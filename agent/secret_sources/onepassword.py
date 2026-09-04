"""1Password (`op` CLI) secret source.

Resolve provider credentials from 1Password ``op://vault/item/field``
references at process startup so they don't have to live in plaintext in
``~/.hermes/.env``.

Design summary
--------------

* Users map environment-variable names to official 1Password secret
  references in ``secrets.onepassword.env``::

      secrets:
        onepassword:
          enabled: true
          env:
            OPENAI_API_KEY: "op://Private/OpenAI/api key"
            ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"

* After ``.env`` loads, each reference is resolved with a single
  ``op read -- <reference>`` call and injected into ``os.environ`` (the
  same point in startup as the Bitwarden source).
* Authentication is whatever the user's ``op`` CLI already uses — a
  service-account token (``OP_SERVICE_ACCOUNT_TOKEN``) for headless boxes,
  or a desktop/interactive session (``OP_SESSION_*``).  Hermes never
  authenticates on the user's behalf; it shells out to an already-trusted,
  already-authenticated CLI.
* Failures NEVER block startup.  A missing ``op`` binary, expired auth, a
  bad reference, or a permission error each surface a one-line warning and
  Hermes continues with whatever credentials ``.env`` already had.

The atomic-write / ``0600`` / TTL cache mechanics are shared with the other
backends via :mod:`agent.secret_sources._cache` — successful, complete pulls
are cached in-process and on disk under ``<hermes_home>/cache/op_cache.json``
so back-to-back short-lived ``hermes`` invocations don't re-shell ``op`` for
every reference.  The disk file holds only resolved secret *values*; auth
material is fingerprinted, never stored.
"""

from __future__ import annotations

import ctypes
import errno
import functools
import hashlib
import logging
import os
import select
import secrets as secrets_module
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agent.secret_sources._cache import (
    CachedFetch,
    DiskCache,
    FetchResult,
    is_valid_env_name,
)
from agent.secret_sources.base import ErrorKind, SecretSource
from agent.secret_sources.base import get_source_environment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# How long to wait for a single `op read`, in seconds.
_OP_RUN_TIMEOUT = 30

# Default env var the official `op` CLI reads for service-account auth.  Users
# can point `service_account_token_env` at a different name; we always export
# the value to the child as OP_SERVICE_ACCOUNT_TOKEN, which is what `op` itself
# looks for.
_DEFAULT_TOKEN_ENV = "OP_SERVICE_ACCOUNT_TOKEN"

# ANSI stripping for `op` diagnostics we surface uses the shared
# tools.ansi_strip.strip_ansi (full ECMA-48: CSI, OSC, DCS/SOS/PM/APC,
# C1) so a control sequence can't reposition the cursor or hide text
# after a redaction marker.

# Env vars the `op` child actually needs.  We build a minimal allowlisted env
# rather than copying all of os.environ (which, post-dotenv, holds every
# provider credential) into the child — tighter blast radius if `op` or
# anything it execs ever misbehaves.  OP_SESSION_* and the token are added
# dynamically in _op_child_env().
_OP_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "SystemRoot",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "OP_ACCOUNT",
    "OP_CONNECT_HOST",
    "OP_CONNECT_TOKEN",
    # Lets a user skip op's desktop-app integration probe (which can hang with
    # no timeout on a wedged desktop container) and go straight to token auth.
    "OP_LOAD_DESKTOP_APP_SETTINGS",
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# In-process cache.  The key folds in str(home_path) so a HERMES_HOME switch
# inside one long-lived process (e.g. the gateway) can't return another
# profile's secrets from L1.  The disk layer omits home from its serialized
# key because the file already lives under the home dir (see _disk_key_str).
_CacheKey = Tuple[str, str, str, str]  # (auth_fp, account, home, refs_fp)
_CACHE: Dict[_CacheKey, CachedFetch] = {}

# `op` can ignore private runtime settings and rendezvous through its global
# daemon.  Serialize the entire cache/fetch/cleanup transaction so two gateway
# reload paths cannot cross-adopt or reuse each other's daemon.
_OP_FETCH_LOCK = threading.RLock()

_DISK_CACHE_BASENAME = "op_cache.json"

# One private, process-scoped CLI socket namespace.  The official CLI's global
# fallback alternates between /var/run/user/<uid> and /run/user/<uid> depending
# on whether XDG_RUNTIME_DIR is available; those aliases can make two clients
# replace the same socket and leave both 24h daemons alive in one service
# cgroup.  A per-Hermes-process name also prevents an old global daemon from
# being reused by a supposedly cache-disabled read.
_OP_SOCKET_NAMESPACE = hashlib.sha256(
    f"{os.getpid()}:{time.time_ns()}".encode("ascii")
).hexdigest()[:16]


def _disk_key_str(cache_key: _CacheKey) -> str:
    """Serialize a cache key for on-disk storage, omitting home_path.

    The disk file is already partitioned by home (it lives under
    ``<home>/cache/``), so the path provides the home dimension; folding it
    into the key string too would be redundant.
    """
    auth_fp, account, _home, refs_fp = cache_key
    return f"{auth_fp}|{account}|{refs_fp}"


_DISK_CACHE: DiskCache = DiskCache(_DISK_CACHE_BASENAME, key_serializer=_disk_key_str)


def _disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Path to the on-disk cache (exposed for tests and direct callers)."""
    return _DISK_CACHE.path(home_path)


# ---------------------------------------------------------------------------
# Reference validation + fingerprinting
# ---------------------------------------------------------------------------


def _validate_references(
    references: Optional[Dict[str, str]],
) -> Tuple[Dict[str, str], List[str]]:
    """Return ``(valid_refs, warnings)`` from an ``env`` mapping.

    A reference is kept only if its target env-var name is a valid POSIX
    name and the value is a stripped ``op://…`` reference string.  Everything
    else produces a warning and is dropped (never fatal).
    """
    valid: Dict[str, str] = {}
    warnings: List[str] = []
    for name, ref in (references or {}).items():
        if not is_valid_env_name(name):
            warnings.append(f"Skipping {name!r}: not a valid env-var name")
            continue
        if not isinstance(ref, str):
            warnings.append(f"Skipping {name!r}: reference is not a string")
            continue
        cleaned = ref.strip()
        if not cleaned.startswith("op://"):
            warnings.append(
                f"Skipping {name!r}: {ref!r} is not an op:// secret reference"
            )
            continue
        valid[name] = cleaned
    return valid, warnings


def _auth_fingerprint(token_env: str) -> str:
    """SHA-256 prefix over the auth material `op` would use.

    Folds in the service-account token, ``OP_ACCOUNT``, the 1Password Connect
    ``OP_CONNECT_HOST``/``OP_CONNECT_TOKEN``, and *all* ``OP_SESSION_*`` vars
    (the names `op` actually exports for interactive sessions —
    ``OP_SESSION_<account_shorthand>``).  Signing out and into a different
    identity therefore changes the cache key, so a value cached under a
    previous identity is never served under a new one.  Never logged or
    displayed; the raw token never leaves this hash.
    """
    source_env = get_source_environment()
    parts: List[str] = [
        f"token={source_env.get(token_env, '')}",
        f"account={source_env.get('OP_ACCOUNT', '')}",
        f"connect_host={source_env.get('OP_CONNECT_HOST', '')}",
        f"connect_token={source_env.get('OP_CONNECT_TOKEN', '')}",
    ]
    for key in sorted(source_env):
        if key.startswith("OP_SESSION_"):
            parts.append(f"{key}={source_env[key]}")
    material = "\n".join(parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _refs_fingerprint(references: Dict[str, str]) -> str:
    """SHA-256 prefix over the configured name→reference mapping."""
    material = "\n".join(f"{name}={references[name]}" for name in sorted(references))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def find_op(binary_path: str = "") -> Optional[Path]:
    """Resolve a usable ``op`` binary, or None.

    When ``binary_path`` is set it is used verbatim and PATH is NOT consulted
    — pinning an absolute path is a way to avoid trusting whatever ``op`` shows
    up first on ``PATH``.  A pinned-but-missing path returns None (the caller
    surfaces a clear error) rather than silently falling back.
    """
    if binary_path:
        pinned = Path(binary_path)
        if pinned.exists() and os.access(pinned, os.X_OK):
            return pinned
        return None
    found = shutil.which("op")
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# `op read` invocation
# ---------------------------------------------------------------------------


def _scrub(text: str) -> str:
    """Remove ANSI control sequences and trim, for safe message surfacing."""
    from tools.ansi_strip import strip_ansi

    # strip_ansi removes well-formed sequences; drop any stray lone ESC too.
    return strip_ansi(text).replace("\x1b", "").strip()


@dataclass(frozen=True)
class _OpRuntimeNamespace:
    """Kernel-bound runtime namespace owned by one 1Password fetch batch."""

    runtime_dir: Path
    child_runtime_dir: Path
    socket_path: Path
    dir_fd: int
    dir_dev: int
    dir_ino: int
    uid: int
    start_ticks_floor: int
    cgroup: bytes


@dataclass(frozen=True)
class _OpDaemonProcess:
    pid: int
    start_ticks: int


@dataclass(frozen=True)
class _OpDaemonEvidence:
    ppid: int
    start_ticks: int
    uid: int
    executable_dev: int
    executable_ino: int
    cmdline: bytes
    cgroup: bytes


def _runtime_root_candidates(source_env: Mapping[str, str], uid: int) -> List[Path]:
    candidates: List[Path] = []
    raw_runtime = source_env.get("XDG_RUNTIME_DIR", "").strip()
    if raw_runtime:
        candidates.append(Path(raw_runtime).expanduser())
    candidates.append(Path("/run/user") / str(uid))
    return candidates


def _runtime_root_is_safe_and_short(runtime_dir: Path, uid: int) -> Optional[Path]:
    try:
        resolved = runtime_dir.resolve(strict=True)
        runtime_stat = resolved.stat()
    except (OSError, RuntimeError):
        return None
    if (
        not resolved.is_absolute()
        or not stat.S_ISDIR(runtime_stat.st_mode)
        or runtime_stat.st_uid != uid
        or stat.S_IMODE(runtime_stat.st_mode) & 0o022
        or not _runtime_root_ancestry_is_safe(resolved, uid)
    ):
        return None
    return resolved


def _runtime_root_ancestry_is_safe(runtime_dir: Path, uid: int) -> bool:
    """Reject replaceable non-sticky ancestry for a canonical runtime root."""
    for ancestor in (runtime_dir, *runtime_dir.parents):
        try:
            ancestor_stat = ancestor.stat()
        except OSError:
            return False
        mode = stat.S_IMODE(ancestor_stat.st_mode)
        if (
            not stat.S_ISDIR(ancestor_stat.st_mode)
            or ancestor_stat.st_uid not in {0, uid}
            or (mode & 0o022 and not mode & stat.S_ISVTX)
        ):
            return False
    return True


def _open_bound_runtime_root(runtime_root: Path, uid: int) -> int:
    """Open and bind the exact safe runtime-root inode used for mkdirat."""
    checked = runtime_root.stat()
    fd = os.open(
        runtime_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        opened = os.fstat(fd)
        current = runtime_root.stat()
        if (
            checked.st_dev != opened.st_dev
            or checked.st_ino != opened.st_ino
            or current.st_dev != opened.st_dev
            or current.st_ino != opened.st_ino
            or not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != uid
            or stat.S_IMODE(opened.st_mode) & 0o022
            or not _runtime_root_ancestry_is_safe(runtime_root, uid)
        ):
            raise RuntimeError("1Password runtime root identity drift; setup HOLD")
        return fd
    except Exception:
        os.close(fd)
        raise


def _fallback_op_runtime_root(source_env: Mapping[str, str], uid: int) -> Path:
    """Create a private persistent root when no system runtime root exists."""
    home_text = source_env.get("HERMES_HOME", "").strip()
    if home_text:
        hermes_home = Path(home_text).expanduser()
    else:
        user_home = source_env.get("HOME", "").strip()
        hermes_home = (
            Path(user_home).expanduser() / ".hermes"
            if user_home
            else Path.home() / ".hermes"
        )
    try:
        hermes_home = hermes_home.resolve(strict=True)
        home_stat = hermes_home.stat()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("no safe 1Password CLI runtime root available") from exc
    if (
        not hermes_home.is_absolute()
        or not stat.S_ISDIR(home_stat.st_mode)
        or home_stat.st_uid != uid
        or stat.S_IMODE(home_stat.st_mode) & 0o022
    ):
        raise RuntimeError("unsafe HERMES_HOME for 1Password runtime fallback")

    parent_fd = os.open(
        hermes_home,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        try:
            os.mkdir(".runtime", mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        runtime_fd = os.open(
            ".runtime",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            runtime_stat = os.fstat(runtime_fd)
            runtime_root = hermes_home / ".runtime"
            if (
                not stat.S_ISDIR(runtime_stat.st_mode)
                or runtime_stat.st_uid != uid
                or stat.S_IMODE(runtime_stat.st_mode) != 0o700
                or runtime_root.resolve(strict=True) != runtime_root
            ):
                raise RuntimeError("unsafe private 1Password runtime fallback")
        finally:
            os.close(runtime_fd)
    finally:
        os.close(parent_fd)

    safe_root = _runtime_root_is_safe_and_short(runtime_root, uid)
    if safe_root is None:
        raise RuntimeError(
            "private 1Password runtime fallback path is unsafe or too long"
        )
    return safe_root


def _safe_op_runtime_root(
    source_env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return a canonical uid-owned runtime root or create a private fallback."""
    source = get_source_environment() if source_env is None else source_env
    uid = os.getuid()  # windows-footgun: ok
    for candidate in _runtime_root_candidates(source, uid):
        safe_root = _runtime_root_is_safe_and_short(candidate, uid)
        if safe_root is not None:
            return safe_root
    return _fallback_op_runtime_root(source, uid)


def _linux_start_ticks_floor() -> int:
    ticks = int(os.sysconf("SC_CLK_TCK"))
    return max(0, int(time.clock_gettime(time.CLOCK_BOOTTIME) * ticks) - 1)


def _create_op_runtime_namespace_inner() -> _OpRuntimeNamespace:
    """Create a private XDG namespace so one batch owns any CLI daemon."""
    if sys.platform != "linux":
        raise RuntimeError("managed 1Password daemon cleanup requires Linux")

    source_env = get_source_environment()
    root = _safe_op_runtime_root(source_env)
    uid = os.getuid()  # windows-footgun: ok
    parent_fd = _open_bound_runtime_root(root, uid)
    name = f"hermes-op-{_OP_SOCKET_NAMESPACE}-{secrets_module.token_hex(8)}"
    runtime_dir = root / name
    dir_fd = -1
    created = False
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        created = True
        dir_fd = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        runtime_stat = os.fstat(dir_fd)
        if (
            not stat.S_ISDIR(runtime_stat.st_mode)
            or runtime_stat.st_uid != uid
            or stat.S_IMODE(runtime_stat.st_mode) != 0o700
            or runtime_dir.resolve(strict=True) != runtime_dir
        ):
            raise RuntimeError("unsafe private 1Password runtime directory")
        child_runtime_dir = Path("/proc") / str(os.getpid()) / "fd" / str(dir_fd)
        if child_runtime_dir.resolve(strict=True) != runtime_dir:
            raise RuntimeError("private 1Password runtime FD binding failed")
        socket_path = child_runtime_dir / "op.sock"
        # sockaddr_un.sun_path is 108 bytes on Linux including its NUL.
        if len(os.fsencode(socket_path)) > 100:
            raise RuntimeError("private 1Password CLI socket path is too long")
        return _OpRuntimeNamespace(
            runtime_dir=runtime_dir,
            child_runtime_dir=child_runtime_dir,
            socket_path=socket_path,
            dir_fd=dir_fd,
            dir_dev=runtime_stat.st_dev,
            dir_ino=runtime_stat.st_ino,
            uid=uid,
            start_ticks_floor=_linux_start_ticks_floor(),
            cgroup=Path("/proc/self/cgroup").read_bytes(),
        )
    except Exception:
        if dir_fd >= 0:
            os.close(dir_fd)
        if created:
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)


def _create_op_runtime_namespace() -> _OpRuntimeNamespace:
    try:
        return _create_op_runtime_namespace_inner()
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(
            "1Password private runtime setup failed; setup HOLD"
        ) from exc


def _read_proc_stat(pid: int) -> Tuple[int, int]:
    raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise RuntimeError("malformed /proc stat")
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError("short /proc stat")
    return int(fields[1]), int(fields[19])  # PPID (field 4), starttime (field 22)


def _read_proc_environment(pid: int) -> Dict[bytes, bytes]:
    raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    return {
        part.split(b"=", 1)[0]: part.split(b"=", 1)[1]
        for part in raw.split(b"\0")
        if b"=" in part
    }


def _read_op_daemon_evidence(pid: int) -> _OpDaemonEvidence:
    proc = Path("/proc") / str(pid)
    ppid, start_ticks = _read_proc_stat(pid)
    proc_stat = proc.stat()
    exe_stat = (proc / "exe").stat()
    return _OpDaemonEvidence(
        ppid=ppid,
        start_ticks=start_ticks,
        uid=proc_stat.st_uid,
        executable_dev=exe_stat.st_dev,
        executable_ino=exe_stat.st_ino,
        cmdline=(proc / "cmdline").read_bytes(),
        cgroup=(proc / "cgroup").read_bytes(),
    )


def _op_daemon_core_identity_matches(
    evidence: _OpDaemonEvidence,
    namespace: _OpRuntimeNamespace,
    binary_stat: os.stat_result,
) -> bool:
    return (
        evidence.ppid == 1
        and evidence.start_ticks >= namespace.start_ticks_floor
        and evidence.uid == namespace.uid
        and evidence.cmdline == b"op\0daemon\0"
        and evidence.executable_dev == binary_stat.st_dev
        and evidence.executable_ino == binary_stat.st_ino
        and evidence.cgroup == namespace.cgroup
    )


def _op_daemon_identity_matches_ignoring_exe(
    evidence: _OpDaemonEvidence,
    namespace: _OpRuntimeNamespace,
) -> bool:
    """Every core-identity axis except the executable inode.

    A re-homed ``op daemon`` whose binary was swapped mid-batch (an ``op``
    upgrade replaces the file, so ``/proc/<pid>/exe`` dev/ino no longer
    matches the freshly stat-ed binary) still matches every other axis. Such
    a daemon must be surfaced as ``foreign`` (reported, never signalled)
    rather than silently dropped as ``unrelated``.
    """
    return (
        evidence.ppid == 1
        and evidence.start_ticks >= namespace.start_ticks_floor
        and evidence.uid == namespace.uid
        and evidence.cmdline == b"op\0daemon\0"
        and evidence.cgroup == namespace.cgroup
    )


def _writable_by_others(st: os.stat_result, fd: int) -> bool:
    """Whether anyone other than the file's owner can write it.

    World-writable is always unsafe. A group-writable file is safe only when
    BOTH: (a) it carries no extended POSIX ACL — a group-write bit can be an
    ACL *mask* concealing a ``u:other:w`` grant (possibly inherited from a
    directory default ACL), which the mode bits alone cannot distinguish from
    a real group permission; and (b) the owning group is the owner's own
    private per-user group — gid == the owner's primary/login gid, named after
    the owner, no secondary members. This is the USERGROUPS / ``umask 002``
    scheme in which each user's primary gid is unique. Any extended ACL, shared
    group, or unresolvable identity fails closed. POSIX-only; Linux path.
    """
    import grp
    import pwd

    mode = stat.S_IMODE(st.st_mode)
    if mode & stat.S_IWOTH:
        return True
    if mode & stat.S_IWGRP:
        try:
            if "system.posix_acl_access" in os.listxattr(fd):
                return True  # extended ACL: the group bit may be a mask
        except OSError:
            pass  # filesystem without xattr/ACL support -> the bit is a real perm
        try:
            owner = pwd.getpwuid(st.st_uid)
            group = grp.getgrgid(st.st_gid)
        except (KeyError, OSError):
            return True
        if (
            st.st_gid != owner.pw_gid
            or group.gr_name != owner.pw_name
            or set(group.gr_mem) - {owner.pw_name}
        ):
            return True
    return False


def _inspect_op_daemon(
    pid: int,
    namespace: _OpRuntimeNamespace,
    binary_stat: os.stat_result,
) -> Tuple[str, Optional[_OpDaemonProcess]]:
    """Classify a PID as gone, unrelated, exact, or foreign-in-namespace."""
    try:
        env = _read_proc_environment(pid)
    except FileNotFoundError:
        return "gone", None
    except (OSError, RuntimeError):
        try:
            evidence = _read_op_daemon_evidence(pid)
        except FileNotFoundError:
            return "gone", None
        except (OSError, RuntimeError, ValueError):
            return "unrelated", None
        return (
            ("foreign", None)
            if _op_daemon_core_identity_matches(evidence, namespace, binary_stat)
            else ("unrelated", None)
        )

    expected_socket = os.fsencode(namespace.socket_path)
    child_runtime_dir = getattr(namespace, "child_runtime_dir", None)
    if not isinstance(child_runtime_dir, Path):
        child_runtime_dir = namespace.runtime_dir
    expected_runtime = os.fsencode(child_runtime_dir)
    socket_matches = env.get(b"OP_SOCK") == expected_socket
    runtime_matches = env.get(b"XDG_RUNTIME_DIR") == expected_runtime
    if not socket_matches and not runtime_matches:
        # CLI 2.38.1 can sanitize OP_SOCK and re-home XDG_RUNTIME_DIR to the
        # global runtime after daemonizing.  A process whose remaining core
        # identity still matches this fetch boundary is therefore ambiguous,
        # not unrelated: fail closed and never signal it.
        try:
            evidence = _read_op_daemon_evidence(pid)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return "unrelated", None
        # An exe dev/ino swap (op upgraded mid-batch) must not downgrade a
        # re-homed daemon that still matches every other axis to a silent
        # "unrelated" pass — cleanup would then remove an empty namespace over
        # a live global-runtime leak. Surface it as foreign so the backstop
        # HOLDs; foreign is never signalled, so this can never kill a
        # genuinely-unrelated process.
        if _op_daemon_core_identity_matches(
            evidence, namespace, binary_stat
        ) or _op_daemon_identity_matches_ignoring_exe(evidence, namespace):
            return "foreign", None
        return "unrelated", None
    if not socket_matches or not runtime_matches:
        return "foreign", None

    try:
        evidence = _read_op_daemon_evidence(pid)
    except FileNotFoundError:
        return "gone", None
    except (OSError, RuntimeError, ValueError):
        return "foreign", None

    if not _op_daemon_core_identity_matches(evidence, namespace, binary_stat):
        return "foreign", None
    return "exact", _OpDaemonProcess(pid=pid, start_ticks=evidence.start_ticks)


def _scan_op_runtime_namespace(
    namespace: _OpRuntimeNamespace,
    binary_stat: os.stat_result,
) -> Tuple[List[_OpDaemonProcess], List[int]]:
    exact: List[_OpDaemonProcess] = []
    foreign: List[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        status, identity = _inspect_op_daemon(pid, namespace, binary_stat)
        if status == "exact" and identity is not None:
            exact.append(identity)
        elif status == "foreign":
            foreign.append(pid)
    return exact, foreign


def _read_op_daemon_pidfile(namespace: _OpRuntimeNamespace) -> Optional[int]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open("op-daemon.pid", flags, dir_fd=namespace.dir_fd)
    except FileNotFoundError:
        return None
    try:
        pid_stat = os.fstat(fd)
        if (
            not stat.S_ISREG(pid_stat.st_mode)
            or pid_stat.st_uid != namespace.uid
            or stat.S_IMODE(pid_stat.st_mode) & 0o077
            or pid_stat.st_nlink != 1
        ):
            raise RuntimeError("unsafe 1Password daemon pidfile; cleanup HOLD")
        raw = os.read(fd, 64).strip()
        if not raw.isdigit() or len(raw) > 20:
            raise RuntimeError("malformed 1Password daemon pidfile; cleanup HOLD")
        pid = int(raw)
        if pid <= 0:
            raise RuntimeError("invalid 1Password daemon pid; cleanup HOLD")
        return pid
    finally:
        os.close(fd)


def _pidfd_open(pid: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_open
    except AttributeError as exc:
        raise RuntimeError("pidfd_open unavailable; 1Password cleanup HOLD") from exc
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _pidfd_send_sigterm(pid_fd: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_send_signal
    except AttributeError as exc:
        raise RuntimeError(
            "pidfd_send_signal unavailable; 1Password cleanup HOLD"
        ) from exc
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    if function(pid_fd, signal.SIGTERM, None, 0) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _wait_pidfd_exit(pid_fd: int, timeout_seconds: float = 5.0) -> bool:
    poller = select.poll()
    poller.register(pid_fd, select.POLLIN | select.POLLHUP | select.POLLERR)
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        try:
            if poller.poll(remaining_ms):
                return True
        except InterruptedError:
            continue
        return False


def _validate_runtime_namespace(namespace: _OpRuntimeNamespace) -> None:
    fd_stat = os.fstat(namespace.dir_fd)
    try:
        path_stat = namespace.runtime_dir.stat()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "1Password runtime directory vanished; cleanup HOLD"
        ) from exc
    if (
        fd_stat.st_dev != namespace.dir_dev
        or fd_stat.st_ino != namespace.dir_ino
        or path_stat.st_dev != namespace.dir_dev
        or path_stat.st_ino != namespace.dir_ino
        or fd_stat.st_uid != namespace.uid
        or stat.S_IMODE(fd_stat.st_mode) != 0o700
        or namespace.child_runtime_dir.resolve(strict=True) != namespace.runtime_dir
    ):
        raise RuntimeError("1Password runtime identity drift; cleanup HOLD")


def _remove_op_runtime_namespace(namespace: _OpRuntimeNamespace) -> None:
    _validate_runtime_namespace(namespace)
    allowed = {"op.sock", "op-daemon.pid"}
    entries = os.listdir(namespace.dir_fd)
    unexpected = sorted(set(entries) - allowed)
    if unexpected:
        raise RuntimeError("unexpected 1Password runtime entries; cleanup HOLD")
    for name in entries:
        entry_stat = os.stat(name, dir_fd=namespace.dir_fd, follow_symlinks=False)
        valid = (
            name == "op.sock"
            and stat.S_ISSOCK(entry_stat.st_mode)
            and entry_stat.st_uid == namespace.uid
        ) or (
            name == "op-daemon.pid"
            and stat.S_ISREG(entry_stat.st_mode)
            and entry_stat.st_uid == namespace.uid
            and not stat.S_IMODE(entry_stat.st_mode) & 0o077
        )
        if not valid:
            raise RuntimeError("unsafe 1Password runtime entry; cleanup HOLD")
        os.unlink(name, dir_fd=namespace.dir_fd)
    _validate_runtime_namespace(namespace)
    os.rmdir(namespace.runtime_dir)
    if os.fstat(namespace.dir_fd).st_nlink != 0:
        raise RuntimeError("1Password runtime inode unlink proof failed; cleanup HOLD")


def _require_op_runtime_quiescence(
    namespace: _OpRuntimeNamespace,
    binary_stat: os.stat_result,
    *,
    runtime_present: bool,
) -> None:
    """Require two bounded zero-process observations for this exact namespace."""
    for observation in range(2):
        pidfile_pid: Optional[int] = None
        if runtime_present:
            _validate_runtime_namespace(namespace)
            pidfile_pid = _read_op_daemon_pidfile(namespace)
        exact, foreign = _scan_op_runtime_namespace(namespace, binary_stat)
        if exact or foreign:
            raise RuntimeError(
                "1Password runtime namespace is not quiescent; cleanup HOLD"
            )
        if pidfile_pid is not None and (Path("/proc") / str(pidfile_pid)).exists():
            status, _ = _inspect_op_daemon(pidfile_pid, namespace, binary_stat)
            if status != "gone":
                raise RuntimeError("unattested 1Password daemon is live; cleanup HOLD")
        if observation == 0:
            time.sleep(0.05)


def _close_op_runtime_namespace_fd(dir_fd: int) -> None:
    os.close(dir_fd)


def _cleanup_op_runtime_namespace(
    namespace: _OpRuntimeNamespace,
    binary: Path,
) -> None:
    """Gracefully stop only the exact daemon created in this fetch namespace."""
    try:
        _validate_runtime_namespace(namespace)
        try:
            binary_stat = binary.resolve(strict=True).stat()
        except OSError as exc:
            raise RuntimeError(
                "1Password binary identity unavailable; cleanup HOLD"
            ) from exc

        pidfile_pid = _read_op_daemon_pidfile(namespace)
        exact, foreign = _scan_op_runtime_namespace(namespace, binary_stat)
        if foreign or len(exact) > 1:
            raise RuntimeError("ambiguous 1Password daemon identity; cleanup HOLD")

        if pidfile_pid is not None:
            status, pidfile_identity = _inspect_op_daemon(
                pidfile_pid, namespace, binary_stat
            )
            if status == "foreign" or status == "unrelated":
                raise RuntimeError(
                    "foreign 1Password daemon pidfile target; cleanup HOLD"
                )
            if (
                status == "exact"
                and pidfile_identity is not None
                and pidfile_identity not in exact
            ):
                exact.append(pidfile_identity)

        if exact:
            identity = exact[0]
            if pidfile_pid != identity.pid:
                raise RuntimeError("1Password daemon pidfile mismatch; cleanup HOLD")
            try:
                pid_fd = _pidfd_open(identity.pid)
            except OSError as exc:
                if exc.errno != errno.ESRCH:
                    raise RuntimeError(
                        "cannot pin 1Password daemon; cleanup HOLD"
                    ) from exc
            else:
                try:
                    status, pinned = _inspect_op_daemon(
                        identity.pid, namespace, binary_stat
                    )
                    if status != "exact" or pinned != identity:
                        raise RuntimeError(
                            "1Password daemon identity changed after pidfd pin; cleanup HOLD"
                        )
                    try:
                        _pidfd_send_sigterm(pid_fd)
                    except OSError as exc:
                        if exc.errno != errno.ESRCH:
                            raise RuntimeError(
                                "cannot terminate 1Password daemon; cleanup HOLD"
                            ) from exc
                    if not _wait_pidfd_exit(pid_fd):
                        raise RuntimeError(
                            "1Password daemon did not exit after SIGTERM; cleanup HOLD"
                        )
                finally:
                    os.close(pid_fd)
        elif pidfile_pid is not None:
            # A private pidfile whose process has already exited is only stale
            # local state; it is safe to remove with the inode-bound directory.
            if (Path("/proc") / str(pidfile_pid)).exists():
                raise RuntimeError("unattested 1Password daemon is live; cleanup HOLD")

        _require_op_runtime_quiescence(namespace, binary_stat, runtime_present=True)
        _remove_op_runtime_namespace(namespace)
        _require_op_runtime_quiescence(namespace, binary_stat, runtime_present=False)
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError("1Password daemon cleanup failed; cleanup HOLD") from exc
    finally:
        primary_error = sys.exc_info()[1]
        try:
            _close_op_runtime_namespace_fd(namespace.dir_fd)
        except OSError as close_error:
            if primary_error is None:
                raise RuntimeError(
                    "1Password runtime namespace close failed; cleanup HOLD"
                ) from close_error
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(
                    f"Additionally failed to close runtime namespace: {close_error}"
                )


def _op_child_env(
    token_value: str,
    runtime_namespace: Optional[_OpRuntimeNamespace] = None,
) -> Dict[str, str]:
    """Build a minimal allowlisted environment for the ``op`` child process."""
    source_env = get_source_environment()
    env: Dict[str, str] = {}
    for key in _OP_ENV_ALLOWLIST:
        val = source_env.get(key)
        if val is not None:
            env[key] = val
    # Desktop / interactive session credentials.
    for key, val in source_env.items():
        if key.startswith("OP_SESSION_"):
            env[key] = val
    # `op` reads OP_SERVICE_ACCOUNT_TOKEN regardless of which env var the user
    # configured Hermes to source it from, so normalize to that name here.
    if token_value:
        env["OP_SERVICE_ACCOUNT_TOKEN"] = token_value
    # CLI 2.38.1 can still launch a Linux service-account cache daemon despite
    # both OP_CACHE=false and --cache=false. Keep both producer controls, then
    # bind that daemon to one private XDG namespace that Hermes cleans exactly.
    env["OP_CACHE"] = "false"
    # The official CLI can still create a daemon while probing desktop-app or
    # biometric integration.  Force both documented producer controls even if
    # a parent environment tried to enable them.
    env["OP_LOAD_DESKTOP_APP_SETTINGS"] = "false"
    env["OP_BIOMETRIC_UNLOCK_ENABLED"] = "false"
    if sys.platform == "linux":
        if runtime_namespace is not None:
            child_runtime_dir = getattr(runtime_namespace, "child_runtime_dir", None)
            runtime_dir = (
                child_runtime_dir
                if isinstance(child_runtime_dir, Path)
                else runtime_namespace.runtime_dir
            )
            socket_path = runtime_namespace.socket_path
        else:
            runtime_dir = _safe_op_runtime_root(source_env)
            socket_path = runtime_dir / f"hermes-op-{_OP_SOCKET_NAMESPACE}.sock"
            if len(os.fsencode(socket_path)) > 100:
                raise RuntimeError("private 1Password CLI socket path is too long")
        # Pin both variables to the same canonical runtime. `op` uses
        # XDG_RUNTIME_DIR for its pidfile even when OP_SOCK points elsewhere;
        # leaving XDG inherited created /tmp and /run aliases in one cgroup.
        env["XDG_RUNTIME_DIR"] = str(runtime_dir)
        env["OP_SOCK"] = str(socket_path)
    env["NO_COLOR"] = "1"
    return env


def _run_op_process(
    cmd: List[str], *, env: Dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run one exact op child with bounded SIGTERM-only timeout handling."""
    process_cmd = cmd
    process_env = env
    helper_fd: Optional[int] = None
    popen_kwargs: Dict[str, Any] = {}
    timeout = float(_OP_RUN_TIMEOUT)
    if sys.platform == "linux":
        helper = Path(__file__).with_name("_op_subreaper.py")
        try:
            helper_fd = os.open(
                helper,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            fd_stat = os.fstat(helper_fd)
            path_stat = helper.stat()
        except OSError as exc:
            if helper_fd is not None:
                os.close(helper_fd)
            raise RuntimeError(
                "unable to bind op lifecycle helper; execution HOLD"
            ) from exc
        if (
            not stat.S_ISREG(fd_stat.st_mode)
            or fd_stat.st_uid != os.getuid()  # windows-footgun: ok
            or _writable_by_others(fd_stat, helper_fd)
            or (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        ):
            os.close(helper_fd)
            raise RuntimeError("unsafe op lifecycle helper; execution HOLD")
        process_cmd = [
            sys.executable,
            "-I",
            "-S",
            f"/proc/self/fd/{helper_fd}",
            "--",
            *cmd,
        ]
        process_env = dict(env)
        process_env["PYTHONDONTWRITEBYTECODE"] = "1"
        popen_kwargs["pass_fds"] = (helper_fd,)
        timeout += 15.0
    try:
        proc = subprocess.Popen(  # noqa: S603 — inode-bound helper / reviewed argv
            process_cmd,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except OSError as exc:
        raise RuntimeError(f"failed to invoke op: {exc}") from exc
    finally:
        if helper_fd is not None:
            os.close(helper_fd)

    pid_fd: Optional[int] = None
    if sys.platform == "linux":
        try:
            pid_fd = _pidfd_open(proc.pid)
        except (OSError, RuntimeError) as exc:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            raise RuntimeError("unable to bind op child pidfd; execution HOLD") from exc
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                if pid_fd is not None:
                    _pidfd_send_sigterm(pid_fd)
                    exited = _wait_pidfd_exit(pid_fd)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                        exited = True
                    except subprocess.TimeoutExpired:
                        exited = False
            except (OSError, RuntimeError) as signal_exc:
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
                raise RuntimeError(
                    f"op read timed out after {_OP_RUN_TIMEOUT}s; "
                    "SIGTERM delivery failed; execution HOLD"
                ) from signal_exc
            if not exited:
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
                raise RuntimeError(
                    f"op read timed out after {_OP_RUN_TIMEOUT}s; "
                    "SIGTERM sent but child still running; execution HOLD"
                ) from exc
            try:
                proc.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
                raise RuntimeError(
                    f"op read timed out after {_OP_RUN_TIMEOUT}s; "
                    "SIGTERM exited child but output pipes remained open; execution HOLD"
                ) from exc
            raise RuntimeError(
                f"op read timed out after {_OP_RUN_TIMEOUT}s; SIGTERM completed"
            ) from exc
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    finally:
        if pid_fd is not None:
            os.close(pid_fd)


def _run_op_read(
    op: Path,
    reference: str,
    *,
    account: str = "",
    token_value: str = "",
    child_env: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve a single ``op://`` reference to its value.

    Raises :class:`RuntimeError` on any failure — including a ``returncode 0``
    with empty output, which would otherwise silently clobber a good
    ``.env``/shell credential with ``""``.
    """
    cmd: List[str] = [str(op), "--cache=false", "read"]
    if account:
        cmd += ["--account", account]
    # `--` terminates option parsing so a reference can never be mis-parsed as
    # an `op` flag even if validation is ever loosened.
    cmd += ["--", reference]

    if child_env is None and sys.platform == "linux":
        namespace = _create_op_runtime_namespace()
        try:
            return _run_op_read(
                op,
                reference,
                account=account,
                token_value=token_value,
                child_env=_op_child_env(token_value, namespace),
            )
        finally:
            _cleanup_op_runtime_namespace(namespace, op)

    proc = _run_op_process(
        cmd,
        env=child_env if child_env is not None else _op_child_env(token_value),
    )

    if proc.returncode != 0:
        err = _scrub(proc.stderr or "")[:200]
        if err:
            raise RuntimeError(f"op read failed for {reference!r}: {err}")
        raise RuntimeError(f"op read exited {proc.returncode} for {reference!r}")

    # `op` appends a trailing newline; strip only that so a value with
    # intentional internal/edge spaces survives.  But a value that is empty or
    # whitespace-only is treated as empty: applying it would silently clobber a
    # good .env/shell credential with effectively nothing.
    value = (proc.stdout or "").rstrip("\r\n")
    if not value.strip():
        raise RuntimeError(f"op read returned an empty value for {reference!r}")
    return value


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _serialize_op_fetch(function):
    @functools.wraps(function)
    def serialized(*args: Any, **kwargs: Any):
        with _OP_FETCH_LOCK:
            return function(*args, **kwargs)

    return serialized


@_serialize_op_fetch
def fetch_onepassword_secrets(
    *,
    references: Dict[str, str],
    account: str = "",
    token_env: str = _DEFAULT_TOKEN_ENV,
    binary: Optional[Path] = None,
    binary_path: str = "",
    use_cache: bool = True,
    cache_ttl_seconds: float = 300,
    home_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Resolve ``references`` (name → ``op://…``) to ``(secrets, warnings)``.

    Raises :class:`RuntimeError` for fatal batch-level conditions: no ``op``
    binary, unsafe runtime setup, or an unverified daemon cleanup. Per-reference
    failures (expired auth, bad reference, empty value) are collected as warnings
    and the reference is dropped, so one bad entry never sinks the rest.

    Only a complete, error-free pull is cached, so a transient auth failure
    isn't frozen in for the whole TTL window.
    """
    valid, warnings = _validate_references(references)
    if not valid:
        return {}, warnings

    token_value = get_source_environment().get(token_env, "").strip()
    cache_key: _CacheKey = (
        _auth_fingerprint(token_env),
        account or "",
        str(home_path) if home_path is not None else "",
        _refs_fingerprint(valid),
    )

    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return dict(cached.secrets), warnings
        disk_cached = _DISK_CACHE.read(cache_key, cache_ttl_seconds, home_path)
        if disk_cached is not None:
            # Promote into L1 so later fetches in this process skip the disk read.
            _CACHE[cache_key] = disk_cached
            return dict(disk_cached.secrets), warnings

    op = binary or find_op(binary_path)
    if op is None:
        raise RuntimeError(
            "op CLI not found.  Install the 1Password CLI "
            "(https://developer.1password.com/docs/cli/get-started/) or set "
            "secrets.onepassword.binary_path to its absolute location."
        )

    secrets: Dict[str, str] = {}
    read_errors = 0
    runtime_namespace: Optional[_OpRuntimeNamespace] = None
    child_env: Optional[Dict[str, str]] = None
    if sys.platform == "linux":
        runtime_namespace = _create_op_runtime_namespace()
        child_env = _op_child_env(token_value, runtime_namespace)

    try:
        for name in sorted(valid):
            try:
                secrets[name] = _run_op_read(
                    op,
                    valid[name],
                    account=account,
                    token_value=token_value,
                    child_env=child_env,
                )
            except RuntimeError as exc:
                warnings.append(str(exc))
                read_errors += 1
    finally:
        if runtime_namespace is not None:
            _cleanup_op_runtime_namespace(runtime_namespace, op)

    if use_cache and not read_errors and secrets:
        entry = CachedFetch(secrets=dict(secrets), fetched_at=time.time())
        _CACHE[cache_key] = entry
        _DISK_CACHE.write(cache_key, entry, cache_ttl_seconds, home_path)

    return secrets, warnings


# ---------------------------------------------------------------------------
# Public entry point — called from hermes_cli.env_loader
# ---------------------------------------------------------------------------


def apply_onepassword_secrets(
    *,
    enabled: bool,
    env: Optional[Dict[str, str]] = None,
    account: str = "",
    service_account_token_env: str = _DEFAULT_TOKEN_ENV,
    binary_path: str = "",
    override_existing: bool = True,
    cache_ttl_seconds: float = 300,
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Resolve configured ``op://`` references and set them on ``os.environ``.

    Called by ``load_hermes_dotenv()`` after the .env files have loaded.
    Intentionally defensive — any failure returns a :class:`FetchResult` with
    ``error`` set (or surfaces warnings); it never raises.

    Parameters mirror the ``secrets.onepassword.*`` config keys so the caller
    can splat the dict in.  References that are already satisfied by the
    current environment (when ``override_existing`` is false) are skipped
    *before* fetching, so ``op`` is never invoked for a value that would be
    discarded.
    """
    result = FetchResult()

    if not enabled:
        return result

    valid, warnings = _validate_references(env)
    result.warnings.extend(warnings)

    # Skip-before-fetch: never resolve a reference we'd only throw away.
    refs_to_fetch: Dict[str, str] = {}
    for name, ref in valid.items():
        if name == service_account_token_env:
            # Never let a resolved secret clobber the very token used to auth.
            result.skipped.append(name)
            continue
        if not override_existing and os.environ.get(name):
            result.skipped.append(name)
            continue
        refs_to_fetch[name] = ref

    if not refs_to_fetch:
        return result

    binary = find_op(binary_path)
    result.binary_path = binary
    if binary is None:
        if binary_path:
            result.error = (
                f"secrets.onepassword.binary_path ({binary_path!r}) is not an "
                "executable op binary."
            )
        else:
            result.error = (
                "secrets.onepassword.enabled is true but the op CLI was not "
                "found on PATH.  Install it "
                "(https://developer.1password.com/docs/cli/get-started/) or set "
                "secrets.onepassword.binary_path."
            )
        return result

    try:
        secrets, fetch_warnings = fetch_onepassword_secrets(
            references=refs_to_fetch,
            account=account,
            token_env=service_account_token_env,
            binary=binary,
            cache_ttl_seconds=cache_ttl_seconds,
            home_path=home_path,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(fetch_warnings)

    for name, value in secrets.items():
        # The token-var and override guards already filtered refs_to_fetch, but
        # re-check defensively in case the fetch layer ever returns extras.
        if name == service_account_token_env:
            if name not in result.skipped:
                result.skipped.append(name)
            continue
        if not override_existing and os.environ.get(name):
            if name not in result.skipped:
                result.skipped.append(name)
            continue
        os.environ[name] = value
        result.applied.append(name)

    return result


# ---------------------------------------------------------------------------
# SecretSource adapter — the registry-facing wrapper around this module.
# ---------------------------------------------------------------------------


class OnePasswordSource(SecretSource):
    """1Password as a registered secret source.

    Thin adapter over the module's fetch machinery.  ``fetch()`` only
    *fetches* — precedence, override semantics, conflict warnings, and
    the ``os.environ`` writes are the orchestrator's job
    (see ``agent.secret_sources.registry.apply_all``).

    1Password is a **mapped** source: the user explicitly binds each env
    var to an ``op://`` reference under ``secrets.onepassword.env``, so
    its claims outrank bulk sources (e.g. a Bitwarden project dump) on
    contested vars.
    """

    name = "onepassword"
    label = "1Password"
    shape = "mapped"
    scheme = "op"

    def override_existing(self, cfg: dict) -> bool:
        # Default True: an explicit VAR→op:// binding is the strongest
        # user intent there is — leaving a stale .env line in place
        # should not silently defeat it (same rotation rationale as
        # Bitwarden).
        return bool(isinstance(cfg, dict) and cfg.get("override_existing", True))

    def protected_env_vars(self, cfg: dict):
        token_env = _DEFAULT_TOKEN_ENV
        if isinstance(cfg, dict):
            token_env = str(cfg.get("service_account_token_env") or token_env)
        return frozenset({token_env})

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "env": {
                "description": "Map of ENV_VAR -> op://vault/item/field reference",
                "default": {},
            },
            "account": {
                "description": "op --account shorthand (empty = default account)",
                "default": "",
            },
            "service_account_token_env": {
                "description": "Env var holding the service-account token "
                "(unset = desktop/interactive session)",
                "default": _DEFAULT_TOKEN_ENV,
            },
            "binary_path": {
                "description": "Pin the op binary (empty = resolve via PATH)",
                "default": "",
            },
            "cache_ttl_seconds": {
                "description": "Disk+memory cache TTL; 0 disables",
                "default": 300,
            },
            "override_existing": {
                "description": "Resolved values overwrite .env/shell values",
                "default": True,
            },
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()

        env_map = cfg.get("env")
        valid, warnings = _validate_references(
            env_map if isinstance(env_map, dict) else None
        )
        result.warnings.extend(warnings)
        if not valid:
            if not warnings:
                result.error = (
                    "secrets.onepassword.enabled is true but the env: map is "
                    "empty.  Add ENV_VAR: op://vault/item/field entries."
                )
                result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        binary_path = str(cfg.get("binary_path") or "")
        binary = find_op(binary_path)
        result.binary_path = binary
        if binary is None:
            if binary_path:
                result.error = (
                    f"secrets.onepassword.binary_path ({binary_path!r}) is "
                    "not an executable op binary."
                )
            else:
                result.error = (
                    "secrets.onepassword.enabled is true but the op CLI was "
                    "not found on PATH.  Install it "
                    "(https://developer.1password.com/docs/cli/get-started/) "
                    "or set secrets.onepassword.binary_path."
                )
            result.error_kind = ErrorKind.BINARY_MISSING
            return result

        try:
            ttl = float(cfg.get("cache_ttl_seconds", 300))
        except (TypeError, ValueError):
            ttl = 300.0

        try:
            secrets, fetch_warnings = fetch_onepassword_secrets(
                references=valid,
                account=str(cfg.get("account") or ""),
                token_env=str(
                    cfg.get("service_account_token_env") or _DEFAULT_TOKEN_ENV
                ),
                binary=binary,
                cache_ttl_seconds=ttl,
                home_path=home_path,
            )
        except RuntimeError as exc:
            result.error = str(exc)
            result.error_kind = _classify_op_error(str(exc))
            return result

        result.secrets = secrets
        result.warnings.extend(fetch_warnings)
        return result

    def remediation(self, kind, cfg: dict) -> str:
        if kind in (ErrorKind.AUTH_FAILED, ErrorKind.AUTH_EXPIRED):
            token_env = _DEFAULT_TOKEN_ENV
            if isinstance(cfg, dict):
                token_env = str(cfg.get("service_account_token_env") or token_env)
            return (
                "Run `hermes secrets onepassword token` to paste a fresh "
                f"service-account token ({token_env}), or `op signin` for an "
                "interactive session."
            )
        if kind == ErrorKind.BINARY_MISSING:
            return (
                "Install the 1Password CLI "
                "(https://developer.1password.com/docs/cli/get-started/) or "
                "set secrets.onepassword.binary_path."
            )
        return super().remediation(kind, cfg)


def _classify_op_error(message: str) -> ErrorKind:
    """Best-effort mapping of op failure text onto the shared taxonomy."""
    lowered = message.lower()
    if "timed out" in lowered:
        return ErrorKind.TIMEOUT
    if (
        "not found on path" in lowered
        or "not an executable" in lowered
        or "failed to invoke" in lowered
    ):
        return ErrorKind.BINARY_MISSING
    if any(
        tok in lowered
        for tok in (
            "unauthorized",
            "not signed in",
            "session expired",
            "authentication",
            "401",
            "403",
        )
    ):
        return ErrorKind.AUTH_FAILED
    if "empty value" in lowered:
        return ErrorKind.EMPTY_VALUE
    if any(tok in lowered for tok in ("network", "connection", "resolve host", "dns")):
        return ErrorKind.NETWORK
    return ErrorKind.INTERNAL


# ---------------------------------------------------------------------------
# Test hook — used by hermetic tests to flush the cache between cases.
# ---------------------------------------------------------------------------


def clear_caches(home_path: Optional[Path] = None) -> None:
    """Drop in-process AND disk caches.

    Used after a token rotation (`hermes secrets onepassword token`) so
    the next startup resolves fresh with the new credential instead of
    serving values cached under the old token's fingerprint.
    """
    _CACHE.clear()
    _DISK_CACHE.clear(home_path)


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process AND disk caches.

    Tests can pass ``home_path`` to scope the disk cleanup to a tmpdir.
    Without it we fall back to the same default resolution as the writer.
    """
    clear_caches(home_path)
