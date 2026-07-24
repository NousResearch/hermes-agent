"""factory_lane.py — registry core cross-agent (HER-54).

Journal per-lane JSONL append-only en `registry/lanes/<KEY>.jsonl`, verrous
par répertoire atomique en `registry/locks/<KEY>/owner.json`, protégés par
`fcntl.flock` sur un fichier de verrou dédié. Python stdlib uniquement.
"""

import argparse
import calendar
import contextlib
import ctypes
import errno
import fcntl
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The admission hook invokes this script by its absolute path from arbitrary
# worktrees. Make the repository package root importable before loading cron's
# shared redaction boundary; relying on the caller's cwd/PYTHONPATH makes the
# documented CLI and the real pre-tool hook fail before they can veto a write.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from cron.redaction import contains_credential, redact_credential_text

# --------------------------------------------------------------------------
# Constantes de contrat
# --------------------------------------------------------------------------

KNOWN_PREFIXES = frozenset({"HER", "SCA", "IMP", "TL", "JYI"})
_TICKET_KEY_RE = re.compile(
    r"^(" + "|".join(sorted(KNOWN_PREFIXES)) + r")-[1-9][0-9]*\Z"
)
_ADHOC_KEY_RE = re.compile(r"^adhoc-[a-z0-9]+(?:-[a-z0-9]+)*\Z")

ALLOWED_MANUAL_EVENTS = frozenset({
    "build_started", "checkpoint", "pr_opened", "ci_green", "review_done",
    "ready_to_merge", "go_jean", "merged", "deployed", "smoke_ok",
    "memory_written", "lane_abandoned",
})

STATUS_LADDER = {
    "lane_claimed": "claimed",
    "build_started": "in_progress",
    "pr_opened": "pr_open",
    "lane_closed": "closed",
}

ALLOWED_HANDOFF_STATUS = frozenset({
    "ready_to_merge", "blocked", "in_progress", "needs_review", "done",
})

MAX_EVIDENCE_LEN = 200
MAX_NEXT_STEP_LEN = 1000
WORKTREE_ACTIVE_SECONDS = 24 * 3600
DEFAULT_TTL_HOURS = 72.0
MAX_FRESHNESS_EVIDENCE_BYTES = 32 * 1024
MAX_TEAM_CONFIG_BYTES = 32 * 1024
MAX_FRESHNESS_SOURCE_ITEMS = 100
# Freshness is operational proof, not durable history: one day is the largest
# controller window so a bad policy cannot make an epoch sentinel look current.
MAX_FRESHNESS_MAX_AGE_SECONDS = 24 * 3600
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_FRESHNESS_VERDICTS = frozenset({"current", "superseded", "duplicate", "needs-rebase"})
_FRESHNESS_EVIDENCE_FIELDS = frozenset({
    "issue", "checked_at", "canonical_branch", "canonical_head", "sources", "verdict",
})
_FRESHNESS_SOURCE_FIELDS = frozenset({
    "newer_linear_issues", "newer_prs_commits", "current_main_behavior",
})
_FRESHNESS_BEHAVIOR_FIELDS = frozenset({"checked", "summary"})
_FRESHNESS_RED_FIELDS = frozenset({"reproduced", "evidence"})

_SECRET_PATTERN = re.compile(
    r"(token|password|passwd|secret|api[_-]?key|credential)", re.IGNORECASE
)

# --------------------------------------------------------------------------
# Constantes context pack (HER-55 Scope A)
# --------------------------------------------------------------------------

CONTEXT_PACK_MAX_LINES = 300
UNMAPPED_BRIEF_MAX_LINES = 20
LINEAR_CACHE_STALE_HOURS = 24.0
UNMAPPED_REPO_MARKER = "UNMAPPED-REPO"
# Motifs de répertoire bloqués en défense en profondeur, même si allow-listés
# par erreur dans context-map.json. `_SECRET_PATTERN` couvre déjà
# secret/token/credential/password (insensible à la casse).
_VAULT_BLOCKED_DIR_PREFIXES = ("transcripts/", "Legal/", "Clients/")


class RegistryError(Exception):
    """Erreur de validation ou d'état attendue — provoque un exit non nul."""


# --------------------------------------------------------------------------
# Validation clé / chemins
# --------------------------------------------------------------------------

def validate_key(key):
    if not isinstance(key, str) or not key:
        raise RegistryError("key must be a non-empty string")
    if _TICKET_KEY_RE.match(key) or _ADHOC_KEY_RE.match(key):
        return
    raise RegistryError(f"invalid lane key: {key!r}")


_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK_FLAG = getattr(os, "O_NONBLOCK", 0)

# Alias système connus (macOS monte `/var`, `/tmp`, `/etc` comme symlinks vers
# `/private/...`). Ce ne sont pas des symlinks créés par un attaquant dans un
# chemin utilisateur : ils sont fixes, racine du système, et la résolution
# réelle est vérifiée avant de les tolérer.
_KNOWN_SYSTEM_ALIAS_ROOTS = frozenset({"/var", "/tmp", "/etc"})


def _is_known_system_alias(ancestor):
    """True seulement pour un alias macOS top-level résolvant vers /private/<nom>."""
    ancestor_str = str(ancestor)
    if ancestor_str not in _KNOWN_SYSTEM_ALIAS_ROOTS:
        return False
    try:
        resolved = os.path.realpath(ancestor_str)
    except OSError:
        return False
    return resolved == f"/private{ancestor_str}"


def _reject_symlink_ancestors(path):
    for ancestor in Path(path).parents:
        if ancestor.exists() and ancestor.is_symlink():
            if _is_known_system_alias(ancestor):
                continue
            raise RegistryError(f"path ancestor must not be a symlink: {ancestor}")


def _open_secure(path, flags, mode=0o600):
    """os.open with O_NOFOLLOW when available; refuses to traverse a symlink."""
    try:
        return os.open(str(path), flags | _NOFOLLOW_FLAG, mode)
    except OSError as exc:
        if _NOFOLLOW_FLAG and exc.errno in (errno.ELOOP, errno.EMLINK):
            raise RegistryError(f"refusing to follow symlink: {path}") from exc
        raise


def _reject_symlink(path, label):
    if Path(path).is_symlink():
        raise RegistryError(f"{label} must not be a symlink: {path}")


def _safe_registry_root(path_str):
    root = Path(path_str)
    _reject_symlink_ancestors(root)
    if root.is_symlink():
        raise RegistryError(f"registry path must not be a symlink: {root}")
    if root.exists():
        if not root.is_dir():
            raise RegistryError(f"registry path is not a directory: {root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _readonly_registry_root(path_str):
    """Inspect a registry root without creating it.

    The runtime admission hook is read-only: an absent root is healthy and
    advisory, while an existing but malformed or unreadable root is ambiguous
    registry state and must be surfaced to the hook as a fail-closed error.
    """
    root = Path(path_str)
    _reject_symlink_ancestors(root)
    if root.is_symlink():
        raise RegistryError(f"registry path must not be a symlink: {root}")
    if not root.exists():
        return None
    if not root.is_dir():
        raise RegistryError(f"registry path is not a directory: {root}")
    try:
        fd = _open_secure(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise RegistryError(f"registry root is unreadable: {root}: {exc}") from exc
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise RegistryError(f"registry path is not a directory: {root}")
    finally:
        os.close(fd)
    return root


def _safe_subdir(root, name):
    p = root / name
    if p.is_symlink():
        raise RegistryError(f"{name} directory must not be a symlink: {p}")
    if p.exists():
        if not p.is_dir():
            raise RegistryError(f"{name} path is not a directory: {p}")
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _validate_metadata_field(value, label):
    if len(value) > MAX_EVIDENCE_LEN:
        raise RegistryError(f"{label} too long")
    if _SECRET_PATTERN.search(value) or contains_credential(value):
        raise RegistryError(f"{label} looks like a credential and was rejected")
    if "\n" in value or "\r" in value:
        raise RegistryError(f"{label} must not contain newlines")


def _validate_evidence(evidence):
    _validate_metadata_field(evidence, "evidence")


# --------------------------------------------------------------------------
# I/O bas niveau : verrou, owner.json, journal JSONL
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _locked(lock_path):
    fd = _open_secure(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _read_json(path):
    fd = _open_secure(path, os.O_RDONLY | _NONBLOCK_FLAG)
    try:
        _require_regular_fd(fd, str(path))
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        return json.load(f)


def _fsync_dir(dir_path):
    """fsync du répertoire (no-follow) pour rendre un os.replace durable."""
    fd = _open_secure(dir_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _parse_jsonl_lines(text, source):
    """Parse fail-closed : toute ligne corrompue lève, aucune ligne n'est ignorée."""
    events = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RegistryError(
                f"corrupt journal {source} at line {lineno}: {exc}"
            ) from exc
    return events


def _read_all_events(lane_file):
    """Lit le journal via ouverture secure no-follow + flock partagé (cohérent
    avec le flock exclusif utilisé pendant l'append)."""
    fd = _open_regular_file_via_chain(lane_file, label=f"journal {lane_file}")
    return _read_all_events_fd(fd, lane_file)


def _read_all_events_at(parent_fd, name, source):
    """Read one regular JSONL journal relative to an anchored directory fd."""
    fd = _open_regular_file_at(parent_fd, name, label=f"journal {source}")
    return _read_all_events_fd(fd, source)


def _read_all_events_fd(fd, source):
    """Parse a JSONL journal already opened with no-follow semantics."""
    try:
        _require_regular_fd(fd, f"journal {source}")
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            text = f.read()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return _parse_jsonl_lines(text, source)


def _same_event(a, b):
    keys = (set(a.keys()) | set(b.keys())) - {"ts"}
    return all(a.get(k) == b.get(k) for k in keys)


# --------------------------------------------------------------------------
# I/O durci dirfd/openat — écriture claim/admit fail-closed contre un swap
# d'ancêtre symlink (registry/locks, registry/lanes) survenant APRÈS la
# validation Path. `O_NOFOLLOW` sur la feuille seule ne suffit pas : chaque
# composant du chemin est ré-ouvert via openat `O_NOFOLLOW|O_DIRECTORY` depuis
# un fd du registry root, donc un ancêtre swappé fait échouer l'openat
# (fail-closed) et aucune écriture ne peut sortir du registry.
# --------------------------------------------------------------------------

_ODIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_RENAME_SUPPORTS_DIR_FD = os.rename in os.supports_dir_fd
_OPATH_FLAG = getattr(os, "O_PATH", 0)
_CLOEXEC_FLAG = getattr(os, "O_CLOEXEC", 0)
_PROC_SELF_FD = "/proc/self/fd"
_SYMLINK_ERRNOS = (errno.ELOOP, errno.ENOTDIR, errno.EMLINK)

try:
    _native_libc = ctypes.CDLL(None, use_errno=True)
    _NATIVE_RENAMEAT = _native_libc.renameat
    _NATIVE_RENAMEAT.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    _NATIVE_RENAMEAT.restype = ctypes.c_int
except (AttributeError, OSError):
    _NATIVE_RENAMEAT = None


class _RegistryRootAnchor:
    """Trusted registry-root descriptor held for one logical operation."""

    def __init__(self, path, fd):
        self.path = os.fspath(path)
        self.fd = fd


def _assert_dirfd_matches_path(fd, path, label):
    """Reject a directory path replaced after its descriptor was opened."""
    try:
        path_stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RegistryError(f"{label} changed during operation: {exc}") from exc
    fd_stat = os.fstat(fd)
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise RegistryError(f"{label} changed during operation")


@contextlib.contextmanager
def _anchored_registry_root(root):
    """Keep one no-follow registry-root fd and reject a later path replacement."""
    if isinstance(root, _RegistryRootAnchor):
        _assert_dirfd_matches_path(root.fd, root.path, "registry root")
        root_path = root.path
        fd = os.dup(root.fd)
    else:
        root_path = os.fspath(root)
        fd = _open_secure(root_path, os.O_RDONLY | _ODIRECTORY_FLAG)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise RegistryError(f"registry path is not a directory: {root_path}")
        anchor = _RegistryRootAnchor(root_path, fd)
        _assert_dirfd_matches_path(anchor.fd, anchor.path, "registry root")
        yield anchor
    finally:
        os.close(fd)


def _ensure_registry_subdirs(root, *names):
    """Create required registry directories from an anchored root descriptor."""
    for name in names:
        fd = _open_dir_chain(root, (name,), create=True)
        os.close(fd)


@contextlib.contextmanager
def _registry_operation(root, *required_subdirs):
    """Anchor a registry root before all subsequent I/O and mutations."""
    with _anchored_registry_root(root) as anchor:
        _ensure_registry_subdirs(anchor, *required_subdirs)
        yield anchor


def _openat_subdir(parent_fd, name, create):
    """openat d'un sous-répertoire via `O_NOFOLLOW|O_DIRECTORY`. Lève
    RegistryError si `name` est un symlink (swap d'ancêtre). Propage
    FileNotFoundError si `name` est simplement absent et `create` est faux."""
    if create:
        with contextlib.suppress(FileExistsError):
            os.mkdir(name, 0o700, dir_fd=parent_fd)
    try:
        return os.open(
            name, os.O_RDONLY | _NOFOLLOW_FLAG | _ODIRECTORY_FLAG, dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno in _SYMLINK_ERRNOS:
            raise RegistryError(
                f"refusing symlinked registry directory: {name!r}"
            ) from exc
        raise


def _open_dir_chain(root_path, parts, create=False):
    """Descend depuis le registry root via openat `O_NOFOLLOW` à chaque composant.

    Retourne un fd du dernier répertoire (à fermer par l'appelant) ; ferme tous
    les fds intermédiaires. Ré-ouvrir cette chaîne à chaque écriture re-valide
    tous les ancêtres et referme la fenêtre TOCTOU du swap symlink."""
    if isinstance(root_path, _RegistryRootAnchor):
        _assert_dirfd_matches_path(root_path.fd, root_path.path, "registry root")
        fd = os.dup(root_path.fd)
    elif isinstance(root_path, int):
        fd = os.dup(root_path)
    else:
        fd = _open_secure(root_path, os.O_RDONLY | _ODIRECTORY_FLAG)
    try:
        for part in parts:
            nxt = _openat_subdir(fd, part, create)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_regular_file_at(parent_fd, name, *, label):
    """Return a readable fd only after pinning and validating a regular leaf."""
    if not _OPATH_FLAG:
        # Darwin has no O_PATH. A portable stat-then-open is only safe when the
        # already-pinned parent is not writable by other local UIDs: they cannot
        # swap the leaf after validation. Refuse an attacker-writable directory
        # before any content-capable leaf open.
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise RegistryError(f"safe parent for {label} is not a directory")
        if parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RegistryError(f"safe leaf pinning requires a trusted directory for {label}")
        if not _NOFOLLOW_FLAG:
            raise RegistryError(f"safe leaf pinning unavailable for {label}")
        try:
            fd = os.open(
                name,
                os.O_RDONLY | _NOFOLLOW_FLAG | _NONBLOCK_FLAG | _CLOEXEC_FLAG,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                raise
            if exc.errno in _SYMLINK_ERRNOS:
                raise RegistryError(f"{label} must not be a symlink") from exc
            raise RegistryError(f"failed to open {label}: {exc}") from exc
        try:
            _require_regular_fd(fd, label)
            return fd
        except BaseException:
            os.close(fd)
            raise

    try:
        pinned_fd = os.open(
            name,
            _OPATH_FLAG | _NOFOLLOW_FLAG | _CLOEXEC_FLAG,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise
        raise RegistryError(f"failed to pin {label}: {exc}") from exc
    try:
        pinned_stat = os.fstat(pinned_fd)
        if not stat.S_ISREG(pinned_stat.st_mode):
            raise RegistryError(f"{label} must be a regular non-symlink file")
        try:
            fd = os.open(
                f"{_PROC_SELF_FD}/{pinned_fd}",
                os.O_RDONLY | _NONBLOCK_FLAG | _CLOEXEC_FLAG,
            )
        except OSError as exc:
            raise RegistryError(f"failed to reopen pinned {label}: {exc}") from exc
        try:
            reopened_stat = os.fstat(fd)
            if (
                not stat.S_ISREG(reopened_stat.st_mode)
                or (pinned_stat.st_dev, pinned_stat.st_ino)
                != (reopened_stat.st_dev, reopened_stat.st_ino)
            ):
                raise RegistryError(f"{label} changed during secure open")
            return fd
        except BaseException:
            os.close(fd)
            raise
    except BaseException:
        raise
    finally:
        os.close(pinned_fd)


def _read_json_at(parent_fd, name):
    fd = _open_regular_file_at(parent_fd, name, label=f"registry file {name}")
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_absolute_path(path):
    """Return an absolute lexical path without resolving user-controlled links."""
    absolute = os.path.abspath(os.fspath(path))
    for alias in _KNOWN_SYSTEM_ALIAS_ROOTS:
        if (_is_known_system_alias(Path(alias))
                and (absolute == alias or absolute.startswith(f"{alias}{os.sep}"))):
            return f"/private{absolute}"
    return absolute


def _open_regular_file_via_chain(path, *, label="freshness evidence"):
    """Pin every ancestor before opening an external regular-file input."""
    parts = Path(_safe_absolute_path(path)).parts
    if len(parts) < 2:
        raise RegistryError(f"{label} must name a regular file")
    parent_fd = _open_secure(parts[0], os.O_RDONLY | _ODIRECTORY_FLAG)
    try:
        for part in parts[1:-1]:
            next_fd = _openat_subdir(parent_fd, part, create=False)
            os.close(parent_fd)
            parent_fd = next_fd
        try:
            return _open_regular_file_at(parent_fd, parts[-1], label=label)
        except OSError as exc:
            if exc.errno in _SYMLINK_ERRNOS:
                raise RegistryError(f"{label} must not be a symlink") from exc
            raise RegistryError(f"failed to open {label}: {exc}") from exc
    finally:
        os.close(parent_fd)


def _require_regular_fd(fd, label):
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise RegistryError(f"{label} is not a regular file")


def _atomic_replace_at(parent_fd, parent_path, tmp_name, final_name):
    """Rename atomique tmp -> final à l'intérieur de `parent_fd`.

    The replace is always relative to the directory fd.  This deliberately
    avoids a post-validation text-path rename on macOS: a swapped ancestor may
    change a path string, but cannot redirect an already-open directory fd.
    """
    del parent_path  # kept in the signature for existing callers and diagnostics.
    if _RENAME_SUPPORTS_DIR_FD:
        os.rename(tmp_name, final_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        return
    if _NATIVE_RENAMEAT is None:
        raise RegistryError("no safe dirfd rename primitive is available")
    if _NATIVE_RENAMEAT(parent_fd, os.fsencode(tmp_name), parent_fd, os.fsencode(final_name)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), final_name)


def _write_json_at(parent_fd, parent_path, name, obj, mode=0o600):
    """Écrit `obj` (JSON) sous `name` dans `parent_fd`, atomiquement."""
    tmp_name = f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    fd = os.open(
        tmp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW_FLAG, mode,
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace_at(parent_fd, parent_path, tmp_name, name)
        os.fsync(parent_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name, dir_fd=parent_fd)
        raise


def _write_text_at(parent_fd, parent_path, name, content, mode=0o600):
    """Write text atomically inside an already verified directory fd."""
    tmp_name = f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    fd = os.open(
        tmp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _NOFOLLOW_FLAG, mode,
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace_at(parent_fd, parent_path, tmp_name, name)
        os.fsync(parent_fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name, dir_fd=parent_fd)
        raise


def _append_event_at(parent_fd, name, source_label, key, event_name, extra=None):
    payload = {"ts": time.time(), "key": key, "event": event_name}
    if extra:
        payload.update(extra)
    fd = os.open(
        name, os.O_CREAT | os.O_RDWR | _NOFOLLOW_FLAG | _NONBLOCK_FLAG, 0o600, dir_fd=parent_fd,
    )
    try:
        _require_regular_fd(fd, f"journal {source_label}")
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            os.fchmod(f.fileno(), 0o600)
            events = _parse_jsonl_lines(f.read(), source_label)
            last = events[-1] if events else None
            if last is not None and _same_event(last, payload):
                return
            f.seek(0, os.SEEK_END)
            f.write(json.dumps(payload, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _dirfd_flock(root_path, parts, name):
    """flock exclusif sur `name` sous la chaîne dirfd `parts` (créée sûrement)."""
    parent_fd = _open_dir_chain(root_path, parts, create=True)
    try:
        fd = os.open(
            name, os.O_CREAT | os.O_RDWR | _NOFOLLOW_FLAG, 0o600, dir_fd=parent_fd,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
    finally:
        os.close(parent_fd)


def _read_owner_via_chain(root_path, key):
    """owner.json de la lane `key` via chaîne dirfd, ou None si absent."""
    try:
        key_fd = _open_dir_chain(root_path, ("locks", key), create=False)
    except FileNotFoundError:
        return None
    try:
        return _read_json_at(key_fd, "owner.json")
    except FileNotFoundError:
        return None
    finally:
        os.close(key_fd)


def _root_text_path(root_path):
    return root_path.path if isinstance(root_path, _RegistryRootAnchor) else os.fspath(root_path)


def _write_owner_via_chain(root_path, key, owner):
    key_fd = _open_dir_chain(root_path, ("locks", key), create=True)
    try:
        parent_path = os.path.join(_root_text_path(root_path), "locks", key)
        _write_json_at(key_fd, parent_path, "owner.json", owner)
    finally:
        os.close(key_fd)


def _unlink_owner_via_chain(root_path, key):
    try:
        key_fd = _open_dir_chain(root_path, ("locks", key), create=False)
    except FileNotFoundError:
        return
    try:
        with contextlib.suppress(FileNotFoundError):
            os.unlink("owner.json", dir_fd=key_fd)
        os.fsync(key_fd)
    finally:
        os.close(key_fd)


def _append_event_via_chain(root_path, key, event_name, extra=None):
    lanes_fd = _open_dir_chain(root_path, ("lanes",), create=True)
    try:
        _append_event_at(
            lanes_fd, f"{key}.jsonl",
            os.path.join(_root_text_path(root_path), "lanes", f"{key}.jsonl"),
            key, event_name, extra=extra,
        )
    finally:
        os.close(lanes_fd)


def _compute_status(events):
    status = "unclaimed"
    for e in events:
        name = e.get("event")
        if name in STATUS_LADDER:
            status = STATUS_LADDER[name]
    return status


# --------------------------------------------------------------------------
# Identité de processus / activité worktree (pour la réclamation de lock)
# --------------------------------------------------------------------------

def _get_process_start_time(pid):
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def _get_process_state_char(pid):
    try:
        result = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out[:1] if out else None


def determine_process_state(owner):
    """Retourne 'alive' | 'zombie' | 'reused' | 'not_found'."""
    pid = owner.get("pid")
    if not isinstance(pid, int):
        return "not_found"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "not_found"
    except PermissionError:
        pass
    except OSError:
        return "not_found"

    if _get_process_state_char(pid) == "Z":
        return "zombie"

    current_start = _get_process_start_time(pid)
    recorded_start = owner.get("process_start_time")
    # Sans empreinte de départ enregistrée, la réutilisation de PID est
    # indémontrable : un process vivant reste 'alive' (jamais 'reused'), sinon
    # un owner vivant sans baseline deviendrait faussement réclamable.
    if recorded_start is None:
        return "alive"
    if current_start is not None and recorded_start != current_start:
        return "reused"
    return "alive"


_WORKTREE_SCAN_MAX_ENTRIES = 5000
_WORKTREE_SCAN_MAX_DEPTH = 8


def _get_worktree_last_active(path):
    """Dernière activité du worktree, avec refus conservateur d'un scan partiel.

    Ne lit aucun contenu et ne suit aucun symlink. Si la profondeur ou le
    nombre maximal d'entrées empêche un scan complet, retourne l'instant
    courant : le worktree est alors considéré actif et le reclaim est refusé.
    """
    if not path:
        return None
    root = Path(path)
    try:
        last = os.stat(root, follow_symlinks=False).st_mtime
    except OSError:
        return None

    remaining = [_WORKTREE_SCAN_MAX_ENTRIES]
    scan_complete = [True]

    def _scan(dir_path, depth):
        nonlocal last
        if depth > _WORKTREE_SCAN_MAX_DEPTH:
            scan_complete[0] = False
            return
        try:
            with os.scandir(dir_path) as it:
                for entry in it:
                    if remaining[0] <= 0:
                        scan_complete[0] = False
                        return
                    remaining[0] -= 1
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        scan_complete[0] = False
                        continue
                    if st.st_mtime > last:
                        last = st.st_mtime
                    if entry.is_dir(follow_symlinks=False):
                        _scan(entry.path, depth + 1)
        except OSError:
            scan_complete[0] = False

    _scan(root, 0)
    return time.time() if not scan_complete[0] else last


def evaluate_reclaim(now, owner, process_state, worktree_last_active):
    """Fonction pure : décide si un lock détenu par `owner` est réclamable.

    Réclamable seulement si le processus n'est PAS vivant, le TTL basé sur
    heartbeat est expiré, ET le worktree n'a pas été touché récemment.
    """
    ttl_seconds = owner["ttl_hours"] * 3600
    heartbeat_age = now - owner["heartbeat_at"]
    ttl_expired = heartbeat_age > ttl_seconds
    worktree_active = (
        worktree_last_active is not None
        and (now - worktree_last_active) < WORKTREE_ACTIVE_SECONDS
    )
    process_stale = process_state in ("zombie", "reused", "not_found")
    reclaimable = process_stale and ttl_expired and not worktree_active
    return {
        "reclaimable": reclaimable,
        "process_stale": process_stale,
        "ttl_expired": ttl_expired,
        "worktree_active": worktree_active,
    }


def _canonical_worktree(worktree):
    return os.path.realpath(str(worktree))


def _resolve_owner_identity(owner_pid, owner_start_time):
    """Valide une identité de process parent transportée par l'appelant.

    Le hard gate/gateway lance `factory_lane.py` en subprocess éphémère : si on
    persistait `os.getpid()`, l'owner porterait un PID mort dès la fin de la
    commande et deviendrait immédiatement réclamable. L'appelant transporte donc
    l'identité de l'agent parent de longue durée via `--owner-pid`
    (+ `--owner-start-time` optionnel, anti-usurpation).

    Retourne `(pid, start_time)` validés, ou `(None, None)` si aucune identité
    n'est transportée (mode CLI autonome — l'appelant EST le process propriétaire).
    Lève `RegistryError` pour un PID absent/mort ou une empreinte incohérente :
    on ne crée jamais un owner déjà mort.
    """
    if owner_pid is None:
        return None, None
    if not isinstance(owner_pid, int) or isinstance(owner_pid, bool) or owner_pid <= 0:
        raise RegistryError(f"invalid owner-pid: {owner_pid!r}")
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError as exc:
        raise RegistryError(f"owner-pid {owner_pid} is not a live process") from exc
    except PermissionError:
        pass  # vivant mais appartenant à un autre utilisateur
    except OSError as exc:
        raise RegistryError(f"owner-pid {owner_pid} is not usable: {exc}") from exc

    actual_start = _get_process_start_time(owner_pid)
    if owner_start_time is not None:
        if actual_start is None or owner_start_time != actual_start:
            raise RegistryError("owner-start-time does not match owner-pid start time")
        return owner_pid, owner_start_time
    return owner_pid, actual_start


def _build_owner(agent, session, worktree, ttl_hours, now, profile=None,
                 gateway_session_key=None, owner_pid=None, owner_start_time=None,
                 team=None, allowed_teams=None, freshness=None):
    if owner_pid is not None:
        pid = owner_pid
        start_time = owner_start_time
    else:
        pid = os.getpid()
        start_time = _get_process_start_time(pid)
    owner = {
        "host": socket.gethostname(),
        "agent": agent,
        "session_id": session,
        "pid": pid,
        "process_start_time": start_time,
        "started_at": now,
        "heartbeat_at": now,
        "ttl_hours": ttl_hours,
        "worktree": _canonical_worktree(worktree),
    }
    if profile:
        owner["profile"] = profile
    if team:
        owner["team"] = team
    if allowed_teams is not None:
        owner["allowed_teams"] = allowed_teams
    if freshness is not None:
        owner["freshness"] = freshness
    if gateway_session_key:
        owner["gateway_session_key"] = gateway_session_key
    return owner


def _is_same_session(owner, agent, session):
    return owner.get("agent") == agent and owner.get("session_id") == session


def _iter_owners_via_chain(root_path):
    """Yield owner records through pinned `locks/` and lane directory fds."""
    locks_fd = _open_dir_chain(root_path, ("locks",), create=False)
    try:
        for key in sorted(os.listdir(locks_fd)):
            validate_key(key)
            key_fd = _openat_subdir(locks_fd, key, create=False)
            try:
                try:
                    owner = _read_json_at(key_fd, "owner.json")
                except FileNotFoundError:
                    continue
                if not isinstance(owner, dict):
                    raise RegistryError(f"owner.json for {key} must be a JSON object")
                yield key, owner
            finally:
                os.close(key_fd)
    finally:
        os.close(locks_fd)


def _find_worktree_claim(root_path, worktree_real):
    for key, owner in _iter_owners_via_chain(root_path):
        claimed = owner.get("worktree")
        if claimed and _canonical_worktree(claimed) == worktree_real:
            return key, owner
    return None


def _find_team_claim(root_path, team):
    if not team:
        return None
    for key, owner in _iter_owners_via_chain(root_path):
        if owner.get("team") == team:
            return key, owner
    return None


def _can_reclaim_owner(now, owner):
    process_state = determine_process_state(owner)
    worktree_last_active = _get_worktree_last_active(owner.get("worktree"))
    verdict = evaluate_reclaim(
        now=now,
        owner=owner,
        process_state=process_state,
        worktree_last_active=worktree_last_active,
    )
    return verdict["reclaimable"]


def _git_dirty(repo):
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=str(repo), capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


# --------------------------------------------------------------------------
# Commandes
# --------------------------------------------------------------------------

def cmd_preflight(root, key, as_json):
    validate_key(key)
    lanes_dir = _safe_subdir(root, "lanes")
    locks_root = _safe_subdir(root, "locks")
    lane_file = lanes_dir / f"{key}.jsonl"

    if lane_file.exists():
        events = _read_all_events(lane_file)
        status = _compute_status(events)
    else:
        status = "unclaimed"

    owner_file = locks_root / key / "owner.json"
    payload = {"key": key, "status": status, "active": owner_file.exists()}
    if as_json:
        print(json.dumps(payload))
    else:
        print(f"{key}: {status}")
    return 0


def _validate_ttl_hours(ttl_hours):
    if not isinstance(ttl_hours, (int, float)) or isinstance(ttl_hours, bool):
        raise RegistryError(f"invalid ttl-hours: {ttl_hours!r}")
    if not math.isfinite(ttl_hours) or ttl_hours <= 0:
        raise RegistryError(f"invalid ttl-hours: {ttl_hours!r}")


def _claim_under_gate(root, key, agent, session, worktree, reclaim, ttl_hours,
                      profile=None, gateway_session_key=None,
                      owner_pid=None, owner_start_time=None, team=None,
                      allowed_teams=None, freshness=None):
    validate_key(key)
    _validate_ttl_hours(ttl_hours)
    # Un `gateway_session_key` ressemblant à un secret (token/password/api_key…)
    # ne doit jamais être persisté en clair dans owner.json — rejet fail-closed.
    if gateway_session_key is not None:
        _validate_metadata_field(gateway_session_key, "gateway_session_key")
    resolved_pid, resolved_start = _resolve_owner_identity(owner_pid, owner_start_time)

    with _registry_operation(root, "lanes", "locks") as root_anchor:
        worktree_real = _canonical_worktree(worktree)

        # Verrou machine-wide (préflight->claim TOCTOU) + verrou par-lane, tous deux
        # ouverts via chaîne dirfd O_NOFOLLOW ; toute écriture (owner.json, journal)
        # ré-ouvre sa chaîne et échoue fermé si un ancêtre a été swappé en symlink.
        with _dirfd_flock(root_anchor, (), ".worktree-admission.lock"), \
                _dirfd_flock(root_anchor, ("locks", key), ".lock"):
            now = time.time()
            owner = _read_owner_via_chain(root_anchor, key)
            if owner is not None:
                if _is_same_session(owner, agent, session):
                    current_wt = owner.get("worktree")
                    if not current_wt or _canonical_worktree(current_wt) != worktree_real:
                        raise RegistryError(
                            "worktree already claimed by this same session from a different worktree"
                        )
                    if resolved_pid is not None:
                        owner["pid"] = resolved_pid
                        owner["process_start_time"] = resolved_start
                    if team:
                        owner["team"] = team
                    if allowed_teams is not None:
                        owner["allowed_teams"] = allowed_teams
                    if freshness is not None:
                        owner["freshness"] = freshness
                    owner["heartbeat_at"] = now
                    _write_owner_via_chain(root_anchor, key, owner)
                    return 0

                if not reclaim:
                    raise RegistryError(
                        f"lane {key} already claimed by "
                        f"{owner.get('agent')}/{owner.get('session_id')}"
                    )

                if not _can_reclaim_owner(now, owner):
                    raise RegistryError(
                        f"lane {key} owner still active, refusing --reclaim"
                    )

                previous_agent = owner.get("agent")
                previous_session = owner.get("session_id")
                new_owner = _build_owner(
                    agent, session, worktree_real, ttl_hours, now,
                    profile=profile, gateway_session_key=gateway_session_key,
                    owner_pid=resolved_pid, owner_start_time=resolved_start,
                    team=team, allowed_teams=allowed_teams, freshness=freshness,
                )
                _write_owner_via_chain(root_anchor, key, new_owner)
                _append_event_via_chain(root_anchor, key, "lock_reclaimed", extra={
                    "previous_agent": previous_agent,
                    "previous_session": previous_session,
                })
                return 0

            conflict = _find_claim_for_worktree(root_anchor, worktree_real)
            if conflict is not None:
                conflict_key, conflict_owner = conflict
                if conflict_key != key:
                    if not reclaim:
                        raise RegistryError(
                            f"worktree already claimed by {conflict_key} "
                            f"{conflict_owner.get('agent')}/{conflict_owner.get('session_id')}"
                        )
                    if not _can_reclaim_owner(now, conflict_owner):
                        raise RegistryError(
                            f"worktree already claimed by active owner {conflict_key}"
                        )
                    _unlink_owner_via_chain(root_anchor, conflict_key)

            team_conflict = _find_team_claim(root_anchor, team)
            if team_conflict is not None and team_conflict[0] != key:
                conflict_key, conflict_owner = team_conflict
                if not reclaim or not _can_reclaim_owner(now, conflict_owner):
                    raise RegistryError(
                        f"team {team} already claimed by {conflict_key} "
                        f"{conflict_owner.get('agent')}/{conflict_owner.get('session_id')}"
                    )
                _unlink_owner_via_chain(root_anchor, conflict_key)

            new_owner = _build_owner(
                agent, session, worktree_real, ttl_hours, now,
                profile=profile, gateway_session_key=gateway_session_key,
                owner_pid=resolved_pid, owner_start_time=resolved_start,
                team=team, allowed_teams=allowed_teams, freshness=freshness,
            )
            _write_owner_via_chain(root_anchor, key, new_owner)
            _append_event_via_chain(root_anchor, key, "lane_claimed")
            return 0


def cmd_claim(root, key, agent, session, worktree, reclaim, ttl_hours,
              profile=None, gateway_session_key=None,
              owner_pid=None, owner_start_time=None):
    return _claim_under_gate(
        root, key, agent, session, worktree, reclaim, ttl_hours,
        profile=profile, gateway_session_key=gateway_session_key,
        owner_pid=owner_pid, owner_start_time=owner_start_time,
    )


def _lane_prefix(key):
    return key.split("-", 1)[0]


def _validate_profile_domain(key, profile, domain_prefixes):
    if not profile or not domain_prefixes:
        return
    allowed = {part.strip() for part in domain_prefixes.split(",") if part.strip()}
    if allowed and _lane_prefix(key) not in allowed:
        raise RegistryError(f"profile {profile} cannot own lane {key}")


def _bounded_json_file(path, label, maximum):
    try:
        fd = _open_regular_file_via_chain(path, label=label)
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            if os.fstat(f.fileno()).st_size > maximum:
                raise RegistryError(f"{label} is too large")
            raw = f.read(maximum + 1)
    except RegistryError as exc:
        if label in str(exc):
            raise
        raise RegistryError(f"failed to read {label}: {exc}") from exc
    except OSError as exc:
        raise RegistryError(f"failed to read {label}: {exc}") from exc
    if len(raw.encode("utf-8")) > maximum:
        raise RegistryError(f"{label} is too large")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{label} is not valid JSON: {exc}") from exc


def load_team_config(path):
    data = _bounded_json_file(path, "team config", MAX_TEAM_CONFIG_BYTES)
    if not isinstance(data, dict) or set(data) != {"freshness", "teams"}:
        raise RegistryError("team config contains unknown or missing fields")
    freshness, teams = data["freshness"], data["teams"]
    if not isinstance(freshness, dict) or set(freshness) != {"canonical_branch", "max_age_seconds"}:
        raise RegistryError("freshness config contains unknown or missing fields")
    branch, maximum_age = freshness["canonical_branch"], freshness["max_age_seconds"]
    if not isinstance(branch, str) or not branch:
        raise RegistryError("freshness.canonical_branch must be a non-empty string")
    if (isinstance(maximum_age, bool) or not isinstance(maximum_age, int)
            or not 0 < maximum_age <= MAX_FRESHNESS_MAX_AGE_SECONDS):
        raise RegistryError(
            "freshness.max_age_seconds must be a whole number between 1 and "
            f"{MAX_FRESHNESS_MAX_AGE_SECONDS}"
        )
    if not isinstance(teams, dict) or set(teams) != {"HER", "SCA"}:
        raise RegistryError("dual-team config must define exactly HER and SCA teams")
    normalized = {"freshness": {"canonical_branch": branch, "max_age_seconds": maximum_age}, "teams": {}}
    profiles = set()
    for team, spec in teams.items():
        if not isinstance(spec, dict) or set(spec) != {"profiles", "allowed_teams", "job_id", "gateway_started_at"}:
            raise RegistryError(f"team {team} config contains unknown or missing fields")
        mapped_profiles, allowed_teams = spec["profiles"], spec["allowed_teams"]
        if (not isinstance(mapped_profiles, list) or len(mapped_profiles) != 1
                or not isinstance(mapped_profiles[0], str) or not mapped_profiles[0]):
            raise RegistryError(f"team {team} must map exactly one profile")
        if mapped_profiles[0] in profiles:
            raise RegistryError("each team profile must have one controller")
        profiles.add(mapped_profiles[0])
        if (not isinstance(allowed_teams, list) or not allowed_teams
                or any(item not in {"HER", "SCA"} for item in allowed_teams)
                or len(set(allowed_teams)) != len(allowed_teams)):
            raise RegistryError(f"team {team} allowed_teams must be a non-empty unique HER/SCA list")
        if not isinstance(spec["job_id"], str) or not spec["job_id"]:
            raise RegistryError(f"team {team} job_id must be a non-empty string")
        if _parse_handoff_timestamp(spec["gateway_started_at"]) is None:
            raise RegistryError(f"team {team} gateway_started_at must be an ISO8601 UTC timestamp")
        normalized["teams"][team] = {
            "profiles": mapped_profiles,
            "allowed_teams": allowed_teams,
            "job_id": spec["job_id"],
            "gateway_started_at": spec["gateway_started_at"],
        }
    return normalized


def _resolve_team_for_profile(config, profile):
    matches = [(team, spec) for team, spec in config["teams"].items() if profile in spec["profiles"]]
    if not matches:
        raise RegistryError(f"profile {profile} is not mapped to any team")
    if len(matches) != 1:
        raise RegistryError(f"profile {profile} maps to multiple teams")
    return matches[0]


def _load_freshness_evidence(path):
    evidence = _bounded_json_file(path, "freshness evidence", MAX_FRESHNESS_EVIDENCE_BYTES)
    if not isinstance(evidence, dict):
        raise RegistryError("freshness evidence must be a JSON object")
    return evidence


def _validate_freshness_evidence(evidence, key, policy):
    if evidence.get("verdict") == "needs-rebase" and "current_main_red" not in evidence:
        raise RegistryError("needs-rebase freshness requires current-main RED evidence")
    allowed_fields = set(_FRESHNESS_EVIDENCE_FIELDS)
    if evidence.get("verdict") == "needs-rebase":
        allowed_fields.add("current_main_red")
    if set(evidence) != allowed_fields:
        raise RegistryError("freshness evidence contains unknown or missing fields")
    if evidence["issue"] != key:
        raise RegistryError("freshness evidence issue must match lane key")
    raw_checked_at = evidence["checked_at"]
    if isinstance(raw_checked_at, float) and not math.isfinite(raw_checked_at):
        raise RegistryError("freshness checked_at must be finite")
    checked_at = _parse_handoff_timestamp(raw_checked_at)
    if checked_at is None or checked_at <= 0:
        raise RegistryError("freshness checked_at must be after epoch")
    if (checked_at > time.time() + 60
            or time.time() - checked_at > policy["max_age_seconds"]):
        raise RegistryError("freshness evidence is stale")
    canonical_branch = evidence["canonical_branch"]
    if not isinstance(canonical_branch, str):
        raise RegistryError("freshness evidence canonical branch must be a string")
    _validate_metadata_field(canonical_branch, "freshness canonical branch")
    if canonical_branch != policy["canonical_branch"]:
        raise RegistryError("freshness evidence canonical branch does not match controller config")
    canonical_head = evidence["canonical_head"]
    if not isinstance(canonical_head, str) or not _SHA40_RE.fullmatch(canonical_head):
        raise RegistryError("freshness evidence canonical_head must be a lowercase 40-character SHA")
    sources = evidence["sources"]
    if not isinstance(sources, dict) or set(sources) != _FRESHNESS_SOURCE_FIELDS:
        raise RegistryError("freshness evidence must record checked sources")
    linear_issues = sources["newer_linear_issues"]
    prs_commits = sources["newer_prs_commits"]
    if not isinstance(linear_issues, list) or not isinstance(prs_commits, list):
        raise RegistryError("freshness evidence sources must be lists")
    if len(linear_issues) > MAX_FRESHNESS_SOURCE_ITEMS or len(prs_commits) > MAX_FRESHNESS_SOURCE_ITEMS:
        raise RegistryError("freshness sources have too many items")
    if (any(not isinstance(item, str) or not _TICKET_KEY_RE.fullmatch(item) for item in linear_issues)
            or any(not isinstance(item, str) or not _SHA40_RE.fullmatch(item) for item in prs_commits)):
        raise RegistryError("freshness sources contain an invalid item")
    behavior = sources["current_main_behavior"]
    if not isinstance(behavior, dict) or set(behavior) != _FRESHNESS_BEHAVIOR_FIELDS:
        raise RegistryError("freshness evidence current-main behavior contains unknown or missing fields")
    if behavior["checked"] is not True or not isinstance(behavior["summary"], str) or not behavior["summary"]:
        raise RegistryError("freshness evidence current-main behavior must be checked")
    _validate_metadata_field(behavior["summary"], "freshness current-main summary")
    verdict = evidence["verdict"]
    if verdict not in _FRESHNESS_VERDICTS:
        raise RegistryError("freshness verdict must be current, superseded, duplicate, or needs-rebase")
    if verdict in {"superseded", "duplicate"}:
        raise RegistryError(f"freshness verdict {verdict} blocks build admission")
    normalized = {
        "issue": key,
        "checked_at": checked_at,
        "canonical_branch": canonical_branch,
        "canonical_head": canonical_head,
        "sources": {
            "newer_linear_issues": list(linear_issues),
            "newer_prs_commits": list(prs_commits),
            "current_main_behavior": {"checked": True, "summary": behavior["summary"]},
        },
        "verdict": verdict,
    }
    if verdict == "needs-rebase":
        red = evidence["current_main_red"]
        if not isinstance(red, dict) or set(red) != _FRESHNESS_RED_FIELDS:
            raise RegistryError("needs-rebase freshness current-main RED contains unknown or missing fields")
        if red["reproduced"] is not True or not isinstance(red["evidence"], str) or not red["evidence"]:
            raise RegistryError("needs-rebase freshness requires current-main RED evidence")
        _validate_metadata_field(red["evidence"], "freshness current-main RED evidence")
        normalized["current_main_red"] = {"reproduced": True, "evidence": red["evidence"]}
    return normalized


def cmd_team_admit(root, key, team_config_path, profile, agent, session, worktree,
                   ttl_hours=DEFAULT_TTL_HOURS, gateway_session_key=None,
                   owner_pid=None, owner_start_time=None, reclaim=False,
                   freshness_evidence_path=None):
    validate_key(key)
    team, spec = _resolve_team_for_profile(load_team_config(team_config_path), profile)
    if _lane_prefix(key) not in spec["allowed_teams"]:
        raise RegistryError(f"profile {profile} cannot own lane {key}")
    if not freshness_evidence_path:
        raise RegistryError("--freshness-evidence is required for team admission")
    config = load_team_config(team_config_path)
    freshness = _validate_freshness_evidence(_load_freshness_evidence(freshness_evidence_path), key, config["freshness"])
    return _claim_under_gate(root, key, agent, session, worktree, reclaim, ttl_hours,
                             profile=profile, gateway_session_key=gateway_session_key,
                             owner_pid=owner_pid, owner_start_time=owner_start_time,
                             team=team, allowed_teams=spec["allowed_teams"], freshness=freshness)


def _load_cron_jobs_by_id():
    try:
        from cron.jobs import list_jobs
        return {job.get("id"): job for job in list_jobs(include_disabled=True)}
    except Exception:
        return {}


def _load_latest_cron_executions(job_ids):
    try:
        from cron.executions import latest_executions
        return latest_executions(list(job_ids))
    except Exception:
        return {}


def _load_latest_builtin_cron_executions(job_ids):
    try:
        from cron.executions import latest_builtin_executions
        return latest_builtin_executions(list(job_ids))
    except Exception:
        return {}


def _execution_after_gateway_start(record, gateway_started_at):
    gateway_started = _parse_handoff_timestamp(gateway_started_at)
    if not record or gateway_started is None:
        return None
    for field in ("finished_at", "started_at", "claimed_at"):
        timestamp = _parse_handoff_timestamp(record.get(field))
        if timestamp is not None:
            return record if timestamp >= gateway_started else None
    return None


_STATUS_EXECUTION_FIELDS = (
    "id", "job_id", "source", "status", "claimed_at", "started_at", "finished_at",
)


def _project_execution_status(record):
    """Expose bounded execution state without replaying untrusted cron errors."""
    if not isinstance(record, dict):
        return None
    projected = {}
    for field in _STATUS_EXECUTION_FIELDS:
        value = record.get(field)
        projected[field] = redact_credential_text(value) if isinstance(value, str) else value
    # A status consumer only needs to know that the attempt failed. The durable
    # error is intentionally not copied, even after persistence-time redaction.
    projected["error"] = "redacted" if record.get("error") is not None else None
    return projected


def build_team_status(root, config):
    _safe_subdir(root, "locks")
    jobs = _load_cron_jobs_by_id()
    ids = [spec["job_id"] for spec in config["teams"].values()]
    latest, builtin = _load_latest_cron_executions(ids), _load_latest_builtin_cron_executions(ids)
    for job_id, record in latest.items():
        if record.get("source") == "builtin":
            builtin.setdefault(job_id, record)
    teams = {}
    for team, spec in config["teams"].items():
        claim = _find_team_claim(root, team)
        owner = claim[1] if claim else None
        job_id, job = spec["job_id"], jobs.get(spec["job_id"])
        teams[team] = {
            "team": team,
            "profile": owner.get("profile") if owner else spec["profiles"][0],
            "lane": claim[0] if claim else None,
            "worktree": owner.get("worktree") if owner else None,
            "runtime_status_source": "registry" if owner else "unclaimed",
            "job_id": job_id,
            "next_run_at": job.get("next_run_at") if job else None,
            "latest_execution": _project_execution_status(latest.get(job_id)),
            "last_builtin_execution": _project_execution_status(builtin.get(job_id)),
            "last_builtin_tick_after_gateway_start": _project_execution_status(
                _execution_after_gateway_start(builtin.get(job_id), spec["gateway_started_at"]),
            ),
            "worker": {"agent": owner.get("agent"), "pid": owner.get("pid"), "process_state": determine_process_state(owner)} if owner else None,
            "heartbeat": owner.get("heartbeat_at") if owner else None,
            "gate": {"freshness": owner.get("freshness") if owner else None},
        }
    return {"teams": teams}


def cmd_team_status(root, team_config_path, as_json=False):
    payload = build_team_status(root, load_team_config(team_config_path))
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for team, status in payload["teams"].items():
            print(f"{team}: {status['lane'] or '-'} profile={status['profile']}")
    return 0


def cmd_admit(root, key, mode, hard, agent, session, worktree, ttl_hours,
              as_json=False, profile=None, gateway_session_key=None,
              domain_prefixes=None, owner_pid=None, owner_start_time=None):
    validate_key(key)
    _validate_ttl_hours(ttl_hours)
    worktree_real = _canonical_worktree(worktree)
    with _registry_operation(root, "locks", "lanes") as root_anchor:
        with _dirfd_flock(root_anchor, (), ".worktree-admission.lock"):
            conflict = _find_claim_for_worktree(root_anchor, worktree_real)
            if mode == "reviewer":
                payload = {
                    "key": key,
                    "worktree": worktree_real,
                    "decision": "reviewer_allowed",
                }
                if conflict:
                    payload["owner_key"] = conflict[0]
                    payload["owner_agent"] = conflict[1].get("agent")
                    payload["owner_session"] = conflict[1].get("session_id")
                if as_json:
                    print(json.dumps(payload, sort_keys=True))
                return 0

            _validate_profile_domain(key, profile, domain_prefixes)
            if hard and conflict is None and _git_dirty(worktree_real):
                raise RegistryError(f"dirty ownerless worktree: {worktree_real}")

        return _claim_under_gate(
            root_anchor, key, agent, session, worktree_real, False, ttl_hours,
            profile=profile, gateway_session_key=gateway_session_key,
            owner_pid=owner_pid, owner_start_time=owner_start_time,
        )


# --------------------------------------------------------------------------
# Hard gate pré-mutation (lecture seule) — appelé par le hook `pre_tool_call`
# --------------------------------------------------------------------------

def evaluate_admission_guard(root, worktree_real, agent, session,
                             profile=None, domain_prefixes=None):
    """Décision de gate pré-mutation, en LECTURE SEULE (aucun owner écrit, aucun
    PID éphémère persisté).

    Retourne `(allowed: bool, reason: str | None)`.

    - Worktree non revendiqué -> `(True, None)` : advisory fail-open (le gate
      n'agit que sur des lanes réellement admises).
    - Profil métier borné (`profile` + `domain_prefixes`) et lane possédant le
      worktree hors domaine -> `(False, ...)` : refus automatique, indépendant de
      la session et de la liveness.
    - Autre session détenant le worktree avec un process VIVANT -> `(False, ...)`
      : anti double-occupation (un seul gagnant par worktree).
    - Sinon -> `(True, None)`.
    """
    with _anchored_registry_root(root) as root_anchor:
        match = _find_claim_for_worktree(root_anchor, worktree_real)
    if match is None:
        return True, None
    key, owner = match

    if profile and domain_prefixes:
        allowed = {p.strip() for p in domain_prefixes.split(",") if p.strip()}
        if allowed and _lane_prefix(key) not in allowed:
            return False, (
                f"profile {profile} cannot mutate lane {key} "
                f"(out of domain {sorted(allowed)})"
            )

    if not _is_same_session(owner, agent, session):
        if determine_process_state(owner) == "alive":
            return False, (
                f"worktree owned by {key} "
                f"{owner.get('agent', '?')}/{owner.get('session_id', '?')}"
            )

    return True, None


def cmd_guard(root, repo, agent, session, profile=None, domain_prefixes=None,
              as_json=False):
    """Hard gate CLI (debug/canary). Le vrai flux runtime importe
    `evaluate_admission_guard` depuis `factory_admission_hook.py`."""
    worktree_real = _git_toplevel_or_none(repo) or os.path.realpath(repo)
    allowed, reason = evaluate_admission_guard(
        root, worktree_real, agent, session,
        profile=profile, domain_prefixes=domain_prefixes,
    )
    if as_json:
        print(json.dumps(
            {"allowed": allowed, "reason": reason, "worktree": worktree_real},
            sort_keys=True,
        ))
    if not allowed:
        if not as_json:
            print(f"BLOCKED: {reason}", file=sys.stderr)
        return 1
    return 0


def cmd_event(root, key, event_name, evidence, pr, commit=None, ci=None, deploy=None):
    validate_key(key)
    if event_name not in ALLOWED_MANUAL_EVENTS:
        raise RegistryError(f"unknown event: {event_name!r}")
    if evidence is not None:
        _validate_evidence(evidence)
    if commit is not None:
        _validate_metadata_field(commit, "commit")
    if ci is not None:
        _validate_metadata_field(ci, "ci")
    if deploy is not None:
        _validate_metadata_field(deploy, "deploy")
    if pr is not None:
        _validate_metadata_field(pr, "pr")
    if event_name == "pr_opened" and not pr:
        raise RegistryError("pr_opened requires --pr")

    with _registry_operation(root, "lanes", "locks") as root_anchor:
        with _dirfd_flock(root_anchor, ("locks", key), ".lock"):
            owner = _read_owner_via_chain(root_anchor, key)
            if owner is None:
                raise RegistryError(f"no active claim for lane {key}")
            owner["heartbeat_at"] = time.time()
            _write_owner_via_chain(root_anchor, key, owner)

            extra = {}
            if evidence is not None:
                extra["evidence"] = evidence
            if pr is not None:
                extra["pr"] = pr
            if commit is not None:
                extra["commit"] = commit
            if ci is not None:
                extra["ci"] = ci
            if deploy is not None:
                extra["deploy"] = deploy
            _append_event_via_chain(root_anchor, key, event_name, extra=extra)
    return 0


def cmd_handoff(root, key, status, next_step, evidence):
    validate_key(key)
    if status not in ALLOWED_HANDOFF_STATUS:
        raise RegistryError(f"unknown handoff status: {status!r}")
    if "\n" in next_step or "\r" in next_step:
        raise RegistryError("next-step must not contain newlines")
    if len(next_step) > MAX_NEXT_STEP_LEN:
        raise RegistryError("next-step too long")
    if _SECRET_PATTERN.search(next_step):
        raise RegistryError("next-step looks like a secret and was rejected")
    if evidence is not None:
        _validate_evidence(evidence)

    with _registry_operation(root, "lanes", "locks", "handoffs") as root_anchor:
        with _dirfd_flock(root_anchor, ("locks", key), ".lock"):
            if _read_owner_via_chain(root_anchor, key) is None:
                raise RegistryError(f"no active claim for lane {key}")

            now = time.time()
            ts_ms = int(now * 1000)
            date_suffix = time.strftime("%Y-%m-%d", time.gmtime(now))
            handoff_name = f"{key}-{date_suffix}.md"
            lines = [
                f"# Handoff {key}",
                f"Status: {status}",
                f"Next step: {next_step}",
            ]
            if evidence is not None:
                lines.append(f"Evidence: {evidence}")
            lines.append(f"Timestamp: {ts_ms}")
            content = "\n".join(lines) + "\n"

            handoffs_fd = _open_dir_chain(root_anchor, ("handoffs",), create=True)
            try:
                _write_text_at(
                    handoffs_fd, os.path.join(_root_text_path(root_anchor), "handoffs"), handoff_name, content,
                )
            finally:
                os.close(handoffs_fd)

            _append_event_via_chain(root_anchor, key, "handoff", extra={
                "status": status, "handoff_file": handoff_name,
            })
    return 0


def cmd_render(root):
    with _registry_operation(root, "lanes", "locks") as root_anchor:
        lanes_fd = _open_dir_chain(root_anchor, ("lanes",), create=False)
        try:
            try:
                lane_names = sorted(name for name in os.listdir(lanes_fd) if name.endswith(".jsonl"))
            except OSError as exc:
                raise RegistryError(f"registry lane scan failed: {exc}") from exc

            entries = []
            for lane_name in lane_names:
                key = Path(lane_name).stem
                events = _read_all_events_at(
                    lanes_fd,
                    lane_name,
                    os.path.join(_root_text_path(root_anchor), "lanes", lane_name),
                )
                if _read_owner_via_chain(root_anchor, key) is None:
                    continue
                entries.append((key, _compute_status(events)))
        finally:
            os.close(lanes_fd)

        lines = ["# LANES", ""]
        if not entries:
            lines.append("_No active lanes._")
        else:
            lines.append("| Key | Status |")
            lines.append("|-----|--------|")
            for key, status in entries:
                lines.append(f"| {key} | {status} |")
        content = "\n".join(lines) + "\n"

        root_fd = _open_dir_chain(root_anchor, (), create=False)
        try:
            _write_text_at(root_fd, _root_text_path(root_anchor), "LANES.md", content, mode=0o644)
        finally:
            os.close(root_fd)
    return 0


def cmd_close(root, key):
    validate_key(key)
    with _registry_operation(root, "lanes", "locks") as root_anchor:
        with _dirfd_flock(root_anchor, ("locks", key), ".lock"):
            if _read_owner_via_chain(root_anchor, key) is None:
                raise RegistryError(f"no active claim for lane {key}")
            _append_event_via_chain(root_anchor, key, "lane_closed")
            _unlink_owner_via_chain(root_anchor, key)
    return 0


# --------------------------------------------------------------------------
# Context pack (HER-55 Scope A) — Hermes -> Claude Code
# --------------------------------------------------------------------------

def _lane_status(root, key):
    """Statut de lane courant, sans effet de bord (aucune création de dossier)."""
    lane_file = root / "lanes" / f"{key}.jsonl"
    if not lane_file.exists():
        return "unclaimed"
    try:
        events = _read_all_events(lane_file)
    except RegistryError:
        return "unclaimed"
    return _compute_status(events)


def _load_context_map(path):
    """Charge `context-map.json`. Toute anomalie (absent, symlink, corrompu)
    dégrade silencieusement vers "aucun repo mappé", jamais une exception :
    un repo mal mappé doit se comporter comme un repo inconnu, pas planter."""
    path = Path(path)
    try:
        if path.is_symlink() or not path.exists():
            return {}
        fd = _open_regular_file_via_chain(path, label="context map")
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, RegistryError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    repos = data.get("repos")
    if not isinstance(repos, dict):
        return {}
    return repos


def _is_blocked_vault_path(relpath):
    """Défense en profondeur : blocage même si `allow_files` liste le chemin."""
    if not isinstance(relpath, str) or not relpath:
        return True
    posix = relpath.replace(os.sep, "/").lstrip("/")
    basename = posix.rsplit("/", 1)[-1]
    if basename == ".env" or basename.startswith(".env"):
        return True
    if _SECRET_PATTERN.search(posix):
        return True
    return any(
        posix == prefix.rstrip("/") or posix.startswith(prefix)
        for prefix in _VAULT_BLOCKED_DIR_PREFIXES
    )


def _read_vault_file_safely(vault_root, relpath):
    """Lit un fichier du vault sans jamais suivre un symlink ni sortir de
    `vault_root`. Retourne None (jamais une exception) sur toute anomalie."""
    if not isinstance(relpath, str) or not relpath or relpath.startswith("/"):
        return None
    target = vault_root / relpath
    try:
        _reject_symlink_ancestors(target)
        if target.is_symlink():
            return None
        resolved = target.resolve(strict=True)
        resolved.relative_to(vault_root)
    except (OSError, ValueError, RegistryError):
        return None
    try:
        fd = _open_regular_file_via_chain(resolved, label="vault context file")
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, RegistryError):
        return None


def _filter_markdown_sections(content, allow_sections):
    """Ne garde que les sections `## Titre` explicitement autorisées.

    Liste vide -> aucune restriction (fichier entier)."""
    if not allow_sections:
        return content
    allowed = set(allow_sections)
    kept = []
    keep = False
    for line in content.splitlines():
        if line.startswith("## "):
            keep = line in allowed
        if keep:
            kept.append(line)
    return "\n".join(kept)


def _linear_cache_banner(root, key):
    """Bannière missing/stale sans jamais planter ; None si cache frais."""
    cache_path = root / "linear-cache" / f"{key}.json"
    try:
        if cache_path.is_symlink() or not cache_path.exists():
            return f"> Linear cache MISSING for {key}."
        fd = _open_regular_file_via_chain(cache_path, label="linear cache")
        with os.fdopen(fd, "r", encoding="utf-8") as source:
            data = json.load(source)
    except (OSError, RegistryError, json.JSONDecodeError):
        return f"> Linear cache MISSING or unreadable for {key}."

    cached_at = data.get("cached_at")
    if not isinstance(cached_at, (int, float)) or isinstance(cached_at, bool):
        return f"> Linear cache STALE for {key} (no cached_at)."

    age_hours = (time.time() - cached_at) / 3600
    if age_hours > LINEAR_CACHE_STALE_HOURS:
        return f"> Linear cache STALE for {key} ({age_hours:.1f}h old)."

    title = data.get("title")
    if isinstance(title, str) and title:
        return f"Linear: {key} — {title}"
    return f"Linear cache: fresh for {key}."


def _render_unmapped_brief(key, repo_real, status):
    lines = [
        f"# Context Claude Code — {key}",
        UNMAPPED_REPO_MARKER,
        f"Repo: {repo_real}",
        f"Lane status: {status}",
    ]
    return "\n".join(lines) + "\n"


def _render_mapped_pack(key, repo_real, status, root, vault_root, entry):
    allow_files = entry.get("allow_files") or []
    allow_sections = entry.get("allow_sections") or []

    lines = [
        f"# Context Claude Code — {key}",
        f"Repo: {repo_real}",
        f"Lane status: {status}",
    ]
    banner = _linear_cache_banner(root, key)
    if banner:
        lines.append(banner)
    lines.append("")

    for relpath in allow_files:
        if _is_blocked_vault_path(relpath):
            continue
        raw = _read_vault_file_safely(vault_root, relpath)
        if raw is None:
            continue
        filtered = _filter_markdown_sections(raw, allow_sections)
        if not filtered.strip():
            continue
        lines.append(f"## Fichier: {relpath}")
        lines.extend(filtered.splitlines())
        lines.append("")

    content = "\n".join(lines).rstrip("\n") + "\n"
    body_lines = content.splitlines()
    if len(body_lines) > CONTEXT_PACK_MAX_LINES:
        body_lines = body_lines[: CONTEXT_PACK_MAX_LINES - 1] + ["_(pack tronqué à 300 lignes)_"]
        content = "\n".join(body_lines) + "\n"
    return content


def _write_context_output(out_path, content):
    out_path = Path(out_path)
    _reject_symlink_ancestors(out_path)
    if out_path.is_symlink():
        raise RegistryError(f"context output must not be a symlink: {out_path}")
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise RegistryError(f"context output directory must not be a symlink: {parent}")

    parent_fd = _open_secure(parent, os.O_RDONLY | _ODIRECTORY_FLAG)
    try:
        _assert_dirfd_matches_path(parent_fd, parent, "context output directory")
        fd, tmp_path = tempfile.mkstemp(dir=str(parent), prefix=".context-", suffix=".tmp")
        tmp_name = Path(tmp_path).name
        try:
            # The temporary file was created through a text path.  Before writing
            # or renaming it, prove that path still names the descriptor we opened.
            # This turns a post-mkstemp ancestor replacement into a clean refusal.
            _assert_dirfd_matches_path(parent_fd, parent, "context output directory")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
                os.fchmod(f.fileno(), 0o600)
            _assert_dirfd_matches_path(parent_fd, parent, "context output directory")
            _atomic_replace_at(parent_fd, str(parent), tmp_name, out_path.name)
            os.fsync(parent_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name, dir_fd=parent_fd)
            raise
    finally:
        os.close(parent_fd)


def _write_registry_context_output(root, registry_output_root, out_path, content):
    """Write a context pack under ``registry/contexts`` through dirfds only.

    Context packs are registry mutations.  Resolve their logical relative path
    without following links, then use the same ``openat``/``renameat`` chain as
    owners and handoffs so a post-validation ancestor swap cannot redirect the
    write outside the registry on macOS.
    """
    root_abs = Path(os.path.abspath(str(registry_output_root)))
    output_abs = Path(os.path.abspath(str(out_path)))
    try:
        relative_output = output_abs.relative_to(root_abs)
    except ValueError as exc:
        raise RegistryError("context output must stay inside registry contexts") from exc
    if not relative_output.parts or relative_output.name in {"", ".", ".."}:
        raise RegistryError("context output must name a file")
    if any(part in {"", ".", ".."} for part in relative_output.parts):
        raise RegistryError("context output path is invalid")

    parent_parts = ("contexts", *relative_output.parts[:-1])
    parent_fd = _open_dir_chain(root, parent_parts, create=True)
    try:
        _write_text_at(parent_fd, "", relative_output.name, content)
    finally:
        os.close(parent_fd)


def _path_is_within(path, root):
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def cmd_context(root, key, repo, vault, context_map_arg, out_arg):
    with _anchored_registry_root(root) as root_anchor:
        return _cmd_context_anchored(
            root_anchor, key, repo, vault, context_map_arg, out_arg,
        )


def _cmd_context_anchored(root, key, repo, vault, context_map_arg, out_arg):
    validate_key(key)
    repo_real = os.path.realpath(repo)
    root_text = Path(_root_text_path(root))
    registry_output_root = root_text / "contexts"
    repo_output_root = Path(repo_real) / ".factory"
    out_path = Path(out_arg) if out_arg else registry_output_root / f"{key}.md"
    output_parent_real = Path(os.path.realpath(str(out_path.parent)))
    allowed_output_roots = (
        Path(os.path.realpath(str(registry_output_root))),
        Path(os.path.realpath(str(repo_output_root))),
    )
    if not any(
        _path_is_within(output_parent_real, allowed_root)
        for allowed_root in allowed_output_roots
    ):
        raise RegistryError(
            "context output must stay inside registry contexts or repo .factory"
        )

    context_map_path = Path(context_map_arg) if context_map_arg else root_text / "context-map.json"
    repos_map = _load_context_map(context_map_path)
    entry = repos_map.get(repo_real)
    status = _lane_status(root_text, key)

    if entry is None:
        # Repo inconnu : aucun accès au vault, pas même `os.path.realpath`.
        content = _render_unmapped_brief(key, repo_real, status)
        if _path_is_within(Path(os.path.abspath(str(out_path))), registry_output_root):
            _write_registry_context_output(root, registry_output_root, out_path, content)
        else:
            _write_context_output(out_path, content)
        return 0

    vault_root = Path(os.path.realpath(vault))
    content = _render_mapped_pack(key, repo_real, status, root_text, vault_root, entry)
    if _path_is_within(Path(os.path.abspath(str(out_path))), registry_output_root):
        _write_registry_context_output(root, registry_output_root, out_path, content)
    else:
        _write_context_output(out_path, content)
    return 0


# --------------------------------------------------------------------------
# Capture handoff (HER-55 Scope B) — Claude Code -> Hermes
# --------------------------------------------------------------------------

MAX_GIT_STATUS_ENTRIES = 5000
MAX_SUMMARY_TEXT_LEN = 1000
MAX_SUMMARY_BYTES = 64 * 1024
MAX_SUMMARY_NODES = 256
MAX_SUMMARY_DEPTH = 8
MAX_SUMMARY_CONTAINER_ITEMS = 128
MAX_HANDOFF_BYTES = 1024 * 1024
SUMMARY_ALLOWED_KEYS = frozenset({
    "tests", "decisions", "blockers", "next_step", "pr", "ci",
})


def _run_git(repo_real, *args, timeout=10):
    try:
        return subprocess.run(
            ["git", *args], cwd=str(repo_real), capture_output=True, text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RegistryError(f"git {' '.join(args)} failed: {exc}") from exc


def _git_toplevel(repo_path):
    result = _run_git(repo_path, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise RegistryError(f"--repo is not a git worktree: {repo_path}")
    return os.path.realpath(result.stdout.strip())


def _git_current_branch(repo_real):
    result = _run_git(repo_real, "rev-parse", "--abbrev-ref", "HEAD")
    if result.returncode != 0:
        raise RegistryError("failed to determine current branch")
    return result.stdout.strip()


def _git_head_commit(repo_real):
    result = _run_git(repo_real, "rev-parse", "HEAD")
    if result.returncode != 0:
        raise RegistryError("failed to determine HEAD commit")
    return result.stdout.strip()


def _git_ref_exists(repo_real, ref):
    result = _run_git(repo_real, "rev-parse", "--verify", "--quiet", ref)
    return result.returncode == 0


def _git_base_commit(repo_real, branch):
    """merge-base avec `main` si elle existe et diffère de `branch`, sinon
    commit racine (premier commit de l'historique)."""
    if branch and branch != "main" and _git_ref_exists(repo_real, "main"):
        result = _run_git(repo_real, "merge-base", branch, "main")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    result = _run_git(repo_real, "rev-list", "--max-parents=0", "HEAD")
    if result.returncode == 0:
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if lines:
            return lines[0]
    raise RegistryError("unable to determine base commit")


def _git_status_entries(repo_real):
    """Chemins + codes de statut seulement — jamais de contenu ni de diff.

    Le format NUL de porcelain v1 préserve les espaces, guillemets et la
    sous-chaîne littérale ` -> `. Pour rename/copy, Git place la destination
    dans le premier record et la source dans le record NUL suivant."""
    result = _run_git(repo_real, "status", "--porcelain=v1", "-z")
    if result.returncode != 0:
        raise RegistryError("failed to read git status")
    records = result.stdout.split("\0")
    entries = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(entries) >= MAX_GIT_STATUS_ENTRIES:
            raise RegistryError("too many git status entries to capture safely")
        if len(record) < 3:
            raise RegistryError("malformed git status record")
        code = record[:2]
        path = record[3:] if record[2:3] == " " else record[2:]
        if "R" in code or "C" in code:
            if index >= len(records) or not records[index]:
                raise RegistryError("malformed git rename/copy status record")
            index += 1  # source path; destination is the first path in -z mode
        entries.append({"path": path, "status": code})
    return entries


def _validate_summary_text(value, label):
    if len(value) > MAX_SUMMARY_TEXT_LEN:
        raise RegistryError(f"summary field {label} too long")
    if "\n" in value or "\r" in value:
        raise RegistryError(f"summary field {label} must not contain newlines")
    if _SECRET_PATTERN.search(value):
        raise RegistryError(f"summary field {label} looks like a secret and was rejected")


def _validate_summary_value(label, value, depth=0, budget=None):
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > MAX_SUMMARY_NODES:
        raise RegistryError("summary has too many nodes")
    if depth > MAX_SUMMARY_DEPTH:
        raise RegistryError("summary is nested too deeply")
    if isinstance(value, str):
        _validate_summary_text(value, label)
        return value
    if value is None or isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        if len(value) > MAX_SUMMARY_CONTAINER_ITEMS:
            raise RegistryError(f"summary field {label} has too many items")
        return [
            _validate_summary_value(f"{label}[]", item, depth + 1, budget)
            for item in value
        ]
    if isinstance(value, dict):
        if len(value) > MAX_SUMMARY_CONTAINER_ITEMS:
            raise RegistryError(f"summary field {label} has too many entries")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RegistryError(f"summary field {label} has a non-string key")
            _validate_summary_text(key, f"{label}.key")
            result[key] = _validate_summary_value(
                f"{label}.{key}", item, depth + 1, budget
            )
        return result
    raise RegistryError(f"summary field {label} has an unsupported type")


def _load_summary(summary_path):
    path = Path(summary_path)
    fd = _open_regular_file_via_chain(path, label="summary path")
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        if os.fstat(f.fileno()).st_size > MAX_SUMMARY_BYTES:
            raise RegistryError("summary JSON is too large")
        raw = f.read(MAX_SUMMARY_BYTES + 1)
    if len(raw.encode("utf-8")) > MAX_SUMMARY_BYTES:
        raise RegistryError("summary JSON is too large")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"summary is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError("summary must be a JSON object")

    unknown = set(data.keys()) - SUMMARY_ALLOWED_KEYS
    if unknown:
        raise RegistryError(f"summary has unsupported keys: {sorted(unknown)}")
    budget = [0]
    return {
        k: _validate_summary_value(k, v, depth=0, budget=budget)
        for k, v in data.items()
    }


def _serialize_handoff(payload):
    content = json.dumps(payload, sort_keys=True) + "\n"
    if len(content.encode("utf-8")) > MAX_HANDOFF_BYTES:
        raise RegistryError("handoff JSON is too large")
    return content


def cmd_capture_handoff(root, key, repo, summary_path):
    validate_key(key)
    summary_fields = {}
    if summary_path is not None:
        summary_fields = _load_summary(summary_path)

    with _registry_operation(root, "lanes", "locks", "handoffs") as root_path, \
            _dirfd_flock(root_path, ("locks", key), ".lock"):
        owner = _read_owner_via_chain(root_path, key)
        if owner is None:
            raise RegistryError(f"no active claim for lane {key}")

        repo_real = _git_toplevel(repo)
        claimed_worktree = owner.get("worktree")
        if claimed_worktree and os.path.realpath(claimed_worktree) != repo_real:
            raise RegistryError(
                f"--repo {repo_real} does not match claimed worktree {claimed_worktree}"
            )

        branch = _git_current_branch(repo_real)
        head_sha = _git_head_commit(repo_real)
        base_commit = _git_base_commit(repo_real, branch)
        git_status = _git_status_entries(repo_real)
        files_changed = sorted({entry["path"] for entry in git_status if entry["path"]})

        now = time.time()
        date_suffix = time.strftime("%Y-%m-%d", time.gmtime(now))
        handoff_name = f"{key}-{date_suffix}.handoff.json"

        payload = {
            "issue": key,
            "agent": owner.get("agent"),
            "session_id": owner.get("session_id"),
            "repo": repo_real,
            "worktree": repo_real,
            "branch": branch,
            "base_commit": base_commit,
            "head_commit": head_sha,
            "git_status": git_status,
            "files_changed": files_changed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        payload.update(summary_fields)

        content = _serialize_handoff(payload)
        handoffs_fd = _open_dir_chain(root_path, ("handoffs",), create=True)
        try:
            _write_text_at(
                handoffs_fd, os.path.join(_root_text_path(root_path), "handoffs"), handoff_name, content,
            )
        finally:
            os.close(handoffs_fd)

        _append_event_via_chain(root_path, key, "handoff_captured", extra={
            "handoff_file": handoff_name,
        })
    return 0


# --------------------------------------------------------------------------
# Réconciliation Claude -> Hermes + hooks Claude Code (HER-55 Scope B suite)
# --------------------------------------------------------------------------

RECONCILE_STALE_HOURS = 24.0


def _git_toplevel_or_none(repo_path, timeout=5):
    """Comme `_git_toplevel` mais ne lève jamais : `None` sur tout échec
    (binaire git absent, timeout, pas un dépôt Git). Usage exclusif hooks
    fail-open — `reconcile` reste fail-closed via `_git_toplevel`."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    if not out:
        return None
    try:
        return os.path.realpath(out)
    except OSError:
        return None


def _find_claim_for_worktree(root, repo_real):
    """Cherche un `owner.json` actif dont le worktree realpath correspond à
    `repo_real`. Retourne `(key, owner)` ou `None`.

    Le gate runtime ne peut pas transformer un scan incomplet en ``ownerless``:
    un locks/owner remplacé par symlink, un fichier non régulier ou un échec de
    lecture est indiscernable d'un owner caché. Le scan descend donc depuis le
    fd du registry avec ``openat(..., O_NOFOLLOW)`` et propage ``RegistryError``
    au hook, qui refuse alors la mutation avant l'outil.
    """
    root_path = Path(root.path) if isinstance(root, _RegistryRootAnchor) else Path(root)
    try:
        locks_fd = _open_dir_chain(root, ("locks",), create=False)
    except FileNotFoundError:
        return None
    except (OSError, RegistryError) as exc:
        raise RegistryError(f"registry lock scan failed: {exc}") from exc
    try:
        def assert_locks_path_stable():
            """Reject a textual locks/ path swapped after its no-follow open."""
            if isinstance(root, _RegistryRootAnchor):
                _assert_dirfd_matches_path(root.fd, root.path, "registry root")
            try:
                path_stat = os.stat(root_path / "locks", follow_symlinks=False)
            except OSError as exc:
                raise RegistryError(f"registry locks changed during owner scan: {exc}") from exc
            fd_stat = os.fstat(locks_fd)
            if (
                not stat.S_ISDIR(path_stat.st_mode)
                or (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise RegistryError("registry locks changed during owner scan")

        try:
            names = sorted(os.listdir(locks_fd))
        except OSError as exc:
            raise RegistryError(f"registry lock scan failed: {exc}") from exc
        assert_locks_path_stable()
        for key in names:
            try:
                validate_key(key)
                key_fd = _openat_subdir(locks_fd, key, create=False)
            except (OSError, RegistryError) as exc:
                raise RegistryError(f"registry lock scan failed for {key!r}: {exc}") from exc
            try:
                try:
                    owner_fd = _open_regular_file_at(
                        key_fd,
                        "owner.json",
                        label=f"owner record for {key!r}",
                    )
                except FileNotFoundError:
                    continue
                except (OSError, RegistryError) as exc:
                    raise RegistryError(f"registry lock scan failed for {key!r}: {exc}") from exc
                try:
                    _require_regular_fd(owner_fd, f"owner record for {key!r}")
                    with os.fdopen(owner_fd, "r", encoding="utf-8") as f:
                        owner = json.load(f)
                except (OSError, ValueError, json.JSONDecodeError, RegistryError) as exc:
                    raise RegistryError(f"registry lock scan failed for {key!r}: {exc}") from exc
                if not isinstance(owner, dict):
                    raise RegistryError(f"registry lock scan failed for {key!r}: owner record is not an object")
                worktree = owner.get("worktree")
                if isinstance(worktree, str) and os.path.realpath(worktree) == repo_real:
                    assert_locks_path_stable()
                    return key, owner
            finally:
                os.close(key_fd)
        assert_locks_path_stable()
        return None
    finally:
        os.close(locks_fd)


def _load_handoff(handoff_path, handoffs_dir):
    path = Path(handoff_path)
    _reject_symlink_ancestors(path)
    if path.is_symlink():
        raise RegistryError(f"handoff path must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(Path(handoffs_dir).resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RegistryError("handoff path must stay inside registry/handoffs") from exc
    fd = _open_regular_file_via_chain(resolved, label="handoff path")
    with os.fdopen(fd, "r", encoding="utf-8") as f:
        if os.fstat(f.fileno()).st_size > MAX_HANDOFF_BYTES:
            raise RegistryError("handoff JSON is too large")
        raw = f.read(MAX_HANDOFF_BYTES + 1)
    if len(raw.encode("utf-8")) > MAX_HANDOFF_BYTES:
        raise RegistryError("handoff JSON is too large")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"handoff is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError("handoff must be a JSON object")
    return data


def _parse_handoff_timestamp(value):
    """epoch seconds (int/float) ou ISO8601 UTC `%Y-%m-%dT%H:%M:%SZ`.
    Toute autre forme -> `None` (traité comme périmé, jamais comme frais)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            return None
    return None


def _test_entry_has_artifact(entry, repo_real):
    """Une déclaration de test ne compte comme preuve que si elle est verte et
    pointe vers un artefact régulier sous `<repo>/.factory/test-artifacts/`.
    Les fichiers système, chemins externes et symlinks ne valent jamais preuve."""
    if not isinstance(entry, dict):
        return False
    if entry.get("result") != "pass":
        return False
    artifact_path = entry.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        return False
    try:
        p = Path(artifact_path)
        _reject_symlink_ancestors(p)
        if p.is_symlink() or not p.is_file():
            return False
        resolved = p.resolve(strict=True)
        allowed_root = (Path(repo_real) / ".factory" / "test-artifacts").resolve(strict=False)
        resolved.relative_to(allowed_root)
        return True
    except (OSError, ValueError, RegistryError):
        return False


def _evaluate_handoff(repo_real, owner, handoff):
    """Fonction pure (hors accès Git en lecture seule) : verdict fermé +
    raisons. Priorité conflicting > stale > partial > verified."""
    reasons = []

    if handoff.get("agent") != owner.get("agent"):
        reasons.append("agent_mismatch")
    if handoff.get("session_id") != owner.get("session_id"):
        reasons.append("session_mismatch")

    claimed_worktree = owner.get("worktree")
    handoff_worktree = handoff.get("worktree")
    worktree_ok = (
        isinstance(handoff_worktree, str)
        and bool(claimed_worktree)
        and os.path.realpath(handoff_worktree) == os.path.realpath(claimed_worktree)
    )
    if not worktree_ok:
        reasons.append("worktree_mismatch")

    try:
        actual_branch = _git_current_branch(repo_real)
    except RegistryError:
        actual_branch = None
    if actual_branch is None or handoff.get("branch") != actual_branch:
        reasons.append("branch_mismatch")

    try:
        actual_head = _git_head_commit(repo_real)
    except RegistryError:
        actual_head = None
    if actual_head is None or handoff.get("head_commit") != actual_head:
        reasons.append("head_commit_mismatch")

    conflicting = bool(reasons)

    ts = _parse_handoff_timestamp(handoff.get("timestamp"))
    stale = ts is None or (time.time() - ts) > (RECONCILE_STALE_HOURS * 3600)
    if stale:
        reasons.append("stale_timestamp")

    tests = handoff.get("tests")
    unproven_tests = isinstance(tests, list) and len(tests) > 0 and any(
        not _test_entry_has_artifact(entry, repo_real) for entry in tests
    )
    if unproven_tests:
        reasons.append("unproven_test_claim")

    if conflicting:
        verdict = "conflicting"
    elif stale:
        verdict = "stale"
    elif unproven_tests:
        verdict = "partial"
    else:
        verdict = "verified"

    return verdict, reasons


def cmd_reconcile(root, key, repo, handoff_path):
    with _registry_operation(root, "lanes", "locks", "handoffs") as root_anchor:
        return _cmd_reconcile_anchored(root_anchor, key, repo, handoff_path)


def _cmd_reconcile_anchored(root, key, repo, handoff_path):
    validate_key(key)
    handoffs_dir = Path(_root_text_path(root)) / "handoffs"

    handoff = _load_handoff(handoff_path, handoffs_dir)

    if handoff.get("issue") != key:
        raise RegistryError(
            f"handoff issue {handoff.get('issue')!r} does not match lane {key!r}"
        )

    repo_real = _git_toplevel(repo)

    handoff_repo = handoff.get("repo")
    if not isinstance(handoff_repo, str) or os.path.realpath(handoff_repo) != repo_real:
        raise RegistryError("handoff repo does not match --repo")

    handoff_canonical = handoff.get("canonical_repo")
    if handoff_canonical is not None:
        if not isinstance(handoff_canonical, str) or os.path.realpath(handoff_canonical) != repo_real:
            raise RegistryError("handoff canonical_repo does not match --repo")

    root_path = root
    with _dirfd_flock(root_path, ("locks", key), ".lock"):
        owner = _read_owner_via_chain(root_path, key)
        if owner is None:
            raise RegistryError(f"no active claim for lane {key}")

        verdict, reasons = _evaluate_handoff(repo_real, owner, handoff)

        _append_event_via_chain(root_path, key, "reconciled", extra={
            "verdict": verdict,
            "reasons": reasons,
            "handoff_head_commit": handoff.get("head_commit"),
        })

    payload = {
        "key": key,
        "issue": key,
        "repo": repo_real,
        "verdict": verdict,
        "reasons": reasons,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def cmd_hook_session_start(root, repo, agent=None, session=None):
    """`SessionStart` : silencieux hors repo suivi/sans claim, brief court
    sinon. Fail-open strict : toute exception -> code 0 sans sortie."""
    try:
        repo_real = _git_toplevel_or_none(repo)
        if repo_real is None:
            return 0

        match = _find_claim_for_worktree(root, repo_real)
        if match is None:
            return 0
        key, owner = match

        status = _lane_status(root, key)
        if agent and session and not _is_same_session(owner, agent, session):
            print(
                "STOP: worktree already owned by "
                f"{key} {owner.get('agent', '?')}/{owner.get('session_id', '?')} "
                f"for {repo_real}"
            )
            return 0

        lines = [
            f"# Hermes — {key}",
            f"Repo: {repo_real}",
            f"Lane status: {status}",
            f"Agent: {owner.get('agent', '?')}",
        ]
        print("\n".join(lines[:UNMAPPED_BRIEF_MAX_LINES]))
        return 0
    except Exception:
        return 0


def cmd_hook_stop(root, repo, key, summary_path=None):
    """`Stop` : capture les faits Git objectifs (jamais transcript/contenu),
    silencieux hors repo suivi/sans claim. Fail-open strict."""
    try:
        validate_key(key)

        repo_real = _git_toplevel_or_none(repo)
        if repo_real is None:
            return 0

        with _anchored_registry_root(root) as root_anchor:
            owner = _read_owner_via_chain(root_anchor, key)
            if owner is None:
                return 0
            claimed_worktree = owner.get("worktree")
            if not claimed_worktree or os.path.realpath(claimed_worktree) != repo_real:
                return 0

            branch = _git_current_branch(repo_real)
            head_sha = _git_head_commit(repo_real)
            git_status = _git_status_entries(repo_real)
            files_changed = sorted({e["path"] for e in git_status if e["path"]})

            payload = {
                "issue": key,
                "agent": owner.get("agent"),
                "session_id": owner.get("session_id"),
                "repo": repo_real,
                "worktree": repo_real,
                "branch": branch,
                "head_commit": head_sha,
                "git_status": git_status,
                "files_changed": files_changed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
            }

            if summary_path is not None:
                try:
                    payload.update(_load_summary(summary_path))
                except RegistryError:
                    pass

            content = _serialize_handoff(payload)
            capture_name = f"{key}-stop-{time.time_ns()}.json"
            handoffs_fd = _open_dir_chain(root_anchor, ("handoffs",), create=True)
            try:
                _write_text_at(
                    handoffs_fd,
                    os.path.join(_root_text_path(root_anchor), "handoffs"),
                    capture_name,
                    content,
                )
            finally:
                os.close(handoffs_fd)

        return 0
    except Exception:
        return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(prog="factory_lane")
    parser.add_argument("--registry", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    p_preflight = sub.add_parser("preflight")
    p_preflight.add_argument("key")
    p_preflight.add_argument("--json", action="store_true")

    p_claim = sub.add_parser("claim")
    p_claim.add_argument("key")
    p_claim.add_argument("--agent", required=True)
    p_claim.add_argument("--session", required=True)
    p_claim.add_argument("--worktree", required=True)
    p_claim.add_argument("--reclaim", action="store_true")
    p_claim.add_argument("--reclaim-worktree", action="store_true")
    p_claim.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    p_claim.add_argument("--profile")
    p_claim.add_argument("--gateway-session-key")
    p_claim.add_argument("--owner-pid", type=int)
    p_claim.add_argument("--owner-start-time")

    p_admit = sub.add_parser("admit")
    p_admit.add_argument("key")
    p_admit.add_argument("--mode", choices=("owner", "reviewer"), required=True)
    p_admit.add_argument("--hard", action="store_true")
    p_admit.add_argument("--agent", required=True)
    p_admit.add_argument("--session", required=True)
    p_admit.add_argument("--worktree", required=True)
    p_admit.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    p_admit.add_argument("--json", action="store_true")
    p_admit.add_argument("--profile")
    p_admit.add_argument("--gateway-session-key")
    p_admit.add_argument("--domain-prefixes")
    p_admit.add_argument("--owner-pid", type=int)
    p_admit.add_argument("--owner-start-time")

    p_team_admit = sub.add_parser("team-admit")
    p_team_admit.add_argument("key")
    p_team_admit.add_argument("--team-config", required=True)
    p_team_admit.add_argument("--profile", required=True)
    p_team_admit.add_argument("--agent", required=True)
    p_team_admit.add_argument("--session", required=True)
    p_team_admit.add_argument("--worktree", required=True)
    p_team_admit.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    p_team_admit.add_argument("--gateway-session-key")
    p_team_admit.add_argument("--owner-pid", type=int)
    p_team_admit.add_argument("--owner-start-time")
    p_team_admit.add_argument("--reclaim", action="store_true")
    p_team_admit.add_argument("--freshness-evidence")

    p_team_status = sub.add_parser("team-status")
    p_team_status.add_argument("--team-config", required=True)
    p_team_status.add_argument("--json", action="store_true")

    p_guard = sub.add_parser("guard")
    p_guard.add_argument("--repo", required=True)
    p_guard.add_argument("--agent", required=True)
    p_guard.add_argument("--session", required=True)
    p_guard.add_argument("--profile")
    p_guard.add_argument("--domain-prefixes")
    p_guard.add_argument("--json", action="store_true")

    p_event = sub.add_parser("event")
    p_event.add_argument("key")
    p_event.add_argument("event_name")
    p_event.add_argument("--evidence")
    p_event.add_argument("--pr")
    p_event.add_argument("--commit")
    p_event.add_argument("--ci")
    p_event.add_argument("--deploy")

    sub.add_parser("render")

    p_handoff = sub.add_parser("handoff")
    p_handoff.add_argument("key")
    p_handoff.add_argument("--status", required=True)
    p_handoff.add_argument("--next-step", required=True)
    p_handoff.add_argument("--evidence")

    p_close = sub.add_parser("close")
    p_close.add_argument("key")

    p_context = sub.add_parser("context")
    p_context.add_argument("key")
    p_context.add_argument("--repo", required=True)
    p_context.add_argument("--vault", required=True)
    p_context.add_argument("--context-map")
    p_context.add_argument("--out")

    p_capture_handoff = sub.add_parser("capture-handoff")
    p_capture_handoff.add_argument("key")
    p_capture_handoff.add_argument("--repo", required=True)
    p_capture_handoff.add_argument("--summary")

    p_reconcile = sub.add_parser("reconcile")
    p_reconcile.add_argument("key")
    p_reconcile.add_argument("--repo", required=True)
    p_reconcile.add_argument("--handoff", required=True)

    p_hook_session_start = sub.add_parser("hook-session-start")
    p_hook_session_start.add_argument("--repo", required=True)
    p_hook_session_start.add_argument("--agent")
    p_hook_session_start.add_argument("--session")

    p_hook_stop = sub.add_parser("hook-stop")
    p_hook_stop.add_argument("--repo", required=True)
    p_hook_stop.add_argument("--key", required=True)
    p_hook_stop.add_argument("--summary")

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        # Team routing is an input-policy gate.  Validate it before creating a
        # registry root so rejected cross-team admits leave no observable or
        # mutable registry state behind.
        if args.command == "team-admit":
            validate_key(args.key)
            team, spec = _resolve_team_for_profile(
                load_team_config(args.team_config), args.profile,
            )
            if _lane_prefix(args.key) not in spec["allowed_teams"]:
                raise RegistryError(f"profile {args.profile} cannot own lane {args.key}")
        root = _safe_registry_root(args.registry)
        if args.command == "preflight":
            return cmd_preflight(root, args.key, args.json)
        if args.command == "claim":
            return cmd_claim(
                root, args.key, args.agent, args.session, args.worktree,
                args.reclaim or args.reclaim_worktree, args.ttl_hours,
                profile=args.profile, gateway_session_key=args.gateway_session_key,
                owner_pid=args.owner_pid, owner_start_time=args.owner_start_time,
            )
        if args.command == "admit":
            return cmd_admit(
                root, args.key, args.mode, args.hard, args.agent, args.session,
                args.worktree, args.ttl_hours, as_json=args.json,
                profile=args.profile,
                gateway_session_key=args.gateway_session_key,
                domain_prefixes=args.domain_prefixes,
                owner_pid=args.owner_pid, owner_start_time=args.owner_start_time,
            )
        if args.command == "team-admit":
            return cmd_team_admit(
                root, args.key, args.team_config, args.profile, args.agent, args.session,
                args.worktree, ttl_hours=args.ttl_hours,
                gateway_session_key=args.gateway_session_key, owner_pid=args.owner_pid,
                owner_start_time=args.owner_start_time, reclaim=args.reclaim,
                freshness_evidence_path=args.freshness_evidence,
            )
        if args.command == "team-status":
            return cmd_team_status(root, args.team_config, as_json=args.json)
        if args.command == "guard":
            return cmd_guard(
                root, args.repo, args.agent, args.session,
                profile=args.profile, domain_prefixes=args.domain_prefixes,
                as_json=args.json,
            )
        if args.command == "event":
            return cmd_event(
                root, args.key, args.event_name, args.evidence, args.pr,
                commit=args.commit, ci=args.ci, deploy=args.deploy,
            )
        if args.command == "render":
            return cmd_render(root)
        if args.command == "handoff":
            return cmd_handoff(root, args.key, args.status, args.next_step, args.evidence)
        if args.command == "close":
            return cmd_close(root, args.key)
        if args.command == "context":
            return cmd_context(
                root, args.key, args.repo, args.vault,
                args.context_map, args.out,
            )
        if args.command == "capture-handoff":
            return cmd_capture_handoff(root, args.key, args.repo, args.summary)
        if args.command == "reconcile":
            return cmd_reconcile(root, args.key, args.repo, args.handoff)
        if args.command == "hook-session-start":
            return cmd_hook_session_start(root, args.repo, args.agent, args.session)
        if args.command == "hook-stop":
            return cmd_hook_stop(root, args.repo, args.key, args.summary)
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 1
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # fail-closed sur toute erreur inattendue
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
