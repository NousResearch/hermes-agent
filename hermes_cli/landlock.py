"""Fail-closed Linux filesystem confinement for a child command.

This module does not launch Pi or a proxy. ``run_landlocked`` starts a short-lived
helper with a clean environment; the helper applies Landlock before replacing
itself with the requested command. Network access is not restricted by this
filesystem-only policy.
"""

from __future__ import annotations

import base64
import ctypes
import errno
import ipaddress
import json
import os
import platform
import re
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from hermes_cli._startup_fast import project_root_str


class LandlockError(RuntimeError):
    """The requested sandbox policy could not be applied safely."""


class LandlockUnavailable(LandlockError):
    """The kernel or platform cannot enforce the required Landlock policy."""


_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

# Linux Landlock access bits. ABI 5 is required so every filesystem right,
# including TRUNCATE and IOCTL_DEV, is handled rather than implicitly allowed.
_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_IOCTL_DEV = 1 << 15
_FS_ALL = (1 << 16) - 1
_FS_READ = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
# Writable roots allow editing existing files and creating regular files and
# directories. Deletion, rename/hard-link, symlink, socket, FIFO and device
# creation are intentionally not granted.
_FS_WORKSPACE = _FS_READ | _FS_WRITE_FILE | _FS_TRUNCATE | _FS_MAKE_REG | _FS_MAKE_DIR

_SYSCALLS = {
    "x86_64": (444, 445, 446),
    "amd64": (444, 445, 446),
    "aarch64": (444, 445, 446),
    "arm64": (444, 445, 446),
}

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


def _syscall_numbers() -> tuple[int, int, int]:
    numbers = _SYSCALLS.get(platform.machine().lower())
    if numbers is None:
        raise LandlockUnavailable("Landlock syscall numbers are not defined for this architecture")
    return numbers


def _syscall(number: int, *args) -> int:
    """Invoke the variadic libc syscall wrapper with explicitly typed arguments."""
    ctypes.set_errno(0)
    result = _LIBC.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno() or errno.EIO
        raise OSError(error, os.strerror(error))
    return int(result)


def probe_landlock_abi() -> int:
    """Return the kernel Landlock ABI, or fail closed when the syscall is unavailable."""
    if not sys.platform.startswith("linux"):
        raise LandlockUnavailable("Landlock is available only on Linux")
    create_ruleset, _, _ = _syscall_numbers()
    try:
        abi = _syscall(
            create_ruleset,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
            raise LandlockUnavailable("the running kernel does not expose Landlock") from exc
        raise LandlockError("could not query the Landlock ABI") from exc
    return abi


def _resolved_path(path: str | os.PathLike[str], *, strict: bool = True) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=strict)
    except (OSError, RuntimeError, ValueError) as exc:
        raise LandlockError("a sandbox path is missing or cannot be resolved") from exc


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _sensitive_workspace_name(name: str) -> bool:
    lower = name.lower()
    if lower in {
        ".env", ".env.local", ".env.production", ".env.development",
        "auth.json", "credentials.json", "credentials", "credentials.yaml",
        "credentials.yml", "token.json", "secrets.json", "secrets.yaml",
        "secrets.yml", "secret.json", "secret.yaml", "secret.yml",
        "kubeconfig", ".git-credentials", ".npmrc", ".netrc", ".pypirc",
        ".pgpass", ".gitconfig", ".terraformrc",
        "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    }:
        return True
    if lower.startswith(".env") and lower not in {".env.example", ".env.sample", ".env.template"}:
        return True
    return lower.endswith((".pem", ".p12", ".pfx", ".key", ".backup"))


# Directories a credential-like file or credential store may never live in the
# sandbox workspace itself; their presence means the workspace is not a clean
# temporary coding tree.
_SENSITIVE_WORKSPACE_DIRS = frozenset(
    {
        ".aws", ".ssh", ".gnupg", ".kube", ".docker", ".config", ".netrc",
        ".wrangler", ".cache", ".azure", ".gcloud", ".terraform.d",
    }
)


def _mount_points_below(workspace: Path) -> list[Path]:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise LandlockUnavailable("cannot verify nested mounts without /proc/self/mountinfo")
    found: list[Path] = []
    unescape = re.compile(r"\\([0-7]{3})")
    try:
        for line in mountinfo.read_text(encoding="utf-8-sig").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            mount_path = Path(unescape.sub(lambda m: chr(int(m.group(1), 8)), fields[4]))
            if mount_path != workspace and workspace in mount_path.parents:
                found.append(mount_path)
    except OSError as exc:
        raise LandlockUnavailable("could not inspect mount points") from exc
    return found


def _preflight_workspace(workspace: Path) -> None:
    """Reject nested mounts, credential stores and special files before granting workspace writes."""
    nested_mounts = _mount_points_below(workspace)
    if nested_mounts:
        raise LandlockError("workspace contains a nested mount; refusing the sandbox policy")

    stack = [workspace]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise LandlockError("workspace cannot be inspected safely") from exc
        for entry in entries:
            if _sensitive_workspace_name(entry.name):
                raise LandlockError("workspace contains a credential-like file; refusing access")
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise LandlockError("workspace changed during sandbox preflight") from exc
            if stat.S_ISLNK(mode):
                continue
            if stat.S_ISDIR(mode):
                if entry.name in _SENSITIVE_WORKSPACE_DIRS:
                    raise LandlockError(
                        "workspace contains a credential-like directory; refusing access"
                    )
                stack.append(Path(entry.path))
                continue
            if not stat.S_ISREG(mode):
                raise LandlockError("workspace contains a special file; refusing the sandbox policy")
            if entry.stat(follow_symlinks=False).st_nlink > 1:
                raise LandlockError("workspace contains a hard-linked file; refusing the sandbox policy")


def _effective_uid() -> int:
    """Return the caller's effective UID, failing closed on non-Linux hosts.

    Landlock itself is Linux-only; keeping the lookup behind a helper means
    the module imports cleanly everywhere and the runner still refuses to run
    outside Linux via ``probe_landlock_abi``.
    """
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        raise LandlockUnavailable("Landlock is available only on Linux")
    return int(geteuid())


def _require_private_owned_directory(path: Path, label: str) -> None:
    """Reject a workspace or sandbox HOME another user can write into.

    A group- or world-writable directory can change between preflight and
    sandbox activation, and an owner other than the current user indicates
    the caller handed over data it does not control. This narrows, but cannot
    eliminate, the residual mutation race by the same user.
    """
    try:
        st = path.stat()
    except OSError as exc:
        raise LandlockError(f"{label} cannot be inspected safely") from exc
    if st.st_uid != _effective_uid():
        raise LandlockError(f"{label} must be owned by the current user")
    if st.st_mode & 0o022:
        raise LandlockError(f"{label} must not be writable by group or others")


def _canonical_roots(paths: Iterable[str | os.PathLike[str]], *, label: str) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = _resolved_path(raw)
        if path not in seen:
            seen.add(path)
            result.append(path)
    if not result:
        return ()
    return tuple(result)


def _validate_protected_paths(
    *,
    read_paths: Sequence[Path],
    write_paths: Sequence[Path],
    protected_paths: Iterable[str | os.PathLike[str]],
) -> None:
    protected = tuple(_resolved_path(path, strict=False) for path in protected_paths)
    for grant in (*read_paths, *write_paths):
        if grant == Path("/"):
            raise LandlockError("a filesystem-root grant is not allowed")
        if any(_overlaps(grant, blocked) for blocked in protected):
            raise LandlockError("a sandbox grant overlaps a protected path")


def _default_protected_paths() -> tuple[Path, ...]:
    """Sensitive roots the runner protects without caller input.

    These are resolved loosely (``strict=False``) so policies are deterministic
    even on hosts where a given directory does not exist; the overlap check is
    path-based and does not require the path to be present.
    """
    home = Path.home()
    parts: list[Path] = []
    for name in (
        ".ssh", ".aws", ".gnupg", ".config", ".kube", ".docker",
        ".cache", ".wrangler", ".netrc", ".pgpass", ".npmrc", ".pypirc",
        ".gitconfig", ".git-credentials", ".azure", ".gcloud", ".terraform.d",
        ".terraformrc",
    ):
        parts.append(home / name)

    hermes_home = os.environ.get("HERMES_HOME")
    hermes_dir = Path(hermes_home).expanduser() if hermes_home else home / ".hermes"
    for name in (
        "auth.json", ".env", ".env.backup", "config.yaml", "config-backups",
        "profiles", "profile.yaml", "profile-backups", "mcp-tokens",
        "kanban.db", "kanban", "state.db", "shared-state.db",
        "gateway_state.json", "state", "state-snapshots", "watcher-state",
        "plugin-data", "memories", ".ssh",
    ):
        parts.append(hermes_dir / name)

    for name in ("/root", "/etc/ssh", "/etc/ssl/private"):
        parts.append(Path(name))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in parts:
        try:
            candidate = path.expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return tuple(resolved)


def _system_read_paths() -> tuple[Path, ...]:
    candidates = (
        "/usr/lib", "/usr/lib64", "/usr/local/lib", "/lib", "/lib64", "/etc/ld.so.cache",
        "/etc/ssl/certs", "/etc/hosts", "/etc/resolv.conf", "/etc/nsswitch.conf",
        "/etc/passwd", "/etc/group", "/dev/null", "/dev/urandom",
    )
    return tuple(Path(path) for path in candidates if Path(path).exists())


def _runtime_read_paths(command: Sequence[str], path_entries: Sequence[Path]) -> tuple[Path, ...]:
    candidates: list[Path] = [
        Path(sys.executable),
        Path(sys.executable).resolve(),
        Path(sys.prefix),
    ]
    runtime_keys = ("stdlib", "platstdlib", "purelib", "platlib")
    candidates.extend(
        Path(value)
        for key, value in sysconfig.get_paths().items()
        if key in runtime_keys and value
    )
    candidates.extend(_system_read_paths())
    for root in (Path(command[0]), Path(command[0]).resolve()):
        candidates.append(root)
        candidates.append(root.parent)
    candidates.extend(path_entries)
    return _canonical_roots((path for path in candidates if Path(path).exists()), label="runtime")


def _validate_proxy_url(proxy_url: str | None) -> None:
    if proxy_url is None:
        return
    try:
        parsed = urlsplit(proxy_url)
        host = parsed.hostname or ""
        is_loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
        _ = parsed.port
    except (ValueError, TypeError):
        is_loopback = False
    if (
        not is_loopback
        or parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/", "/v1"}
    ):
        raise LandlockError("the model proxy URL must be a plain loopback HTTP endpoint")


def _clean_child_environment(home: Path, path_entries: Sequence[Path], proxy_url: str | None) -> dict[str, str]:
    env = {
        "HOME": str(home),
        "TMPDIR": str(home),
        "PATH": os.pathsep.join(str(path) for path in path_entries),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if proxy_url is not None:
        env["OPENAI_BASE_URL"] = proxy_url.rstrip("/")
        env["OPENAI_API_KEY"] = "hermes-local-proxy"
    return env


def _add_path_rule(ruleset_fd: int, path: Path, rights: int) -> None:
    path_fd = os.open(path, getattr(os, "O_PATH", 0) | os.O_CLOEXEC)
    try:
        # READ_DIR is meaningful only for directories. Passing it for regular
        # runtime files (e.g. the interpreter or ld.so.cache) makes the kernel
        # reject the rule with EINVAL.
        if not path.is_dir():
            rights &= ~_FS_READ_DIR
        attr = _PathBeneathAttr(allowed_access=rights, parent_fd=path_fd, reserved=0)
        _, add_rule, _ = _syscall_numbers()
        try:
            _syscall(
                add_rule,
                ctypes.c_int32(ruleset_fd),
                ctypes.c_uint32(_LANDLOCK_RULE_PATH_BENEATH),
                ctypes.byref(attr),
                ctypes.c_uint32(0),
            )
        except OSError as exc:
            raise LandlockError(
                f"the kernel refused a filesystem rule for {path}: {exc.strerror}"
            ) from exc
    finally:
        os.close(path_fd)


def apply_landlock(
    *,
    read_paths: Iterable[str | os.PathLike[str]],
    write_paths: Iterable[str | os.PathLike[str]],
    protected_paths: Iterable[str | os.PathLike[str]] = (),
) -> int:
    """Restrict the current process to explicit read roots and limited write roots.

    Call this only in a disposable helper process: Landlock restrictions are
    inherited by children and cannot be removed. The caller must close any file
    descriptors it does not intend the child to inherit before calling.
    """
    abi = probe_landlock_abi()
    if abi < 5:
        raise LandlockUnavailable("Landlock ABI 5 or newer is required for complete filesystem rights")

    read_roots = _canonical_roots(read_paths, label="read")
    write_roots = _canonical_roots(write_paths, label="write")
    if not write_roots:
        raise LandlockError("at least one writable workspace root is required")
    for index, left in enumerate(write_roots):
        if any(_overlaps(left, right) for right in write_roots[index + 1 :]):
            raise LandlockError("writable roots must be disjoint")
    _validate_protected_paths(
        read_paths=read_roots,
        write_paths=write_roots,
        protected_paths=protected_paths,
    )

    create_ruleset, _, restrict_self = _syscall_numbers()
    attr = _RulesetAttr(handled_access_fs=_FS_ALL)
    ruleset_fd = _syscall(
        create_ruleset,
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint32(0),
    )
    try:
        for path in read_roots:
            _add_path_rule(ruleset_fd, path, _FS_READ)
        for path in write_roots:
            _add_path_rule(ruleset_fd, path, _FS_WORKSPACE)
        if _LIBC.prctl(
            ctypes.c_int(_PR_SET_NO_NEW_PRIVS),
            ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0),
        ) != 0:
            error = ctypes.get_errno() or errno.EPERM
            raise LandlockError(f"the kernel refused no_new_privs: {os.strerror(error)}")
        try:
            _syscall(restrict_self, ctypes.c_int32(ruleset_fd), ctypes.c_uint32(0))
        except OSError as exc:
            raise LandlockError(f"the kernel refused the Landlock policy: {exc.strerror}") from exc
    finally:
        os.close(ruleset_fd)
    return abi


def run_landlocked(
    command: Sequence[str],
    *,
    workspace: str | os.PathLike[str],
    home: str | os.PathLike[str],
    read_paths: Iterable[str | os.PathLike[str]] = (),
    protected_paths: Iterable[str | os.PathLike[str]] = (),
    path_entries: Iterable[str | os.PathLike[str]] = (),
    proxy_url: str | None = None,
    cwd: str | os.PathLike[str] | None = None,
    capture_output: bool = False,
    text: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a child command under Landlock with a clean, non-secret environment.

    Only ``workspace`` and the empty ``home`` directory receive limited write
    rights. Runtime paths are read-only. The runner protects, without caller
    input, the well-known credential and configuration directories under the
    real HOME (``.ssh``, ``.aws``, ``.gnupg``, ``.config``, ``.kube``,
    ``.docker``, ``.cache``, ``.wrangler``, ``.netrc``, ``.pgpass``,
    ``.npmrc``, ``.pypirc``, ``.gitconfig``, ``.git-credentials``, ``.azure``,
    ``.gcloud``, ``.terraform.d``, ...), the Hermes data dir (``HERMES_HOME``
    or ``~/.hermes``) and its well-known credential/config/state files
    (``auth.json``, ``.env``, ``config.yaml``, ``profiles``, ``mcp-tokens``,
    ``kanban.db``, ``state.db``, ``plugin-data``, ``memories``, ...), the
    system roots ``/root``, ``/etc/ssh`` and ``/etc/ssl/private``, and
    ``workspace/.binder-data`` (including a symlink target). No grant --
    caller-supplied or runtime -- may overlap any of them. These lists are
    allowlist-based guardrails on top of Landlock's deny-by-default boundary:
    paths not enumerated are still unreachable unless explicitly granted. A
    workspace that itself contains ``.binder-data`` is deliberately refused
    (a sandboxed child can create that entry, so shared workspaces are
    disposable). The workspace must not contain the helper implementation,
    and the workspace and sandbox HOME must be owned by the current user and
    not writable by group or others; the workspace may not contain
    credential-like entries, nested mounts or special files, and a sandbox
    HOME must be empty. ``protected_paths`` adds further caller-known data
    roots. Landlock cannot restrict network destinations, and a same-user
    race between preflight and activation cannot be fully eliminated. This
    helper is not yet wired into any Pi launcher.
    """
    if not command or any(not isinstance(arg, str) or "\0" in arg for arg in command):
        raise LandlockError("command must be a non-empty sequence of strings")
    executable = Path(command[0])
    if not executable.is_absolute() or not executable.exists():
        raise LandlockError("the sandbox command must name an existing absolute executable")

    workspace_path = _resolved_path(workspace)
    home_path = _resolved_path(home)
    if not workspace_path.is_dir() or not home_path.is_dir():
        raise LandlockError("workspace and sandbox HOME must be existing directories")
    module_root = Path(project_root_str())
    if _overlaps(workspace_path, module_root):
        raise LandlockError(
            "the sandbox workspace must not contain the helper implementation"
        )
    _require_private_owned_directory(workspace_path, "workspace")
    _require_private_owned_directory(home_path, "sandbox HOME")
    if _overlaps(workspace_path, home_path):
        raise LandlockError("workspace and sandbox HOME must be separate")
    if any(home_path.iterdir()):
        raise LandlockError("sandbox HOME must be empty")
    working_directory = _resolved_path(cwd or workspace_path)
    if working_directory != workspace_path and workspace_path not in working_directory.parents:
        raise LandlockError("working directory must be inside the writable workspace")

    _validate_proxy_url(proxy_url)
    _preflight_workspace(workspace_path)

    entries = [Path(path) for path in path_entries]
    entries.extend((executable.parent, executable.resolve().parent))
    normalized_entries = _canonical_roots(
        (path for path in entries if path.exists()), label="PATH"
    )
    if not normalized_entries:
        raise LandlockError("no executable PATH entries are available")

    binder_data = workspace_path / ".binder-data"
    protected = [*_default_protected_paths(), *protected_paths]
    if binder_data.exists() or binder_data.is_symlink():
        protected.append(binder_data.resolve(strict=False))

    runtime_roots = _runtime_read_paths(command, normalized_entries)
    explicit_roots = _canonical_roots(read_paths, label="read")
    all_read_roots = _canonical_roots((*runtime_roots, *explicit_roots), label="read")
    write_roots = (workspace_path, home_path)
    _validate_protected_paths(
        read_paths=all_read_roots,
        write_paths=write_roots,
        protected_paths=protected,
    )

    child_env = _clean_child_environment(home_path, normalized_entries, proxy_url)
    helper_env = {
        "HOME": str(home_path),
        "TMPDIR": str(home_path),
        "PATH": os.pathsep.join(str(path) for path in normalized_entries),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(module_root),
    }
    payload = {
        "read_paths": [str(path) for path in all_read_roots],
        "write_paths": [str(path) for path in write_roots],
        "protected_paths": [str(path) for path in protected],
        "child_env": child_env,
    }
    encoded_policy = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).decode()
    helper_command = [
        sys.executable,
        "-m",
        "hermes_cli.landlock",
        "--internal-landlock-exec",
        encoded_policy,
        "--",
        *command,
    ]
    return subprocess.run(
        helper_command,
        cwd=working_directory,
        env=helper_env,
        close_fds=True,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        check=False,
    )


def _helper_main(argv: Sequence[str]) -> int:
    if len(argv) < 4 or argv[0] != "--internal-landlock-exec" or argv[2] != "--":
        return 125
    try:
        payload = json.loads(base64.urlsafe_b64decode(argv[1].encode()).decode())
        command = list(argv[3:])
        if not command or not isinstance(payload, dict):
            return 125
        apply_landlock(
            read_paths=payload["read_paths"],
            write_paths=payload["write_paths"],
            protected_paths=payload["protected_paths"],
        )
        os.closerange(3, 1_048_576)
        os.execve(command[0], command, payload["child_env"])
    except Exception as exc:  # health: allow BLE001 -- the helper's only job is to fail closed: any policy, decode or exec error must exit 125 before the child can run, and the message is the caller's only diagnostic
        print(
            f"Landlock refused to execute the child ({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )
        return 125
    return 125


if __name__ == "__main__":  # pragma: no cover - exercised through run_landlocked
    raise SystemExit(_helper_main(sys.argv[1:]))
