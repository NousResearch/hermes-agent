"""Filesystem hardening and privilege-drop helpers.

This is the "lock everything down" layer. Two levels of isolation exist:

  1. Filesystem permissions (this module): each employee's HERMES_HOME is 0700,
     secret files 0600. On a shared OS user this stops *accidental* cross-tenant
     reads but is NOT a boundary against a malicious/injected agent (same UID).

  2. OS/kernel boundary (backend layer): a dedicated OS user per worker
     (`run_as_user` + drop_privileges_cmd) or a container/microVM. This is the
     real security boundary. See backends/ and README "Security model".
"""
from __future__ import annotations

import os
from pathlib import Path


def ensure_dir(path: Path, mode: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, mode)


def write_secret(path: Path, content: str, mode: int) -> None:
    """Write a secret file with restrictive perms set BEFORE content lands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create with restrictive mode atomically via O_CREAT/O_EXCL-ish flow.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(path, mode)  # enforce even if umask/pre-existing


def harden_tree(root: Path, dir_mode: int, secret_names: set[str], secret_mode: int) -> None:
    """Recursively enforce perms across an employee home."""
    os.chmod(root, dir_mode)
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            os.chmod(Path(dirpath) / d, dir_mode)
        for f in filenames:
            p = Path(dirpath) / f
            os.chmod(p, secret_mode if f in secret_names else 0o600)


def chown_recursive(root: Path, user: str) -> None:
    """chown -R root to `user` (requires privilege). Best-effort; raises on failure."""
    import pwd
    pw = pwd.getpwnam(user)
    os.chown(root, pw.pw_uid, pw.pw_gid)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            os.chown(Path(dirpath) / name, pw.pw_uid, pw.pw_gid)


def drop_privileges_cmd(base_cmd: list[str], run_as_user: str | None) -> list[str]:
    """Wrap a command so it runs as `run_as_user` (per-worker OS boundary).

    Returns the command unchanged when run_as_user is None (dev mode).
    Uses `sudo -u` which requires a sudoers rule for the control-plane user.
    """
    if not run_as_user:
        return base_cmd
    return ["sudo", "-n", "-u", run_as_user, "--", *base_cmd]


def seatbelt_wrap(
    cmd: list[str],
    own_home: Path,
    runtime_reads: list[Path],
    home_root: Path = Path("/Users"),
) -> list[str]:
    """Wrap a command in a macOS sandbox that confines it to its OWN tenant home.

    Model: allow the SYSTEM to work (so binaries run), but **deny all of the
    human home root** (``/Users`` — private keys, docs, other repos, other users,
    AND the global ~/.hermes secrets), then re-allow only:
      * bare metadata/stat under the home root (so runtimes that walk parent
        dirs — e.g. Hermes probing for .git — don't crash; no content, no writes)
      * read-only CONTENT for the specific runtime paths needed to execute
      * read+write for this tenant's own home ONLY

    This closes the earlier hole where the agent (allow-default) could read the
    operator's entire laptop. It confines the worker daemon and every subprocess
    it spawns. Still a dev stand-in for the container/microVM boundary; and note
    it does NOT block unix-socket connect() — the per-worker token does that."""
    own = own_home.resolve()
    root = home_root.resolve()
    parts = [
        "(version 1)",
        "(allow default)",
        f'(deny file* (subpath "{root}"))',
        f'(allow file-read-metadata (subpath "{root}"))',
    ]
    seen = set()
    for p in runtime_reads:
        rp = Path(p).resolve()
        if rp.exists() and str(rp) not in seen:
            seen.add(str(rp))
            parts.append(f'(allow file-read* (subpath "{rp}"))')
    parts.append(f'(allow file* (subpath "{own}"))')
    return ["sandbox-exec", "-p", " ".join(parts), *cmd]


def runtime_read_paths(hermes_bin: str) -> list[Path]:
    """Best-effort list of read-only paths a worker needs to execute Hermes +
    the orchard daemon, so the confinement sandbox can re-allow just those."""
    import sys
    import orchard

    home = Path.home()
    paths: list[Path] = [
        Path(orchard.__file__).resolve().parent,   # orchard package (editable)
        Path(sys.prefix),                            # control-plane venv
        home / ".local",                             # node / uv-managed tools
        home / ".hermes" / "node",                   # shared node runtime
        home / ".hermes" / "bin",                    # shared tool binaries
    ]
    hb = Path(hermes_bin)
    if hb.is_absolute() and len(hb.parents) >= 3:
        paths.append(hb.parents[2])                  # the hermes checkout/venv root
    return paths


def assert_contained(path: Path, root: Path) -> Path:
    """Guard against path traversal: `path` must resolve inside `root`."""
    rp = path.resolve()
    rr = root.resolve()
    if rr not in rp.parents and rp != rr:
        raise ValueError(f"path {rp} escapes tenant root {rr}")
    return rp
