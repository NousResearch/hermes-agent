"""Rollout helpers for profile-to-UID setup: dedicated board, copy, SHA, toolchain."""

from __future__ import annotations

import os
import shlex
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

DEFAULT_DEV_WORKSPACE = "/home/matt/Documents/WorkoutTracker"
DEFAULT_FLUTTER_SDK = "/home/matt/flutter"
DEFAULT_ANDROID_SDK = "/home/matt/Android/Sdk"
DEFAULT_JDK_HOME = "/home/matt/.local/opt/jdk-17"
DEFAULT_GH_BIN = "/usr/bin/gh"
DEFAULT_GIT_BIN = "/usr/bin/git"
VERSIONED_RUNTIME_ROOT = "/opt/hermes/kanban-os-users"
GITHUB_LS_REMOTE_URL = "https://github.com/NousResearch/hermes-agent.git"
_SECRET_BASENAMES = frozenset({
    ".env",
    "auth.json",
    "id_rsa",
    "id_ecdsa",
    "id_ed25519",
    "id_dsa",
    "credentials.json",
})


def _quote_cmd(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in argv)


def self_hermes_argv(hermes_argv: Optional[Sequence[str]] = None) -> list[str]:
    """Argv for *this* tree's CLI. Never assume an installed `hermes` binary."""
    if hermes_argv:
        return [str(p) for p in hermes_argv]
    return [sys.executable, "-m", "hermes_cli.main"]


def apply_command_hint(hermes_argv: Optional[Sequence[str]] = None) -> str:
    argv = self_hermes_argv(hermes_argv) + [
        "kanban",
        "os-users",
        "setup",
        "--apply",
    ]
    return "sudo " + _quote_cmd(argv)


def feature_source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def feature_source_sha() -> Optional[str]:
    root = feature_source_root()
    git = root / ".git"
    try:
        if git.is_file():
            text = git.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return None
            gitdir = Path(text.split(":", 1)[1].strip())
            if not gitdir.is_absolute():
                gitdir = (root / gitdir).resolve()
        elif git.is_dir():
            gitdir = git
        else:
            return None
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            return (gitdir / ref).read_text(encoding="utf-8").strip()[:40]
        return head[:40]
    except OSError:
        return None


def hermes_argv_covers_feature(
    hermes_argv: Sequence[str], feature_root: Optional[Path] = None
) -> bool:
    root = str((feature_root or feature_source_root()).resolve())
    for part in hermes_argv:
        raw = str(part)
        try:
            resolved = str(Path(raw).resolve())
        except OSError:
            resolved = raw
        if root in resolved or resolved.startswith(root):
            return True
        if root in raw:
            return True
    return False


def dedicated_shared_db_path(hermes_root: Path) -> Path:
    return Path(hermes_root) / "kanban" / "kanban.db"


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return str(a) == str(b)


def shared_board_acl_layout(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Retarget WAL/write ACLs off the Hermes root onto a dedicated kanban dir."""
    live_db = Path(paths["kanban_db"])
    hermes_root = Path(paths["hermes_root"]) if "hermes_root" in paths else None
    if "kanban_dir" in paths:
        kdir = Path(paths["kanban_dir"])
    elif hermes_root is not None:
        kdir = hermes_root / "kanban"
    else:
        kdir = live_db.parent / "kanban"
    needs_migration = False
    if hermes_root is not None and _same_path(live_db.parent, hermes_root):
        writable_dir = kdir
        target_db = kdir / "kanban.db"
        try:
            needs_migration = live_db.exists() and not _same_path(live_db, target_db)
        except OSError:
            needs_migration = live_db.exists()
    else:
        writable_dir = live_db.parent
        target_db = live_db
    if hermes_root is not None and _same_path(writable_dir, hermes_root):
        raise ValueError(
            f"refusing write ACL on Hermes root {hermes_root}; "
            "shared board must live in a dedicated directory"
        )
    return {
        "hermes_root": hermes_root,
        "writable_dir": writable_dir,
        "target_db": target_db,
        "live_db": live_db,
        "needs_migration": needs_migration,
        "kanban_dir": kdir,
    }


def setfacl_is_write(argv: Sequence[str]) -> bool:
    blob = " ".join(str(p) for p in argv)
    if (
        ":--x" in blob
        and ":wx" not in blob
        and ":rw" not in blob
        and ":rwx" not in blob
    ):
        return False
    return any(tok in blob for tok in (":wx", ":rwx", ":rw"))


def reject_write_acl_on_hermes_root(steps: Sequence[Any], hermes_root: Path) -> None:
    root = str(hermes_root)
    for step in steps:
        argv = list(getattr(step, "argv", []) or [])
        if not argv or argv[0] != "setfacl":
            continue
        if str(argv[-1]) != root:
            continue
        if setfacl_is_write(argv):
            raise ValueError(f"refusing write ACL on Hermes root: {_quote_cmd(argv)}")


def sqlite_backup_copy(src: str | Path, dst: str | Path) -> None:
    """Online-safe SQLite backup. Never raw-copy a live DB or its WAL/SHM."""
    import sqlite3

    src_p = Path(src)
    dst_p = Path(dst)
    if src_p.is_symlink() or dst_p.is_symlink():
        raise RuntimeError("refusing to backup via symlink")
    if not src_p.is_file():
        raise RuntimeError(f"source DB {src_p} does not exist")
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    src_con = sqlite3.connect(f"file:{src_p}?mode=ro", uri=True, timeout=30)
    dst_con = sqlite3.connect(str(dst_p), timeout=30)
    try:
        src_con.backup(dst_con)
        dst_con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dst_con.commit()
    finally:
        dst_con.close()
        src_con.close()


def summarize_tree(src: Path) -> dict[str, Any]:
    files = 0
    dirs = 0
    symlinks = 0
    if not src.exists():
        return {"files": 0, "dirs": 0, "symlinks": 0, "missing": True}
    if src.is_symlink():
        return {"files": 0, "dirs": 0, "symlinks": 1, "missing": False}
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        kept: list[str] = []
        for name in dirnames:
            child = Path(dirpath) / name
            if child.is_symlink():
                symlinks += 1
            else:
                kept.append(name)
                dirs += 1
        dirnames[:] = kept
        for name in filenames:
            child = Path(dirpath) / name
            if child.is_symlink():
                symlinks += 1
            else:
                files += 1
    return {"files": files, "dirs": dirs, "symlinks": symlinks, "missing": False}


def _maybe_chown(path: Path, owner: str, group: str) -> None:
    if os.geteuid() != 0:
        return
    try:
        import grp
        import pwd

        os.chown(path, pwd.getpwnam(owner).pw_uid, grp.getgrnam(group).gr_gid)
    except (KeyError, LookupError, OSError, PermissionError):
        return


def copy_tree_reject_symlinks(
    src: str | Path,
    dst: str | Path,
    *,
    owner: str,
    group: str,
) -> dict[str, int]:
    """Copy a directory tree. Rejects symlinks and never prints file contents."""
    src_p = Path(src)
    dst_p = Path(dst)
    if src_p.is_symlink() or dst_p.is_symlink():
        raise RuntimeError(f"refusing symlink copy root {src_p} -> {dst_p}")
    if ".." in src_p.parts or ".." in dst_p.parts:
        raise RuntimeError("refusing path with ..")
    if not src_p.is_dir():
        raise RuntimeError(f"copy-tree source is not a directory: {src_p}")
    stats = {"files": 0, "dirs": 1, "rejected_symlinks": 0}
    dst_p.mkdir(parents=True, exist_ok=True)
    os.chmod(dst_p, 0o700)
    _maybe_chown(dst_p, owner, group)
    for dirpath, dirnames, filenames in os.walk(src_p, followlinks=False):
        rel = Path(dirpath).relative_to(src_p)
        kept: list[str] = []
        for name in dirnames:
            child = Path(dirpath) / name
            if child.is_symlink():
                stats["rejected_symlinks"] += 1
                continue
            kept.append(name)
            dest_dir = dst_p / rel / name
            dest_dir.mkdir(parents=True, exist_ok=True)
            mode = stat.S_IMODE(child.stat().st_mode) & 0o755
            os.chmod(dest_dir, mode or 0o700)
            _maybe_chown(dest_dir, owner, group)
            stats["dirs"] += 1
        dirnames[:] = kept
        for name in filenames:
            child = Path(dirpath) / name
            if child.is_symlink():
                stats["rejected_symlinks"] += 1
                continue
            dest_file = dst_p / rel / name
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(child, "rb") as inf, open(dest_file, "wb") as outf:
                while True:
                    chunk = inf.read(1024 * 1024)
                    if not chunk:
                        break
                    outf.write(chunk)
            mode = stat.S_IMODE(child.stat().st_mode)
            if child.name in _SECRET_BASENAMES or child.name == ".env":
                os.chmod(dest_file, 0o600)
            else:
                os.chmod(dest_file, mode & 0o644)
            _maybe_chown(dest_file, owner, group)
            stats["files"] += 1
    return stats


def copy_tree_argv(
    src: Path,
    dst: Path,
    *,
    owner: str,
    group: str,
    hermes_argv: Optional[Sequence[str]] = None,
) -> list[str]:
    return self_hermes_argv(hermes_argv) + [
        "kanban",
        "os-users",
        "copy-tree",
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--owner",
        owner,
        "--group",
        group,
    ]


def migrate_db_argv(
    src: Path,
    dst: Path,
    *,
    hermes_argv: Optional[Sequence[str]] = None,
) -> list[str]:
    return self_hermes_argv(hermes_argv) + [
        "kanban",
        "os-users",
        "migrate-db",
        "--from",
        str(src),
        "--to",
        str(dst),
    ]


def default_toolchain_if_present() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, path in (
        ("flutter_sdk", DEFAULT_FLUTTER_SDK),
        ("android_sdk", DEFAULT_ANDROID_SDK),
        ("jdk_home", DEFAULT_JDK_HOME),
    ):
        if Path(path).exists():
            out[key] = path
    return out


def extra_sudoers_bins(
    *,
    flutter_sdk: Optional[str] = None,
    android_sdk: Optional[str] = None,
    jdk_home: Optional[str] = None,
) -> list[str]:
    bins: list[str] = []
    for path in (DEFAULT_GH_BIN, DEFAULT_GIT_BIN):
        if Path(path).is_file():
            bins.append(path)
    if flutter_sdk:
        flutter = Path(flutter_sdk) / "bin" / "flutter"
        dart = Path(flutter_sdk) / "bin" / "dart"
        if flutter.is_file():
            bins.append(str(flutter))
        if dart.is_file():
            bins.append(str(dart))
    if jdk_home:
        java = Path(jdk_home) / "bin" / "java"
        if java.is_file():
            bins.append(str(java))
    _ = android_sdk
    return bins


def format_rollout_and_rollback(*, hermes_argv: Optional[Sequence[str]] = None) -> str:
    hint = apply_command_hint(hermes_argv)
    sha = feature_source_sha() or "(unknown)"
    runtime = f"{VERSIONED_RUNTIME_ROOT}/{sha}"
    return "\n".join([
        "## Ordered rollout (do not skip; mid-step failure must leave the live board recoverable)",
        "1. Deploy this reviewed commit first. Preferred: wait for upstream merge + upgrade.",
        f"   Alternative: install SHA {sha} into {runtime} and point HERMES_BIN + gateway unit + sudoers at that argv.",
        "   `sudo hermes` may invoke the old installed CLI without os-users — do not use it.",
        f"   Apply with: {hint}",
        "2. Create private groups/users and specialist homes. Do not enable profile_os_users yet.",
        "3. Create the dedicated shared dir (~/.hermes/kanban/) with group wx. Hermes root gets traverse-only (--x), never write.",
        "4. Quiesce the gateway (stop/restart window you control). Live ~/.hermes/kanban.db stays untouched until backup succeeds.",
        "5. sqlite backup API (migrate-db), not cp of a live DB/WAL/SHM, into ~/.hermes/kanban/kanban.db.",
        "6. Pin HERMES_KANBAN_DB to the dedicated path in the gateway service. Restart onto the versioned runtime.",
        "7. Toolchain: narrow u:hermes-dev:r-x on Flutter/SDK/JDK; private caches under /home/hermes-dev/.cache. No recursive ACL on /home/matt.",
        "8. Manual gates: --migrate-profile-files (bounded copy-tree), then gh auth login as hermes-dev (never copy SSH keys).",
        "9. hermes kanban os-users check must pass SHA/runtime, WAL-in-dedicated-dir, gh/git, toolchain, and secret denials.",
        "10. Only then enable kanban.profile_os_users in the *gateway* config and restart. Isolation is false until the live dispatcher runs this SHA.",
        "",
        "## Rollback (recover the current gateway/board)",
        "1. Remove profile_os_users from gateway config and restart onto the previous HERMES_BIN. Mapping-off is the recoverability gate.",
        "2. Point HERMES_KANBAN_DB back at the pre-cutover file if the dedicated copy is untrusted. Keep the backup; do not delete the live DB first.",
        "3. Remove the sudoers drop-in and visudo -c.",
        "4. Reverse recorded ACLs. userdel/groupdel only principals created by this setup (state file).",
        "5. Group membership and toolchain ACLs can wait; the board is already served by the previous runtime.",
        "",
        "Do not cat/print .env, auth.json, gh tokens, or SSH keys.",
    ])


def github_manual_gate_lines(dev_user: str = "hermes-dev") -> list[str]:
    return [
        "# GitHub continuity (secret-safe manual gate — never copy ~/.ssh or print tokens)",
        f"# As root, in a tty you start: sudo -u {dev_user} -H gh auth login --hostname github.com --git-protocol https",
        f"# Prove: sudo -n -u {dev_user} -- {DEFAULT_GH_BIN} api user",
        f"# Prove: sudo -n -u {dev_user} -- {DEFAULT_GIT_BIN} ls-remote {GITHUB_LS_REMOTE_URL} HEAD",
        "# Do not copy Matt's SSH keys into the specialist home.",
        f"# Optional workspace: {DEFAULT_DEV_WORKSPACE}",
        f"# Isolation does not include world-readable code trees; prove denial of .ssh and credentials instead.",
    ]
