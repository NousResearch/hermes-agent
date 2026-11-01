"""Refuse an update whose target predates the gateway runtime while a state.db already depends on it.

Releases before the gateway cutover know nothing of the runtime ledgers (``session_admissions`` /
``worker_executions``, ``ON DELETE RESTRICT``) and stamp the same ``schema_version``, so moving the
code back under a store that holds ledger rows leaves every session with that history undeletable
there. Downgrading is unsupported: the update refuses before anything moves and names the recovery
point that predates the cutover. Read-only throughout; a store or target it cannot read is never a
reason to refuse, and a target that descends from the running commit is a forward update.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from typing import Optional

# The runtime ledger DDL: a target whose state-store modules lack it predates the cutover.
_RUNTIME_SCHEMA_MARKER = "CREATE TABLE IF NOT EXISTS session_admissions"
STATE_MODULES = "hermes_state*.py"  # git pathspec and fnmatch pattern alike (``*`` crosses ``/`` in both)
_LEDGER_PROBE = ("SELECT EXISTS(SELECT 1 FROM session_admissions) "
                 "OR EXISTS(SELECT 1 FROM worker_executions)")
_DOCS = "https://hermes-agent.nousresearch.com/docs/getting-started/updating#rolling-back-to-an-older-release"


def _updater_git(git_cmd, root: Path) -> Callable[..., int]:
    import subprocess

    from hermes_cli.update_custody import run_git

    def git(*args: str) -> int:
        try:
            return run_git(git_cmd, list(args), cwd=str(root), capture_output=True, timeout=120).returncode
        except (OSError, subprocess.SubprocessError):
            return -1
    return git


def defines_runtime(sources: Iterable[str]) -> bool:
    """Whether any of a tree's ``STATE_MODULES`` sources defines the runtime ledger. Lazy: stops at
    the first match, so a caller reading sources over the network fetches no more than it needs."""
    return any(_RUNTIME_SCHEMA_MARKER in source for source in sources)


def target_predates_runtime(git_cmd, root: Path, target_ref: str, *,
                            git: Optional[Callable[..., int]] = None,
                            remote: Optional[Callable[[str], Optional[bool]]] = None) -> bool:
    """True when ``target_ref`` does not descend from HEAD and none of its ``hermes_state*.py`` files
    defines the runtime ledger. The pathspec keeps a blobless partial clone from fetching the whole
    tree. Any git failure proves nothing (False). ``git`` runs one git command and returns its exit
    code; the update check passes its own read-only probe.

    ``remote`` answers the same question for a target whose commit, tree or schema blobs are missing
    locally (True, False, or None when it cannot tell). The update check passes one because it never
    fetches; the updater stays git-only."""
    git = git or _updater_git(git_cmd, root)
    if git("merge-base", "--is-ancestor", "HEAD", target_ref) == 0:
        return False  # forward update: whatever the target's schema, it was written knowing this one
    # grep can return 1 for unreadable blobs too: establish completeness before trusting no match.
    if git("rev-list", "--objects", "--no-walk", "--sparse", "--missing=error", target_ref, "--", STATE_MODULES) != 0:
        return remote is not None and remote(target_ref) is True
    # Exit 1 = no match in a complete local schema; 0 = found; anything else is a git error.
    result = git("grep", "-q", "-F", _RUNTIME_SCHEMA_MARKER, target_ref, "--", STATE_MODULES)
    return result == 1 if result in (0, 1) else remote is not None and remote(target_ref) is True


def _has_ledger_rows(db_path: Path) -> bool:
    if not db_path.is_file():
        return False
    from hermes_state_holders import read_only_db_uri
    try:
        with closing(sqlite3.connect(read_only_db_uri(db_path), uri=True, timeout=5.0)) as conn:
            return bool(conn.execute(_LEDGER_PROBE).fetchone()[0])
    except sqlite3.Error:
        return False  # no ledger tables yet, or unreadable: nothing an older release would trip on


def _pre_runtime_snapshot(home: Path) -> str | None:
    """Newest quick snapshot of ``home`` holding a state.db from before the runtime ledger."""
    from hermes_cli.backup import _quick_snapshot_root
    from hermes_state_holders import read_only_db_uri
    root = _quick_snapshot_root(home)
    for snap in sorted(root.iterdir(), reverse=True) if root.is_dir() else ():
        db = snap / "state.db"
        if snap.name.startswith(".") or not db.is_file():
            continue
        try:
            with closing(sqlite3.connect(read_only_db_uri(db), uri=True)) as conn:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_admissions'").fetchone() is None:
                    return snap.name
        except sqlite3.Error:
            continue
    return None


def _named_profile(home: Path) -> str:
    """The ``-p`` name that opens ``home`` (``hermes -p acme update`` passes ``profiles/acme``);
    ``""`` for the default root or a custom HERMES_HOME, which bare ``hermes`` opens."""
    from hermes_cli.profiles import _PROFILE_ID_RE, _get_profiles_root
    parent = _get_profiles_root().resolve()
    resolved = home.resolve()
    named = resolved.parent == parent and _PROFILE_ID_RE.match(resolved.name) and resolved.name != "default"
    return resolved.name if named else ""


def _ledgered_profiles(home: Path | None) -> list[tuple[str, Path]]:
    from hermes_cli.backup import _sibling_profile_homes
    from hermes_constants import get_hermes_home

    home = Path(home or get_hermes_home())
    profiles = [(_named_profile(home), home), *_sibling_profile_homes(home)]
    return [(name, h) for name, h in profiles if _has_ledger_rows(h / "state.db")]


def would_refuse(root: Path, target_ref: str, home: Path | None = None, *, git: Callable[..., int],
                 remote: Optional[Callable[[str], Optional[bool]]] = None) -> bool:
    """Whether ``hermes update`` would refuse ``target_ref`` here: the update check's read-only twin of
    :func:`downgrade_refusal`, so a release the updater refuses is never offered as an update.
    The store scan runs first, so ``remote`` is only consulted when some store has runtime history."""
    return bool(_ledgered_profiles(home)) and target_predates_runtime(
        None, root, target_ref, git=git, remote=remote)


def downgrade_refusal(git_cmd, root: Path, target_ref: str, home: Path | None = None) -> str | None:
    """Why the update must not move to ``target_ref``, or ``None``: refuses only a pre-runtime target
    while this profile's or a sibling profile's state.db holds runtime ledger rows."""
    ledgered = _ledgered_profiles(home)
    if not ledgered or not target_predates_runtime(git_cmd, root, target_ref):
        return None
    lines = [f"✗ Refusing to move this install to {target_ref[:12]}: that release predates the gateway runtime "
             "these session stores now use. Downgrading is unsupported.",
             *(f"  {h / 'state.db'} has gateway runtime history." for _name, h in ledgered),
             "  The older release could not delete or prune those sessions.",
             "  To go back anyway: run `hermes gateway stop`, check out the older release with git, then put back "
             "each store's copy from before the upgrade:"]
    for name, h in ledgered:
        snapshot = _pre_runtime_snapshot(h)
        hermes = f"hermes -p {name}" if name else "hermes"
        lines.append(f"    {h}: run `{hermes}` and enter `/snapshot restore {snapshot}`." if snapshot else
                     f"    {h}: no pre-update snapshot predates the upgrade; restore a full backup taken before "
                     f"it with `{hermes} import <backup.zip>`.")
    lines += ["  Sessions created since the upgrade are not in that copy.", f"  Guide: {_DOCS}"]
    return "\n".join(lines)
