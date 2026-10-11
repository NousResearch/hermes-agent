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
from contextlib import closing
from pathlib import Path

# The runtime ledger DDL: a target whose state-store modules lack it predates the cutover.
_RUNTIME_SCHEMA_MARKER = "CREATE TABLE IF NOT EXISTS session_admissions"
_LEDGER_PROBE = ("SELECT EXISTS(SELECT 1 FROM session_admissions) "
                 "OR EXISTS(SELECT 1 FROM worker_executions)")
_DOCS = "https://hermes-agent.nousresearch.com/docs/getting-started/updating#rolling-back-to-an-older-release"


def target_predates_runtime(git_cmd, root: Path, target_ref: str) -> bool:
    """True when ``target_ref`` does not descend from HEAD and none of its ``hermes_state*.py`` files
    defines the runtime ledger. The pathspec keeps a blobless partial clone from fetching the whole
    tree. Any git failure proves nothing (False)."""
    import subprocess

    from hermes_cli.update_custody import run_git

    def git(*args: str) -> int:
        try:
            return run_git(git_cmd, list(args), cwd=str(root), capture_output=True, timeout=120).returncode
        except (OSError, subprocess.SubprocessError):
            return -1

    if git("merge-base", "--is-ancestor", "HEAD", target_ref) == 0:
        return False  # forward update: whatever the target's schema, it was written knowing this one
    if git("cat-file", "-e", f"{target_ref}^{{commit}}") != 0:
        return False
    # Exit 1 = no match (the target lacks the ledger); 0 = found; anything else is a git error.
    return git("grep", "-q", "-F", _RUNTIME_SCHEMA_MARKER, target_ref, "--", "hermes_state*.py") == 1


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


def downgrade_refusal(git_cmd, root: Path, target_ref: str, home: Path | None = None) -> str | None:
    """Why the update must not move to ``target_ref``, or ``None``: refuses only a pre-runtime target
    while this profile's or a sibling profile's state.db holds runtime ledger rows."""
    from hermes_cli.backup import _sibling_profile_homes
    from hermes_constants import get_hermes_home

    home = Path(home or get_hermes_home())
    profiles = [("", home), *_sibling_profile_homes(home)]
    ledgered = [(name, h) for name, h in profiles if _has_ledger_rows(h / "state.db")]
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
