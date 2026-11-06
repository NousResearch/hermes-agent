"""`hermes update --rollback`: undo the last update from its own receipt.

Resets the checkout to the pre-update commit recorded in the latest
success/partial receipt, restores the pre-update quick snapshot (state files;
through the SQLite backup API so a running gateway keeps serving), then
re-syncs dependencies so the environment matches the restored code.

Runs UNDER the update lock (wired in ``cmd_update`` after a successful
acquire), so "no live update" is guaranteed by construction rather than
re-checked. Refuses a dirty tree — rollback moves HEAD, never user work.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

ROLLBACK_OK = 0
ROLLBACK_NOTHING = 0
ROLLBACK_REFUSED = 2
ROLLBACK_FAILED = 1

_ROLLBACK_TARGETS = ("success", "partial")


def _m():
    from hermes_cli import main

    return main


def _rollback_target() -> tuple[dict[str, Any], str, str | None]:
    """(receipt, pre_sha, snapshot_id) or raise with the user-facing reason."""
    from hermes_cli.update_receipt import read_latest_receipt

    record = read_latest_receipt()
    if not isinstance(record, dict):
        raise RuntimeError("no update receipt yet — nothing has updated this home, nothing to roll back.")
    outcome = record.get("outcome")
    if outcome not in _ROLLBACK_TARGETS:
        raise RuntimeError(
            f"latest update {outcome or 'has no outcome'} ({record.get('started_at', '?')}) — "
            "rollback only follows an update that committed (outcome: success/partial)."
        )
    pre = record.get("pre_update") or {}
    sha = pre.get("sha") if isinstance(pre, dict) else None
    if not sha:
        raise RuntimeError("latest receipt records no pre-update commit — cannot choose a rollback target.")
    snapshot_id = None
    for step in record.get("steps") or []:
        if isinstance(step, dict) and step.get("name") == "pre_update_backup":
            detail = str(step.get("detail") or "")
            if detail.startswith("snapshot="):
                snapshot_id = detail[len("snapshot="):]
    return record, sha, snapshot_id


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, timeout=120,
        check=False,  # callers branch on returncode; a failed git is a refusal, not an exception
    )


def _cmd_update_rollback(project_root: Path) -> int:
    """Reset to the pre-update commit, restore state, re-sync deps. Returns exit code."""
    root = Path(project_root)
    try:
        record, pre_sha, snapshot_id = _rollback_target()
    except RuntimeError as exc:
        print(f"✗ Cannot roll back: {exc}")
        return ROLLBACK_REFUSED

    if not (root / ".git").exists() and _git(root, "rev-parse", "--git-dir").returncode != 0:
        # Same retarget already ran in cmd_update; if we are still not in a repo,
        # the owning checkout is gone (zip install, moved tree).
        print("✗ Cannot roll back: not a git checkout. Reinstall instead:")
        print("  npm i -g hermes-agent  |  brew install hermes-agent  |  hermes setup --uninstall, then reinstall")
        return ROLLBACK_REFUSED

    head = _git(root, "rev-parse", "HEAD")
    if head.returncode == 0 and head.stdout.strip() == pre_sha:
        print(f"✓ Already on the pre-update commit ({pre_sha[:12]}) — nothing to roll back.")
        return ROLLBACK_NOTHING

    dirty = _git(root, "status", "--porcelain")
    if dirty.returncode == 0 and dirty.stdout.strip():
        print("✗ Cannot roll back: the checkout has uncommitted changes. Commit or stash them first:")
        for line in dirty.stdout.splitlines()[:10]:
            print(f"    {line}")
        return ROLLBACK_REFUSED

    started = record.get("started_at", "?")
    print(f"↩ Rolling back the update from {started} → {pre_sha[:12]} ...")

    reset = _git(root, "reset", "--hard", pre_sha)
    if reset.returncode != 0:
        print(f"✗ git reset failed: {(reset.stderr or reset.stdout).strip()[:300]}")
        return ROLLBACK_FAILED
    print(f"  code: checkout reset to {pre_sha[:12]}")

    if snapshot_id:
        try:
            from hermes_cli.backup import restore_quick_snapshot
        except Exception as exc:  # pragma: no cover - import failure is environmental
            print(f"  ⚠ state: snapshot {snapshot_id} found but the restore path failed to load ({exc}); code is rolled back, state is not.")
        else:
            if restore_quick_snapshot(snapshot_id):
                print(f"  state: restored pre-update snapshot {snapshot_id}")
            else:
                print(f"  ⚠ state: snapshot {snapshot_id} refused or failed (live gateway keeps current state); code is rolled back, state is not.")
    else:
        print("  ⚠ state: that update recorded no snapshot (backup was off) — state files stay as they are.")

    try:
        from hermes_cli.venv_sync import sync as venv_sync
    except Exception as exc:  # pragma: no cover - import failure is environmental
        print(f"  ⚠ deps: could not load the dependency sync ({exc}); run `hermes update` to roll forward instead.")
        return ROLLBACK_FAILED
    result = venv_sync(root) if callable(venv_sync) else {}
    if isinstance(result, dict) and result.get("ok", True):
        print("  deps: environment re-synced to the restored code")
    else:
        detail = result.get("detail", "") if isinstance(result, dict) else ""
        print(f"  ⚠ deps: re-sync reported a problem ({detail}); the code is rolled back — re-run `hermes update` to roll forward.")
        return ROLLBACK_FAILED

    from hermes_cli.update_cmd import _record_update_step

    _record_update_step("rollback", True, f"reset to {pre_sha[:12]}")
    print(f"✓ Rolled back to {pre_sha[:12]}. Gateways pick it up on their next restart (`hermes gateway restart`).")
    return ROLLBACK_OK
