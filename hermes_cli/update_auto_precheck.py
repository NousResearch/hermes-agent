"""Read-only recovery evidence before skipping a scheduled no-update run."""

from __future__ import annotations

import json
from pathlib import Path
import stat

from hermes_cli.update_auto_state import AutoUpdateContext


def _present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _nonempty_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(info.st_mode):
        return True
    return next(path.iterdir(), None) is not None


def _paths_in_scope(context: AutoUpdateContext) -> tuple[list[Path], list[Path], Path]:
    from hermes_constants import get_default_hermes_root, get_hermes_home
    from hermes_cli import update_host_obligation, update_pause_record
    from hermes_cli._early_recovery import ZIP_SWAP_JOURNAL, interrupted_pull_marker
    from hermes_cli.update_required_backup import _affected_homes
    from hermes_cli.venv_sync import completion_pending_path
    from pm.environments import owning_home_root

    root = context.install.resolve()
    if get_default_hermes_root(home=get_hermes_home()).resolve() != context.home:
        raise ValueError("Recovery evidence belongs to another data root")
    # These two frozen owners derive their install key from their code location.
    # A generation/borrowed entrypoint must let the canonical child settle scope.
    owners = (update_host_obligation, update_pause_record)
    if any(Path(module.__file__).resolve().parents[1] != root for module in owners):
        raise ValueError("Recovery evidence belongs to another code location")
    if owning_home_root(root) is not None:
        raise ValueError("Recovery evidence belongs to the installation's owning home")
    host = update_host_obligation.host_obligation_path()
    pause = update_pause_record.record_path()
    homes = _affected_homes()
    paths = [completion_pending_path(root), host,
             host.with_name(update_host_obligation.HOST_OBLIGATION_NAME), pause,
             root / ".update-incomplete", root / ".lazy-refresh-incomplete",
             interrupted_pull_marker(root), root / ZIP_SWAP_JOURNAL]
    paths.extend(home / update_host_obligation.PROFILE_MARKER_NAME for home in homes)
    directories = [home / "serve_restart_pending" for home in homes]
    return paths, directories, pause


def _pause_claim_present(pause: Path) -> bool:
    try:
        entries = list(pause.parent.iterdir())
    except FileNotFoundError:
        return False
    return any(path.name.startswith(pause.name + ".") and path.name.endswith(".claim")
               for path in entries)


def _read_receipt(path: Path) -> dict | None:
    if not _present(path):
        return None
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Latest update receipt is not a regular file")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Latest update receipt is not an object")
    return data


def _receipt_needs_recovery(receipt: dict, current_sha: str) -> bool:
    if receipt.get("outcome") != "success" or not receipt.get("finished_at"):
        return True
    if receipt.get("exit_code") not in (None, 0):
        return True
    if any(receipt.get(key) for key in ("followups", "user_action", "pending_manual_serves", "carried_manual_serves")):
        return True
    post = receipt.get("post_update")
    if not isinstance(post, dict) or post.get("sha") != current_sha:
        return True
    restart = receipt.get("gateway_restart", {})
    if not isinstance(restart, dict) or restart.get("incomplete") or restart.get("phase_error"):
        return True
    fleet = receipt.get("fleet", [])
    if not isinstance(fleet, list):
        return True
    return any(not isinstance(row, dict) or row.get("state") not in {"current", "external"}
               for row in fleet)


def recovery_needed(context: AutoUpdateContext, current_sha: str | None) -> bool:
    """Unknown evidence keeps the canonical recovery path; absence is a clean first run.

    Do not use the fleet's apparent read helpers here: some discharge markers or
    create manual-serve reminders. This precheck only examines durable evidence.
    """
    if not isinstance(current_sha, str) or not current_sha:
        return True
    try:
        paths, directories, pause = _paths_in_scope(context)
        if any(_present(path) for path in paths):
            return True
        if any(_nonempty_directory(path) for path in directories) or _pause_claim_present(pause):
            return True
        points = {context.receipt_directory / "latest.json",
                  context.home / "logs" / "update_receipts" / "latest.json"}
        for point in points:
            receipt = _read_receipt(point)
            if receipt is not None and _receipt_needs_recovery(receipt, current_sha):
                return True
    except (OSError, ValueError, TypeError):
        return True
    return False
