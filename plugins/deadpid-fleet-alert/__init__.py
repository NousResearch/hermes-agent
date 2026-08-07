"""Dead-PID fleet alert relay plugin.

The kanban dispatcher fires ``kanban_failure_alert`` when a worker-death
fingerprint reaches ``consecutive_failures >= 3``. This plugin turns that strict
hook into one compact fleet notification and deduplicates by fingerprint so a
storm of cards produces one message per window, not one per task.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict

from hermes_cli.config import cfg_get, load_config

logger = logging.getLogger(__name__)

_DEFAULT_TARGET = "discord:#fleet-reports"
_DEFAULT_DEDUP_WINDOW_SECONDS = 30 * 60
_seen_by_fingerprint: Dict[str, float] = {}


def _settings() -> tuple[str, int]:
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    target = cfg_get(
        cfg,
        "kanban",
        "failure_alert",
        "target",
        default=_DEFAULT_TARGET,
    )
    window = cfg_get(
        cfg,
        "kanban",
        "failure_alert",
        "dedup_window_seconds",
        default=_DEFAULT_DEDUP_WINDOW_SECONDS,
    )
    try:
        dedup_window = max(1, int(window))
    except (TypeError, ValueError):
        dedup_window = _DEFAULT_DEDUP_WINDOW_SECONDS
    if not isinstance(target, str) or not target.strip():
        target = _DEFAULT_TARGET
    return target.strip(), dedup_window


def _prune(now: float, window: int) -> None:
    stale = [fp for fp, ts in _seen_by_fingerprint.items() if now - ts >= window]
    for fp in stale:
        _seen_by_fingerprint.pop(fp, None)


def _format_message(
    *,
    task_id: str,
    board: str | None,
    assignee: str | None,
    consecutive_failures: int,
    fingerprint: str,
    error: str,
    run_id: int | None = None,
    **_: Any,
) -> str:
    board_part = board or "default"
    assignee_part = assignee or "unassigned"
    run_part = f" run={run_id}" if run_id is not None else ""
    return (
        "KANBAN FAILURE ALERT: dead-PID worker failures reached "
        f"cf={consecutive_failures} on {board_part}/{task_id}{run_part} "
        f"assignee={assignee_part} fp={fingerprint!r} error={error[:240]!r}"
    )


def _on_kanban_failure_alert(**kwargs: Any) -> Dict[str, Any] | None:
    fingerprint = str(kwargs.get("fingerprint") or "").strip()
    if not fingerprint:
        return None
    target, window = _settings()
    now = time.time()
    _prune(now, window)
    if fingerprint in _seen_by_fingerprint:
        logger.info("deadpid-fleet-alert: deduped fingerprint %s", fingerprint)
        return {"deduped": True, "fingerprint": fingerprint}
    _seen_by_fingerprint[fingerprint] = now

    message = _format_message(**kwargs)
    proc = subprocess.run(
        ["hermes", "send", "--to", target, "--quiet", message],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        _seen_by_fingerprint.pop(fingerprint, None)
        raise RuntimeError(
            "deadpid-fleet-alert delivery failed "
            f"target={target!r} rc={proc.returncode} stderr={proc.stderr.strip()!r}"
        )
    return {"sent": True, "target": target, "fingerprint": fingerprint}


def register(ctx) -> None:
    ctx.register_hook("kanban_failure_alert", _on_kanban_failure_alert)
