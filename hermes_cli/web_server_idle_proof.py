"""Diagnostic work snapshot for a Desktop-pooled ``hermes serve`` child.

Reads process admission, session/worker, delegation, cron and human-input ledgers. Unreadable
state is indeterminate, never idle. This snapshot alone is NOT permission to retire: only
``backend_retirement.RetirementFence.prepare`` freezes admission before taking the snapshot,
and only a confirmed commit permits the client to stop that exact backend generation.
The separate messaging gateway process owns its own lifecycle and drain protocol.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from hermes_cli.web_server_idle_exit import busy_ledger

_log = logging.getLogger(__name__)

_input_probe_failure_logged = False


def pending_human_input() -> Optional[int]:
    """Count of prompts waiting on a human (clarify/approval/sudo/secret requests plus queued
    gateway approvals); ``None`` when a ledger cannot be read."""
    global _input_probe_failure_logged
    try:
        from tui_gateway import server_requests
        from tools.approval import pending_gateway_approval_count

        return server_requests.open_request_count() + pending_gateway_approval_count()
    except Exception:
        if not _input_probe_failure_logged:
            _input_probe_failure_logged = True
            _log.warning("idle-proof input probe unavailable; this backend will report indeterminate",
                         exc_info=True)
        return None


def idle_proof(turn_probe: Callable[[], bool | str | None] = busy_ledger,
               input_probe: Callable[[], Optional[int]] = pending_human_input) -> dict:
    """``{"idle": True | False | None, "reason": str | None}`` plus ``"detail"`` naming the busy ledger.

    ``True`` only when no turn is in flight (session table AND cron ledger) and nothing is waiting
    on a human. ``None`` whenever either probe is indeterminate — the caller treats it exactly like
    busy. ``turn_probe`` may answer with a bool or with :func:`busy_ledger`'s name; the name is
    surfaced as ``detail`` so a backend that refuses to retire says which work it is protecting.
    """
    turn = turn_probe()
    if turn is None:
        return {"idle": None, "reason": "turn_probe_unavailable"}
    if turn:
        return {"idle": False, "reason": "turn_in_flight", "detail": turn if isinstance(turn, str) else None}
    pending = input_probe()
    if pending is None:
        return {"idle": None, "reason": "input_probe_unavailable"}
    if pending:
        return {"idle": False, "reason": "awaiting_human_input"}
    return {"idle": True, "reason": None}
