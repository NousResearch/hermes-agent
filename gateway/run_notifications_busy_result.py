"""Which finished background results may ride a running turn, and the process-completion carry.

``delegation.busy_result_mode`` governs subagent results and
``display.background_process_busy_result_mode`` governs ``notify_on_complete`` process completions.
With ``steer`` a result that finishes while its session's turn runs is appended to that turn's next
tool result (``agent.background_results``) instead of queueing a new turn.

A process completion has no durable claim to hold, so its delivery call waits on the carry instead:
the future resolves True once a tool result carried it and False when the turn ended without one,
and the caller then queues it as the next turn exactly as ``queue`` mode does.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any, Optional

logger = logging.getLogger("gateway.run")

BUSY_RESULT_MODE_KEYS = {
    "async_delegation": ("delegation", "busy_result_mode"),
    "completion": ("display", "background_process_busy_result_mode"),
}


@dataclasses.dataclass(frozen=True)
class CarriedCompletion:
    """A claimed completion riding a running turn (``delegation.busy_result_mode: steer``): what
    settling it later needs once a tool result carried it, or the turn ended without one."""

    evt: dict
    delegation_id: str
    claim_id: str
    identity: Any
    siblings: tuple
    waiter: Optional[asyncio.Future] = None  # a process completion's carry outcome


def busy_result_mode(evt_type: str) -> str:
    """The profile in scope's busy-result mode for ``evt_type``: ``queue`` (default) or ``steer``."""
    from gateway.run import _load_gateway_config
    from hermes_cli.config import cfg_get

    section, key = BUSY_RESULT_MODE_KEYS[evt_type]
    mode = str(cfg_get(_load_gateway_config(), section, key, default="") or "queue").strip().lower()
    if mode not in ("queue", "steer"):
        logger.warning("%s.%s=%r is not 'queue' or 'steer'; using 'queue'", section, key, mode)
        return "queue"
    return mode


def offer_process_completion(agent: Any, carried: Any, synth_text: str) -> asyncio.Future:
    """Offer a process completion to ``agent``'s next tool result; return the carry outcome future."""
    from agent.background_results import PROCESS_RESULT_MARKER, offer, withdraw

    waiter = asyncio.get_running_loop().create_future()
    carried = dataclasses.replace(carried, waiter=waiter)
    session_ids = [str(sid or "") for sid in carried.evt.get("_batch_session_ids") or [carried.evt.get("session_id")]]

    def already_read() -> bool:
        # The agent read every output with process(wait/log) this turn: showing it again is noise.
        import tools.process_registry as process_registry_module
        return all(process_registry_module.process_registry.is_completion_consumed(sid) for sid in session_ids)

    offer(agent, carried, synth_text, marker=PROCESS_RESULT_MARKER, already_read=already_read)
    # A cancelled delivery (shutdown) is retried by its watcher, so the turn must not carry it too.
    waiter.add_done_callback(lambda done: done.cancelled() and withdraw(agent, carried))
    logger.info("Carrying process completion %s into the running turn of %s",
                ", ".join(session_ids), carried.evt.get("session_key"))
    return waiter
