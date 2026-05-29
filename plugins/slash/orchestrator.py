"""/resume + /skip slash command handlers.

Both handlers parse a phase id from the raw slash-command arguments, connect
to the Temporal frontend, and signal the running ``drain-tier-graph`` parent
workflow with ``manual_resume`` + the appropriate action.

The handlers are synchronous (``fn(raw_args: str) -> str``) per
``PluginContext.register_command`` contract, but spin up an event loop to run
the async temporalio client call. We use ``asyncio.run`` so each invocation
gets a fresh loop — Hermes's gateway dispatches plugin commands in a
thread-pool, so no parent loop is in scope.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phase ids look like ``<plan-id>-<letter>``:
#   ``020-A``, ``015-1C``, ``atlas-019-B``, ``hub-020-E``.
# The trailing token is one or more digits + a single A-Z letter.
PHASE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-[0-9]*[A-Z]$")

MANUAL_RESUME_SIGNAL = "manual_resume"
DRAIN_WORKFLOW_ID = "drain-tier-graph"

# Surface a constant Temporal Web URL so the operator can watch the retry.
# Defaults to the laptop bridge port shipped by 020-D; overridable for
# Fargate/Tailscale once 020-F lands.
_DEFAULT_TEMPORAL_WEB_URL = "http://localhost:8233"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_phase_id(raw_args: str) -> str | None:
    """Pull the first whitespace-delimited token and validate its shape.

    Returns the normalized phase id, or None if the input doesn't look like
    a valid phase id (we reject loudly to avoid signalling the workflow with
    garbage payloads).
    """
    if not raw_args:
        return None
    token = raw_args.strip().split()[0] if raw_args.strip() else ""
    if not token:
        return None
    if not PHASE_ID_PATTERN.match(token):
        return None
    return token


def _temporal_host() -> str:
    return os.environ.get("TEMPORAL_HOST", "localhost:7233")


def _temporal_namespace() -> str:
    return os.environ.get("TEMPORAL_NAMESPACE", "default")


def _temporal_web_url() -> str:
    return os.environ.get("TEMPORAL_WEB_URL", _DEFAULT_TEMPORAL_WEB_URL)


async def _signal_drain_workflow(phase_id: str, action: str) -> None:
    """Open a Temporal client and signal the drain workflow.

    Imports ``temporalio`` lazily so the plugin can register on Hermes
    instances where temporalio isn't installed (it only becomes a hard
    requirement at signal time, and a clean error message is better than
    refusing to load the plugin).
    """
    try:
        from temporalio.client import Client  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "temporalio not installed in the Hermes environment — "
            "`pip install 'temporalio>=1.7,<2'` to enable /resume + /skip."
        ) from exc

    client = await Client.connect(_temporal_host(), namespace=_temporal_namespace())
    handle = client.get_workflow_handle(DRAIN_WORKFLOW_ID)
    payload: dict[str, Any] = {"phase_id": phase_id, "action": action}
    await handle.signal(MANUAL_RESUME_SIGNAL, payload)


def _signal_sync(phase_id: str, action: str) -> None:
    """Synchronous wrapper around ``_signal_drain_workflow``.

    Tests monkeypatch this function (rather than the async one) so they don't
    need an event loop to assert on call args.
    """
    asyncio.run(_signal_drain_workflow(phase_id, action))


# ---------------------------------------------------------------------------
# Slash command handlers
# ---------------------------------------------------------------------------


def _usage(cmd: str) -> str:
    return (
        f"Usage: /{cmd} <phase_id>\n"
        f"Example: /{cmd} 020-E\n"
        "Phase ids look like <plan-id>-<letter> "
        "(e.g. 020-E, 015-1C, atlas-019-B)."
    )


def handle_resume(raw_args: str) -> str:
    """``/resume <phase_id>`` — signal drainTierGraph to retry the phase."""
    phase_id = _parse_phase_id(raw_args)
    if phase_id is None:
        return _usage("resume")
    try:
        _signal_sync(phase_id, "retry")
    except Exception as exc:  # noqa: BLE001 — return a Slack-friendly error
        logger.exception("Failed to signal manual_resume retry for %s", phase_id)
        return (
            f":x: Failed to signal manual_resume for {phase_id}: {exc}. "
            "Check that the drain-tier-graph workflow is running and that "
            "TEMPORAL_HOST is reachable."
        )
    return (
        f":white_check_mark: Resume signal sent for {phase_id}. "
        f"Watch {_temporal_web_url()} for the new attempt."
    )


def handle_skip(raw_args: str) -> str:
    """``/skip <phase_id>`` — mark the phase permanently Blocked."""
    phase_id = _parse_phase_id(raw_args)
    if phase_id is None:
        return _usage("skip")
    try:
        _signal_sync(phase_id, "skip")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to signal manual_resume skip for %s", phase_id)
        return (
            f":x: Failed to signal skip for {phase_id}: {exc}. "
            "Check that the drain-tier-graph workflow is running."
        )
    return (
        f":white_check_mark: {phase_id} marked permanently Blocked. "
        "Downstream phases will stay Todo."
    )
