"""orchestrator-slash plugin — /resume and /skip slash commands.

Wires Slack-side handlers for the agentic-hub Plan 020 Temporal orchestrator.
When the parent `drainTierGraph` workflow emits `event=phase_blocked` (via the
`log_phase_blocked` activity), Blake gets pinged in Slack. He responds with
either:

    /resume <phase_id>   — re-queue the phase for a fresh attempt
    /skip   <phase_id>   — mark the Linear issue permanently Blocked

Both handlers signal the running `drain-tier-graph` workflow via temporalio's
client with payload ``{phase_id, action: "retry"|"skip"}``. The signal handler
on `DrainTierGraph` (already shipped in 020-D) queues the request and the main
loop processes it on the next iteration.

Auth: this plugin trusts the Hermes gateway's existing ``SLACK_ALLOWED_USERS``
gate. The Slack platform layer (gateway/platforms/slack.py) rejects events from
unauthorized user IDs before they reach plugin handlers.

Connection target is controlled by the env var ``TEMPORAL_HOST`` (default
``localhost:7233``) and ``TEMPORAL_NAMESPACE`` (default ``default``). The
workflow id is fixed at ``drain-tier-graph`` per Plan 020-D.
"""

from __future__ import annotations

from .orchestrator import (
    handle_resume,
    handle_skip,
    PHASE_ID_PATTERN,
    DRAIN_WORKFLOW_ID,
)
from .draft import handle_draft
from .daily import handle_daily

__all__ = [
    "register",
    "handle_resume",
    "handle_skip",
    "handle_draft",
    "handle_daily",
    "PHASE_ID_PATTERN",
    "DRAIN_WORKFLOW_ID",
]


def register(ctx) -> None:
    """Plugin entry point — registers /resume, /skip, and /draft slash commands."""
    ctx.register_command(
        "resume",
        handler=handle_resume,
        description="Retry a blocked orchestrator phase (signals drainTierGraph).",
        args_hint="<phase_id>",
    )
    ctx.register_command(
        "skip",
        handler=handle_skip,
        description="Mark a blocked orchestrator phase permanently Blocked.",
        args_hint="<phase_id>",
    )
    # Plan 030-A — Atlas-aware /draft skeleton. Atlas context fetch + LLM
    # draft generation are stubbed; 030-B and 030-C fill them in.
    ctx.register_command(
        "draft",
        handler=handle_draft,
        description="Draft a follow-up email (Atlas-aware; 030-A skeleton).",
        args_hint="<recipient> <context>",
    )
    # Plan 026-A — /daily morning brief with parallel fan-out across
    # Calendar, Inbox, Atlas (commitments / contradictions / contacts),
    # and orchestrator drain status.
    ctx.register_command(
        "daily",
        handler=handle_daily,
        description="Morning brief — top 3 items across Atlas + Calendar + Inbox + orchestrator.",
        args_hint="",
    )
