"""
session_orchestration — managed external-agent session layer for Hermes.

This package teaches Hermes to orchestrate external coding-agent CLI
sessions (claude-code, omp, and future agents) on the user's behalf.

Sub-modules
-----------
types
    Core types: ``SessionLifecycle``, ``SessionHandle``, ``Capabilities``.
adapters.base
    ``AgentAdapter`` ABC — the contract every concrete adapter implements.
registry
    SQLite-backed session registry (``session_orchestration`` table in
    ``state.db``) under the single-writer pattern: the cron watcher is the
    sole mutator of registry rows and counters; webhook-adopt and Discord-drive
    append intents to a queue table the cron drains.

Planned sub-modules (implemented in later tasks)
-------------------------------------------------
watcher
    Cron-driven liveness watcher (turn-change, heartbeat, hang layers).
relay
    Discord-message → tmux relay (holds per-session lock around send).

Implemented sub-modules
-----------------------
ingest
    Webhook ingest handler (HMAC verification, persistent event_id dedup,
    per-source rate limit, correlate/adopt, feed push).  Entry point:
    ``session_orchestration.ingest.process_z_harness_alert``.

Configuration
-------------
All behaviour is gated behind ``session_orchestration.enabled`` in Hermes
config (``~/.hermes/config.yaml``).  When disabled, the import of this
package has zero side-effects and no new network calls are made.
"""

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.registry import (
    SessionOrchestrationRegistry,
    canonical_repo_id,
)
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle

__all__ = [
    "AgentAdapter",
    "Capabilities",
    "SessionHandle",
    "SessionLifecycle",
    "SessionOrchestrationRegistry",
    "canonical_repo_id",
]
