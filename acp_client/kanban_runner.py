"""Opt-in ACP transport runner for the Kanban external-coder lane (default-off).

This is the seam a (future) ``kanban-acp-worker.sh`` wrapper would call instead
of ``claude -p`` *when explicitly configured*.  It is structured so all the
decision logic is pure and unit-testable, and the only code that launches a
subprocess is gated behind **both** the transport opt-in / launch guard
(:mod:`acp_client.transport`) **and** an explicit ``allow_launch=True``.

Nothing in this module launches an external CLI unless :func:`build_launch_plan`
returns an ACP plan **and** the caller invokes :func:`run_acp_lane` with
``allow_launch=True``.  Tests drive everything through the pure planners and an
injected ``connection_factory``, so no real ``claude``/``codex`` process is ever
started.

This module is **not wired** into the live Kanban worker, ``delegate_task``, or
any runtime path.  Activating it (or the real-launch path) is a separate
Filip-approval gate; see the Phase-2 implementation report.

Provenance: concepts adopted from prototype ``72bd6be09``
(``acp_client/kanban_runner.py``, task ``t_7514d8c1``) and **adapted** onto the
current ``main`` ``acp_client`` API surface (``OutboundConnection.spawn``,
``DEFAULT_REGISTRY``, ``normalize_stop_reason``, ``EventTranslator(on_event=)``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from acp_client import AcpClientUnavailable
from acp_client import transport as _transport
from acp_client.transport_registry import DEFAULT_REGISTRY, TransportRegistry

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "claude"
PROGRESS_FILENAME = "progress.jsonl"


@dataclass
class LaunchPlan:
    """The resolved plan for one worker invocation.

    When ``transport == 'pty'`` the caller must fall back to the existing
    ``kanban-claude-code-worker.sh`` behaviour.  When ``transport == 'acp'`` the
    remaining fields describe how to launch the external agent.
    """

    transport: str
    reason: str
    backend: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    fell_back: bool = False
    refusal: Optional[str] = None

    @property
    def use_acp(self) -> bool:
        return self.transport == _transport.TRANSPORT_ACP


def build_launch_plan(
    *,
    workspace: str,
    backend: str = DEFAULT_BACKEND,
    env: Optional[Dict[str, str]] = None,
    strict: bool = False,
    registry: Optional[TransportRegistry] = None,
) -> LaunchPlan:
    """Resolve transport + backend into a launch plan.  Never launches anything.

    Args:
        workspace: task workspace path (cwd for the agent; reserved for future
            arg templating).
        backend: which opt-in backend to use (default ``claude``).
        env: source environment (defaults to ``os.environ`` inside
            :func:`~acp_client.transport.resolve_transport`).
        strict: if True and ACP was requested but is unavailable, raise
            :class:`AcpClientUnavailable` instead of falling back to PTY.
        registry: opt-in transport registry (defaults to ``DEFAULT_REGISTRY``).
    """
    decision = _transport.resolve_transport(env)

    if not decision.is_acp:
        if decision.fell_back and strict and decision.refusal:
            raise AcpClientUnavailable(decision.refusal)
        return LaunchPlan(
            transport=_transport.TRANSPORT_PTY,
            reason=decision.reason,
            fell_back=decision.fell_back,
            refusal=decision.refusal,
        )

    # ACP requested and gated on.  Resolve the backend against the opt-in
    # registry (deny-by-default: an unknown name is never silently launched).
    reg = registry or DEFAULT_REGISTRY
    try:
        spec = reg.resolve(backend)
    except KeyError as exc:
        msg = str(exc)
        if strict:
            raise AcpClientUnavailable(msg) from exc
        return LaunchPlan(
            transport=_transport.TRANSPORT_PTY,
            reason=f"{msg} Falling back to PTY lane.",
            fell_back=True,
            refusal=msg,
        )

    return LaunchPlan(
        transport=_transport.TRANSPORT_ACP,
        reason=decision.reason,
        backend=spec.name,
        command=spec.command,
        args=list(spec.args),
        env=spec.resolve_env(env),
    )


@dataclass
class WritebackDecision:
    """What the worker should do with the Kanban card after the run."""

    action: str  # complete | block
    lane_status: str  # done | blocked
    reason: str
    summary: str


# Structured replacement for the PTY lane's substring scraping of
# ``Status: DONE`` / ``Status: BLOCKED`` (design §1.4).  Only an explicit
# ``end_turn`` ("done") completes the card; every other terminal state is a
# conservative block that surfaces for review rather than auto-completing.
_WRITEBACK_BY_CATEGORY: Dict[str, tuple[str, str]] = {
    "done": ("complete", "done"),
    "limit": ("block", "blocked"),
    "cancelled": ("block", "blocked"),
    "refusal": ("block", "blocked"),
    "error": ("block", "blocked"),
}


def decide_writeback(stop_reason: Optional[str], summary: str) -> WritebackDecision:
    """Map a final ACP ``stop_reason`` to a Kanban writeback action.

    Uses :func:`acp_client.connection.normalize_stop_reason` so the runner
    branches on a stable ``done``/``limit``/``cancelled``/``refusal``/``error``
    vocabulary instead of raw backend strings.
    """
    from acp_client.connection import normalize_stop_reason

    category = normalize_stop_reason(stop_reason)
    action, lane_status = _WRITEBACK_BY_CATEGORY.get(category, ("block", "blocked"))
    return WritebackDecision(
        action=action,
        lane_status=lane_status,
        reason=f"stop_reason={stop_reason!r} -> {category}",
        summary=(summary or "")[-3500:],
    )


class ProgressWriter:
    """Append-only writer for ``${WORKSPACE}/progress.jsonl``.

    Compatible as an ``on_event`` sink for :class:`~acp_client.event_translator.EventTranslator`
    and as an ``audit_log`` sink for :class:`~acp_client.permission_relay.PermissionRelay`
    (the relay passes a ``PermissionDecision``, normalised here to a dict).
    """

    def __init__(self, workspace: str, *, filename: str = PROGRESS_FILENAME):
        self.path = Path(workspace) / filename

    def write(self, record: Any) -> None:
        payload = record if isinstance(record, dict) else _coerce_record(record)
        line = json.dumps(payload, default=str, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _coerce_record(record: Any) -> Dict[str, Any]:
    """Best-effort normalisation of a non-dict event (e.g. a dataclass)."""
    for attr in ("__dict__",):
        data = getattr(record, attr, None)
        if isinstance(data, dict):
            return {"event": "permission", **data}
    return {"event": "raw", "value": str(record)}


async def run_acp_lane(
    plan: LaunchPlan,
    *,
    workspace: str,
    prompt_text: str,
    sessions: Any = None,
    progress: Optional[ProgressWriter] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    allow_launch: bool = False,
    connection_factory: Optional[Callable[..., Any]] = None,
    registry: Optional[TransportRegistry] = None,
    base_env: Optional[Dict[str, str]] = None,
) -> WritebackDecision:
    """Drive one task through the external ACP agent.

    Safety: this refuses to actually spawn a process unless ``allow_launch`` is
    True.  Tests pass a ``connection_factory`` that yields a fake connection, so
    no real CLI is launched even when ``allow_launch`` is True under test.

    Args:
        plan: an ACP :class:`LaunchPlan` from :func:`build_launch_plan`.
        workspace: task workspace path (cwd for the agent).
        prompt_text: the rendered task prompt.
        sessions: optional :class:`~acp_client.outbound_session.OutboundSessionManager`.
        progress: optional :class:`ProgressWriter` for ``progress.jsonl``.
        on_event: optional extra event sink (in addition to ``progress``).
        allow_launch: must be True to spawn the real subprocess.
        connection_factory: optional injectable factory with the same signature
            as :meth:`acp_client.connection.OutboundConnection.spawn`, used in
            tests in place of the real spawn (so nothing is launched).
        registry: opt-in transport registry forwarded to the real spawn path.
        base_env: env source forwarded to the real spawn path.
    """
    if not plan.use_acp:
        raise ValueError("run_acp_lane called with a non-ACP plan; use the PTY lane.")

    if connection_factory is None and not allow_launch:
        raise AcpClientUnavailable(
            "Refusing to launch an external ACP agent without allow_launch=True. "
            "This is the final safety gate; set it only in an explicitly approved "
            "context."
        )

    assistant_chunks: List[str] = []

    def _sink(event: Dict[str, Any]) -> None:
        if isinstance(event, dict) and event.get("type") == "agent_message_chunk":
            assistant_chunks.append(str(event.get("text") or ""))
        if progress is not None:
            progress.write(event)
        if on_event is not None:
            on_event(event)

    if connection_factory is None:
        from acp_client.connection import OutboundConnection

        factory = OutboundConnection.spawn
    else:
        factory = connection_factory

    if progress is not None:
        progress.write({
            "event": "lane_start",
            "backend": plan.backend,
            "transport": "acp",
        })

    cm = factory(
        plan.backend,
        cwd=workspace,
        workspace_path=workspace,
        registry=registry,
        base_env=base_env,
        sessions=sessions,
        on_event=_sink,
    )
    async with cm as conn:
        state = await conn.create_session(cwd=workspace)
        session_id = getattr(state, "session_id", None) or str(state)
        resp = await conn.prompt(session_id, prompt_text)
        stop_reason = getattr(resp, "stop_reason", None)

    summary = "".join(assistant_chunks)
    decision = decide_writeback(stop_reason, summary)
    if progress is not None:
        progress.write({
            "event": "lane_end",
            "stop_reason": decision.lane_status,
            "action": decision.action,
        })
    return decision
