# SPDX-License-Identifier: MIT
"""OTel *log-record* emitter for the observability/otel plugin.

Companion to :mod:`emitter` (which emits GenAI **spans**). The TraceVerse
``/agent-coding`` dashboard is built from OTel **log records** — the
``claude_code.*`` event model — not spans. Claude Code / Codex / Copilot emit
log events, so they populate the dashboard; a spans-only agent does not.

This module turns the same three Hermes observability concepts — *session*,
*LLM API call*, *tool call* — into **log records** shaped like the records the
dashboard already aggregates, so a Hermes session shows up in the Activity /
Sessions / Leaderboards / breakdown views exactly like a Claude Code session.

Field contract the dashboard aggregates on (see the agent-coding API routes):

    attributes.event.name      session_start | api_response | tool_result
    attributes.model           model id            (by_model breakdown)
    attributes.tool_name       tool name           (by_tool breakdown)
    attributes.cost_usd        float               (leaderboard cost sums)
    attributes.input_tokens    int                 (leaderboard token sums)
    attributes.output_tokens   int                 (leaderboard token sums)
    attributes.decision_type   accept | reject ... (tool_result outcome)
    attributes.decision_source where the decision came from

Identity lives on the OTel **resource** (set in ``provider.build_logger``):

    resource.service.name      agent.coding.hermes
    resource.user.id / organization.id / team.id / session.id

``session.id`` is per-session, so it is emitted as a *record* attribute too and
the ingest ``agent-coding-router-pipeline`` promotes ``attributes.session.id``
onto ``resource.session.id`` at index time (idempotent) — matching how real
Claude Code telemetry (which also carries IDs as event attributes) is handled.

Like every observability path here, this NEVER raises out into the agent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

# Event names — `agent_coding.*` keeps them namespaced and obviously ours, while
# the dashboard keys only on the bare `event.name` value (session_start /
# api_response / tool_result) via the attribute promotions below.
EVENT_SESSION_START = "session_start"
EVENT_API_RESPONSE = "api_response"
EVENT_TOOL_RESULT = "tool_result"

GEN_AI_SYSTEM = "hermes"


def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    """Drop None / empty values so log records stay clean (OTel rejects None)."""
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        out[k] = v
    return out


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any, limit: int = 8192) -> str:
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:  # pragma: no cover - defensive
        return "<unrepr-able>"
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


class OtelLogEmitter:
    """Emit dashboard-shaped OTel log records for agent-coding sessions.

    Parameters
    ----------
    otel_logger:
        An OpenTelemetry ``Logger`` (from ``provider.build_logger``). When
        ``None`` every method is a no-op, so callers need no extra guards.
    agent_name:
        Stamped as ``attributes.agent_type`` (mirrors Claude Code's field).
    capture_content:
        When true, prompt/response/tool text is attached to records.
    """

    def __init__(
        self,
        otel_logger: Any,
        *,
        agent_name: str = "hermes",
        capture_content: bool = False,
    ) -> None:
        self._logger = otel_logger
        self._agent_name = agent_name
        self._capture_content = capture_content
        self._seq = 0

    @property
    def enabled(self) -> bool:
        return self._logger is not None

    # -- low-level emit ----------------------------------------------------

    def _emit(self, event_name: str, attributes: Mapping[str, Any], *, body: str | None = None) -> None:
        if self._logger is None:
            return
        try:
            from opentelemetry._logs import LogRecord, SeverityNumber

            self._seq += 1
            now_ns = time.time_ns()
            attrs = _clean(
                {
                    "event.name": event_name,
                    "agent_type": self._agent_name,
                    "gen_ai.system": GEN_AI_SYSTEM,
                    "event.sequence": self._seq,
                    **dict(attributes),
                }
            )
            record = LogRecord(
                timestamp=now_ns,
                observed_timestamp=now_ns,
                severity_number=SeverityNumber.INFO,
                severity_text="INFO",
                body=body if body is not None else event_name,
                attributes=attrs,
            )
            self._logger.emit(record)
        except Exception:  # pragma: no cover - fail-open
            logger.debug("OtelLogEmitter: failed to emit %s record", event_name, exc_info=True)

    # -- session ----------------------------------------------------------

    def session_start(
        self,
        *,
        session_id: str | None,
        model: str | None = None,
        user_prompt: str | None = None,
    ) -> None:
        """Emit the session-boundary record (opens a session in the dashboard)."""
        attrs: dict[str, Any] = {
            "session.id": session_id,
            "model": model,
        }
        if self._capture_content and user_prompt:
            attrs["prompt"] = _safe_str(user_prompt)
        self._emit(EVENT_SESSION_START, attrs, body=user_prompt if self._capture_content else None)

    # -- LLM call ---------------------------------------------------------

    def api_response(
        self,
        *,
        session_id: str | None,
        request_model: str | None,
        response_model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        finish_reasons: Sequence[str] | None = None,
        ttft_ms: float | None = None,
        response_text: str | None = None,
    ) -> None:
        """Emit one ``api_response`` record per LLM call.

        Carries model + token usage + cost so the by_model breakdown and the
        per-user / per-org leaderboard cost & token sums populate.
        """
        attrs: dict[str, Any] = {
            "session.id": session_id,
            "model": response_model or request_model,
            "input_tokens": _as_int(input_tokens),
            "output_tokens": _as_int(output_tokens),
            "cost_usd": _as_float(cost_usd),
        }
        if finish_reasons:
            attrs["finish_reason"] = list(finish_reasons)[0]
        ttft = _as_float(ttft_ms)
        if ttft is not None:
            attrs["ttft_ms"] = ttft
        # Response text rides on the record body (privacy-gated), not as a
        # duplicate attribute — matching session_start's prompt handling.
        self._emit(EVENT_API_RESPONSE, attrs, body=response_text if self._capture_content else None)

    # -- tool call --------------------------------------------------------

    def tool_result(
        self,
        *,
        session_id: str | None,
        tool_name: str,
        tool_type: str = "function",
        arguments: Mapping[str, Any] | None = None,
        result: Any = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Emit one ``tool_result`` record per tool call.

        Drives the by_tool breakdown and the post-execution tool-decision funnel
        (decision_type / decision_source).
        """
        ok = error is None
        attrs: dict[str, Any] = {
            "session.id": session_id,
            "tool_name": tool_name,
            "tool_type": tool_type,
            "status_code": "success" if ok else "error",
            # The dashboard's post-exec funnel reads decision_type/source on
            # tool_result events; for an auto-run agent the outcome is "executed".
            "decision_type": "executed" if ok else "error",
            "decision_source": "agent",
        }
        if not ok:
            attrs["error"] = _safe_str(error if isinstance(error, str) else repr(error))
            attrs["error_type"] = type(error).__name__ if isinstance(error, BaseException) else "ToolError"
        if self._capture_content and arguments is not None:
            attrs["tool_input"] = _safe_str(arguments)
        if self._capture_content and result is not None:
            attrs["tool_output"] = _safe_str(result)
        self._emit(EVENT_TOOL_RESULT, attrs)
