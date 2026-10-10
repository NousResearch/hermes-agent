"""CaMeL-style capability guard implemented on Hermes' native plugin hooks.

The plugin never edits the system prompt, wraps tool messages, or replaces
Hermes' executor.  ``pre_llm_call`` captures trusted user intent,
``post_tool_call`` records which tool outputs entered the turn as untrusted
data, and ``pre_tool_call`` gates later side effects.  Hermes continues to
construct every tool result through ``make_tool_result_message()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import threading
from typing import Any, Mapping, Sequence

# Hermes loads standalone plugin directories as packages with
# ``submodule_search_locations``; ty cannot infer that dynamic package root.
from .policy import (  # ty: ignore[unresolved-import]
    CAPABILITY_LABELS,
    CLASSIFIER_INSTRUCTIONS,
    CLASSIFIER_SCHEMA,
    CapabilityPlan,
    TurnState,
    capability_for,
    is_untrusted_output,
    normalize_mode,
    normalized_capabilities,
)


@dataclass(frozen=True)
class GuardSettings:
    mode: str = "off"
    trace_enabled: bool = False
    trace_max_events: int = 200
    classifier_timeout_seconds: float = 12.0


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _load_settings(ctx: Any) -> GuardSettings:
    return GuardSettings(
        mode=normalize_mode(ctx.get_config("mode", "off")),
        trace_enabled=ctx.get_config("trace_enabled", False) is True,
        trace_max_events=_bounded_int(
            ctx.get_config("trace_max_events", 200),
            default=200,
            minimum=20,
            maximum=2000,
        ),
        classifier_timeout_seconds=_bounded_float(
            ctx.get_config("classifier_timeout_seconds", 12.0),
            default=12.0,
            minimum=2.0,
            maximum=120.0,
        ),
    )


def _text_only_user_message(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ""
    pieces: list[str] = []
    for part in value:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            pieces.append(text.strip())
    return "\n".join(pieces)


def _history_untrusted_sources(history: Any) -> set[str]:
    sources: set[str] = set()
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return sources
    for message in history:
        if not isinstance(message, Mapping) or message.get("role") != "tool":
            continue
        name = str(message.get("tool_name") or message.get("name") or "").strip()
        if is_untrusted_output(name):
            sources.add(name)
    return sources


def _strict_classifier_capabilities(payload: Mapping[str, Any], key: str) -> set[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"classifier field {key} must be a list")
    if any(
        not isinstance(item, str) or item not in CAPABILITY_LABELS for item in value
    ):
        raise ValueError(f"classifier field {key} contains an invalid capability")
    normalized = normalized_capabilities(value)
    if len(normalized) != len(value):
        raise ValueError(f"classifier field {key} contains duplicate capabilities")
    return normalized


class CamelGuardRuntime:
    """Session/turn-scoped adapter around the generic Hermes hook surface."""

    def __init__(self, ctx: Any, settings: GuardSettings) -> None:
        self._ctx = ctx
        self.settings = settings
        self._lock = threading.RLock()
        self._turns: dict[tuple[str, str], TurnState] = {}
        self._current_turn_by_scope: dict[str, str] = {}

    @staticmethod
    def _scope_id(session_id: str = "", task_id: str = "") -> str:
        return str(session_id or task_id or "process-default")

    def _purge_scope(self, scope_id: str) -> None:
        with self._lock:
            self._current_turn_by_scope.pop(scope_id, None)
            stale = [key for key in self._turns if key[0] == scope_id]
            for key in stale:
                self._turns.pop(key, None)

    def _state_for(
        self,
        *,
        session_id: str = "",
        task_id: str = "",
        turn_id: str = "",
    ) -> TurnState | None:
        scope_id = self._scope_id(session_id, task_id)
        with self._lock:
            resolved_turn = str(
                turn_id or self._current_turn_by_scope.get(scope_id) or ""
            )
            if not resolved_turn:
                return None
            return self._turns.get((scope_id, resolved_turn))

    def _source_snapshot(self, state: TurnState) -> list[str]:
        # Concurrent executor workers can finish untrusted tools while a
        # sibling reaches the policy hook. Never iterate a mutating set.
        with self._lock:
            return sorted(state.untrusted_sources)

    def on_pre_llm_call(
        self,
        *,
        user_message: Any = "",
        conversation_history: Any = None,
        session_id: str = "",
        task_id: str = "",
        turn_id: str = "",
        **_: Any,
    ) -> None:
        scope_id = self._scope_id(session_id, task_id)
        if self.settings.mode == "off":
            self._purge_scope(scope_id)
            return None

        resolved_turn = str(turn_id or "turn-unknown")
        state = TurnState(
            scope_id=scope_id,
            turn_id=resolved_turn,
            trusted_user_message=_text_only_user_message(user_message),
            untrusted_sources=_history_untrusted_sources(conversation_history),
        )
        with self._lock:
            stale = [key for key in self._turns if key[0] == scope_id]
            for key in stale:
                self._turns.pop(key, None)
            self._turns[(scope_id, resolved_turn)] = state
            self._current_turn_by_scope[scope_id] = resolved_turn
        # Returning context would mutate the user-message wire payload.  The
        # guard is an execution policy, so it returns nothing here.
        return None

    def on_post_tool_call(
        self,
        *,
        tool_name: str = "",
        status: str = "",
        session_id: str = "",
        task_id: str = "",
        turn_id: str = "",
        **_: Any,
    ) -> None:
        if self.settings.mode == "off":
            return None
        if status in {"blocked", "cancelled"} or not is_untrusted_output(tool_name):
            return None
        state = self._state_for(
            session_id=session_id,
            task_id=task_id,
            turn_id=turn_id,
        )
        if state is not None:
            with self._lock:
                state.untrusted_sources.add(str(tool_name))
        return None

    def _classify(self, state: TurnState) -> CapabilityPlan:
        with state.classification_lock:
            if state.plan is not None:
                return state.plan
            try:
                result = self._ctx.llm.complete_structured(
                    instructions=CLASSIFIER_INSTRUCTIONS,
                    input=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {"trusted_user_message": state.trusted_user_message},
                                ensure_ascii=False,
                            ),
                        }
                    ],
                    json_schema=CLASSIFIER_SCHEMA,
                    schema_name="camel_guard_capabilities",
                    temperature=0,
                    max_tokens=260,
                    timeout=self.settings.classifier_timeout_seconds,
                    purpose="classify trusted tool capability intent",
                )
                payload = result.parsed
                if not isinstance(payload, Mapping):
                    raise ValueError("classifier did not return a structured object")
                required_fields = {
                    "allowed_capabilities",
                    "denied_capabilities",
                    "rationale",
                }
                if set(payload) != required_fields:
                    raise ValueError("classifier returned an invalid field set")
                if not isinstance(payload.get("rationale"), str):
                    raise ValueError("classifier rationale must be a string")
                allowed = _strict_classifier_capabilities(
                    payload,
                    "allowed_capabilities",
                )
                denied = _strict_classifier_capabilities(
                    payload,
                    "denied_capabilities",
                )
                allowed.difference_update(denied)
                state.plan = CapabilityPlan(
                    allowed=frozenset(allowed),
                    denied=frozenset(denied),
                    status="ok",
                    rationale=str(payload.get("rationale") or "")[:240],
                )
            except Exception as exc:
                # The classifier is part of the security boundary.  Failure is
                # a read-only plan; monitor observes it, enforce blocks it.
                state.plan = CapabilityPlan(
                    status="fallback_read_only",
                    rationale=type(exc).__name__,
                )
            return state.plan

    def _record_event(
        self,
        *,
        state: TurnState | None,
        tool_name: str,
        capability: str,
        outcome: str,
        reason_code: str,
        classifier_status: str,
    ) -> None:
        if (
            not self.settings.trace_enabled
            or self.settings.mode == "off"
            or state is None
        ):
            return
        event = {
            "at": datetime.now(timezone.utc).isoformat(),
            "session_id": state.scope_id,
            "turn_id": state.turn_id,
            "mode": self.settings.mode,
            "tool_name": str(tool_name),
            "capability": capability,
            "outcome": outcome,
            "reason_code": reason_code,
            "classifier_status": classifier_status,
            "untrusted_sources": self._source_snapshot(state),
        }
        # Trace data is explicitly opt-in, bounded, and contains neither the
        # user request nor tool arguments/results.
        with self._lock:
            events = self._ctx.state.get("decision_events", default=[])
            if not isinstance(events, list):
                events = []
            events.append(event)
            events = events[-self.settings.trace_max_events :]
            self._ctx.state.set("decision_events", events)

    def on_pre_tool_call(
        self,
        *,
        tool_name: str = "",
        args: Any = None,
        session_id: str = "",
        task_id: str = "",
        turn_id: str = "",
        **_: Any,
    ) -> dict[str, str] | None:
        if self.settings.mode == "off":
            return None
        capability = capability_for(
            tool_name,
            args if isinstance(args, Mapping) else {},
        )
        if not capability:
            return None

        state = self._state_for(
            session_id=session_id,
            task_id=task_id,
            turn_id=turn_id,
        )
        if state is None:
            reason_code = "missing_trusted_turn"
            classifier_status = "not_run"
            allowed = False
            sources: list[str] = []
        else:
            sources = self._source_snapshot(state)
            if not sources:
                # No untrusted data influenced this turn, so legacy behavior
                # is preserved and no auxiliary call is needed.
                return None
            plan = self._classify(state)
            classifier_status = plan.status
            if capability in plan.denied:
                allowed = False
                reason_code = "explicitly_denied"
            elif capability in plan.allowed:
                allowed = True
                reason_code = "trusted_intent_authorized"
            else:
                allowed = False
                reason_code = "trusted_intent_missing"

        if allowed:
            self._record_event(
                state=state,
                tool_name=tool_name,
                capability=capability,
                outcome="allow",
                reason_code=reason_code,
                classifier_status=classifier_status,
            )
            return None

        outcome = "would_block" if self.settings.mode == "monitor" else "block"
        self._record_event(
            state=state,
            tool_name=tool_name,
            capability=capability,
            outcome=outcome,
            reason_code=reason_code,
            classifier_status=classifier_status,
        )
        if self.settings.mode == "monitor":
            return None

        source_text = ", ".join(sources) if sources else "unknown turn context"
        label = CAPABILITY_LABELS.get(capability, capability)
        return {
            "action": "block",
            "message": (
                f"CaMeL guard blocked {tool_name}: {label} was not authorized "
                f"by trusted user intent after untrusted data from {source_text}. "
                f"Policy result: {reason_code} ({classifier_status})."
            ),
        }

    def on_session_end(
        self,
        *,
        session_id: str = "",
        task_id: str = "",
        **_: Any,
    ) -> None:
        self._purge_scope(self._scope_id(session_id, task_id))


_runtime: CamelGuardRuntime | None = None


def register(ctx: Any) -> None:
    global _runtime
    _runtime = CamelGuardRuntime(ctx, _load_settings(ctx))
    ctx.register_hook("pre_llm_call", _runtime.on_pre_llm_call)
    ctx.register_hook("pre_tool_call", _runtime.on_pre_tool_call)
    ctx.register_hook("post_tool_call", _runtime.on_post_tool_call)
    ctx.register_hook("on_session_end", _runtime.on_session_end)
    ctx.register_hook("on_session_reset", _runtime.on_session_end)
