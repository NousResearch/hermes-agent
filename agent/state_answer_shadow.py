"""Side-effect-free Shadow adapter for State Answer Gate.

This module intentionally has no Runtime, database, model, or delivery imports.
It observes explicit provider metadata without enforcing a model-call decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import threading
import uuid
from collections import deque
from typing import Any, Callable, Protocol

from .state_answer_gate import StateAnswerGateResult, evaluate_answer_gate
from .state_candidate_evaluator import StateCandidateResult

_MAX_EVENT_BYTES = 8192
_MAX_CANDIDATES = 128
_MAX_REQUESTED_KEYS = 128
_MAX_CANDIDATE_KEY_PAIRS = 32
_MAX_FIELD_CHARS = 256


class InputStatus(str, Enum):
    READY = "ready"
    INPUT_UNAVAILABLE = "input_unavailable"
    DISABLED = "disabled"
    PROVIDER_ERROR = "provider_error"
    MALFORMED_INPUT = "malformed_input"


class ShadowDecision(str, Enum):
    EVALUATED = "evaluated"
    NOT_EVALUATED = "not_evaluated"


class TerminalStatus(str, Enum):
    UNOBSERVED = "unobserved"
    PREFLIGHT_RETURN = "preflight_return"
    RETRY_EXHAUSTED = "retry_exhausted"
    PROVIDER_ERROR = "provider_error"
    INTERRUPTED = "interrupted"
    EXCEPTION = "exception"
    COMPLETED = "completed"


class ReceiptStatus(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNOBSERVED = "unobserved"


class ShadowCollector(Protocol):
    def enqueue_initial(self, event: dict[str, Any]) -> None: ...

    def enqueue_terminal(self, event_id: str, update: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class ShadowInput:
    status: InputStatus
    run_id: str = ""
    turn_id: str = ""
    active_scope: str = ""
    answer_scope: str = ""
    current_state: Any = None
    evidence: Any = None
    current_state_validated: bool = False
    evidence_validated: bool = False
    candidates: tuple[StateCandidateResult, ...] = ()
    requested_state_keys: tuple[str, ...] = ()


class StateAnswerShadowInputProvider(Protocol):
    def get_input(self) -> ShadowInput: ...


def _opaque_id() -> str:
    return uuid.uuid4().hex


def _is_opaque_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(
        char in "0123456789abcdef" for char in value
    )


def _opaque_ref(value: str) -> str:
    del value
    return uuid.uuid4().hex[:16]


_ISSUED_EVENT_IDS: dict[str, str] = {}
_PENDING_TERMINALS: dict[str, tuple[ShadowCollector, dict[str, Any]]] = {}
_ISSUED_EVENT_ORDER: deque[str] = deque()
_ISSUED_EVENT_IDS_LOCK = threading.Lock()
_COLLECTOR_SLOTS = threading.BoundedSemaphore(32)
_COLLECTOR_FAILURES = 0
_CORRELATION_EXPIRIES = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enum_value(value: Enum | str, allowed: set[str]) -> str | None:
    result = value.value if isinstance(value, Enum) else value
    return result if isinstance(result, str) and result in allowed else None


def _safe_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Allowlist and size-check event metadata before collector access."""
    allowed = {
        "schema_version", "shadow_event_id", "origin", "mode", "input_status",
        "shadow_decision", "gate_decision", "model_call_status", "terminal_status",
        "persistence_receipt", "persistence_receipt_status", "reason_code", "candidate_count",
        "run_id", "turn_id", "created_at", "candidate_outcomes",
        "active_scope_status", "current_state_status", "evidence_status", "candidate_status",
    }
    if set(event) - allowed:
        return None
    if any(not _is_opaque_id(event.get(name)) for name in ("shadow_event_id", "run_id", "turn_id")):
        return None
    if event.get("origin") != "normal_turn" or event.get("mode") != "shadow":
        return None
    try:
        encoded = json.dumps(event, ensure_ascii=True, separators=(",", ":")).encode()
    except (TypeError, ValueError):
        return None
    if len(encoded) > _MAX_EVENT_BYTES:
        return None
    return dict(event)


def collector_failure_count() -> int:
    return _COLLECTOR_FAILURES


def correlation_expiry_count() -> int:
    return _CORRELATION_EXPIRIES


def _dispatch_nonblocking(
    fn: Callable[..., None], *args: Any, on_failure: Callable[[], None] | None = None,
    on_success: Callable[[], None] | None = None,
) -> bool:
    global _COLLECTOR_FAILURES
    if not _COLLECTOR_SLOTS.acquire(blocking=False):
        _COLLECTOR_FAILURES += 1
        return False

    def invoke() -> None:
        global _COLLECTOR_FAILURES
        try:
            fn(*args)
            if on_success is not None:
                on_success()
        except Exception:
            _COLLECTOR_FAILURES += 1
            if on_failure is not None:
                try:
                    on_failure()
                except Exception:
                    _COLLECTOR_FAILURES += 1
        finally:
            _COLLECTOR_SLOTS.release()

    try:
        threading.Thread(target=invoke, name="hermes-shadow-collector", daemon=True).start()
    except Exception:
        _COLLECTOR_FAILURES += 1
        _COLLECTOR_SLOTS.release()
        return False
    return True


def _base_event(event_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "shadow_event_id": event_id,
        "run_id": _opaque_id(),
        "turn_id": _opaque_id(),
        "created_at": _now(),
        "origin": "normal_turn",
        "mode": "shadow",
        "model_call_status": "unobserved",
        "terminal_status": TerminalStatus.UNOBSERVED.value,
        "persistence_receipt_status": ReceiptStatus.UNOBSERVED.value,
        "active_scope_status": "unavailable",
        "current_state_status": "unavailable",
        "evidence_status": "unavailable",
        "candidate_status": "unavailable",
    }


def observe_state_answer_shadow(
    provider: StateAnswerShadowInputProvider,
    collector: ShadowCollector,
    *,
    gate_evaluator: Callable[..., StateAnswerGateResult] = evaluate_answer_gate,
) -> str:
    """Evaluate explicit inputs in Shadow mode and emit sanitized metadata.

    The provider is the only input owner. Non-ready inputs are not coerced into
    the existing fail-closed gate contract; they are recorded as not evaluated.
    Collector failures are swallowed by design and never affect the caller.
    """
    global _CORRELATION_EXPIRIES
    event_id = _opaque_id()
    with _ISSUED_EVENT_IDS_LOCK:
        _ISSUED_EVENT_IDS[event_id] = "initial_pending"
        _ISSUED_EVENT_ORDER.append(event_id)
        while len(_ISSUED_EVENT_ORDER) > 4096:
            expired = _ISSUED_EVENT_ORDER.popleft()
            if _ISSUED_EVENT_IDS.pop(expired, None) is not None:
                _CORRELATION_EXPIRIES += 1
            _PENDING_TERMINALS.pop(expired, None)
    event = _base_event(event_id)
    try:
        supplied = provider.get_input()
    except Exception:
        supplied = ShadowInput(InputStatus.PROVIDER_ERROR)
    if not isinstance(supplied, ShadowInput):
        supplied = ShadowInput(InputStatus.PROVIDER_ERROR)
    if _is_opaque_id(supplied.run_id):
        event["run_id"] = supplied.run_id
    if _is_opaque_id(supplied.turn_id):
        event["turn_id"] = supplied.turn_id
    status = _enum_value(supplied.status, {item.value for item in InputStatus})
    if status is None:
        status = InputStatus.PROVIDER_ERROR.value
    event["input_status"] = status
    if supplied.current_state is not None and supplied.current_state_validated is True:
        event["current_state_status"] = "available"
    if supplied.evidence is not None and supplied.evidence_validated is True:
        event["evidence_status"] = "available"
    if not isinstance(supplied.requested_state_keys, (tuple, list)):
        status = InputStatus.MALFORMED_INPUT.value
        event["input_status"] = status
    if not isinstance(supplied.candidates, (tuple, list)) or len(supplied.candidates) > _MAX_CANDIDATES:
        candidates = []
        status = InputStatus.MALFORMED_INPUT.value
        event["input_status"] = status
    else:
        candidates = list(supplied.candidates)
    event["candidate_count"] = len(candidates)

    if status == InputStatus.READY.value:
        try:
            if any(
                not isinstance(candidate, StateCandidateResult)
                or not isinstance(candidate.scope, str)
                or not isinstance(candidate.state_key, str)
                or not isinstance(candidate.candidate_id, str)
                or not candidate.scope.strip()
                or not candidate.state_key.strip()
                or not candidate.candidate_id.strip()
                or len(candidate.scope) > _MAX_FIELD_CHARS
                or len(candidate.state_key) > _MAX_FIELD_CHARS
                or len(candidate.candidate_id) > _MAX_FIELD_CHARS
                for candidate in candidates
            ):
                raise ValueError("malformed candidate")
            if (
                not isinstance(supplied.active_scope, str)
                or not supplied.active_scope.strip()
                or len(supplied.active_scope) > _MAX_FIELD_CHARS
                or not isinstance(supplied.answer_scope, str)
                or not supplied.answer_scope.strip()
                or len(supplied.answer_scope) > _MAX_FIELD_CHARS
                or supplied.active_scope != supplied.answer_scope
                or any(
                    not isinstance(candidate.scope, str)
                    or candidate.scope != supplied.active_scope
                    for candidate in candidates
                )
            ):
                raise ValueError("scope mismatch")
            if not isinstance(supplied.requested_state_keys, (tuple, list)):
                raise ValueError("malformed requested keys")
            requested_keys = list(supplied.requested_state_keys)
            if len(requested_keys) > _MAX_REQUESTED_KEYS:
                raise ValueError("malformed requested keys")
            if any(not isinstance(key, str) or not key.strip() for key in requested_keys):
                raise ValueError("malformed requested keys")
            if any(len(key) > _MAX_FIELD_CHARS for key in requested_keys):
                raise ValueError("malformed requested keys")
            if len(candidates) * len(requested_keys) > _MAX_CANDIDATE_KEY_PAIRS:
                raise ValueError("malformed candidate projection")
            gates = []
            outcomes = []
            event["active_scope_status"] = "available"
            event["candidate_status"] = "available"
            for candidate in candidates:
                gate = gate_evaluator(
                    candidate, requested_state_keys=requested_keys, answer_scope=supplied.answer_scope
                )
                gates.append(gate)
                policy = _enum_value(gate.decision.response_policy, {
                    "continue", "ask_confirmation", "hold", "conflict_stop",
                })
                if policy is None:
                    raise ValueError("invalid gate decision")
                outcomes.append({
                    "candidate_ref": _opaque_ref(candidate.candidate_id),
                    "key_outcomes": [
                        {
                            "key_ref": _opaque_ref(key),
                            "decision": policy if key == candidate.state_key else "continue",
                            "reason_code": "evaluated" if key == candidate.state_key else "not_dependent",
                        }
                        for key in requested_keys
                    ],
                })
            dependent_policies = [
                _enum_value(gate.decision.response_policy, {
                    "continue", "ask_confirmation", "hold", "conflict_stop",
                })
                for candidate, gate in zip(candidates, gates)
                if candidate.state_key in requested_keys
            ]
            if any(policy is None for policy in dependent_policies):
                raise ValueError("invalid gate decision")
            valid_policies = [policy for policy in dependent_policies if policy is not None]
            precedence = {"continue": 0, "ask_confirmation": 1, "hold": 2, "conflict_stop": 3}
            event["shadow_decision"] = ShadowDecision.EVALUATED.value
            event["gate_decision"] = max(valid_policies, key=lambda item: precedence[item]) if valid_policies else "continue"
            event["candidate_outcomes"] = outcomes
            event["reason_code"] = "evaluated"
        except ValueError as exc:
            event["shadow_decision"] = ShadowDecision.NOT_EVALUATED.value
            event["gate_decision"] = "unobserved"
            event["reason_code"] = (
                "scope_mismatch" if str(exc) == "scope mismatch"
                else "malformed_input" if str(exc) in {
                    "malformed requested keys", "malformed candidate", "malformed candidate projection"
                }
                else "evaluation_error"
            )
        except Exception:
            event["shadow_decision"] = ShadowDecision.NOT_EVALUATED.value
            event["gate_decision"] = "unobserved"
            event["reason_code"] = "evaluation_error"
    else:
        event["shadow_decision"] = ShadowDecision.NOT_EVALUATED.value
        event["gate_decision"] = "unobserved"
        event["reason_code"] = status

    safe = _safe_event(event)
    if safe is not None:
        def initial_failed() -> None:
            with _ISSUED_EVENT_IDS_LOCK:
                _ISSUED_EVENT_IDS.pop(event_id, None)
                _PENDING_TERMINALS.pop(event_id, None)

        def initial_succeeded() -> None:
            with _ISSUED_EVENT_IDS_LOCK:
                if event_id not in _ISSUED_EVENT_IDS:
                    return
                pending = _PENDING_TERMINALS.pop(event_id, None)
                _ISSUED_EVENT_IDS[event_id] = "terminal_pending" if pending is not None else "initial_accepted"
            if pending is not None:
                pending_collector, pending_update = pending
                def pending_failed() -> None:
                    with _ISSUED_EVENT_IDS_LOCK:
                        if _ISSUED_EVENT_IDS.get(event_id) == "terminal_pending":
                            _ISSUED_EVENT_IDS[event_id] = "initial_accepted"
                def pending_succeeded() -> None:
                    with _ISSUED_EVENT_IDS_LOCK:
                        _ISSUED_EVENT_IDS.pop(event_id, None)
                if not _dispatch_nonblocking(
                    pending_collector.enqueue_terminal, event_id, pending_update,
                    on_failure=pending_failed, on_success=pending_succeeded,
                ):
                    pending_failed()

        if not _dispatch_nonblocking(
            collector.enqueue_initial, safe,
            on_failure=initial_failed, on_success=initial_succeeded,
        ):
            with _ISSUED_EVENT_IDS_LOCK:
                _ISSUED_EVENT_IDS.pop(event_id, None)
    else:
        with _ISSUED_EVENT_IDS_LOCK:
            _ISSUED_EVENT_IDS.pop(event_id, None)
    return event_id


def observe_state_answer_shadow_terminal(
    collector: ShadowCollector,
    event_id: str,
    *,
    terminal_status: TerminalStatus,
    model_call_status: str,
    persistence_confirmed: bool | None,
) -> None:
    """Emit a sanitized terminal/receipt update; never invent a receipt."""
    if not _is_opaque_id(event_id):
        return
    terminal = _enum_value(terminal_status, {
        item.value for item in TerminalStatus if item is not TerminalStatus.UNOBSERVED
    })
    model = _enum_value(model_call_status, {"observed", "not_called", "failed", "unobserved"})
    if terminal is None or model is None:
        return
    receipt = (
        ReceiptStatus.TRUE.value
        if persistence_confirmed is True and terminal == TerminalStatus.COMPLETED.value
        and model == "observed"
        else ReceiptStatus.FALSE.value if persistence_confirmed is False
        else ReceiptStatus.UNOBSERVED.value
    )
    update = {
        "terminal_status": terminal,
        "model_call_status": model,
        "persistence_receipt_status": receipt,
    }
    with _ISSUED_EVENT_IDS_LOCK:
        lifecycle = _ISSUED_EVENT_IDS.get(event_id)
        if lifecycle == "initial_pending":
            if event_id in _PENDING_TERMINALS:
                return
            _PENDING_TERMINALS[event_id] = (collector, update)
            return
        if lifecycle != "initial_accepted":
            return
        _ISSUED_EVENT_IDS[event_id] = "terminal_pending"
    def terminal_failed() -> None:
        with _ISSUED_EVENT_IDS_LOCK:
            if _ISSUED_EVENT_IDS.get(event_id) == "terminal_pending":
                _ISSUED_EVENT_IDS[event_id] = "initial_accepted"

    def terminal_succeeded() -> None:
        with _ISSUED_EVENT_IDS_LOCK:
            _ISSUED_EVENT_IDS.pop(event_id, None)

    if not _dispatch_nonblocking(
        collector.enqueue_terminal, event_id, update,
        on_failure=terminal_failed, on_success=terminal_succeeded,
    ):
        terminal_failed()
