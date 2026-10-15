"""Small, loss-bounded incident state for repair-oriented context hygiene.

The transcript remains available to explicit session-search/recovery tools, but the active
repair context is reduced to six fields.  This module deliberately uses deterministic
extraction: a failed repair must not require another model call to decide what history to
keep, and raw tool output must not be copied into the next prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional


DEFAULT_SUMMARY_TOKEN_CAP = 1500
INCIDENT_STATE_MESSAGE_KEY = "_incident_state"
INCIDENT_STATE_MARKER = "[CONTEXT COMPACTION — INCIDENT STATE]"
INCIDENT_RESET_ACTION = (
    "Hard reset after two failed repair attempts: re-read authoritative files/config/logs "
    "from current machine state before changing anything; continue from first principles."
)

FIELD_NAMES = (
    "original_failure_requested_outcome",
    "current_root_cause_hypothesis",
    "supporting_evidence",
    "files_config_changed",
    "latest_test_result",
    "next_action_current_observed_state",
)
FIELD_LABELS = (
    "Original failure/requested outcome",
    "Current root-cause hypothesis",
    "Supporting evidence",
    "Files/config changed",
    "Latest test result",
    "Next action/current observed state",
)
_DEFAULT_FIELD = "Not recorded."
_MESSAGE_SUMMARY_LIMIT = 240
_SECRET_RE = re.compile(
    r"(?i)\b(api[_ -]?key|token|password|secret|authorization|cookie)\b\s*[:=]\s*[^\s,;]+"
)
_PATH_RE = re.compile(
    r"(?<![\w-])(?:~|/|\.{0,2}/)?(?:[\w.-]+/)*[\w.-]+\.(?:py|ya?ml|json|md|toml|ini|sh|js|ts|tsx|jsx|plist|env|service|cfg)"
)
_HYPOTHESIS_RE = re.compile(r"\b(root cause|hypothesis|caused by|because|due to)\b", re.I)
_EVIDENCE_RE = re.compile(r"\b(error|failed|failure|observed|evidence|exit\s*code|blocked|timeout|reproduced)\b", re.I)
_TEST_RE = re.compile(r"\b(test|pytest|unittest|validation|validated|passed|pass|failed|failure|exit\s*code|check)\b", re.I)
_NEXT_RE = re.compile(r"\b(next action|next step|current state|observed state|before changing)\b", re.I)


def _clip_text(value: Any, limit: int = 700) -> str:
    text = str(value or "").replace("\x00", " ")
    text = re.sub(r"```[^\n]*", "", text)
    text = _SECRET_RE.sub(r"\1=[REDACTED]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    keep = max(40, (limit - 24) // 2)
    return f"{text[:keep]} …[truncated]… {text[-keep:]}"


def _message_content(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    if content is None:
        return ""
    return str(content)


def _compact_tool_content(content: Any) -> str:
    text = _message_content({"content": content})
    try:
        payload = json.loads(text) if isinstance(text, str) else None
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        pieces = []
        for key in (
            "error", "exit_code", "success", "status", "path", "file", "bytes_written", "changed", "message",
        ):
            if key in payload and payload[key] not in (None, ""):
                pieces.append(f"{key}={payload[key]}")
        if pieces:
            text = "; ".join(pieces)
        else:
            text = ""
    elif text:
        # Keep one relevant line from non-JSON output; full stdout stays retrievable
        # evidence instead of becoming part of the next active prompt.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = next((line for line in lines if _EVIDENCE_RE.search(line)), "")
    return _clip_text(text, 240)


def _message_summary(message: Any) -> str:
    if not isinstance(message, dict):
        return _clip_text(message)
    if message.get("role") == "tool":
        name = str(message.get("name") or "tool")
        result = _compact_tool_content(message.get("content"))
        return f"{name}: {result}" if result else name
    return _clip_text(_message_content(message), _MESSAGE_SUMMARY_LIMIT)


def _focused_summary(text: str, pattern: re.Pattern[str]) -> str:
    """Keep only one relevant line from narrative text for a state field."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    match = next((line for line in reversed(lines) if pattern.search(line)), "")
    return _clip_text(match, _MESSAGE_SUMMARY_LIMIT)


def _is_incident_message(message: Any) -> bool:
    return isinstance(message, dict) and (
        bool(message.get(INCIDENT_STATE_MESSAGE_KEY))
        or INCIDENT_STATE_MARKER in _message_content(message)
    )


def _meaningful(value: Any) -> bool:
    return bool(str(value or "").strip()) and str(value).strip() != _DEFAULT_FIELD


@dataclass(frozen=True)
class IncidentState:
    """The only fields allowed in the active incident checkpoint."""

    original_failure_requested_outcome: str = _DEFAULT_FIELD
    current_root_cause_hypothesis: str = _DEFAULT_FIELD
    supporting_evidence: str = _DEFAULT_FIELD
    files_config_changed: str = _DEFAULT_FIELD
    latest_test_result: str = _DEFAULT_FIELD
    next_action_current_observed_state: str = _DEFAULT_FIELD

    @classmethod
    def from_mapping(cls, value: Any) -> Optional["IncidentState"]:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return None
        return cls(**{
            name: _clip_text(value.get(name, _DEFAULT_FIELD), 1400) or _DEFAULT_FIELD
            for name in FIELD_NAMES
        })

    def as_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name) or _DEFAULT_FIELD) for name in FIELD_NAMES}

    def render(self, token_cap: int = DEFAULT_SUMMARY_TOKEN_CAP) -> str:
        try:
            cap = int(token_cap)
        except (TypeError, ValueError):
            cap = DEFAULT_SUMMARY_TOKEN_CAP
        cap = max(1000, min(2000, cap))
        limit = cap * 4
        values = [_clip_text(getattr(self, name), 1400) or _DEFAULT_FIELD for name in FIELD_NAMES]
        fixed = len(INCIDENT_STATE_MARKER) + sum(len(label) + 4 for label in FIELD_LABELS) + 20
        budget = max(6 * 64, limit - fixed)
        weights = (1.0, 1.0, 1.2, 1.0, 1.0, 1.2)
        total_weight = sum(weights)
        values = [
            _clip_text(value, max(64, int(budget * weight / total_weight))) or _DEFAULT_FIELD
            for value, weight in zip(values, weights)
        ]
        rendered = "\n".join(
            [INCIDENT_STATE_MARKER]
            + [f"{label}: {value}" for label, value in zip(FIELD_LABELS, values)]
        )
        return rendered[:limit]


def parse_incident_state(text: Any) -> Optional[IncidentState]:
    body = _message_content({"content": text})
    if INCIDENT_STATE_MARKER not in body:
        return None
    values: dict[str, str] = {}
    for line in body.splitlines():
        for name, label in zip(FIELD_NAMES, FIELD_LABELS):
            prefix = f"{label}:"
            if line.startswith(prefix):
                values[name] = _clip_text(line[len(prefix):].strip(), 1400) or _DEFAULT_FIELD
                break
    return IncidentState.from_mapping(values) if values else None


def extract_incident_state(messages: Iterable[Any]) -> Optional[IncidentState]:
    for message in reversed(list(messages or [])):
        if not _is_incident_message(message):
            continue
        parsed = parse_incident_state(_message_content(message))
        if parsed:
            return parsed
    return None


def _tool_call_args(message: Any) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    result = []
    for call in calls or []:
        function = getattr(call, "function", None)
        if function is not None:
            raw = function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", None)
        elif isinstance(call, dict):
            function = call.get("function")
            raw = function.get("arguments") if isinstance(function, dict) else call.get("arguments")
        else:
            raw = None
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            args = None
        if isinstance(args, dict):
            result.append(args)
    return result


def _paths_from_messages(messages: Iterable[Any]) -> str:
    paths: list[str] = []
    for message in messages:
        text = _message_content(message)
        for args in _tool_call_args(message):
            text += " " + " ".join(str(args.get(key, "")) for key in ("path", "file", "file_path", "config"))
        for path in _PATH_RE.findall(text):
            if path not in paths:
                paths.append(path)
    return ", ".join(paths[-8:]) or _DEFAULT_FIELD


def build_incident_state(
    messages: Iterable[Any], previous: Any = None, *, hard_reset: bool = False
) -> IncidentState:
    rows = [message for message in (messages or []) if not _is_incident_message(message)]
    # Runtime state is authoritative. Do not resurrect a checkpoint from an older
    # incident left in the durable transcript after a successful turn.
    prior = IncidentState.from_mapping(previous)
    summaries = [(message.get("role") if isinstance(message, dict) else "", _message_summary(message)) for message in rows]
    summaries = [(role, text) for role, text in summaries if text]
    user_texts = [text for role, text in summaries if role == "user"]

    original = prior.original_failure_requested_outcome if prior else _DEFAULT_FIELD
    if not _meaningful(original) and user_texts:
        # With no active checkpoint, the latest real user turn is the new incident.
        original = user_texts[-1]

    hypothesis = prior.current_root_cause_hypothesis if prior else _DEFAULT_FIELD
    for _role, text in reversed(summaries):
        if _HYPOTHESIS_RE.search(text):
            hypothesis = _focused_summary(text, _HYPOTHESIS_RE)
            break

    evidence_items: list[str] = []
    for role, text in reversed(summaries):
        candidate = text if role == "tool" else _focused_summary(text, _EVIDENCE_RE)
        if candidate:
            if candidate not in evidence_items:
                evidence_items.append(candidate)
        if len(evidence_items) == 3:
            break
    evidence = "; ".join(reversed(evidence_items)) if evidence_items else (prior.supporting_evidence if prior else _DEFAULT_FIELD)

    test_result = prior.latest_test_result if prior else _DEFAULT_FIELD
    for _role, text in reversed(summaries):
        candidate = _focused_summary(text, _TEST_RE)
        if candidate:
            test_result = candidate
            break

    next_action = prior.next_action_current_observed_state if prior else _DEFAULT_FIELD
    if prior and not hard_reset and prior.next_action_current_observed_state == INCIDENT_RESET_ACTION:
        # The reset directive is consumed by the first fresh-machine action; do
        # not carry it forward as the next action for a later failed mutation.
        next_action = _DEFAULT_FIELD
    for _role, text in reversed(summaries):
        candidate = _focused_summary(text, _NEXT_RE)
        if candidate:
            next_action = candidate
            break
    if not _meaningful(next_action) and summaries:
        role, text = summaries[-1]
        next_action = f"Current observed state: {text}" if role == "tool" else "Current state requires an authoritative reread."
    if hard_reset:
        next_action = INCIDENT_RESET_ACTION

    return IncidentState(
        original_failure_requested_outcome=_clip_text(original) or _DEFAULT_FIELD,
        current_root_cause_hypothesis=_clip_text(hypothesis) or _DEFAULT_FIELD,
        supporting_evidence=_clip_text(evidence, 1100) or _DEFAULT_FIELD,
        files_config_changed=_paths_from_messages(rows),
        latest_test_result=_clip_text(test_result) or _DEFAULT_FIELD,
        next_action_current_observed_state=_clip_text(next_action) or _DEFAULT_FIELD,
    )


def incident_state_message(state: Any, *, token_cap: int = DEFAULT_SUMMARY_TOKEN_CAP, hard_reset: bool = False) -> dict[str, Any]:
    normalized = IncidentState.from_mapping(state) or IncidentState()
    if hard_reset:
        normalized = replace(normalized, next_action_current_observed_state=INCIDENT_RESET_ACTION)
    return {
        "role": "user",
        "content": normalized.render(token_cap),
        "_compressed_summary": True,
        "_compressed_summary_has_user_turn": True,
        INCIDENT_STATE_MESSAGE_KEY: True,
    }


def _tool_call_name(call: Any) -> str:
    function = getattr(call, "function", None)
    if function is not None:
        return str(getattr(function, "name", "") or "")
    if isinstance(call, dict):
        function = call.get("function")
        return str((function or {}).get("name") or call.get("name") or "") if isinstance(function, dict) else str(call.get("name") or "")
    return ""


def _tool_call_id(call: Any) -> str:
    return str(getattr(call, "id", "") or (call.get("id", "") if isinstance(call, dict) else ""))


def _repair_mutation(call: Any) -> bool:
    name = _tool_call_name(call)
    if name in {"write_file", "patch"}:
        return True
    if name != "terminal":
        return False
    args = _tool_call_args({"tool_calls": [call]})
    command = str((args[0] if args else {}).get("command") or (args[0] if args else {}).get("cmd") or "")
    try:
        from agent.tool_dispatch_helpers import _is_destructive_command
        return _is_destructive_command(command)
    except Exception:
        return bool(re.search(r"\b(rm|mv|cp|install|truncate|dd|shred|sed\s+-i|git\s+(reset|clean|checkout))\b|>", command))


def _call_failed(call: Any, messages: list[Any], result_start: int) -> bool:
    name = _tool_call_name(call)
    call_id = _tool_call_id(call)
    candidates = []
    for message in messages[result_start:]:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if name and message.get("name") not in (None, name):
            continue
        if call_id and message.get("tool_call_id") not in (None, call_id):
            continue
        candidates.append(message)
    if not candidates:
        return False
    try:
        from agent.display import _detect_tool_failure
        return any(_detect_tool_failure(name, _message_content(message))[0] for message in candidates)
    except Exception:
        return any("error" in _message_content(message).lower() or "failed" in _message_content(message).lower() for message in candidates)


@dataclass(frozen=True)
class IncidentUpdate:
    state: Optional[IncidentState]
    mutation_attempt: bool = False
    failed: bool = False
    failed_attempts: int = 0
    compact: bool = False
    hard_reset: bool = False


def update_after_tool_round(agent: Any, assistant_message: Any, messages: list[Any], result_start: int) -> IncidentUpdate:
    """Record a repair mutation and decide whether the transcript needs an incident boundary."""
    if not getattr(agent, "incident_context_enabled", True):
        return IncidentUpdate(None)
    calls = list(getattr(assistant_message, "tool_calls", None) or [])
    mutation_calls = [call for call in calls if _repair_mutation(call)]
    previous = IncidentState.from_mapping(getattr(agent, "_incident_state", None))
    recovered_from_history = False
    if previous is None:
        # Resume an unresolved incident after a process restart, but only when the
        # latest real user turn is still the persisted checkpoint. A newer user
        # turn marks a genuinely fresh incident and must not inherit old state.
        for message in reversed(messages or []):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                if _is_incident_message(message):
                    previous = extract_incident_state(messages)
                    recovered_from_history = previous is not None
                break
    if not mutation_calls and previous is None:
        return IncidentUpdate(None)

    mutation_attempt = bool(mutation_calls)
    failed = mutation_attempt and any(_call_failed(call, messages, result_start) for call in mutation_calls)
    failed_attempts = int(getattr(agent, "_incident_failed_repair_attempts", 0) or 0)
    if recovered_from_history and failed_attempts == 0 and previous:
        # A persisted ordinary checkpoint proves one failed mutation happened
        # before a process restart. A reset checkpoint starts a new attempt zero.
        if previous.next_action_current_observed_state != INCIDENT_RESET_ACTION:
            failed_attempts = 1
    if failed:
        failed_attempts += 1
    if previous is None and not failed:
        return IncidentUpdate(None, mutation_attempt=mutation_attempt)

    reset_after = max(1, int(getattr(agent, "incident_reset_after_failed_repairs", 2) or 2))
    hard_reset = failed and failed_attempts >= reset_after
    state = build_incident_state(messages, previous, hard_reset=hard_reset)
    agent._incident_state = state.as_dict()
    agent._incident_failed_repair_attempts = failed_attempts
    return IncidentUpdate(
        state=state,
        mutation_attempt=mutation_attempt,
        failed=failed,
        failed_attempts=failed_attempts,
        # Preserve newly retrieved read/test output for the next model turn. A
        # checkpoint is a boundary after a failed mutation, not after every read.
        compact=failed,
        hard_reset=hard_reset,
    )


__all__ = [
    "DEFAULT_SUMMARY_TOKEN_CAP", "FIELD_NAMES", "FIELD_LABELS", "INCIDENT_RESET_ACTION",
    "INCIDENT_STATE_MARKER", "INCIDENT_STATE_MESSAGE_KEY", "IncidentState", "IncidentUpdate",
    "build_incident_state", "extract_incident_state", "incident_state_message", "parse_incident_state",
    "update_after_tool_round",
]
