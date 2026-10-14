"""Classify tool outcomes and require bounded follow-through in the current turn."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional



_OUTCOME_HOUSEKEEPING_TOOLS = frozenset({"memory", "todo_list", "skill_manage", "session_search"})

_OUTCOME_PENDING_STATUSES = frozenset({"pending", "retryable_pending", "awaiting_receipt", "awaiting_reconciliation"})

_OUTCOME_UNKNOWN_STATUSES = frozenset({"unknown", "outcome_unknown", "unknown_outcome", "reconcile_required"})

_OUTCOME_CONFLICT_STATUSES = frozenset({
    "conflict", "identity_conflict", "provenance_conflict", "cas_conflict", "permission_denied", "unauthorized",
})

_OUTCOME_TERMINAL_STATUSES = frozenset({"delivered", "processed", "completed", "archived", "merged", "resolved"})

_OUTCOME_SAFETY_RE = re.compile(
    r"\b(?:permission denied|not authorized|unauthorized|forbidden|approval required|user approval|"
    r"identity (?:mismatch|conflict)|provenance (?:mismatch|conflict)|cas conflict|source conflict|"
    r"duplicate (?:paid )?(?:call|write|operation))\b",
    re.I,
)

_ACTIONABLE_NEXT_STEP_RE = re.compile(
    r"\b(?:next step|next action|should now|need(?:s)? to|will need to|i recommend|we should|i should)\b",
    re.I,
)

_REQUESTED_EXECUTION_RE = re.compile(
    r"\b(?:implement|fix|repair|change|update|run|verify|deploy|publish|complete|finish|execute|"
    r"create|build|write|modify|inspect|diagnose|resolve|deliver|transcribe|import|delete|send|archive|merge)\b",
    re.I,
)

_USER_ONLY_RE = re.compile(
    r"\b(?:need(?:s)? your approval|user approval required|requires? (?:your )?approval|"
    r"waiting for (?:your )?approval|requires? your confirmation|waiting for your confirmation|"
    r"must be approved)\b",
    re.I,
)

_SECRET_RE = re.compile(
    r"(?i)\b(api[_ -]?key|token|password|secret|authorization|cookie)\b\s*[:=]\s*[^\s,;]+"
)

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

@dataclass(frozen=True)
class OutcomeBlocker:
    """Typed blocker reconstructed from the current turn's real tool results."""

    kind: str
    tool_name: str
    reason: str
    evidence: str
    attempts: int
    repeated_without_progress: bool
    has_followup_action: bool
    has_successful_followup: bool

def _tool_rounds(messages: Iterable[Any], start: int = 0) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for index, message in enumerate(list(messages or [])[start:], start):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and message.get("tool_calls"):
            if current:
                rounds.append(current)
            current = {"index": index, "calls": list(message.get("tool_calls") or []), "results": []}
        elif message.get("role") == "tool" and current is not None:
            current["results"].append(message)
        elif message.get("role") in {"assistant", "user"} and current is not None:
            rounds.append(current)
            current = None
    if current:
        rounds.append(current)
    return rounds

def _tool_call_details(call: Any) -> tuple[str, str, dict[str, Any]]:
    name = _tool_call_name(call)
    call_id = _tool_call_id(call)
    args = _tool_call_args({"tool_calls": [call]})
    return name, call_id, args[0] if args else {}

def _outcome_json(content: Any) -> Any:
    text = _message_content({"content": content})
    try:
        return json.loads(text) if isinstance(text, str) else None
    except (TypeError, ValueError):
        return None

def _classify_outcome(tool_name: str, result: Any, effect_disposition: Any = None) -> Optional[tuple[str, str, str]]:
    text = _message_content({"content": result})
    data = _outcome_json(text)
    status = str(data.get("status") or "").strip().lower() if isinstance(data, dict) else ""
    reason = ""
    evidence = ""
    if isinstance(data, dict):
        reason = str(data.get("reason") or data.get("error") or data.get("message") or "")
        evidence = str(
            data.get("evidence") or data.get("receipt") or data.get("operation_id")
            or data.get("readback") or text
        )
        if tool_name == "terminal" and isinstance(data.get("exit_code"), int):
            # Terminal stdout is arbitrary target data: a successful GitHub readback can
            # contain words like "provenance conflict" or "failed" without failing the command.
            if status in _OUTCOME_CONFLICT_STATUSES or _OUTCOME_SAFETY_RE.search(reason):
                kind = "safety_conflict"
            elif status in _OUTCOME_PENDING_STATUSES:
                kind = "pending"
            elif status in _OUTCOME_UNKNOWN_STATUSES:
                kind = "unknown"
            elif data["exit_code"] == 0 and not data.get("error"):
                return None
            elif status == "blocked" or str(effect_disposition or "").lower() == "none":
                # Tool-level rejection is known to happen before the command runs.
                kind = "recoverable"
            else:
                # A shell command may fail after an external write; require readback before retry.
                kind = "unknown"
            return (
                kind,
                _clip_text(reason or f"terminal exited with code {data['exit_code']}", 360),
                _clip_text(str(data.get("output") or evidence), 640),
            )
    else:
        reason = text
        evidence = text
    if status in _OUTCOME_CONFLICT_STATUSES or _OUTCOME_SAFETY_RE.search(reason or text):
        kind = "safety_conflict"
    elif status in _OUTCOME_PENDING_STATUSES:
        kind = "pending"
    elif status in _OUTCOME_UNKNOWN_STATUSES or str(effect_disposition or "").lower() == "unknown":
        kind = "unknown"
    else:
        try:
            from agent.display import _detect_tool_failure
            failed = _detect_tool_failure(tool_name, text)[0]
        except Exception:
            failed = bool(re.search(r"\b(error|failed|failure|timed out|exit code [1-9])\b", text, re.I))
        if not failed:
            return None
        kind = "recoverable"
    return (
        kind,
        _clip_text(reason or f"{tool_name} returned {status or 'an error'}", 360),
        _clip_text(evidence or text, 640),
    )

def _is_terminal_readback(result: Any) -> bool:
    data = _outcome_json(_message_content({"content": result}))
    if not isinstance(data, dict):
        return False
    status = str(data.get("status") or "").strip().lower()
    receipt = any(data.get(key) is True for key in ("receipt_verified", "target_readback", "readback_verified"))
    return status in _OUTCOME_TERMINAL_STATUSES and receipt

def _has_followthrough_action(messages: list[Any], after_index: int) -> bool:
    """Whether the model performed a real tool action after the last nudge.

    A guardrail/scope-blocked call is not an action. A real failed diagnostic is
    still an action because it supplies new evidence for the next recovery choice.
    """
    for tool_round in _tool_rounds(messages, after_index):
        calls = [_tool_call_details(call) for call in tool_round["calls"]]
        for result in tool_round["results"]:
            name = str(result.get("name") or result.get("tool_name") or "")
            call_id = str(result.get("tool_call_id") or "")
            call = next((item for item in calls if call_id and item[1] == call_id), None)
            if call is None:
                call = next((item for item in calls if item[0] == name), (name, call_id, {}))
            if call[0] in _OUTCOME_HOUSEKEEPING_TOOLS or result.get("effect_disposition") == "none":
                continue
            data = _outcome_json(result.get("content"))
            if isinstance(data, dict) and data.get("guardrail"):
                continue
            return True
    return False

def _current_turn_start(messages: Iterable[Any]) -> int:
    synthetic = (
        "_verification_stop_synthetic", "_pre_verify_synthetic", "_kanban_stop_synthetic",
        "_dropped_toolcall_nudge",
    )
    start = 0
    for index, message in enumerate(messages or []):
        if (
            isinstance(message, dict) and message.get("role") == "user"
            and not any(message.get(flag) for flag in synthetic)
        ):
            start = index
    return start



def latest_outcome_blocker(messages: Iterable[Any]) -> Optional[OutcomeBlocker]:
    """Return the latest typed failure in this user turn, including repeated exact blockers."""
    rows = list(messages or [])
    rounds = _tool_rounds(rows, _current_turn_start(rows))
    failures: list[dict[str, Any]] = []
    successful_rounds: set[int] = set()
    for round_index, tool_round in enumerate(rounds):
        calls = [_tool_call_details(call) for call in tool_round["calls"]]
        for result in tool_round["results"]:
            tool_name = str(result.get("name") or result.get("tool_name") or "")
            call_id = str(result.get("tool_call_id") or "")
            call = next((item for item in calls if call_id and item[1] == call_id), None)
            if call is None:
                call = next((item for item in calls if item[0] == tool_name), (tool_name, call_id, {}))
            if call[0] in _OUTCOME_HOUSEKEEPING_TOOLS:
                continue
            classification = _classify_outcome(
                tool_name or call[0], result.get("content"), result.get("effect_disposition")
            )
            if classification is None:
                if call[0] not in _OUTCOME_HOUSEKEEPING_TOOLS:
                    successful_rounds.add(round_index)
                continue
            kind, reason, evidence = classification
            action_sig = hashlib.sha256(json.dumps(
                [call[0], call[2]], sort_keys=True, default=str, separators=(",", ":")
            ).encode("utf-8", "replace")).hexdigest()
            blocker_sig = hashlib.sha256(
                f"{kind}\0{reason}\0{evidence}".encode("utf-8", "replace")
            ).hexdigest()
            result_sig = hashlib.sha256(_message_content({"content": result.get("content")}).encode(
                "utf-8", "replace"
            )).hexdigest()
            failures.append({
                "round": round_index, "kind": kind, "tool": tool_name or call[0],
                "reason": reason, "evidence": evidence, "action_sig": action_sig,
                "blocker_sig": blocker_sig, "result_sig": result_sig,
            })
    if not failures:
        return None
    latest = failures[-1]
    latest_terminal_round = max((
        round_index
        for round_index, tool_round in enumerate(rounds)
        for result in tool_round["results"]
        if _is_terminal_readback(result.get("content"))
    ), default=-1)
    if latest["kind"] in {"pending", "unknown"} and latest_terminal_round > latest["round"]:
        return None
    prior_same = [item for item in failures[:-1] if item["blocker_sig"] == latest["blocker_sig"]]
    consecutive_exact = 0
    last_round: Optional[int] = None
    for item in reversed(failures):
        if (
            item["blocker_sig"] != latest["blocker_sig"]
            or item["action_sig"] != latest["action_sig"]
            or item["result_sig"] != latest["result_sig"]
            or (last_round is not None and item["round"] < last_round - 1)
        ):
            break
        consecutive_exact += 1
        last_round = item["round"]
    later_action = any(
        tool_round["index"] > rounds[latest["round"]]["index"]
        and any(_tool_call_details(call)[0] not in _OUTCOME_HOUSEKEEPING_TOOLS for call in tool_round["calls"])
        for tool_round in rounds[latest["round"] + 1:]
    )
    later_success = any(round_index > latest["round"] for round_index in successful_rounds)
    return OutcomeBlocker(
        kind=latest["kind"], tool_name=latest["tool"], reason=latest["reason"],
        evidence=latest["evidence"], attempts=1 + len(prior_same),
        repeated_without_progress=consecutive_exact > 1,
        has_followup_action=later_action,
        has_successful_followup=later_success,
    )

def outcome_followthrough_nudge(agent: Any, final_response: Any, messages: Iterable[Any]) -> Optional[str]:
    """Keep recoverable blockers in the same run; unresolved boundaries fail the run, never succeed."""
    rows = list(messages or [])
    response = str(final_response or "")
    announced_next_action = bool(_ACTIONABLE_NEXT_STEP_RE.search(response))
    turn_start = _current_turn_start(rows)
    user_text = " ".join(
        _message_content(message)
        for message in rows[turn_start:]
        if isinstance(message, dict) and message.get("role") == "user"
        and not any(message.get(flag) for flag in (
            "_verification_stop_synthetic", "_pre_verify_synthetic", "_kanban_stop_synthetic",
            "_dropped_toolcall_nudge",
        ))
    )
    blocker = latest_outcome_blocker(rows)
    if _USER_ONLY_RE.search(response) and (
        blocker is not None or _REQUESTED_EXECUTION_RE.search(user_text)
    ):
        agent._outcome_ownership_failure = f"user_only_decision: {_clip_text(response, 360)}"
        return None
    if getattr(agent, "_outcome_followthrough_pending", False):
        try:
            followthrough_start = int(getattr(agent, "_outcome_followthrough_start_index", len(rows)))
        except (TypeError, ValueError):
            followthrough_start = len(rows)
        if not _has_followthrough_action(rows, followthrough_start):
            summary = blocker.reason if blocker is not None else _clip_text(response, 240)
            agent._outcome_ownership_failure = f"followthrough_action_not_taken: {_clip_text(summary, 360)}"
            return None
        agent._outcome_followthrough_pending = False
        agent._outcome_followthrough_start_index = None
    if blocker is not None and blocker.kind == "safety_conflict":
        agent._outcome_ownership_failure = f"{blocker.kind}: {blocker.reason}"
        return None

    if blocker is None:
        # Recommendation-only stops also occur before a tool has returned an
        # error. Close that gap for execution requests while leaving ordinary
        # explanatory answers alone.
        if not (announced_next_action and _REQUESTED_EXECUTION_RE.search(user_text)):
            return None
        instruction = (
            "Your response names an unfinished next action for the user's requested task. If it is bounded, "
            "available, and authorized, perform it now and verify the requested outcome in this run. If a true "
            "user-only decision or safety boundary blocks it, stop and report that exact boundary without claiming success."
        )
        evidence = _clip_text(response, 360)
        blocker_kind, blocker_tool, blocker_reason, blocker_attempts = "recommended_action", "assistant", "", 0
        repeated = False
    else:
        needs_followthrough = (
            not blocker.has_followup_action
            or not blocker.has_successful_followup
            or blocker.kind in {"pending", "unknown"}
            or blocker.repeated_without_progress
            or announced_next_action
        )
        if not needs_followthrough:
            return None

        blocker_kind, blocker_tool = blocker.kind, blocker.tool_name
        blocker_reason, evidence = blocker.reason, blocker.evidence
        blocker_attempts = blocker.attempts
        repeated = blocker.repeated_without_progress

    attempts = int(getattr(agent, "_outcome_recovery_nudges", 0) or 0)
    # Reuse the existing per-turn model iteration budget. A fixed repair-attempt
    # threshold would turn an ordinary fixable test/code error into escalation.
    try:
        followthrough_budget = max(1, int(getattr(agent, "max_iterations", 1) or 1))
    except (TypeError, ValueError):
        followthrough_budget = 1
    if attempts >= followthrough_budget or not getattr(agent, "valid_tool_names", None):
        agent._outcome_ownership_failure = f"{blocker_kind}: {blocker_reason or evidence}"
        return None
    agent._outcome_recovery_nudges = attempts + 1
    agent._outcome_followthrough_pending = True
    agent._outcome_followthrough_start_index = len(rows)
    if blocker is None:
        pass
    elif blocker.kind == "pending":
        instruction = (
            "The last tool returned a typed pending state. Preserve its reason/evidence and attempt history. "
            "Use an existing read/reconciliation action to check the same operation's current state; do not "
            "turn pending into unknown or repeat the side effect."
        )
    elif blocker.kind == "unknown":
        instruction = (
            "The last tool has an unknown outcome. Reconcile the existing receipt or target state with a "
            "read-only action before any retry; never repeat an uncertain paid or external side effect."
        )
    else:
        instruction = (
            "The last tool failed. Inspect its exact evidence, diagnose the cause, then perform an already-"
            "authorized bounded recovery or alternate diagnostic action before reporting."
        )
    if repeated:
        instruction += " The same blocker and exact action repeated without progress; choose a different authorized path."
    if announced_next_action:
        instruction += " You named an actionable next step; execute it now if it is bounded and already authorized."
    return (
        f"{instruction}\n"
        f"Typed state: {blocker_kind}; attempt {blocker_attempts}; tool {blocker_tool}.\n"
        f"Reason: {blocker_reason or 'No tool error has been reported.'}\nEvidence: {evidence}\n"
        "Do not claim success from a report, plan, PR, test, status comment, or intended write. "
        "If only user approval, a true identity/provenance/CAS conflict, or a safety boundary can unblock this, "
        "stop without bypassing it and report the exact unresolved state."
    )
