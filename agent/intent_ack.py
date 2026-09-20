"""Intent-acknowledgment detection for unstarted tool work."""

import json
import re
from typing import Any, Dict, List


# Governed, word-bounded actions adapted from ClaySecAI's #69779. Keep the
# current-turn execution gate in BOTH modes; retrying completed work is unsafe.
_ACK_ACTION_ALT = (
    r"look\s+(?:into|at)|inspect(?:ing)?|scan(?:ning)?|check(?:ing)?(?!\s+in\b)|"
    r"analy[sz](?:e|ing)|review(?:ing)?|explor(?:e|ing)|read(?:ing)?(?!\s+(?:up|you|to)\b)|"
    r"open(?:ing)?(?!\s+(?:with|to|by)\b)|run(?:ning)?(?!\s+(?:with|through)\b)|"
    r"test(?:ing)?|fix(?:ing)?|debug(?:ging)?|search(?:ing)?|find(?:ing)?|report\s+back|"
    r"summari[sz](?:e|ing)|deploy(?:ing)?|build(?:ing)?(?!\s+on\b)|verif(?:y|ying)"
)
_ACK_LEAD = (
    r"(?:^|[.!]\s+)(?:(?:sure|understood|okay|ok)[,!.—\s]+)?"
    r"(?:i['’]ll|i\s+will|i(?:['’]m|\s+am)\s+going\s+to|let\s+me)\s+"
    r"(?:(?:first|now)\s+)?(?:start\s+by\s+)?"
)
_ACK_ANNOUNCE_RE = re.compile(_ACK_LEAD + r"(?:" + _ACK_ACTION_ALT + r")\b")
_ACK_TERMINAL_RE = re.compile(
    r"\b(?:let me know|if you|would you|feel free|happy to help|"
    r"approval|permission|confirmation|credentials?|password|token|"
    r"(?:once|when|after|until) you|go.ahead|"
    r"wait|waiting|awaiting|blocked|unable|cannot|can['’]t|won['’]t|"
    r"never|do not|don['’]t|cancel(?:led|ed)?|stop|hold off|"
    r"done|finished|complete(?:d)?|already|passed|next week|later|"
    r"background|still running|in flight)\b"
)
_ACK_WORKSPACE_RE = re.compile(
    r"\b(?:director(?:y|ies)|current dir|cwd|repos?|repository|codebase|"
    r"projects?|folders?|filesystem|file tree|files?|paths?)\b"
)
_EDIT_REQUEST_RE = re.compile(
    r"^(?:please\s+)?(?:go ahead and\s+)?"
    r"(?:implement|fix|change|update|edit|build|add|remove)\b"
)
_CLARIFIED_ACTION_RE = re.compile(
    _ACK_LEAD + r"(?:implement|change|update|edit|add|remove|show|hide|keep)\b"
)
_DECLINED_RE = re.compile(
    r"(?:^no\b|\b(?:cancel(?:led|ed)?|abort|stop|wait|declin(?:e|ed)|deny|skip|hold off|not yet|"
    r"approval|permission|confirmation|credentials?|password|"
    r"do not|don['’]t|did not provide a response)\b)"
)


def _answered_clarification(content: Any) -> bool:
    try:
        result = json.loads(content)
    except (TypeError, ValueError):
        return False
    if not isinstance(result, dict) or result.get("timed_out") or result.get("error"):
        return False
    responses = result.get("responses", [result])
    if not isinstance(responses, list) or not responses:
        return False
    for response in responses:
        if not isinstance(response, dict):
            return False
        answer = response.get("user_response")
        answers = answer if isinstance(answer, list) else [answer]
        if not answers or any(
            not isinstance(a, str) or not a.strip() or _DECLINED_RE.search(a.strip().lower())
            for a in answers
        ):
            return False
    return True


def _clarification_only_turn(messages: List[Dict[str, Any]]):
    """Return answered clarification count, or None if execution may have begun.

    Pair calls with results rather than trusting a result's display name. Unknown
    tools, missing results, and partial/declined answers conservatively end recovery.
    """
    pending = set()
    count = 0
    start = next((i + 1 for i in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[i], dict) and messages[i].get("role") == "user"), 0)
    if start:
        from agent.conversation_compression import _is_real_user_message, _message_text

        latest = messages[start - 1]
        text = _message_text(latest).strip()
        # Other recovery/verification rounds are not new, unstarted human work.
        if (not _is_real_user_message(latest)
                or text.startswith(("[System:", "[OUT-OF-BAND USER MESSAGE"))
                or _DECLINED_RE.search(text.lower())):
            return None
    for msg in messages[start:]:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict) or call.get("function", {}).get("name") != "clarify":
                return None
            call_id = call.get("id")
            if not call_id or call_id in pending:
                return None
            pending.add(call_id)
        if msg.get("role") == "tool":
            call_id = msg.get("tool_call_id")
            if call_id not in pending or not _answered_clarification(msg.get("content")):
                return None
            pending.remove(call_id)
            count += 1
    return None if pending else count


def has_live_ack_work(agent) -> bool:
    """An earlier turn may still own background work; do not duplicate it."""
    from tools.async_delegation import has_live_for_session
    from tools.process_registry import process_registry

    if has_live_for_session(parent_session_id=getattr(agent, "session_id", "") or ""):
        return True
    owners = getattr(agent, "_process_owner_task_ids", ())
    return bool(owners) and any(
        row["owner_task_id"] in owners and row["status"] == "running"
        for row in process_registry.list_sessions()
    )


def looks_like_codex_intermediate_ack(
    agent, user_message: Any, assistant_content: str, messages: List[Dict[str, Any]],
    require_workspace: bool = True,
) -> bool:
    """Detect a short action announcement before execution in the current turn.

    Opt-in ``require_workspace=False`` relaxes only the workspace requirement.
    Prior turns cannot prove that the current request has been acted on (#69778).
    """
    clarifications = _clarification_only_turn(messages)
    if clarifications is None:
        return False
    from agent.codex_responses_adapter import _summarize_user_message_for_log
    user_text = _summarize_user_message_for_log(user_message).strip().lower()
    # Clarification refines an existing execution request; it is not approval.
    if _DECLINED_RE.search(user_text) or (clarifications and not _EDIT_REQUEST_RE.search(user_text)):
        return False
    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text or len(assistant_text) > 1200:
        return False
    if "?" in assistant_text or _ACK_TERMINAL_RE.search(assistant_text):
        return False
    # Quoted examples and code are content, not the assistant's own commitment.
    prose = re.sub(
        r'''`[^`]*`|"[^"]*"|“[^”]*”|(?<!\w)'[^\n]*?'(?!\w)|‘[^\n]*?’(?!\w)|(?m:^>.*$)''',
        "", assistant_text,
    )
    if not (_ACK_ANNOUNCE_RE.search(prose) or (clarifications and _CLARIFIED_ACTION_RE.search(prose))):
        return False
    if not require_workspace:
        return True

    return (
        bool(_ACK_WORKSPACE_RE.search(user_text))
        or "/" in user_text
        or bool(_ACK_WORKSPACE_RE.search(prose))
    )
