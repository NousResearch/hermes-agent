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
    r"^(?:please\s+)?(?:(?:proceed|continue)\s+with\s+|go ahead and\s+)?"
    r"(?:implement(?:ing)?|fix(?:ing)?|chang(?:e|ing)|updat(?:e|ing)|"
    r"edit(?:ing)?|build(?:ing)?|add(?:ing)?|remov(?:e|ing))\b"
)
_RESUME_REQUEST_RE = re.compile(r"\b(?:proceed|continue|go ahead)[.!\s]*$")
_REPORTED_REQUEST_RE = re.compile(r"\b(?:said|says|told|reported|quote|quoted|example)\b")
_TASK_COMPLETION_RE = re.compile(r"\b(?:done|finished|complete(?:d)?|no (?:changes|work) remain)\b")
_CONTRASTIVE_ANSWER_RE = re.compile(r"^no,\s+[^.!?;]+,\s+not\s+[^.!?;]+[.!]?$")
_CLARIFIED_ACTION_RE = re.compile(
    _ACK_LEAD + r"(?:implement|change|update|edit|add|remove|show|hide|keep)\b"
)
_DECLINED_RE = re.compile(
    r"(?:^no\b|\b(?:cancel(?:led|ed)?|abort|stop|wait|declin(?:e|ed)|deny|skip|hold off|not yet|"
    r"approval|permission|confirmation|credentials?|password|nothing|not now|not to|"
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
        if not answers:
            return False
        for answer in answers:
            if not isinstance(answer, str) or not answer.strip():
                return False
            text = answer.strip().lower()
            # A contrastive answer corrects the question's premise; a bare No
            # still declines. Explicit stop/wait language wins in either form.
            if _CONTRASTIVE_ANSWER_RE.fullmatch(text):
                text = text[3:].lstrip()
            if _DECLINED_RE.search(text):
                return False
    return True


def _unquoted_prose(text: str) -> str:
    return re.sub(
        r'''`[^`]*`|"[^"]*"|“[^”]*”|(?<!\w)'[^\n]*?'(?!\w)|‘[^\n]*?’(?!\w)|(?m:^>.*$)''',
        "", text,
    )


def _clarification_task_request(user_text: str, messages: List[Dict[str, Any]]) -> str:
    """Resolve a direct request or its adjacent, still-unstarted continuation.

    This selects context for a bounded re-query, not permission to execute. Never
    mine tool output or assistant prose for authority, or jump over performed work
    to resurrect a stale request. A quoted/reported 'proceed' is not a continuation.
    """
    from agent.conversation_compression import _is_real_user_message, _message_text

    text = _unquoted_prose(user_text).strip()
    if "?" in text or _DECLINED_RE.search(text) or _REPORTED_REQUEST_RE.search(text):
        return ""
    if _EDIT_REQUEST_RE.search(text):
        return text
    if not _RESUME_REQUEST_RE.search(text):
        return ""
    latest = next((i for i in range(len(messages) - 1, -1, -1)
                   if isinstance(messages[i], dict) and messages[i].get("role") == "user"), 0)
    for row in reversed(messages[:latest]):
        if not isinstance(row, dict):
            continue
        if row.get("role") == "tool" or row.get("tool_calls"):
            return ""
        if row.get("role") == "assistant" and _TASK_COMPLETION_RE.search(
            _unquoted_prose(_message_text(row).lower())
        ):
            return ""
        if row.get("role") == "user":
            if not _is_real_user_message(row):
                return ""
            request = _unquoted_prose(_message_text(row).lower()).strip()
            return request if (
                _EDIT_REQUEST_RE.search(request) and "?" not in request
                and not _DECLINED_RE.search(request) and not _REPORTED_REQUEST_RE.search(request)
            ) else ""
    return ""


def has_current_turn_clarification(messages: List[Dict[str, Any]]) -> bool:
    """Route even denied clarification through the same visible-ack safety gate."""
    for row in reversed(messages):
        if not isinstance(row, dict):
            continue
        if row.get("role") == "user":
            break
        if row.get("role") == "tool" and row.get("name") == "clarify":
            return True
        if any(call.get("function", {}).get("name") == "clarify"
               for call in row.get("tool_calls") or [] if isinstance(call, dict)):
            return True
    return False


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
    if _DECLINED_RE.search(user_text):
        return False
    if clarifications:
        user_text = _clarification_task_request(user_text, messages)
        if not user_text:
            return False
    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text or len(assistant_text) > 1200:
        return False
    if "?" in assistant_text or _ACK_TERMINAL_RE.search(assistant_text):
        return False
    # Quoted examples and code are content, not the assistant's own commitment.
    prose = _unquoted_prose(assistant_text)
    from agent.agent_runtime_helpers import trailing_continue_intent

    if not (_ACK_ANNOUNCE_RE.search(prose) or (clarifications and (
        _CLARIFIED_ACTION_RE.search(prose) or trailing_continue_intent(prose)
    ))):
        return False
    if not require_workspace:
        return True

    return (
        bool(_ACK_WORKSPACE_RE.search(user_text))
        or "/" in user_text
        or bool(_ACK_WORKSPACE_RE.search(prose))
    )
