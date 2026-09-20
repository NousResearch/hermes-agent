"""Intent-acknowledgment detection for unstarted tool work."""

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
    r"wait|waiting|awaiting|blocked|unable|cannot|can['’]t|won['’]t|"
    r"never|do not|don['’]t|cancel(?:led|ed)?|stop|hold off|"
    r"done|finished|complete(?:d)?|already|passed|next week|later|"
    r"background|still running|in flight)\b"
)
_ACK_WORKSPACE_RE = re.compile(
    r"\b(?:director(?:y|ies)|current dir|cwd|repos?|repository|codebase|"
    r"projects?|folders?|filesystem|file tree|files?|paths?)\b"
)


def looks_like_codex_intermediate_ack(
    agent, user_message: Any, assistant_content: str, messages: List[Dict[str, Any]],
    require_workspace: bool = True,
) -> bool:
    """Detect a short action announcement before execution in the current turn.

    Opt-in ``require_workspace=False`` relaxes only the workspace requirement.
    Prior turns cannot prove that the current request has been acted on (#69778).
    """
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            break
        if msg.get("role") == "tool" or msg.get("tool_calls"):
            return False
    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text or len(assistant_text) > 1200:
        return False
    if "?" in assistant_text or _ACK_TERMINAL_RE.search(assistant_text):
        return False
    # Quoted examples and code are content, not the assistant's own commitment.
    prose = re.sub(r'`[^`]*`|"[^"]*"|“[^”]*”|(?m:^>.*$)', "", assistant_text)
    if not _ACK_ANNOUNCE_RE.search(prose):
        return False
    if not require_workspace:
        return True
    # ``user_message`` may be a multi-part content list (vision via the OpenAI-compat server); a
    # list survives ``or ""`` and ``.strip()`` raises, so flatten first.
    from agent.codex_responses_adapter import _summarize_user_message_for_log
    user_text = _summarize_user_message_for_log(user_message).strip().lower()
    return (
        bool(_ACK_WORKSPACE_RE.search(user_text))
        or "/" in user_text
        or bool(_ACK_WORKSPACE_RE.search(prose))
    )
