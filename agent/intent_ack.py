"""Intent-acknowledgment detection for unstarted tool work."""

import re
from typing import Any, Dict, List


_ACK_FUTURE_RE = re.compile(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b")
_ACK_ACTION_MARKERS = (
    "look into", "look at", "inspect", "scan", "check", "analyz", "review", "explore", "read", "open",
    "run", "test", "fix", "debug", "search", "find", "walkthrough", "report back", "summarize",
)
_ACK_WORKSPACE_MARKERS = (
    "directory", "current directory", "current dir", "cwd", "repo", "repository", "codebase",
    "project", "folder", "filesystem", "file tree", "files", "path",
)


def looks_like_codex_intermediate_ack(
    agent, user_message: Any, assistant_content: str, messages: List[Dict[str, Any]],
    require_workspace: bool = True,
) -> bool:
    """Detect a planning/ack message that should continue instead of ending the turn.
    ``require_workspace=False`` (opt-in for all api_modes) drops the filesystem/repo reference
    requirement; future-ack + short-content + no-prior-tools + action-verb checks always apply."""
    if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
        return False
    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text or len(assistant_text) > 1200:
        return False
    if not _ACK_FUTURE_RE.search(assistant_text):
        return False
    if not any(marker in assistant_text for marker in _ACK_ACTION_MARKERS):
        return False
    # Opted-in (all-api_mode) path: future-ack + action verb + no prior tool call suffices.
    if not require_workspace:
        return True
    # ``user_message`` may be a multi-part content list (vision via the OpenAI-compat server); a
    # list survives ``or ""`` and ``.strip()`` raises, so flatten first.
    from agent.codex_responses_adapter import _summarize_user_message_for_log
    user_text = _summarize_user_message_for_log(user_message).strip().lower()
    return (
        any(marker in user_text for marker in _ACK_WORKSPACE_MARKERS)
        or "~/" in user_text
        or "/" in user_text
        or any(marker in assistant_text for marker in _ACK_WORKSPACE_MARKERS)
    )
