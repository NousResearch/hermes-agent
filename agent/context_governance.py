"""Optional plugin admission gates for recalled context and compression commits.

No hook means the existing Hermes behavior. Once a hook is registered, every
callback must explicitly allow; malformed, failed, or timed-out policy hooks
deny. The raw context stays in-process and is never written to diagnostics.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)
POLICY_HOOKS = frozenset({"pre_tool_call", "pre_memory_context", "pre_compression_commit"})


def policy_hook_required(event: str) -> bool:
    """Read the active profile's behavioral requirement, never a process env flag."""
    from hermes_cli.config import load_config_readonly
    plugins = load_config_readonly().get("plugins", {})
    value = plugins.get("required_policy_hooks", []) if isinstance(plugins, dict) else None
    if not isinstance(value, list) or any(not isinstance(item, str) or item not in POLICY_HOOKS for item in value):
        raise ValueError("invalid plugins.required_policy_hooks")
    return event in value


def _active(event: str) -> bool | None:
    """False = no gate, None = discovery failed (fail closed)."""
    try:
        from hermes_cli.plugins import has_hook
        required = policy_hook_required(event)
        active = has_hook(event)
        if required and not active:
            logger.warning("context governance event=%s decision=block reason=required_hook_missing", event)
            return None
        return active
    except Exception:
        logger.warning("context governance event=%s decision=block reason=config_or_discovery_error", event)
        return None


def _allowed(event: str, **payload: Any) -> bool:
    try:
        from hermes_cli.lifecycle import invoke_hook
        results = invoke_hook(event, **payload)
        allowed = bool(results) and all(
            isinstance(result, dict) and result.get("action") == "allow"
            for result in results
        )
    except Exception:
        allowed = False
    logger.info("context governance event=%s decision=%s", event, "allow" if allowed else "block")
    return allowed


def _terms(value: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", value)}


def _content(messages: list) -> str:
    parts = []
    for row in messages[:200]:
        if isinstance(row, dict) and isinstance(row.get("content"), str):
            parts.append(row["content"][:10000])
    return "\n".join(parts)


def memory_context_allowed(*, provider: str, query: str, context: str, session_id: str) -> bool:
    """Gate one provider's text before it enters a turn's API context."""
    active = _active("pre_memory_context")
    if active is not True:
        return active is False
    try:
        qterms, cterms = _terms(query), _terms(context)
        state = {"provider_kind": "builtin" if provider == "builtin" else "external",
                 "query_chars": min(len(query), 100000), "context_chars": min(len(context), 100000),
                 "query_terms": len(qterms), "context_terms": len(cterms),
                 "shared_terms": len(qterms & cterms)}
    except Exception:
        logger.warning("context governance event=pre_memory_context decision=block reason=invalid_input")
        return False
    return _allowed("pre_memory_context", state=state, session_id=session_id)


def compression_commit_allowed(*, original: list, candidate: list, session_id: str) -> bool:
    """Gate a valid local compression candidate before durable session mutation."""
    active = _active("pre_compression_commit")
    if active is not True:
        return active is False
    try:
        before, after = _content(original), _content(candidate)
        before_terms, after_terms = _terms(before), _terms(after)
        state = {"original_messages": len(original), "candidate_messages": len(candidate),
                 "original_chars": min(len(before), 1000000), "candidate_chars": min(len(after), 1000000),
                 "original_terms": len(before_terms), "retained_terms": len(before_terms & after_terms)}
    except Exception:
        logger.warning("context governance event=pre_compression_commit decision=block reason=invalid_input")
        return False
    return _allowed("pre_compression_commit", state=state, session_id=session_id)
