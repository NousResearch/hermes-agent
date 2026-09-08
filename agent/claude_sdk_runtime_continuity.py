"""Session continuity for the claude-agent-sdk runtime: the persisted SDK resume
id and the digest that primes a fresh SDK session from the Hermes transcript.
Extracted from ``claude_sdk_runtime.py``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agent.claude_sdk_runtime_state import _SdkTurnState

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


_SDK_RESUME_BINDING_PREFIX = "hermes-sdk-resume-v1:"


def _canonical_sdk_cwd(value: Optional[str] = None) -> str:
    """Return the canonical CWD identity used to bind SDK resume IDs."""
    if value is None:
        from agent.runtime_cwd import resolve_agent_cwd

        value = str(resolve_agent_cwd())
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(str(value)))))


def _encode_sdk_resume_binding(session_id: str, *, cwd: Optional[str] = None) -> str:
    payload = {
        "cwd": _canonical_sdk_cwd(cwd),
        "id": str(session_id),
    }
    return _SDK_RESUME_BINDING_PREFIX + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def _persisted_sdk_session_id(agent) -> Optional[str]:
    """The SDK session id stored on the Hermes session row (or None)."""
    if getattr(agent, "_persist_disabled", False):
        return None
    if not (getattr(agent, "_session_db", None) and getattr(agent, "session_id", None)):
        return None
    try:
        row = agent._session_db.get_session(agent.session_id) or {}
        raw = row.get("claude_sdk_session_id") or None
        if not isinstance(raw, str) or not raw:
            return None
        current_cwd = _canonical_sdk_cwd()
        if raw.startswith(_SDK_RESUME_BINDING_PREFIX):
            try:
                payload = json.loads(raw[len(_SDK_RESUME_BINDING_PREFIX) :])
                session_id = payload["id"]
                bound_cwd = payload["cwd"]
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("empty SDK session id")
                if not isinstance(bound_cwd, str) or not bound_cwd:
                    raise ValueError("empty SDK resume cwd")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                _store_sdk_session_id(agent, None)
                return None
        else:
            # Legacy raw IDs and unknown envelope versions have no trustworthy
            # creation-workspace provenance.  The session row's CWD is mutable,
            # so it cannot retroactively authorize a cross-workspace resume.
            _store_sdk_session_id(agent, None)
            return None
        if _canonical_sdk_cwd(bound_cwd) != current_cwd:
            logger.info(
                "claude-agent-sdk: declining resume id bound to a different cwd"
            )
            return None
        return session_id
    except Exception:
        logger.debug("resume-id read failed", exc_info=True)
        return None


def _store_sdk_session_id(
    agent, value: Optional[str], *, cwd: Optional[str] = None
) -> None:
    """Persist (or clear, with None) the SDK session id on the session row."""
    if getattr(agent, "_persist_disabled", False):
        # A review/curator fork shares the parent's session_id — it must
        # never write its own resume id onto the parent's row.
        return
    if not (getattr(agent, "_session_db", None) and getattr(agent, "session_id", None)):
        return
    try:
        if value is not None and cwd is None:
            # The workspace this id belongs to was never sampled. Encoding it
            # here would canonicalise whatever cwd happens to be live at
            # persist time, which then matches on every subsequent turn and
            # makes the check a silent no-op. Write nothing rather than a
            # guess. Deliberately not a clear: callers that mean "clear" pass
            # None explicitly, and the read path already re-validates a stored
            # binding's cwd on every turn, so an older binding for this same
            # workspace is exactly what should still be resumable.
            logger.debug("resume-id not written: turn workspace unknown")
            return
        stored = (
            _encode_sdk_resume_binding(value, cwd=cwd) if value is not None else None
        )
        agent._session_db.update_claude_sdk_session_id(agent.session_id, stored)
    except Exception:
        logger.debug("resume-id write failed", exc_info=True)


_CONTINUITY_DIGEST_MAX_CHARS = 4000


def _render_continuity_digest(prior_messages: List[Dict[str, Any]]) -> str:
    """Bounded text preamble for a FRESH SDK session that has prior Hermes
    history (resume impossible: no stored id, or the stored one went stale).
    Reuses _digest_history's compaction, then flattens to capped text."""
    # Projected background results are the agent's OWN answers, already
    # delivered outbound; re-presenting them here is the double-presentation
    # pathology the background lane exists to kill. Filter before the
    # compaction pass — _digest_history may rebuild dicts and drop the mark.
    prior_messages = [
        m for m in (prior_messages or [])
        if not (
            isinstance(m, dict)
            and m.get("display_kind") == "sdk_background_result"
        )
    ]
    try:
        from agent.background_review import _digest_history

        msgs = _digest_history(list(prior_messages or []), tail=8)
    except Exception:  # pragma: no cover - compaction is best-effort
        msgs = list(prior_messages or [])[-8:]
    lines: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        text = str(content).replace("\n", " ").strip()
        if text:
            lines.append(f"{role.upper()}: {text[:400]}")
    if not lines:
        return ""
    body = "\n".join(lines)
    if len(body) > _CONTINUITY_DIGEST_MAX_CHARS:
        body = body[-_CONTINUITY_DIGEST_MAX_CHARS:]
    return (
        "[Continuity digest — the runtime restarted and the live model "
        "context was lost; recent turns from the stored transcript, oldest "
        "first:]\n" + body + "\n[End digest. The user's new message follows.]\n\n"
    )


def _persist_turn(agent, state: _SdkTurnState) -> None:
    """Flush the projected rows FIRST, then persist the SDK resume id."""
    turn = state.turn
    messages = state.messages
    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        # Early-return path bypasses conversation_loop's per-step persistence;
        # flush the new projected rows ourselves (idempotent via the intrinsic
        # _DB_PERSISTED_MARKER — the user turn was flushed at turn start).
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "claude-sdk projected-message flush failed", exc_info=True
                )

    if not getattr(turn, "should_retire", False) and state.failover_reason is None:
        # Persist the SDK session id for restart/eviction/interrupt resume.
        # AFTER the flush on purpose: the flush's _ensure_db_session retry is
        # what (re)creates the session row when turn-start persistence hit a
        # transient lock — storing first would silently discard the id.
        thread_id = getattr(turn, "thread_id", None)
        if thread_id:
            _store_sdk_session_id(agent, thread_id, cwd=state.turn_session_cwd)
