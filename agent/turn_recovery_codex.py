"""Codex Responses recovery rungs: stale encrypted-reasoning strip and the replay verdict reset."""
from __future__ import annotations

import logging
from typing import Any

from agent.turn_retry_state import TurnRetryState

logger = logging.getLogger(__name__)


def _is_codex_token_expired(agent: Any, api_error: Exception) -> bool:
    """401 ``token_expired`` from the Codex backend (#88510). It rejects a stale replayed
    ``encrypted_content`` blob with this auth signature, so a persisted session loops on "sign
    in again" while a fresh session on the same bearer works. The caller treats it like
    ``invalid_encrypted_content`` — but only while cached reasoning items remain to strip."""
    if getattr(api_error, "status_code", None) != 401:
        return False
    reason = agent._extract_api_error_context(api_error).get("reason")
    return isinstance(reason, str) and reason.strip().lower() == "token_expired"


def reset_codex_reasoning_replay(agent: Any) -> None:
    """The replay verdict belongs to the route that earned it: a ``/model`` switch, fallback
    activation or primary restore starts the new route with replay on (#61552)."""
    agent._codex_reasoning_replay_enabled = True
    agent._codex_reasoning_replay_rejected = False


def _recover_stale_codex_reasoning(
    agent: Any, _retry: TurnRetryState, messages: list[dict[str, Any]], api_messages: Any,
) -> bool:
    """Stale ``codex_reasoning_items`` blob rejected by the provider: strip cached items (mutates
    persisted ``messages``) and retry once. The first rung drops only the replayed blobs not stamped by
    exactly this endpoint+model (it consumes its own precondition, so it cannot loop); a repeat goes on
    to strip every item. The first full strip keeps replay on, since blobs the
    route mints from now on are sealed with its current key; a repeat rejection means the route
    cannot round-trip its own blobs, so replay is disabled for the session."""
    if (
        _retry.invalid_encrypted_content_retry_attempted
        or agent.api_mode != "codex_responses"
        or not bool(getattr(agent, "_codex_reasoning_replay_enabled", True))
        or not any(
            isinstance(_m, dict)
            and _m.get("role") == "assistant"
            and isinstance(_m.get("codex_reasoning_items"), list)
            and _m.get("codex_reasoning_items")
            for _m in messages
        )
    ):
        return False
    from agent.turn_recovery import _vlines  # facade owns the verbose printer; late import breaks the cycle

    transport = agent._get_transport()
    if transport is not None and (removed := transport.drop_unverified_replay(messages, api_messages)):
        _vlines(agent, f"⚠️  Encrypted reasoning replay was rejected — dropped {removed} item(s) minted elsewhere, retrying...")
        return True
    _retry.invalid_encrypted_content_retry_attempted = True
    keep_replay = not getattr(agent, "_codex_reasoning_replay_rejected", False)
    agent._codex_reasoning_replay_rejected = True
    replay_stats = agent._disable_codex_reasoning_replay(messages, keep_replay=keep_replay)
    # The retry is rebuilt from the request copy; with replay kept on it would resend the stale blob.
    for _m in api_messages if isinstance(api_messages, list) else []:
        if isinstance(_m, dict):
            _m.pop("codex_reasoning_items", None)
    action = "stripped stale" if keep_replay else "disabled replay for this session and stripped"
    _vlines(
        agent,
        f"⚠️  Encrypted reasoning replay was rejected by the provider — "
        f"{action} {replay_stats['items']} item(s) from {replay_stats['messages']} message(s), retrying...",
    )
    logger.warning(
        "%sInvalid encrypted reasoning recovery: %s %d items from %d messages",
        agent.log_prefix, action, replay_stats["items"], replay_stats["messages"],
    )
    return True
