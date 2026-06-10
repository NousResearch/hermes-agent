"""Fallback Context Bridge — context translation across provider switches.

When a mid-turn fallback activates and the api_mode changes (e.g.
``codex_responses`` → ``anthropic_messages`` / ``chat_completions``), the
already-built ``api_messages`` list contains provider-specific fields that
the new provider rejects or misinterprets:

  * ``codex_reasoning_items``   — Responses-API reasoning blobs; the new
    provider does not recognise this key and strict gateways 422 on it.
  * ``codex_message_items``     — Codex intermediate output items; same issue.
  * ``reasoning_content`` pads  — injected by the Codex path; may collide
    with the new provider's own echo-back convention.

``apply_fallback_context_bridge()`` is called by ``conversation_loop.py``
immediately after fallback activation is detected, before ``_build_api_kwargs``
runs.  It performs two jobs in one pass:

  1. **Context Bridge (Option 1)** — strip / neutralise provider-specific
     fields from every message in ``api_messages`` so the new provider sees
     clean, standard Chat-Completions or Anthropic-Messages format.

  2. **Skill Re-injection (Option 2)** — rebuild the system message at
     ``api_messages[0]`` with the freshly-built ``agent._cached_system_prompt``
     (which was invalidated by ``try_activate_fallback`` when the api_mode
     changed).  This ensures the new provider receives an up-to-date skill
     index rather than the stale Codex-shaped prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Fields that are specific to the OpenAI Responses / Codex API and must be
# stripped when switching to a provider that uses Chat-Completions or the
# Anthropic Messages API.
_CODEX_SPECIFIC_FIELDS = frozenset({
    "codex_reasoning_items",
    "codex_message_items",
})

# Internal Hermes bookkeeping keys that no external API should receive.
_HERMES_INTERNAL_FIELDS = frozenset({
    "tool_name",
    "finish_reason",
    "reasoning",          # kept internally for trajectories; copied to reasoning_content where needed
    "_thinking_prefill",
})


def _strip_codex_fields_from_messages(api_messages: List[Dict[str, Any]]) -> int:
    """Remove Codex Responses-API-specific fields from every message.

    Operates in-place on *api_messages*.  Returns the count of messages that
    were modified.
    """
    modified = 0
    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        changed = False
        for field in _CODEX_SPECIFIC_FIELDS:
            if field in msg:
                del msg[field]
                changed = True
        # Strip any remaining underscore-prefixed internal Hermes scaffolding
        # that the Codex path may have left on the message.
        internal_keys = [k for k in msg if isinstance(k, str) and k.startswith("_")]
        for k in internal_keys:
            del msg[k]
            changed = True
        if changed:
            modified += 1
    return modified


def _rebuild_system_message(agent: Any, api_messages: List[Dict[str, Any]]) -> bool:
    """Replace the system message in *api_messages* with a freshly-built prompt.

    Builds ``agent._cached_system_prompt`` (it was invalidated when the
    api_mode changed) and swaps the ``role: system`` entry at index 0.
    Returns True when the system message was replaced, False otherwise.
    """
    # Rebuild _cached_system_prompt for the new provider.
    try:
        fresh_prompt = agent._build_system_prompt()
        agent._cached_system_prompt = fresh_prompt
    except Exception as exc:
        logger.warning(
            "fallback_context_bridge: failed to rebuild system prompt: %s", exc
        )
        return False

    if not fresh_prompt:
        return False

    # Persist the updated prompt so subsequent turns in this session reuse it.
    try:
        if agent._session_db and agent.session_id:
            agent._session_db.update_system_prompt(agent.session_id, fresh_prompt)
    except Exception as exc:
        logger.warning(
            "fallback_context_bridge: could not persist rebuilt system prompt: %s", exc
        )

    new_sys = {"role": "system", "content": fresh_prompt}
    if api_messages and api_messages[0].get("role") == "system":
        api_messages[0] = new_sys
        return True

    # No system message present yet — prepend one.
    api_messages.insert(0, new_sys)
    return True


def apply_fallback_context_bridge(
    agent: Any,
    api_messages: List[Dict[str, Any]],
    old_api_mode: str,
    new_api_mode: str,
) -> None:
    """Translate *api_messages* after a cross-mode provider fallback.

    Called by ``conversation_loop.run_conversation`` immediately before
    ``_build_api_kwargs`` when ``agent._pending_context_bridge`` is set.

    Step 1 — Context Bridge: strip Codex-specific fields when leaving the
    ``codex_responses`` api_mode, so the new provider receives clean messages.

    Step 2 — Skill Re-injection: rebuild the system message so the new
    provider sees the current skill index (plus any skills created or updated
    during the Codex session).
    """
    leaving_codex = old_api_mode == "codex_responses"
    entering_codex = new_api_mode == "codex_responses"

    # ── Step 1: Context Bridge ─────────────────────────────────────────────
    if leaving_codex:
        stripped = _strip_codex_fields_from_messages(api_messages)
        if stripped:
            logger.info(
                "fallback_context_bridge: stripped Codex-specific fields from "
                "%d messages (codex_responses → %s)",
                stripped, new_api_mode,
            )
        # Disable reasoning replay — encrypted_content blobs minted by Codex
        # are sealed to that endpoint and will be rejected by the new provider.
        try:
            stats = agent._disable_codex_reasoning_replay(api_messages)
            if stats.get("messages"):
                logger.info(
                    "fallback_context_bridge: disabled codex reasoning replay "
                    "(%d messages, %d items)",
                    stats["messages"], stats.get("items", 0),
                )
        except Exception as exc:
            logger.warning(
                "fallback_context_bridge: _disable_codex_reasoning_replay failed: %s", exc
            )

    if entering_codex:
        # Entering Codex from another provider: strip anthropic-specific fields
        # (reasoning_content pads, cache_control markers) that Codex rejects.
        _strip_anthropic_fields_from_messages(api_messages)

    # ── Step 2: Skill Re-injection ─────────────────────────────────────────
    # _cached_system_prompt was set to None by try_activate_fallback() when the
    # api_mode changed.  Rebuild it now so the new provider receives a prompt
    # that reflects the current skill index (including skills created/updated
    # during the outgoing provider's session).
    rebuilt = _rebuild_system_message(agent, api_messages)
    if rebuilt:
        logger.info(
            "fallback_context_bridge: rebuilt system prompt for new provider "
            "(%s → %s), skill context re-injected",
            old_api_mode, new_api_mode,
        )
    else:
        logger.warning(
            "fallback_context_bridge: system prompt rebuild skipped or failed "
            "(%s → %s)",
            old_api_mode, new_api_mode,
        )


def _strip_anthropic_fields_from_messages(api_messages: List[Dict[str, Any]]) -> int:
    """Strip Anthropic-Messages-specific fields when switching TO Codex.

    ``cache_control`` markers and ``reasoning_content`` pads written by the
    Anthropic path confuse the Responses API.  Operates in-place.
    """
    modified = 0
    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        changed = False
        # cache_control is only valid for the Anthropic Messages API.
        if "cache_control" in msg:
            del msg["cache_control"]
            changed = True
        # Strip cache_control from content parts (Anthropic injects it there).
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "cache_control" in part:
                    del part["cache_control"]
                    changed = True
        # reasoning_content pads are Anthropic/DeepSeek-specific.
        if "reasoning_content" in msg:
            del msg["reasoning_content"]
            changed = True
        if changed:
            modified += 1
    return modified
