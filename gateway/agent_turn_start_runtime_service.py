"""Production turn-start / session bootstrap for gateway foreground runs.

Promoted from ``GatewayRunner._handle_message_with_agent`` so there is a single
bootstrap path.  Callers must pass a live runner; this module does not reimplement
session store / pin / lease semantics.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from gateway.direct_shortcut_runtime_service import (
    try_handle_direct_gateway_shortcuts,
)
from gateway.empty_response_fallback import (
    explicit_group_reply_context_note as _explicit_group_reply_context_note,
)
from gateway.session import build_session_context


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _load_gateway_config() -> dict:
    """Load the same config dict production uses (call-time import avoids cycle)."""
    from gateway.run import _load_gateway_config as _lgc
    return _lgc()


@dataclass(slots=True)
class GatewayPreparedAgentTurnStart:
    """Result of production turn bootstrap (session → history → early outs)."""

    session_entry: Any = None
    session_key: str = ""
    context: Any = None
    history: list = field(default_factory=list)
    history_for_agent: list = field(default_factory=list)
    context_prompt: str = ""
    immediate_response: str | None = None
    turn_sidecar_notes: list[str] = field(default_factory=list)
    session_env_tokens: Any = None
    source: Any = None
    aborted: bool = False
    abort_response: str | None = None


async def prepare_gateway_agent_turn_start(
    *,
    runner: Any,
    event: Any,
    source: Any,
    quick_key: str,
    run_generation: int,
    logger: Any,
) -> GatewayPreparedAgentTurnStart:
    """Resolve session, pin context, collect sidecar notes, load history, early-outs.

    Side effects on ``runner`` (conversation scope clear, agent eviction, turn
    lease tokens, topic bindings) match the former inline bootstrap in
    ``_handle_message_with_agent``.
    """
    # Bind name used by extracted body
    _quick_key = quick_key

    # Get or create session
    # Topic-mode DMs: rewrite a stale/foreign thread_id to the user's
    # last-active topic so a cross-topic Reply or stripped plain reply
    # doesn't fragment the conversation across sessions.
    recovered = await asyncio.to_thread(runner._recover_telegram_topic_thread_id, source)
    if recovered is not None:
        logger.info(
            "telegram topic recovery: chat=%s user=%s %r -> %s",
            source.chat_id, source.user_id, source.thread_id, recovered,
        )
        source = dataclasses.replace(source, thread_id=recovered)
        try:
            event.source = source
        except Exception:
            pass

    session_entry = await runner.async_session_store.get_or_create_session(source)
    session_key = session_entry.session_key
    pinned_session_id = str(
        (getattr(event, "metadata", None) or {}).get("gateway_session_id") or ""
    ).strip()
    if pinned_session_id and pinned_session_id != session_entry.session_id:
        # Fail closed (#55578): the spawning session may have ENDED since
        # dispatch (user /new-reset, compression rotation whose parent was
        # closed). switch_session() re-opens ended sessions, so pinning
        # blindly would RESURRECT a conversation the user explicitly
        # ended and inject into it — the same illicit-revival class as
        # the ws_orphan_reap loop (#60609). A completion whose spawning
        # session is dead is dropped from injection; the subagent's
        # output remains in the delegation records.
        pinned_row = None
        try:
            if runner._session_db is not None:
                # AsyncSessionDB already offloads to a thread.
                pinned_row = await runner._session_db.get_session(pinned_session_id)
        except Exception:
            pinned_row = None
        if pinned_row is None or pinned_row.get("ended_at"):
            logger.warning(
                "Async-delegation completion pinned to session %s, which is "
                "%s — dropping injection instead of resurrecting it "
                "(#55578 fail-closed; result remains in the delegation "
                "records).",
                pinned_session_id,
                "unknown" if pinned_row is None else "ended",
            )
            return GatewayPreparedAgentTurnStart(
                aborted=True,
                abort_response=None,
            )
        prior_session_id = session_entry.session_id
        switched = await runner.async_session_store.switch_session(session_key, pinned_session_id)
        if switched is not None:
            session_entry = switched
            logger.info(
                "Pinned async-delegation completion to spawning session %s "
                "(was %s) for routing key %s (#57498)",
                pinned_session_id,
                prior_session_id,
                session_key,
            )
    runner._cache_session_source(session_key, source)
    if await asyncio.to_thread(runner._is_telegram_topic_lane, source):
        try:
            binding = (await runner._session_db.get_telegram_topic_binding(
                chat_id=str(source.chat_id),
                thread_id=str(source.thread_id),
            )) if runner._session_db else None
        except Exception:
            logger.debug("Failed to read Telegram topic binding", exc_info=True)
            binding = None
        if binding:
            bound_session_id = str(binding.get("session_id") or "")
            # Heal bindings that point at a pre-compression parent: walk
            # the compression-continuation chain forward to its tip so the
            # next message resumes the compressed child instead of
            # reloading the oversized parent transcript (#20470/#29712/
            # #33414). Returns the input unchanged when the session isn't
            # a compression parent, so this is cheap and safe.
            if bound_session_id and runner._session_db is not None:
                try:
                    canonical_session_id = await runner._session_db.get_compression_tip(
                        bound_session_id,
                    )
                except Exception:
                    logger.debug(
                        "compression-tip lookup failed for %s",
                        bound_session_id, exc_info=True,
                    )
                    canonical_session_id = bound_session_id
                if (
                    canonical_session_id
                    and canonical_session_id != bound_session_id
                ):
                    bound_session_id = canonical_session_id
            if bound_session_id and bound_session_id != session_entry.session_id:
                # Route the override through SessionStore so the session_key
                # → session_id mapping is persisted to disk and the previous
                # lane session is ended cleanly. Mutating session_entry in
                # place here created a split-brain state where the JSON
                # index pointed at one id but code downstream used another.
                switched = await runner.async_session_store.switch_session(session_key, bound_session_id)
                if switched is not None:
                    session_entry = switched
            # If the stored binding pointed at a parent, rewrite it to the
            # canonical descendant now that we've followed the chain.
            if (
                bound_session_id
                and bound_session_id != str(binding.get("session_id") or "")
            ):
                await asyncio.to_thread(
                    runner._sync_telegram_topic_binding,
                    source, session_entry, reason="compression-tip-walk",
                )
        else:
            try:
                await asyncio.to_thread(runner._record_telegram_topic_binding, source, session_entry)
            except Exception:
                logger.debug("Failed to record Telegram topic binding", exc_info=True)
    # Capture and immediately consume was_auto_reset so it does not
    # re-fire on subsequent messages — preventing the cleanup from
    # wiping model/reasoning overrides set between turns (Closes #48031).
    _was_auto_reset = getattr(session_entry, "was_auto_reset", False)
    if _was_auto_reset:
        # Treat auto-reset as a full conversation boundary — clear every
        # conversation-scoped per-session dict in one funnel call so the
        # fresh session does not inherit the previous conversation's
        # model/reasoning overrides, a queued "/model switched" note, or
        # a stale resolved-model cache (#48031, #58403). See
        # _CONVERSATION_SCOPED_STATE.
        runner._clear_conversation_scope(session_key, reason="auto_reset")
        # Evict the cached agent so the fresh session does not inherit the
        # previous conversation's context_compressor._previous_summary —
        # the cache is keyed on the stable session_key, so an auto-reset
        # otherwise reuses the old agent and leaks prior history into new
        # compaction summaries. Mirrors /reset and the compression-exhausted
        # path (#9893). Covers daily/idle/suspended auto-reset.
        runner._evict_cached_agent(session_key)
        session_entry.was_auto_reset = False

    # Emit session:start for new or auto-reset sessions
    _is_new_session = (
        session_entry.created_at == session_entry.updated_at
        or _was_auto_reset
        or getattr(session_entry, "is_fresh_reset", False)
    )
    # Consume the is_fresh_reset flag immediately so it doesn't leak
    # onto subsequent messages in the same session (issue #6508).
    if getattr(session_entry, "is_fresh_reset", False):
        session_entry.is_fresh_reset = False
    if _is_new_session:
        await runner.hooks.emit("session:start", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_id": session_entry.session_id,
            "session_key": session_key,
        })

    # Build session context (include admin identity for direct-control tools)
    admin_user_ids = runner._configured_admin_user_ids(source.platform)
    is_admin_user = runner._is_admin_user(source) if admin_user_ids else None
    context = build_session_context(
        source,
        runner.config,
        session_entry,
        admin_user_ids=admin_user_ids,
        is_admin_user=is_admin_user,
    )

    # Set session context variables for tools (task-local, concurrency-safe)
    _session_env_tokens = runner._set_session_env(context)

    # Read privacy.redact_pii from config (re-read per message)
    _redact_pii = False
    persist_user_message = None
    persist_user_timestamp = None
    try:
        _pcfg = _load_gateway_config()
        _redact_pii = bool((_pcfg.get("privacy") or {}).get("redact_pii", False))
    except Exception:
        pass

    # Build the context prompt to inject.  The render is pinned per
    # session, keyed by a hash of the exact renderer inputs
    # (_ephemeral_change_key).  A key hit reuses the pinned bytes verbatim
    # so the composed system prompt cannot drift turn-over-turn; a key
    # miss (thread rename, /sethome, redact_pii flip, ...) re-renders
    # once — the only legitimate cache busts.
    context_prompt = runner._pinned_session_context_prompt(
        context, _redact_pii, session_key
    )

    # Per-turn must-deliver notes.  These used to be appended to
    # context_prompt (the ephemeral system prompt), which guaranteed a
    # turn1→turn2 system-prompt diff and a full agent rebuild.  They now
    # ride the current user message via the api_content sidecar instead
    # (staged below, consumed in run_sync → build_turn_context).
    turn_sidecar_notes: List[str] = []

    # Explicit group address (@bot / name trigger) must not be answered with
    # [[NO_REPLY]] / empty — inject a hard turn note for the model.
    _group_reply_note = _explicit_group_reply_context_note(event)
    if _group_reply_note:
        turn_sidecar_notes.append(_group_reply_note)

    # If the previous session expired and was auto-reset, deliver a notice
    # so the agent knows this is a fresh conversation (not an intentional /reset).
    if _was_auto_reset:
        reset_reason = getattr(session_entry, 'auto_reset_reason', None) or 'idle'
        if reset_reason == "suspended":
            context_note = "[System note: The user's previous session was stopped and suspended. This is a fresh conversation with no prior context.]"
        elif reset_reason == "daily":
            context_note = "[System note: The user's session was automatically reset by the daily schedule. This is a fresh conversation with no prior context.]"
        elif reset_reason == "resume_pending_expired":
            context_note = "[System note: The previous gateway session could not be recovered after a restart (API recovery timed out). This is a fresh conversation — use /resume to restore history if needed.]"
        else:
            context_note = "[System note: The user's previous session expired due to inactivity. This is a fresh conversation with no prior context.]"
        turn_sidecar_notes.append(context_note)

        # Send a user-facing notification explaining the reset, unless:
        # - notifications are disabled in config
        # - the platform is excluded (e.g. api_server, webhook)
        # - the expired session had no activity (nothing was cleared)
        try:
            policy = runner.session_store.config.get_reset_policy(
                platform=source.platform,
                session_type=getattr(source, 'chat_type', 'dm'),
            )
            platform_name = source.platform.value if source.platform else ""
            had_activity = getattr(session_entry, 'reset_had_activity', False)
            # Suspended and restart-recovery-expired sessions always notify
            # regardless of policy.notify — the user had an active session
            # that was silently replaced, so they need to know they can
            # /resume it.  Idle/daily resets respect the policy flag.
            should_notify = reset_reason in {"suspended", "resume_pending_expired"} or (
                policy.notify
                and had_activity
                and platform_name not in policy.notify_exclude_platforms
            )
            if should_notify:
                adapter = runner._adapter_for_source(source)
                if adapter:
                    if reset_reason == "suspended":
                        reason_text = "previous session was stopped or interrupted"
                    elif reset_reason == "resume_pending_expired":
                        reason_text = "gateway restart recovery timed out"
                    elif reset_reason == "daily":
                        reason_text = f"daily schedule at {policy.at_hour}:00"
                    else:
                        hours = policy.idle_minutes // 60
                        mins = policy.idle_minutes % 60
                        duration = f"{hours}h" if not mins else f"{hours}h {mins}m" if hours else f"{mins}m"
                        reason_text = f"inactive for {duration}"
                    notice = (
                        f"◐ Session automatically reset ({reason_text}). "
                        f"Conversation history cleared.\n"
                        f"Use /resume to browse and restore a previous session.\n"
                        f"Adjust reset timing in config.yaml under session_reset."
                    )
                    try:
                        session_info = await asyncio.to_thread(
                            runner._reset_notice_session_info, source
                        )
                        if session_info:
                            notice = f"{notice}\n\n{session_info}"
                    except Exception:
                        pass
                    await adapter.send(
                        source.chat_id, notice,
                        metadata=runner._thread_metadata_for_source(source),
                    )
        except Exception as e:
            logger.debug("Auto-reset notification failed (non-fatal): %s", e)

        # was_auto_reset is already consumed in the cleanup block above
        # (single source of truth); only the reset reason needs clearing here.
        session_entry.auto_reset_reason = None

    # Auto-load skill(s) for topic/channel bindings (Telegram DM Topics,
    # Discord channel_skill_bindings).  Supports a single name or ordered list.
    # Only inject on NEW sessions — ongoing conversations already have the
    # skill content in their conversation history from the first message.
    _auto = getattr(event, "auto_skill", None)
    if _is_new_session and _auto:
        _skill_names = [_auto] if isinstance(_auto, str) else list(_auto)
        try:
            from agent.skill_commands import _load_skill_payload, _build_skill_message
            _combined_parts: list[str] = []
            _loaded_names: list[str] = []
            for _sname in _skill_names:
                _loaded = _load_skill_payload(_sname, task_id=_quick_key)
                if _loaded:
                    _loaded_skill, _skill_dir, _display_name = _loaded
                    _note = (
                        f'[IMPORTANT: The "{_display_name}" skill is auto-loaded. '
                        f"Follow its instructions for this session.]"
                    )
                    _part = _build_skill_message(_loaded_skill, _skill_dir, _note)
                    if _part:
                        _combined_parts.append(_part)
                        _loaded_names.append(_sname)
                else:
                    logger.warning("[Gateway] Auto-skill '%s' not found", _sname)
            if _combined_parts:
                # Append the user's original text after all skill payloads
                _combined_parts.append(event.text)
                event.text = "\n\n".join(_combined_parts)
                logger.info(
                    "[Gateway] Auto-loaded skill(s) %s for session %s",
                    _loaded_names, session_key,
                )
        except Exception as e:
            logger.warning("[Gateway] Failed to auto-load skill(s) %s: %s", _skill_names, e)

    # ── Turn lease (#64934) ────────────────────────────────────────
    # Session resolution is FINAL here (get_or_create → async-delegation
    # pinning → topic tip-walk switch_session are all above). Serialize
    # the [load history → run → flush] region per resolved SESSION_ID:
    # when a second routing key is mapped to this same session_id, its
    # turn waits here for the previous turn's flush instead of loading a
    # stale history base and interleaving transcript writes. Same-key
    # messages never reach this point mid-turn (adapter + runner guards
    # hold them), so the lock is uncontended outside the alias-key route.
    # Fail-open: on timeout the token comes back degraded and the turn
    # proceeds unserialized (never a wedged session). Released in
    # _handle_message's finally via _release_turn_lease — granted per
    # (routing key, run generation) so a stale unwind can't release a
    # newer turn's lease.
    _lease_registry = getattr(runner, "_turn_leases", None)
    if _lease_registry is not None:
        _lease_token = await _lease_registry.acquire(
            session_entry.session_id,
            owner_key=_quick_key,
            generation=run_generation,
            timeout=_float_env("HERMES_AGENT_TIMEOUT", 1800),
        )
        if _lease_token is not None:
            if not hasattr(runner, "_turn_lease_tokens"):
                runner._turn_lease_tokens = {}
            runner._turn_lease_tokens[(_quick_key, run_generation)] = _lease_token

    # Load conversation history from transcript
    history = await runner.async_session_store.load_transcript(session_entry.session_id)

    # Direct oral shortcuts (QQ admin send/group/intel/social, runtime status,
    # background-job status). Must run before agent so oral ops never spend a
    # full model turn — and never risk wrong-tool chatter on the personal QQ.
    try:
        from gateway.direct_shortcut_runtime_service import (
            try_handle_direct_gateway_shortcuts,
        )

        direct_shortcut_response = try_handle_direct_gateway_shortcuts(
            runner,
            event,
            conversation_history=list(history or []),
            logger=logger,
        )
    except Exception as exc:
        logger.warning("Direct gateway shortcut handling failed: %s", exc)
        direct_shortcut_response = None
    if direct_shortcut_response is not None:
        return GatewayPreparedAgentTurnStart(
            session_entry=session_entry,
            session_key=session_key,
            context=context,
            history=history,
            history_for_agent=[],
            context_prompt=context_prompt,
            immediate_response=direct_shortcut_response,
            turn_sidecar_notes=list(turn_sidecar_notes),
            session_env_tokens=_session_env_tokens,
            source=source,
        )

    # Auto-detach obvious long-running work when enabled for this platform.
    try:
        auto_background_response = runner._maybe_auto_background_turn(
            event=event,
            source=source,
            context=context,
            session_key=session_key,
            history=history,
            context_prompt=context_prompt,
            session_id=session_entry.session_id,
            logger=logger,
        )
    except Exception as exc:
        logger.warning("Auto-background dispatch failed: %s", exc)
        auto_background_response = None
    if auto_background_response is not None:
        return GatewayPreparedAgentTurnStart(
            session_entry=session_entry,
            session_key=session_key,
            context=context,
            history=history,
            history_for_agent=list(history or []),
            context_prompt=context_prompt,
            immediate_response=auto_background_response,
            turn_sidecar_notes=list(turn_sidecar_notes),
            session_env_tokens=_session_env_tokens,
            source=source,
        )

    return GatewayPreparedAgentTurnStart(
        session_entry=session_entry,
        session_key=session_key,
        context=context,
        history=history,
        history_for_agent=list(history or []),
        context_prompt=context_prompt,
        turn_sidecar_notes=list(turn_sidecar_notes),
        session_env_tokens=_session_env_tokens,
        source=source,
    )

