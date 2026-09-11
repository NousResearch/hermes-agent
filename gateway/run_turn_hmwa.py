"""run_turn_hmwa — the _hmwa* methods split out of GatewayTurnMixin (mechanical extraction)."""

from __future__ import annotations
import logging
import asyncio
import dataclasses
import json
import os
import time
from agent.i18n import t
from contextlib import suppress
from contextvars import copy_context
from gateway.config import Platform
from gateway.session import build_channel_continuity_note, build_session_context
from gateway.session_transcript import TranscriptReadError
from gateway.turn_lease import DEFAULT_LEASE_WAIT, TurnLeaseTimeoutError
from typing import List
logger = logging.getLogger("gateway.run")

class GatewayTurnHmwaMixin:
    async def _hmwa_resolve_session(self, event, source):
        """Resolve ``source`` to its session entry (topic recovery, internal-route guards, Telegram
        topic-binding heal). Returns ``(source, session_entry, session_key)`` or ``None`` to drop
        the event."""
        # Topic-mode DMs: rewrite a stale/foreign thread_id to the user's last-active topic so a
        # cross-topic Reply doesn't fragment the conversation.
        event_metadata = getattr(event, "metadata", None) or {}
        expected_session_key = str(event_metadata.get("gateway_session_key") or "").strip()
        recovered = (await asyncio.to_thread(self._recover_telegram_topic_thread_id, source)
                     if not expected_session_key else None)
        if recovered is not None:
            logger.info(
                "telegram topic recovery: chat=%s user=%s %r -> %s",
                source.chat_id, source.user_id, source.thread_id, recovered,
            )
            source = dataclasses.replace(source, thread_id=recovered)
            with suppress(Exception):
                event.source = source

        if expected_session_key:
            derived_session_key = self._session_key_for_source(source)
            if derived_session_key != expected_session_key:
                logger.warning(
                    "Dropping internally routed event after route recovery: expected session=%s derived=%s",
                    expected_session_key, derived_session_key,
                )
                return

        strict_session = bool(event_metadata.get("gateway_session_strict"))
        pinned_session_id = str(event_metadata.get("gateway_session_id") or "").strip()
        if strict_session:
            session_entry = await self.async_session_store.lookup_by_session_key(expected_session_key)
            if session_entry is None or not pinned_session_id or session_entry.session_id != pinned_session_id:
                logger.warning(
                    "Dropping internally routed event: expected session id=%s is no longer current for key=%s",
                    pinned_session_id or "missing", expected_session_key or "missing",
                )
                return
        else:
            # Internal wakes observe reset policy without counting as user activity, or periodic
            # notifications keep the routing key alive across every daily/idle boundary.
            session_entry = await self.async_session_store.get_or_create_session(
                source, touch_activity=not bool(getattr(event, "internal", False)),
            )
        session_key = session_entry.session_key
        if not strict_session and pinned_session_id:
            resolved_entry = await self._resolve_async_delegation_session(session_entry, pinned_session_id)
            if resolved_entry is None:
                return
            session_entry = resolved_entry
        self._cache_session_source(session_key, source)
        if not bool(getattr(event, "internal", False)):
            try:
                from gateway.wisdom_mediation import schedule as observe_wisdom_session

                await observe_wisdom_session(
                    self, self._adapter_for_source(source), source,
                    str(session_entry.session_id), observe_only=True,
                )
            except Exception:
                logger.debug("Wisdom session activity unavailable", exc_info=True)
        if await asyncio.to_thread(self._is_telegram_topic_lane, source):
            session_entry = await self._hmwa_heal_telegram_topic_binding(source, session_entry, session_key)
        from gateway.run_heartbeat_acceptance import resolve_heartbeat_owner
        if not await resolve_heartbeat_owner(self, event, session_entry):
            return
        return source, session_entry, session_key

    async def _hmwa_heal_telegram_topic_binding(self, source, session_entry, session_key):
        """Follow the (chat_id, thread_id) topic binding — healed to its compression tip — or record
        a fresh one. Returns the (possibly switched) session entry."""
        binding = None
        try:
            if self._session_db:
                binding = await self._session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id), thread_id=str(source.thread_id),
                    profile_name=self._telegram_topic_profile_name(source),
                )
        except Exception:
            logger.debug("Failed to read Telegram topic binding", exc_info=True)
        if not binding:
            try:
                await asyncio.to_thread(self._record_telegram_topic_binding, source, session_entry)
            except Exception:
                logger.debug("Failed to record Telegram topic binding", exc_info=True)
            return session_entry
        stored_session_id = str(binding.get("session_id") or "")
        bound_session_id = stored_session_id
        # A binding pointing at a pre-compression parent is walked forward to the tip so the next
        # message resumes the compressed child instead of reloading the oversized parent.
        # Returns the input unchanged when the session isn't a compression parent, so this is cheap and
        # safe. See #20470, #29712, #33414.
        if bound_session_id and self._session_db is not None:
            try:
                canonical_session_id = await self._session_db.get_compression_tip(bound_session_id)
            except Exception:
                logger.debug("compression-tip lookup failed for %s", bound_session_id, exc_info=True)
                canonical_session_id = bound_session_id
            if canonical_session_id and canonical_session_id != bound_session_id:
                bound_session_id = canonical_session_id
        if bound_session_id and bound_session_id != session_entry.session_id:
            # Route through SessionStore so the key → id mapping persists and the previous lane
            # session ends cleanly (in-place mutation split-brained the JSON index).
            switched = await self.async_session_store.switch_session(session_key, bound_session_id)
            if switched is not None:
                session_entry = switched
        if bound_session_id and bound_session_id != stored_session_id:
            # The stored binding pointed at a parent: rewrite it to the canonical descendant.
            await asyncio.to_thread(
                self._sync_telegram_topic_binding, source, session_entry, reason="compression-tip-walk",
            )
        return session_entry

    async def _hmwa_open_session(self, session_entry, session_key, source):
        """Consume auto-reset / fresh-reset flags and emit ``session:start`` for new sessions.
        Returns ``(_was_auto_reset, _is_new_session)``."""
        # Consume was_auto_reset immediately so it cannot re-fire and wipe overrides set between turns.
        # Capture and immediately consume was_auto_reset so it does not re-fire on subsequent messages —
        # preventing the cleanup from wiping model/reasoning overrides set between turns (Closes #48031).
        _was_auto_reset = getattr(session_entry, "was_auto_reset", False)
        if _was_auto_reset:
            # Conversation boundary: the funnel clears every conversation-scoped dict; evict the cached
            # agent so context_compressor._previous_summary cannot leak into new summaries.
            # Treat auto-reset as a full conversation boundary — clear every conversation-scoped per-session
            # dict in one funnel call so the fresh session does not inherit the previous conversation's
            # model/reasoning overrides, a queued "/model switched" note, or a stale resolved-model cache
            # (#48031, #58403). See _CONVERSATION_SCOPED_STATE.
            self._clear_conversation_scope(session_key, reason="auto_reset")
            self._evict_cached_agent(session_key)
            session_entry.was_auto_reset = False

        _is_fresh_reset = getattr(session_entry, "is_fresh_reset", False)
        _is_new_session = session_entry.created_at == session_entry.updated_at or _was_auto_reset or _is_fresh_reset
        # Consume is_fresh_reset so it doesn't leak onto later messages in the same session.
        if _is_fresh_reset:
            # See #6508.
            session_entry.is_fresh_reset = False
        if _is_new_session:
            await self.hooks.emit("session:start", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            })
        return _was_auto_reset, _is_new_session

    async def _hmwa_deliver_auto_reset_notice(self, session_entry, source, turn_sidecar_notes):
        """Stage the auto-reset sidecar note for the agent and notify the user (policy-gated)."""
        from gateway.run import _AUTO_RESET_CONTEXT_NOTES
        reset_reason = getattr(session_entry, 'auto_reset_reason', None) or 'suspended'
        context_note = _AUTO_RESET_CONTEXT_NOTES.get(reset_reason, _AUTO_RESET_CONTEXT_NOTES["suspended"])
        # Long-lived channels: point the agent at the prior same-channel session for session_search.
        try:
            # Returns None (appends nothing) for other platforms or when there's no prior activity to
            # recall. Deterministic — no extra API/DB calls (#36220).
            continuity_note = build_channel_continuity_note(session_entry, source)
        except Exception:
            continuity_note = None
        if continuity_note:
            context_note = context_note + "\n\n" + continuity_note
        turn_sidecar_notes.append(context_note)

        try:
            should_notify = reset_reason == "suspended"
            adapter = self._adapter_for_source(source) if should_notify else None
            if adapter:
                notice = (
                    "◐ Session reset after being stopped. "
                    f"Conversation history cleared.\n"
                    f"Use /resume to browse and restore a previous session.\n"
                )
                with suppress(Exception):
                    session_info = await asyncio.to_thread(self._reset_notice_session_info, source)
                    if session_info:
                        notice = f"{notice}\n\n{session_info}"
                await adapter.send(source.chat_id, notice, metadata=self._thread_metadata_for_source(source))
        except Exception as e:
            logger.debug("Auto-reset notification failed (non-fatal): %s", e)

        # was_auto_reset was consumed in _hmwa_open_session; only the reason needs clearing.
        session_entry.auto_reset_reason = None

    def _hmwa_auto_load_skills(self, event, _auto, _quick_key, session_key):
        """Prepend topic/channel-bound skill payload(s) to ``event.text`` on a new session."""
        _skill_names = [_auto] if isinstance(_auto, str) else list(_auto)
        try:
            from agent.skill_commands import _load_skill_payload, _build_skill_message
            _combined_parts: list[str] = []
            _loaded_names: list[str] = []
            for _sname in _skill_names:
                _loaded = _load_skill_payload(_sname, task_id=_quick_key)
                if not _loaded:
                    logger.warning("[Gateway] Auto-skill '%s' not found", _sname)
                    continue
                _loaded_skill, _skill_dir, _display_name = _loaded
                _part = _build_skill_message(
                    _loaded_skill, _skill_dir,
                    f'[IMPORTANT: The "{_display_name}" skill is auto-loaded. '
                    f"Follow its instructions for this session.]",
                )
                if _part:
                    _combined_parts.append(_part)
                    _loaded_names.append(_sname)
            if _combined_parts:
                _combined_parts.append(event.text)  # user's original text after the payloads
                event.text = "\n\n".join(_combined_parts)
                logger.info("[Gateway] Auto-loaded skill(s) %s for session %s", _loaded_names, session_key)
        except Exception as e:
            logger.warning("[Gateway] Failed to auto-load skill(s) %s: %s", _skill_names, e)

    async def _hmwa_acquire_turn_lease(self, _quick_key, run_generation, session_entry, _session_env_tokens):
        """Serialize [load history → run → flush] per resolved SESSION_ID so another routing key on
        the same session waits for the prior flush. Fail-closed on timeout (outer dispatch returns
        a resend notice). Released in _handle_message's finally, granted per (routing key, run
        generation) so a stale unwind can't release a newer turn's."""
        from gateway.run import _float_env
        _lease_registry = getattr(self, "_turn_leases", None)
        if _lease_registry is None:
            return
        try:
            _lease_token = await _lease_registry.acquire(
                session_entry.session_id, owner_key=_quick_key, generation=run_generation,
                timeout=_float_env("HERMES_TURN_LEASE_TIMEOUT", DEFAULT_LEASE_WAIT),
            )
        except TurnLeaseTimeoutError:
            # The cleanup finally starts later; restore the tokens here or this exit leaks identity.
            self._clear_session_env(_session_env_tokens)
            raise
        if _lease_token is not None:
            _lease_state = self._session_state(_quick_key).turn
            _lease_state.lease_token = _lease_token
            _lease_state.lease_generation = run_generation

    @staticmethod
    def _hmwa_hygiene_read_config(hs, data):
        """Apply model / compression knobs from the gateway config onto ``hs`` (invalid values keep the defaults)."""
        # Resolve model name (same logic as run_sync)
        _model_cfg = data.get("model", {})
        if isinstance(_model_cfg, str):
            hs.model = _model_cfg
        elif isinstance(_model_cfg, dict):
            hs.model = _model_cfg.get("default") or _model_cfg.get("model") or hs.model
            _raw_ctx = _model_cfg.get("context_length")
            if _raw_ctx is not None:
                with suppress(TypeError, ValueError):
                    hs.config_context_length = int(_raw_ctx)
            hs.provider = _model_cfg.get("provider") or None
            hs.base_url = _model_cfg.get("base_url") or None

        # Only the enabled flag is shared with the agent's compression config (hygiene runs higher).
        _comp_cfg = data.get("compression", {})
        if not isinstance(_comp_cfg, dict):
            return
        hs.compression_enabled = str(_comp_cfg.get("enabled", True)).lower() in {"true", "1", "yes"}

        def _knob(key, current, cast, allow_zero=False):
            raw = _comp_cfg.get(key)
            if raw is None:
                return current
            try:
                parsed = cast(raw)
            except (TypeError, ValueError):
                return current
            return parsed if (parsed >= 0 if allow_zero else parsed > 0) else current

        hs.hard_msg_limit = _knob("hygiene_hard_message_limit", hs.hard_msg_limit, int)
        hs.timeout_seconds = _knob("hygiene_timeout_seconds", hs.timeout_seconds, float)
        hs.total_ceiling_seconds = _knob("hygiene_total_ceiling_seconds", hs.total_ceiling_seconds, float)
        # The ceiling can never be tighter than one idle window, or the extension loop would be dead code.
        hs.total_ceiling_seconds = max(hs.total_ceiling_seconds, hs.timeout_seconds)
        hs.max_turn_hold_seconds = _knob("hygiene_max_turn_hold_seconds", hs.max_turn_hold_seconds, float)
        hs.failure_cooldown_seconds = _knob(
            "hygiene_failure_cooldown_seconds", hs.failure_cooldown_seconds, float, allow_zero=True,
        )

    async def _hmwa_hygiene_settings(self, source, session_key):
        """Resolve model/provider/context-length + hygiene knobs (fail-soft: errors keep defaults).

        The 0.85 threshold is deliberately HIGHER than the agent's compressor (0.50): a safety net
        for sessions that grew between turns. ``max_turn_hold_seconds`` bounds the TURN wait
        (compressor keeps running detached, commit fenced); kept below transport idle-timeouts."""
        from gateway.run import _load_gateway_config
        hs = self._HygieneSettings(
            model="anthropic/claude-sonnet-4.6", threshold_pct=0.85, compression_enabled=True,
            hard_msg_limit=5000, timeout_seconds=30.0, total_ceiling_seconds=600.0,
            max_turn_hold_seconds=10.0, failure_cooldown_seconds=300.0, config_context_length=None,
            provider=None, base_url=None, api_key=None, data={},
        )
        try:
            hs.data = _load_gateway_config()
            if hs.data:
                self._hmwa_hygiene_read_config(hs, hs.data)
            configured_model, configured_provider, configured_base_url = hs.model, hs.provider, hs.base_url

            with suppress(Exception):
                hs.model, _hyg_runtime = self._resolve_session_agent_runtime(
                    source=source, session_key=session_key,
                    user_config=hs.data if isinstance(hs.data, dict) else None,
                )
                hs.provider = _hyg_runtime.get("provider") or hs.provider
                hs.base_url = _hyg_runtime.get("base_url") or hs.base_url
                hs.api_key = _hyg_runtime.get("api_key") or hs.api_key

            if hs.config_context_length is not None:
                try:
                    from hermes_cli.route_identity import should_clear_context_pin_async

                    if await should_clear_context_pin_async(
                        configured_model, hs.model, configured_base_url, hs.base_url,
                        configured_provider, hs.provider,
                    ):
                        hs.config_context_length = None
                except Exception:
                    hs.config_context_length = None

            # custom_providers per-model context_length fallback (as in run_agent.py); needs base_url.
            if hs.config_context_length is None and hs.base_url:
                with suppress(TypeError, ValueError):
                    try:
                        from hermes_cli.config import (
                            get_compatible_custom_providers as _gw_gcp,
                            get_custom_provider_context_length as _gw_gccl,
                        )
                        _hyg_custom_providers = _gw_gcp(hs.data)
                    except Exception:
                        _hyg_custom_providers = hs.data.get("custom_providers")
                        if not isinstance(_hyg_custom_providers, list):
                            _hyg_custom_providers = []
                    _hyg_custom_ctx = _gw_gccl(
                        model=hs.model, base_url=hs.base_url, custom_providers=_hyg_custom_providers,
                    )
                    if _hyg_custom_ctx:
                        hs.config_context_length = int(_hyg_custom_ctx)
        except Exception:
            pass
        return hs

    async def _hmwa_hygiene_plan(self, hs, history, session_entry, session_key):
        """Decide whether hygiene compression fires this turn (token/message thresholds, DB-backed
        failure cooldown, in-flight compression)."""
        from agent.model_metadata import estimate_messages_tokens_rough, get_model_context_length_async
        _hyg_context_length = await get_model_context_length_async(
            hs.model, base_url=hs.base_url or "", api_key=hs.api_key or "",
            config_context_length=hs.config_context_length, provider=hs.provider or "",
        )
        _compress_token_threshold = int(_hyg_context_length * hs.threshold_pct)
        _warn_token_threshold = int(_hyg_context_length * 0.95)
        _msg_count = len(history)

        # Real usage decides: the API-reported prompt count, else the anchor persisted on the session
        # row (real count + delta of what was appended since, survives gateway restarts), else the
        # rough estimate (runs 30-50% high, which only fires hygiene early — safe). Do NOT compensate
        # with a threshold multiplier.
        from agent.image_token_cost import image_cost_context, learned_image_token_cost
        _anchored = None
        # Images in any local delta/estimate are priced at the cost learned from this model's usage.
        with image_cost_context(learned_image_token_cost(hs.model, hs.base_url)):
            if session_entry.last_prompt_tokens <= 0:
                from agent.usage_anchor import persisted_anchor_tokens
                _session_db = getattr(self, "_session_db", None)
                _anchored = persisted_anchor_tokens(
                    getattr(_session_db, "_db", _session_db), session_entry.session_id, history,
                )
            if session_entry.last_prompt_tokens > 0:
                _approx_tokens, _token_source = session_entry.last_prompt_tokens, "actual"
            elif _anchored is not None:
                _approx_tokens, _token_source = _anchored, "anchored"
            else:
                _approx_tokens, _token_source = estimate_messages_tokens_rough(history), "estimated"

        # Hard safety valve: force compression at an extreme message count regardless of tokens,
        # breaking the disconnect → no token data → no compression spiral. 5000 clears 1M+ sessions.
        _needs_compress = _approx_tokens >= _compress_token_threshold or _msg_count >= hs.hard_msg_limit

        if _needs_compress:
            # DB-backed cooldown (shared with context_compressor.py): survives gateway restarts, so a
            # failing compression is not re-triggered on every restart.
            # The in-memory dict was reset on every restart, re-triggering the same failing compression and
            # wedging session storage (#74136).
            _session_db = getattr(self, "_session_db", None)
            _getter = getattr(getattr(_session_db, "_db", _session_db), "get_compression_failure_cooldown", None)
            if _getter is not None:
                _cooldown_state = None
                with suppress(Exception):
                    _cooldown_state = _getter(session_entry.session_id)
                if _cooldown_state and _cooldown_state.get("remaining_seconds", 0) > 0:
                    logger.info(
                        "Session hygiene: skipping compression for %s; "
                        "previous failure cooldown active for %.1fs",
                        session_entry.session_id, _cooldown_state["remaining_seconds"],
                    )
                    _needs_compress = False

        if _needs_compress and await self._session_has_compression_in_flight(session_key):
            # A prior compression still holds the durable lock (e.g. a shielded worker left by /stop):
            # another attempt would wait up to 600s behind a commit the fence will refuse.
            logger.info(
                "Session hygiene: skipping compression for %s; "
                "another compression is already in flight", session_entry.session_id,
            )
            _needs_compress = False

        if _needs_compress:
            logger.info(
                "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
                "(threshold: %s%% of %s = %s tokens)",
                _msg_count, f"{_approx_tokens:,}", _token_source,
                int(hs.threshold_pct * 100), f"{_hyg_context_length:,}", f"{_compress_token_threshold:,}",
            )
        return self._HygienePlan(_needs_compress, _approx_tokens, _msg_count, _warn_token_threshold)

    async def _hmwa_hygiene_wait_for_summary(self, attempt, hs, session_entry):
        """Progress-aware inline wait for the detached hygiene compressor. Returns the compressed
        transcript; raises ``HygieneTurnHoldExceeded`` (turn-hold budget) or
        ``asyncio.TimeoutError`` (idle/ceiling/fence cancel) for the caller's handlers.

        Idle timeout (fence ticks per streamed token) + hard ceiling + turn-hold cap."""
        from gateway.run import HygieneTurnHoldExceeded, hygiene_wait_should_extend
        fence = attempt.commit_fence
        while True:
            if fence.is_cancelled:
                raise asyncio.TimeoutError
            # Charge the idle budget from the LAST PROGRESS event, else silence can approach 2x timeout.
            _hyg_waited = time.monotonic() - attempt.wait_started
            _slice = min(
                max(hs.timeout_seconds - fence.seconds_since_progress(), 0.005),
                max(hs.total_ceiling_seconds - _hyg_waited, 0.005),
            )
            # Cap the slice at the remaining turn-hold budget so a continuously-streaming worker can't
            # hold the turn until the ceiling. Budget exhausted → immediate timeout → abandonment.
            _turn_hold_remaining = hs.max_turn_hold_seconds - (time.monotonic() - attempt.wait_started)
            _slice = 0.005 if _turn_hold_remaining <= 0 else min(_slice, max(_turn_hold_remaining, 0.005))
            # Short poll so a /stop or /restart cancel is not stuck behind a full idle window.
            _idle_left = max(hs.timeout_seconds - fence.seconds_since_progress(), 0.005)
            _slice = min(_slice, 0.25)
            try:
                _compressed, _ = await asyncio.wait_for(asyncio.shield(attempt.future), timeout=_slice)
                return _compressed
            except asyncio.TimeoutError:
                if fence.is_cancelled:
                    raise
                _hyg_waited = time.monotonic() - attempt.wait_started
                _idle = fence.seconds_since_progress()
                # Never hold the TURN past the budget even while the summary streams: proceed on the
                # uncompressed transcript so the wire never trips a transport idle-timeout.
                if _hyg_waited >= hs.max_turn_hold_seconds:
                    logger.info(
                        "Session hygiene compression for session %s exceeded the turn-hold "
                        "budget (%.1fs >= %.1fs) — abandoning inline wait, proceeding "
                        "without compression this turn",
                        session_entry.session_id, _hyg_waited, hs.max_turn_hold_seconds,
                    )
                    raise HygieneTurnHoldExceeded(
                        f"turn-hold budget {hs.max_turn_hold_seconds:.1f}s "
                        f"elapsed after {_hyg_waited:.1f}s"
                    )
                if hygiene_wait_should_extend(
                    idle=_idle, timeout=hs.timeout_seconds, waited=_hyg_waited,
                    ceiling=hs.total_ceiling_seconds, fence_cancelled=fence.is_cancelled,
                ):
                    if _slice >= _idle_left - 1e-9:
                        logger.info(
                            "Session hygiene compression for session %s still streaming after "
                            "%.0fs (last progress %.1fs ago) — extending wait (ceiling %.0fs)",
                            session_entry.session_id, _hyg_waited, _idle, hs.total_ceiling_seconds,
                        )
                    continue
                raise

    async def _hmwa_hygiene_cancel_or_adopt(self, attempt, context):
        """Cancel the worker at the commit fence; on success release its lease and defer agent
        cleanup, returning ``None``. When the worker already crossed into its commit, consume and
        return the compressed transcript instead (a successful compaction is never a timeout; the
        turn may be held past the budget by up to the commit duration — by design). The lock-free
        ``commit_in_flight`` marker keeps the poll from spinning on a hung commit."""
        fence = attempt.commit_fence
        while not fence.commit_in_flight:
            cancelled = fence.try_cancel_before_commit()
            if cancelled is None:
                await asyncio.sleep(0.025)
            elif cancelled:
                fence.release_cancelled_compression_lock()
                self._hmwa_hygiene_defer_cleanup(attempt, context)
                return None
            else:
                break
        _compressed, _ = await attempt.future
        return _compressed

    def _hmwa_hygiene_defer_cleanup(self, attempt, context):
        """Hand the agent's cleanup to the still-running worker future and mark it deferred."""
        self._defer_agent_cleanup_until_future_done(attempt.future, attempt.agent, context=context)
        attempt.cleanup_deferred = True

    @staticmethod
    def _hmwa_hygiene_stamp(agent, desc, provenance_name, debug_label):
        from agent.session_activity import ActivityProvenance
        from gateway.run import _stamp_hygiene_compression_provenance
        _stamp_hygiene_compression_provenance(agent, desc, getattr(ActivityProvenance, provenance_name), debug_label)

    async def _hmwa_hygiene_notify(self, source, meta, message, what):
        """Best-effort user notice on the hygiene thread; failure is logged, never raised."""
        try:
            _adapter = self._adapter_for_source(source)
            if _adapter and source.chat_id:
                await _adapter.send(source.chat_id, message, metadata=meta)
        except Exception as _werr:
            logger.warning("Failed to deliver %s to user: %s", what, _werr)

    async def _hmwa_hygiene_record_failure_cooldown(self, hs, session_key, session_id, reason):
        """Escalate the failure streak (off-loop) and persist the cooldown, when enabled."""
        from gateway.run import _hygiene_cooldown_for_failure, _record_hygiene_cooldown
        if hs.failure_cooldown_seconds < 0:
            return
        _hyg_cooldown = await asyncio.to_thread(
            _hygiene_cooldown_for_failure, self, session_key, hs.failure_cooldown_seconds,
        )
        _record_hygiene_cooldown(self, session_id, _hyg_cooldown, reason)

    async def _hmwa_hygiene_on_turn_hold(self, attempt, hs, session_entry, session_key, source):
        """``except HygieneTurnHoldExceeded`` body: keep or cancel the worker's commit admission,
        notify the user, and re-raise; returns the compressed transcript only when the worker
        was already committing.

        Turn-hold expiry is an availability boundary, not a failure: the streak must NOT advance,
        only flat retry spacing is recorded. A watermark-fenced commit (rows appended after
        compression start survive as cloned tail) KEEPS admission: the turn proceeds uncompressed
        now and the summary is adopted at the worker's fenced commit — always cancelling burned
        every attempt for thinking summary models. Without the fence a late commit could clobber
        newer turns, so cancel."""
        from gateway.run import (
            _HYGIENE_TURNHOLD_RETRY_SECONDS, _record_hygiene_cooldown, _reset_hygiene_failure_streak
        )
        fence = attempt.commit_fence
        _hyg_keep_admission = bool(getattr(fence, "commit_watermark_fenced", False)) and not fence.is_cancelled
        if _hyg_keep_admission:
            self._hmwa_hygiene_defer_cleanup(attempt, "session hygiene turn-hold")
            # NO retry-after here (it would also block the agent-side preflight compressor); spacing
            # comes from the durable compression lock. The done-callback records the flat retry-after
            # ONLY if the worker ends without committing anything.
            _sid, _skey, _agent = session_entry.session_id, session_key, attempt.agent

            def _hyg_adopt_or_space_retry(_fut, _gw=self, _sid=_sid, _skey=_skey, _agent=_agent):
                try:
                    _exc = _fut.exception()
                except (asyncio.CancelledError, Exception):
                    _committed = False
                else:
                    _committed = _exc is None and (
                        bool(getattr(_agent, "_last_compaction_in_place", False))
                        or getattr(_agent, "session_id", _sid) != _sid
                    )
                if _committed:
                    logger.info(
                        "Session hygiene compression for session %s finished after the "
                        "turn-hold was released — summary adopted at the watermark-fenced "
                        "commit boundary (#97963)", _sid,
                    )
                    try:
                        _reset_hygiene_failure_streak(_gw, _skey)
                    except Exception as _rs_err:
                        logger.debug("hygiene streak reset after deferred adoption failed: %s", _rs_err)
                else:
                    # Nothing to adopt (summary failed / fence refused / superseded): flat spacing so
                    # sustained traffic doesn't spawn and abandon a compressor every turn.
                    _record_hygiene_cooldown(
                        _gw, _sid, _HYGIENE_TURNHOLD_RETRY_SECONDS,
                        "hygiene compression deferred: turn-hold budget expired and the "
                        "detached attempt did not commit",
                    )

            attempt.future.add_done_callback(_hyg_adopt_or_space_retry)
            _log_suffix = (
                " — the watermark-fenced worker keeps its commit admission and the summary "
                "will be adopted when it finishes"
            )
        else:
            _adopted = await self._hmwa_hygiene_cancel_or_adopt(attempt, "session hygiene turn-hold")
            if _adopted is not None:
                return _adopted
            # Short flat retry-after, else every turn re-spawns, holds and cancels a compressor.
            _record_hygiene_cooldown(
                self, session_entry.session_id, _HYGIENE_TURNHOLD_RETRY_SECONDS,
                "hygiene compression deferred: turn-hold budget expired while the "
                "summary was still streaming",
            )
            _log_suffix = ""
        self._hmwa_hygiene_stamp(
            attempt.agent, "session hygiene compression turn-hold",
            "AGENT_COMPRESSION_TURNHOLD", "hygiene compression turn-hold activity stamp failed",
        )
        logger.info(
            "Session hygiene compression for session %s exceeded turn-hold budget (%.1fs); "
            "proceeding without compression this turn%s",
            session_entry.session_id, time.monotonic() - attempt.wait_started, _log_suffix,
        )
        await self._hmwa_hygiene_notify(
            source, attempt.meta, t("gateway.compress.turnhold_deferred"), "compression-turnhold notice",
        )
        raise

    async def _hmwa_hygiene_on_timeout(self, attempt, hs, session_entry, session_key, source):
        """``except asyncio.TimeoutError`` body: cancel at the commit fence, record the failure
        cooldown, warn the user, and re-raise; returns the compressed transcript only when the
        worker crossed the commit boundary first."""
        from gateway.run import _hygiene_compression_timeout_message
        fence = attempt.commit_fence
        _hyg_waited = time.monotonic() - attempt.wait_started
        _hyg_total_exhausted = _hyg_waited >= hs.total_ceiling_seconds or fence.deadline_exceeded
        if _hyg_total_exhausted:
            # The worker checks this deadline between digest calls; keep its lease until it exits so
            # an unchanged session cannot overlap a retry (the release below is then a no-op).
            fence.retain_compression_lock_until_worker_done()
        # Capture fence state BEFORE try_cancel (which itself sets is_cancelled).
        _hyg_fence_cancelled = fence.is_cancelled
        _adopted = await self._hmwa_hygiene_cancel_or_adopt(attempt, "session hygiene timeout")
        if _adopted is not None:
            return _adopted
        await self._hmwa_hygiene_record_failure_cooldown(
            hs, session_key, session_entry.session_id,
            "session hygiene compression " + (
                "cancelled at commit fence" if _hyg_fence_cancelled
                else "total ceiling exhausted" if _hyg_total_exhausted
                else "timed out with no output from the summary model"
            ),
        )
        self._hmwa_hygiene_stamp(
            attempt.agent,
            "session hygiene compression cancelled at commit fence" if _hyg_fence_cancelled
            else "session hygiene compression timed out",
            "AGENT_COMPRESSION_TIMEOUT", "hygiene compression timeout activity stamp failed",
        )
        if _hyg_fence_cancelled:
            logger.warning(
                "Session hygiene compression for session %s was cancelled at the "
                "commit fence; continuing without compression", session_entry.session_id,
            )
            raise
        _hyg_elapsed = time.monotonic() - attempt.wait_started
        if _hyg_total_exhausted:
            logger.warning(
                "Session hygiene compression for session %s reached its total ceiling after "
                "%.1fs (progress observed=%s); continuing without compression",
                session_entry.session_id, _hyg_elapsed, fence.progress_observed,
            )
        else:
            logger.warning(
                "Session hygiene compression for session %s made no progress for %.1fs "
                "(total wait %.1fs, ceiling %.1fs); continuing without compression",
                session_entry.session_id, fence.seconds_since_progress(), _hyg_elapsed, hs.total_ceiling_seconds,
            )
        await self._hmwa_hygiene_notify(
            source, attempt.meta,
            _hygiene_compression_timeout_message(
                total_exhausted=_hyg_total_exhausted, elapsed=_hyg_elapsed,
                idle_timeout=hs.timeout_seconds, progress_observed=fence.progress_observed,
            ),
            "compression-timeout warning",
        )
        raise

    def _hmwa_hygiene_on_unwind(self, attempt, hs, session_entry, session_key):
        """``except BaseException`` body (caller re-raises): revoke commit admission BEFORE the host
        unwinds so the detached worker can never commit later, and record a cooldown — otherwise
        the next turn re-arms hygiene and waits up to 600s behind a fence that refuses again."""
        from gateway.run import _hygiene_cooldown_for_failure, _record_hygiene_cooldown
        attempt.commit_fence.revoke_commit_admission()
        if not attempt.cleanup_deferred:
            self._hmwa_hygiene_defer_cleanup(attempt, "session hygiene unwind")
        if hs.failure_cooldown_seconds >= 0:
            try:
                _record_hygiene_cooldown(
                    self, session_entry.session_id,
                    _hygiene_cooldown_for_failure(self, session_key, hs.failure_cooldown_seconds),
                    "session hygiene compression cancelled at commit fence",
                )
            except Exception as _cd_err:
                logger.debug("hygiene unwind cooldown record failed: %s", _cd_err)

    async def _hmwa_hygiene_adopt_transcript(
        self, attempt, _compressed, history, plan, *, session_entry, source, _quick_key, run_generation,
    ):
        """Adopt a finished compression (rotation / in-place / refused); publishes the transcript to
        continue with on ``attempt.history``. Returns ``(rotated, in_place, new_count, new_tokens)``.

        Rewrite only on rotation (NEW session id): in-place compaction already soft-archived the
        old rows and rewrite_transcript() would DELETE them; neither rotation nor in-place signals
        FAILURE and an unconditional rewrite would leave only the summary. Write-before-repoint:
        a repoint-then-failed-rewrite would point the live entry at an empty session."""
        from agent.model_metadata import estimate_messages_tokens_rough
        _hyg_agent = attempt.agent
        # _compress_context rotates to a NEW session_id so the old transcript stays intact/searchable.
        _hyg_new_sid = _hyg_agent.session_id
        _hyg_rotated = _hyg_new_sid != session_entry.session_id
        _hyg_in_place = bool(getattr(_hyg_agent, "_last_compaction_in_place", False))
        # Anti-growth guard: refuse a compression that did not shrink the transcript (seen 427K→598K).
        _hyg_in_toks = estimate_messages_tokens_rough(history)
        _hyg_out_toks = estimate_messages_tokens_rough(_compressed)
        if _hyg_rotated and _hyg_out_toks > _hyg_in_toks:
            logger.warning(
                "Gateway hygiene compression for session %s would grow transcript (~%s -> ~%s "
                "tokens); keeping the original transcript unchanged",
                session_entry.session_id, f"{_hyg_in_toks:,}", f"{_hyg_out_toks:,}",
            )
            _hyg_rotated = False
            _compressed = history
        # Only rewrite the transcript when rotation produced a NEW session id. In-place compaction does NOT
        # need a rewrite: archive_and_compact() has already soft-archived the previous active rows and
        # inserted the compacted messages as the new active set inside _compress_context(). Calling
        # rewrite_transcript() after in-place compaction would invoke replace_messages(active_only=False)
        # which DELETEs ALL rows — including the archived turns that archive_and_compact() deliberately
        # preserved (silent data loss, #61145). The danger this guards against (mirrors the /compress fix
        # #44794/#39704): if _compress_context returns a summary but neither rotates nor completes
        # archive_and_compact(), the session_id is unchanged for a FAILURE reason, and an unconditional
        # rewrite_transcript() would DELETE the original messages and replace them with only the compressed
        # summary (permanent data loss, #21301). Write-before-repoint (mirrors manual /compress): if we
        # repointed session_entry onto the child SID and rewrite_transcript then failed (lock/ENOSPC), the
        # live entry would already reference a brand-new empty session while the turn continues — the
        # conversation silently vanishes. Persist the child transcript first; only then rebind the live
        # entry.
        if _hyg_rotated:
            if not await self.async_session_store.rewrite_transcript(_hyg_new_sid, _compressed):
                logger.error(
                    "Session hygiene: failed to persist compressed transcript for rotated session "
                    "%s → %s; keeping the live entry on the original session so the "
                    "conversation is not dropped", session_entry.session_id, _hyg_new_sid,
                )
                # Fail closed: treat like no rotation.
                _hyg_rotated = False
                _hyg_in_place = False
            else:
                session_entry.session_id = _hyg_new_sid
                # The held turn lease follows the rotation (alias keys still serialize on this turn).
                self._rebind_turn_lease(_quick_key, run_generation, _hyg_new_sid)
                await self.async_session_store._save()
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding, source, session_entry, reason="hygiene-compression",
                )

        if _hyg_rotated or _hyg_in_place:
            # Rewritten (rotation) or persisted by archive_and_compact() (in-place): reset token count.
            session_entry.last_prompt_tokens = 0
            attempt.history = _compressed
            _new_count = len(_compressed)
            _new_tokens = estimate_messages_tokens_rough(_compressed)
        else:
            # No rewrite happened — post-compression counts equal the pre-compression ones.
            _new_count = plan.msg_count
            _new_tokens = plan.approx_tokens
            logger.warning(
                "Gateway hygiene compression for session %s did not rotate or compact in place (no "
                "session_db on the hygiene agent) — preserving the original transcript instead "
                "of overwriting it with the summary (#21301).", session_entry.session_id,
            )

        logger.info(
            "Session hygiene: compressed %s → %s msgs, ~%s → ~%s tokens",
            plan.msg_count, _new_count, f"{plan.approx_tokens:,}", f"{_new_tokens:,}",
        )
        if _new_tokens >= plan.warn_token_threshold:
            logger.warning("Session hygiene: still ~%s tokens after compression", f"{_new_tokens:,}")
        return _hyg_rotated, _hyg_in_place, _new_count, _new_tokens

    async def _hmwa_hygiene_apply_result(
        self, attempt, hs, _compressed, history, plan, *,
        session_entry, session_key, source, _quick_key, run_generation,
    ):
        """Adopt a finished hygiene compression, rebind the session + turn lease, record
        streak/cooldown, and warn the user on abort."""
        from gateway.run import _reset_hygiene_failure_streak, hygiene_compaction_recovered
        _hyg_rotated, _hyg_in_place, _new_count, _new_tokens = await self._hmwa_hygiene_adopt_transcript(
            attempt, _compressed, history, plan, session_entry=session_entry, source=source,
            _quick_key=_quick_key, run_generation=run_generation,
        )
        # Summary failure aborts the compressor (nothing dropped). Warn the user visibly — agent.log
        # is invisible on TG/Discord — so they know the chat is "frozen" and can /compress or /reset.
        _comp = getattr(attempt.agent, "context_compressor", None)
        _hyg_aborted = _comp is not None and getattr(_comp, "_last_compress_aborted", False)
        # A fence-cancelled _compress_context returns the original transcript with
        # _last_compress_aborted False: treat that no-op as an abort so hygiene records a cooldown
        # instead of retrying into the 600s wait. A committed rotate/in-place is never an abort.
        _hyg_fence_cancelled = bool(attempt.commit_fence.is_cancelled and not _hyg_rotated and not _hyg_in_place)
        if _hyg_fence_cancelled:
            _hyg_aborted = True
        # Recovery decision lives in the unit-tested predicate: the "neither rotated nor in place"
        # path reuses pre-compression counts, so a numbers-only check would read a no-op as success.
        if not _hyg_aborted and hygiene_compaction_recovered(
            aborted=_hyg_aborted, rotated=_hyg_rotated, in_place=_hyg_in_place,
            msg_count=plan.msg_count, new_count=_new_count, approx_tokens=plan.approx_tokens,
            new_tokens=_new_tokens,
        ):
            await asyncio.to_thread(_reset_hygiene_failure_streak, self, session_key)
        if _hyg_aborted:
            await self._hmwa_hygiene_record_failure_cooldown(
                hs, session_key, session_entry.session_id,
                "session hygiene compression cancelled at commit fence" if _hyg_fence_cancelled
                else getattr(_comp, "_last_summary_error", None),
            )
            self._hmwa_hygiene_stamp(
                attempt.agent, "session hygiene compression aborted",
                "AGENT_COMPRESSION_COOLDOWN", "hygiene compression abort activity stamp failed",
            )
            if not _hyg_fence_cancelled:
                # Force-redact: provider exception text may contain credentials; this reaches users.
                from agent.redact import redact_sensitive_text
                _err = redact_sensitive_text(getattr(_comp, "_last_summary_error", None) or "unknown error", force=True)
                await self._hmwa_hygiene_notify(
                    source, attempt.meta, "⚠️ Context compression aborted "
                    f"({_err}). No messages were dropped — "
                    "conversation is unchanged. Run /compress to retry, /reset for a clean "
                    "session, or check your auxiliary.compression model configuration.",
                    "compression-failure warning",
                )
        # Configured aux model failed, recovered on the main model: only the user can fix that config.
        elif _comp is not None and getattr(_comp, "_last_aux_model_failure_model", None):
            _aux_model = getattr(_comp, "_last_aux_model_failure_model", "")
            _aux_err = getattr(_comp, "_last_aux_model_failure_error", None) or "unknown error"
            await self._hmwa_hygiene_notify(
                source, attempt.meta, f"ℹ️ Configured compression model `{_aux_model}` "
                f"failed ({_aux_err}). Recovered using your main "
                "model — context is intact — but you may want to "
                "check `auxiliary.compression.model` in config.yaml.",
                "aux-model-fallback notice",
            )

    async def _hmwa_hygiene_codex_compaction(self, hs, plan, history, session_entry, session_key, _hyg_runtime):
        """codex app-server runtime: the real context is the server-side thread, not the transcript
        mirror. The detached-agent path would only rewrite the mirror and its finally-eviction
        would destroy the live thread (next turn starts blank), so use the cached agent's
        thread/compact/start and KEEP it cached."""
        from gateway.run import run_codex_hygiene_compaction
        # codex app-server runtime: the model's real context is the app-server's server-side thread, not the
        # transcript mirror. See #73503.
        _hyg_codex_auto = "native"
        _hyg_comp_cfg = hs.data.get("compression") if isinstance(hs.data, dict) else None
        if isinstance(_hyg_comp_cfg, dict):
            _hyg_codex_auto = str(_hyg_comp_cfg.get("codex_app_server_auto", "native") or "native")
        _hyg_codex_outcome = await run_codex_hygiene_compaction(
            self, session_key, session_entry.session_id, auto_mode=_hyg_codex_auto, history=history,
            approx_tokens=plan.approx_tokens, timeout_seconds=hs.total_ceiling_seconds,
            failure_cooldown_seconds=hs.failure_cooldown_seconds,
        )
        logger.info(
            "Session hygiene (codex app-server): %s (session=%s, mode=%s, ~%s tokens)",
            _hyg_codex_outcome, session_entry.session_id, _hyg_codex_auto, f"{plan.approx_tokens:,}",
        )

    async def _hmwa_hygiene_build_agent(self, _hyg_model, _hyg_runtime, session_entry):
        """Build the detached hygiene ``AIAgent`` with the live session's system prompt. Returns
        ``(agent, sync_session_db)``."""
        from gateway.run import _GATEWAY_HYGIENE_PLATFORM, _seed_hygiene_system_prompt
        from run_agent import AIAgent
        try:
            _hyg_session_row = await self._session_db.get_session(session_entry.session_id)
        except Exception as exc:
            _hyg_session_row = None
            logger.warning(
                "Session hygiene could not restore the system prompt for session %s: %s. "
                "Preserving an empty prompt so the live turn rebuilds it with its "
                "configured providers.", session_entry.session_id, exc, exc_info=True,
            )
        _hyg_session_db = getattr(self._session_db, "_db", self._session_db)
        # With compression.checkpoint_required on, load the memory provider so the checkpoint exists
        # before any mutation; otherwise keep the fast path (no provider init).
        from hermes_cli.config import load_config as _load_cfg
        from utils import is_truthy_value as _is_truthy

        _hyg_checkpoint_required = _is_truthy(
            ((_load_cfg() or {}).get("compression") or {}).get("checkpoint_required"), default=False,
        )
        _hyg_agent = AIAgent(
            **_hyg_runtime, model=_hyg_model, max_iterations=4, quiet_mode=True,
            skip_memory=not _hyg_checkpoint_required, enabled_toolsets=["memory"],
            session_id=session_entry.session_id, session_db=_hyg_session_db,
        )
        _seed_hygiene_system_prompt(_hyg_agent, _hyg_session_row)
        # A rebuilt (not retained) prompt is deliberately stale for every real gateway surface.
        _hyg_agent.platform = _GATEWAY_HYGIENE_PLATFORM
        return _hyg_agent, _hyg_session_db

    async def _hmwa_hygiene_detached_attempt(
        self, attempt, hs, plan, history, _hyg_msgs, _hyg_model, _hyg_runtime,
        source, session_entry, session_key, _quick_key, run_generation,
    ):
        """Run one detached hygiene compression attempt end to end; publishes the transcript to
        continue with (compressed or original) on ``attempt.history``."""
        from gateway.run import HygieneTurnHoldExceeded
        from agent.conversation_compression import CompressionCommitFence
        _hyg_agent, _hyg_session_db = await self._hmwa_hygiene_build_agent(_hyg_model, _hyg_runtime, session_entry)
        attempt.agent = _hyg_agent
        try:
            # Hygiene owns the session binding, so prefer in-place compaction over minting a
            # continuation child. Without a SessionDB this stays False.
            _hyg_agent.compression_in_place = True
            _bind_hyg_state = getattr(getattr(_hyg_agent, "context_compressor", None), "bind_session_state", None)
            if callable(_bind_hyg_state):
                _bind_hyg_state(_hyg_session_db, session_entry.session_id)
            # Never finalize on close() — that would end the live gateway session row.
            _hyg_agent._end_session_on_close = False
            _hyg_agent._print_fn = lambda *a, **kw: None

            loop = asyncio.get_running_loop()
            _hyg_commit_fence = CompressionCommitFence(total_ceiling_seconds=hs.total_ceiling_seconds)
            # Default executor (NOT self._get_executor): a hung summary must never occupy an
            # agent-work slot. MUST run in the caller's contextvars (multiplex secret scope).
            attempt.commit_fence = _hyg_commit_fence
            attempt.future = loop.run_in_executor(
                None,
                # But it MUST run inside the caller's contextvars: under multiplex_profiles the profile
                # secret scope / HERMES_HOME override live in ContextVars, and a bare run_in_executor worker
                # starts with an empty Context — the summary model's get_secret(<PROVIDER>_API_KEY) then
                # fails closed (UnscopedSecretError) and every hygiene compaction silently degrades to a
                # lossy truncation (#100849 bundle).
                copy_context().run,
                lambda: _hyg_agent._compress_context(
                    _hyg_msgs, "", approx_tokens=plan.approx_tokens, commit_fence=_hyg_commit_fence,
                ),
            )
            attempt.wait_started = time.monotonic()
            try:
                _compressed = await self._hmwa_hygiene_wait_for_summary(attempt, hs, session_entry)
            except HygieneTurnHoldExceeded:
                _compressed = await self._hmwa_hygiene_on_turn_hold(attempt, hs, session_entry, session_key, source)
            except asyncio.TimeoutError:
                _compressed = await self._hmwa_hygiene_on_timeout(attempt, hs, session_entry, session_key, source)
            except BaseException:
                self._hmwa_hygiene_on_unwind(attempt, hs, session_entry, session_key)
                raise

            await self._hmwa_hygiene_apply_result(
                attempt, hs, _compressed, history, plan, session_entry=session_entry,
                session_key=session_key, source=source, _quick_key=_quick_key,
                run_generation=run_generation,
            )
        finally:
            # Evict the cached agent so the next turn rebuilds its system prompt.
            self._evict_cached_agent(session_key)
            if not attempt.cleanup_deferred:
                await self._cleanup_agent_resources_off_loop(_hyg_agent, context="session hygiene")

    async def _hmwa_run_session_hygiene(
        self, event, source, session_entry, session_key, history, _quick_key, run_generation,
    ):
        """Auto-compress pathologically large transcripts before the agent starts so oversized
        histories don't cause repeated truncation/context failures. Token source: the API's
        prompt_tokens from the last turn, else a char/4 estimate."""
        from gateway.run import HygieneTurnHoldExceeded
        if not history or len(history) < 4:
            return history

        hs = await self._hmwa_hygiene_settings(source, session_key)
        if not hs.compression_enabled:
            return history
        plan = await self._hmwa_hygiene_plan(hs, history, session_entry, session_key)
        if not plan.needs_compress:
            return history

        attempt = self._HygieneAttempt(agent=None, meta=self._event_thread_metadata(event, source), history=history)
        try:
            _hyg_model, _hyg_runtime = self._resolve_session_agent_runtime(
                source=source, session_key=session_key,
                user_config=hs.data if isinstance(hs.data, dict) else None,
            )
            if str(_hyg_runtime.get("api_mode") or "").lower() == "codex_app_server":
                await self._hmwa_hygiene_codex_compaction(hs, plan, history, session_entry, session_key, _hyg_runtime)
            elif _hyg_runtime.get("api_key"):
                # Pass the FULL transcript (tool results included) as the agent loop does: filtering
                # to user/assistant starved the compressor (tool results are the bulk of context).
                _hyg_msgs = [m for m in history if m.get("role") in {"user", "assistant", "tool"}]
                if len(_hyg_msgs) >= 4:
                    await self._hmwa_hygiene_detached_attempt(
                        attempt, hs, plan, history, _hyg_msgs, _hyg_model, _hyg_runtime,
                        source, session_entry, session_key, _quick_key, run_generation,
                    )
        except HygieneTurnHoldExceeded:
            # Availability boundary, not a failure — already logged at INFO by the turn-hold handler.
            # Must not hit the generic "auto-compress failed" warning below: that log is how thinking-model
            # deployments read as permanently broken (#97963; surfaced by @686f6c61 in PR #99657).
            pass
        except Exception as e:
            logger.warning("Session hygiene auto-compress failed: %s", e)
        return attempt.history

    async def _hmwa_first_contact_notes(self, source, history, turn_sidecar_notes):
        """First-ever-message onboarding note + one-time 'no home channel' prompt (both only when
        the session has no history). Delivered on the user message (sidecar), NOT the ephemeral
        system prompt: present-on-turn-1/absent-on-turn-2 was a guaranteed prompt diff + rebuild."""
        from gateway.run import _hermes_home, _home_target_env_var, _load_gateway_config
        if history:
            return
        if not await self.async_session_store.has_any_sessions():
            _intro_note = (
                "[System note: This is the user's very first message ever. "
                "Briefly introduce yourself and mention that /help shows available commands. "
                "Keep the introduction concise -- one or two sentences max.]"
            )
            # onboarding.profile_build == "ask" (default) and not yet offered: swap the plain intro for
            # a consent-gated profile-build directive. Fires at most once.
            try:
                from agent.onboarding import (
                    PROFILE_BUILD_FLAG, is_seen, mark_seen, profile_build_directive,
                    profile_build_mode,
                )
                _onb_cfg = _load_gateway_config()
                if profile_build_mode(_onb_cfg) == "ask" and not is_seen(_onb_cfg, PROFILE_BUILD_FLAG):
                    turn_sidecar_notes.append(profile_build_directive().strip())
                    mark_seen(_hermes_home / "config.yaml", PROFILE_BUILD_FLAG)
                else:
                    turn_sidecar_notes.append(_intro_note)
            except Exception as _pb_err:
                logger.debug("Profile-build onboarding directive failed, using plain intro: %s", _pb_err)
                turn_sidecar_notes.append(_intro_note)

        # One-time prompt if no home channel is set (webhooks deliver to configured targets instead).
        if not source.platform or source.platform in (Platform.LOCAL, Platform.WEBHOOK):
            return
        platform_name = source.platform.value
        env_key = _home_target_env_var(platform_name)
        # Multiplex: the home channel may live only in the profile secret scope, not os.environ.
        home_env = ""
        if env_key:
            with suppress(Exception):
                from agent.secret_scope import get_secret
                home_env = (get_secret(env_key) or "").strip()
            home_env = home_env or (os.getenv(env_key) or "").strip()
        # Also honor in-memory / yaml home_channel on this platform.
        with suppress(Exception):
            if not home_env and self.config.get_home_channel(source.platform):
                home_env = "set"
        # Secondary-profile platforms may only exist under that profile's config — re-read in scope.
        if not home_env:
            with suppress(Exception):
                from gateway.config import load_gateway_config as _lgc
                prof = (getattr(source, "profile", None) or "").strip()
                if prof and prof != "default" and _lgc().get_home_channel(source.platform):
                    home_env = "set"
        if not home_env:
            # Slack routes every command through the parent `/hermes`; bare `/sethome` would fail.
            sethome_cmd = "/hermes sethome" if source.platform == Platform.SLACK else "/sethome"
            await self._deliver_platform_notice(
                source, f"📬 No home channel is set for {platform_name.title()}. "
                f"A home channel is where Hermes delivers cron job results and cross-platform "
                f"messages.\n\nType {sethome_cmd} to make this chat your home channel, or ignore "
                f"to skip.",
            )

    def _hmwa_apply_message_timestamp(self, event, message_text):
        """Capture the platform event time as message metadata and keep the persisted transcript
        clean (strip any leading timestamp prefix) regardless of the toggle; only the in-context
        RENDER is gated behind gateway.message_timestamps.enabled (default OFF)."""
        from gateway.run import _load_gateway_config, _message_timestamps_enabled
        persist_user_message = None
        persist_user_timestamp = None
        try:
            from hermes_time import get_timezone as _get_evt_tz
            from gateway.message_timestamps import (
                coerce_message_timestamp as _coerce_msg_ts,
                render_user_content_with_timestamp as _render_msg_ts,
                strip_leading_message_timestamps as _strip_msg_ts,
            )
            _evt_tz = _get_evt_tz()
            if message_text and isinstance(message_text, str):
                _clean_message_text, _embedded_ts = _strip_msg_ts(message_text, tz=_evt_tz)
                persist_user_message = _clean_message_text
                _event_epoch = _coerce_msg_ts(getattr(event, "timestamp", None), tz=_evt_tz)
                persist_user_timestamp = _event_epoch if _event_epoch is not None else _embedded_ts
                if _message_timestamps_enabled(_load_gateway_config()):
                    message_text = _render_msg_ts(_clean_message_text, persist_user_timestamp, tz=_evt_tz)
                else:
                    # Toggle off: the model sees the clean message; timestamp stored for later opt-in.
                    message_text = _clean_message_text
        except Exception as _ts_err:
            logger.debug("Message timestamp injection failed (non-fatal): %s", _ts_err)
        return message_text, persist_user_message, persist_user_timestamp

    async def _hmwa_stop_typing_for_turn(self, event, source):
        """Stop the typing indicator (never raises). Slack AI status is scoped to a thread/
        workspace, so preserve the routing metadata used by the response delivery path."""
        with suppress(Exception):
            _typing_adapter = self._adapter_for_source(source)
            _kind = type(_typing_adapter)
            if _typing_adapter and callable(getattr(_kind, "_stop_typing_with_metadata", None)):
                await _typing_adapter._stop_typing_with_metadata(source.chat_id, self._event_thread_metadata(event, source))
            elif _typing_adapter and callable(getattr(_kind, "stop_typing", None)):
                await _typing_adapter.stop_typing(source.chat_id)

    async def _hmwa_shape_agent_response(
        self, agent_result, source, history, session_entry, session_key,
        _quick_key, run_generation, _run_start_session_id, _platform_name, _msg_start_time,
    ):
        """Turn the raw agent result into the outbound text: sentinel/silence handling, response
        logging, resume-pending clear, empty-response normalization, and identity-guarded
        post-compression session_id propagation. Returns
        ``(response, _intentional_silence, agent_messages)``."""
        from gateway.run import (
            _is_gateway_hidden_reasoning_incomplete_turn, _normalize_empty_agent_response,
            _sanitize_gateway_final_response, _should_clear_resume_pending_after_turn,
        )
        response = agent_result.get("final_response") or ""
        # Hidden-reasoning-only retry exhaustion: the loop's sentinel text doubles as final_response
        # and would be delivered verbatim (peer agents would ingest it as a completed turn).
        if _is_gateway_hidden_reasoning_incomplete_turn(agent_result):
            response = ""
        _intentional_silence = self._is_intentional_silence(agent_result, response)

        # "(empty)" = the model produced no visible content after exhausting all retries.
        if response == "(empty)" and not _intentional_silence:
            response = (
                "⚠️ The model returned no response after processing tool results. This can happen "
                "with some models — try again or rephrase your question."
            )
        agent_messages = agent_result.get("messages", [])
        logger.info(
            "response ready: platform=%s chat=%s time=%.1fs api_calls=%d response=%d chars",
            _platform_name, source.chat_id or "unknown",
            time.time() - _msg_start_time, agent_result.get("api_calls", 0), len(response),
        )

        # Successful turn: clear the consecutive-restart stuck-loop counter and resume_pending (set
        # by drain-timeout shutdown) so later messages don't get the restart-interruption note.
        if session_key and _should_clear_resume_pending_after_turn(agent_result):
            await self._clear_restart_failure_count(session_key)
            try:
                await self.async_session_store.clear_resume_pending(session_key)
            except Exception as _e:
                logger.debug("clear_resume_pending failed for %s: %s", session_key, _e)

        # Normalize empty responses: surface errors, partial failures, and work-without-text.
        # Fix for #18765.
        if not _intentional_silence:
            response = _normalize_empty_agent_response(agent_result, response, history_len=len(history))
            response = _sanitize_gateway_final_response(source.platform, response)

        # The agent thread already updated the contextvar; propagate to SessionEntry + _save() only
        # if the binding still points at the session this run was launched against.
        if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
            if session_entry.session_id == _run_start_session_id:
                session_entry.session_id = agent_result["session_id"]
                # The held turn lease follows the rotation (persistence writes to the NEW id).
                self._rebind_turn_lease(_quick_key, run_generation, session_entry.session_id)
                await self.async_session_store._save()
                await self.async_session_store._record_gateway_session_peer(
                    session_entry.session_id, session_key, source,
                )
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding, source, session_entry, reason="agent-result-compression",
                )
            else:
                logger.info(
                    "Skipping agent-result session split sync for %s because the session binding "
                    "moved from %s to %s before compression finished",
                    session_key or "?", _run_start_session_id, session_entry.session_id,
                )
        return response, _intentional_silence, agent_messages

    def _hmwa_prepend_reasoning(self, agent_result, response, source, _intentional_silence):
        """Prepend the last reasoning block when show_reasoning is on for this platform. Mattermost
        requires an explicit per-platform opt-in (scratch text, not final-answer content)."""
        from gateway.run import _load_gateway_config, _platform_config_key, _resolve_gateway_display_bool
        try:
            _show_reasoning_effective = _resolve_gateway_display_bool(
                _load_gateway_config(), _platform_config_key(source.platform), "show_reasoning",
                default=bool(getattr(self, "_show_reasoning", False)), platform=source.platform,
                require_platform_override_for={Platform.MATTERMOST},
            )
        except Exception:
            _show_reasoning_effective = (
                False if source.platform == Platform.MATTERMOST else getattr(self, "_show_reasoning", False)
            )
        last_reasoning = agent_result.get("last_reasoning")
        if not (_show_reasoning_effective and response and not _intentional_silence and last_reasoning):
            return response
        from gateway.stream_consumer_fences import escape_code_fences_for_display
        # Collapse long reasoning to keep messages readable
        lines = last_reasoning.strip().splitlines()
        if len(lines) > 15:
            display_reasoning = "\n".join(lines[:15]) + f"\n_... ({len(lines) - 15} more lines)_"
        else:
            display_reasoning = last_reasoning.strip()
        # Per-platform render style: Discord defaults to "-# " subtext, others keep the code block.
        try:
            from gateway.display_config import resolve_display_setting
            _reasoning_style = resolve_display_setting(
                _load_gateway_config(), _platform_config_key(source.platform), "reasoning_style", "code",
            )
        except Exception:
            _reasoning_style = "code"
        _quote = self._REASONING_QUOTE_STYLES.get(_reasoning_style)
        if _quote:
            header, prefix, empty = _quote
            _quoted = "\n".join(f"{prefix}{ln}" if ln else empty for ln in display_reasoning.splitlines())
            return f"{header}\n{_quoted}\n\n{response}"
        # Escape ``` inside reasoning so inner fences don't break the outer code block.
        display_reasoning = escape_code_fences_for_display(display_reasoning)
        return f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"

    def _hmwa_runtime_footer_line(self, agent_result, source, _turn_seconds):
        """Runtime-metadata footer for the FINAL message of the turn; off by default
        (display.runtime_footer.enabled=false)."""
        from gateway.run import _load_gateway_config, _platform_config_key, _terminal_scope_cwd
        try:
            from gateway.runtime_footer import build_footer_line as _bfl
            return _bfl(
                user_config=_load_gateway_config(),
                platform_key=_platform_config_key(source.platform), model=agent_result.get("model"),
                context_tokens=agent_result.get("last_prompt_tokens", 0) or 0,
                context_length=agent_result.get("context_length") or None,
                cwd=_terminal_scope_cwd(""), turn_seconds=_turn_seconds,
            )
        except Exception as _footer_err:
            logger.debug("runtime_footer build failed: %s", _footer_err)
            return ""

    async def _hmwa_post_turn_hooks(self, hook_ctx, agent_result, response):
        """agent:end hook, process-watcher scheduling, and watch-notification drain."""
        await self.hooks.emit("agent:end", {
            **hook_ctx, "response": (response or "")[:500], "model": agent_result.get("model", ""),
            "provider": agent_result.get("provider", ""),
        })

        # Pending process watchers (check_interval on background processes)
        try:
            from tools.process_registry import process_registry
            # Detach the batch atomically (reassign, not clear()) so concurrent appends aren't dropped.
            watchers = process_registry.pending_watchers
            process_registry.pending_watchers = []
            for i, watcher in enumerate(watchers):
                asyncio.create_task(self._run_process_watcher(watcher))
                if i % 100 == 99:
                    await asyncio.sleep(0)
        except Exception as e:
            logger.error("Process watcher setup error: %s", e)

        # Drain watch notifications that arrived during the run; the queue also carries process /
        # async-delegation completions owned elsewhere — inject only watch-type events.
        try:
            from tools.process_registry import process_registry as _pr
            await self._drain_watch_notifications(_pr.completion_queue)
        except Exception as e:
            logger.debug("Watch queue drain error: %s", e)

    def _hmwa_classify_turn_failure(self, agent_result, history, session_entry):
        """Classify a finished turn for transcript persistence. Returns
        ``(agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure)``.

        Context-overflow failures must NOT persist the user message (session would grow and
        reproduce the failure forever); transient failures (429/timeout/5xx) DO."""
        from gateway.run import _is_gateway_hidden_reasoning_incomplete_turn
        from gateway.run_turn import is_context_overflow_failure_result
        # Save the full conversation to the transcript, including tool calls. This preserves the complete
        # agent loop (tool_calls, tool results, intermediate reasoning) so sessions can be resumed with full
        # context and transcripts are useful for debugging and training data. IMPORTANT: For
        # context-overflow failures (compression exhausted, generic 400 on large sessions) we must NOT
        # persist the user's message — doing so would grow the session further and cause the same failure on
        # the next attempt, an infinite loop. (#1630, #9893) Transient failures (429, timeout, connection
        # error, provider 5xx) are different: the session is not oversized, and silently dropping the user
        # message causes severe context loss on retry — the agent forgets what was just asked. Persist the
        # user turn so the conversation is preserved. (#7100)
        agent_failed_early = bool(agent_result.get("failed"))
        hidden_reasoning_incomplete = _is_gateway_hidden_reasoning_incomplete_turn(agent_result)
        is_context_overflow_failure = is_context_overflow_failure_result(agent_result, len(history))
        if is_context_overflow_failure:
            logger.info(
                "Skipping transcript persistence for context-overflow "
                "failure in session %s to prevent session growth loop.", session_entry.session_id,
            )
        elif agent_failed_early:
            logger.info(
                "Transient agent failure in session %s — persisting user "
                "message so conversation context is preserved on retry.", session_entry.session_id,
            )
        elif hidden_reasoning_incomplete:
            logger.warning(
                "Suppressing hidden-reasoning-only incomplete gateway turn for session %s: %s",
                session_entry.session_id, agent_result.get("error", "processing incomplete"),
            )
        return agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure

    async def _hmwa_compression_exhaustion_reset(
        self, agent_result, response, session_entry, session_key, source,
    ):
        """Auto-reset a permanently oversized session so the next message starts fresh instead of
        replaying the oversized context forever. Never on a lock-contended defer — that is the
        OPPOSITE case (a concurrent path holds the lock and is shrinking it). Returns
        ``(response, session_entry)``."""
        # When compression is exhausted, the session is permanently too large to process. (#9893) Never wipe
        # the session for that — retry-next-message semantics apply (#69870 lock-skip consumer; salvaged
        # from #49874).
        if agent_result.get("compression_deferred"):
            logger.info(
                "Compression deferred for session %s — the compression "
                "lock is held by a concurrent compressor. Keeping the "
                "session intact; the next message retries normally.",
                session_entry.session_id if session_entry else "?",
            )
        elif agent_result.get("compression_exhausted") and session_entry and session_key:
            logger.info("Auto-resetting session %s after compression exhaustion.", session_entry.session_id)
            new_entry = await self.async_session_store.reset_session(session_key)
            self._evict_cached_agent(session_key)
            # Conversation boundary: the funnel clears every conversation-scoped per-session dict.
            self._clear_conversation_scope(session_key, reason="compression_exhausted_reset")
            if new_entry is not None:
                # Re-point the Telegram topic binding at the fresh session, or the binding-heal walk
                # switches the next message back onto the bloated child and re-triggers exhaustion
                # forever. No-op on non-topic lanes.
                # Compression rotated session_entry.session_id to the oversized compressed child earlier
                # this turn (the agent-result sync above), and that _sync also rewrote the (chat_id,
                # thread_id) -> bloated-child binding. reset_session swaps in a clean, parentless session,
                # but without re-syncing the binding the next inbound message in this topic gets
                # switch_session'd back onto the bloated child by the binding-heal walk, reloads the
                # oversized transcript, and re-triggers compression exhaustion forever (#35809 — regression
                # of the #9893/#10063 auto-reset).
                session_entry = new_entry
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding, source, session_entry, reason="compression-exhausted-reset",
                )
            response = (response or "") + (
                "\n\n🔄 Session auto-reset — the conversation exceeded the maximum context size and "
                "could not be compressed further. Your next message will start a fresh session."
            )
        return response, session_entry

    @staticmethod
    def _hmwa_user_transcript_entry(event, prepared, ts):
        """Transcript row for the inbound user turn (clean text + event time when captured)."""
        # Transient failure (429/timeout/5xx): persist only the user message so the next message can load a
        # transcript that reflects what was said. Skip the assistant error text since it's a
        # gateway-generated hint, not model output. Hidden- reasoning-only incomplete turns follow the same
        # persistence rule so peer-agent channels don't ingest them as completed assistant turns. (#7100,
        # #51628)
        _user_entry = {
            "role": "user",
            "content": (
                prepared.persist_user_message if prepared.persist_user_message is not None
                else prepared.message_text
            ),
            "timestamp": prepared.persist_user_timestamp if prepared.persist_user_timestamp is not None else ts,
        }
        if prepared.persist_user_display_kind:
            _user_entry["display_kind"] = prepared.persist_user_display_kind
        if prepared.persistence_owner:
            _user_entry["display_metadata"] = {"gateway_input_owner": prepared.persistence_owner}
        if getattr(event, "message_id", None):
            _user_entry["message_id"] = str(event.message_id)
        return _user_entry

    async def _hmwa_persist_turn_transcript(
        self, *, event, source, session_entry, session_key, agent_result, agent_messages,
        prepared, response, agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure,
    ):
        """Persist this turn to the transcript (session_meta on first turn, user-only on transient
        failure, nothing on context overflow), update last_prompt_tokens, and re-baseline the
        cached agent's message count."""
        from gateway.run import _resolve_gateway_model
        ts = time.time()  # Unix epoch float — consistent with DB storage
        store = self.async_session_store
        sid = session_entry.session_id
        history = prepared.history
        # The agent already persisted this turn's rows (codex app-server reports agent_persisted=True
        # too); skip the DB write. Default = a session DB exists; non-persisting runtimes pass False.
        # The agent already persisted these messages to SQLite via _flush_messages_to_session_db(), so skip
        # the DB write here to prevent the duplicate-write bug (#860 / #42039). This holds for the codex
        # app-server runtime too: although it early-returns and bypasses conversation_loop's per-step
        # flushes, it flushes its own projected assistant/tool messages before returning and reports
        # agent_persisted=True (see agent/codex_runtime.py). Reading the flag (default = self._session_db is
        # not None) keeps the persistence contract explicit and lets any future non-persisting runtime opt
        # into a gateway-side write by returning False.
        agent_persisted = agent_result.get("agent_persisted", self._session_db is not None)
        _user_row = self._hmwa_user_transcript_entry(event, prepared, ts)

        if is_context_overflow_failure:
            pass  # Skip all transcript writes — don't grow a broken session
        else:
            if not history:
                # Fresh session: the tool definitions (as sent in the API request) make the transcript
                # self-describing.
                await store.append_to_transcript(sid, {
                    "role": "session_meta",
                    "tools": agent_result.get("tools", []) or [],
                    "model": _resolve_gateway_model(),
                    "platform": source.platform.value if source.platform else "",
                    "timestamp": ts,
                })
            if agent_failed_early or hidden_reasoning_incomplete:
                # Transient failure / hidden-reasoning incomplete: persist only the user message (the
                # assistant error text is a gateway hint, not model output). Dedupe on platform
                # message_id (Telegram retries after transient failures).
                if event.message_id and await store.has_platform_message_id(sid, str(event.message_id)):
                    logger.info(
                        "Skipping duplicate user turn (message_id=%s) in session %s",
                        event.message_id, sid,
                    )
                else:
                    await store.append_to_transcript(sid, _user_row, skip_db=agent_persisted)
            else:
                # Only the NEW messages: history_offset (what the agent saw), not len(history), which
                # counts session_meta entries stripped before the agent saw them.
                history_len = agent_result.get("history_offset", len(history))
                new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []
                if not new_messages:
                    # Edge case: fall back to simple user/assistant rows.
                    await store.append_to_transcript(sid, _user_row, skip_db=agent_persisted)
                    if response:
                        await store.append_to_transcript(
                            sid, {"role": "assistant", "content": response, "timestamp": ts},
                            skip_db=agent_persisted,
                        )
                else:
                    # Attach the inbound platform message_id to the first user entry so platform-level
                    # quote-resolution (e.g. Yuanbao) can find earlier @bot messages by original id.
                    _user_msg_id_attached = False
                    for msg in new_messages:
                        if msg.get("role") == "system":
                            continue  # rebuilt each run
                        entry = {**msg, "timestamp": ts}
                        if (
                            not _user_msg_id_attached
                            and msg.get("role") == "user"
                            and event.message_id
                            and "message_id" not in entry
                        ):
                            entry["message_id"] = str(event.message_id)
                            _user_msg_id_attached = True
                        await store.append_to_transcript(sid, entry, skip_db=agent_persisted)

        # The agent persists token counts/model itself; keep only last_prompt_tokens for hygiene.
        await store.update_session(
            session_entry.session_key, last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
            touch_activity=not bool(getattr(event, "internal", False)),
        )

        # Re-baseline the cached agent's message_count now that ALL of this turn's writes are done:
        # the coherence guard snapshots at agent-BUILD time, so our own writes would otherwise
        # trigger a rebuild next turn (destroying prompt caching).
        await self._refresh_agent_cache_message_count(session_key, sid)

    async def _hmwa_deliver_turn_response(
        self, event, source, session_entry, session_key, run_generation,
        agent_result, agent_messages, response, _footer_line, _intentional_silence,
    ):
        """Final delivery decisions: intentional silence, voice reply, streamed-turn media/footer.
        Returns the text for the adapter to send, or ``None`` when already delivered."""
        # Intentional silence is a delivery decision: the [SILENT] turn stays persisted (alternation).
        if _intentional_silence:
            logger.info("Suppressing intentional silence marker for session %s", session_entry.session_id)
            response = ""

        adapter = self._adapter_for_source(source)
        # Auto voice reply (TTS audio before the text) unless streaming TTS already delivered audio.
        _streaming_tts_done = adapter is not None and bool(
            getattr(adapter, "_streaming_tts_turn_completed", lambda *_a, **_k: False)(session_key, run_generation)
        )
        if not _streaming_tts_done and self._should_send_voice_reply(
            event, response, agent_messages, already_sent=bool(agent_result.get("already_sent")),
        ):
            await self._send_voice_reply(event, response)

        # Streamed responses still need MEDIA: files delivered (chunks carry the tags verbatim). Never
        # skip when the agent failed: the error text is new content streaming didn't show.
        if agent_result.get("already_sent") and not agent_result.get("failed"):
            if response and adapter:
                await self._deliver_media_from_response(response, event, adapter)
            # Streaming delivered the body, but the footer was held back (`not already_sent` gate).
            if _footer_line and adapter:
                try:
                    await adapter.send(source.chat_id, _footer_line, metadata=self._event_thread_metadata(event, source))
                except Exception as _e:
                    logger.debug("trailing footer send failed: %s", _e)
            # Return None so the body isn't sent twice; stash the delivered text on the event for the
            # /loop and /goal hooks that read the return value.
            with suppress(Exception):
                event._streamed_final_response = str(response or "")
            return None

        return response

    async def _hmwa_agent_error_reply(self, e, event, source, session_entry, session_key, prepared):
        """``except Exception`` body of the agent turn: stop typing, log, persist the inbound user
        turn once, and build the sanitized user-facing error reply."""
        # Retain Slack thread/workspace routing so a failed turn cannot leave its status visible.
        await self._hmwa_stop_typing_for_turn(event, source)
        logger.exception("Agent error in session %s", session_key)
        # Replay can coalesce inputs; only this input's durable marker establishes ownership.
        try:
            if prepared.message_text is not None and session_entry is not None:
                _owned = await self.async_session_store.has_input_owner(
                    prepared.persistence_session_id, prepared.persistence_owner,
                )
                if not _owned:
                    await self.async_session_store.append_to_transcript(
                        session_entry.session_id, self._hmwa_user_transcript_entry(event, prepared, time.time()),
                    )
        except Exception:
            logger.debug("Failed to persist inbound user message after agent exception", exc_info=True)
        # Never expose raw exception types/messages to end users (info-leakage risk).
        status_code = getattr(e, "status_code", None)
        status_hint = self._STATUS_HINTS.get(status_code, "")
        if status_code == 429:
            # Plan usage limit (resets on a schedule) vs a transient rate limit
            _err_json = {}
            with suppress(Exception):
                _err_json = e.response.json().get("error", {})
            if not isinstance(_err_json, dict):
                _err_json = {}
            _resets_in = _err_json.get("resets_in_seconds")
            if _err_json.get("type") != "usage_limit_reached":
                status_hint = " You are being rate-limited. Please wait a moment and try again."
            elif _resets_in and _resets_in > 0:
                import math
                status_hint = f" Your plan's usage limit has been reached. It resets in ~{math.ceil(_resets_in / 3600)}h."
            else:
                status_hint = " Your plan's usage limit has been reached. Please wait until it resets."
        elif status_code in {400, 500}:
            # 400/500 on a large session: context overflow / payload too large.
            if len(prepared.history) > 50:
                return (
                    "⚠️ Session too large for the model's context window.\nUse /compact to "
                    "compress the conversation, or /reset to start fresh."
                )
            elif status_code == 400:
                status_hint = " The request was rejected by the API."
        return (
            f"Sorry, I encountered an unexpected error.{status_hint}\n"
            "Try again or use /reset to start a fresh session."
        )

    def _hmwa_discard_stale_result(self, source, _quick_key, run_generation):
        """A newer run generation superseded this turn: drop its deferred post-delivery callback."""
        logger.info(
            "Discarding stale agent result for %s — generation %d is no longer current",
            _quick_key or "?", run_generation,
        )
        self._pop_post_delivery_callback(self._adapter_for_source(source), _quick_key, run_generation)

    async def _hmwa_prepare_turn(self, event, source, session_entry, session_key, _quick_key, run_generation):
        """Everything between session resolution and the agent run: session open, task-local env,
        context prompt, sidecar notes, turn lease, transcript load + hygiene, inbound text. Returns
        ``(_PreparedTurn, env_tokens)``; a ``str`` first element is a reply to send instead of
        running (history unreadable); ``None`` drops the turn (inbound text rejected)."""
        from gateway.run import _load_gateway_config
        _was_auto_reset, _is_new_session = await self._hmwa_open_session(session_entry, session_key, source)
        context = build_session_context(source, self.config, session_entry)
        # Session context variables for tools (task-local, concurrency-safe)
        _session_env_tokens = self._set_session_env(context)
        # Self-injected turns (MessageEvent(internal=True)) persist with a DB-only display_kind so
        # UIs render timeline notices, not user bubbles; role/content untouched.
        persist_user_display_kind = "internal_notification" if getattr(event, "internal", False) else None
        _redact_pii = False  # privacy.redact_pii, re-read per message
        with suppress(Exception):
            _redact_pii = bool((_load_gateway_config().get("privacy") or {}).get("redact_pii", False))

        # The context prompt render is pinned per session, keyed by a hash of the renderer inputs, so
        # the system prompt cannot drift turn-over-turn; a miss (thread rename, /sethome) re-renders.
        context_prompt = self._pinned_session_context_prompt(context, _redact_pii, session_key)

        # Per-turn notes ride the user message via the api_content sidecar, NOT context_prompt
        # (appending to the ephemeral system prompt forced a full agent rebuild).
        turn_sidecar_notes: List[str] = []
        if _was_auto_reset:
            await self._hmwa_deliver_auto_reset_notice(session_entry, source, turn_sidecar_notes)

        # Auto-load bound skill(s) only on NEW sessions; ongoing ones carry the content in history.
        _auto = getattr(event, "auto_skill", None)
        if _is_new_session and _auto:
            self._hmwa_auto_load_skills(event, _auto, _quick_key, session_key)

        await self._hmwa_acquire_turn_lease(_quick_key, run_generation, session_entry, _session_env_tokens)

        # A turn becomes durable recovery work only after it owns the per-session lease; marking
        # earlier would falsely recover a message that never began processing.
        await self._mark_durable_active_turn(event, session_entry.session_key)

        # An unreadable store is not an empty conversation: stop before the agent invents continuity
        # from []. Restore task-local context here (before the broad cleanup finally).
        try:
            history = await self.async_session_store.load_transcript(session_entry.session_id)
            history = await self._hmwa_run_session_hygiene(
                event, source, session_entry, session_key, history, _quick_key, run_generation,
            )
        except TranscriptReadError:
            self._clear_session_env(_session_env_tokens)
            return (
                "⚠️ This session's history is temporarily unavailable, so this message was not "
                "processed. Ask the operator to inspect state.db, then resend after it is healthy. "
                "Use /reset only if you intentionally want to start a new conversation."
            ), _session_env_tokens

        await self._hmwa_first_contact_notes(source, history, turn_sidecar_notes)

        # Voice channel state rides the user message ONLY when changed (in the system prompt it
        # forced a rebuild + prompt-cache re-key per message).
        _vc_note = self._voice_channel_sidecar_note(event, source, session_key)
        if _vc_note:
            turn_sidecar_notes.append(_vc_note)

        # Auto-analyze user images so the model gets a description plus the local path.
        message_text = await self._prepare_profile_scoped_inbound_message_text(
            event=event, source=source, history=history, session_key=session_key,
        )
        if message_text is None:
            return None, _session_env_tokens

        message_text, persist_user_message, persist_user_timestamp = (
            self._hmwa_apply_message_timestamp(event, message_text)
        )

        # Stage the notes (one-shot; consumed in run_sync) AFTER the early-out so an aborted turn
        # cannot leak them into the next turn.
        if turn_sidecar_notes and session_key:
            self._set_pending_turn_sidecar_notes(session_key, turn_sidecar_notes)

        # Bind this run generation to the adapter so deferred post-delivery callbacks are released
        # by the run that registered them.
        self._bind_adapter_run_generation(self._adapter_for_source(source), session_key, run_generation)
        # Delivery IDs are only unique in their transport namespace. Keyless turns
        # need their own identity, even when another process writes to this session.
        import uuid
        namespace = [source.platform.value, source.profile, source.scope_id,
                     source.chat_id, source.thread_id, str(event.message_id)]
        owner = (str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(namespace)))
                 if event.message_id else str(uuid.uuid4()))
        return self._PreparedTurn(
            history, context_prompt, message_text, persist_user_message, persist_user_timestamp,
            persist_user_display_kind, session_entry.session_id, owner,
        ), _session_env_tokens
