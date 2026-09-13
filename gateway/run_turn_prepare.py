"""Gateway turn prepare phases; inherited by GatewayTurnMixin."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import dataclasses
import inspect
import json
import os
import queue
import threading
import time
from agent.i18n import t
from agent.session_activity import format_iteration_progress
from contextlib import nullcontext, suppress
from contextvars import copy_context
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.session import (
    SessionSource, _session_key_namespace, build_channel_continuity_note,
    build_session_context,
)
from gateway.session_transcript import TranscriptReadError
from gateway.turn_context import TurnContext
from gateway.turn_lease import DEFAULT_LEASE_WAIT, TurnLeaseTimeoutError
from hermes_constants import get_hermes_home_override
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils import base_url_hostname

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayTurnPrepareMixin:
    def _resolve_session_agent_runtime(
        self, *, source: Optional[SessionSource] = None, session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Resolve model/runtime for a session.

        Priority (highest first): session ``/model`` → ``channel_overrides`` → global config/env
        (``_resolve_gateway_model(user_config)`` and default provider resolution)."""
        from gateway.run import (
            _credential_pool_for_provider, _get_channel_override, _resolve_gateway_model,
            _resolve_runtime_agent_kwargs, _resolve_runtime_agent_kwargs_for_provider,
        )
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(self, source) if source is not None else None
        if policy is not None:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            from gateway.run import _runtime_agent_kwargs
            from gateway.session_policy import launch_key
            from hermes_cli.runtime_provider_custom import _resolve_named_custom_runtime
            from gateway.session_authorities import active_authority
            authority = active_authority(self)
            frozen = policy.config(authority)
            key = launch_key(authority, policy)
            import json
            runtime = _resolve_named_custom_runtime(requested_provider=policy.provider,
                explicit_api_key=key, explicit_base_url=json.loads(policy.request_json).get('base_url'),
                target_model=policy.model, config=frozen)
            if runtime is None:
                if key is None:
                    key = frozen.get('model', {}).get('api_key')
                runtime = resolve_runtime_provider(requested=policy.provider,
                    explicit_api_key=key, explicit_base_url=policy.base_url, target_model=policy.model)
            return policy.model, _runtime_agent_kwargs(runtime)
        skey = self._resolve_session_key_or_none(source, session_key)

        model = _resolve_gateway_model(user_config)
        if skey:
            self._rehydrate_session_model_override(skey)
        _override_state = self._peek_session_state(skey) if skey else None
        override = _override_state.conversation.model_override if _override_state else None
        if override:
            override_model = override.get("model", model)
            override_runtime = {
                k: override.get(k) for k in (
                    "provider", "requested_provider", "api_key", "base_url", "api_mode",
                    "max_tokens", "credential_pool", "request_overrides", "capabilities",
                )
            }
            override_runtime["capabilities"] = dict(override_runtime["capabilities"] or {})
            if override_runtime.get("api_key"):
                if override_runtime.get("credential_pool") is None:
                    override_runtime["credential_pool"] = _credential_pool_for_provider(override.get("provider"))
                logger.debug(
                    "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                    skey or "", model, override_model, override_runtime.get("provider"),
                )
                return override_model, override_runtime
            # No api_key on the override: env-based resolution below, override model/provider on top.
            logger.debug(
                "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
                skey or "", model, override_model,
            )
        else:
            logger.debug(
                "No session model override: session=%s config_model=%s override_keys=%s",
                skey or "", model,
                [
                    _key for _key, _st in list(self._sessions_map().items())
                    if _st.conversation.model_override is not None
                ][:5] or "[]",
            )

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            logger.info("Runtime provider supplied explicit model override: %s -> %s", model, runtime_model)
            model = runtime_model

        cfg = getattr(self, "config", None)  # getattr: bare object.__new__ test runners
        if cfg and source is not None:
            ch = _get_channel_override(
                cfg, source.platform, str(source.chat_id) if source.chat_id else "",
                thread_id=str(source.thread_id) if getattr(source, "thread_id", None) else None,
                parent_id=str(source.parent_chat_id) if getattr(source, "parent_chat_id", None) else None,
            )
            if ch:
                if ch.model:
                    model = ch.model
                if ch.provider:
                    runtime_kwargs = _resolve_runtime_agent_kwargs_for_provider(ch.provider)
                    ch_runtime_model = runtime_kwargs.pop("model", None)
                    # Adopt the provider's bundled model only when the override named none.
                    if ch_runtime_model and not ch.model:
                        model = ch_runtime_model

        if override and skey:
            model, runtime_kwargs = self._apply_session_model_override(skey, model, runtime_kwargs)

        # Provider resolved but no model.default (`hermes auth add` without `hermes model`): use the
        # provider's first catalog model.
        if not model and runtime_kwargs.get("provider"):
            with suppress(Exception):
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s", model, runtime_kwargs["provider"],
                    )

        # Final safety net: an empty model (transient config-cache miss) makes every API call 400 and
        # the session goes silent — reuse the last model resolved for this session, else process-wide.
        if not model:
            _lr_state = self._peek_session_state(skey) if skey else None
            _lr_star = self._peek_session_state("*")
            _recovered = (
                (_lr_state.conversation.last_resolved_model if _lr_state else "")
                or (_lr_star.conversation.last_resolved_model if _lr_star else "")
            )
            if _recovered:
                logger.warning(
                    "Empty model resolved for session=%s — recovering "
                    "last-known-good model %s (config read likely returned "
                    "empty; see #35314)", skey or "", _recovered,
                )
                model = _recovered
        else:
            # Cache the good resolution for future recovery turns.
            if skey:
                self._session_state(skey).conversation.last_resolved_model = model
            self._session_state("*").conversation.last_resolved_model = model

        return model, runtime_kwargs

    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        """Effective model/runtime config for one turn. With `/fast` priority on, fast-mode
        ``request_overrides`` are deep-merged OVER the per-provider ones so both reach the model."""
        from gateway.run import _deep_merge_request_overrides
        from hermes_cli.models import resolve_fast_mode_overrides
        # Tests bind this method onto bare namespaces, so no class-level tables here.
        runtime = {
            k: runtime_kwargs.get(k) for k in (
                "api_key", "base_url", "provider", "requested_provider", "api_mode", "command", "args",
                "credential_pool", "max_tokens", "capabilities",
            )
        }
        runtime["args"] = list(runtime["args"] or [])
        runtime["capabilities"] = dict(runtime["capabilities"] or {})
        base_request_overrides = dict(runtime_kwargs.get("request_overrides") or {})
        route = {
            "model": model,
            "runtime": runtime,
            "signature": (
                model, runtime["provider"], runtime["requested_provider"], runtime["base_url"],
                runtime["api_mode"], runtime["command"], tuple(runtime["args"]),
            ),
        }
        if getattr(self, "_service_tier", None) != "priority":
            # None / auto / cold: the bounded window is applied per request by agent.fast_mode.
            route["request_overrides"] = base_request_overrides
            return route
        try:
            overrides = resolve_fast_mode_overrides(
                route["model"], provider=runtime["provider"], base_url=runtime["base_url"],
            )
        except Exception:
            overrides = None
        # Fast-mode keys (service_tier / speed) are top-level and don't collide with extra_body.
        route["request_overrides"] = _deep_merge_request_overrides(base_request_overrides, overrides or {})
        return route

    def _sync_session_model_from_agent(self, session_id: str, agent: Any) -> None:
        """Persist the runtime model/provider a gateway turn actually used (provider fallback can
        switch them after the row was created). Runs in the ``run_sync`` executor thread, so it
        uses the sync ``SessionDB`` (``_db``), not the AsyncSessionDB forwarder."""
        if not session_id or agent is None or self._session_db is None:
            return
        model = getattr(agent, "model", None)
        if not model:
            return
        runtime = {k: getattr(agent, k, None) for k in ("provider", "base_url", "api_mode")}
        runtime["fallback_active"] = bool(getattr(agent, "_fallback_activated", False))
        runtime = {k: v for k, v in runtime.items() if v not in (None, "")}
        try:
            db = self._session_db._db
            row = db.get_session(session_id)
            if not row:
                return
            # Legacy backfill: canonical Bot Chats created BEFORE the follow_profile_config contract existed
            # carry no marker, yet they are still the plugin-owned forever-DM. The plugin's own identity
            # rule is "the profile's session titled exactly 'Bot Chat'" (UNIQUE(title) makes that an exact
            # registry, and pre-policy rows may be visible OR hidden), so mirror that rule here. Without
            # this, every Bot Chat that already exists in the field stays pinned to its stale stored
            # provider until the user deletes it — the exact live-report shape (#89497 / #94818).
            raw_config = row.get("model_config")
            config = {}
            with suppress(Exception):
                config = json.loads(raw_config) if raw_config else {}
            if not isinstance(config, dict):
                config = {}
            gateway_runtime = dict(config.get("gateway_runtime") or {})
            if row.get("model") == model and all(gateway_runtime.get(k) == v for k, v in runtime.items()):
                return
            config["gateway_runtime"] = runtime
            db.update_session_meta(session_id, json.dumps(config), model=model)
        except Exception:
            logger.debug("Failed to sync gateway session model metadata", exc_info=True)

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
            self._session_state(_quick_key).turn.lease_tokens[run_generation] = _lease_token

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
            # A secondary with no home channel must not borrow the default profile's from
            # os.environ; only an UNSCOPED single-profile read may fall back to the process env.
            from agent.secret_scope import UnscopedSecretError, get_secret
            try:
                home_env = (get_secret(env_key) or "").strip()
            except UnscopedSecretError:
                home_env = (os.getenv(env_key) or "").strip()
            except Exception:
                home_env = ""
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

    @dataclasses.dataclass
    class _PreparedTurn:
        """Inputs to the agent run assembled by ``_hmwa_prepare_turn``."""

        history: Any
        context_prompt: str
        message_text: Any
        persist_user_message: Any
        persist_user_timestamp: Any
        persist_user_display_kind: Optional[str]
        persistence_session_id: Optional[str] = None
        persistence_owner: Optional[str] = None

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
