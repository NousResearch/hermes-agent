"""Agent-turn execution for GatewayRunner: _handle_message_with_agent, _run_agent*, proxy path,
background tasks, MCP reload. Bound onto ``GatewayRunner`` via the MRO; ``gateway.run`` internals
are imported lazily inside method bodies (import cycle) so ``patch("gateway.run.X")`` still works.
"""

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
from contextlib import nullcontext, suppress
from contextvars import copy_context
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
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


class GatewayTurnMixin:
    """Agent-turn execution for GatewayRunner (see module docstring)."""

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

    def _event_thread_metadata(self, event, source):
        """Thread metadata for a send that replies to ``event`` on ``source``."""
        return self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

    @staticmethod
    def _pop_post_delivery_callback(adapter, key, generation):
        """Pop the adapter's deferred post-delivery callback for ``key`` (legacy dict fallback)."""
        if getattr(type(adapter), "pop_post_delivery_callback", None) is not None:
            return adapter.pop_post_delivery_callback(key, generation=generation)
        if adapter and hasattr(adapter, "_post_delivery_callbacks"):
            return adapter._post_delivery_callbacks.pop(key, None)
        return None

    @staticmethod
    def _is_intentional_silence(agent_result, response) -> bool:
        try:
            from gateway.response_filters import is_intentional_silence_agent_result
            return is_intentional_silence_agent_result(agent_result, response)
        except Exception:
            return False

    async def _hmwa_resolve_session(self, event, source):
        """Resolve ``source`` to its session entry (topic recovery, internal-route guards, Telegram
        topic-binding heal). Returns ``(source, session_entry, session_key)`` or ``None`` to drop
        the event."""
        # Topic-mode DMs: rewrite a stale/foreign thread_id to the user's last-active topic so a
        # cross-topic Reply doesn't fragment the conversation.
        recovered = await asyncio.to_thread(self._recover_telegram_topic_thread_id, source)
        if recovered is not None:
            logger.info(
                "telegram topic recovery: chat=%s user=%s %r -> %s",
                source.chat_id, source.user_id, source.thread_id, recovered,
            )
            source = dataclasses.replace(source, thread_id=recovered)
            with suppress(Exception):
                event.source = source

        event_metadata = getattr(event, "metadata", None) or {}
        expected_session_key = str(event_metadata.get("gateway_session_key") or "").strip()
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
        from gateway.run import _AUTO_RESET_CONTEXT_NOTES, _auto_reset_reason_text
        reset_reason = getattr(session_entry, 'auto_reset_reason', None) or 'idle'
        context_note = _AUTO_RESET_CONTEXT_NOTES.get(reset_reason, _AUTO_RESET_CONTEXT_NOTES["idle"])
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
            policy = self.session_store.config.get_reset_policy(
                platform=source.platform, session_type=getattr(source, 'chat_type', 'dm'),
            )
            # Check pairing store. A pairing entry is a first-class authorization grant, created only by a
            # trusted operator approving a pairing code (hermes gateway pairing approve / the authenticated
            # dashboard) — an inbound sender can never reach approve_code, so this is not an
            # attacker-controlled path. Honored as a UNION with the allowlist: a paired user is authorized
            # regardless of the allowlist, and when an allowlist IS configured, operator approval also
            # writes the user into that allowlist (see PairingStore._approve_user), keeping a single
            # operator-visible source of truth. (#23778: the original bypass was the inbound
            # message/approval-button gate, not this gate; that gate is fixed separately.) In multiplex
            # gateways, route to the per-profile PairingStore so each profile's whitelist is isolated; falls
            # back to the global store when the source has no profile or the profile isn't registered.
            platform_name = source.platform.value if source.platform else ""
            # Suspended / restart-recovery-expired sessions always notify (the user must learn they
            # can /resume); idle/daily resets respect policy.notify + excluded platforms + activity.
            should_notify = reset_reason in {"suspended", "resume_pending_expired"} or (
                policy.notify
                and getattr(session_entry, 'reset_had_activity', False)
                and platform_name not in policy.notify_exclude_platforms
            )
            adapter = self._adapter_for_source(source) if should_notify else None
            if adapter:
                notice = (
                    f"◐ Session automatically reset ({_auto_reset_reason_text(reset_reason, policy)}). "
                    f"Conversation history cleared.\n"
                    f"Use /resume to browse and restore a previous session.\n"
                    f"Adjust reset timing in config.yaml under session_reset."
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

    @dataclasses.dataclass
    class _HygienePlan:
        """Hygiene pre-check outcome for one turn."""

        needs_compress: bool
        approx_tokens: int
        msg_count: int
        warn_token_threshold: int

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

        # Prefer the API-reported prompt tokens over the rough estimate (runs 30-50% high, which only
        # fires hygiene early — safe). Do NOT compensate with a threshold multiplier.
        if session_entry.last_prompt_tokens > 0:
            _approx_tokens, _token_source = session_entry.last_prompt_tokens, "actual"
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

    # reasoning_style → (header line, per-line quote prefix for blank / non-blank lines)
    _REASONING_QUOTE_STYLES = {
        "subtext": ("-# 💭 Reasoning", "-# ", "-#"), "blockquote": ("> 💭 **Reasoning:**", "> ", ">")
    }

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

    _CONTEXT_OVERFLOW_ERROR_PHRASES = (
        "context length", "context size", "context window",
        "maximum context", "token limit", "too many tokens",
        "reduce the length", "exceeds the limit",
        "request entity too large", "prompt is too long",
        "payload too large", "input is too long",
    )

    def _hmwa_classify_turn_failure(self, agent_result, history, session_entry):
        """Classify a finished turn for transcript persistence. Returns
        ``(agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure)``.

        Context-overflow failures must NOT persist the user message (session would grow and
        reproduce the failure forever); transient failures (429/timeout/5xx) DO."""
        from gateway.run import _is_gateway_hidden_reasoning_incomplete_turn
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
        _err = str(agent_result.get("error", "")).lower()
        # Multi-word phrases (not bare "exceed"/"token") avoid matching "rate limit exceeded".
        is_context_overflow_failure = agent_failed_early and (
            bool(agent_result.get("compression_exhausted"))
            or any(p in _err for p in self._CONTEXT_OVERFLOW_ERROR_PHRASES)
            or ("400" in _err and len(history) > 50)
        )
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

    _STATUS_HINTS = {
        401: " Check your API key or run `claude /login` to refresh OAuth credentials.",
        402: " Your API balance or quota is exhausted. Check your provider dashboard.",
        529: " The API is temporarily overloaded. Please try again shortly.",
    }

    async def _hmwa_agent_error_reply(self, e, event, source, session_entry, session_key, prepared):
        """``except Exception`` body of the agent turn: stop typing, log, persist the inbound user
        turn once, and build the sanitized user-facing error reply."""
        # Retain Slack thread/workspace routing so a failed turn cannot leave its status visible.
        await self._hmwa_stop_typing_for_turn(event, source)
        logger.exception("Agent error in session %s", session_key)
        # Failures before run_conversation() (provider/httpx init) can't persist the inbound turn:
        # append the user message here once, unless the latest user row already matches it.
        try:
            if prepared.message_text is not None and session_entry is not None:
                try:
                    _recent_transcript = await self.async_session_store.load_transcript(session_entry.session_id)
                except Exception:
                    _recent_transcript = []
                _expected_user_content = (
                    prepared.persist_user_message if prepared.persist_user_message is not None
                    else prepared.message_text
                )
                _last_user = next(
                    (_msg for _msg in reversed(_recent_transcript[-10:]) if _msg.get("role") == "user"), None,
                )
                if _last_user is None or _last_user.get("content") != _expected_user_content:
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

    @dataclasses.dataclass
    class _PreparedTurn:
        """Inputs to the agent run assembled by ``_hmwa_prepare_turn``."""

        history: Any
        context_prompt: str
        message_text: Any
        persist_user_message: Any
        persist_user_timestamp: Any
        persist_user_display_kind: Optional[str]

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
        except TranscriptReadError:
            self._clear_session_env(_session_env_tokens)
            return (
                "⚠️ This session's history is temporarily unavailable, so this message was not "
                "processed. Ask the operator to inspect state.db, then resend after it is healthy. "
                "Use /reset only if you intentionally want to start a new conversation."
            ), _session_env_tokens

        history = await self._hmwa_run_session_hygiene(
            event, source, session_entry, session_key, history, _quick_key, run_generation,
        )

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
        return self._PreparedTurn(
            history, context_prompt, message_text, persist_user_message, persist_user_timestamp,
            persist_user_display_kind,
        ), _session_env_tokens

    async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
        """Inner handler that runs under the _running_agents sentinel guard."""
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        logger.info(
            "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r",
            _platform_name, source.user_name or source.user_id or "unknown",
            source.chat_id or "unknown", (event.text or "")[:80].replace("\n", " "),
            getattr(event, "reply_to_message_id", None),
            (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " "),
        )

        resolved = await self._hmwa_resolve_session(event, source)
        if resolved is None:
            return
        source, session_entry, session_key = resolved
        prepared, _session_env_tokens = await self._hmwa_prepare_turn(
            event, source, session_entry, session_key, _quick_key, run_generation,
        )
        if not isinstance(prepared, self._PreparedTurn):
            return prepared
        history, message_text = prepared.history, prepared.message_text

        try:
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "chat_id": source.chat_id or "",
                "thread_id": str(source.thread_id) if getattr(source, "thread_id", None) else "",
                "chat_type": getattr(source, "chat_type", "") or "",
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)

            # Capture the launch session id so post-run compression publication is identity-guarded
            # (a /new may move session_entry.session_id while the old run is still unwinding).
            _run_start_session_id = session_entry.session_id
            _turn_started_monotonic = time.monotonic()
            agent_result = await self._run_agent(
                message=message_text, context_prompt=prepared.context_prompt, history=history, source=source,
                session_id=_run_start_session_id, session_key=session_key,
                run_generation=run_generation, event_message_id=self._reply_anchor_for_event(event),
                inbound_message_id=str(event.message_id) if event.message_id else None,
                channel_prompt=event.channel_prompt, moa_config=getattr(event, "_moa_config", None),
                persist_user_message=prepared.persist_user_message,
                persist_user_timestamp=prepared.persist_user_timestamp,
                persist_user_display_kind=prepared.persist_user_display_kind,
                message_type=event.message_type,
                startup_resume=bool(getattr(event, "startup_resume", False)),
            )
            _turn_seconds = time.monotonic() - _turn_started_monotonic

            await self._hmwa_stop_typing_for_turn(event, source)

            if not self._is_session_run_current(_quick_key, run_generation):
                self._hmwa_discard_stale_result(source, _quick_key, run_generation)
                return None

            response, _intentional_silence, agent_messages = await self._hmwa_shape_agent_response(
                agent_result, source, history, session_entry, session_key,
                _quick_key, run_generation, _run_start_session_id, _platform_name, _msg_start_time,
            )
            response = self._hmwa_prepend_reasoning(agent_result, response, source, _intentional_silence)
            _footer_line = self._hmwa_runtime_footer_line(agent_result, source, _turn_seconds)
            # Streaming already delivered the body: the footer goes out as a trailing send instead.
            if _footer_line and response and not agent_result.get("already_sent") and not _intentional_silence:
                response = f"{response}\n\n{_footer_line}"
            await self._hmwa_post_turn_hooks(hook_ctx, agent_result, response)

            agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure = (
                self._hmwa_classify_turn_failure(agent_result, history, session_entry)
            )
            response, session_entry = await self._hmwa_compression_exhaustion_reset(
                agent_result, response, session_entry, session_key, source,
            )
            await self._hmwa_persist_turn_transcript(
                event=event, source=source, session_entry=session_entry, session_key=session_key,
                agent_result=agent_result, agent_messages=agent_messages, prepared=prepared,
                response=response, agent_failed_early=agent_failed_early,
                hidden_reasoning_incomplete=hidden_reasoning_incomplete,
                is_context_overflow_failure=is_context_overflow_failure,
            )
            return await self._hmwa_deliver_turn_response(
                event, source, session_entry, session_key, run_generation,
                agent_result, agent_messages, response, _footer_line, _intentional_silence,
            )

        except Exception as e:
            return await self._hmwa_agent_error_reply(e, event, source, session_entry, session_key, prepared)
        finally:
            # Restore session context variables to their pre-handler state
            self._clear_session_env(_session_env_tokens)

    def _profile_scope_for_source(self, source: SessionSource):
        """``_profile_runtime_scope`` for ``source``'s profile when multiplexing, else a no-op context.

        Under multiplexing config/skills/memory resolve to the source profile's home AND credentials
        come from its secret scope (never process-global ``os.environ``)."""
        from gateway.run import _profile_runtime_scope
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return _profile_runtime_scope(self._resolve_profile_home_for_source(source))
        return nullcontext()

    def _reset_notice_session_info(self, source: SessionSource) -> str:
        """Session-info block for the auto-reset notice, resolved inside the profile serving ``source``.

        Call via ``asyncio.to_thread``: resolution can block (credential refresh, context-length
        probes), and the scope is entered here so contextvars behave in the worker thread."""
        with self._profile_scope_for_source(source):
            return self._format_session_info()

    def _format_session_info(self) -> str:
        """Model / provider / context-length / endpoint block so users can spot bad context detection."""
        from gateway.run import _resolve_gateway_model_context
        resolved = _resolve_gateway_model_context()
        context_length = resolved.context_length
        ctx_source = {
            "config": "config",
            "default": "default — set model.context_length in config to override",
        }.get(resolved.context_source, "detected")
        ctx_display = (
            f"{context_length / 1_000_000:.1f}M" if context_length >= 1_000_000
            else f"{context_length // 1_000}K" if context_length >= 1_000 else str(context_length)
        )
        lines = [
            f"◆ Model: `{resolved.model}`",
            f"◆ Provider: {resolved.provider or 'openrouter'}",
            f"◆ Context: {ctx_display} tokens ({ctx_source})",
        ]
        base_url = resolved.base_url
        if base_url and base_url_hostname(base_url) in ("localhost", "127.0.0.1", "0.0.0.0"):
            lines.append(f"◆ Endpoint: {base_url}")
        return "\n".join(lines)

    async def _run_background_task(
        self, prompt: str, source: "SessionSource", task_id: str,
        event_message_id: Optional[str] = None, media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Profile-scoping wrapper around the background agent task (mirrors ``_run_agent``)."""
        with self._profile_scope_for_source(source):
            return await self._run_background_task_inner(
                prompt, source, task_id, event_message_id, media_urls, media_types,
            )

    def _resolve_enabled_toolsets_for_source(
        self, user_config: dict, source: "SessionSource", platform_key: str,
    ) -> list:
        """Enabled toolsets for an agent run, honoring an adapter ``toolsets_for_source()`` override
        validated through the SAME ``_get_platform_tools`` path (unknown / platform-restricted
        toolsets dropped, not trusted)."""
        from hermes_cli.tools_config import _get_platform_tools
        try:
            adapter = self._adapter_for_source(source)
            override = adapter.toolsets_for_source(source) if adapter is not None else None
        except Exception:
            override = None
        if override and isinstance(override, list):
            pts = dict(user_config.get("platform_toolsets") or {})
            pts[platform_key] = [str(x) for x in override]
            user_config = {**user_config, "platform_toolsets": pts}
        return sorted(_get_platform_tools(user_config, platform_key))

    def _resolve_turn_toolsets(self, user_config: dict, source: "SessionSource", platform_key: str):
        """``(enabled_toolsets, disabled_toolsets)`` for an agent run on ``source``."""
        from agent.skill_utils import parse_config_string_list
        enabled = self._resolve_enabled_toolsets_for_source(user_config, source, platform_key)
        disabled = parse_config_string_list((user_config.get("agent") or {}).get("disabled_toolsets")) or None
        return enabled, disabled

    async def _run_background_task_inner(
        self, prompt: str, source: "SessionSource", task_id: str,
        event_message_id: Optional[str] = None, media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        from gateway.run import (
            _checkpoint_agent_kwargs, _current_max_iterations, _load_gateway_config,
            _platform_config_key,
        )
        from run_agent import AIAgent
        media_urls = media_urls or []
        media_types = media_types or []
        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return
        _thread_metadata = self._thread_metadata_for_source(source, event_message_id)

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(source=source, user_config=user_config)
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    f"❌ Background task {task_id} failed: no provider credentials configured.",
                    metadata=_thread_metadata,
                )
                return

            platform_key = _platform_config_key(source.platform)
            enabled_toolsets, disabled_toolsets = self._resolve_turn_toolsets(user_config, source, platform_key)
            pr = self._provider_routing
            max_iterations = _current_max_iterations()
            reasoning_config = self._resolve_session_reasoning_config(source=source, model=model)
            self._reasoning_config = reasoning_config
            self._service_tier = self._resolve_session_service_tier(source=source)
            turn_route = self._resolve_turn_agent_config(prompt, model, runtime_kwargs)

            # Enrich the prompt with image descriptions (same as the main flow).
            enriched_prompt = prompt
            image_paths = [
                path for i, path in enumerate(media_urls)
                if (media_types[i] if i < len(media_types) else "").startswith("image/")
            ]
            if image_paths:
                try:
                    enriched_prompt = await self._enrich_message_with_vision(prompt, image_paths)
                except Exception as e:
                    logger.warning("Background task vision enrichment failed: %s", e)

            def run_sync():
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    **_checkpoint_agent_kwargs(user_config),
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    **{k: getattr(source, k) for k in (
                        "user_id", "user_id_alt", "user_name", "chat_id", "chat_name", "chat_type", "thread_id",
                    )},
                    session_db=getattr(self._session_db, "_db", self._session_db),
                    # Reload from disk — do not reuse the startup snapshot.
                    # See #60955.
                    fallback_model=self._refresh_fallback_model(),
                )
                try:
                    return agent.run_conversation(user_message=enriched_prompt, task_id=task_id)
                finally:
                    self._cleanup_agent_resources(agent)

            result = await self._run_in_executor_with_context(run_sync)

            response = result.get("final_response", "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"
            # Fresh conversation, so history_offset=0: every message in the run belongs to this turn.
            if response:
                response = repair_explicit_computer_use_media_paths(response, result.get("messages", []))

            preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
            header = f'✅ Background task complete\nPrompt: "{preview}"\n\n'
            images, media_files, text_content = [], [], ""
            if response:
                media_files, response = adapter.extract_media(response)
                media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
                images, text_content = adapter.extract_images(response)
            if text_content:
                await adapter.send(chat_id=source.chat_id, content=header + text_content, metadata=_thread_metadata)
            elif not images and not media_files:
                await adapter.send(
                    chat_id=source.chat_id, content=header + "(No response generated)", metadata=_thread_metadata,
                )
            for image_url, alt_text in (images or []):
                with suppress(Exception):
                    await adapter.send_image(
                        chat_id=source.chat_id, image_url=image_url, caption=alt_text, metadata=_thread_metadata,
                    )
            # Route each media file by type (voice bubble / video / image / document), as the
            # streaming + kanban paths do.
            from gateway.platforms.base import should_send_media_as_audio as _should_send_media_as_audio
            from gateway.run_notifications import _IMAGE_EXTS, _VIDEO_EXTS
            for media_path, _is_voice in (media_files or []):
                _ext = os.path.splitext(media_path)[1].lower()
                with suppress(Exception):
                    if _should_send_media_as_audio(source.platform, _ext, _is_voice):
                        await adapter.send_voice(
                            chat_id=source.chat_id, audio_path=media_path, metadata=_thread_metadata,
                            is_voice=_is_voice,
                        )
                    else:
                        sender, key = (
                            (adapter.send_video, "video_path") if _ext in _VIDEO_EXTS
                            else (adapter.send_image_file, "image_path") if _ext in _IMAGE_EXTS
                            else (adapter.send_document, "file_path")
                        )
                        await sender(chat_id=source.chat_id, metadata=_thread_metadata, **{key: media_path})

        except Exception as e:
            logger.exception("Background task %s failed", task_id)
            with suppress(Exception):
                await adapter.send(
                    chat_id=source.chat_id, content=f"❌ Background task {task_id} failed: {e}",
                    metadata=_thread_metadata,
                )

    def _mcp_reload_refresh_cached_agents(self, multiplex: bool, profile) -> None:
        """Refresh cached agents so existing sessions see new MCP tools on their next turn without
        a history-destroying ``/new``. Each agent keeps its build-time toolset selection EXACTLY: a
        session built with restricted enabled_toolsets (e.g. ["safe"]) must NOT silently gain tools."""
        try:
            from tools.mcp_tool_agent import refresh_agent_mcp_tools
            _cache = getattr(self, "_agent_cache", None)
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            if _cache_lock is None or not _cache:
                return
            # Multiplex: only this profile's sessions (another profile's agent would get this registry).
            _ns_prefix = _session_key_namespace(profile) + ":" if multiplex else None
            with _cache_lock:
                for _sess_key, _entry in list(_cache.items()):
                    if _ns_prefix and not str(_sess_key).startswith(_ns_prefix):
                        continue
                    _agent = _entry[0] if isinstance(_entry, tuple) else _entry
                    if _agent is not None:
                        refresh_agent_mcp_tools(_agent, quiet_mode=True)
        except Exception as _exc:
            logger.debug("Failed to update cached agent tools after MCP reload: %s", _exc)

    async def _execute_mcp_reload(self, event: MessageEvent) -> str:
        """Disconnect, reconnect, and notify MCP tool changes (shared by button / text / no-confirm paths).

        Under multiplex the reload runs inside the requesting profile's runtime scope (entered here
        when the caller did not) and only that profile's servers are torn down and rediscovered.

        See #95518.
        """
        from gateway.run import _profile_runtime_scope
        multiplex = bool(getattr(self.config, "multiplex_profiles", False))
        if multiplex and not get_hermes_home_override():
            profile_home = self._resolve_profile_home_for_source(event.source)
            with _profile_runtime_scope(Path(profile_home)):
                return await self._execute_mcp_reload(event)
        try:
            from tools.mcp_tool_lifecycle import shutdown_mcp_servers
            from tools.mcp_tool_discovery import discover_mcp_tools
            from tools.mcp_tool import _servers, _lock, _server_scope_keys
            from tools.mcp_tool_agent import reprobe_tool_availability
            from tools.registry import registry

            reload_scope = registry.current_scope_key() if multiplex else None

            def _scoped_server_names() -> set:
                with _lock:
                    return {
                        name for name in _servers
                        if reload_scope is None or _server_scope_keys.get(name) == reload_scope
                    }

            old_servers = _scoped_server_names()
            await self._run_in_executor_with_context(lambda: shutdown_mcp_servers(scope=reload_scope))
            # Explicit reload also re-probes tool availability (check_fn).
            reprobe_tool_availability()
            # Reconnect by discovering tools (reads config.yaml fresh).
            new_tools = await self._run_in_executor_with_context(discover_mcp_tools)

            connected_servers = _scoped_server_names()
            if reload_scope is not None:
                from tools.mcp_tool import _mcp_tool_server_names
                with _lock:
                    new_tools = [n for n in new_tools if _mcp_tool_server_names.get(n) in connected_servers]
            # (label, i18n key, names); i18n lines list reconnected first, the injected note added first.
            changes = (
                ("Reconnected", "gateway.reload_mcp.reconnected", connected_servers & old_servers),
                ("Added", "gateway.reload_mcp.added", connected_servers - old_servers),
                ("Removed", "gateway.reload_mcp.removed", old_servers - connected_servers),
            )
            lines = [t("gateway.reload_mcp.header")] + [
                t(key, names=", ".join(sorted(names))) for _label, key, names in changes if names
            ]
            if not connected_servers:
                lines.append(t("gateway.reload_mcp.none_connected"))
            else:
                lines.append(t("gateway.reload_mcp.tools_available", tools=len(new_tools), servers=len(connected_servers)))

            self._mcp_reload_refresh_cached_agents(multiplex, event.source.profile)

            # Append a note at the END of the history (preserves the prompt-cache prefix).
            change_parts = [
                f"{label} servers: {', '.join(sorted(names))}"
                for label, _key, names in (changes[1], changes[2], changes[0]) if names
            ]
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            with suppress(Exception):  # Best-effort; don't fail the reload over a transcript write
                session_entry = await self.async_session_store.get_or_create_session(event.source)
                await self.async_session_store.append_to_transcript(session_entry.session_id, reload_msg)

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return t("gateway.reload_mcp.failed", error=e)

    def _get_proxy_url(self) -> Optional[str]:
        """Proxy URL if proxy mode is configured (GATEWAY_PROXY_URL env wins over ``gateway.proxy_url``)."""
        from gateway.run import _load_gateway_config
        url = os.getenv("GATEWAY_PROXY_URL", "").strip()
        if not url:
            url = ((_load_gateway_config().get("gateway") or {}).get("proxy_url") or "").strip()
        return url.rstrip("/") if url else None

    def _build_stream_consumer_config(
        self, source: "SessionSource", scfg: Any, adapter: Any, *, on_missing_cursor: str,
    ) -> "tuple[Any, Optional[Callable[[], None]]]":
        """Build the shared ``StreamConsumerConfig`` and optional Telegram pause-typing closure.
        For non-editing adapters ``on_missing_cursor="fallback"`` streams with an empty cursor;
        ``"raise"`` raises ``RuntimeError`` so the caller skips streaming entirely."""
        from gateway.stream_consumer import StreamConsumerConfig
        _pause_typing_before_finalize = None
        if source.platform == Platform.TELEGRAM and hasattr(adapter, "pause_typing_for_chat"):
            def _pause_typing_before_finalize(_adapter=adapter, _chat_id=source.chat_id) -> None:
                _adapter.pause_typing_for_chat(_chat_id)
        # Non-editing platforms (QQ, WeChat) skip streaming — the partial first message could never
        # be updated — unless they have a native-streaming transport (WeCom msgtype "stream").
        _adapter_supports_edit = getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True)
        _adapter_supports_native_stream = bool(getattr(adapter, "SUPPORTS_NATIVE_STREAMING", False))
        if not _adapter_supports_edit and not _adapter_supports_native_stream and on_missing_cursor == "raise":
            raise RuntimeError("skip streaming for non-editable platform")
        _effective_cursor = scfg.cursor if _adapter_supports_edit else ""
        # Some Matrix clients render the cursor as tofu: stream text, no cursor.
        _buffer_only = source.platform == Platform.MATRIX
        if _buffer_only:
            _effective_cursor = ""
        # Fresh-final applies to Telegram only (others edit in place cheaply).
        # Fresh-final applies to Telegram only — other platforms either edit in place cheaply (Discord,
        # Slack) or don't have the timestamp-on-edit / edit-timestamp-stays-stale problem. (Ported from
        # openclaw/openclaw#72038.)
        _fresh_final_secs = (
            float(getattr(scfg, "fresh_final_after_seconds", 0.0) or 0.0)
            if source.platform == Platform.TELEGRAM else 0.0
        )
        _consumer_cfg = StreamConsumerConfig(
            edit_interval=scfg.edit_interval, buffer_threshold=scfg.buffer_threshold,
            cursor=_effective_cursor, buffer_only=_buffer_only,
            fresh_final_after_seconds=_fresh_final_secs, transport=scfg.transport or "edit",
            chat_type=getattr(source, "chat_type", "") or "",
        )
        return _consumer_cfg, _pause_typing_before_finalize

    def _run_still_current_fn(self, session_key: Optional[str], run_generation: Optional[int]) -> Callable[[], bool]:
        """Predicate: does this run's generation still own ``session_key``? (always True when untracked)."""
        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)
        return _run_still_current

    @staticmethod
    def _proxy_error_result(text: str) -> Dict[str, Any]:
        return {"final_response": text, "messages": [], "api_calls": 0, "tools": []}

    def _proxy_stream_consumer(self, source: "SessionSource", event_message_id, _thread_metadata, _run_still_current):
        """Platform stream consumer for the proxy path when streaming is enabled, else ``None``."""
        from gateway.run import _load_gateway_config, _platform_config_key
        _scfg = getattr(getattr(self, "config", None), "streaming", None)
        # #60671 — streaming TTS consumer is created on the outer event-loop thread before run_sync
        # launches.  run_sync only reads it via ``streaming_tts_consumer_holder[0]`` for delta callback
        # wiring.
        if _scfg is None:
            from gateway.config import StreamingConfig
            _scfg = StreamingConfig()
        from gateway.display_config import resolve_display_setting
        _plat_streaming = resolve_display_setting(_load_gateway_config(), _platform_config_key(source.platform), "streaming")
        _streaming_enabled = (
            _scfg.enabled and _scfg.transport != "off" if _plat_streaming is None else bool(_plat_streaming)
        )
        if not _streaming_enabled:
            return None
        try:
            from gateway.stream_consumer import GatewayStreamConsumer
            _adapter = self._adapter_for_source(source)
            if not _adapter:
                return None
            _consumer_cfg, _pause_typing_before_finalize = self._build_stream_consumer_config(
                source, _scfg, _adapter, on_missing_cursor="fallback",
            )
            return GatewayStreamConsumer(
                adapter=_adapter, chat_id=source.chat_id, config=_consumer_cfg,
                metadata=_thread_metadata, on_before_finalize=_pause_typing_before_finalize,
                initial_reply_to_id=event_message_id, run_still_current=_run_still_current,
            )
        except Exception as _sc_err:
            logger.debug("Proxy: could not set up stream consumer: %s", _sc_err)
            return None

    async def _run_agent_via_proxy(
        self, message: str, context_prompt: str, history: List[Dict[str, Any]],
        source: "SessionSource", session_id: str, session_key: str = None,
        run_generation: Optional[int] = None, event_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward the message to a remote Hermes API server instead of running a local AIAgent.

        Lets a Docker container handle Matrix E2EE while the agent runs on the host with full
        access to local files, memory, skills, and a unified session store."""
        from gateway.run import _GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS
        try:
            from aiohttp import ClientSession as _AioClientSession, ClientTimeout
        except ImportError:
            return self._proxy_error_result("⚠️ Proxy mode requires aiohttp. Install with: pip install aiohttp")

        proxy_url = self._get_proxy_url()
        if not proxy_url:
            return self._proxy_error_result("⚠️ Proxy URL not configured (GATEWAY_PROXY_URL or gateway.proxy_url)")

        # The proxy key is a per-profile credential: honor the installed secret scope under multiplex.
        # Only UnscopedSecretError / import failures fall back to the env; any other get_secret()
        # error propagates (same as BASE) rather than silently degrading to the ambient key.
        try:
            from agent.secret_scope import UnscopedSecretError, get_secret

            try:
                proxy_key = (get_secret("GATEWAY_PROXY_KEY") or "").strip()
            except UnscopedSecretError:
                proxy_key = os.getenv("GATEWAY_PROXY_KEY", "").strip()
        except Exception:
            proxy_key = os.getenv("GATEWAY_PROXY_KEY", "").strip()

        _run_still_current = self._run_still_current_fn(session_key, run_generation)

        def _stale_result(what: str) -> Dict[str, Any]:
            logger.info(
                "Discarding stale proxy %s for %s — generation %d is no longer current",
                what, session_key or "?", run_generation or 0,
            )
            return {
                "final_response": "", "messages": [], "api_calls": 0, "tools": [],
                "history_offset": len(history), "session_id": session_id, "response_previewed": False,
            }

        # OpenAI chat format. The remote keeps continuity via X-Hermes-Session-Id; send the current
        # message plus a compact text-only history for a remote that has none yet.
        api_messages: List[Dict[str, str]] = [{"role": "system", "content": context_prompt}] if context_prompt else []
        api_messages += [
            {"role": msg.get("role"), "content": msg.get("content")}
            for msg in history if msg.get("role") in {"user", "assistant"} and msg.get("content")
        ]
        api_messages.append({"role": "user", "content": message})

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if proxy_key:
            headers["Authorization"] = f"Bearer {proxy_key}"
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id
        body = {"model": "hermes-agent", "messages": api_messages, "stream": True}

        _thread_metadata: Optional[Dict[str, Any]] = self._thread_metadata_for_source(source, event_message_id)
        _stream_consumer = self._proxy_stream_consumer(source, event_message_id, _thread_metadata, _run_still_current)
        stream_task = asyncio.create_task(_stream_consumer.run()) if _stream_consumer else None

        _adapter = self._adapter_for_source(source)
        if _adapter:
            with suppress(Exception):
                await _adapter.send_typing(source.chat_id, metadata=_thread_metadata)

        full_response = ""
        _start = time.time()
        try:
            _timeout = ClientTimeout(total=0, sock_read=1800)
            async with _AioClientSession(timeout=_timeout) as session:
                async with session.post(f"{proxy_url}/v1/chat/completions", json=body, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning("Proxy error (%d) from %s: %s", resp.status, proxy_url, error_text[:500])
                        return self._proxy_error_result(f"⚠️ Proxy error ({resp.status}): {error_text[:300]}")

                    buffer = ""
                    async for chunk in resp.content.iter_any():
                        if not _run_still_current():
                            return _stale_result("stream")
                        buffer += chunk.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                choices = json.loads(data).get("choices", [])
                            except json.JSONDecodeError:
                                continue
                            content = choices[0].get("delta", {}).get("content", "") if choices else ""
                            if content:
                                full_response += content
                                if _stream_consumer:
                                    _stream_consumer.on_delta(content)
                        if len(buffer) > _GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS:
                            raise ValueError("Proxy SSE stream exceeded max buffer size without a line boundary")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Proxy connection error to %s: %s", proxy_url, e)
            if not full_response:
                return self._proxy_error_result(f"⚠️ Proxy connection error: {e}")
            # Partial response — return what we got
        finally:
            if _stream_consumer:
                _stream_consumer.finish()
            if stream_task:
                try:
                    await asyncio.wait_for(stream_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stream_task.cancel()

        _elapsed = time.time() - _start
        if not _run_still_current():
            return _stale_result("result")
        logger.info(
            "proxy response: url=%s session=%s time=%.1fs response=%d chars",
            proxy_url, (session_id or "")[:20], _elapsed, len(full_response),
        )
        return {
            "final_response": full_response or "(No response from remote agent)",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response},
            ],
            "api_calls": 1,
            "tools": [],
            "history_offset": len(history),
            "session_id": session_id,
            "response_previewed": _stream_consumer is not None and bool(full_response),
        }

    async def _run_agent(
        self, message: str, context_prompt: str, history: List[Dict[str, Any]],
        source: SessionSource, session_id: str, **turn_kwargs,
    ) -> Dict[str, Any]:
        """Profile-scoping wrapper around ``_run_agent_inner`` (same keyword parameters; pass-through
        when multiplexing is off)."""
        with self._profile_scope_for_source(source):
            return await self._run_agent_inner(message, context_prompt, history, source, session_id, **turn_kwargs)

    def _run_agent_display_settings(self, source: SessionSource) -> "GatewayRunner._RunAgentDisplay":
        """Resolve per-platform display, progress, status and streaming-surface settings for a turn."""
        from gateway.run import (
            _gateway_platform_value, _has_platform_display_override, _load_gateway_config,
            _platform_config_key,
        )
        from gateway.display_config import resolve_display_setting
        from gateway.status_phrases import choose_status_phrase, resolve_status_phrase_catalog
        user_config = _load_gateway_config()
        platform_key = _platform_config_key(source.platform)
        enabled_toolsets, disabled_toolsets = self._resolve_turn_toolsets(user_config, source, platform_key)
        adapter = self._adapter_for_source(source)
        # display.platforms.<platform>.<key> → display.<key> → built-in platform defaults.
        _display_cfg = user_config.get("display", {})
        if not isinstance(_display_cfg, dict):
            _display_cfg = {}

        # Tool preview length (0 = no limit) and friendly tool labels (default on), per-platform.
        for _setter, _setting, _default, _cast in (
            ("set_tool_preview_max_len", "tool_preview_length", 0, lambda v: int(v) if v else 0),
            ("set_friendly_tool_labels", "friendly_tool_labels", True, bool),
        ):
            with suppress(Exception):
                from agent import display as _agent_display
                _val = resolve_display_setting(user_config, platform_key, _setting, _default)
                getattr(_agent_display, _setter)(_cast(_val))

        # Tool progress mode; HERMES_TOOL_PROGRESS_MODE wins only when the config never set it.
        _resolved_tp = resolve_display_setting(user_config, platform_key, "tool_progress")
        _env_tp = os.getenv("HERMES_TOOL_PROGRESS_MODE")
        _platform_cfg = (_display_cfg.get("platforms") or {}).get(platform_key) or {}
        _legacy_tp_overrides = _display_cfg.get("tool_progress_overrides") or {}
        _tool_progress_configured = "tool_progress" in _display_cfg or any(
            isinstance(cfg, dict) and key in cfg
            for cfg, key in ((_platform_cfg, "tool_progress"), (_legacy_tp_overrides, platform_key))
        )
        progress_mode = _env_tp if _env_tp and not _tool_progress_configured else (_resolved_tp or _env_tp or "all")
        # "accumulate" (edit one bubble) or "separate" (one msg per tool)
        progress_grouping = resolve_display_setting(user_config, platform_key, "tool_progress_grouping") or "accumulate"
        _generic_status_recent: List[str] = []
        _generic_status_catalog = resolve_status_phrase_catalog(user_config, platform_key)

        def _display_surface_mode(
            setting: str, *, default: bool = False,
            require_platform_override_for: set[Any] | None = None, allow_generic: bool = False,
        ) -> str:
            """Return off|raw|generic for a gateway visibility surface."""
            if require_platform_override_for:
                current_platform = _gateway_platform_value(source.platform)
                platform_only = {_gateway_platform_value(item) for item in require_platform_override_for}
                if (
                    current_platform in platform_only
                    and not _has_platform_display_override(user_config, platform_key, setting)
                ):
                    return "off"
            value = resolve_display_setting(user_config, platform_key, setting, default)
            if isinstance(value, str) and value.strip().lower() == "generic":
                return "generic" if allow_generic else "off"
            return "raw" if bool(value) else "off"

        def _generic_status_phrase(kind: str, *, tool_name: str | None = None, preview: str | None = None, args: Any = None) -> str:
            try:
                return choose_status_phrase(
                    kind, tool_name=tool_name, preview=preview, args=args,
                    recent=_generic_status_recent, catalog=_generic_status_catalog,
                )
            except Exception as _phrase_err:
                logger.debug("generic status phrase selection failed: %s", _phrase_err)
                return "still on it" if kind in {"heartbeat", "waiting", "long_running", "status"} else "one sec"

        # Webhooks can't edit messages, so tool progress / log mode are off there.
        is_webhook = source.platform == Platform.WEBHOOK
        tool_progress_enabled = progress_mode not in {"off", "log"} and not is_webhook
        # Live status for text-rendering typing indicators (Slack); independent of tool_progress.
        _live_status_mode = resolve_display_setting(user_config, platform_key, "live_status", "full")
        _live_status_adapter = (
            adapter if getattr(adapter, "supports_status_text", False) and _live_status_mode != "off" else None
        )
        # "log" mode: tool calls go to ~/.hermes/logs/tool_calls.log instead of the chat. Gateway-only.
        log_mode_enabled = progress_mode == "log" and not is_webhook
        # Interim assistant messages and thinking_progress are independent of tool progress (same
        # queue). Mattermost requires a per-platform opt-in: scratch text leaks into public threads.
        interim_assistant_messages_mode = _display_surface_mode(
            "interim_assistant_messages", default=True, require_platform_override_for={Platform.MATTERMOST},
        )
        interim_assistant_messages_enabled = not is_webhook and interim_assistant_messages_mode != "off"
        _thinking_enabled = _display_surface_mode(
            "thinking_progress", default=False, require_platform_override_for={Platform.MATTERMOST},
        ) != "off"
        # Slack-native task cards need the progress queue even with text tool_progress off.
        # Slack-native task cards (#29483): when the Slack adapter's opt-in is set, tool progress renders as
        # native plan/task cards via chat.startStream — the progress queue is needed even though Slack keeps
        # ordinary text tool_progress off by default (requiring both flags would silently leave the native
        # feature inactive).
        _native_slack_task_cards = False
        if source.platform == Platform.SLACK and hasattr(adapter, "native_task_cards_enabled"):
            try:
                _native_slack_task_cards = bool(adapter.native_task_cards_enabled())
            except Exception:
                logger.debug("Slack native task-card config check failed", exc_info=True)
        return self._RunAgentDisplay(
            user_config=user_config, platform_key=platform_key, enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets, resolve_display_setting=resolve_display_setting,
            progress_mode=progress_mode, progress_grouping=progress_grouping,
            _display_surface_mode=_display_surface_mode,
            tool_progress_enabled=tool_progress_enabled, _live_status_mode=_live_status_mode,
            _live_status_adapter=_live_status_adapter, log_mode_enabled=log_mode_enabled,
            log_queue=queue.Queue() if log_mode_enabled else None,
            interim_assistant_messages_enabled=interim_assistant_messages_enabled,
            _thinking_enabled=_thinking_enabled, _native_slack_task_cards=_native_slack_task_cards,
            needs_progress_queue=tool_progress_enabled or _thinking_enabled or _native_slack_task_cards,
            _generic_status_phrase=_generic_status_phrase,
        )

    # _RunAgentDisplay fields copied verbatim onto the TurnContext.
    _DISPLAY_TO_TURN_CTX = (
        "_live_status_adapter", "_live_status_mode", "_thinking_enabled", "progress_mode",
        "progress_grouping", "tool_progress_enabled", "log_queue", "resolve_display_setting",
        "user_config", "enabled_toolsets", "disabled_toolsets", "log_mode_enabled",
        "interim_assistant_messages_enabled", "needs_progress_queue", "_native_slack_task_cards",
    )

    def _run_agent_build_turn_context(
        self, disp: "GatewayRunner._RunAgentDisplay", AIAgent: Any, *, message: str, source: SessionSource,
        session_key: Optional[str], run_generation: Optional[int], **turn_params,
    ) -> Tuple[TurnContext, TurnRunner, Any]:
        """Build the ``TurnContext`` and its ``TurnRunner``; ``turn_params`` (history, context_prompt,
        session_id, persist_user_*, …) are stored verbatim. Returns ``(turn_ctx, turn_runner,
        cleanup_adapter)``."""
        from gateway.run_turn_runner import TurnRunner
        # Discord voice "verbal ack" on the FIRST tool call (discord.voice_fx.enabled): resolve the
        # guild whose voice connection is bound to this text channel (mirrors DiscordAdapter.play_tts).
        _voice_ack_guild: List[Optional[int]] = [None]
        if source.platform == Platform.DISCORD:
            _va = self.adapters.get(Platform.DISCORD)
            _vtc = getattr(_va, "_voice_text_channels", None)
            if isinstance(_vtc, dict) and hasattr(_va, "voice_mixer_active"):
                _voice_ack_guild[0] = next(
                    (_gid for _gid, _tc in _vtc.items() if str(_tc) == str(source.chat_id) and _va.voice_mixer_active(_gid)),
                    None,
                )

        # Auto-cleanup of temporary progress bubbles needs a real ``delete_message`` (getattr on the
        # type: a fake adapter without it means "can't delete", not a crash).
        _cleanup_progress = bool(
            disp.resolve_display_setting(disp.user_config, disp.platform_key, "cleanup_progress")
        )
        _cleanup_adapter = self._adapter_for_source(source) if _cleanup_progress else None
        if _cleanup_adapter is not None and getattr(type(_cleanup_adapter), "delete_message", None) in (
            None, BasePlatformAdapter.delete_message,
        ):
            _cleanup_progress = False
            _cleanup_adapter = None

        # The one-slot progress/holder containers shared with the callbacks are TurnContext defaults.
        turn_ctx = TurnContext(
            source=source, message=message, AIAgent=AIAgent, session_key=session_key,
            run_generation=run_generation, _cleanup_progress=_cleanup_progress,
            _run_still_current=self._run_still_current_fn(session_key, run_generation),
            progress_queue=queue.Queue() if disp.needs_progress_queue else None,
            _voice_ack_guild=_voice_ack_guild, _voice_ack_loop=asyncio.get_running_loop(),
            **{name: getattr(disp, name) for name in self._DISPLAY_TO_TURN_CTX}, **turn_params,
        )
        turn_runner = TurnRunner(self, turn_ctx)
        # Agent tool-lifecycle callbacks live on the runner (bound methods, same signatures).
        turn_ctx.progress_callback = turn_runner.progress_callback
        turn_ctx.voice_ack_callback = turn_runner.voice_ack_callback
        turn_ctx.native_tool_start_callback = turn_runner.combined_tool_start_callback
        turn_ctx.native_tool_complete_callback = turn_runner.native_tool_complete_callback
        return turn_ctx, turn_runner, _cleanup_adapter

    def _thread_metadata_for_progress(
        self, source: SessionSource, event_message_id: Optional[str], _progress_thread_id: Any,
        _relay_prospective_thread_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Thread metadata for a progress-lane send; relay Discord auto-thread lane falls back to the reply anchor.

        The connector will auto-thread on the reply anchor (thread is born on its FIRST send), so
        carrying it routes progress / status bubbles into the same thread as the final reply."""
        if not _progress_thread_id:
            metadata = None
        elif _progress_thread_id == source.thread_id:
            metadata = self._thread_metadata_for_source(source, event_message_id)
        else:
            metadata = self._thread_metadata_for_target(
                source.platform, source.chat_id, _progress_thread_id,
                chat_type=getattr(source, "chat_type", None), reply_to_message_id=event_message_id,
            )
        if metadata is None and _relay_prospective_thread_id:
            metadata = {"reply_to_message_id": event_message_id}
        return metadata

    def _run_agent_progress_threading(
        self, source: SessionSource, event_message_id: Optional[str], _native_slack_task_cards: bool
    ) -> Tuple[Optional[dict], Optional[str], Optional[dict]]:
        """Resolve where progress bubbles are threaded (platform-specific).

        Returns ``(progress_metadata, progress_reply_to, status_thread_metadata)``; the latter is
        for status / approval / stream sends (Feishu topics need the triggering message id via the
        reply API, so carry it as a fallback). Slack and Buzz honour the user's reply_in_thread
        opt-out: never synthesise a thread for progress, or every later reply inherits it."""
        from gateway.run import _non_conversational_metadata, _resolve_progress_thread_id
        is_buzz = str(getattr(source.platform, "value", source.platform) or "").lower() == "buzz"
        _progress_reply_in_thread = True
        _adapter = self._adapter_for_source(source) if source.platform == Platform.SLACK or is_buzz else None
        if _adapter is not None:
            try:
                if is_buzz:
                    _progress_reply_in_thread = getattr(_adapter, "_reply_to_mode", "first") != "off"
                else:
                    # Relay lane: the adapter owns mode resolution; native lane: flat extra key.
                    _mode_fn = getattr(_adapter, "_effective_reply_in_thread", None)
                    _progress_reply_in_thread = bool(
                        _mode_fn() if callable(_mode_fn) else _adapter.config.extra.get("reply_in_thread", True)
                    )
            except Exception:
                _progress_reply_in_thread = True
        _progress_thread_id = _resolve_progress_thread_id(
            source.platform, source.thread_id, event_message_id, reply_in_thread=_progress_reply_in_thread,
        )
        # Relay Discord auto-thread lane: the connector stamps prospective_thread_id at ingest.
        _relay_prospective_thread_id = (
            str(getattr(source, "prospective_thread_id", None))
            if source.platform == Platform.DISCORD
            and getattr(source, "delivered_via_upstream_relay", False)
            and getattr(source, "prospective_thread_id", None)
            and not source.thread_id
            else None
        )
        _progress_metadata = _non_conversational_metadata(
            self._thread_metadata_for_progress(
                source, event_message_id, _progress_thread_id, _relay_prospective_thread_id,
            ),
            # Freshness-gate stale resume_pending zombies (#46934) — but honor an explicit
            # ``session_reset.mode: none``: the user opted out of ALL automatic resets, so an expired resume
            # marker must fall through to a normal resume of the preserved transcript, never a silent fresh
            # session (#61052).
            platform=source.platform,
        )
        if _native_slack_task_cards:
            # chat.startStream in channels requires the recipient team/user pair; harmless elsewhere.
            _progress_metadata = dict(_progress_metadata or {})
            if source.scope_id:
                _progress_metadata.setdefault("recipient_team_id", source.scope_id)
                _progress_metadata.setdefault("slack_team_id", source.scope_id)
            if source.user_id:
                _progress_metadata.setdefault("recipient_user_id", source.user_id)
        # Buzz has no native thread_id: thread via reply-to unless the user opted out.
        _progress_reply_to = (
            event_message_id
            if (source.platform in (Platform.FEISHU, Platform.MATTERMOST) and source.thread_id and event_message_id)
            or (is_buzz and event_message_id and _progress_reply_in_thread)
            or _relay_prospective_thread_id
            else None
        )
        if source.platform == Platform.FEISHU and source.thread_id and event_message_id:
            _status_thread_metadata = {"thread_id": _progress_thread_id, "reply_to_message_id": event_message_id}
        else:
            _status_thread_metadata = self._thread_metadata_for_progress(
                source, event_message_id, _progress_thread_id, _relay_prospective_thread_id,
            )
        return _progress_metadata, _progress_reply_to, _status_thread_metadata

    async def _run_agent_write_tool_log(self, log_queue: Any) -> None:
        """Drain log_queue and append tool-call lines to tool_calls.log (tool_progress=log).

        RotatingFileHandler (5MB × 3) bounds the log; RedactingFormatter keeps secrets off disk."""
        from gateway.run import _hermes_home
        if log_queue is None:
            return
        from logging.handlers import RotatingFileHandler
        from agent.redact import RedactingFormatter

        log_dir = _hermes_home / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "tool_calls.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
        )
        file_handler.setFormatter(RedactingFormatter("%(message)s"))
        tool_logger = logging.getLogger(f"hermes.tool_calls.{id(log_queue)}")
        tool_logger.setLevel(logging.INFO)
        tool_logger.propagate = False
        tool_logger.addHandler(file_handler)
        try:
            while True:
                try:
                    tool_logger.info("%s", log_queue.get_nowait())
                except queue.Empty:
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error("write_tool_log error: %s", e)
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            # Drain remaining entries so late tool calls from the final iteration aren't lost.
            with suppress(Exception):
                while True:
                    tool_logger.info("%s", log_queue.get_nowait())
            tool_logger.removeHandler(file_handler)
            with suppress(Exception):
                file_handler.flush()
                file_handler.close()

    def _run_agent_start_streaming_tts(
        self, source: SessionSource, message_type: Optional[str],
        _status_thread_metadata: Optional[Dict[str, Any]], streaming_tts_consumer_holder: list,
    ) -> None:
        """Start the streaming-TTS consumer for a voice-input turn on an auto-TTS chat.

        Created on the gateway loop thread (not run_sync's executor); an inactive consumer leaves
        the holder None so the whole-file fallback path runs."""
        # Skip when streaming TTS already delivered audio for this turn (#60671).
        # This avoids a cross-scope NameError: the outer interrupt / finalisation paths reference the
        # consumer via ``streaming_tts_consumer_holder[0]``. Gates: voice input, auto-TTS enabled for this
        # chat, adapter supports streaming, and a usable streaming TTS provider configured. See #60671.
        _stts_adapter = self._adapter_for_source(source)
        _is_voice_input = (
            message_type is not None
            and str(getattr(message_type, "value", message_type)).lower() == "voice"
        )
        if _stts_adapter is None or not _is_voice_input or not _stts_adapter._should_auto_tts_for_chat(source.chat_id):
            return
        try:
            from gateway.streaming_tts_consumer import StreamingTTSConsumer
            from tools.tts_tool import _load_tts_config
            _stts_consumer = StreamingTTSConsumer(
                adapter=_stts_adapter, chat_id=source.chat_id, tts_config=_load_tts_config(),
                loop=self._gateway_loop or asyncio.get_event_loop(),
                metadata=_status_thread_metadata,
            )
            if _stts_consumer.active:
                streaming_tts_consumer_holder[0] = _stts_consumer
                _stts_consumer.start()
        except Exception as _stts_err:
            logger.debug("Could not set up streaming TTS consumer: %s", _stts_err)

    async def _run_agent_stream_consumer_task(self, stream_consumer_holder: list) -> None:
        """Wait (up to 10s) for the stream consumer to be created inside run_sync, then run it."""
        for _ in range(200):
            if stream_consumer_holder[0] is not None:
                await stream_consumer_holder[0].run()
                return
            await asyncio.sleep(0.05)

    @staticmethod
    async def _await_stream_task(stream_task) -> None:
        """Give the stream consumer task 5s to flush, then cancel it."""
        try:
            await asyncio.wait_for(stream_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task

    async def _run_agent_track_agent(self, turn_ctx: TurnContext) -> None:
        """Track this agent as running for the session (interrupt support) once it is created — only
        if this run is still current, else leave the newer run's slot alone."""
        session_key, run_generation, agent_holder = turn_ctx.session_key, turn_ctx.run_generation, turn_ctx.agent_holder
        while agent_holder[0] is None:
            await asyncio.sleep(0.05)
        if not session_key:
            return
        if run_generation is not None and not self._is_session_run_current(session_key, run_generation):
            logger.info(
                "Skipping stale agent promotion for %s — generation %s is no longer current",
                session_key or "", run_generation,
            )
            return
        self._session_state(session_key).turn.agent = agent_holder[0]
        if self._draining:
            self._update_runtime_status("draining")

    async def _run_agent_fire_pending_interrupt(
        self, adapter: Any, agent: Any, source: SessionSource, session_key: str,
        _interrupt_detected: "asyncio.Event", streaming_tts_consumer_holder: list, *,
        log_context: str, log: Callable[[], None],
    ) -> None:
        """Peek the adapter's pending event, transcribe voice, then signal the agent + abort streaming TTS.

        Peek WITHOUT consuming: the event must stay for the post-run ``_dequeue_pending_event()``
        (popping races the agent finishing). Transcribe BEFORE signaling so voice interrupts carry
        the real transcript."""
        from gateway.run import _build_media_placeholder
        _peek_event = adapter._pending_messages.get(session_key)
        pending_text = None
        if _peek_event is not None:
            pending_text = _peek_event.text or ""
            if self._pending_event_audio_paths(_peek_event):
                pending_text, _ = await self._transcribe_and_echo_pending_voice(
                    _peek_event, adapter, source, pending_text, log_context=log_context,
                    metadata={"thread_id": source.thread_id} if source.thread_id else None,
                )
            elif not pending_text and (getattr(_peek_event, "media_urls", None) or []):
                pending_text = _build_media_placeholder(_peek_event)
        log()
        agent.interrupt(pending_text)
        _interrupt_detected.set()
        # Abort streaming TTS on barge-in.
        # See #60671.
        # See #60671.
        # See #60671.
        # Finalize the streaming-TTS consumer (#60671). finish() is called from the outer event-loop thread
        # (not the executor worker) so early returns from run_sync are also finalised.  wait_complete()
        # drains queued audio; on timeout the consumer is aborted unconditionally — if audio was audible,
        # suppression is preserved so the gateway does not replay from the beginning; if no audio was
        # audible, the whole-file fallback path is permitted.
        _stts = streaming_tts_consumer_holder[0]
        if _stts is not None:
            _stts.abort("barge-in")

    async def _run_agent_monitor_for_interrupt(self, turn_ctx: TurnContext, _interrupt_detected: "asyncio.Event") -> None:
        """Poll the adapter for interrupts (new messages) every 200ms and signal the agent.

        Level 1 (base.py) catches regular text before _handle_message(); the inactivity poll loop
        has a BACKUP check in case this task dies. Keyed by session_key, NOT source.chat_id."""
        source, session_key, agent_holder = turn_ctx.source, turn_ctx.session_key, turn_ctx.agent_holder
        streaming_tts_consumer_holder = turn_ctx.streaming_tts_consumer_holder
        if not session_key:
            return
        while True:
            await asyncio.sleep(0.2)
            try:
                # Re-resolve the adapter each iteration so reconnects don't leave a stale reference.
                _adapter = self._adapter_for_source(source)
                if not _adapter:
                    continue
                if hasattr(_adapter, 'has_pending_interrupt') and _adapter.has_pending_interrupt(session_key):
                    agent = agent_holder[0]
                    if agent:
                        await self._run_agent_fire_pending_interrupt(
                            _adapter, agent, source, session_key, _interrupt_detected,
                            streaming_tts_consumer_holder, log_context="Voice-interrupt",
                            log=lambda: logger.debug("Interrupt detected from adapter, signaling agent..."),
                        )
                        break
            except asyncio.CancelledError:
                raise
            except Exception as _mon_err:
                logger.debug("monitor_for_interrupt error (will retry): %s", _mon_err)

    async def _run_agent_backup_interrupt_check(
        self, turn_ctx: TurnContext, _interrupt_detected: "asyncio.Event", interrupt_monitor: "asyncio.Task",
    ) -> None:
        """Backup interrupt check: if the monitor task died or missed the interrupt, catch it here."""
        source, session_key = turn_ctx.source, turn_ctx.session_key
        if _interrupt_detected.is_set() or not session_key:
            return
        _backup_adapter = self._adapter_for_source(source)
        _backup_agent = turn_ctx.agent_holder[0]
        if (_backup_adapter and _backup_agent
                and hasattr(_backup_adapter, 'has_pending_interrupt')
                and _backup_adapter.has_pending_interrupt(session_key)):
            await self._run_agent_fire_pending_interrupt(
                _backup_adapter, _backup_agent, source, session_key, _interrupt_detected,
                turn_ctx.streaming_tts_consumer_holder,
                log_context="Voice-backup-interrupt",
                log=lambda: logger.info(
                    "Backup interrupt detected for session %s (monitor task state: %s)",
                    session_key, "done" if interrupt_monitor.done() else "running",
                ),
            )

    @staticmethod
    def _run_agent_stream_confirmed_final_delivery(consumer, final_text: str, *, previewed: bool = False) -> bool:
        """True only when the actual final reply reached the user: a finalize call may carry only the
        last preview snapshot, so reconcile against the recorded payload — a demonstrable mismatch
        (False) overrides the flag; None keeps legacy trust."""
        if consumer is None:
            return False
        if getattr(consumer, "final_response_sent", False):
            matcher = getattr(consumer, "delivered_final_matches", None)
            if callable(matcher):
                with suppress(Exception):
                    if matcher(final_text) is False:
                        return False
            return True
        if previewed:
            has_delivered_text = getattr(consumer, "has_delivered_text", None)
            if callable(has_delivered_text):
                try:
                    return bool(has_delivered_text(final_text))
                except Exception:
                    return False
        return False

    def _run_agent_start_turn_worker(self, turn_ctx: TurnContext, run_sync: Callable[[], Any]) -> "GatewayRunner._RunAgentWorker":
        """Schedule ``run_sync`` on the executor plus the inactivity watchdog thread.

        *Inactivity* timeout (agent.gateway_timeout / HERMES_AGENT_TIMEOUT, env wins; 0 = unlimited),
        not wall-clock. The daemon watchdog is independent of asyncio: cgroup memory reclaim can
        starve the loop that runs the normal timeout poll."""
        from gateway.run import _float_env, _watch_gateway_turn_inactivity
        from tools.process_registry import process_registry
        agent_holder, session_key, run_generation = turn_ctx.agent_holder, turn_ctx.session_key, turn_ctx.run_generation
        _agent_timeout, _agent_warning = (
            v if v > 0 else None
            for v in (_float_env("HERMES_AGENT_TIMEOUT", 1800), _float_env("HERMES_AGENT_TIMEOUT_WARNING", 900))
        )

        # background=true processes survive a turn: reap only children created by THIS turn on timeout.
        _turn_task_id = turn_ctx.session_id or ""
        # The daemon watchdog is independent of asyncio: cgroup memory reclaim may starve the event loop
        # that runs the normal timeout poll, but it need not also postpone cleanup until the loop recovers
        # (#76115).
        _turn_process_baseline = process_registry.snapshot_running_ids(_turn_task_id)
        turn_ctx.process_task_id = _turn_task_id
        turn_ctx.process_baseline = _turn_process_baseline
        # task_id is session-scoped: gate the reap on this claim still being current so a replacement
        # turn's fresh process isn't killed by this turn's stale baseline.
        worker = self._RunAgentWorker(
            agent_timeout=_agent_timeout, agent_warning=_agent_warning, task_id=_turn_task_id,
            process_baseline=_turn_process_baseline, worker_done=threading.Event(),
            timeout_fired=threading.Event(), cleanup_lock=threading.Lock(),
            is_current=(
                (lambda: self._is_session_run_current(session_key, run_generation))
                if run_generation is not None
                else (lambda: True)
            ),
        )

        def _run_sync_with_timeout_lifecycle():
            try:
                return run_sync()
            finally:
                worker.worker_done.set()
                # `.turn.agent` stays reachable until the *next* turn is claimed; clearing the
                # ownership markers now means a /stop on the finished turn no longer reaps background
                # work it left running.
                # `.turn.agent` on the session state is only reset to _AGENT_PENDING_SENTINEL when the
                # *next* turn is claimed (see _session_state(...).turn.agent = ... at claim time), so a
                # stale reference to this exact agent instance stays reachable from
                # _interrupt_and_clear_session() until then. See #76115.
                _finished_agent = agent_holder[0] if agent_holder else None
                if _finished_agent is not None:
                    _finished_agent._gateway_turn_process_task_id = ""
                    _finished_agent._gateway_turn_process_baseline = frozenset()

        if _agent_timeout is not None:
            threading.Thread(
                target=_watch_gateway_turn_inactivity,
                kwargs={
                    "agent_holder": agent_holder, "timeout": _agent_timeout, "poll_interval": 5.0,
                    **self._reaper_kwargs(worker),
                },
                name=f"gateway-turn-watchdog-{_turn_task_id[:12]}",
                daemon=True,
            ).start()
        worker.executor_task = asyncio.ensure_future(
            self._run_in_executor_with_context(_run_sync_with_timeout_lifecycle)
        )
        return worker

    @staticmethod
    def _reaper_kwargs(worker: "GatewayRunner._RunAgentWorker") -> dict:
        """Shared kwargs of the watchdog + timeout-reaper threads."""
        return {
            **{k: getattr(worker, k) for k in ("task_id", "process_baseline", "worker_done", "timeout_fired", "cleanup_lock")},
            "is_still_current": worker.is_current,
        }

    @staticmethod
    def _agent_activity_summary(agent: Any) -> dict:
        """``agent.get_activity_summary()`` or ``{}`` when unavailable / failing."""
        if agent and hasattr(agent, "get_activity_summary"):
            with suppress(Exception):
                return agent.get_activity_summary()
        return {}

    async def _run_agent_inactivity_warning(self, worker, source, _status_thread_metadata) -> None:
        """Staged one-shot warning before the inactivity timeout escalates."""
        from gateway.run import _interim_metadata
        _warn_adapter = self._adapter_for_source(source)
        if not _warn_adapter:
            return
        try:
            await _warn_adapter.send(
                source.chat_id, f"⚠️ No activity for {int(worker.agent_warning // 60) or 1} min. "
                "If the agent does not respond soon, it will be timed out in "
                f"{int((worker.agent_timeout - worker.agent_warning) // 60) or 1} min. "
                "You can continue waiting or use /reset.",
                metadata=_interim_metadata(_status_thread_metadata),
            )
        except Exception as _warn_err:
            logger.debug("Inactivity warning send error: %s", _warn_err)

    def _run_agent_timeout_result(self, worker, turn_ctx: TurnContext) -> dict:
        """Synthetic failed run dict for an inactivity timeout, with the activity-tracker diagnostic;
        interrupts the agent if it is still running so the thread pool worker is freed."""
        from gateway.run import _INTERRUPT_REASON_TIMEOUT, request_hard_interrupt
        session_key, result_holder, tools_holder = turn_ctx.session_key, turn_ctx.result_holder, turn_ctx.tools_holder
        _timed_out_agent = turn_ctx.agent_holder[0]
        _activity = self._agent_activity_summary(_timed_out_agent)
        _last_desc = _activity.get("last_activity_desc", "unknown")
        _secs_ago = _activity.get("seconds_since_activity", 0)
        _cur_tool = _activity.get("current_tool")
        _iter_n = _activity.get("api_call_count", 0)
        _iter_max = _activity.get("max_iterations", 0)
        logger.error(
            "Agent idle for %.0fs (timeout %.0fs) in session %s "
            "| last_activity=%s | iteration=%s/%s | tool=%s",
            _secs_ago, worker.agent_timeout, session_key, _last_desc, _iter_n, _iter_max,
            _cur_tool or "none",
        )
        if _timed_out_agent:
            request_hard_interrupt(_timed_out_agent, _INTERRUPT_REASON_TIMEOUT)
        _timeout_mins = int(worker.agent_timeout // 60) or 1
        _diag_lines = [
            f"⏱️ Agent inactive for {_timeout_mins} min — no tool calls or API responses."
        ]
        if _cur_tool:
            _diag_lines.append(
                f"The agent appears stuck on tool `{_cur_tool}` ({_secs_ago:.0f}s since last "
                f"activity, iteration {_iter_n}/{_iter_max})."
            )
        else:
            _diag_lines.append(
                f"Last activity: {_last_desc} ({_secs_ago:.0f}s ago, "
                f"iteration {_iter_n}/{_iter_max}). "
                "The agent may have been waiting on an API response."
            )
        _diag_lines.append(
            "To increase the limit, set agent.gateway_timeout in config.yaml (value in seconds, 0 "
            "= no limit) and restart the gateway.\nTry again, or use /reset to start fresh."
        )
        return {
            "final_response": "\n".join(_diag_lines),
            "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
            "api_calls": _iter_n,
            "tools": tools_holder[0] or [],
            "history_offset": 0,
            "failed": True,
        }

    async def _run_agent_await_turn_worker(
        self, worker: "GatewayRunner._RunAgentWorker", turn_ctx: TurnContext,
        _interrupt_detected: "asyncio.Event", interrupt_monitor: "asyncio.Task",
    ) -> Any:
        """Poll the executor future (inactivity timeout + backup interrupt checks); return its result,
        or a synthetic failed run dict on inactivity timeout. Polls even with an unlimited timeout
        so the backup interrupt check runs if monitor_for_interrupt() silently died."""
        from gateway.run import _abandon_timed_out_gateway_turn
        agent_holder = turn_ctx.agent_holder
        _warning_fired = False
        while True:
            done, _ = await asyncio.wait({worker.executor_task}, timeout=5.0)
            if done:
                # Prefer the real result even if the watchdog fired in the same window (the run already
                # persisted its reply).
                return worker.executor_task.result()
            if worker.agent_timeout is not None:
                if worker.timeout_fired.is_set():
                    break
                _idle_secs = self._agent_activity_summary(agent_holder[0]).get("seconds_since_activity", 0.0)
                if not _warning_fired and worker.agent_warning is not None and _idle_secs >= worker.agent_warning:
                    _warning_fired = True
                    await self._run_agent_inactivity_warning(worker, turn_ctx.source, turn_ctx._status_thread_metadata)
                if _idle_secs >= worker.agent_timeout:
                    threading.Thread(
                        target=_abandon_timed_out_gateway_turn,
                        kwargs={"agent_holder": agent_holder, **self._reaper_kwargs(worker)},
                        name=f"gateway-turn-reaper-{worker.task_id[:12]}", daemon=True,
                    ).start()
                    break
            await self._run_agent_backup_interrupt_check(turn_ctx, _interrupt_detected, interrupt_monitor)
        return self._run_agent_timeout_result(worker, turn_ctx)

    def _run_agent_evict_on_fallback(self, turn_ctx: TurnContext) -> None:
        """Evict the cached agent when a fallback model activated on a SUCCESSFUL run (so /model shows
        the active model and the next message retries the primary). Skip failed runs: evicting
        would loop bad model → fallback → evict → recreate."""
        from gateway.run import _resolve_gateway_model
        session_key = turn_ctx.session_key
        _agent = turn_ctx.agent_holder[0]
        _result_for_fb = turn_ctx.result_holder[0]
        if _agent is None or not hasattr(_agent, 'model') or (_result_for_fb and _result_for_fb.get("failed")):
            return
        _cfg_model = _resolve_gateway_model()
        # Normalize as AIAgent.__init__ does (vendor prefix stripped on native providers), else the
        # cached agent is evicted every turn, destroying prompt caching.
        with suppress(Exception):
            from hermes_cli.model_normalize import _AGGREGATOR_PROVIDERS, normalize_model_for_provider
            _agent_provider = getattr(_agent, 'provider', '') or ''
            if _agent_provider and _agent_provider not in _AGGREGATOR_PROVIDERS:
                _cfg_model = normalize_model_for_provider(_cfg_model, _agent_provider)
        if _agent.model != _cfg_model and not self._is_intentional_model_switch(session_key, _agent.model):
            self._evict_cached_agent(session_key)

    async def _run_agent_finalize_streaming_tts(self, turn_ctx: TurnContext, adapter: Any) -> None:
        """Finalize the streaming-TTS consumer on the outer event-loop thread (covers early returns
        from run_sync). On drain timeout abort to free the task — audible streams keep whole-file
        suppression, silent streams stay eligible for the whole-file fallback."""
        _stts = turn_ctx.streaming_tts_consumer_holder[0]
        if _stts is None:
            return
        _stts.finish()
        try:
            await _stts.wait_complete(timeout=10.0)
        except Exception as _stts_done_err:
            logger.debug("streaming TTS wait_complete error: %s", _stts_done_err)
        if not _stts.done:
            _stts.abort("streaming TTS finalisation timeout")
            await _stts.wait_complete(timeout=2.0)
        if _stts.suppress_whole_file and adapter is not None:
            _mark_turn = getattr(adapter, "_mark_streaming_tts_completed_turn", None)
            if callable(_mark_turn):
                _mark_turn(turn_ctx.session_key, turn_ctx.run_generation)

    async def _run_agent_drain_pending(
        self, result: Any, adapter: Any, source: SessionSource, session_key: Optional[str]
    ) -> Tuple[Any, Optional[str]]:
        """Dequeue the adapter's pending / interrupt / leftover-steer follow-up as ``(pending_event, pending)``.

        Keyed by session_key (not source.chat_id) to match the adapter's storage keys."""
        from gateway.run import (
            _build_media_placeholder, _dequeue_pending_event, _is_control_interrupt_message
        )
        pending_event = None
        pending = None
        if result and adapter and session_key:
            pending_event = _dequeue_pending_event(adapter, session_key)
            # /queue overflow: promote the next queued event into the consumed "next-up" slot so the
            # recursive drain sees it (keeps FIFO order; a mid-chain /queue can't jump the queue).
            pending_event = self._promote_queued_event(session_key, adapter, pending_event)
            if result.get("interrupted") and not pending_event and result.get("interrupt_message"):
                interrupt_message = result.get("interrupt_message")
                if _is_control_interrupt_message(interrupt_message):
                    logger.info(
                        "Ignoring control interrupt message for session %s: %s",
                        session_key or "?", interrupt_message,
                    )
                else:
                    pending = interrupt_message
            elif pending_event:
                # Transcribe audio BEFORE it becomes the next user turn (real transcript, not a path).
                _pending_text = pending_event.text or ""
                if self._pending_event_audio_paths(pending_event):
                    pending, _ = await self._transcribe_and_echo_pending_voice(
                        pending_event, adapter, source, _pending_text, log_context="Voice-drain",
                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                    )
                    pending = pending or _build_media_placeholder(pending_event)
                else:
                    pending = _pending_text or _build_media_placeholder(pending_event)
                if pending:
                    logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])

        # Leftover /steer (arrived after the last tool batch): deliver as the next user turn.
        if result and not pending and not pending_event and result.get("pending_steer"):
            pending = result.get("pending_steer")
            logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])

        # Safety net: a pending slash command is never passed to the agent as user input.
        if pending and pending.strip().startswith("/"):
            _pending_cmd_word = pending.strip().split(None, 1)[0][1:].lower()
            if _pending_cmd_word:
                with suppress(Exception):
                    from hermes_cli.commands import resolve_command as _rc_pending
                    if _rc_pending(_pending_cmd_word):
                        logger.info(
                            "Discarding command '/%s' from pending queue — "
                            "commands must not be passed as agent input", _pending_cmd_word,
                        )
                        pending_event = None
                        pending = None

        if self._draining and (pending_event or pending):
            logger.info(
                "Discarding pending follow-up for session %s during gateway %s",
                session_key or "?", self._status_action_label(),
            )
            pending_event = None
            pending = None
        return pending_event, pending

    async def _run_agent_deliver_first_response(
        self, turn_ctx: TurnContext, adapter: Any, response: Any, result: Any, stream_task: Any,
    ) -> None:
        """Deliver the first response before a queued follow-up runs, unless streaming already did."""
        session_key = turn_ctx.session_key
        _sc = turn_ctx.stream_consumer_holder[0]
        if _sc and stream_task:
            try:
                await self._await_stream_task(stream_task)
            except Exception as e:
                logger.debug("Stream consumer wait before queued message failed: %s", e)
        # Delivery uses the finalized task result (empty/failure normalization), not raw ``result``.
        _delivery_result = response if isinstance(response, dict) else (result or {})
        first_response = _delivery_result.get("final_response", "")
        _already_streamed = self._run_agent_stream_confirmed_final_delivery(
            _sc, first_response, previewed=bool(_delivery_result.get("response_previewed")),
        )
        # Same silence predicate as the normal path, else this branch leaks the literal marker.
        if self._is_intentional_silence(_delivery_result, first_response):
            logger.info(
                "Queued follow-up for session %s: suppressing intentional silence marker before continuing.",
                session_key or "?",
            )
        elif first_response:
            logger.info(
                "Queued follow-up for session %s: final text delivery confirmed; delivering explicit media before continuing."
                if _already_streamed else
                "Queued follow-up for session %s: final stream delivery not confirmed; sending first response before continuing.",
                session_key or "?",
            )
            try:
                await self._deliver_queued_first_response(
                    first_response, source=turn_ctx.source, adapter=adapter,
                    metadata=turn_ctx._status_thread_metadata, event_message_id=turn_ctx.event_message_id,
                    text_already_delivered=_already_streamed,
                    deliver_media=not _delivery_result.get("failed"), stream_consumer=_sc,
                )
            except Exception as e:
                logger.warning("Failed to send first response before queued message: %s", e)
        # Release deferred bg-review notifications: pop (no double-fire in base.py's finally) and call.
        _bg_cb = self._pop_post_delivery_callback(adapter, session_key, turn_ctx.run_generation)
        if callable(_bg_cb):
            with suppress(Exception):
                _bg_result = _bg_cb()
                if inspect.isawaitable(_bg_result):
                    await _bg_result

    async def _run_agent_queued_followup(
        self, turn_ctx: TurnContext, adapter: Any, pending: Optional[str], pending_event: Any,
        response: Any, result: Any, stream_task: Any,
    ) -> Any:
        """Run the queued / interrupting follow-up as the next turn (recursive ``_run_agent``)."""
        from gateway.platforms.base import merge_pending_message_event
        from gateway.run import _preserve_queued_followup_history_offset
        source, session_id, session_key, run_generation = (
            turn_ctx.source, turn_ctx.session_id, turn_ctx.session_key, turn_ctx.run_generation,
        )
        _interrupt_depth, history, _status_thread_metadata = (
            turn_ctx._interrupt_depth, turn_ctx.history, turn_ctx._status_thread_metadata,
        )
        logger.debug("Processing pending message: '%s...'", pending[:40])

        # Clear the interrupt event so the recursive _run_agent isn't re-interrupted (infinite loop).
        _active = getattr(adapter, "_active_sessions", None) if adapter else None
        if _active and session_key and session_key in _active:
            _active[session_key].clear()

        # Cap recursion depth (user keeps sending while the agent keeps failing).
        # (#816)
        if _interrupt_depth >= self._MAX_INTERRUPT_DEPTH:
            logger.warning(
                "Interrupt recursion depth %d reached for session %s — "
                "queueing message instead of recursing.", _interrupt_depth, session_key,
            )
            adapter = self._adapter_for_source(source)
            if adapter and pending_event:
                merge_pending_message_event(adapter._pending_messages, session_key, pending_event)
            elif adapter and hasattr(adapter, 'queue_message'):
                adapter.queue_message(session_key, pending)
            return turn_ctx.result_holder[0] or {"final_response": response, "messages": history}

        # Interrupted: discard the response ("Operation interrupted." is noise).
        if not result.get("interrupted"):
            await self._run_agent_deliver_first_response(turn_ctx, adapter, response, result, stream_task)

        updated_history = result.get("messages", history)
        next_source, next_message, next_session_key = source, pending, session_key
        # message_type is carried into the recursive call so queued voice turns can stream TTS.
        next_message_id = next_channel_prompt = next_message_type = None
        # See #60671.
        if pending_event is not None:
            next_source = getattr(pending_event, "source", None) or source
            if self._is_goal_continuation_event(pending_event) and not self._goal_still_active_for_session(session_id):
                logger.info(
                    "Discarding stale goal continuation for session %s — goal is no longer active",
                    session_key or "?",
                )
                return result
            # Resolve the follow-up's session key BEFORE preparing the inbound text: native image
            # paths are buffered under the key given and consumed under next_session_key.
            try:
                next_session_key = self._session_key_for_source(next_source)
            except Exception:
                logger.debug(
                    "Queued follow-up session-key resolution failed; reusing %s",
                    session_key or "?", exc_info=True,
                )
            next_message = await self._prepare_profile_scoped_inbound_message_text(
                event=pending_event, source=next_source, history=updated_history, session_key=next_session_key,
            )
            if next_message is None:
                return result
            next_message_id = self._reply_anchor_for_event(pending_event)
            next_channel_prompt = getattr(pending_event, "channel_prompt", None)
            next_message_type = getattr(pending_event, "message_type", None)

        # Clear the prior turn's streaming-TTS completion marker so the recursive turn isn't suppressed.
        # See #60671.
        _clear_adapter = self._adapter_for_source(source)
        _completed_turns = getattr(_clear_adapter, "_streaming_tts_completed_turns", None)
        _prior_key = getattr(_clear_adapter, "_streaming_tts_turn_key", None)
        if _completed_turns is not None and callable(_prior_key) and session_key and run_generation is not None:
            _pk = _prior_key(session_key, run_generation)
            if _pk:
                _completed_turns.discard(_pk)

        # Restart the typing indicator; the outer typing task may be stale.
        if _clear_adapter:
            with suppress(Exception):
                await _clear_adapter.send_typing(source.chat_id, metadata=_status_thread_metadata)

        # Re-baseline the cached agent's message_count before recursing, else the coherence guard
        # rebuilds on OUR OWN flushed rows (the outer handler re-baselines only after the chain).
        # Re-baseline the cached agent's message_count snapshot before recursing into the in-band queued
        # (/queue) follow-up turn. The first turn has completed and flushed its own user + assistant rows to
        # the SessionDB, so the cross-process coherence guard (#45966) — which this recursive _run_agent
        # call re-enters — would otherwise see the grown on-disk count against the stale build-time snapshot
        # and rebuild the agent on THIS process's OWN writes, destroying the prompt-cache prefix #46237 was
        # merged to preserve. The existing re-baseline in _handle_message_with_agent only runs after the
        # whole _run_agent chain unwinds — too late for the in-band follow-up. Use the same (session_key,
        # session_id) the recursive call runs under so the snapshot matches exactly what the follow-up's
        # guard will consult. Fail-safe in helper.
        await self._refresh_agent_cache_message_count(session_key, session_id)

        followup_result = await self._run_agent(
            message=next_message, context_prompt=turn_ctx.context_prompt, history=updated_history,
            source=next_source, session_id=session_id, session_key=next_session_key,
            run_generation=run_generation, _interrupt_depth=_interrupt_depth + 1,
            event_message_id=next_message_id, channel_prompt=next_channel_prompt,
            message_type=next_message_type,
        )
        return _preserve_queued_followup_history_offset(result, followup_result)

    async def _run_agent_cleanup_turn_tasks(
        self, turn_ctx: TurnContext, *, progress_task: Any, log_task: Any, interrupt_monitor: "asyncio.Task",
        _notify_task: "asyncio.Task", tracking_task: "asyncio.Task", stream_task: Any,
    ) -> None:
        """``finally`` half of a turn: cancel background tasks, flush stream, release the session slot."""
        stream_consumer_holder, session_key = turn_ctx.stream_consumer_holder, turn_ctx.session_key
        for task in (progress_task, log_task, interrupt_monitor, _notify_task):
            if task:
                task.cancel()

        if stream_task:
            # No stream consumer was created: nothing to flush, cancel instead of waiting out 5s.
            if not (stream_consumer_holder and stream_consumer_holder[0] is not None):
                stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stream_task
            else:
                await self._await_stream_task(stream_task)

        # Abort + bounded wait for streaming TTS: covers paths where normal finalisation was skipped.
        _stts_finally = turn_ctx.streaming_tts_consumer_holder[0]
        # See #60671.
        if _stts_finally is not None and not _stts_finally.done:
            _stts_finally.abort("cleanup")
            with suppress(Exception):
                await _stts_finally.wait_complete(timeout=2.0)

        tracking_task.cancel()
        if session_key:
            # Release the slot only if this run's generation still owns it (/stop or /new may have
            # installed its own state).
            self._release_running_agent_state(session_key, run_generation=turn_ctx.run_generation)
        if self._draining:
            self._update_runtime_status("draining")

        for task in (progress_task, log_task, interrupt_monitor, tracking_task, _notify_task):
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # A background task that died of a real error must not abort the cleanup path.
                    logger.debug("background turn task failed during cleanup", exc_info=True)

    async def _run_agent_edit_streamed_message(
        self, _sc, source, response, content, *, _sk, ok, fail_result, fail_exc,
    ) -> None:
        """Edit the stream consumer's message in place with ``content``; on success mark
        ``response["already_sent"]`` and log ``ok``. ``fail_result`` (None = trust the call) logs a
        returned failure as ``(session, error)``; ``fail_exc`` logs an exception as ``(session, exc)``."""
        try:
            _res = await _sc.adapter.edit_message(
                chat_id=source.chat_id, message_id=_sc.message_id, content=content, finalize=True,
            )
        except Exception as _edit_err:
            logger.warning(fail_exc, _sk, _edit_err)
            return
        if fail_result is not None and not getattr(_res, "success", True):
            logger.warning(fail_result, _sk, getattr(_res, "error", None))
            return
        response["already_sent"] = True
        logger.info(*ok)

    async def _run_agent_mark_streamed_delivery(self, response: Any, turn_ctx: TurnContext) -> None:
        """Set ``response["already_sent"]`` when streaming already delivered the final reply.

        Never when the agent failed (the error is unseen content) or on "(empty)". Both suppression
        flags reflect call success, not content, so reconcile against the recorded turn-final
        payload: a mismatch (False, incl. payload-less split delivery) never suppresses; None (no
        record) keeps legacy trust."""
        _sc, source, session_key = turn_ctx.stream_consumer_holder[0], turn_ctx.source, turn_ctx.session_key
        if not isinstance(response, dict) or response.get("failed"):
            return
        _final = response.get("final_response") or ""
        _is_empty_sentinel = not _final or _final == "(empty)"
        # response_previewed: only suppress if that EXACT text was delivered, not unrelated commentary.
        # Unrelated commentary/progress must not be mistaken for the final response (#14238).
        _previewed = bool(response.get("response_previewed"))
        _content_delivered = bool(_sc and getattr(_sc, "final_content_delivered", False))
        # #71643: a *successful* finalize edit can still carry only the last preview snapshot — deltas
        # generated between that edit and stream completion never reach any API call, and both suppression
        # flags are set from the call's success rather than its content. Reconcile the consumer's recorded
        # turn-final payload against the completed response: on a demonstrable mismatch (False) neither
        # final_response_sent nor final_content_delivered may suppress the normal final send. False also
        # covers payload-less multi-message split delivery (#78541). None (no record on a non-split legacy
        # path) keeps legacy trust; the failed-finalize family (#51828 / #33793) is unaffected because those
        # paths leave the flags False or record the complete fallback payload.
        _stale_finalized = False
        if _content_delivered and not _is_empty_sentinel:
            _matcher = getattr(_sc, "delivered_final_matches", None)
            if callable(_matcher):
                with suppress(Exception):
                    _stale_finalized = _matcher(_final) is False
            if _stale_finalized:
                _content_delivered = False
        # Plugin hooks may append content after streaming finished — then send the final version.
        _transformed = bool(response.get("response_transformed"))
        # Suppress the normal send only when the actual final reply reached the user.
        _streamed = self._run_agent_stream_confirmed_final_delivery(_sc, _final, previewed=_previewed)
        if _is_empty_sentinel:
            return
        _sk = session_key or "?"
        if not _transformed and (_streamed or _content_delivered):
            logger.info(
                "Suppressing normal final send for session %s: final delivery already confirmed (streamed=%s previewed=%s content_delivered=%s).",
                _sk, _streamed, _previewed, _content_delivered,
            )
            response["already_sent"] = True
        elif not _transformed and _stale_finalized and _sc is not None:
            # Stale finalize: edit the streamed message up to the complete response (on failure the
            # normal send delivers). Not for split delivery — message_id is only the LAST chunk.
            _sc_msg_id = _sc.message_id
            if getattr(_sc, "_turn_split_delivery", False):
                logger.info(
                    "Stale streamed finalize detected for session %s on a multi-message split; skipping the in-place reconciliation edit and delivering the complete response via normal final send (#78541).",
                    _sk,
                )
            elif _sc_msg_id and _sc_msg_id != "__no_edit__" and getattr(_sc, "adapter", None) is not None:
                await self._run_agent_edit_streamed_message(
                    _sc, source, response, _final, _sk=_sk,
                    ok=("Reconciled stale streamed finalize for session %s: edited message %s with the complete response (#71643).", _sk, _sc_msg_id),
                    fail_result="Stale-finalize reconciliation edit failed for session %s (%s); sending complete response via normal final send.",
                    fail_exc="Stale-finalize reconciliation edit failed for session %s: %s; sending complete response via normal final send.",
                )
            else:
                logger.info(
                    "Stale streamed finalize detected for session %s with no editable message; delivering complete response via normal final send (#71643).",
                    _sk,
                )
        elif _transformed and _sc is not None:
            # Transformed after streaming: edit the streamed message instead of sending a duplicate.
            if _sc.message_id:
                await self._run_agent_edit_streamed_message(
                    _sc, source, response, response["final_response"], _sk=_sk,
                    ok=("Edited streamed message %s for session %s to include plugin-transformed content.", _sc.message_id, _sk),
                    fail_result=None, fail_exc="Failed to edit streamed message for session %s: %s",
                )
        elif _sc is not None:
            # DUPLICATE-RISK DIAGNOSTIC: a stream consumer existed but suppression did NOT fire; log the
            # decision inputs ("signal never set" vs "ack-pending race").
            logger.warning(
                "Normal final-send NOT suppressed despite active stream consumer for session %s: "
                "streamed=%s previewed=%s content_delivered=%s transformed=%s final_len=%d — "
                "possible duplicate send (see wecom ack-timeout RCA).",
                _sk, _streamed, _previewed, _content_delivered, _transformed, len(_final),
            )

    def _run_agent_schedule_bubble_cleanup(self, response: Any, _cleanup_adapter: Any, turn_ctx: TurnContext) -> None:
        """Schedule deletion of tracked temporary progress bubbles after the final response lands.

        Failed runs keep them as breadcrumbs. Only on adapters with ``delete_message``; failures swallowed."""
        from gateway.run import safe_schedule_threadsafe
        _cleanup_msg_ids, session_key = turn_ctx._cleanup_msg_ids, turn_ctx.session_key
        if not (
            turn_ctx._cleanup_progress
            and _cleanup_adapter is not None
            and _cleanup_msg_ids
            and session_key
            and isinstance(response, dict)
            and not response.get("failed")
            and hasattr(_cleanup_adapter, "register_post_delivery_callback")
        ):
            return
        _ids_snapshot = list(_cleanup_msg_ids)
        _chat_id_snapshot = turn_ctx.source.chat_id
        _loop_snapshot = asyncio.get_running_loop()

        def _cleanup_temp_bubbles() -> None:
            async def _delete_all() -> None:
                for _mid in _ids_snapshot:
                    with suppress(Exception):
                        await _cleanup_adapter.delete_message(_chat_id_snapshot, _mid)
            with suppress(Exception):
                safe_schedule_threadsafe(
                    _delete_all(), _loop_snapshot, logger=logger,
                    log_message="Temp bubble cleanup scheduling error",
                )

        try:
            _cleanup_adapter.register_post_delivery_callback(
                session_key, _cleanup_temp_bubbles, generation=turn_ctx.run_generation,
            )
        except Exception as _rpe:
            logger.debug("Post-delivery cleanup registration failed: %s", _rpe)

    def _run_agent_bind_turn_wiring(
        self, turn_ctx: TurnContext, turn_runner: TurnRunner, source: SessionSource,
        event_message_id: Optional[str], _native_slack_task_cards: bool,
    ) -> Optional[Dict[str, Any]]:
        """Resolve progress threading, then publish progress metadata and the sync→async bridges onto
        ``turn_ctx`` (the one-slot holders shared with run_sync's executor thread are TurnContext
        defaults). Returns ``_status_thread_metadata``."""
        turn_ctx._progress_metadata, turn_ctx._progress_reply_to, _status_thread_metadata = (
            self._run_agent_progress_threading(source, event_message_id, _native_slack_task_cards)
        )
        # Bridges: sync step/event/status callbacks → async hooks.emit and adapter.send.
        turn_ctx._loop_for_step = asyncio.get_running_loop()
        turn_ctx._hooks_ref = self.hooks
        turn_ctx._step_callback_sync = turn_runner._step_callback_sync
        turn_ctx._event_callback_sync = turn_runner._event_callback_sync
        turn_ctx._status_callback_sync = turn_runner._status_callback_sync
        turn_ctx._status_adapter = self._adapter_for_source(source)
        turn_ctx._status_chat_id = source.chat_id
        turn_ctx._status_thread_metadata = _status_thread_metadata
        return _status_thread_metadata

    async def _run_agent_notify_long_running(
        self, disp: "GatewayRunner._RunAgentDisplay", turn_ctx: TurnContext, _executor_task_holder: list,
    ) -> None:
        """Periodic "still working" heartbeat, edited in place where supported. Stops once this run
        no longer owns the session slot or the executor finished. ``_executor_task_holder[0]`` is
        bound just after this task is scheduled (reads as None until then).

        Interval: agent.gateway_notify_interval / HERMES_AGENT_NOTIFY_INTERVAL (default 180s; 0 or
        long_running_notifications=off disables)."""
        from gateway.run import _float_env, _interim_metadata, _non_conversational_metadata
        _notify_start = time.time()
        _NOTIFY_INTERVAL = _float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180)
        _long_running_mode = disp._display_surface_mode("long_running_notifications", default=True, allow_generic=True)
        if _NOTIFY_INTERVAL <= 0 or _long_running_mode == "off":
            return
        source, session_key, agent_holder = turn_ctx.source, turn_ctx.session_key, turn_ctx.agent_holder
        _status_thread_metadata = turn_ctx._status_thread_metadata
        _notify_adapter = self._adapter_for_source(source)
        if not _notify_adapter:
            return
        _heartbeat_msg_id: Optional[str] = None
        while True:
            await asyncio.sleep(_NOTIFY_INTERVAL)
            if not self._should_emit_long_running_notification(
                session_key, agent_holder[0], _executor_task_holder[0]
            ):
                break
            _elapsed_mins = int((time.time() - _notify_start) // 60)
            # Terse heartbeat by default; the iteration counter is gated on busy_ack_detail.
            _status_detail = ""
            _want_iteration_detail = bool(
                disp.resolve_display_setting(disp.user_config, disp.platform_key, "busy_ack_detail", True)
            )
            _a = self._agent_activity_summary(agent_holder[0])
            with suppress(Exception):
                if _a:
                    _parts = []
                    if _want_iteration_detail:
                        _parts.append(f"iteration {_a['api_call_count']}/{_a['max_iterations']}")
                    _action = _a.get("current_tool") or _a.get("last_activity_desc")
                    if _action:
                        _parts.append(str(_action))
                    if _parts:
                        _status_detail = " — " + ", ".join(_parts)
            _heartbeat_text = (
                disp._generic_status_phrase("status")
                if _long_running_mode == "generic"
                else f"⏳ Working — {_elapsed_mins} min{_status_detail}"
            )
            try:
                _notify_res = None
                if _heartbeat_msg_id:
                    try:
                        _notify_res = await _notify_adapter.edit_message(source.chat_id, _heartbeat_msg_id, _heartbeat_text)
                    except Exception as _ee:
                        logger.debug("Heartbeat edit failed: %s", _ee)
                        _notify_res = None
                if not (_notify_res and getattr(_notify_res, "success", False)):
                    _notify_res = await _notify_adapter.send(
                        source.chat_id, _heartbeat_text,
                        metadata=_interim_metadata(_non_conversational_metadata(_status_thread_metadata, platform=source.platform)),
                    )
                    if getattr(_notify_res, "success", False) and getattr(_notify_res, "message_id", None):
                        _heartbeat_msg_id = str(_notify_res.message_id)
                        if turn_ctx._cleanup_progress:
                            turn_ctx._cleanup_msg_ids.append(_heartbeat_msg_id)
            except Exception as _ne:
                logger.debug("Long-running notification error: %s", _ne)

    async def _run_agent_inner(
        self, message: str, context_prompt: str, history: List[Dict[str, Any]],
        source: SessionSource, session_id: str, session_key: str = None,
        run_generation: Optional[int] = None, _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None, inbound_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None, moa_config: Optional[dict] = None,
        persist_user_message: Optional[Any] = None, persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None, message_type: Optional[str] = None,
        startup_resume: bool = False,
    ) -> Dict[str, Any]:
        """Run the agent; returns the full run_conversation result dict.

        Keys: "final_response", "messages", "api_calls", "completed"."""
        if self._get_proxy_url():
            return await self._run_agent_via_proxy(
                message=message, context_prompt=context_prompt, history=history, source=source,
                session_id=session_id, session_key=session_key, run_generation=run_generation,
                event_message_id=event_message_id,
            )

        from run_agent import AIAgent

        disp = self._run_agent_display_settings(source)
        turn_ctx, turn_runner, _cleanup_adapter = self._run_agent_build_turn_context(
            disp, AIAgent, message=message, source=source, session_key=session_key,
            run_generation=run_generation, context_prompt=context_prompt, history=history,
            session_id=session_id, _interrupt_depth=_interrupt_depth,
            event_message_id=event_message_id, inbound_message_id=inbound_message_id,
            channel_prompt=channel_prompt, moa_config=moa_config,
            persist_user_message=persist_user_message,
            persist_user_timestamp=persist_user_timestamp,
            persist_user_display_kind=persist_user_display_kind,
            startup_resume=startup_resume,
        )
        _status_thread_metadata = self._run_agent_bind_turn_wiring(
            turn_ctx, turn_runner, source, event_message_id, disp._native_slack_task_cards,
        )
        self._run_agent_start_streaming_tts(
            source, message_type, _status_thread_metadata, turn_ctx.streaming_tts_consumer_holder,
        )

        # Progress sender drains BOTH tool-progress lines and thinking bubbles (needs_progress_queue).
        spawn = asyncio.create_task
        progress_task = spawn(turn_runner.send_progress_messages()) if disp.needs_progress_queue else None
        log_task = spawn(self._run_agent_write_tool_log(disp.log_queue)) if disp.log_mode_enabled else None
        # The stream consumer is created inside run_sync; this task polls for it.
        stream_task = spawn(self._run_agent_stream_consumer_task(turn_ctx.stream_consumer_holder))
        tracking_task = spawn(self._run_agent_track_agent(turn_ctx))
        _interrupt_detected = asyncio.Event()  # shared with backup check
        interrupt_monitor = spawn(self._run_agent_monitor_for_interrupt(turn_ctx, _interrupt_detected))
        # Periodic "still working" notifications so the user knows the agent hasn't died.
        _executor_task_holder: list = [None]  # bound once the executor future exists (see below)
        _notify_task = spawn(self._run_agent_notify_long_running(disp, turn_ctx, _executor_task_holder))

        try:
            # run_sync is TurnRunner.run_sync (bound method; executor call unchanged).
            worker = self._run_agent_start_turn_worker(turn_ctx, turn_runner.run_sync)
            _executor_task_holder[0] = worker.executor_task  # read late by _notify_long_running
            response = await self._run_agent_await_turn_worker(worker, turn_ctx, _interrupt_detected, interrupt_monitor)
            self._run_agent_evict_on_fallback(turn_ctx)

            # Interrupted OR queued message (/queue)?
            result = turn_ctx.result_holder[0]
            adapter = self._adapter_for_source(source)
            await self._run_agent_finalize_streaming_tts(turn_ctx, adapter)
            pending_event, pending = await self._run_agent_drain_pending(result, adapter, source, session_key)
            if pending_event or pending:
                return await self._run_agent_queued_followup(
                    turn_ctx, adapter, pending, pending_event, response, result, stream_task,
                )
        finally:
            await self._run_agent_cleanup_turn_tasks(
                turn_ctx, progress_task=progress_task, log_task=log_task, interrupt_monitor=interrupt_monitor,
                _notify_task=_notify_task, tracking_task=tracking_task, stream_task=stream_task,
            )

        await self._run_agent_mark_streamed_delivery(response, turn_ctx)
        self._run_agent_schedule_bubble_cleanup(response, _cleanup_adapter, turn_ctx)
        return response
