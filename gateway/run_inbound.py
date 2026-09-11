"""run_inbound.py — facade: composes the turn mixins (split from one 2063-line class)."""

from __future__ import annotations
from gateway.run_inbound_hm import GatewayInboundHmMixin
from gateway.run_inbound_enrich import GatewayInboundEnrichMixin
from gateway.run_inbound_prepend import GatewayInboundPrependMixin

import logging
import asyncio
import concurrent.futures
import dataclasses
import os
import re
import time
from contextlib import suppress
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_common import _UNSET
from gateway.session import (
    SessionSource, is_shared_multi_user_session, neutralize_untrusted_inline_text
)
from gateway.turn_lease import TurnLeaseTimeoutError
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner  # noqa: F401

logger = logging.getLogger("gateway.run")

class GatewayInboundMixin(GatewayInboundHmMixin, GatewayInboundEnrichMixin, GatewayInboundPrependMixin):
    """Inbound message pipeline (_handle_message, text/media preparation, durable-turn markers, plugin injection) for GatewayRunner."""

    # Reply → choice for a pending slash-confirm prompt; the command spelling wins over the
    # bang/slash-stripped free-text spelling.
    _SLASH_CONFIRM_CMD_CHOICES = {
        "approve": "once", "yes": "once", "ok": "once", "confirm": "once",
        "always": "always", "remember": "always",
        "cancel": "cancel", "no": "cancel", "deny": "cancel", "nevermind": "cancel",
    }

    _SLASH_CONFIRM_TEXT_CHOICES = {
        "approve": "once", "approve once": "once", "once": "once",
        "always": "always", "always approve": "always",
        "cancel": "cancel", "nevermind": "cancel", "no": "cancel",
    }

    # Idle-path built-ins with bespoke flow (confirmations, prompt rewrites, one-shot MoA), each
    # handled by ``_hm_cmd_<name>`` → ``(handled, result)``; ``(False, None)`` falls through to the agent.
    _HM_CANONICAL_COMMANDS = frozenset({
        "new", "start", "egress", "learn", "plan", "init", "blueprint", "undo", "queue", "steer", "moa",
    })

    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """Handle an incoming message from any platform: auth → command check → running-agent
        interrupt → get/create session → build context → run agent → return response."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        _admitted = await self._hm_admit_event(event)
        if _admitted is None:
            return None
        event, source, is_internal = _admitted
        # TERMINAL-DECLINE LATCH TEARDOWN. Deliberately placed AFTER admission,
        # not on the adapter's raw inbound: profile routing, the ignored-channel
        # guard, plugin hooks and user authorization all reject events above,
        # and a rejected event must not be able to clear a refusal belonging to
        # an active turn. This is also the single entry point every lane shares
        # — Discord interaction passthrough builds its own MessageEvent and
        # calls handle_message directly, so a teardown on the relay's inbound
        # handler left those turns muted.

        _paused_notice = self._hm_estop_gate(event, source, is_internal)
        if _paused_notice is not None:
            return _paused_notice

        _quick_key = self._session_key_for_source(source)
        _reply = await self._hm_pending_reply_intercepts(event, source, _quick_key)
        if _reply is not None:
            return _reply

        # Evict a leaked/reaped ``_running_agents`` slot before the busy-session fast-path.
        self._hm_evict_idle_stale_agent(_quick_key)
        if self._is_session_running(_quick_key):
            self._hm_evict_reaped_agent(_quick_key)
        if self._is_session_running(_quick_key):
            return await self._hm_handle_running_session_message(event, source, _quick_key)

        _handled, _result = await self._hm_dispatch_idle_commands(event, source, _quick_key)
        if _handled:
            return _result

        # Pending exec approvals go through /approve and /deny only — no bare-text matching, or a
        # conversational "yes" would execute a dangerous command.
        if not is_internal:
            if await asyncio.to_thread(self._is_telegram_topic_root_lobby, source):
                # Debounced so a user who forgets about topic mode doesn't get ten reminders.
                if self._should_send_telegram_lobby_reminder(source):
                    return self._telegram_topic_root_lobby_message()
                return None
            # External-drain new-turn gate: when NAS engaged an external drain (.drain_request.json,
            # seen by _drain_control_watcher), refuse to START new turns so the in-flight set can
            # only fall to zero. Reversible.
            if self._external_drain_active:
                logger.info("Refusing new turn for session %s — external drain active.", _quick_key)
                return (
                    "⏳ This agent is draining for a maintenance action and isn't "
                    "accepting new turns right now. It'll be back in a moment — "
                    "please resend shortly."
                )

        # Claim this session before any await: many awaits sit between here and _run_agent
        # registering the real AIAgent; without this sentinel a second message during any of them
        # passes the "already running" guard and spins up a duplicate agent for the same session.
        _active_session_lease, _limit_message = self._claim_active_session_slot(_quick_key, source)
        if _limit_message is not None:
            logger.info("Rejecting new active session %s: max_concurrent_sessions reached", _quick_key)
            return _limit_message

        event, source, is_internal = self._hm_rescue_orphaned_fifo(event, source, is_internal, _quick_key)

        _claim_state = self._session_state(_quick_key)
        if _active_session_lease is not None:
            _claim_state.turn.lease = _active_session_lease
        _claim_state.turn.agent = _AGENT_PENDING_SENTINEL
        _claim_state.turn.started_ts = time.time()
        self._persist_active_agents()
        _run_generation = self._begin_session_run_generation(_quick_key)

        try:
            try:
                _agent_result = await self._handle_message_with_agent(event, source, _quick_key, _run_generation)
            except TurnLeaseTimeoutError as exc:
                # A rejected message, not a completed turn: return before the /goal judge so it
                # cannot consume the resend notice and enqueue a synthetic continuation loop.
                logger.error(
                    "Rejecting turn for routing key %s on session %s after "
                    "turn-lease timeout; transcript load was not started and "
                    "the user must resend",
                    _quick_key, exc.session_id,
                )
                return (
                    "⏳ Another turn is still running on this session. To "
                    "protect the transcript, this message was not processed. "
                    "Wait for the active turn to finish, then resend it."
                )
            try:
                await self._run_post_turn_hooks(
                    agent_result=_agent_result, source=source, is_internal=is_internal, event=event,
                )
            except Exception as _goal_exc:
                logger.debug("post-turn hook failed: %s", _goal_exc)
            return _agent_result
        finally:
            # MoA one-shot restore must run on EVERY exit path (success, exception, interrupt):
            # the restore data lives on the per-turn event and would leak permanently otherwise.
            self._restore_moa_one_shot(event, _quick_key)
            self._restore_pending_one_turn_model_override(_quick_key)
            # SIGKILL/OOM skips finally, leaving the durable marker for the next unclean startup's
            # recovery pass.
            await self._clear_durable_active_turn(event)
            # Unconditional, idempotent release without a run_generation guard: evicts the zombie
            # left when session_reset bumps the generation mid-flight (gen-N's guarded release in
            # _run_agent returns False; a sentinel-only check would lock forever).
            self._release_running_agent_state(_quick_key)
            # Turn lease is keyed by (routing key, run generation) so this unwind can only free
            # the lease its own turn acquired, never a newer turn's.
            # Unconditional release covers every exit path. _release_running_agent_state is idempotent
            # (pop-on-absent is harmless) and, called without a run_generation guard, always clears the slot
            # regardless of which generation it holds. This evicts the zombie left when session_reset bumps
            # the generation (N -> N+1) mid-flight: gen-N's guarded release inside _run_agent returns False,
            # and the old sentinel-only check here missed the leftover real agent — locking the session out
            # forever (#28686).
            self._release_turn_lease(_quick_key, _run_generation)

    def _restore_moa_one_shot(self, event: "MessageEvent", quick_key: str) -> None:
        """Revert a ``/moa <prompt>`` one-shot model override after its turn (called from the
        message-handling ``finally``). ``_moa_restore_override`` holds the prior per-session
        override (``None`` = clear the MoA override outright)."""
        if not getattr(event, "_moa_disable_after_turn", False):
            return
        with suppress(Exception):
            self._session_state(quick_key).conversation.model_override = getattr(event, "_moa_restore_override", None)
            self._evict_cached_agent(quick_key)

    def _restore_pending_one_turn_model_override(self, session_key: str) -> None:
        """Restore a per-session model override after ``/model --once`` runs."""
        if not session_key:
            return
        try:
            _otr_state = self._peek_session_state(session_key)
            snapshot = _otr_state.conversation.one_turn_restore if _otr_state else None
            if _otr_state is not None:
                _otr_state.conversation.one_turn_restore = None
            if snapshot:
                self._restore_session_model_override(session_key, snapshot)
        except Exception:
            logger.debug("Failed to restore one-turn model override", exc_info=True)

    def _prefix_inbound_sender_context(self, event: MessageEvent, source: SessionSource, message_text: str) -> str:
        """Attribute the sender in shared multi-user sessions and prepend history-backfill channel context."""
        _is_shared_multi_user = is_shared_multi_user_session(
            source, group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
        )
        if _is_shared_multi_user and source.user_name:
            # Display names are attacker-influenceable: neutralize newlines/control chars or a
            # hostile name masquerades as a fake markdown section (mirrors build_session_context_prompt).
            _safe_user_name = neutralize_untrusted_inline_text(source.user_name)
            # Slack: expose the CURRENT speaker's verifiable `<@U...>` id so "mention me again" has a
            # trusted target (display names are ambiguous). user_id comes from the envelope, not user-editable.
            # See #17916.
            if source.platform == Platform.SLACK and source.user_id:
                _safe_user_name = f"{_safe_user_name} | Slack user <@{source.user_id}>"
            message_text = f"[{_safe_user_name}] {message_text}"
        # After the sender-prefix so the prefix applies only to the trigger message, not the backfill.
        if getattr(event, "channel_context", None):
            message_text = f"{event.channel_context}\n\n[New message]\n{message_text}"
        return message_text

    @staticmethod
    def _classify_inbound_media(
        event: MessageEvent, pending_stt_prepared: bool
    ) -> Tuple[list, list, list, list]:
        """Split ``event.media_urls`` into (image, STT-voice, audio-file, video) paths. Per-attachment
        MIME wins over the message-level type (a document sent alongside an image must not be routed
        as an image). MessageType.AUDIO / mixed DOCUMENT audio is a file attachment, never STT."""
        from gateway.run import _event_media_is_audio, _event_media_is_image, _event_media_is_stt_input
        image_paths, audio_paths, audio_file_paths, video_paths = [], [], [], []
        for i, path in enumerate(event.media_urls or []):
            mtype = event.media_types[i] if i < len(event.media_types) else ""
            if _event_media_is_image(event, i):
                image_paths.append(path)
            if _event_media_is_audio(event, i):
                if event.message_type in {MessageType.AUDIO, MessageType.DOCUMENT}:
                    audio_file_paths.append(path)
                elif not pending_stt_prepared and _event_media_is_stt_input(event, i):
                    audio_paths.append(path)
            if mtype.startswith("video/") or (not mtype and event.message_type == MessageType.VIDEO):
                video_paths.append(path)
        return image_paths, audio_paths, audio_file_paths, video_paths

    async def _echo_stt_transcripts(
        self, adapter, source: SessionSource, transcripts: List[str], *, metadata=None, log_context: str = "Transcript"
    ) -> None:
        """Send each transcript back as ``🎙️ "…"`` (best-effort; failures are logged, never raised)."""
        for tx in transcripts:
            try:
                await adapter.send(source.chat_id, f'🎙️ "{tx}"', metadata=metadata)
            except Exception as echo_exc:
                logger.debug("%s echo failed (non-fatal): %s", log_context, echo_exc)

    @staticmethod
    def _inbound_attachment_display_name(path: str) -> Tuple[str, str]:
        """``(display_name, agent_visible_path)``: cache filename is ``<id>_<id>_<original>``; the
        path is translated to the in-container mount under a Docker backend."""
        from tools.credential_files import to_agent_visible_cache_path
        basename = os.path.basename(path)
        parts = basename.split("_", 2)
        return re.sub(r'[^\w.\- ]', '_', parts[2] if len(parts) >= 3 else basename), to_agent_visible_cache_path(path)

    async def _inbound_model_context_length(self, source: SessionSource, session_key: str) -> int:
        """Context length of the model this turn runs on. A global ``model.context_length`` pin
        belongs to the configured model, not a /model or channel override; custom-provider limits win."""
        from gateway.run import _load_gateway_config
        from agent.model_metadata import get_model_context_length_async

        _msg_config_ctx = None
        _msg_cfg = None
        _msg_model_cfg = {}
        _msg_custom_providers = []
        with suppress(Exception):
            _msg_cfg = _load_gateway_config()
            _msg_model_cfg = _msg_cfg.get("model", {})
            if isinstance(_msg_model_cfg, dict):
                _msg_raw_ctx = _msg_model_cfg.get("context_length")
                if _msg_raw_ctx is not None:
                    _msg_config_ctx = int(_msg_raw_ctx)
            try:
                from hermes_cli.config import get_compatible_custom_providers

                _msg_custom_providers = get_compatible_custom_providers(_msg_cfg)
            except Exception:
                _msg_custom_providers = _msg_cfg.get("custom_providers") or []
        # GatewayRunner has no self._model/self._base_url; resolve the session's actual runtime.
        _msg_model, _msg_runtime = self._resolve_session_agent_runtime(
            source=source, session_key=session_key, user_config=_msg_cfg,
        )
        _msg_base_url = _msg_runtime.get("base_url") or ""
        if isinstance(_msg_model_cfg, dict):
            _msg_configured_model = _msg_model_cfg.get("default") or _msg_model_cfg.get("model")
        else:
            _msg_configured_model = _msg_model_cfg  # (no dict → no pin was read; ctx is already None)
        if _msg_model != _msg_configured_model:
            _msg_config_ctx = None
        if _msg_config_ctx is not None:
            try:
                from hermes_cli.route_identity import should_clear_context_pin_async

                if await should_clear_context_pin_async(
                    None, None,  # model match already checked above
                    _msg_model_cfg.get("base_url"), _msg_base_url,
                    _msg_model_cfg.get("provider"), _msg_runtime.get("provider"),
                ):
                    _msg_config_ctx = None
            except Exception:
                _msg_config_ctx = None
        if _msg_custom_providers and _msg_base_url:
            with suppress(Exception):
                from hermes_cli.config import get_custom_provider_context_length

                _msg_config_ctx = get_custom_provider_context_length(
                    model=_msg_model, base_url=_msg_base_url, custom_providers=_msg_custom_providers,
                ) or _msg_config_ctx
        return await get_model_context_length_async(
            _msg_model, base_url=_msg_base_url, api_key=_msg_runtime.get("api_key") or "",
            config_context_length=_msg_config_ctx, provider=_msg_runtime.get("provider") or "",
            custom_providers=_msg_custom_providers,
        )

    async def _expand_inbound_context_references(
        self, source: SessionSource, session_key: str, message_text: str
    ) -> Optional[str]:
        """Expand ``@`` context references; returns None when the injection was refused (user notified)."""
        try:
            from agent.context_references import preprocess_context_references_async

            try:
                from tools.terminal_scope import terminal_env as _ts_env
            except ImportError:
                _ts_env = os.environ.get
            _msg_cwd = _ts_env("TERMINAL_CWD", os.path.expanduser("~"))
            _msg_ctx_len = await self._inbound_model_context_length(source, session_key)
            _ctx_result = await preprocess_context_references_async(
                message_text, cwd=_msg_cwd, context_length=_msg_ctx_len, allowed_root=_msg_cwd
            )
            if _ctx_result.blocked:
                _adapter = self._adapter_for_source(source)
                if _adapter:
                    await _adapter.send(
                        source.chat_id,
                        "\n".join(_ctx_result.warnings) or "Context injection refused.",
                    )
                return None
            if _ctx_result.expanded:
                message_text = _ctx_result.message
        except Exception as exc:
            logger.warning("@ context reference expansion failed: %s", exc)
            logger.debug("@ context reference expansion failure detail", exc_info=True)
        return message_text

    async def _prepare_inbound_message_text(
        self, *, event: MessageEvent, source: SessionSource, history: List[Dict[str, Any]],
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Prepare inbound event text for the agent. Shared by the normal inbound and queued
        follow-up paths so attribution, image enrichment, STT, document notes, reply context and
        @ references behave the same. Side effect: buffers per-session native image paths when the
        model supports native vision; the caller consumes that buffer at ``run_conversation``."""
        _pending_stt_prepared = hasattr(event, "_gateway_pending_stt_text")
        message_text = (event._gateway_pending_stt_text if _pending_stt_prepared else event.text) or ""
        # Prefer the caller's resolved session key so this write key matches the consume key at the
        # run_conversation site; derive it here only for tests and legacy standalone callers.
        session_key = session_key or self._session_key_for_source(source)
        # Reset only this session's per-call buffer; other sessions may be concurrently preparing.
        self._consume_pending_native_image_paths(session_key)

        message_text = self._prefix_inbound_sender_context(event, source, message_text)
        image_paths, audio_paths, audio_file_paths, video_paths = self._classify_inbound_media(event, _pending_stt_prepared)
        if image_paths:
            message_text = await self._enrich_inbound_images(source, session_key, message_text, image_paths)
        if audio_paths:
            message_text = await self._enrich_inbound_voice(event, source, message_text, audio_paths)
        message_text = self._prepend_inbound_media_file_notes(message_text, audio_file_paths, video_paths)
        message_text = self._prepend_inbound_document_notes(event, message_text)
        if "@" in message_text:
            message_text = await self._expand_inbound_context_references(source, session_key, message_text)
            if message_text is None:
                return None
        # After expansion: the quoted reply is someone else's text and stays literal — an
        # ``@file:`` inside it must never read a local file on the replier's behalf.
        return self._prepend_inbound_reply_context(event, source, message_text)

    async def _prepare_profile_scoped_inbound_message_text(
        self, *, event: MessageEvent, source: SessionSource, history: List[Dict[str, Any]],
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Run inbound preprocessing under the routed profile when multiplexed."""
        from gateway.run import _async_profile_runtime_scope
        kwargs = dict(event=event, source=source, history=history, session_key=session_key)
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            async with _async_profile_runtime_scope(self._resolve_profile_home_for_source(source)):
                return await self._prepare_inbound_message_text(**kwargs)
        return await self._prepare_inbound_message_text(**kwargs)

    async def _prepare_clarify_reply_text(self, event) -> str:
        """Return raw text or successful voice transcripts for a clarify reply."""
        if not self._pending_event_audio_paths(event):
            return (event.text or "").strip()
        _, successful_transcripts = await self._transcribe_pending_audio_event_once(event, "")
        return "\n\n".join(t.strip() for t in successful_transcripts if t.strip())

    def _consume_pending_native_image_paths(self, session_key: str) -> List[str]:
        state = self._peek_session_state(session_key)
        paths = list(state.persistent.native_image_paths or []) if state is not None else []
        if paths:
            state.persistent.native_image_paths = []
        return paths

    async def _mark_durable_active_turn(self, event: "MessageEvent", session_key: str) -> bool:
        """Persist the exact resolved routing key for this running turn."""
        try:
            token = await self.async_session_store.mark_turn_active(session_key)
        except Exception as exc:
            logger.warning("Could not persist active-turn marker for %s: %s", session_key, exc)
            return False
        if not token:
            return False
        # Private event attributes are process-local ownership state: keep the token out of public
        # metadata, transcripts, and platform payloads.
        event._gateway_active_turn_session_key = session_key
        event._gateway_active_turn_token = token
        return True

    async def _clear_durable_active_turn(self, event: "MessageEvent") -> bool:
        """Best-effort CAS clear of the marker owned by *event* (3 attempts; never blocks agent/lease
        release — a stale marker is bounded by the agent timeout and clean-start discard)."""
        session_key = getattr(event, "_gateway_active_turn_session_key", None)
        token = getattr(event, "_gateway_active_turn_token", None)
        try:
            if not session_key or not token:
                return False
            last_error: Optional[Exception] = None
            for attempt in range(1, 4):
                try:
                    return bool(await self.async_session_store.clear_turn_active(session_key, token))
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        logger.debug(
                            "Retrying active-turn marker cleanup for %s (%d/3): %s",
                            session_key, attempt, exc,
                        )
            logger.warning(
                "Could not clear active-turn marker for %s after 3 attempts: %s", session_key, last_error,
            )
            return False
        finally:
            for attr in ("_gateway_active_turn_session_key", "_gateway_active_turn_token"):
                with suppress(AttributeError):
                    delattr(event, attr)

    def _install_plugin_message_injector(self) -> None:
        """Publish this live gateway's plugin message scheduler."""
        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().set_gateway_message_injector(
            self, self._schedule_plugin_message_injection
        )

    def _clear_plugin_message_injector(self) -> None:
        """Remove this runner's scheduler without clobbering a newer owner."""
        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().clear_gateway_message_injector(self)

    def _schedule_plugin_message_injection(
        self, *, session_key: str, content: str, plugin_id: str
    ) -> bool:
        """Schedule a plugin-triggered turn on the live gateway loop (thread-safe)."""
        from gateway.run import safe_schedule_threadsafe
        loop = getattr(self, "_gateway_loop", None)
        if not getattr(self, "_running", False) or loop is None or loop.is_closed():
            return False

        coro = self._dispatch_plugin_message_injection(
            session_key=session_key, content=content, plugin_id=plugin_id,
        )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is loop:
            try:
                future = loop.create_task(coro)
            except Exception:
                coro.close()
                logger.warning("Plugin message injection scheduling failed", exc_info=True)
                return False
            self._background_tasks.add(future)
            future.add_done_callback(self._background_tasks.discard)
        else:
            future = safe_schedule_threadsafe(
                coro, loop, logger=logger, log_message="Plugin message injection scheduling failed",
                log_level=logging.WARNING,
            )
            if future is None:
                return False

        def _log_result(completed) -> None:
            try:
                if completed.result():
                    return
                what, exc = "was not routed", None
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                return
            except Exception as err:
                what, exc = "failed", err
            logger.warning(
                "Plugin message injection %s: plugin=%s session=%s", what, plugin_id, session_key, exc_info=exc,
            )

        future.add_done_callback(_log_result)
        return True

    async def _dispatch_plugin_message_injection(
        self, *, session_key: str, content: str, plugin_id: str
    ) -> bool:
        """Route a plugin-triggered turn through the session's live adapter."""
        def _accepting() -> bool:
            return getattr(self, "_running", False) and not getattr(self, "_draining", False)

        if not _accepting():
            return False
        entry = await self.async_session_store.lookup_by_session_key(session_key)
        if entry is None or entry.origin is None or not _accepting():
            return False

        source = dataclasses.replace(entry.origin)
        try:
            authorized = self._is_user_authorized(source, allow_adapter_delegation=False)
        except Exception:
            logger.warning(
                "Plugin message injection authorization check failed: plugin=%s session=%s",
                plugin_id, session_key, exc_info=True,
            )
            return False
        if not authorized:
            logger.warning(
                "Plugin message injection denied by current gateway authorization: "
                "plugin=%s session=%s", plugin_id, session_key,
            )
            return False

        adapter = self._adapter_for_source(source)
        if adapter is None:
            return False

        await adapter.handle_message(MessageEvent(
            text=content, message_type=MessageType.TEXT, source=source, internal=True,
            allow_gateway_control=False,
            metadata={
                "hermes_plugin_id": plugin_id, "hermes_plugin_injection": True,
                "gateway_session_key": session_key, "gateway_session_id": entry.session_id,
                "gateway_session_strict": True,
            },
        ))
        logger.info(
            "Plugin message injection dispatched: plugin=%s session=%s session_id=%s",
            plugin_id, session_key, entry.session_id,
        )
        return True

    def _decide_image_input_mode(
        self, *, source: Optional[SessionSource] = None, session_key: Optional[str] = None,
        user_config: Optional[dict] = None, provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Resolve image-input routing (``"native"`` / ``"text"``) for the effective model this turn
        (see agent/image_routing.py). Sessions can carry /model overrides and this runs before AIAgent
        sets the auxiliary_client runtime globals, so resolve the per-session runtime bundle the
        upcoming turn will use, not just the persisted default."""
        try:
            from agent.image_routing import decide_image_input_mode
            from agent.auxiliary_client import _read_main_model, _read_main_provider
            from hermes_cli.config import load_config

            cfg = user_config if isinstance(user_config, dict) else load_config()
            resolved_provider = (provider or "").strip()
            resolved_model = (model or "").strip()
            resolved_requested_provider = ""

            if (not resolved_provider or not resolved_model) and (source is not None or session_key):
                try:
                    turn_model, runtime_kwargs = self._resolve_session_agent_runtime(
                        source=source, session_key=session_key, user_config=cfg,
                    )
                    rk = runtime_kwargs if isinstance(runtime_kwargs, dict) else {}
                    if not resolved_model and isinstance(turn_model, str):
                        resolved_model = turn_model.strip()
                    if not resolved_provider and isinstance(rk.get("provider"), str):
                        resolved_provider = rk["provider"].strip()
                    if isinstance(rk.get("requested_provider"), str):
                        resolved_requested_provider = rk["requested_provider"].strip()
                except Exception as exc:
                    logger.debug(
                        "image_routing: session runtime resolution failed, falling back to config — %s",
                        exc,
                    )

            return decide_image_input_mode(
                resolved_provider or _read_main_provider(), resolved_model or _read_main_model(),
                cfg, requested_provider=resolved_requested_provider,
            )
        except Exception as exc:
            logger.debug("image_routing: decision failed, falling back to text — %s", exc)
            return "text"

    _EMPTY_TEXT_PLACEHOLDER = "(The user sent a message with no text content)"

    @staticmethod
    def _untranscribed_audio_note(path: str) -> str:
        """One minimal neutral marker for every STT failure. Never mention "no STT provider" or setup
        steps — persisted in history they make the model keep volunteering STT-setup advice."""
        from tools.credential_files import to_agent_visible_cache_path
        agent_path = to_agent_visible_cache_path(os.path.abspath(path))
        return f"[voice message could not be transcribed automatically; the audio is available at: {agent_path}]"

    async def _transcribe_one_clip(self, path: str, transcribe_audio, transcribe_audio_local_fallback) -> Tuple[Optional[str], str]:
        """``(transcript_or_None, note)`` for one clip via configured STT with local fallback."""
        result = await asyncio.to_thread(transcribe_audio, path, None, "gateway")
        if not result.get("success"):
            fallback = await asyncio.to_thread(transcribe_audio_local_fallback, path)
            if fallback.get("success"):
                logger.info("Configured STT failed for %s; recovered with local STT", path)
                result = fallback
        if not result["success"]:
            logger.info("Voice transcription failed for %s: %s", path, result.get("error", "unknown error"))
            return None, self._untranscribed_audio_note(path)
        transcript = result["transcript"]
        # STT may return success=True with an empty/whitespace transcript (silence, cut-off);
        # empty quotes make the agent reply to nothing and can loop, so emit a sentinel note.
        # See #41603.
        if not (transcript or "").strip():
            return None, (
                "[The user sent a voice message but it came through "
                "empty or inaudible — speech-to-text returned no "
                "words. Do not guess at the content; ask the user "
                "to resend or type it out.]"
            )
        # Plain quoted line: a "The user sent a voice message..." wrapper read as a meta-instruction
        # and made the LLM comment on voice mode instead.
        return transcript, f'"{transcript}"'

    def _pending_event_audio_paths(self, event) -> List[str]:
        """Return STT-eligible paths from a pending voice message."""
        from gateway.run import _event_media_is_stt_input
        return [
            path for i, path in enumerate(getattr(event, "media_urls", None) or [])
            if _event_media_is_stt_input(event, i)
        ]

    async def _transcribe_pending_audio_event_once(
        self, event, user_text: Optional[str] = None
    ) -> tuple[str | None, List[str]]:
        """Transcribe a pending audio event once and cache the result on the event: the interrupt
        monitor and the pending-drain path both need it — one STT call and one echo per message."""
        if hasattr(event, "_gateway_pending_stt_text"):
            return event._gateway_pending_stt_text, list(getattr(event, "_gateway_pending_stt_transcripts", []) or [])
        audio_paths = self._pending_event_audio_paths(event)
        if not audio_paths:
            return user_text if user_text is not None else (getattr(event, "text", None) or None), []
        text = user_text if user_text is not None else (getattr(event, "text", "") or "")
        enriched_text, successful_transcripts = await self._enrich_message_with_transcription(text, audio_paths)
        event._gateway_pending_stt_text = enriched_text
        event._gateway_pending_stt_transcripts = list(successful_transcripts)
        return enriched_text, successful_transcripts

    async def _echo_pending_stt_transcripts_once(
        self, event, adapter, source, transcripts: List[str], *, metadata=None,
        log_context: str = "Transcript",
    ) -> None:
        """Echo pending-event STT transcripts to the chat at most once. Tracked as a COUNT (not a
        set — identical transcripts are distinct deliveries): ``merge_pending_message_event`` can
        append a second voice note and invalidate the cache; the re-run returns earlier transcripts
        as a prefix, so only the unsent tail is echoed."""
        if not transcripts or not self._should_echo_stt_transcripts() or adapter is None:
            return
        already_echoed = int(getattr(event, "_gateway_pending_stt_echoed", 0) or 0)
        event._gateway_pending_stt_echoed = max(already_echoed, len(transcripts))
        await self._echo_stt_transcripts(
            adapter, source, transcripts[already_echoed:], metadata=metadata, log_context=log_context,
        )

    async def _transcribe_and_echo_pending_voice(
        self, event, adapter, source, text: str, *, log_context: str, metadata=_UNSET
    ) -> tuple[str, List[str]]:
        """Transcribe a pending voice event and echo transcripts once → ``(enriched_text,
        transcripts)`` for ``agent.interrupt()`` or the pending-drain flow; ``(text, [])`` when there
        is no STT-eligible media (caller owns the ``_build_media_placeholder`` fallback)."""
        if not self._pending_event_audio_paths(event):
            return text, []
        try:
            enriched_text, transcripts = await self._transcribe_pending_audio_event_once(event, text)
            if metadata is _UNSET:
                metadata = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
            await self._echo_pending_stt_transcripts_once(
                event, adapter, source, transcripts, metadata=metadata, log_context=log_context
            )
            return enriched_text or text, transcripts
        except Exception as trans_exc:
            logger.warning("%s transcription failed: %s", log_context, trans_exc)
            return text, []
