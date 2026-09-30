"""Inbound context, reply and media preparation for GatewayRunner."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_inbound_media import rehome_inbound_media
from gateway.run_inbound_turn_context import prepend_turn_context_note, turn_context_update
from gateway.session import SessionSource, is_shared_multi_user_session, neutralize_untrusted_inline_text

if TYPE_CHECKING:
    from gateway.run import GatewayRunner
    from gateway.run_inbound import GatewayInboundMixin

logger = logging.getLogger("gateway.run")


def discord_triggering_note(message_id: Any) -> str:
    """Model-facing routing note for a Discord turn (rides the API-bound user message only)."""
    return (
        f"[Triggering message id: `{message_id}` — use as `message_id` for reply/react/pin "
        f"via the discord tools.]"
    )


def strip_discord_triggering_note(event: Any, message_text: Any) -> Any:
    """Authored text for the durable user row: peel off exactly the note
    ``_prepend_inbound_reply_context`` added for THIS event, if present. The note is a
    model instruction, not something the user wrote — persisted as ``content`` it renders
    verbatim in every transcript surface and pollutes FTS/memory (#71304, #114719). It
    keeps riding ``message_text`` (and the replay-only ``api_content`` sidecar)."""
    message_id = getattr(event, "message_id", None)
    if not message_id or not isinstance(message_text, str):
        return message_text
    prefix = f"{discord_triggering_note(message_id)}\n\n"
    return message_text[len(prefix):] if message_text.startswith(prefix) else message_text


class GatewayInboundContextMixin:
    def _prefix_inbound_sender_context(self: GatewayRunner, event: MessageEvent, source: SessionSource, message_text: str) -> str:
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
        return event.text_with_channel_context(message_text)


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


    async def _enrich_inbound_images(
        self: GatewayRunner, source: SessionSource, session_key: str, message_text: str, image_paths: list[str]
    ) -> str:
        """Route images natively (attach pixels at run_conversation) or pre-analyze them into text."""
        # See agent/image_routing.py. Offloaded to a thread: the decision does blocking network I/O
        # (models.dev fetch on cache miss, Ollama /api/show probe) that would stall the event loop.
        _img_mode = await asyncio.to_thread(
            self._decide_image_input_mode, source=source, session_key=session_key,
        )
        if _img_mode == "native":
            self._session_state(session_key).persistent.native_image_paths = list(image_paths)
            logger.info(
                "Image routing: native (model supports vision). %d image(s) will be attached inline.",
                len(image_paths),
            )
            return message_text
        logger.info(
            "Image routing: text (mode=%s). Pre-analyzing %d image(s) via vision_analyze.",
            _img_mode, len(image_paths),
        )
        # Vision enrichment runs before AIAgent.run_conversation(), so bind this session's resolved
        # runtime explicitly rather than consulting process-global compatibility mirrors.
        vision_runtime = None
        try:
            turn_model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source, session_key=session_key,
            )
            vision_runtime = {**(runtime_kwargs or {}), "model": turn_model}
        except Exception:
            logger.debug("vision enrichment: session runtime resolution failed", exc_info=True)

        from agent.auxiliary_client import scoped_runtime_main

        with scoped_runtime_main(vision_runtime):
            return await self._enrich_message_with_vision(message_text, image_paths)


    async def _echo_stt_transcripts(
        self, adapter, source: SessionSource, transcripts: List[str], *, metadata=None, log_context: str = "Transcript"
    ) -> None:
        """Send each transcript back as ``🎙️ "…"`` (best-effort; failures are logged, never raised)."""
        for tx in transcripts:
            try:
                await adapter.send(source.chat_id, t("gateway.voice.transcript_echo_short", text=tx), metadata=metadata)
            except Exception as echo_exc:
                logger.debug("%s echo failed (non-fatal): %s", log_context, echo_exc)


    async def _enrich_inbound_voice(
        self: GatewayRunner, event: MessageEvent, source: SessionSource, message_text: str, audio_paths: list[str]
    ) -> str:
        message_text, _successful_transcripts = await self._enrich_message_with_transcription(
            message_text, audio_paths,
        )
        # Echo each successful transcript back immediately when configured so users can verify STT
        # quality in real time. On transcription failure do NOT send a hardcoded notice: that
        # bypassed the LLM and produced two replies; enrichment leaves one neutral marker instead.
        if _successful_transcripts and self._should_echo_stt_transcripts():
            _echo_adapter = self._delivery_adapter_for(source)
            if _echo_adapter:
                _echo_meta = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
                await self._echo_stt_transcripts(_echo_adapter, source, _successful_transcripts, metadata=_echo_meta)
        return message_text


    @staticmethod
    def _inbound_attachment_display_name(path: str) -> Tuple[str, str]:
        """``(display_name, agent_visible_path)``: cache filename is ``<id>_<id>_<original>``; the
        path is translated to the in-container mount under a Docker backend."""
        from tools.credential_files import to_agent_visible_cache_path
        basename = os.path.basename(path)
        parts = basename.split("_", 2)
        return re.sub(r'[^\w.\- ]', '_', parts[2] if len(parts) >= 3 else basename), to_agent_visible_cache_path(path)


    @classmethod
    def _prepend_inbound_media_file_notes(cls, message_text: str, audio_file_paths: list[str], video_paths: list[str]) -> str:
        """Prepend a path-pointing note per audio-file / video attachment (content is not inlined)."""
        for kind, noun, verb, tool, paths in (
            ("an audio file attachment", "audio", "transcribe or process", "a transcription or media tool", audio_file_paths),
            ("a video attachment", "video", "inspect or process", "a video analysis or media tool", video_paths),
        ):
            for _path in paths:
                _display, _agent_path = cls._inbound_attachment_display_name(_path)
                message_text = (
                    f"[The user sent {kind}: '{_display}'. "
                    f"It is saved at: {_agent_path}. "
                    f"Its content is not inlined here. If the user's request involves "
                    f"what the {noun} contains, {verb} it yourself — for "
                    f"example by passing the path to {tool} — "
                    f"instead of asking the user to describe it. Only ask what to do "
                    f"with it if their intent is genuinely unclear.]"
                    f"\n\n{message_text}"
                )
        return message_text


    @classmethod
    def _prepend_inbound_document_notes(cls, event: MessageEvent, message_text: str) -> str:
        """Prepend a context note per non-media attachment (anything not routed as image/audio/video)."""
        from gateway.run import (
            _build_document_context_note, _event_media_is_audio, _event_media_is_image,
            _event_media_is_video,
        )
        if not event.media_urls:
            return message_text
        import mimetypes as _mimetypes

        _TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
        inline_flags = getattr(event, "media_text_inlined", None) or []
        for i, path in enumerate(event.media_urls):
            # A document mixed into a PHOTO/VOICE message (message-level type != DOCUMENT) still
            # reaches the agent; only genuine non-media files get a note.
            if any(f(event, i) for f in (_event_media_is_image, _event_media_is_audio, _event_media_is_video)):
                continue
            mtype = event.media_types[i] if i < len(event.media_types) else ""
            if mtype in {"", "application/octet-stream"}:
                _is_text = os.path.splitext(path)[1].lower() in _TEXT_EXTENSIONS
                mtype = "text/plain" if _is_text else (_mimetypes.guess_type(path)[0] or "application/octet-stream")
            # Every accepted file gets a note — a non-text/non-application MIME (font/*, model/*)
            # must still tell the agent the file exists.
            display_name, agent_path = cls._inbound_attachment_display_name(path)
            inline_flag = inline_flags[i] if i < len(inline_flags) else None
            context_note = _build_document_context_note(
                display_name, agent_path, mtype, content_inlined=inline_flag is not False,
            )
            message_text = f"{context_note}\n\n{message_text}"
        return message_text


    @staticmethod
    def _prepend_inbound_reply_context(
        event: MessageEvent, source: SessionSource, message_text: str, *, redact_pii: bool = False,
    ) -> str:
        """Prepend the reply-to pointer, then the Discord triggering-message note (outermost)."""
        if getattr(event, "reply_to_text", None) and event.reply_to_message_id:
            # Always inject the reply-to pointer even when the quoted text is already in history:
            # it's disambiguation (*which* prior message), not deduplication.
            # Adapters resolve the original message (or the user's native partial quote).
            # A preview here silently loses later list items and code; keep that context intact.
            reply_text = event.reply_to_text
            if getattr(event, "reply_to_is_own_message", False):
                pointer = "Replying to your previous message: "
            elif event.reply_to_author_authorized is None:
                # Some adapters fill reply_to_author_id with a phone number. Identify the author
                # only when the adapter has checked their authorisation, and hash a bare ID
                # under redact_pii.
                pointer = "Replying to: "
            else:
                from gateway.session import _hash_sender_id, _should_redact_pii, neutralize_untrusted_inline_text

                trust = "[unverified] " if event.reply_to_author_authorized is False else ""
                author = event.reply_to_author_name
                if not author and event.reply_to_author_id:
                    author = event.reply_to_author_id
                    if _should_redact_pii(source.platform, redact_pii):
                        author = _hash_sender_id(author)
                pointer = (
                    f"Replying to {trust}{neutralize_untrusted_inline_text(author)}: " if author
                    else f"Replying to: {trust}"
                )
            message_text = f'[{pointer}"{reply_text}"]\n\n{message_text}'

        # Discord: the triggering message id goes on the per-turn user message, never the cached
        # system prompt — it changes every turn and would bust the agent-cache signature. It is
        # the OUTERMOST prefix so strip_discord_triggering_note can peel exactly it off the
        # persisted transcript row without touching the reply pointer.
        if (
            source is not None
            and getattr(source, "platform", None) == Platform.DISCORD
            and getattr(event, "message_id", None)
        ):
            from gateway.session import _discord_tools_loaded as _disc_tools_loaded
            if _disc_tools_loaded():
                message_text = f"{discord_triggering_note(event.message_id)}\n\n{message_text}"
        return message_text


    async def _inbound_model_context_length(self: GatewayRunner, source: SessionSource, session_key: str) -> int:
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
        self: GatewayRunner, source: SessionSource, session_key: str, message_text: str
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
                _adapter = self._delivery_adapter_for(source)
                if _adapter:
                    await _adapter.send(
                        source.chat_id,
                        "\n".join(_ctx_result.warnings) or t("gateway.notify.context_injection_refused"),
                    )
                return None
            if _ctx_result.expanded:
                message_text = _ctx_result.message
        except Exception as exc:
            logger.warning("@ context reference expansion failed: %s", exc)
            logger.debug("@ context reference expansion failure detail", exc_info=True)
        return message_text


    async def _prepare_inbound_message_text(
        self: GatewayRunner, *, event: MessageEvent, source: SessionSource, history: List[Dict[str, Any]],
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Prepare inbound event text for the agent. Shared by the normal inbound and queued
        follow-up paths so attribution, image enrichment, STT, document notes, reply context and
        @ references behave the same. Side effect: buffers per-session native image paths when the
        model supports native vision; the caller consumes that buffer at ``run_conversation``."""
        rehome_inbound_media(event)  # before any consumer (vision, STT, document notes) reads media_urls
        _pending_stt_prepared = hasattr(event, "_gateway_pending_stt_text")
        message_text: str = getattr(event, "_gateway_pending_stt_text", event.text) or ""
        # Prefer the caller's resolved session key so this write key matches the consume key at the
        # run_conversation site; derive it here only for tests and legacy standalone callers.
        session_key = session_key or self._session_key_for_source(source)
        # Reset only this session's per-call buffer; other sessions may be concurrently preparing.
        self._consume_pending_native_image_paths(session_key)

        if "@" in message_text:
            expanded_message_text = await self._expand_inbound_context_references(source, session_key, message_text)
            if expanded_message_text is None:
                return None
            message_text = expanded_message_text

        adapter = self._intake_adapter_for(source)
        context_snapshot = None
        fetch_inbound_context = getattr(type(adapter), "fetch_inbound_context", None)
        if callable(fetch_inbound_context):
            context_snapshot = await fetch_inbound_context(adapter, event)
            context_snapshot.use_turn_context(await turn_context_update(
                self, event=event, source=source, session_key=session_key, history=history,
            ))
        message_text = self._prefix_inbound_sender_context(event, source, message_text)
        media_event = context_snapshot.media_event(event) if context_snapshot is not None else event
        image_paths, audio_paths, audio_file_paths, video_paths = self._classify_inbound_media(media_event, _pending_stt_prepared)
        authored_images = ()
        if image_paths and context_snapshot is not None:
            from gateway.inbound_context import ImageEnrichment

            authored_images = await ImageEnrichment.enrich_each(self, source, session_key, image_paths)
        elif image_paths:
            message_text = await self._enrich_inbound_images(source, session_key, message_text, image_paths)
        if audio_paths:
            message_text = await self._enrich_inbound_voice(event, source, message_text, audio_paths)
        message_text = self._prepend_inbound_media_file_notes(message_text, audio_file_paths, video_paths)
        message_text = self._prepend_inbound_document_notes(event, message_text)
        redact_pii = False
        if event.reply_to_text or (context_snapshot is not None and event.reply_to_message_id):
            from gateway.run import _load_gateway_config

            with suppress(Exception):
                redact_pii = bool((_load_gateway_config().get("privacy") or {}).get("redact_pii", False))
        if context_snapshot is None:
            message_text = self._prepend_inbound_reply_context(event, source, message_text, redact_pii=redact_pii)
            return await prepend_turn_context_note(
                self, event=event, source=source, session_key=session_key, history=history,
                message_text=message_text,
            )
        from gateway.inbound_context import ImageEnrichment, PreparedInboundMessage

        await context_snapshot.refresh()
        prepared = PreparedInboundMessage(
            context_snapshot, event, message_text, redact_pii=redact_pii, authored_images=authored_images,
        )
        quoted_images = context_snapshot.reply_image_paths()
        if quoted_images:
            prepared.quoted_images = await ImageEnrichment.enrich_each(self, source, session_key, quoted_images)
            await context_snapshot.refresh()
        event._prepared_inbound = prepared
        message_text = prepared.render(self)
        state = self._peek_session_state(session_key)
        if state is not None:
            state.persistent.native_image_paths = prepared.retained_image_paths(state.persistent.native_image_paths or [])
        return message_text


    async def _prepare_profile_scoped_inbound_message_text(
        self: GatewayRunner, *, event: MessageEvent, source: SessionSource, history: List[Dict[str, Any]],
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Run inbound preprocessing under the routed profile when multiplexed."""
        from gateway.run import _async_profile_runtime_scope
        kwargs = dict(event=event, source=source, history=history, session_key=session_key)
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            async with _async_profile_runtime_scope(self._resolve_profile_home_for_source(source)):
                return await self._prepare_inbound_message_text(**kwargs)
        return await self._prepare_inbound_message_text(**kwargs)


    async def _prepare_clarify_reply_text(self: GatewayInboundMixin, event) -> str:
        """Return raw text or successful voice transcripts for a clarify reply."""
        if not self._pending_event_audio_paths(event):
            return (event.text or "").strip()
        _, successful_transcripts = await self._transcribe_pending_audio_event_once(event, "")
        return "\n\n".join(t.strip() for t in successful_transcripts if t.strip())


    def _consume_pending_native_image_paths(self: GatewayRunner, session_key: str) -> List[str]:
        state = self._peek_session_state(session_key)
        if state is None:
            return []
        paths = list(state.persistent.native_image_paths or [])
        if paths:
            state.persistent.native_image_paths = []
        return paths
