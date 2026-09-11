"""run_inbound_prepend — the _prepend* methods split out of GatewayInboundMixin (mechanical extraction)."""

from __future__ import annotations
import os
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

class GatewayInboundPrependMixin:
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
    def _prepend_inbound_reply_context(event: MessageEvent, source: SessionSource, message_text: str) -> str:
        """Prepend the Discord triggering-message id and the reply-to pointer."""
        # Discord: the triggering message id goes on the per-turn user message, never the cached
        # system prompt — it changes every turn and would bust the agent-cache signature.
        if (
            source is not None
            and getattr(source, "platform", None) == Platform.DISCORD
            and getattr(event, "message_id", None)
        ):
            from gateway.session import _discord_tools_loaded as _disc_tools_loaded
            if _disc_tools_loaded():
                message_text = (
                    f"[Triggering message id: `{event.message_id}` — use as "
                    f"`message_id` for reply/react/pin via the discord tools.]\n\n"
                    f"{message_text}"
                )

        if getattr(event, "reply_to_text", None) and event.reply_to_message_id:
            # Always inject the reply-to pointer even when the quoted text is already in history:
            # it's disambiguation (*which* prior message), not deduplication.
            # Adapters resolve the original message (or the user's native partial quote).
            # A preview here silently loses later list items and code; keep that context intact.
            reply_text = event.reply_to_text
            _who = " your previous message" if getattr(event, "reply_to_is_own_message", False) else ""
            message_text = f'[Replying to{_who}: "{reply_text}"]\n\n{message_text}'
        return message_text

    @classmethod
    def _prepend_media_prefix(cls, prefix: str, user_text: str) -> str:
        """``prefix`` + the user's text; the Discord empty-content placeholder is dropped as redundant."""
        if user_text and user_text.strip() != cls._EMPTY_TEXT_PLACEHOLDER:
            return f"{prefix}\n\n{user_text}"
        return prefix
