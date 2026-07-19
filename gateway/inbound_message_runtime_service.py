"""Inbound message text preparation for gateway turns.

Promoted from ``GatewayRunner._prepare_inbound_message_text`` so the normal
inbound path and queued follow-ups share one preprocessing pipeline
(sender prefix, vision/STT, document notes, reply context, @ refs).

``GatewayRunner`` methods remain thin delegates so existing tests keep calling
``runner._prepare_inbound_message_text``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

from gateway.config import Platform
from gateway.platforms.base import MessageType
from gateway.session import (
    is_shared_multi_user_session,
    neutralize_untrusted_inline_text,
)

logger = logging.getLogger("gateway.run")  # match gateway.run logger name used by tests?


def _event_media_is_image(event, index: int) -> bool:
    from gateway.run import _event_media_is_image as _fn
    return _fn(event, index)


def _event_media_is_audio(event, index: int) -> bool:
    from gateway.run import _event_media_is_audio as _fn
    return _fn(event, index)


def _event_media_is_stt_input(event, index: int) -> bool:
    from gateway.run import _event_media_is_stt_input as _fn
    return _fn(event, index)


def _event_media_is_video(event, index: int) -> bool:
    from gateway.run import _event_media_is_video as _fn
    return _fn(event, index)


def _build_document_context_note(display_name: str, agent_path: str, mtype: str) -> str:
    from gateway.run import _build_document_context_note as _fn
    return _fn(display_name, agent_path, mtype)


def _load_gateway_config() -> dict:
    from gateway.run import _load_gateway_config as _fn
    return _fn()


async def prepare_inbound_message_text(
    *,
    runner: Any,
    event: Any,
    source: Any,
    history: List[Dict[str, Any]] | None = None,
    session_key: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    """Prepare inbound event text for the agent (production semantics)."""
    if logger is None:
        logger = logging.getLogger(__name__)

    history = history or []
    _pending_stt_prepared = hasattr(event, "_gateway_pending_stt_text")
    message_text = (
        getattr(event, "_gateway_pending_stt_text", None)
        if _pending_stt_prepared
        else event.text
    ) or ""
    _group_sessions_per_user = getattr(runner.config, "group_sessions_per_user", True)
    _thread_sessions_per_user = getattr(runner.config, "thread_sessions_per_user", False)
    # Prefer the already resolved session key from the caller so this write
    # key matches the consume key at the run_conversation site. Fall back
    # to deriving it here for tests and legacy standalone callers.
    session_key = session_key or runner._session_key_for_source(source)
    # Reset only this session's per-call buffer; other sessions may be
    # concurrently preparing multimodal turns on the same runner.
    runner._consume_pending_native_image_paths(session_key)

    _is_shared_multi_user = is_shared_multi_user_session(
        source,
        group_sessions_per_user=_group_sessions_per_user,
        thread_sessions_per_user=_thread_sessions_per_user,
    )
    if _is_shared_multi_user and source.user_name:
        # source.user_name is the platform display name — attacker-
        # influenceable on any platform that lets participants set their
        # own name. Neutralize embedded newlines/control chars before
        # interpolating it into every message in the shared session, or
        # a hostile name can masquerade as a fake markdown section
        # (mirrors the same field's treatment in
        # build_session_context_prompt via _format_untrusted_prompt_value).
        _safe_user_name = neutralize_untrusted_inline_text(source.user_name)
        message_text = f"[{_safe_user_name}] {message_text}"

    # Prepend channel context from history backfill (if any).  This
    # happens after sender-prefix so the prefix only applies to the
    # trigger message, not the backfill block.
    if getattr(event, "channel_context", None):
        message_text = f"{event.channel_context}\n\n[New message]\n{message_text}"

    # Declare at outer scope so the audio-file-paths handling block below
    # remains safe when ``event.media_urls`` is empty (no inner block runs).
    audio_file_paths: list[str] = []
    video_paths: list[str] = []

    if event.media_urls:
        image_paths = []
        audio_paths = []
        for i, path in enumerate(event.media_urls):
            mtype = event.media_types[i] if i < len(event.media_types) else ""
            # Classify images per-attachment: trust this attachment's own
            # MIME, and only honour the message-level PHOTO type when the
            # per-attachment MIME is unknown. Otherwise a document (or any
            # non-image) sent alongside an image in the same message gets
            # mis-routed here as an image and the provider 400s.
            if _event_media_is_image(event, i):
                image_paths.append(path)
            # MessageType.AUDIO = audio file attachment (e.g. .mp3, .m4a) — never STT
            # MessageType.VOICE = voice message (Opus/OGG) — always STT
            if event.message_type == MessageType.AUDIO:
                audio_file_paths.append(path)
            elif not _pending_stt_prepared and _event_media_is_stt_input(event, i):
                audio_paths.append(path)
            if mtype.startswith("video/") or (not mtype and event.message_type == MessageType.VIDEO):
                video_paths.append(path)

        if image_paths:
            # Decide routing: native (attach pixels) vs text (vision_analyze
            # pre-run + prepend description).  See agent/image_routing.py.
            # Offload to a worker thread: the decision does blocking network
            # I/O — a models.dev fetch on cache miss, and the Ollama
            # ``/api/show`` capability probe for local servers — whose
            # request timeout would otherwise stall the whole gateway event
            # loop (every session) while a single image is routed.
            _img_mode = await asyncio.to_thread(
                runner._decide_image_input_mode,
                source=source,
                session_key=session_key,
            )
            if _img_mode == "native":
                # Defer attachment to the run_conversation call site.
                pending_native = getattr(runner, "_pending_native_image_paths_by_session", None)
                if pending_native is None:
                    pending_native = {}
                    runner._pending_native_image_paths_by_session = pending_native
                pending_native[session_key] = list(image_paths)
                logger.info(
                    "Image routing: native (model supports vision). %d image(s) will be attached inline.",
                    len(image_paths),
                )
            else:
                logger.info(
                    "Image routing: text (mode=%s). Pre-analyzing %d image(s) via vision_analyze.",
                    _img_mode, len(image_paths),
                )
                # Vision enrichment runs before AIAgent.run_conversation(),
                # so bind this session's resolved runtime explicitly rather
                # than consulting process-global compatibility mirrors.
                vision_runtime = None
                try:
                    turn_model, runtime_kwargs = runner._resolve_session_agent_runtime(
                        source=source,
                        session_key=session_key,
                    )
                    vision_runtime = dict(runtime_kwargs or {})
                    vision_runtime["model"] = turn_model
                except Exception:
                    logger.debug(
                        "vision enrichment: session runtime resolution failed",
                        exc_info=True,
                    )

                from agent.auxiliary_client import scoped_runtime_main

                with scoped_runtime_main(vision_runtime):
                    message_text = await runner._enrich_message_with_vision(
                        message_text,
                        image_paths,
                    )

        if audio_paths:
            message_text, _successful_transcripts = await runner._enrich_message_with_transcription(
                message_text,
                audio_paths,
            )
            # Echo each successful transcript back to the user immediately
            # when configured. Lets users verify STT quality in real-time,
            # while allowing quiet STT for users who only want the agent to
            # receive the transcription.
            if _successful_transcripts and runner._should_echo_stt_transcripts():
                _echo_adapter = runner._adapter_for_source(source)
                _echo_meta = runner._thread_metadata_for_source(source, runner._reply_anchor_for_event(event))
                if _echo_adapter:
                    for _tx in _successful_transcripts:
                        try:
                            await _echo_adapter.send(
                                source.chat_id,
                                f'🎙️ "{_tx}"',
                                metadata=_echo_meta,
                            )
                        except Exception as _echo_exc:
                            logger.debug(
                                "Transcript echo failed (non-fatal): %s", _echo_exc,
                            )
            # NOTE: Previously, when transcription failed (e.g. no STT
            # provider configured), the gateway also emitted a hardcoded
            # English notice via `_stt_adapter.send()`. That bypassed the
            # LLM and produced two replies — one pre-canned English clip
            # (which TTS then spoke aloud, in the wrong language) and one
            # correct, localized LLM reply from the enriched message text.
            # The enrichment step now leaves a single neutral marker in the
            # prompt, so the LLM produces one coherent reply in the user's
            # language. The hardcoded send has therefore been removed.

    if audio_file_paths:
        from tools.credential_files import to_agent_visible_cache_path as _to_agent_path
        for _apath in audio_file_paths:
            _basename = os.path.basename(_apath)
            _parts = _basename.split("_", 2)
            _display = _parts[2] if len(_parts) >= 3 else _basename
            _display = re.sub(r'[^\w.\- ]', '_', _display)
            _agent_path = _to_agent_path(_apath)
            _note = (
                f"[The user sent an audio file attachment: '{_display}'. "
                f"It is saved at: {_agent_path}. "
                f"Its content is not inlined here. If the user's request involves "
                f"what the audio contains, transcribe or process it yourself — for "
                f"example by passing the path to a transcription or media tool — "
                f"instead of asking the user to describe it. Only ask what to do "
                f"with it if their intent is genuinely unclear.]"
            )
            message_text = f"{_note}\n\n{message_text}"

    if video_paths:
        from tools.credential_files import to_agent_visible_cache_path as _to_agent_path
        for _vpath in video_paths:
            _basename = os.path.basename(_vpath)
            _parts = _basename.split("_", 2)
            _display = _parts[2] if len(_parts) >= 3 else _basename
            _display = re.sub(r'[^\w.\- ]', '_', _display)
            _agent_path = _to_agent_path(_vpath)
            _note = (
                f"[The user sent a video attachment: '{_display}'. "
                f"It is saved at: {_agent_path}. "
                f"Its content is not inlined here. If the user's request involves "
                f"what the video contains, inspect or process it yourself — for "
                f"example by passing the path to a video analysis or media tool — "
                f"instead of asking the user to describe it. Only ask what to do "
                f"with it if their intent is genuinely unclear.]"
            )
            message_text = f"{_note}\n\n{message_text}"

    if event.media_urls:
        import mimetypes as _mimetypes
        from tools.credential_files import to_agent_visible_cache_path

        _TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
        for i, path in enumerate(event.media_urls):
            # Per-attachment document handling. Skip anything already routed
            # as image / audio / video by the buckets above — only genuine
            # non-media files get a path-pointing context note. This makes a
            # document mixed into a PHOTO/VOICE message (whole-message type
            # != DOCUMENT) still reach the agent as a readable cached file,
            # instead of being silently dropped because the message-level
            # type wasn't DOCUMENT.
            if (
                _event_media_is_image(event, i)
                or _event_media_is_audio(event, i)
                or _event_media_is_video(event, i)
            ):
                continue
            mtype = event.media_types[i] if i < len(event.media_types) else ""
            if mtype in {"", "application/octet-stream"}:
                _ext = os.path.splitext(path)[1].lower()
                if _ext in _TEXT_EXTENSIONS:
                    mtype = "text/plain"
                else:
                    guessed, _ = _mimetypes.guess_type(path)
                    if guessed:
                        mtype = guessed
                    else:
                        mtype = "application/octet-stream"
            # Any accepted file gets a path-pointing context note — we accept
            # all file types now, so a non-text/non-application MIME (font/*,
            # model/*, etc.) must still tell the agent the file exists.

            basename = os.path.basename(path)
            parts = basename.split("_", 2)
            display_name = parts[2] if len(parts) >= 3 else basename
            display_name = re.sub(r'[^\w.\- ]', '_', display_name)

            # Translate host cache path to in-container path if running under Docker backend.
            # This ensures the agent receives a path it can open inside its sandbox, as the
            # cache directories are auto-mounted at /root/.hermes/cache/* by get_cache_directory_mounts().
            agent_path = to_agent_visible_cache_path(path)

            context_note = _build_document_context_note(display_name, agent_path, mtype)
            message_text = f"{context_note}\n\n{message_text}"

    # Discord: surface the triggering message id per-turn on the user
    # message rather than in the cached system prompt. message_id changes
    # every turn, so baking it into build_session_context_prompt() would
    # bust the agent-cache signature and rebuild the AIAgent every message
    # (destroying prompt caching). The static IDs block points the agent
    # here; the volatile id rides the per-turn user content.
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
        # Always inject the reply-to pointer — even when the quoted text
        # already appears in history. The prefix isn't deduplication, it's
        # disambiguation: it tells the agent *which* prior message the user
        # is referencing. History can contain the same or similar text
        # multiple times, and without an explicit pointer the agent has to
        # guess (or answer for both subjects). Token overhead is minimal.
        reply_snippet = event.reply_to_text[:500]
        if getattr(event, "reply_to_is_own_message", False):
            message_text = (
                f'[Replying to your previous message: "{reply_snippet}"]\n\n'
                f"{message_text}"
            )
        else:
            message_text = f'[Replying to: "{reply_snippet}"]\n\n{message_text}'

    if "@" in message_text:
        try:
            from agent.context_references import preprocess_context_references_async
            from agent.model_metadata import get_model_context_length_async

            _msg_cwd = os.environ.get("TERMINAL_CWD", os.path.expanduser("~"))
            _msg_config_ctx = None
            _msg_cfg = None
            _msg_model_cfg = {}
            _msg_custom_providers = []
            try:
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
            except Exception:
                pass
            # Resolve the session's actual model/provider/base_url the
            # same way the hygiene compression block does (~11080).
            # GatewayRunner has no runner._model/runner._base_url attrs
            # (that was copy-pasted from HermesCLI, which does carry
            # runner.model/runner.base_url), so using them here always raised
            # AttributeError, silently caught below, meaning this feature
            # never ran.
            _msg_model, _msg_runtime = runner._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
                user_config=_msg_cfg,
            )
            _msg_base_url = _msg_runtime.get("base_url") or ""
            # A global model.context_length belongs to the configured
            # model, not a session /model or channel override. Prefer a
            # matching per-custom-provider model limit when available.
            _msg_configured_model = (
                _msg_model_cfg.get("default") or _msg_model_cfg.get("model")
                if isinstance(_msg_model_cfg, dict)
                else _msg_model_cfg
            )
            if _msg_model != _msg_configured_model:
                _msg_config_ctx = None
            if _msg_custom_providers and _msg_base_url:
                try:
                    from hermes_cli.config import get_custom_provider_context_length

                    _msg_custom_ctx = get_custom_provider_context_length(
                        model=_msg_model,
                        base_url=_msg_base_url,
                        custom_providers=_msg_custom_providers,
                    )
                    if _msg_custom_ctx:
                        _msg_config_ctx = _msg_custom_ctx
                except Exception:
                    pass
            _msg_ctx_len = await get_model_context_length_async(
                _msg_model,
                base_url=_msg_base_url,
                api_key=_msg_runtime.get("api_key") or "",
                config_context_length=_msg_config_ctx,
                provider=_msg_runtime.get("provider") or "",
                custom_providers=_msg_custom_providers,
            )
            _ctx_result = await preprocess_context_references_async(
                message_text,
                cwd=_msg_cwd,
                context_length=_msg_ctx_len,
                allowed_root=_msg_cwd,
            )
            if _ctx_result.blocked:
                _adapter = runner._adapter_for_source(source)
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

