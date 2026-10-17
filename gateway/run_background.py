"""Finite background tasks launched by the gateway /bg command."""

from contextlib import suppress
import logging
import os
from typing import Optional

from agent.i18n import t
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


def _bg_prompt_preview(prompt: str, limit: int = 60) -> str:
    """Short single-line quote of a /bg prompt for its failure notice (the task id means nothing to the user)."""
    text = " ".join(str(prompt or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class GatewayBackgroundMixin:
    async def _run_background_task(
        self, prompt: str, source: SessionSource, task_id: str,
        event_message_id: Optional[str] = None, media_urls: Optional[list[str]] = None,
        media_types: Optional[list[str]] = None,
    ) -> None:
        """Profile-scoping wrapper around the background agent task (mirrors ``_run_agent``)."""
        with self._profile_scope_for_source(source):
            return await self._run_background_task_inner(
                prompt, source, task_id, event_message_id, media_urls, media_types,
            )

    async def _run_background_task_inner(
        self, prompt: str, source: SessionSource, task_id: str,
        event_message_id: Optional[str] = None, media_urls: Optional[list[str]] = None,
        media_types: Optional[list[str]] = None,
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        from gateway.run import (
            _checkpoint_agent_kwargs, _current_max_iterations, _load_gateway_config,
            _platform_config_key,
        )
        from run_agent import AIAgent
        media_urls = media_urls or []
        media_types = media_types or []
        adapter = self._delivery_adapter_for(source)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return
        _thread_metadata = self._thread_metadata_for_source(source, event_message_id)

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(source=source, user_config=user_config)
            if not runtime_kwargs.get("api_key"):
                await adapter.send(source.chat_id, t("gateway.background.no_credentials"), metadata=_thread_metadata)
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
                response = t("gateway.shared.error_prefix", error=result["error"])
            # Fresh conversation, so history_offset=0: every message in the run belongs to this turn.
            if response:
                response = repair_explicit_computer_use_media_paths(response, result.get("messages", []))

            preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
            header = t("gateway.background.complete_header", preview=preview)
            images, media_files, text_content = [], [], ""
            if response:
                media_files, response = adapter.extract_media(response)
                media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
                images, text_content = adapter.extract_images(response)
            if text_content:
                await adapter.send(chat_id=source.chat_id, content=header + text_content, metadata=_thread_metadata)
            elif not images and not media_files:
                await adapter.send(
                    chat_id=source.chat_id, content=header + t("gateway.background.no_response"), metadata=_thread_metadata,
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

        except Exception:
            logger.exception("Background task %s failed", task_id)
            # Automatic failure diagnostic (the task produced no requested result to deliver).
            with suppress(Exception):
                await adapter.emit_warning(
                    source.chat_id,
                    t("gateway.background.failed", preview=_bg_prompt_preview(prompt)),
                    metadata=_thread_metadata, logical_platform=source.platform,
                )
