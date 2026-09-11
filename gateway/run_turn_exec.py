"""run_turn_exec — the _run* methods split out of GatewayTurnMixin (mechanical extraction)."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import asyncio
import inspect
import json
import os
import queue
import threading
import time
from agent.session_activity import format_iteration_progress
from contextlib import suppress
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from typing import Any, Callable, Dict, List, Optional, Tuple
if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner
logger = logging.getLogger("gateway.run")

class GatewayTurnExecMixin:
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

    def _run_still_current_fn(self, session_key: Optional[str], run_generation: Optional[int]) -> Callable[[], bool]:
        """Predicate: does this run's generation still own ``session_key``? (always True when untracked)."""
        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)
        return _run_still_current

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
        # Operator-facing log keeps the raw resolved value; only the user-facing lines hide the sentinel.
        logger.error(
            "Agent idle for %.0fs (timeout %.0fs) in session %s "
            "| last_activity=%s | iteration=%s/%s | tool=%s",
            _secs_ago, worker.agent_timeout, session_key, _last_desc, _iter_n, _iter_max,
            _cur_tool or "none",
        )
        if _timed_out_agent:
            request_hard_interrupt(_timed_out_agent, _INTERRUPT_REASON_TIMEOUT)
        _timeout_mins = int(worker.agent_timeout // 60) or 1
        _iter_progress = format_iteration_progress(_iter_n, _iter_max)
        _diag_lines = [
            f"⏱️ Agent inactive for {_timeout_mins} min — no tool calls or API responses."
        ]
        if _cur_tool:
            _diag_lines.append(
                f"The agent appears stuck on tool `{_cur_tool}` ({_secs_ago:.0f}s since last "
                f"activity, {_iter_progress})."
            )
        else:
            _diag_lines.append(
                f"Last activity: {_last_desc} ({_secs_ago:.0f}s ago, "
                f"{_iter_progress}). "
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
        if _agent.model != _cfg_model and not self._is_intentional_model_switch(session_key, _agent, _cfg_model):
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
                    # The text send records a delivery-ledger obligation under this key, keyed on
                    # the raw inbound id (the anchor above is only the reply target).
                    session_key=session_key, inbound_message_id=turn_ctx.inbound_message_id,
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
        # The raw inbound id keys the delivery-ledger obligation for the follow-up's own final send,
        # distinct from the reply anchor above (None in forum topics). Carry it or two chained
        # topic turns with the same text would collide on one obligation id (queued-final-ledger).
        next_inbound_id = None
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
            next_inbound_id = str(pending_event.message_id) if getattr(pending_event, "message_id", None) else None
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
            event_message_id=next_message_id, inbound_message_id=next_inbound_id,
            channel_prompt=next_channel_prompt, message_type=next_message_type,
        )
        merged = _preserve_queued_followup_history_offset(result, followup_result)
        # The TERMINAL turn of the chain owns the ledger identity for the outer final send, which
        # the adapter brackets against the event that OPENED the chain. Without this the terminal
        # reply is recorded under the first message's id, so a first reply that was refused (flood
        # control) has its outstanding row replaced and marked delivered by an identical-text
        # terminal reply, and is never redelivered. A deeper recursion has already set its own id,
        # so only fill the key while it is still absent: the innermost turn wins.
        if isinstance(merged, dict) and "queued_terminal_inbound_id" not in merged:
            merged = {**merged, "queued_terminal_inbound_id": next_inbound_id}
        return merged

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
                        _parts.append(format_iteration_progress(_a["api_call_count"], _a["max_iterations"]))
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
        persist_user_display_metadata: Optional[dict] = None,
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
            persist_user_display_metadata=persist_user_display_metadata,
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
