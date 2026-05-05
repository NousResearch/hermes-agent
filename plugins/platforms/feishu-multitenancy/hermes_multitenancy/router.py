"""Pre-gateway-dispatch hook callback (sync) + async dispatch entry point.

Wires together: SQLite RoutingTable (production) + in-memory _SPIKE_ROUTING
(fallback) + LRU RuntimePool (cached profile runtimes) + Hermes-derived slash
command dispatch.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # Shared Hermes Feishu card/typewriter transport.
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig  # type: ignore
except Exception:  # pragma: no cover - allows plugin unit tests without gateway on path
    GatewayStreamConsumer = None  # type: ignore
    StreamConsumerConfig = None  # type: ignore

# Per-user in-flight dispatch tasks — used by /stop to cancel the right task.
_user_inflight_tasks: dict[str, asyncio.Task] = {}

# Multi-turn session memory: maps (profile_name, user_key) → list of messages.
# user_key prefers the real Feishu open_id (ou_*) used for routing, falling
# back to union_id / alternate identifiers for legacy rows. SessionStore
# persists the same key so memory survives gateway restarts.
_SESSION_HISTORY_MAX = 20  # keep last N messages (user + assistant alternating)
_session_history: dict[tuple[str, str], list[dict]] = {}
# Tracks which (profile, user_key) pairs we've lazy-loaded from SessionStore
# so we only hit SQLite once per pair per process lifetime.
_session_loaded: set[tuple[str, str]] = set()


def _history_key(profile_name: str, sender: str, sender_alt: Optional[str]) -> tuple[str, str]:
    """Return the per-(profile, user) key used to look up conversation history."""
    return (profile_name, sender_alt or sender)


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep at most ``_SESSION_HISTORY_MAX`` most recent messages."""
    if len(history) <= _SESSION_HISTORY_MAX:
        return history
    return history[-_SESSION_HISTORY_MAX:]


def _load_history(key: tuple[str, str]) -> list[dict]:
    """Get history for ``key`` — first call hydrates from SessionStore, subsequent calls hit cache."""
    if key in _session_loaded:
        return list(_session_history.get(key, []))
    store = _get_session_store()
    if store is not None:
        try:
            persisted = store.load_recent(key[0], key[1], _SESSION_HISTORY_MAX)
        except Exception as exc:
            logger.debug("multitenancy: SessionStore.load_recent failed (%s)", exc)
            persisted = []
        if persisted:
            _session_history[key] = persisted
    _session_loaded.add(key)
    return list(_session_history.get(key, []))


def _persist_turn(key: tuple[str, str], user_msg: dict, assistant_text: str) -> None:
    """Append a (user, assistant) turn to both in-memory cache and SessionStore."""
    new_history = _session_history.get(key, []) + [
        user_msg,
        {"role": "assistant", "content": assistant_text},
    ]
    _session_history[key] = _trim_history(new_history)
    store = _get_session_store()
    if store is None:
        return
    try:
        # _build_user_message always sets content as a str — no cast needed.
        store.append(key[0], key[1], user_msg["role"], user_msg["content"])
        store.append(key[0], key[1], "assistant", assistant_text)
    except Exception as exc:
        logger.debug("multitenancy: SessionStore.append failed (%s)", exc)


def _clear_history(key: tuple[str, str]) -> bool:
    """Drop a user's history from cache + store. Returns True if anything was cleared."""
    had_cache = _session_history.pop(key, None) is not None
    _session_loaded.discard(key)
    store = _get_session_store()
    if store is None:
        return had_cache
    try:
        rows = store.clear(key[0], key[1])
    except Exception as exc:
        logger.debug("multitenancy: SessionStore.clear failed (%s)", exc)
        rows = 0
    return had_cache or rows > 0


def _build_user_message(event: Any, *, text_override: Optional[str] = None) -> dict:
    """Construct the OpenAI-format user message, splicing in reply context if any.

    Reply context: hermes mainstream sets ``event.reply_to_text`` when the
    user is quoting an earlier message. We surface it inline so the model
    knows what's being replied to.

    ``text_override`` lets ``_enrich_with_vision`` rewrite the text before
    the message is built (so reply context still wraps the enriched text).
    """
    text = text_override if text_override is not None else (getattr(event, "text", "") or "")
    reply_to_text = getattr(event, "reply_to_text", None)
    if reply_to_text:
        text = f"(replying to: {reply_to_text})\n{text}"
    return {"role": "user", "content": text}


def _is_feishu_open_id(value: Any) -> bool:
    return bool(value) and str(value).startswith("ou_")


def _nested_get(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _current_sender_open_id() -> Optional[str]:
    """Return the adapter-provided Feishu open_id context, if available."""
    try:
        from tools import feishu_oapi_client as feishu_oapi  # type: ignore

        candidate = feishu_oapi.current_sender_open_id.get()
    except Exception:
        return None
    return str(candidate) if _is_feishu_open_id(candidate) else None


def _resolve_sender_for_routing(event: Any, *, fallback: str = "unknown") -> str:
    """Pick the stable Feishu user key used by the multitenancy router.

    Feishu SDK events can expose ``source.user_id`` as a short tenant-local ID
    such as ``g41a5b5g``. The UAT files and explicit routes are keyed by the
    real app-scoped open_id (``ou_*``), so prefer that when the adapter or raw
    event makes it available.
    """
    source = getattr(event, "source", None)
    direct_candidates = (
        _current_sender_open_id(),
        getattr(event, "sender_open_id", None),
        getattr(source, "open_id", None) if source is not None else None,
        getattr(source, "user_id", None) if source is not None else None,
        getattr(source, "user_id_alt", None) if source is not None else None,
    )
    for candidate in direct_candidates:
        if _is_feishu_open_id(candidate):
            return str(candidate)

    raw_candidates = (
        getattr(event, "raw", None),
        getattr(event, "raw_event", None),
        getattr(event, "event", None),
    )
    paths = (
        ("sender", "sender_id", "open_id"),
        ("event", "sender", "sender_id", "open_id"),
        ("event", "message", "sender", "sender_id", "open_id"),
        ("message", "sender", "sender_id", "open_id"),
        ("sender_id", "open_id"),
    )
    for raw in raw_candidates:
        for path in paths:
            candidate = _nested_get(raw, path)
            if _is_feishu_open_id(candidate):
                return str(candidate)

    return fallback


def _event_with_text(event: Any, text: str) -> Any:
    """Return an event-shaped object whose ``text`` matches the agent prompt."""
    if text == (getattr(event, "text", "") or ""):
        return event
    cloned = copy.copy(event)
    setattr(cloned, "text", text)
    return cloned


def _clean_stream_display_text(text: str) -> str:
    """Hide native media-delivery directives from visible streaming text."""
    try:
        from gateway.stream_consumer import GatewayStreamConsumer  # type: ignore

        return GatewayStreamConsumer._clean_for_display(text)
    except Exception:
        cleaned = str(text or "").replace("[[audio_as_voice]]", "")
        cleaned = re.sub(r'''[`"']?MEDIA:\s*\S+[`"']?''', "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.rstrip()


async def _deliver_media_from_stream_response(gateway: Any, response: str, event: Any, adapter: Any) -> None:
    """Delegate post-stream media attachment delivery to Hermes' native gateway path."""
    deliver = getattr(gateway, "_deliver_media_from_response", None)
    if not callable(deliver):
        return
    await deliver(response, event, adapter)


async def _enrich_via_hermes_pipeline(event: Any, gateway: Any) -> Optional[str]:
    """Delegate inbound preprocessing to hermes' ``_prepare_inbound_message_text``.

    This is the single call that mainstream uses to:
      - run vision_analyze_tool on attached images
      - run transcribe_audio on voice messages
      - inject text-file content (.txt / .md / .csv / etc.)
      - prepend reply-quoted context
      - attribute multi-user shared sessions

    By calling the same gateway method, the plugin behaves *identically* to
    mainstream for every multimodal input — no re-implementation, no drift.

    Caveat: this depends on a private GatewayRunner method. If hermes-agent
    refactors ``_prepare_inbound_message_text``, swap to local fallbacks
    (``_local_enrich_with_vision_only`` below as a minimal safety net).

    Returns:
        Enriched text string, or None on failure (caller falls back to event.text).
    """
    if gateway is None:
        return None
    prep = getattr(gateway, "_prepare_inbound_message_text", None)
    if prep is None or not callable(prep):
        logger.debug("multitenancy: gateway._prepare_inbound_message_text unavailable")
        return await _local_enrich_with_vision_only(event)
    source = getattr(event, "source", None)
    if source is None:
        return None
    try:
        return await prep(event=event, source=source, history=[])
    except Exception as exc:
        logger.debug("multitenancy: gateway._prepare_inbound_message_text failed (%s)", exc)
        return await _local_enrich_with_vision_only(event)


async def _local_enrich_with_vision_only(event: Any) -> Optional[str]:
    """Local fallback if hermes' ``_prepare_inbound_message_text`` is unavailable.

    Only handles images (the most common multimodal input). Audio / files /
    reply context degrade gracefully — the model will see ``event.text`` only.
    """
    media_urls = getattr(event, "media_urls", None) or []
    media_types = getattr(event, "media_types", None) or []
    if not media_urls:
        return None
    try:
        from tools.vision_tools import vision_analyze_tool  # type: ignore
    except ImportError:
        return None
    import json as _json
    descriptions: list[str] = []
    for path, mtype in zip(media_urls, media_types or [""] * len(media_urls)):
        if mtype and not mtype.startswith("image"):
            continue
        try:
            result_json = await vision_analyze_tool(
                image_url=path,
                user_prompt="Describe this image in thorough detail.",
            )
            result = _json.loads(result_json) if isinstance(result_json, str) else result_json
            if result.get("success"):
                descriptions.append(f"[Image: {result.get('analysis', '')}]")
        except Exception as exc:
            logger.debug("multitenancy: local vision fallback error on %s: %s", path, exc)
    if not descriptions:
        return None
    base = getattr(event, "text", "") or ""
    return "\n".join(descriptions) + ("\n" + base if base else "")


# -- Hook entry point --------------------------------------------------------


def on_pre_gateway_dispatch(*, event: Any, gateway: Any, session_store: Any = None, **_kwargs) -> dict:
    """Sync hook callback (registered to ``pre_gateway_dispatch``).

    Schedules the async work as a background task and returns immediately
    with ``action: skip`` so the gateway main flow halts for this event.
    """
    try:
        if _should_defer_gateway_processing_complete(event):
            _defer_gateway_processing_complete(event, gateway)
        loop = asyncio.get_running_loop()
        task = loop.create_task(handle_async(event=event, gateway=gateway))
        task.add_done_callback(_log_task_failure)
    except RuntimeError:
        # Test-only fallback: hook called from sync context (no running loop).
        if os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                asyncio.run(handle_async(event=event, gateway=gateway))
            except Exception as exc:
                logger.warning("multitenancy: sync fallback dispatch failed: %s", exc)
        else:
            logger.error(
                "multitenancy: pre_gateway_dispatch invoked without a running loop "
                "and not in pytest — dropping event"
            )
    except Exception as exc:
        logger.warning("multitenancy: failed to schedule handle_async: %s", exc)
    return {"action": "skip", "reason": "multitenancy router took over"}


def _should_defer_gateway_processing_complete(event: Any) -> bool:
    """Return True when the async router owns visible processing lifecycle."""
    from .commands import parse_command

    source = getattr(event, "source", None)
    fallback_sender = getattr(source, "user_id", "unknown") if source else "unknown"
    sender = _resolve_sender_for_routing(event, fallback=fallback_sender)
    sender_alt = getattr(source, "user_id_alt", None) if source else None
    text = getattr(event, "text", "") or ""
    if parse_command(text) is not None:
        return False
    _profile_name, profile_home = _resolve_route(sender, alt_id=sender_alt)
    return profile_home is not None or _auto_provision_enabled()


def _defer_gateway_processing_complete(event: Any, gateway: Any) -> None:
    adapter = _get_feishu_adapter(gateway)
    defer = getattr(adapter, "defer_processing_complete", None)
    if not callable(defer):
        return
    try:
        defer(event)
    except Exception as exc:
        logger.debug("multitenancy: defer_processing_complete failed: %s", exc)


# -- Async dispatch ----------------------------------------------------------


async def handle_async(*, event: Any, gateway: Any) -> None:
    """Async dispatch — orchestrates routing + pool + adapter calls + commands."""
    from .commands import parse_command

    try:
        source = getattr(event, "source", None)
        chat_id = getattr(source, "chat_id", "unknown") if source else "unknown"
        fallback_sender = getattr(source, "user_id", "unknown") if source else "unknown"
        sender = _resolve_sender_for_routing(event, fallback=fallback_sender)
        text = getattr(event, "text", "") or ""

        sender_alt = getattr(source, "user_id_alt", None) if source else None

        # Slash command short-circuit (resolve route first so /status / /new
        # know which profile's history to inspect). When _resolve_route signals
        # a miss with profile_home=None, surface profile_name=None so command
        # handlers reply "未路由" instead of leaking the sender id.
        cmd_pair = parse_command(text)
        if cmd_pair is not None:
            cmd_profile_name, cmd_profile_home = _resolve_route(sender, alt_id=sender_alt)
            cmd_profile = cmd_profile_name if cmd_profile_home is not None else None
            await _handle_command(
                cmd_pair,
                sender,
                sender_alt,
                cmd_profile,
                cmd_profile_home,
                chat_id,
                gateway,
                event,
            )
            return

        # Routing: SQLite table first, then in-memory spike fallback.
        profile_name, profile_home = _resolve_or_auto_provision_route(sender, alt_id=sender_alt)
        if profile_home is None:
            logger.info("multitenancy: no route for sender=%s, ignoring", sender)
            return

        # Register self in the user's in-flight slot (replace previous)
        current = asyncio.current_task()
        prev = _user_inflight_tasks.get(sender)
        if prev is not None and not prev.done() and prev is not current:
            prev.cancel()
        if current is not None:
            _user_inflight_tasks[sender] = current

        adapter = _get_feishu_adapter(gateway)
        # Detect whether adapter supports the streaming/reaction APIs we use.
        # Real FeishuAdapter does; unit-test mocks typically don't.
        feishu_full = (
            adapter is not None
            and hasattr(adapter, "edit_message")
            and hasattr(adapter, "on_processing_start")
            and hasattr(adapter, "on_processing_complete")
        )

        outcome_failed = False
        if feishu_full:
            try:
                await adapter.on_processing_start(event)
            except Exception as exc:
                logger.debug("multitenancy: on_processing_start failed: %s", exc)

        # Multi-modal enrichment — delegate to hermes' canonical pipeline so
        # vision (images), STT (audio), text-file inject (.txt/.md/.csv etc.),
        # reply-context wrapping, and multi-user attribution all behave EXACTLY
        # like mainstream. Falls back to local vision-only on missing API.
        enriched_text = await _enrich_via_hermes_pipeline(event, gateway)

        # Build the conversation: prior history + current user message (with
        # reply context spliced in). The runner prepends the profile's SOUL.
        # First lookup for a (profile, user) pair hydrates from SessionStore.
        hist_key = _history_key(profile_name, sender, sender_alt)
        prior = _load_history(hist_key)
        user_msg = _build_user_message(event, text_override=enriched_text)
        conversation = prior + [user_msg]
        agent_event = _event_with_text(event, user_msg["content"])

        try:
            if feishu_full:
                # Streaming path — card stream when available; text edit fallback.
                response_text = await _stream_into_feishu(
                    adapter, chat_id, profile_name, profile_home, agent_event,
                    messages=conversation,
                )
                if response_text:
                    await _deliver_media_from_stream_response(
                        gateway, response_text, agent_event, adapter
                    )
            else:
                # Mock / minimal adapter — old non-stream path (send_typing + pool.dispatch + send)
                if adapter is not None:
                    await _safe_call(adapter.send_typing, chat_id)
                response_text = await _get_pool().dispatch(profile_name, profile_home, agent_event)
                if adapter is not None:
                    await _safe_call(adapter.send, chat_id, response_text)

            # Record turn into history + persist to SessionStore.
            if response_text and isinstance(response_text, str):
                _persist_turn(hist_key, user_msg, response_text)

            _touch_route(sender, sender_alt)
        except Exception:
            outcome_failed = True
            raise
        finally:
            if feishu_full:
                try:
                    out = _processing_outcome(failed=outcome_failed)
                    complete_deferred = getattr(adapter, "complete_deferred_processing", None)
                    if callable(complete_deferred):
                        await complete_deferred(event, out)
                    else:
                        await adapter.on_processing_complete(event, out)
                except Exception as exc:
                    logger.debug("multitenancy: on_processing_complete failed: %s", exc)
            if _user_inflight_tasks.get(sender) is current:
                _user_inflight_tasks.pop(sender, None)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("multitenancy: handle_async failed: %s", exc)


def _processing_outcome(*, failed: bool) -> Any:
    """Return Hermes' ProcessingOutcome enum, or a string-compatible fallback."""
    try:
        from gateway.platforms.base import ProcessingOutcome  # type: ignore

        return ProcessingOutcome.FAILURE if failed else ProcessingOutcome.SUCCESS
    except Exception:
        class _FallbackOutcome:
            def __str__(self) -> str:
                status = "FAILURE" if failed else "SUCCESS"
                return f"ProcessingOutcome.{status}"

        return _FallbackOutcome()


# -- Command dispatch --------------------------------------------------------


async def _handle_command(
    pair: tuple[str, str],
    sender: str,
    sender_alt: Optional[str],
    profile_name: Optional[str],
    profile_home: Optional[Path],
    chat_id: str,
    gateway: Any,
    event: Any,
) -> None:
    """Execute a parsed slash command and reply via the shared adapter."""
    cmd, _args = pair
    adapter = _get_feishu_adapter(gateway)

    if cmd == "stop":
        task = _user_inflight_tasks.pop(sender, None)
        if task is not None and not task.done():
            task.cancel()
            reply = "已停止当前任务"
        else:
            reply = "没有进行中的任务"
    elif cmd == "status":
        task = _user_inflight_tasks.get(sender)
        running = task is not None and not task.done()
        # Surface session memory size + profile so the user knows their context.
        if profile_name:
            hist = _session_history.get(_history_key(profile_name, sender, sender_alt), [])
            hist_len = len(hist)
        else:
            hist_len = 0
        reply = (
            f"状态: {'运行中' if running else '空闲'}\n"
            f"profile: {profile_name or '(未路由)'}\n"
            f"会话历史: {hist_len} 条消息"
        )
    elif cmd in ("new", "reset"):
        # Clear this user's per-profile history (cache + persistent SessionStore).
        if profile_name:
            key = _history_key(profile_name, sender, sender_alt)
            had = _clear_history(key)
            reply = "会话已重置 ✅" if had else "会话已重置 (本来也是空的)"
        else:
            reply = "(未路由的用户) 没有历史可重置"
    elif cmd == "help":
        reply = _gateway_help_text()
    else:
        dispatched = await _dispatch_gateway_command(
            cmd,
            event,
            gateway,
            sender=sender,
            sender_alt=sender_alt,
            profile_name=profile_name,
            profile_home=profile_home,
            chat_id=chat_id,
        )
        if dispatched is not None:
            reply = dispatched
        else:
            from .commands import is_known_command, unknown_command_message

            reply = (
                f"Command `/{cmd}` is recognized by Hermes, but this gateway does not "
                "expose a reusable command dispatcher yet."
                if is_known_command(cmd)
                else unknown_command_message(cmd)
            )

    if adapter is not None:
        await _safe_call(adapter.send, chat_id, reply)


async def _dispatch_gateway_command(
    cmd: str,
    event: Any,
    gateway: Any,
    *,
    sender: str,
    sender_alt: Optional[str],
    profile_name: Optional[str],
    profile_home: Optional[Path],
    chat_id: str,
) -> Optional[str]:
    """Delegate a Hermes-known slash command to the gateway when possible."""
    _ensure_command_event_methods(event, cmd)

    dispatcher = getattr(gateway, "_dispatch_slash_command", None)
    if callable(dispatcher):
        with _profile_gateway_context(
            gateway,
            event,
            sender=sender,
            sender_alt=sender_alt,
            profile_name=profile_name,
            profile_home=profile_home,
            chat_id=chat_id,
        ):
            try:
                result = dispatcher(event, multitenancy_context={
                    "profile_name": profile_name,
                    "profile_home": str(profile_home) if profile_home else "",
                    "sender_open_id": sender,
                    "session_key_override": _multitenant_gateway_session_key(
                        event,
                        profile_name=profile_name,
                        sender=sender,
                        sender_alt=sender_alt,
                        chat_id=chat_id,
                    ),
                })
            except TypeError:
                result = dispatcher(event)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result) if result is not None else None

    handler = _gateway_handler_for_command(gateway, cmd)
    if handler is None:
        return None
    with _profile_gateway_context(
        gateway,
        event,
        sender=sender,
        sender_alt=sender_alt,
        profile_name=profile_name,
        profile_home=profile_home,
        chat_id=chat_id,
    ):
        result = handler(event)
        if asyncio.iscoroutine(result):
            result = await result
    return str(result) if result is not None else None


def _gateway_handler_for_command(gateway: Any, cmd: str) -> Any:
    """Return Hermes' handler method using naming conventions, not a command table."""
    normalized = cmd.replace("-", "_")
    candidates = [f"_handle_{normalized}_command"]
    if normalized == "sethome":
        candidates.append("_handle_set_home_command")
    for name in candidates:
        handler = getattr(gateway, name, None)
        if callable(handler):
            return handler
    return None


def _ensure_command_event_methods(event: Any, cmd: str) -> None:
    """Add minimal MessageEvent command helpers for tests/fallback objects."""
    text = getattr(event, "text", "") or ""
    args = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else ""
    if not callable(getattr(event, "get_command", None)):
        setattr(event, "get_command", lambda: cmd)
    if not callable(getattr(event, "get_command_args", None)):
        setattr(event, "get_command_args", lambda: args)


def _profile_gateway_context(
    gateway: Any,
    event: Any,
    *,
    sender: str,
    sender_alt: Optional[str],
    profile_name: Optional[str],
    profile_home: Optional[Path],
    chat_id: str,
):
    """Temporarily scope Hermes gateway helpers to the routed profile."""
    class _Context:
        def __enter__(self):
            self._old_home = os.environ.get("HERMES_HOME")
            self._had_home = "HERMES_HOME" in os.environ
            self._old_session_key_for_source = getattr(gateway, "_session_key_for_source", None)
            if profile_home is not None:
                os.environ["HERMES_HOME"] = str(profile_home)
            session_key = _multitenant_gateway_session_key(
                event,
                profile_name=profile_name,
                sender=sender,
                sender_alt=sender_alt,
                chat_id=chat_id,
            )
            if session_key:
                def _scoped_session_key_for_source(source):
                    return session_key

                setattr(gateway, "_session_key_for_source", _scoped_session_key_for_source)
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._old_session_key_for_source is not None:
                setattr(gateway, "_session_key_for_source", self._old_session_key_for_source)
            elif hasattr(gateway, "_session_key_for_source"):
                try:
                    delattr(gateway, "_session_key_for_source")
                except Exception:
                    pass
            if self._had_home:
                os.environ["HERMES_HOME"] = self._old_home or ""
            else:
                os.environ.pop("HERMES_HOME", None)
            return False

    return _Context()


def _multitenant_gateway_session_key(
    event: Any,
    *,
    profile_name: Optional[str],
    sender: str,
    sender_alt: Optional[str],
    chat_id: str,
) -> Optional[str]:
    if not profile_name:
        return None
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", None) or "feishu"
    user_key = sender_alt or sender
    return f"multitenancy:{platform}:{profile_name}:{chat_id}:{user_key}"


def _gateway_help_text() -> str:
    """Render help from Hermes' central command registry when available."""
    try:
        from hermes_cli.commands import gateway_help_lines  # type: ignore

        lines = gateway_help_lines()
        if lines:
            return "📖 可用命令\n" + "\n".join(lines[:30])
    except Exception:
        pass
    return (
        "📖 可用命令\n"
        "/help    — 显示这条帮助\n"
        "/status  — 查看当前 profile + 历史长度\n"
        "/new     — 重置会话历史 (开始新对话)\n"
        "/reset   — /new 的别名\n"
        "/stop    — 取消正在运行的任务\n"
        "/model   — 切换当前会话模型\n"
        "/reasoning — 管理推理强度\n"
        "/voice   — 切换语音回复模式\n"
    )


# -- Routing resolution ------------------------------------------------------


def _resolve_route(sender: str, *, alt_id: Optional[str] = None) -> tuple[str, Optional[Path]]:
    """Resolve sender → (profile_name, profile_home).

    The routing table's ``open_id`` column is overloaded as "any stable user
    identifier" — it can hold a real Feishu open_id (``ou_xxx``), a union_id
    (``on_xxx``), or any other tenant-stable token chosen by feishu-sync.

    Lookup order:
      1. SQLite RoutingTable WHERE open_id = sender (typical: real Feishu ou_* open_id)
      2. SQLite RoutingTable WHERE open_id = alt_id (legacy: source.user_id_alt = union_id)
      3. In-memory ``_SPIKE_ROUTING`` dict (Phase 1 compat / unit tests)

    Returns (sender, None) when no route hits.
    """
    from .runtime import resolve_profile_home as _spike_resolve

    table = _get_routing_table()
    candidates: list[str] = [sender]
    if alt_id and alt_id != sender:
        candidates.append(alt_id)

    if table is not None:
        for candidate in candidates:
            try:
                row = table.lookup_by_open_id(candidate)
            except Exception as exc:
                logger.debug("multitenancy: routing lookup failed (%s)", exc)
                continue
            if row is not None:
                return (row.profile_name, _profile_name_to_home(row.profile_name))

    # Fallback: in-memory spike routing dict
    for candidate in candidates:
        spike_home = _spike_resolve(candidate)
        if spike_home is not None:
            return (spike_home.name, spike_home)
    return (sender, None)


def _resolve_or_auto_provision_route(
    sender: str,
    *,
    alt_id: Optional[str] = None,
) -> tuple[str, Optional[Path]]:
    """Resolve an existing route, or create a dedicated profile for a new sender."""
    profile_name, profile_home = _resolve_route(sender, alt_id=None)
    if profile_home is not None:
        _repair_auto_profile(profile_name, profile_home, route_key=sender, sender=sender)
        return profile_name, profile_home
    if _auto_provision_enabled():
        provisioned = _auto_provision_route(sender, alt_id=alt_id)
        if provisioned is not None:
            return provisioned
        return profile_name, None
    return _resolve_route(sender, alt_id=alt_id)


def _repair_auto_profile(
    profile_name: str,
    profile_home: Path,
    *,
    route_key: str,
    sender: str,
) -> None:
    if not profile_name.startswith("feishu_"):
        return
    try:
        _ensure_auto_profile(profile_name, profile_home, route_key=route_key, sender=sender)
    except Exception as exc:
        logger.debug("multitenancy: auto profile repair failed for %s: %s", profile_name, exc)


def _auto_provision_enabled() -> bool:
    value = os.environ.get("HERMES_MULTITENANCY_AUTO_PROVISION", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _auto_provision_route(
    sender: str,
    *,
    alt_id: Optional[str] = None,
) -> Optional[tuple[str, Path]]:
    """Create a route/profile for an unseen Feishu user, then return it."""
    if not _auto_provision_enabled():
        return None
    table = _get_routing_table()
    if table is None:
        return None

    route_key = sender
    if not route_key or route_key == "unknown":
        return None

    profile_name = _auto_profile_name(route_key)
    profile_home = _profile_name_to_home(profile_name)
    try:
        _ensure_auto_profile(profile_name, profile_home, route_key=route_key, sender=sender)
        table.upsert(
            user_id=route_key,
            profile_name=profile_name,
            open_id=route_key,
            union_id=alt_id if alt_id and alt_id != sender else None,
        )
    except Exception as exc:
        logger.warning(
            "multitenancy: auto-provision failed sender=%s alt=%s: %s",
            sender,
            alt_id,
            exc,
        )
        return None

    logger.info(
        "multitenancy: auto-provisioned sender=%s alt=%s profile=%s",
        sender,
        alt_id,
        profile_name,
    )
    return profile_name, profile_home


def _auto_profile_name(route_key: str) -> str:
    """Return a deterministic, filesystem-safe profile name for a tenant key."""
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", route_key).strip("_")
    if not clean:
        clean = "unknown"
    return f"feishu_{clean}"


def _ensure_auto_profile(
    profile_name: str,
    profile_home: Path,
    *,
    route_key: str,
    sender: str,
) -> None:
    """Create the on-disk profile skeleton required by AIAgent."""
    profile_home.mkdir(parents=True, exist_ok=True)
    shared_home = _shared_home_for_profile(profile_home)

    config_path = profile_home / "config.yaml"
    if config_path.exists():
        _normalize_profile_config_file(config_path, shared_home=shared_home)
    else:
        config_path.write_text(
            _dump_profile_config(_profile_config_from_shared_home(shared_home)),
            encoding="utf-8",
        )

    soul_path = profile_home / "SOUL.md"
    if not soul_path.exists():
        soul_path.write_text(
            "\n".join(
                [
                    f"# Hermes Profile {profile_name}",
                    "",
                    f"You are the dedicated Hermes tenant profile for Feishu route `{route_key}`.",
                    f"The current Feishu sender open_id is `{sender}`.",
                    "Keep tools, files, memory, and responses isolated to this profile.",
                    "Do not claim to be another Hermes profile.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    for name in ("auth.json", ".env"):
        _ensure_shared_profile_file(profile_home, shared_home, name)


def _shared_home_for_profile(profile_home: Path) -> Path:
    """Infer the shared Hermes root from a profile path."""
    if profile_home.parent.name == "profiles":
        return profile_home.parent.parent
    return Path.home() / ".hermes"


def _profile_config_from_shared_home(shared_home: Path) -> dict[str, Any]:
    """Build a minimal profile config from the shared Hermes config."""
    config: dict[str, Any] = {}
    shared_config = shared_home / "config.yaml"
    if shared_config.exists():
        try:
            import yaml

            loaded = yaml.safe_load(shared_config.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                for key in ("model", "fallback", "platform_toolsets"):
                    value = loaded.get(key)
                    if value:
                        config[key] = value
                feishu_platform = ((loaded.get("platforms") or {}).get("feishu") or None)
                if feishu_platform:
                    config["platforms"] = {"feishu": feishu_platform}
        except Exception as exc:
            logger.debug("multitenancy: failed to read shared config %s: %s", shared_config, exc)

    return _normalize_profile_config(config)


def _normalize_profile_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if isinstance(model, dict) and model.get("default"):
        default_model = str(model.get("default") or "").strip()
        provider = str(model.get("provider") or "").strip()
        if default_model and provider and "/" not in default_model:
            model["default"] = f"{provider}/{default_model}"
    return config


def _normalize_profile_config_file(config_path: Path, *, shared_home: Path) -> None:
    try:
        import yaml

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.debug("multitenancy: failed to normalize profile config %s: %s", config_path, exc)
        return
    if not isinstance(loaded, dict):
        return
    before = json.dumps(loaded, sort_keys=True, ensure_ascii=True)
    _merge_shared_feishu_platform(loaded, shared_home)
    normalized = _normalize_profile_config(loaded)
    after = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
    if after != before:
        config_path.write_text(_dump_profile_config(normalized), encoding="utf-8")


def _merge_shared_feishu_platform(config: dict[str, Any], shared_home: Path) -> None:
    shared_config = shared_home / "config.yaml"
    if not shared_config.exists():
        return
    try:
        import yaml

        loaded = yaml.safe_load(shared_config.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.debug("multitenancy: failed to read shared Feishu platform config %s: %s", shared_config, exc)
        return
    if not isinstance(loaded, dict):
        return
    feishu_platform = ((loaded.get("platforms") or {}).get("feishu") or None)
    if not feishu_platform:
        return
    platforms = config.setdefault("platforms", {})
    if isinstance(platforms, dict):
        platforms["feishu"] = feishu_platform


def _dump_profile_config(config: dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(config, sort_keys=False, allow_unicode=False)
    except Exception:
        return json.dumps(config, indent=2, ensure_ascii=True) + "\n"


def _ensure_shared_profile_file(profile_home: Path, shared_home: Path, name: str) -> None:
    source = shared_home / name
    target = profile_home / name
    if target.exists() or target.is_symlink() or not source.exists():
        return
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _profile_name_to_home(profile_name: str) -> Path:
    """Map profile_name to its on-disk profile home directory.

    Mirrors ``hermes_cli/profiles.py`` convention: ``~/.hermes/profiles/<name>``.
    """
    return Path.home() / ".hermes" / "profiles" / profile_name


def _touch_route(sender: str, sender_alt: Optional[str] = None) -> None:
    """Best-effort last_active_at update; no-op if no SQLite table or row.

    Touch both identifiers because auto-provisioned rows are keyed by
    app-scoped ``sender`` while older synced rows may still be keyed by
    ``sender_alt`` / union_id.
    """
    table = _get_routing_table()
    if table is None:
        return
    keys = [sender]
    if sender_alt and sender_alt != sender:
        keys.append(sender_alt)
    for key in keys:
        try:
            table.touch_active(key)
        except Exception as exc:
            logger.debug("multitenancy: touch_active failed: %s", exc)


# -- Lazy singletons (RoutingTable + RuntimePool) ----------------------------


_routing_table: Any = None
_routing_db_path: Optional[str] = None
_pool: Any = None
_session_store: Any = None
_session_db_path: Optional[str] = None  # None → DEFAULT_DB_PATH inside SessionStore


def _get_session_store():
    """Lazy-init module-level SessionStore. Returns None if init fails (in-memory only)."""
    global _session_store
    if _session_store is None:
        try:
            from .sessions import SessionStore
            _session_store = SessionStore(_session_db_path)
        except Exception as exc:
            logger.debug("multitenancy: SessionStore init failed (%s)", exc)
            return None
    return _session_store


def override_session_store(store_or_path) -> None:
    """Test helper — inject a SessionStore (or db path string, or None to disable)."""
    global _session_store, _session_db_path, _session_loaded
    if _session_store is not None and _session_store is not store_or_path:
        try:
            _session_store.close()
        except Exception:
            pass
    _session_loaded.clear()
    if store_or_path is None or isinstance(store_or_path, (str, Path)):
        _session_store = None
        _session_db_path = str(store_or_path) if store_or_path is not None else None
    else:
        _session_store = store_or_path
        _session_db_path = None


def _get_routing_table():
    """Lazy-init module-level RoutingTable. Returns None if init fails."""
    global _routing_table
    if _routing_table is None:
        try:
            from .routing import RoutingTable
            _routing_table = RoutingTable(_routing_db_path)
        except Exception as exc:
            logger.debug("multitenancy: RoutingTable init failed (%s)", exc)
            return None
    return _routing_table


def _get_pool():
    """Lazy-init module-level RuntimePool."""
    global _pool
    if _pool is None:
        from .pool import RuntimePool
        _pool = RuntimePool()
    return _pool


def override_routing_table(db_path: Optional[str | Path]) -> None:
    """Test helper — reset the routing-table singleton, optionally pointing it at a different db."""
    global _routing_table, _routing_db_path
    if _routing_table is not None:
        try:
            _routing_table.close()
        except Exception:
            pass
    _routing_table = None
    _routing_db_path = str(db_path) if db_path is not None else None


def override_pool(pool) -> None:
    """Test helper — inject a custom RuntimePool (or None to reset)."""
    global _pool
    _pool = pool


# -- Adapter resolution ------------------------------------------------------


def _get_feishu_adapter(gateway: Any) -> Any:
    """Pull the FeishuAdapter from the gateway, returning None if unavailable."""
    if gateway is None:
        return None
    adapters = getattr(gateway, "adapters", None)
    if adapters is None:
        return None
    try:
        from gateway.platforms.base import Platform  # type: ignore
        result = adapters.get(Platform.FEISHU)
        if result is not None:
            return result
    except Exception as exc:  # pragma: no cover — only triggers when import fails
        logger.debug("multitenancy: Platform import unavailable (%s)", exc)
    if isinstance(adapters, dict):
        return adapters.get("feishu")
    return None


async def _safe_call(fn, *args, **kwargs):
    """Await fn(*args, **kwargs) whether it is sync or async."""
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


# Detect Feishu rate-limit errors. Strings vary across SDK versions, so we
# match on broad hints rather than exact codes.
_RATE_LIMIT_HINTS = ("429", "rate limit", "too many requests", "ratelimit")


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = (str(exc) or "").lower()
    return any(h in msg for h in _RATE_LIMIT_HINTS)


async def _edit_with_retry(adapter, chat_id, message_id, content, *, finalize=False):
    """Wrap adapter.edit_message with exponential backoff on 429.

    On 429: backoff 0.5 → 1.0 → 2.0s, max 3 retries (4 total attempts), then
    log a warning and return None so the caller can continue streaming.
    On non-429: 1 retry after 0.2s, then propagate.
    """
    backoffs = (0.5, 1.0, 2.0)
    for attempt in range(4):
        try:
            if finalize:
                try:
                    return await adapter.edit_message(chat_id, message_id, content, finalize=True)
                except TypeError:
                    return await adapter.edit_message(chat_id, message_id, content)
            return await adapter.edit_message(chat_id, message_id, content)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                if attempt < len(backoffs):
                    logger.debug(
                        "multitenancy: edit_message 429, backoff %ss (attempt %d)",
                        backoffs[attempt], attempt + 1,
                    )
                    await asyncio.sleep(backoffs[attempt])
                    continue
                logger.warning("multitenancy: edit_message rate-limited 4x, giving up")
                return None
            if attempt == 0:
                logger.debug("multitenancy: edit_message non-429 retry: %s", exc)
                await asyncio.sleep(0.2)
                continue
            raise
    return None


def _adapter_supports_streaming_card(adapter) -> bool:
    """Return True when the shared Feishu adapter can drive card streaming."""
    if adapter is None:
        return False
    supports = getattr(adapter, "supports_streaming_card", None)
    if callable(supports):
        try:
            return bool(supports())
        except Exception as exc:
            logger.debug("multitenancy: supports_streaming_card failed: %s", exc)
            return False
    return bool(getattr(adapter, "SUPPORTS_STREAMING_CARD", False))


async def _start_feishu_stream_target(adapter, chat_id) -> tuple[str, Optional[str]]:
    """Start a card stream when possible, otherwise create the text placeholder."""
    if _adapter_supports_streaming_card(adapter):
        starter = getattr(adapter, "start_streaming_card", None)
        updater = getattr(adapter, "update_streaming_card", None)
        if callable(starter) and callable(updater):
            try:
                result = await starter(chat_id=chat_id, reply_to=None, metadata=None)
            except Exception as exc:
                logger.debug("multitenancy: start_streaming_card failed: %s", exc)
            else:
                message_id = getattr(result, "message_id", None)
                if getattr(result, "success", False) and message_id:
                    logger.info("multitenancy: streaming_card started message_id=%s", message_id)
                    return ("card", str(message_id))
                logger.debug(
                    "multitenancy: start_streaming_card unsuccessful: %s",
                    getattr(result, "error", None),
                )

    placeholder_send = await adapter.send(chat_id, "...")
    message_id = (
        placeholder_send.message_id
        if getattr(placeholder_send, "success", False)
        else None
    )
    return ("edit", str(message_id) if message_id else None)


async def _update_feishu_stream_target(
    adapter, chat_id, message_id, content, *, mode: str, finalize: bool = False
):
    """Update the current Feishu streaming surface."""
    if mode == "card":
        result = await adapter.update_streaming_card(
            chat_id=chat_id,
            message_id=message_id,
            content=content,
            finalize=finalize,
        )
        if not getattr(result, "success", False):
            logger.debug(
                "multitenancy: update_streaming_card unsuccessful: %s",
                getattr(result, "error", None),
            )
        return result
    return await _edit_with_retry(
        adapter, chat_id, message_id, content, finalize=finalize
    )


async def _abort_feishu_stream_target(
    adapter, chat_id, message_id, content, *, mode: str
):
    """Force the current streaming surface into an aborted terminal state."""
    if mode == "card":
        aborter = getattr(adapter, "abort_streaming_card", None)
        if callable(aborter):
            result = await aborter(
                chat_id=chat_id,
                message_id=message_id,
                content=content,
            )
            if not getattr(result, "success", False):
                logger.debug(
                    "multitenancy: abort_streaming_card unsuccessful: %s",
                    getattr(result, "error", None),
                )
            return result
    return await _update_feishu_stream_target(
        adapter,
        chat_id,
        message_id,
        content or "Aborted.",
        mode=mode,
        finalize=True,
    )


async def _run_terminal_stream_update(update_coro, *, label: str):
    """Run terminal card/edit update even if the caller is being cancelled."""
    task = asyncio.create_task(update_coro)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            result = await task
            logger.info("multitenancy: %s completed while task was cancelling", label)
            return result
        except Exception as exc:
            logger.debug("multitenancy: %s failed while task was cancelling: %s", label, exc)
        raise


async def _update_feishu_stream_reasoning(
    adapter, chat_id, message_id, content, *, mode: str
):
    """Update reasoning/status text without polluting the final answer block."""
    if mode == "card":
        updater = getattr(adapter, "update_streaming_card_reasoning", None)
        if callable(updater):
            return await updater(
                chat_id=chat_id,
                message_id=message_id,
                content=content,
            )
    return await _update_feishu_stream_target(
        adapter, chat_id, message_id, content, mode=mode
    )


async def _update_feishu_stream_status(
    adapter, chat_id, message_id, content, *, mode: str
):
    """Update an ephemeral in-progress status without retaining it as reasoning."""
    if mode == "card":
        updater = getattr(adapter, "update_streaming_card_status", None)
        if callable(updater):
            return await updater(
                chat_id=chat_id,
                message_id=message_id,
                content=content,
            )
    return await _update_feishu_stream_reasoning(
        adapter, chat_id, message_id, content, mode=mode
    )


async def _update_feishu_stream_tool_event(
    adapter, chat_id, message_id, payload, *, mode: str, completed: bool
):
    """Update active/completed tool state on the streaming surface."""
    payload = payload if isinstance(payload, dict) else {"name": str(payload or "tool")}
    tool_name = str(payload.get("name") or payload.get("tool_name") or "tool")
    if mode == "card":
        method_name = (
            "update_streaming_card_tool_completed"
            if completed
            else "update_streaming_card_tool_started"
        )
        updater = getattr(adapter, method_name, None)
        if callable(updater):
            if completed:
                return await updater(
                    chat_id=chat_id,
                    message_id=message_id,
                    tool_name=tool_name,
                    duration=payload.get("duration"),
                    is_error=bool(payload.get("is_error")),
                )
            return await updater(
                chat_id=chat_id,
                message_id=message_id,
                tool_name=tool_name,
                preview=payload.get("preview"),
                args=payload.get("args"),
            )

    status = (
        f"✅ 工具完成: {tool_name}"
        if completed and not payload.get("is_error")
        else f"⚠️ 工具失败: {tool_name}"
        if completed
        else f"🔧 正在调用工具: {tool_name}"
    )
    return await _update_feishu_stream_target(
        adapter, chat_id, message_id, status, mode=mode
    )


# Throttle edit_message calls. Hermes mainstream uses 1.5s between edits
# (run.py:9502 _PROGRESS_EDIT_INTERVAL); we mirror that as the floor for the
# content phase. CardKit reasoning can update more often because it streams
# into a stable card element; legacy edit_message keeps the wider heartbeat.
_STREAM_CONTENT_MIN_CHARS = 60
_STREAM_CONTENT_MIN_SECONDS = 1.0
_STREAM_THINKING_MIN_SECONDS = 2.0
_STREAM_CARD_REASONING_MIN_CHARS = 40
_STREAM_CARD_REASONING_MIN_SECONDS = 0.8
_STREAM_CARD_PRIME_STATUS = "Hermes 正在准备响应..."
_STREAM_CARD_IDLE_HEARTBEAT_SECONDS = 2.5
_STREAM_ABORT_FALLBACK = "Aborted."
_STREAM_MAX_VISIBLE_CHARS = 3_000
_STREAM_TRUNCATION_SUFFIX = "\n\n...[已截断: 回复过长，已保留前半部分以保证卡片及时完成]"


def _stream_card_idle_status(tick: int) -> str:
    """Return a visibly changing pre-token status for CardKit typewriter keepalive."""
    dots = "." * (3 + (tick % 3))
    return f"Hermes 正在准备响应{dots}"


async def _stream_into_feishu_shared_consumer(
    adapter, chat_id, profile_name, profile_home, event, *, messages: Optional[list[dict]] = None
) -> Optional[str]:
    """Stream Feishu card output through Hermes' shared GatewayStreamConsumer.

    Returns None when the shared card surface cannot be started, allowing the
    caller to fall back to the legacy text-edit transport.
    """
    if GatewayStreamConsumer is None or StreamConsumerConfig is None:
        return None

    import time
    from .agent_real import stream_run_agent, real_run_agent
    from .runtime import _PROFILE_HOME_VAR

    stream_started_at = time.monotonic()
    consumer = GatewayStreamConsumer(
        adapter,
        chat_id,
        StreamConsumerConfig(
            edit_interval=_STREAM_CONTENT_MIN_SECONDS,
            buffer_threshold=_STREAM_CONTENT_MIN_CHARS,
            cursor=" ▉",
        ),
        metadata=None,
    )
    consumer_task: Optional[asyncio.Task] = None
    idle_heartbeat_task: Optional[asyncio.Task] = None
    terminal_update_sent = False
    first_agent_event_seen = False
    content_delta_seen = False
    content = ""
    thinking = ""
    last_reasoning_edit = 0.0
    last_reasoning_len = 0

    async def _idle_card_heartbeat() -> None:
        tick = 1
        while True:
            await asyncio.sleep(_STREAM_CARD_IDLE_HEARTBEAT_SECONDS)
            if first_agent_event_seen or content_delta_seen:
                return
            try:
                await consumer.update_streaming_card_status(_stream_card_idle_status(tick))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("multitenancy: shared card idle heartbeat failed: %s", exc)
            tick += 1

    async def _stop_idle_card_heartbeat() -> None:
        if idle_heartbeat_task is None or idle_heartbeat_task.done():
            return
        idle_heartbeat_task.cancel()
        try:
            await idle_heartbeat_task
        except asyncio.CancelledError:
            pass

    def _abort_content() -> str:
        raw = content if content else (thinking if thinking else _STREAM_ABORT_FALLBACK)
        return _clean_stream_display_text(raw)

    async def _finish_consumer() -> None:
        if consumer_task is None:
            return
        consumer.finish()
        await consumer_task

    try:
        start_task = asyncio.create_task(consumer.ensure_streaming_card_started())
        try:
            started = await asyncio.shield(start_task)
        except asyncio.CancelledError:
            try:
                started = await start_task
            except Exception as exc:
                logger.debug("multitenancy: shared card start failed while cancelling: %s", exc)
                started = False
            if started:
                try:
                    await consumer.abort_streaming_card(_STREAM_ABORT_FALLBACK)
                except Exception as exc:
                    logger.debug("multitenancy: shared card abort-after-start failed: %s", exc)
            raise

        if not started:
            logger.debug("multitenancy: shared card start unavailable; falling back")
            return None

        logger.info(
            "multitenancy: shared stream card ready elapsed=%.3fs",
            time.monotonic() - stream_started_at,
        )
        consumer_task = asyncio.create_task(consumer.run())

        prime_task = asyncio.create_task(
            consumer.update_streaming_card_status(_STREAM_CARD_PRIME_STATUS)
        )
        try:
            await asyncio.shield(prime_task)
        except asyncio.CancelledError:
            try:
                await prime_task
            except Exception as exc:
                logger.debug("multitenancy: shared card prime failed while cancelling: %s", exc)
            try:
                await consumer.abort_streaming_card(_STREAM_ABORT_FALLBACK)
            except Exception as exc:
                logger.debug("multitenancy: shared card abort-after-prime failed: %s", exc)
            raise

        idle_heartbeat_task = asyncio.create_task(_idle_card_heartbeat())

        token = _PROFILE_HOME_VAR.set(profile_home)
        try:
            try:
                async for kind, delta in stream_run_agent(event, profile_home, messages=messages):
                    if not first_agent_event_seen:
                        first_agent_event_seen = True
                        logger.info(
                            "multitenancy: shared stream first agent event kind=%s total=%.3fs",
                            kind,
                            time.monotonic() - stream_started_at,
                        )

                    if kind == "thinking":
                        thinking += str(delta or "")
                        now = time.monotonic()
                        if (
                            not last_reasoning_len
                            or len(thinking) - last_reasoning_len >= _STREAM_CARD_REASONING_MIN_CHARS
                            or now - last_reasoning_edit >= _STREAM_CARD_REASONING_MIN_SECONDS
                        ):
                            await consumer.update_streaming_card_reasoning(thinking)
                            last_reasoning_len = len(thinking)
                            last_reasoning_edit = now
                        continue

                    if kind == "tool_started":
                        payload = delta if isinstance(delta, dict) else {"name": str(delta or "tool")}
                        await consumer.update_streaming_card_tool_started(
                            str(payload.get("name") or payload.get("tool_name") or "tool"),
                            preview=payload.get("preview"),
                            args=payload.get("args"),
                        )
                        continue

                    if kind == "tool_completed":
                        payload = delta if isinstance(delta, dict) else {"name": str(delta or "tool")}
                        await consumer.update_streaming_card_tool_completed(
                            str(payload.get("name") or payload.get("tool_name") or "tool"),
                            duration=payload.get("duration"),
                            is_error=bool(payload.get("is_error")),
                        )
                        continue

                    if kind == "done":
                        continue

                    piece = str(delta or "")
                    if not piece:
                        continue
                    remaining = _STREAM_MAX_VISIBLE_CHARS - len(content)
                    if remaining <= 0:
                        continue
                    if len(piece) > remaining:
                        piece = piece[:remaining] + _STREAM_TRUNCATION_SUFFIX
                        content = content[:_STREAM_MAX_VISIBLE_CHARS] + _STREAM_TRUNCATION_SUFFIX
                        consumer.on_delta(piece)
                        content_delta_seen = True
                        logger.info(
                            "multitenancy: shared stream content truncated max_chars=%s",
                            _STREAM_MAX_VISIBLE_CHARS,
                        )
                        break
                    content += piece
                    consumer.on_delta(piece)
                    content_delta_seen = True
            except Exception as exc:
                logger.info("multitenancy: shared streaming failed (%s) — falling back to non-stream", exc)
                try:
                    content = await real_run_agent(event, profile_home, messages=messages)
                except Exception as fallback_exc:
                    logger.warning("multitenancy: LLM fully unavailable: %s", fallback_exc)
                    content = (
                        "⚠️ 模型暂时不可用 (LLM provider rejected the request).\n"
                        "请检查 profile 的 config.yaml 模型/凭据, 或稍后再试。"
                    )
                if not content_delta_seen:
                    consumer.on_delta(_clean_stream_display_text(content))
                    content_delta_seen = True
        finally:
            _PROFILE_HOME_VAR.reset(token)
            await _stop_idle_card_heartbeat()

        full = content if content else (thinking if thinking else "(empty response)")
        if not content_delta_seen:
            consumer.on_delta(_clean_stream_display_text(full))

        await _finish_consumer()
        terminal_update_sent = True
        return full
    except asyncio.CancelledError:
        await _stop_idle_card_heartbeat()
        if not terminal_update_sent:
            try:
                await _run_terminal_stream_update(
                    consumer.abort_streaming_card(_abort_content()),
                    label="shared stream abort update",
                )
            except asyncio.CancelledError:
                raise
            except Exception as abort_exc:
                logger.debug("multitenancy: shared stream abort update failed: %s", abort_exc)
        if consumer_task is not None:
            consumer.finish()
            try:
                await consumer_task
            except Exception:
                pass
        raise


async def _stream_into_feishu(
    adapter, chat_id, profile_name, profile_home, event, *, messages: Optional[list[dict]] = None
) -> str:
    """Stream LLM tokens into Feishu.

    Uses OpenClaw-style interactive card streaming when the shared Feishu
    adapter supports it. Falls back to the legacy text placeholder +
    ``edit_message`` loop for older adapters.

    Falls back to a single non-streamed send() if streaming returns empty or
    the initial stream target fails. Returns the final concatenated text.

    ``messages`` (optional): full conversation including prior history. When
    omitted the runner builds a single-turn system+user prompt from the event.
    """
    import time
    from .agent_real import stream_run_agent, real_run_agent
    from .runtime import _PROFILE_HOME_VAR

    stream_started_at = time.monotonic()

    # Without an adapter we can still produce text (used in unit tests).
    if adapter is None:
        token = _PROFILE_HOME_VAR.set(profile_home)
        try:
            content_parts: list[str] = []
            async for kind, c in stream_run_agent(event, profile_home, messages=messages):
                if kind == "content":
                    content_parts.append(c)
            return (
                "".join(content_parts)
                or await real_run_agent(event, profile_home, messages=messages)
            )
        finally:
            _PROFILE_HOME_VAR.reset(token)

    if _adapter_supports_streaming_card(adapter):
        shared_response = await _stream_into_feishu_shared_consumer(
            adapter,
            chat_id,
            profile_name,
            profile_home,
            event,
            messages=messages,
        )
        if shared_response is not None:
            return shared_response

    stream_mode = "edit"
    placeholder_id: Optional[str] = None
    target_ready_at = stream_started_at
    thinking = ""
    content = ""
    last_edit_time = 0.0
    last_render_len = 0
    last_reasoning_render_len = 0
    content_started = False
    first_agent_event_seen = False
    terminal_update_sent = False
    card_reasoning_sent = False
    idle_heartbeat_task: Optional[asyncio.Task] = None

    async def _idle_card_heartbeat() -> None:
        tick = 1
        while True:
            await asyncio.sleep(_STREAM_CARD_IDLE_HEARTBEAT_SECONDS)
            if first_agent_event_seen or content_started or placeholder_id is None:
                return
            try:
                await _update_feishu_stream_status(
                    adapter,
                    chat_id,
                    placeholder_id,
                    _stream_card_idle_status(tick),
                    mode=stream_mode,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("multitenancy: card idle heartbeat failed: %s", exc)
            tick += 1

    async def _stop_idle_card_heartbeat() -> None:
        if idle_heartbeat_task is None or idle_heartbeat_task.done():
            return
        idle_heartbeat_task.cancel()
        try:
            await idle_heartbeat_task
        except asyncio.CancelledError:
            pass

    def render() -> str:
        if content:
            return _clean_stream_display_text(content)
        preview = thinking[-160:].strip() if thinking else ""
        return f"💭 思考中…\n{preview}" if preview else "💭 思考中…"

    def abort_content() -> str:
        raw = content if content else (thinking if thinking else _STREAM_ABORT_FALLBACK)
        return _clean_stream_display_text(raw)

    try:
        # Create/send can complete remotely after this task is cancelled. Shield
        # it so we can still obtain the message_id and close the card instead of
        # leaving a Generating card behind.
        start_task = asyncio.create_task(_start_feishu_stream_target(adapter, chat_id))
        try:
            stream_mode, placeholder_id = await asyncio.shield(start_task)
        except asyncio.CancelledError:
            try:
                stream_mode, placeholder_id = await start_task
                logger.info(
                    "multitenancy: stream target start completed while task was cancelling "
                    "mode=%s message_id=%s",
                    stream_mode,
                    placeholder_id,
                )
            except Exception as exc:
                logger.debug("multitenancy: stream target start failed while cancelling: %s", exc)
            raise

        target_ready_at = time.monotonic()
        logger.info(
            "multitenancy: stream target ready mode=%s message_id=%s elapsed=%.3fs",
            stream_mode,
            placeholder_id,
            target_ready_at - stream_started_at,
        )

        if placeholder_id is None:
            # Couldn't get a message to edit — degrade to one-shot non-stream.
            text = await real_run_agent(event, profile_home, messages=messages)
            await adapter.send(chat_id, text)
            return text

        if stream_mode == "card":
            try:
                await _update_feishu_stream_status(
                    adapter,
                    chat_id,
                    placeholder_id,
                    _STREAM_CARD_PRIME_STATUS,
                    mode=stream_mode,
                )
                logger.info(
                    "multitenancy: stream card primed message_id=%s elapsed=%.3fs",
                    placeholder_id,
                    time.monotonic() - target_ready_at,
                )
                idle_heartbeat_task = asyncio.create_task(_idle_card_heartbeat())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("multitenancy: card prime update failed: %s", exc)

        token = _PROFILE_HOME_VAR.set(profile_home)
        try:
            try:
                async for kind, delta in stream_run_agent(event, profile_home, messages=messages):
                    if not first_agent_event_seen:
                        first_agent_event_seen = True
                        logger.info(
                            "multitenancy: stream first agent event kind=%s "
                            "since_target=%.3fs total=%.3fs",
                            kind,
                            time.monotonic() - target_ready_at,
                            time.monotonic() - stream_started_at,
                        )
                    if kind == "thinking":
                        thinking += str(delta or "")
                        if stream_mode == "card" and thinking:
                            now = time.monotonic()
                            should_update_card_reasoning = (
                                not card_reasoning_sent
                                or len(thinking) - last_reasoning_render_len >= _STREAM_CARD_REASONING_MIN_CHARS
                                or now - last_edit_time >= _STREAM_CARD_REASONING_MIN_SECONDS
                            )
                            if should_update_card_reasoning:
                                try:
                                    await _update_feishu_stream_reasoning(
                                        adapter,
                                        chat_id,
                                        placeholder_id,
                                        thinking,
                                        mode=stream_mode,
                                    )
                                    card_reasoning_sent = True
                                    last_reasoning_render_len = len(thinking)
                                except Exception as exc:
                                    logger.debug("multitenancy: card reasoning update failed: %s", exc)
                                last_edit_time = now
                                last_render_len = len(render())
                            continue
                    elif kind == "tool_started":
                        try:
                            await _update_feishu_stream_tool_event(
                                adapter,
                                chat_id,
                                placeholder_id,
                                delta,
                                mode=stream_mode,
                                completed=False,
                            )
                        except Exception as exc:
                            logger.debug("multitenancy: card tool-start update failed: %s", exc)
                        last_edit_time = time.monotonic()
                        last_render_len = len(render())
                        continue
                    elif kind == "tool_completed":
                        try:
                            await _update_feishu_stream_tool_event(
                                adapter,
                                chat_id,
                                placeholder_id,
                                delta,
                                mode=stream_mode,
                                completed=True,
                            )
                        except Exception as exc:
                            logger.debug("multitenancy: card tool-complete update failed: %s", exc)
                        last_edit_time = time.monotonic()
                        last_render_len = len(render())
                        continue
                    elif kind == "done":
                        continue
                    else:
                        content += str(delta or "")
                        if len(content) > _STREAM_MAX_VISIBLE_CHARS:
                            content = (
                                content[:_STREAM_MAX_VISIBLE_CHARS]
                                + _STREAM_TRUNCATION_SUFFIX
                            )
                            logger.info(
                                "multitenancy: stream content truncated and finalized "
                                "message_id=%s max_chars=%s",
                                placeholder_id,
                                _STREAM_MAX_VISIBLE_CHARS,
                            )
                            break
                        if not content_started:
                            # Force an immediate edit on phase transition so the user
                            # sees the answer start the moment reasoning ends.
                            content_started = True
                            try:
                                await _update_feishu_stream_target(
                                    adapter,
                                    chat_id,
                                    placeholder_id,
                                    render(),
                                    mode=stream_mode,
                                )
                            except Exception as exc:
                                logger.debug(
                                    "multitenancy: phase-transition stream update failed: %s",
                                    exc,
                                )
                            last_edit_time = time.monotonic()
                            last_render_len = len(render())
                            continue

                    now = time.monotonic()
                    rendered = render()
                    if content_started:
                        should_edit = (
                            len(rendered) - last_render_len >= _STREAM_CONTENT_MIN_CHARS
                            or now - last_edit_time >= _STREAM_CONTENT_MIN_SECONDS
                        )
                    else:
                        # Reasoning phase — heartbeat-only edits, no char threshold.
                        should_edit = now - last_edit_time >= _STREAM_THINKING_MIN_SECONDS
                    if should_edit:
                        try:
                            await _update_feishu_stream_target(
                                adapter,
                                chat_id,
                                placeholder_id,
                                rendered,
                                mode=stream_mode,
                            )
                        except Exception as exc:
                            logger.debug("multitenancy: stream update mid-stream failed: %s", exc)
                        last_edit_time = now
                        last_render_len = len(rendered)
            except Exception as exc:
                logger.info("multitenancy: streaming failed (%s) — falling back to non-stream", exc)
                try:
                    content = await real_run_agent(event, profile_home, messages=messages)
                except Exception as fallback_exc:
                    # Both stream + non-stream LLM paths failed (e.g. region block,
                    # exhausted credentials). Surface a user-visible error instead
                    # of leaving the "..." placeholder hanging.
                    logger.warning("multitenancy: LLM fully unavailable: %s", fallback_exc)
                    content = (
                        "⚠️ 模型暂时不可用 (LLM provider rejected the request).\n"
                        "请检查 profile 的 config.yaml 模型/凭据, 或稍后再试。"
                    )
        finally:
            _PROFILE_HOME_VAR.reset(token)
            await _stop_idle_card_heartbeat()

        full = content if content else (thinking if thinking else "(empty response)")
        display_full = _clean_stream_display_text(full)

        # 3. Final commit. finalize=True signals end of stream to Feishu.
        try:
            await _run_terminal_stream_update(
                _update_feishu_stream_target(
                    adapter,
                    chat_id,
                    placeholder_id,
                    display_full,
                    mode=stream_mode,
                    finalize=True,
                ),
                label="stream final update",
            )
            terminal_update_sent = True
        except Exception as exc:
            logger.debug("multitenancy: final stream update failed: %s", exc)

        return full
    except asyncio.CancelledError:
        if placeholder_id is not None and not terminal_update_sent:
            full = abort_content()
            logger.info(
                "multitenancy: stream cancelled; aborting target mode=%s message_id=%s content_len=%s",
                stream_mode,
                placeholder_id,
                len(full),
            )
            try:
                await _run_terminal_stream_update(
                    _abort_feishu_stream_target(
                        adapter,
                        chat_id,
                        placeholder_id,
                        full,
                        mode=stream_mode,
                    ),
                    label="stream abort update",
                )
            except asyncio.CancelledError:
                raise
            except Exception as abort_exc:
                logger.debug("multitenancy: stream abort update failed: %s", abort_exc)
        raise


def _log_task_failure(task: asyncio.Task) -> None:
    """Done-callback for fire-and-forget tasks — surfaces silent exceptions."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("multitenancy: background task crashed: %r", exc)
