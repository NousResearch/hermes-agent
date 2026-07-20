"""Gateway agent run path (foreground ``_run_agent_inner`` production body).

Promoted from ``GatewayRunner._run_agent_inner``. The runner keeps a thin
``_run_agent`` / ``_run_agent_inner`` delegate for callers and tests.
"""

from __future__ import annotations

import inspect
import logging
import asyncio
import os
import time
import json
import threading
import copy
import re
import datetime
from typing import Any, Dict, List, Optional

logger_default = logging.getLogger("gateway.run")



def _auto_continue_freshness_window(*args, **kwargs):
    from gateway.run import _auto_continue_freshness_window as _fn
    return _fn(*args, **kwargs)


def _build_gateway_agent_history(*args, **kwargs):
    from gateway.run import _build_gateway_agent_history as _fn
    return _fn(*args, **kwargs)


def _build_media_placeholder(*args, **kwargs):
    from gateway.run import _build_media_placeholder as _fn
    return _fn(*args, **kwargs)


def _build_progress_message_for_surface(*args, **kwargs):
    from gateway.run import _build_progress_message_for_surface as _fn
    return _fn(*args, **kwargs)


def _collect_auto_append_media_tags(*args, **kwargs):
    from gateway.run import _collect_auto_append_media_tags as _fn
    return _fn(*args, **kwargs)


def _collect_history_media_paths(*args, **kwargs):
    from gateway.run import _collect_history_media_paths as _fn
    return _fn(*args, **kwargs)


def _current_max_iterations(*args, **kwargs):
    from gateway.run import _current_max_iterations as _fn
    return _fn(*args, **kwargs)


def _dequeue_pending_event(*args, **kwargs):
    from gateway.run import _dequeue_pending_event as _fn
    return _fn(*args, **kwargs)


def _float_env(*args, **kwargs):
    from gateway.run import _float_env as _fn
    return _fn(*args, **kwargs)


def _format_exec_approval_fallback(*args, **kwargs):
    from gateway.run import _format_exec_approval_fallback as _fn
    return _fn(*args, **kwargs)


def _gateway_platform_value(*args, **kwargs):
    from gateway.run import _gateway_platform_value as _fn
    return _fn(*args, **kwargs)


def _has_platform_display_override(*args, **kwargs):
    from gateway.run import _has_platform_display_override as _fn
    return _fn(*args, **kwargs)


def _is_control_interrupt_message(*args, **kwargs):
    from gateway.run import _is_control_interrupt_message as _fn
    return _fn(*args, **kwargs)


def _is_fresh_gateway_interruption(*args, **kwargs):
    from gateway.run import _is_fresh_gateway_interruption as _fn
    return _fn(*args, **kwargs)


def _last_transcript_timestamp(*args, **kwargs):
    from gateway.run import _last_transcript_timestamp as _fn
    return _fn(*args, **kwargs)


def _load_gateway_config(*args, **kwargs):
    from gateway.run import _load_gateway_config as _fn
    return _fn(*args, **kwargs)


def _message_timestamps_enabled(*args, **kwargs):
    from gateway.run import _message_timestamps_enabled as _fn
    return _fn(*args, **kwargs)


def _new_stream_consumer(*args, **kwargs):
    from gateway.run import _new_stream_consumer as _fn
    return _fn(*args, **kwargs)


def _non_conversational_metadata(*args, **kwargs):
    from gateway.run import _non_conversational_metadata as _fn
    return _fn(*args, **kwargs)


def _normalize_empty_agent_response(*args, **kwargs):
    from gateway.run import _normalize_empty_agent_response as _fn
    return _fn(*args, **kwargs)


def _platform_config_key(*args, **kwargs):
    from gateway.run import _platform_config_key as _fn
    return _fn(*args, **kwargs)


def _prepare_gateway_status_message(*args, **kwargs):
    from gateway.run import _prepare_gateway_status_message as _fn
    return _fn(*args, **kwargs)


def _preserve_queued_followup_history_offset(*args, **kwargs):
    from gateway.run import _preserve_queued_followup_history_offset as _fn
    return _fn(*args, **kwargs)


def _profile_runtime_scope(*args, **kwargs):
    from gateway.run import _profile_runtime_scope as _fn
    return _fn(*args, **kwargs)


def _redact_approval_command(*args, **kwargs):
    from gateway.run import _redact_approval_command as _fn
    return _fn(*args, **kwargs)


def _redact_gateway_user_facing_secrets(*args, **kwargs):
    from gateway.run import _redact_gateway_user_facing_secrets as _fn
    return _fn(*args, **kwargs)


def _resolve_gateway_model(*args, **kwargs):
    from gateway.run import _resolve_gateway_model as _fn
    return _fn(*args, **kwargs)


def _resolve_hermes_browser_cdp_url(*args, **kwargs):
    from gateway.run import _resolve_hermes_browser_cdp_url as _fn
    return _fn(*args, **kwargs)


def _resolve_progress_thread_id(*args, **kwargs):
    from gateway.run import _resolve_progress_thread_id as _fn
    return _fn(*args, **kwargs)


def _sanitize_gateway_final_response(*args, **kwargs):
    from gateway.run import _sanitize_gateway_final_response as _fn
    return _fn(*args, **kwargs)


def _select_cached_agent_history(*args, **kwargs):
    from gateway.run import _select_cached_agent_history as _fn
    return _fn(*args, **kwargs)


async def _send_or_update_status_coro(*args, **kwargs):
    from gateway.run import _send_or_update_status_coro as _fn
    return await _fn(*args, **kwargs)


def _should_surface_tool_progress_detail(*args, **kwargs):
    from gateway.run import _should_surface_tool_progress_detail as _fn
    return _fn(*args, **kwargs)


def _wrap_current_message_with_observed_context(*args, **kwargs):
    from gateway.run import _wrap_current_message_with_observed_context as _fn
    return _fn(*args, **kwargs)


def build_resume_recovery_note(*args, **kwargs):
    from gateway.run import build_resume_recovery_note as _fn
    return _fn(*args, **kwargs)


def render_notice_line(*args, **kwargs):
    from gateway.run import render_notice_line as _fn
    return _fn(*args, **kwargs)


def _join_turn_sidecar_notes(*args, **kwargs):
    from gateway.run import _join_turn_sidecar_notes as _fn
    return _fn(*args, **kwargs)


def _strip_stale_dangerous_confirmations(*args, **kwargs):
    from gateway.run import _strip_stale_dangerous_confirmations as _fn
    return _fn(*args, **kwargs)



def cfg_get(*args, **kwargs):
    from gateway.run import cfg_get as _fn
    return _fn(*args, **kwargs)


def is_truthy_value(*args, **kwargs):
    from gateway.run import is_truthy_value as _fn
    return _fn(*args, **kwargs)


def merge_pending_message_event(*args, **kwargs):
    from gateway.run import merge_pending_message_event as _fn
    return _fn(*args, **kwargs)


def safe_schedule_threadsafe(*args, **kwargs):
    from gateway.run import safe_schedule_threadsafe as _fn
    return _fn(*args, **kwargs)


async def run_gateway_agent_inner(
    *,
    runner: Any,
    message: str,
    context_prompt: str,
    history: List[Dict[str, Any]],
    source: Any,
    session_id: str,
    session_key: str = None,
    run_generation: Optional[int] = None,
    _interrupt_depth: int = 0,
    event_message_id: Optional[str] = None,
    channel_prompt: Optional[str] = None,
    moa_config: Optional[dict] = None,
    persist_user_message: Optional[Any] = None,
    persist_user_timestamp: Optional[float] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """
    Run the agent with the given message and context.

    Returns the full result dict from run_conversation, including:
      - "final_response": str (the text to send back)
      - "messages": list (full conversation including tool calls)
      - "api_calls": int
      - "completed": bool

    This is run in a thread pool to not block the event loop.
    Supports interruption via new messages.
    """
    if logger is None:
        logger = logger_default
    from gateway.run import (
        _AGENT_PENDING_SENTINEL,
        _INTERRUPT_REASON_TIMEOUT,
        BasePlatformAdapter,
        _hermes_home,
    )

    # ---- Proxy mode: delegate to remote API server ----
    if runner._get_proxy_url():
        return await runner._run_agent_via_proxy(
            message=message,
            context_prompt=context_prompt,
            history=history,
            source=source,
            session_id=session_id,
            session_key=session_key,
            run_generation=run_generation,
            event_message_id=event_message_id,
        )

    from run_agent import AIAgent
    import queue

    def _run_still_current() -> bool:
        if run_generation is None or not session_key:
            return True
        return runner._is_session_run_current(session_key, run_generation)

    user_config = _load_gateway_config()
    platform_key = _platform_config_key(source.platform)

    from hermes_cli.tools_config import _get_platform_tools
    enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))
    agent_cfg_local = user_config.get("agent") or {}
    disabled_toolsets = agent_cfg_local.get("disabled_toolsets") or None

    display_config = user_config.get("display", {})
    if not isinstance(display_config, dict):
        display_config = {}

    # Per-platform display settings — resolve via display_config module
    # which checks display.platforms.<platform>.<key> first, then
    # display.<key> global, then built-in platform defaults.
    from gateway.display_config import resolve_display_setting

    # Apply tool preview length config (0 = no limit)
    try:
        from agent.display import set_tool_preview_max_len
        _tpl = resolve_display_setting(user_config, platform_key, "tool_preview_length", 0)
        set_tool_preview_max_len(int(_tpl) if _tpl else 0)
    except Exception:
        pass

    # Apply friendly tool labels config (default on) — per-platform aware
    try:
        from agent.display import set_friendly_tool_labels
        _ftl = resolve_display_setting(user_config, platform_key, "friendly_tool_labels", True)
        set_friendly_tool_labels(bool(_ftl))
    except Exception:
        pass

    # Tool progress mode — resolved per-platform with env var fallback
    _resolved_tp = resolve_display_setting(user_config, platform_key, "tool_progress")
    _env_tp = os.getenv("HERMES_TOOL_PROGRESS_MODE")
    _display_cfg = display_config if isinstance(display_config, dict) else {}
    _platforms_cfg = _display_cfg.get("platforms") or {}
    _platform_cfg = _platforms_cfg.get(platform_key) or {}
    _legacy_tp_overrides = _display_cfg.get("tool_progress_overrides") or {}
    _tool_progress_configured = (
        "tool_progress" in _display_cfg
        or (
            isinstance(_platform_cfg, dict)
            and "tool_progress" in _platform_cfg
        )
        or (
            isinstance(_legacy_tp_overrides, dict)
            and platform_key in _legacy_tp_overrides
        )
    )
    progress_mode = (
        _env_tp
        if _env_tp and not _tool_progress_configured
        else (_resolved_tp or _env_tp or "all")
    )
    # Tool progress grouping: "accumulate" (edit one bubble) or "separate" (one msg per tool)
    progress_grouping = resolve_display_setting(user_config, platform_key, "tool_progress_grouping") or "accumulate"
    from gateway.status_phrases import choose_status_phrase, resolve_status_phrase_catalog
    _generic_status_recent: List[str] = []
    _generic_status_catalog = resolve_status_phrase_catalog(user_config, platform_key)

    def _display_surface_mode(
        setting: str,
        *,
        default: bool = False,
        require_platform_override_for: set[Any] | None = None,
        allow_generic: bool = False,
    ) -> str:
        """Return off|raw|generic for a gateway visibility surface."""
        if require_platform_override_for:
            current_platform = _gateway_platform_value(source.platform)
            platform_only = {
                _gateway_platform_value(item)
                for item in require_platform_override_for
            }
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
                kind,
                tool_name=tool_name,
                preview=preview,
                args=args,
                recent=_generic_status_recent,
                catalog=_generic_status_catalog,
            )
        except Exception as _phrase_err:
            logger.debug("generic status phrase selection failed: %s", _phrase_err)
            return "still on it" if kind in {"heartbeat", "waiting", "long_running", "status"} else "one sec"
    # Disable tool progress for webhooks - they don't support message editing,
    # so each progress line would be sent as a separate message.
    from gateway.config import Platform
    tool_progress_enabled = progress_mode not in {"off", "log"} and source.platform != Platform.WEBHOOK
    # Live working-state status for text-rendering typing indicators
    # (Slack's assistant status line). Independent of tool_progress —
    # Slack defaults tool_progress off (permanent lines spam channels)
    # but the status line is ephemeral, so live status stays useful
    # there. Rendering rides the existing _keep_typing refresh: the
    # callback only stores a phrase on the adapter, costing zero extra
    # platform API calls.
    _live_status_mode = resolve_display_setting(
        user_config, platform_key, "live_status", "full"
    )
    _live_status_adapter = runner._adapter_for_source(source)
    if not getattr(_live_status_adapter, "supports_status_text", False):
        _live_status_adapter = None
    if _live_status_mode == "off":
        _live_status_adapter = None
    # "log" mode: tool calls are written to ~/.hermes/logs/tool_calls.log
    # instead of the chat (#3459 / #3458). Gateway-only by design.
    log_mode_enabled = progress_mode == "log" and source.platform != Platform.WEBHOOK
    log_queue: "queue.Queue | None" = queue.Queue() if log_mode_enabled else None
    # Natural assistant status messages are intentionally independent from
    # tool progress and token streaming. Users can keep tool_progress quiet
    # in chat platforms while opting into concise mid-turn updates.
    interim_assistant_messages_mode = _display_surface_mode(
        "interim_assistant_messages",
        default=True,
        require_platform_override_for={Platform.MATTERMOST},
    )
    interim_assistant_messages_enabled = (
        source.platform != Platform.WEBHOOK
        and interim_assistant_messages_mode != "off"
    )
    # thinking_progress is independent — if enabled, we need the progress
    # queue even when tool_progress is off (thinking relay uses same infra).
    # Mattermost requires a per-platform opt-in: global scratch-text display
    # is too easy to leak into busy public threads.
    _thinking_mode = _display_surface_mode(
        "thinking_progress",
        default=False,
        require_platform_override_for={Platform.MATTERMOST},
    )
    _thinking_enabled = _thinking_mode != "off"
    needs_progress_queue = tool_progress_enabled or _thinking_enabled


    # Queue for progress messages (thread-safe)
    progress_queue = queue.Queue() if needs_progress_queue else None
    last_tool = [None]  # Mutable container for tracking in closure
    last_progress_msg = [None]  # Track last message for dedup
    repeat_count = [0]  # How many times the same message repeated
    # True when the previously enqueued progress line was a terminal
    # fenced code block — consecutive terminal calls then drop the
    # repeated "💻 terminal" header and render back-to-back blocks.
    last_was_terminal_block = [False]

    # ── Discord voice "verbal ack before tool calls" ────────────────
    # When the bot is in a voice channel with the continuous mixer
    # installed (discord.voice_fx.enabled), speak a short phrase ("let me
    # look into that") over the ambient idle bed on the FIRST tool call of
    # the turn.  Fires from tool_start_callback (independent of the
    # tool-progress text gate), at most once per turn.  No-op on every
    # other platform / when not in a voice channel.
    _voice_ack_fired = [False]
    _voice_ack_guild: List[Optional[int]] = [None]
    if source.platform == Platform.DISCORD:
        _va = runner.adapters.get(Platform.DISCORD)
        # source.chat_id is the linked text channel; resolve the guild whose
        # voice connection is bound to it (mirrors DiscordAdapter.play_tts).
        _vtc = getattr(_va, "_voice_text_channels", None)
        if isinstance(_vtc, dict) and hasattr(_va, "voice_mixer_active"):
            for _gid, _tc in _vtc.items():
                if str(_tc) == str(source.chat_id) and _va.voice_mixer_active(_gid):
                    _voice_ack_guild[0] = _gid
                    break
    _voice_ack_loop = asyncio.get_running_loop()

    def voice_ack_callback(call_id, tool_name, args):
        """tool_start_callback: speak a one-time ack in the voice channel."""
        if _voice_ack_fired[0] or _voice_ack_guild[0] is None:
            return
        if not _run_still_current():
            return
        _voice_ack_fired[0] = True
        _adapter = runner.adapters.get(Platform.DISCORD)
        if _adapter is None or not hasattr(_adapter, "play_ack_in_voice"):
            return
        try:
            safe_schedule_threadsafe(
                _adapter.play_ack_in_voice(_voice_ack_guild[0]),
                _voice_ack_loop,
                logger=logger,
                log_message="voice ack scheduling error",
            )
        except Exception as _ack_err:
            logger.debug("voice ack schedule failed: %s", _ack_err)

    # Auto-cleanup of temporary progress bubbles (Telegram + any adapter
    # that implements ``delete_message``). When enabled via
    # ``display.platforms.<platform>.cleanup_progress: true``, message IDs
    # from the tool-progress / "⏳ Working — N min" / status-callback bubbles
    # are collected here and deleted after the final response lands.
    # Failed runs skip cleanup so the bubbles remain as breadcrumbs.
    _cleanup_progress = bool(
        resolve_display_setting(user_config, platform_key, "cleanup_progress")
    )
    _cleanup_adapter = runner._adapter_for_source(source) if _cleanup_progress else None
    if _cleanup_adapter is not None and (
        type(_cleanup_adapter).delete_message is BasePlatformAdapter.delete_message
    ):
        # Adapter doesn't support deletion — silently disable.
        _cleanup_progress = False
        _cleanup_adapter = None
    _cleanup_msg_ids: List[str] = []
    # First-touch onboarding latch: fires at most once per run, even if
    # several tools exceed the threshold.
    long_tool_hint_fired = [False]
    _LONG_TOOL_THRESHOLD_S = 30.0

    def progress_callback(event_type: str, tool_name: str = None, preview: str = None, args: dict = None, **kwargs):
        """Callback invoked by agent on tool lifecycle events."""
        # Live status line (Slack's assistant status): stash the current
        # tool phrase on the adapter; the _keep_typing refresh renders it
        # within a couple of seconds. Handled before every other gate
        # because it's independent of progress bubbles and queues (Slack
        # keeps tool_progress off by default, but the ephemeral status
        # line is always safe). Plain dict write — safe from the agent's
        # sync worker thread, no event-loop hop needed.
        if (
            _live_status_adapter is not None
            and _live_status_mode != "off"
            and tool_name != "_thinking"
        ):
            try:
                if event_type == "tool.started" and tool_name and _run_still_current():
                    from agent.display import build_status_phrase
                    _phrase = build_status_phrase(
                        tool_name,
                        args if _live_status_mode == "full" else None,
                    )
                    _live_status_adapter.set_status_text(source.chat_id, _phrase)
                elif event_type == "tool.completed":
                    # Between tools the model is genuinely "thinking"
                    # again — revert to the static default.
                    _live_status_adapter.set_status_text(source.chat_id, None)
            except Exception as _ls_err:
                logger.debug("live status update failed: %s", _ls_err)
        # "log" mode: append tool.started lines to the log queue and stay
        # silent in chat. Handled before the progress_queue guard because
        # log mode runs without a chat progress queue.
        if log_queue is not None:
            if event_type == "tool.started" and tool_name and tool_name != "_thinking":
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                preview_str = f' "{preview}"' if preview else ""
                log_queue.put(f"{ts}  {tool_name}:{preview_str}".rstrip())
            if not progress_queue:
                return
        if not progress_queue or not _run_still_current():
            return

        # First-touch onboarding: the first time a tool takes longer than
        # _LONG_TOOL_THRESHOLD_S during a run that's streaming every tool
        # (progress_mode == "all"), append a one-time hint suggesting
        # /verbose.  We only fire when (a) the user hasn't seen the hint
        # before and (b) /verbose is actually usable on this platform
        # (gateway gate must be open).  The CLI has its own trigger.
        if event_type == "tool.completed" and not long_tool_hint_fired[0]:
            try:
                duration = kwargs.get("duration") or 0
                if duration >= _LONG_TOOL_THRESHOLD_S and progress_mode == "all":
                    from agent.onboarding import (
                        TOOL_PROGRESS_FLAG,
                        is_seen,
                        mark_seen,
                        tool_progress_hint_gateway,
                    )
                    _cfg = _load_gateway_config()
                    gate_on = is_truthy_value(
                        cfg_get(_cfg, "display", "tool_progress_command"),
                        default=False,
                    )
                    if gate_on and not is_seen(_cfg, TOOL_PROGRESS_FLAG):
                        long_tool_hint_fired[0] = True
                        progress_queue.put(tool_progress_hint_gateway())
                        mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
            except Exception as _hint_err:
                logger.debug("tool-progress onboarding hint failed: %s", _hint_err)
            return

        # "_thinking" is assistant scratch text between tool calls.  It
        # is never ordinary tool progress: only relay it when the platform
        # explicitly opted into thinking_progress.  Handle both legacy
        # callback shapes: ("_thinking", text) and
        # ("reasoning.available", "_thinking", text, ...).
        if event_type == "_thinking" or tool_name == "_thinking":
            if not _thinking_enabled:
                return
            thinking_text = preview if tool_name == "_thinking" else tool_name
            msg = f"💬 {thinking_text}" if thinking_text else None
            if msg:
                progress_queue.put(msg)
            return

        # If tool_progress is off, only _thinking passes through (above).
        # Regular tool calls are suppressed.
        if not tool_progress_enabled:
            return

        # Only act on tool.started events (ignore tool.completed, reasoning.available, etc.)
        if event_type not in {"tool.started",}:
            return

        # Suppress tool-progress bubbles once the user has sent `stop`.
        # When the LLM response carries N parallel tool calls, the agent
        # fires N "tool.started" events back-to-back before checking for
        # interrupts — without this guard, a late `stop` still renders
        # all N as 🔍 bubbles, making the interrupt feel ignored.
        # (agent lives in run_sync's scope; agent_holder[0] is the shared
        # handle across nested scopes — see line ~9607.)
        try:
            _agent_for_interrupt = agent_holder[0] if agent_holder else None
            if _agent_for_interrupt is not None and getattr(
                _agent_for_interrupt, "is_interrupted", False
            ):
                return
        except Exception:
            pass

        # "new" mode: only report when tool changes
        if progress_mode == "new" and tool_name == last_tool[0]:
            return
        last_tool[0] = tool_name

        # Build progress message with primary argument preview
        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(tool_name, default="⚙️")

        # Markdown-capable platforms render a terminal command as a fenced
        # code block instead of the compact `terminal: "cmd…"` preview.
        # Gated on the adapter's ``supports_code_blocks`` capability so
        # plain-text platforms keep the short line.  No language tag is
        # emitted — Slack mrkdwn renders the tag as a literal first code
        # line ("bash"), and a bare fence renders correctly everywhere
        # that supports blocks.
        #
        # Verbose mode shows the FULL command.  Non-verbose ("all"/"new")
        # modes still wrap in a fence but truncate to a single line capped
        # at ``tool_preview_length`` (default 40) so a long or multi-line
        # command doesn't render as a huge block — matching the budget the
        # non-terminal preview path already applies (#42634).
        _code_block_full = None
        _code_block_short = None
        try:
            _progress_adapter = runner._adapter_for_source(source)
        except Exception:
            _progress_adapter = None
        if (
            getattr(_progress_adapter, "supports_code_blocks", False)
            and tool_name == "terminal"
            and isinstance(args, dict)
            and isinstance(args.get("command"), str)
            and args["command"].strip()
        ):
            from agent.display import get_tool_preview_max_len
            _cmd_full = args["command"].rstrip()
            # Consecutive terminal calls: drop the repeated
            # "💻 terminal" header so back-to-back commands render as
            # adjacent code blocks under a single header.
            _block_header = (
                "" if last_was_terminal_block[0] else f"{emoji} {tool_name}\n"
            )
            _code_block_full = f"{_block_header}```\n{_cmd_full}\n```"
            # Single-line, capped preview for non-verbose modes.
            _pl = get_tool_preview_max_len()
            _cap = _pl if _pl > 0 else 40
            _lines = _cmd_full.splitlines()
            _cmd_short = _lines[0] if _lines else _cmd_full
            _multiline = len(_lines) > 1
            if len(_cmd_short) > _cap:
                _cmd_short = _cmd_short[:_cap - 3] + "..."
            elif _multiline:
                _cmd_short = _cmd_short + " ..."
            _code_block_short = f"{_block_header}```\n{_cmd_short}\n```"

        # Verbose mode: show detailed arguments, respects tool_preview_length
        if progress_mode == "verbose":
            if _code_block_full is not None:
                last_was_terminal_block[0] = True
                progress_queue.put(_code_block_full)
                return
            last_was_terminal_block[0] = False
            if args:
                from agent.display import get_tool_preview_max_len
                _pl = get_tool_preview_max_len()
                args_str = json.dumps(args, ensure_ascii=False, default=str)
                # When tool_preview_length is 0 (default), don't truncate
                # in verbose mode — the user explicitly asked for full
                # detail.  Platform message-length limits handle the rest.
                if _pl > 0 and len(args_str) > _pl:
                    args_str = args_str[:_pl - 3] + "..."
                msg = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
            elif preview:
                msg = f"{emoji} {tool_name}: \"{preview}\""
            else:
                msg = f"{emoji} {tool_name}..."
            progress_queue.put(msg)
            return

        # "all" / "new" modes: short preview, respects tool_preview_length
        # config (defaults to 40 chars when unset to keep gateway messages
        # compact — unlike CLI spinners, these persist as permanent messages).
        # Terminal commands on markdown platforms get a single-line capped
        # fenced block (built above) instead of the truncated preview.
        if _code_block_short is not None:
            msg = _code_block_short
            last_was_terminal_block[0] = True
        elif preview:
            from agent.display import (
                get_tool_preview_max_len,
                get_tool_verb,
                tool_verb_connector,
                verb_drops_preview,
            )
            _pl = get_tool_preview_max_len()
            _cap = _pl if _pl > 0 else 40
            if len(preview) > _cap:
                preview = preview[:_cap - 3] + "..."
            # Friendly labels: render a human-phrased line for built-in
            # tools ("🔍 Searching the web for ...") by prefixing the verb
            # onto the preview the callback already computed (so the
            # command/url/query is preserved).  Custom/plugin/MCP tools
            # have no verb and fall back to the raw "tool_name: ..." form.
            _verb = get_tool_verb(tool_name)
            if _verb:
                if verb_drops_preview(tool_name):
                    msg = f"{emoji} {_verb}"
                else:
                    msg = f"{emoji} {_verb}{tool_verb_connector(tool_name)}{preview}"
            else:
                msg = f"{emoji} {tool_name}: \"{preview}\""
            last_was_terminal_block[0] = False
        else:
            msg = f"{emoji} {tool_name}..."
            last_was_terminal_block[0] = False

        # Dedup: collapse consecutive identical progress messages.
        # Common with execute_code where models iterate with the same
        # code (same boilerplate imports → identical previews).
        if msg == last_progress_msg[0]:
            repeat_count[0] += 1
            # Update the last line in progress_lines with a counter
            # via a special "dedup" queue message.
            progress_queue.put(("__dedup__", msg, repeat_count[0]))
            return
        last_progress_msg[0] = msg
        repeat_count[0] = 0

        progress_queue.put(msg)

    # Background task to send progress messages
    # Accumulates tool lines into a single message that gets edited.
    #
    # Threading metadata is platform-specific:
    # - Slack DM threading needs event_message_id fallback (reply thread)
    # - Telegram forum topics use message_thread_id; Hermes-created private
    #   DM topic lanes require both thread metadata and a reply anchor
    # - Feishu only honors reply_in_thread when sending a reply, so topic
    #   progress uses the triggering event message as the reply target
    # - Other platforms should use explicit source.thread_id only
    _progress_thread_id = _resolve_progress_thread_id(
        source.platform, source.thread_id, event_message_id,
    )
    _progress_metadata = (
        runner._thread_metadata_for_source(source, event_message_id)
        if _progress_thread_id == source.thread_id
        else {"thread_id": _progress_thread_id}
    ) if _progress_thread_id else None
    _progress_metadata = _non_conversational_metadata(_progress_metadata, platform=source.platform)
    _progress_reply_to = (
        event_message_id
        if source.platform in (Platform.FEISHU, Platform.MATTERMOST) and source.thread_id and event_message_id
        else None
    )

    async def write_tool_log():
        """Drain log_queue and append tool-call lines to tool_calls.log.

        Only active when ``display.tool_progress`` is ``log``. Uses a
        RotatingFileHandler (5MB × 3 backups) so the audit log can't grow
        unbounded, and the shared RedactingFormatter so secrets never land
        on disk.
        """
        if log_queue is None:
            return
        from logging.handlers import RotatingFileHandler

        from agent.redact import RedactingFormatter

        log_dir = _hermes_home / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "tool_calls.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
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
            # Drain remaining entries before closing so late tool calls
            # from the final iteration aren't lost.
            while True:
                try:
                    tool_logger.info("%s", log_queue.get_nowait())
                except queue.Empty:
                    break
                except Exception:
                    break
            tool_logger.removeHandler(file_handler)
            try:
                file_handler.flush()
                file_handler.close()
            except Exception:
                pass

    async def send_progress_messages():
        if not progress_queue:
            return

        adapter = runner._adapter_for_source(source)
        if not adapter:
            return

        # Skip tool progress for platforms that don't support message
        # editing (e.g. iMessage/BlueBubbles) — each progress update
        # would become a separate message bubble, which is noisy.
        if type(adapter).edit_message is BasePlatformAdapter.edit_message:
            while not progress_queue.empty():
                try:
                    progress_queue.get_nowait()
                except Exception:
                    break
            return

        progress_lines = []      # Accumulated tool lines for the CURRENT editable bubble
        progress_msg_id = None   # ID of the current progress message to edit
        can_edit = progress_grouping != "separate"  # "separate" = one message per tool (pre-v0.9 behavior)
        _last_edit_ts = 0.0      # Throttle edits to avoid Telegram flood control
        _PROGRESS_EDIT_INTERVAL = 1.5  # Minimum seconds between edits

        _progress_len_fn = (
            adapter.message_len_fn
            if isinstance(adapter, BasePlatformAdapter)
            else len
        )
        try:
            _raw_progress_limit = int(getattr(adapter, "MAX_MESSAGE_LENGTH", 4000) or 4000)
        except Exception:
            _raw_progress_limit = 4000
        # Leave a little room for platform quirks / formatting.  For tiny
        # test adapters keep the limit usable instead of clamping to 500+.
        _PROGRESS_TEXT_LIMIT = max(
            1,
            _raw_progress_limit - (64 if _raw_progress_limit > 128 else 0),
        )

        # Detect whether the adapter's edit_message accepts metadata so
        # overflow edits preserve Telegram topic/thread routing (#27487).
        _edit_accepts_metadata = False
        if _progress_metadata:
            try:
                _edit_params = inspect.signature(adapter.edit_message).parameters
                _edit_accepts_metadata = (
                    "metadata" in _edit_params
                    or any(
                        param.kind is inspect.Parameter.VAR_KEYWORD
                        for param in _edit_params.values()
                    )
                )
            except (TypeError, ValueError):
                _edit_accepts_metadata = False

        async def _edit_progress_message(message_id: str, content: str):
            kwargs = {
                "chat_id": source.chat_id,
                "message_id": message_id,
                "content": content,
            }
            if getattr(adapter, "REQUIRES_EDIT_FINALIZE", False):
                kwargs["finalize"] = True
            if _edit_accepts_metadata:
                kwargs["metadata"] = _progress_metadata
            return await adapter.edit_message(**kwargs)

        def _progress_text(lines: list) -> str:
            return "\n".join(str(line) for line in lines)

        def _split_progress_groups(lines: list) -> list[list]:
            """Partition progress lines into platform-sized editable bubbles."""
            groups: list[list] = []
            current: list = []
            for line in lines:
                candidate = current + [line]
                if current and _progress_len_fn(_progress_text(candidate)) > _PROGRESS_TEXT_LIMIT:
                    groups.append(current)
                    current = [line]
                else:
                    current = candidate
            if current:
                groups.append(current)
            return groups

        def _track_progress_result(result) -> None:
            if (
                _cleanup_progress
                and getattr(result, "success", False)
                and getattr(result, "message_id", None)
            ):
                _cleanup_msg_ids.append(str(result.message_id))

        async def _send_progress_text(text: str):
            result = await adapter.send(
                chat_id=source.chat_id,
                content=text,
                reply_to=_progress_reply_to,
                metadata=_progress_metadata,
            )
            _track_progress_result(result)
            return result

        async def _roll_progress_overflow_if_needed() -> bool:
            """Start fresh editable progress bubbles before a bubble exceeds limit.

            Returns True when it delivered/split the current buffer and the
            caller should skip the normal send/edit path for this tick.
            """
            nonlocal progress_msg_id, progress_lines, can_edit
            if not progress_lines or not can_edit:
                return False
            groups = _split_progress_groups(progress_lines)
            if len(groups) <= 1:
                return False

            first_text = _progress_text(groups[0])
            if progress_msg_id is not None:
                result = await _edit_progress_message(progress_msg_id, first_text)
                if not result.success:
                    can_edit = False
                    # Fall back to the existing non-edit behavior below.
                    return False
            else:
                result = await _send_progress_text(first_text)
                if result.success and result.message_id:
                    progress_msg_id = result.message_id

            for group in groups[1:]:
                result = await _send_progress_text(_progress_text(group))
                if result.success and result.message_id:
                    progress_msg_id = result.message_id

            # The newest continuation is now the only mutable bubble.  Keep
            # just its lines so subsequent edits update it instead of
            # replaying the full historical transcript into new messages.
            progress_lines = groups[-1]
            return True

        while True:
            try:
                if not _run_still_current():
                    while not progress_queue.empty():
                        try:
                            progress_queue.get_nowait()
                        except Exception:
                            break
                    return

                raw = progress_queue.get_nowait()

                # Drain silently when interrupted: events queued in the
                # window between tool parse and interrupt processing
                # should not render as bubbles.  The "⚡ Interrupting
                # current task" message is sent separately and is the
                # last progress-flavored bubble the user should see.
                try:
                    _agent_for_interrupt = agent_holder[0] if agent_holder else None
                    if _agent_for_interrupt is not None and getattr(
                        _agent_for_interrupt, "is_interrupted", False
                    ):
                        # Drop this event and continue draining.
                        await asyncio.sleep(0)
                        continue
                except Exception:
                    pass

                # Handle dedup messages: update last line with repeat counter
                if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                    _, base_msg, count = raw
                    if progress_lines:
                        progress_lines[-1] = f"{base_msg} (×{count + 1})"
                    msg = progress_lines[-1] if progress_lines else base_msg
                elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                    # Content bubble just landed on the platform — close off
                    # the current tool-progress bubble so the next tool
                    # starts a fresh bubble below the content. Without this,
                    # tool lines keep editing the ORIGINAL progress message
                    # above the new content, making the chat appear out of
                    # order. Mirrors GatewayStreamConsumer.on_segment_break
                    # on the content side. (Issue: tool + content
                    # linearization regression after PR #7885.)
                    progress_msg_id = None
                    progress_lines = []
                    last_progress_msg[0] = None
                    repeat_count[0] = 0
                    continue
                else:
                    msg = raw
                    progress_lines.append(msg)

                if await _roll_progress_overflow_if_needed():
                    _last_edit_ts = time.monotonic()
                    await asyncio.sleep(0.3)
                    if _run_still_current():
                        await adapter.send_typing(source.chat_id, metadata=_progress_metadata)
                    continue

                # Throttle edits: batch rapid tool updates into fewer
                # API calls to avoid hitting Telegram flood control.
                # (grammY auto-retry pattern: proactively rate-limit
                # instead of reacting to 429s.)
                _now = time.monotonic()
                _remaining = _PROGRESS_EDIT_INTERVAL - (_now - _last_edit_ts)
                if _remaining > 0:
                    # Wait out the throttle interval, then loop back to
                    # drain any additional queued messages before sending
                    # a single batched edit.
                    await asyncio.sleep(_remaining)
                    continue

                if not _run_still_current():
                    return

                if can_edit and progress_msg_id is not None:
                    # Try to edit the existing progress message
                    full_text = "\n".join(progress_lines)
                    result = await _edit_progress_message(progress_msg_id, full_text)
                    if not result.success:
                        _err = (getattr(result, "error", "") or "").lower()
                        # Transient network errors (ConnectError, timeouts)
                        # must not permanently disable progress-message
                        # editing — the next cycle can catch up.  Only
                        # permanent failures (flood control, message not
                        # found, permissions) should set can_edit = False.
                        if getattr(result, "retryable", False):
                            logger.debug(
                                "[%s] Transient edit failure — keeping can_edit=True",
                                adapter.name,
                            )
                            continue
                        if "flood" in _err or "retry after" in _err:
                            # Flood control hit — backoff but keep editing.
                            # Only disable edits for non-recoverable errors.
                            logger.info(
                                "[%s] Progress edit flood control, backing off",
                                adapter.name,
                            )
                            _last_edit_ts = time.monotonic()
                        else:
                            can_edit = False
                        _flood_result = await adapter.send(
                            chat_id=source.chat_id,
                            content=msg,
                            reply_to=_progress_reply_to,
                            metadata=_progress_metadata,
                        )
                        if (
                            _cleanup_progress
                            and getattr(_flood_result, "success", False)
                            and getattr(_flood_result, "message_id", None)
                        ):
                            _cleanup_msg_ids.append(str(_flood_result.message_id))
                else:
                    if can_edit:
                        # First tool: send all accumulated text as new message
                        full_text = "\n".join(progress_lines)
                        result = await adapter.send(
                            chat_id=source.chat_id,
                            content=full_text,
                            reply_to=_progress_reply_to,
                            metadata=_progress_metadata,
                        )
                    else:
                        # Editing unsupported: send just this line
                        result = await adapter.send(
                            chat_id=source.chat_id,
                            content=msg,
                            reply_to=_progress_reply_to,
                            metadata=_progress_metadata,
                        )
                    if result.success and result.message_id:
                        progress_msg_id = result.message_id
                        if _cleanup_progress:
                            _cleanup_msg_ids.append(str(result.message_id))

                _last_edit_ts = time.monotonic()

                # Restore typing indicator
                await asyncio.sleep(0.3)
                if _run_still_current():
                    await adapter.send_typing(source.chat_id, metadata=_progress_metadata)

            except queue.Empty:
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                # Drain remaining queued messages
                while not progress_queue.empty():
                    try:
                        raw = progress_queue.get_nowait()
                        if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                            _, base_msg, count = raw
                            if progress_lines:
                                progress_lines[-1] = f"{base_msg} (×{count + 1})"
                                await _roll_progress_overflow_if_needed()
                        elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                            # Content-bubble marker during drain: close off
                            # the current progress bubble and start a fresh
                            # one for any tool lines that arrived after.
                            await _roll_progress_overflow_if_needed()
                            if can_edit and progress_lines and progress_msg_id:
                                _pending_text = _progress_text(progress_lines)
                                try:
                                    await _edit_progress_message(progress_msg_id, _pending_text)
                                except Exception:
                                    pass
                            progress_msg_id = None
                            progress_lines = []
                            last_progress_msg[0] = None
                            repeat_count[0] = 0
                        else:
                            progress_lines.append(raw)
                            await _roll_progress_overflow_if_needed()
                    except Exception:
                        break
                # Final edit with all remaining tools (only if editing works)
                if can_edit and progress_lines and progress_msg_id:
                    await _roll_progress_overflow_if_needed()
                if can_edit and progress_lines and progress_msg_id:
                    full_text = _progress_text(progress_lines)
                    try:
                        await _edit_progress_message(progress_msg_id, full_text)
                    except Exception:
                        pass
                return
            except Exception as e:
                logger.error("Progress message error: %s", e)
                await asyncio.sleep(1)

    # We need to share the agent instance for interrupt support
    agent_holder = [None]  # Mutable container for the agent instance
    result_holder = [None]  # Mutable container for the result
    tools_holder = [None]   # Mutable container for the tool definitions
    stream_consumer_holder = [None]  # Mutable container for stream consumer

    # Bridge sync step_callback → async hooks.emit for agent:step events
    _loop_for_step = asyncio.get_running_loop()
    _hooks_ref = runner.hooks

    def _step_callback_sync(iteration: int, prev_tools: list) -> None:
        if not _run_still_current():
            return
        # prev_tools may be list[str] or list[dict] with "name"/"result"
        # keys.  Normalise to keep "tool_names" backward-compatible for
        # user-authored hooks that do ', '.join(tool_names)'.
        _names: list[str] = []
        for _t in (prev_tools or []):
            if isinstance(_t, dict):
                _names.append(_t.get("name") or "")
            else:
                _names.append(str(_t))
        safe_schedule_threadsafe(
            _hooks_ref.emit("agent:step", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_id,
                "iteration": iteration,
                "tool_names": _names,
                "tools": prev_tools,
            }),
            _loop_for_step,
            logger=logger,
            log_message="agent:step hook scheduling error",
        )

    # Bridge sync event_callback → async hooks.emit for lifecycle events
    # (e.g. session:compress fires after context compression splits a session)
    def _event_callback_sync(event_type: str, context: dict) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                _hooks_ref.emit(event_type, context),
                _loop_for_step,
            )
        except Exception as _e:
            logger.debug("event_callback hook error: %s", _e)

    # Bridge sync status_callback → async adapter.send for context pressure
    _status_adapter = runner._adapter_for_source(source)
    _status_chat_id = source.chat_id
    if source.platform == Platform.FEISHU and source.thread_id and event_message_id:
        # Feishu topics only keep messages inside the topic when they are
        # sent via the reply API with reply_in_thread=true. Status/interim,
        # approval, and stream-consumer paths usually only receive metadata,
        # so carry the triggering message id as a Feishu-specific fallback.
        _status_thread_metadata: Optional[Dict[str, Any]] = {
            "thread_id": _progress_thread_id,
            "reply_to_message_id": event_message_id,
        }
    else:
        _status_thread_metadata = runner._thread_metadata_for_source(source, event_message_id) if _progress_thread_id else None

    def _status_callback_sync(event_type: str, message: str) -> None:
        if not _status_adapter or not _run_still_current():
            return
        prepared_message = _prepare_gateway_status_message(
            source.platform,
            event_type,
            message,
        )
        if prepared_message is None:
            logger.debug(
                "status_callback suppressed for %s/%s: %s",
                source.platform.value if source.platform else "unknown",
                event_type,
                _redact_gateway_user_facing_secrets(str(message or ""))[:160],
            )
            return
        _fut = safe_schedule_threadsafe(
            _send_or_update_status_coro(_status_adapter, _status_chat_id, event_type, prepared_message, _status_thread_metadata),
            _loop_for_step,
            logger=logger,
            log_message=f"status_callback ({event_type}) scheduling error",
        )
        if _fut is None:
            return
        if _cleanup_progress:
            def _track_status_id(fut) -> None:
                try:
                    res = fut.result()
                except Exception:
                    return
                mid = getattr(res, "message_id", None)
                if getattr(res, "success", False) and mid:
                    _cleanup_msg_ids.append(str(mid))
            _fut.add_done_callback(_track_status_id)

    def run_sync():
        # The conditional re-assignment of `message` further below
        # (prepending model-switch notes) makes Python treat it as a
        # local variable in the entire function.  `nonlocal` lets us
        # read *and* reassign the outer `_run_agent` parameter without
        # triggering an UnboundLocalError on the earlier read at
        # `_resolve_turn_agent_config(message, …)`.
        nonlocal message

        # session_key is propagated via contextvars in _set_session_env()
        # (_SESSION_KEY) and via set_current_session_key() (_approval_session_key)
        # below — both concurrency-safe and inherited by tool worker threads.
        # We deliberately do NOT write os.environ["HERMES_SESSION_KEY"] here:
        # os.environ is process-global, so concurrent gateway sessions (e.g.
        # two Discord threads) would clobber each other's value, and a tool
        # thread whose contextvar is unset would fall back to os.environ and
        # read the wrong session key — misrouting command-approval prompts to
        # the wrong thread (#24100). The non-gateway surfaces don't depend on
        # this write: CLI and cron bind the session via contextvars
        # (set_current_session_key / session context), and only the TUI
        # slash-worker *subprocess* exports HERMES_SESSION_KEY (from its own
        # --session-key argv, a separate process) — so removing this in-process
        # gateway write does not affect any of them.

        # Map platform enum to the platform hint key the agent understands.
        # Platform.LOCAL ("local") maps to "cli"; others pass through as-is.
        platform_key = "cli" if source.platform == Platform.LOCAL else source.platform.value

        # Combine platform context, YAML channel_prompts hint for this chat,
        # channel_overrides system_prompt (or global ephemeral), and gateway
        # ephemeral prompt from _get_system_prompt_for_channel.
        combined_ephemeral = context_prompt or ""
        event_channel_prompt = (channel_prompt or "").strip()
        if event_channel_prompt:
            combined_ephemeral = (combined_ephemeral + "\n\n" + event_channel_prompt).strip()
        cfg_channel_prompt = runner._get_system_prompt_for_channel(
            source.platform,
            source.chat_id or "",
            thread_id=getattr(source, "thread_id", None),
            parent_id=getattr(source, "parent_chat_id", None),
        )
        if cfg_channel_prompt:
            combined_ephemeral = (combined_ephemeral + "\n\n" + cfg_channel_prompt).strip()

        max_iterations = _current_max_iterations()

        try:
            model, runtime_kwargs = runner._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
                user_config=user_config,
            )
            logger.debug(
                "run_agent resolved: model=%s provider=%s session=%s",
                model, runtime_kwargs.get("provider"), session_key or "",
            )
        except Exception as exc:
            return {
                "final_response": f"⚠️ Provider authentication failed: {exc}",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        pr = runner._provider_routing
        reasoning_config = runner._resolve_session_reasoning_config(
            source=source,
            session_key=session_key,
            model=model,
        )
        runner._reasoning_config = reasoning_config
        runner._service_tier = runner._load_service_tier()
        # Set up stream consumer for token streaming or interim commentary.
        _stream_consumer = None
        _stream_delta_cb = None
        _scfg = getattr(getattr(runner, 'config', None), 'streaming', None)
        if _scfg is None:
            from gateway.config import StreamingConfig
            _scfg = StreamingConfig()

        # Per-platform streaming gate: display.platforms.<plat>.streaming
        # can disable streaming for specific platforms even when the global
        # streaming config is enabled.
        _plat_streaming = resolve_display_setting(
            user_config, platform_key, "streaming"
        )
        # None = no per-platform override → follow global config
        _streaming_enabled = (
            _scfg.enabled and _scfg.transport != "off"
            if _plat_streaming is None
            else bool(_plat_streaming)
        )
        _want_stream_deltas = _streaming_enabled
        _want_interim_messages = interim_assistant_messages_enabled
        _want_interim_consumer = _want_interim_messages
        if _want_stream_deltas or _want_interim_consumer:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
                _adapter = runner._adapter_for_source(source)
                if _adapter:
                    _pause_typing_before_finalize = None
                    if source.platform == Platform.TELEGRAM and hasattr(_adapter, "pause_typing_for_chat"):
                        def _pause_typing_before_finalize(
                            _adapter=_adapter,
                            _chat_id=source.chat_id,
                        ) -> None:
                            _adapter.pause_typing_for_chat(_chat_id)
                    # Platforms that don't support editing sent messages
                    # (e.g. QQ, WeChat) should skip streaming entirely —
                    # without edit support, the consumer sends a partial
                    # first message that can never be updated, resulting in
                    # duplicate messages (partial + final).
                    _adapter_supports_edit = getattr(_adapter, "SUPPORTS_MESSAGE_EDITING", True)
                    if not _adapter_supports_edit:
                        raise RuntimeError("skip streaming for non-editable platform")
                    _effective_cursor = _scfg.cursor
                    # Some Matrix clients render the streaming cursor
                    # as a visible tofu/white-box artifact.  Keep
                    # streaming text on Matrix, but suppress the cursor.
                    _buffer_only = False
                    if source.platform == Platform.MATRIX:
                        _effective_cursor = ""
                        _buffer_only = True
                    # Fresh-final applies to Telegram only — other
                    # platforms either edit in place cheaply or don't
                    # have the edit-timestamp-stays-stale problem.
                    # (Ported from openclaw/openclaw#72038.)
                    _fresh_final_secs = (
                        float(getattr(_scfg, "fresh_final_after_seconds", 0.0) or 0.0)
                        if source.platform == Platform.TELEGRAM
                        else 0.0
                    )
                    _consumer_cfg = StreamConsumerConfig(
                        edit_interval=_scfg.edit_interval,
                        buffer_threshold=_scfg.buffer_threshold,
                        cursor=_effective_cursor,
                        buffer_only=_buffer_only,
                        fresh_final_after_seconds=_fresh_final_secs,
                        transport=_scfg.transport or "edit",
                        chat_type=getattr(source, "chat_type", "") or "",
                    )
                    _stream_consumer = GatewayStreamConsumer(
                        adapter=_adapter,
                        chat_id=source.chat_id,
                        config=_consumer_cfg,
                        metadata=_status_thread_metadata,
                        on_new_message=(
                            (lambda: progress_queue.put(("__reset__",)))
                            if progress_queue is not None
                            else None
                        ),
                        on_before_finalize=_pause_typing_before_finalize,
                        initial_reply_to_id=event_message_id,
                        run_still_current=_run_still_current,
                    )
                    if _want_stream_deltas:
                        def _stream_delta_cb(text: str) -> None:
                            if _run_still_current():
                                _stream_consumer.on_delta(text)
                    stream_consumer_holder[0] = _stream_consumer
            except Exception as _sc_err:
                logger.debug("Could not set up stream consumer: %s", _sc_err)

        def _interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
            if not _run_still_current():
                return
            display_text = text
            if _stream_consumer is not None:
                if already_streamed:
                    _stream_consumer.on_segment_break()
                else:
                    _stream_consumer.on_commentary(display_text)
                return
            if already_streamed or not _status_adapter or not str(display_text or "").strip():
                return
            safe_schedule_threadsafe(
                _status_adapter.send(
                    _status_chat_id,
                    display_text,
                    metadata=_status_thread_metadata,
                ),
                _loop_for_step,
                logger=logger,
                log_message="interim_assistant_callback scheduling error",
            )

        turn_route = runner._resolve_turn_agent_config(message, model, runtime_kwargs)

        # Check agent cache — reuse the AIAgent from the previous message
        # in this session to preserve the frozen system prompt and tool
        # schemas for prompt cache hits.
        _sig = runner._agent_config_signature(
            turn_route["model"],
            turn_route["runtime"],
            enabled_toolsets,
            combined_ephemeral,
            cache_keys=runner._extract_cache_busting_config(user_config),
            user_id=getattr(source, "user_id", None),
            user_id_alt=getattr(source, "user_id_alt", None),
        )
        agent = None
        reused_cached_agent = False
        _cache_lock = getattr(runner, "_agent_cache_lock", None)
        _cache = getattr(runner, "_agent_cache", None)

        # Peek at the cached entry's snapshot session_id (if any) so we can
        # check, OUTSIDE the cache lock, whether THAT session_id is a DEAD
        # session in state.db. This closes a gap in the #54947 fix: that
        # fix treats "cached session_id != current session_id" as an
        # intentional /resume-style switch and reuses the agent unchanged.
        # But the #54878 self-heal produces the exact same tuple shape
        # when it recovers a routing key away from a session that was
        # already ended — the cached AIAgent still belongs to the DEAD
        # session, not a valid sibling conversation. Reusing it lets that
        # turn's post-run "session split" sync write the routing key
        # straight back onto the dead session_id, undoing the self-heal
        # and looping every message until an interrupt happens to race in
        # first (the #54878 x #54947 interaction — no existing upstream
        # issue tracks this combination as of 2026-07-12).
        _peek_cached_sid = None
        if _cache_lock and _cache is not None:
            with _cache_lock:
                _peek_entry = _cache.get(session_key)
            if _peek_entry and len(_peek_entry) > 3:
                _peek_cached_sid = _peek_entry[3]
        _cached_sid_is_dead = False
        if (
            _peek_cached_sid is not None
            and session_id is not None
            and _peek_cached_sid != session_id
        ):
            try:
                _cached_sid_is_dead = runner.session_store._is_session_ended_in_db(
                    _peek_cached_sid
                )
            except Exception:
                _cached_sid_is_dead = False

        # Detect cross-process writes: when another process (e.g. hermes
        # dashboard) appends to the same session in the shared SessionDB,
        # the cached agent's in-memory transcript becomes stale.  Compare
        # the session's current message_count against the count recorded
        # when the agent was cached; on mismatch, invalidate the cache
        # so a fresh agent re-reads from disk. (#45966)
        _current_msg_count = None
        if runner._session_db is not None and session_id:
            try:
                # run_sync is off-loop (executor); sync DB is fine.
                _sess_row = runner._session_db._db.get_session(session_id)
                if _sess_row:
                    _current_msg_count = _sess_row.get("message_count", 0)
            except Exception:
                pass

        _xproc_evicted_agent = None
        if _cache_lock and _cache is not None:
            with _cache_lock:
                cached = _cache.get(session_key)
                if cached and cached[1] == _sig:
                    # cached[2] is the message_count at cache time;
                    # stale when a second process appended rows.
                    # cached[3] (when present) is the session_id the
                    # snapshot was taken for — used to skip the guard
                    # when the active session_id differs (#54947).
                    _cached_mc = cached[2] if len(cached) > 2 else None
                    _cached_sid = cached[3] if len(cached) > 3 else None
                    # If the snapshot belongs to a different session_id
                    # (same session_key, different conversation), the
                    # message_count comparison is meaningless — the
                    # counts track DIFFERENT DB rows.  REUSE the cached
                    # agent rather than rebuild and bust the prompt cache
                    # on every session switch (#54947).
                    _session_id_mismatch = (
                        _cached_sid is not None
                        and session_id is not None
                        and _cached_sid != session_id
                    )
                    # Re-validate the OUTSIDE-lock dead-session peek
                    # against the tuple actually read under THIS lock —
                    # the cache entry could have been replaced between
                    # the peek and this lock acquisition, and a stale
                    # "dead" verdict must never be applied to a
                    # different (possibly live) cached agent.
                    _stale_dead_sid_reuse = (
                        _session_id_mismatch
                        and _cached_sid_is_dead
                        and _cached_sid == _peek_cached_sid
                    )
                    if _stale_dead_sid_reuse:
                        # #54878 x #54947 interaction: the routing key
                        # was just self-healed away from a session that
                        # state.db already marked ended, but the cached
                        # AIAgent here still belongs to that DEAD
                        # session_id. The #54947 "different session_id
                        # under the same key = intentional switch, reuse
                        # freely" rule does not hold here — this isn't a
                        # sibling conversation, it's a stale agent left
                        # over from before the self-heal. Reusing it lets
                        # this turn's post-run "session split" sync write
                        # the routing key straight back onto the dead
                        # session_id, undoing the self-heal and looping
                        # every message until an interrupt happens to
                        # race in first. Discard and rebuild fresh
                        # instead, same as a genuine cross-process write.
                        logger.info(
                            "Agent cache invalidated for session %s: "
                            "cached agent's session_id %s is ended in "
                            "state.db (stale self-heal artifact, "
                            "#54878 x #54947) — discarding instead of "
                            "reusing across the routing recovery",
                            session_key, _cached_sid,
                        )
                        evicted = runner._agent_cache.pop(session_key, None)
                        _ev_agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
                        if _ev_agent and _ev_agent is not _AGENT_PENDING_SENTINEL:
                            # Same deferred-cleanup rationale as the
                            # cross-process branch below (#52197): don't
                            # block the event loop / cache lock on
                            # memory-provider shutdown or socket teardown.
                            _xproc_evicted_agent = _ev_agent
                    elif (
                        not _session_id_mismatch
                        and _cached_mc is not None
                        and _current_msg_count is not None
                        and _current_msg_count != _cached_mc
                    ):
                        # Cross-process write detected — discard stale
                        # agent so it rebuilds from fresh DB transcript.
                        logger.info(
                            "Agent cache invalidated for session %s: "
                            "message_count changed (%s -> %s), "
                            "possible cross-process write",
                            session_key, _cached_mc, _current_msg_count,
                        )
                        evicted = runner._agent_cache.pop(session_key, None)
                        _ev_agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
                        if _ev_agent and _ev_agent is not _AGENT_PENDING_SENTINEL:
                            # Defer cleanup until AFTER the lock is
                            # released — _cleanup_agent_resources /
                            # release_clients can block on memory-provider
                            # shutdown and socket teardown, and running it
                            # here would stall the gateway event loop while
                            # _sweep_idle_cached_agents (session-expiry
                            # watcher) waits on the same lock, blocking
                            # Discord heartbeats (#52197).  The same session
                            # rebuilds a fresh agent immediately below, so
                            # use the SOFT release that preserves the
                            # session's terminal sandbox / browser / bg
                            # processes for the rebuilt agent to inherit —
                            # mirrors _evict_cached_agent / idle-sweep.
                            _xproc_evicted_agent = _ev_agent
                    else:
                        agent = cached[0]
                        # Refresh LRU order so the cap enforcement evicts
                        # truly-oldest entries, not the one we just used.
                        if hasattr(_cache, "move_to_end"):
                            try:
                                _cache.move_to_end(session_key)
                            except KeyError:
                                pass
                        runner._init_cached_agent_for_turn(agent, _interrupt_depth)
                        # Refresh agent max_iterations from current config
                        # (cached agent may have been created with old config)
                        agent.max_iterations = max_iterations
                        logger.debug("Reusing cached agent for session %s", session_key)
                        reused_cached_agent = True

        # Lock released — refresh the fallback chain from disk for the
        # reused agent OUTSIDE the cache lock (config.yaml read is disk
        # I/O; the idle-sweep watcher contends on this lock and stalls
        # Discord heartbeats — same reasoning as #52197).  A chain
        # configured after this agent was cached (or after gateway start)
        # must reach the next turn (#60955).  Per-session turn
        # serialization (_running_agents) keeps this safe post-lock.
        if reused_cached_agent and agent is not None:
            runner._apply_fallback_chain_to_agent(
                agent, runner._refresh_fallback_model(),
            )

        # Lock released — now schedule cleanup of any cross-process-evicted
        # agent on a daemon thread so memory-provider shutdown / socket
        # teardown never blocks the gateway event loop or the cache lock
        # the session-expiry watcher needs (#52197).
        if _xproc_evicted_agent is not None:
            try:
                threading.Thread(
                    target=runner._release_evicted_agent_soft,
                    args=(_xproc_evicted_agent,),
                    daemon=True,
                    name=f"agent-xproc-evict-{str(session_key)[:24]}",
                ).start()
            except Exception:
                # Interpreter shutdown or thread-spawn failure — release
                # inline as a best-effort fallback.
                try:
                    runner._release_evicted_agent_soft(_xproc_evicted_agent)
                except Exception:
                    pass

        if agent is None:
            # Config changed or first message — create fresh agent
            agent = AIAgent(
                model=turn_route["model"],
                **turn_route["runtime"],
                max_iterations=max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                ephemeral_system_prompt=combined_ephemeral or None,
                prefill_messages=runner._prefill_messages or None,
                reasoning_config=reasoning_config,
                service_tier=runner._service_tier,
                request_overrides=turn_route.get("request_overrides"),
                providers_allowed=pr.get("only"),
                providers_ignored=pr.get("ignore"),
                providers_order=pr.get("order"),
                provider_sort=pr.get("sort"),
                provider_require_parameters=pr.get("require_parameters", False),
                provider_data_collection=pr.get("data_collection"),
                session_id=session_id,
                platform=platform_key,
                user_id=source.user_id,
                user_id_alt=source.user_id_alt,
                user_name=source.user_name,
                chat_id=source.chat_id,
                chat_name=source.chat_name,
                chat_type=source.chat_type,
                thread_id=source.thread_id,
                gateway_session_key=session_key,
                session_db=getattr(runner._session_db, "_db", runner._session_db),
                # Reload from disk — do not reuse the startup snapshot (#60955).
                fallback_model=runner._refresh_fallback_model(),
            )
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    # Record the session_id the snapshot was taken for
                    # alongside the message_count, so the cross-process
                    # guard can skip the (meaningless) count comparison
                    # when the active session_id later switches under
                    # the same session_key (#54947).
                    _cache[session_key] = (
                        agent, _sig, _current_msg_count, session_id,
                    )
                    runner._enforce_agent_cache_cap()
            logger.debug("Created new agent for session %s (sig=%s)", session_key, _sig)

        # Per-message state — callbacks and reasoning config change every
        # turn and must not be baked into the cached agent constructor.
        # Gate on needs_progress_queue (tool_progress OR thinking_progress)
        # rather than tool_progress alone: the progress_callback also relays
        # _thinking assistant scratch text, which is gated on
        # thinking_progress and is intentionally independent of tool
        # progress. With the old `tool_progress_enabled`-only gate, a user
        # who set thinking_progress:true but kept tool_progress:off got a
        # None callback — so _thinking scratch bubbles never relayed even
        # though the progress queue was created for them.
        agent.tool_progress_callback = (
            progress_callback
            if (
                needs_progress_queue
                or log_mode_enabled
                or _live_status_adapter is not None
            )
            else None
        )
        # Discord voice verbal-ack hook (fires once per turn on first tool
        # call; armed only when in a voice channel with the mixer running).
        agent.tool_start_callback = (
            voice_ack_callback if _voice_ack_guild[0] is not None else None
        )
        agent.step_callback = _step_callback_sync if _hooks_ref.loaded_hooks else None
        agent.stream_delta_callback = _stream_delta_cb
        agent.interim_assistant_callback = _interim_assistant_cb if _want_interim_messages else None
        agent.status_callback = _status_callback_sync
        # Credits / out-of-band notices (usage bands, depletion, restored).
        # Messaging has no persistent status bar, so each notice is a
        # standalone push: render to a single plaintext line and deliver via
        # the shared _deliver_platform_notice rail (honors private/public +
        # thread metadata). Fires from the agent's sync worker thread, so we
        # hop onto the gateway loop with safe_schedule_threadsafe - same
        # pattern as _status_callback_sync. The fired-once latch lives on the
        # cached agent and persists across turns, so a band crosses -> one
        # push (no per-turn re-nag). Recovery ("✓ Credit access restored")
        # rides the same show path (it's emitted as a success notice, not a
        # clear). The clear callback is a no-op: a sent platform message
        # can't be cleanly retracted, and the band already fired once.
        def _notice_callback_sync(notice) -> None:
            if not _status_adapter or not _run_still_current():
                return
            try:
                line = render_notice_line(notice)
            except Exception:
                logger.debug("render_notice_line failed", exc_info=True)
                return
            if not line:
                return
            safe_schedule_threadsafe(
                runner._deliver_platform_notice(source, line),
                _loop_for_step,
                logger=logger,
                log_message="notice_callback delivery scheduling error",
            )

        agent.notice_callback = _notice_callback_sync
        agent.notice_clear_callback = None
        agent.event_callback = _event_callback_sync
        agent.reasoning_config = reasoning_config
        agent.service_tier = runner._service_tier
        agent.request_overrides = turn_route.get("request_overrides") or {}
        # Must-deliver notes for THIS turn ride the current user message
        # (api_content sidecar), never the system prompt: staged by
        # _handle_message_with_agent (auto-reset note, first-contact
        # intro, voice-channel change).  Assigned unconditionally so a
        # reused cached agent never replays a stale note.
        agent._gateway_turn_context_notes = _join_turn_sidecar_notes(
            runner._consume_pending_turn_sidecar_notes(session_key)
        )

        _bg_review_release = threading.Event()
        _bg_review_pending: list[str] = []
        _bg_review_pending_lock = threading.Lock()

        def _deliver_bg_review_message(message: str) -> None:
            if not _status_adapter or not _run_still_current():
                return
            safe_schedule_threadsafe(
                _status_adapter.send(
                    _status_chat_id,
                    message,
                    metadata=_non_conversational_metadata(_status_thread_metadata, platform=source.platform),
                ),
                _loop_for_step,
                logger=logger,
                log_message="background_review_callback scheduling error",
            )

        def _release_bg_review_messages() -> None:
            _bg_review_release.set()
            with _bg_review_pending_lock:
                pending = list(_bg_review_pending)
                _bg_review_pending.clear()
            for queued in pending:
                _deliver_bg_review_message(queued)

        # Background review delivery — send "💾 Memory updated" etc. to user
        def _bg_review_send(message: str) -> None:
            if not _status_adapter or not _run_still_current():
                return
            if not _bg_review_release.is_set():
                with _bg_review_pending_lock:
                    if not _bg_review_release.is_set():
                        _bg_review_pending.append(message)
                        return
            _deliver_bg_review_message(message)

        agent.background_review_callback = _bg_review_send
        # Register the release hook on the adapter so base.py's finally
        # block can fire it after delivering the main response.
        if _status_adapter and session_key:
            if getattr(type(_status_adapter), "register_post_delivery_callback", None) is not None:
                _status_adapter.register_post_delivery_callback(
                    session_key,
                    _release_bg_review_messages,
                    generation=run_generation,
                )
            else:
                _pdc = getattr(_status_adapter, "_post_delivery_callbacks", None)
                if _pdc is not None:
                    _pdc[session_key] = _release_bg_review_messages
        # Memory update notifications in chat.  Config: display.memory_notifications
        #   off     — no chat notification (still logged to stdout)
        #   on      — generic "💾 Memory updated" (default)
        #   verbose — content preview: "💾 Memory ➕ Hermes Repo..."
        _mem_notif = user_config.get("display", {}).get("memory_notifications")
        if isinstance(_mem_notif, bool):
            _mem_notif = "on" if _mem_notif else "off"
        agent.memory_notifications = str(_mem_notif).lower() if _mem_notif else "on"

        # ------------------------------------------------------------------
        # Clarify callback: present a clarify prompt and block on a response.
        #
        # Runs on the agent's worker thread (see clarify_tool's synchronous
        # callback contract).  Bridges sync→async by scheduling the
        # adapter's send_clarify on the gateway event loop, then blocks on
        # the clarify primitive's threading.Event with a configurable
        # timeout.  Returns the user's response string, or a sentinel
        # explaining that no response arrived (so the agent can adapt
        # rather than hang forever).
        # ------------------------------------------------------------------
        def _clarify_callback_sync(question: str, choices) -> str:
            from tools import clarify_gateway as _clarify_mod
            import uuid as _uuid

            if not _status_adapter:
                return ""

            clarify_id = _uuid.uuid4().hex[:10]
            _clarify_mod.register(
                clarify_id=clarify_id,
                session_key=session_key or "",
                question=question,
                choices=list(choices) if choices else None,
            )

            # Pause typing — like approval, we don't want a "thinking..."
            # status to obscure the prompt or block the user from typing
            # an "Other" response on platforms that disable input while
            # typing is active (Slack Assistant API).
            try:
                _status_adapter.pause_typing_for_chat(_status_chat_id)
            except Exception:
                pass

            send_ok = False
            fut = safe_schedule_threadsafe(
                _status_adapter.send_clarify(
                    chat_id=_status_chat_id,
                    question=question,
                    choices=list(choices) if choices else None,
                    clarify_id=clarify_id,
                    session_key=session_key or "",
                    metadata=_status_thread_metadata,
                ),
                _loop_for_step,
                logger=logger,
                log_message="Clarify send failed to schedule",
            )
            if fut is None:
                send_ok = False
            else:
                try:
                    result = fut.result(timeout=15)
                    send_ok = bool(getattr(result, "success", False))
                except Exception as exc:
                    logger.warning("Clarify send failed: %s", exc)
                    send_ok = False

            if not send_ok:
                # Couldn't deliver the prompt — clean up and return
                # sentinel so the agent can fall back to a sensible
                # default rather than hanging.
                _clarify_mod.clear_session(session_key or "")
                return "[clarify prompt could not be delivered]"

            timeout = _clarify_mod.get_clarify_timeout()
            response = _clarify_mod.wait_for_response(clarify_id, timeout=float(timeout))
            if response is None or response == "":
                # Timeout or session-boundary cancellation
                return f"[user did not respond within {int(timeout / 60)}m]"
            return response

        agent.clarify_callback = _clarify_callback_sync

        # Show assistant thinking between tool calls — independent of
        # tool_progress mode. Mattermost needs an explicit per-platform
        # opt-in so global scratch-text display does not leak into threads.
        agent.thinking_progress = _thinking_enabled
        # Store agent reference for interrupt support
        agent_holder[0] = agent
        # Capture the full tool definitions for transcript logging
        tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None

        # Convert history to agent format.
        # Two cases:
        #   1. Normal path (from transcript): simple {role, content, timestamp} dicts
        #      - Strip timestamps, keep role+content
        #   2. Interrupt path (from agent result["messages"]): full agent messages
        #      that may include tool_calls, tool_call_id, reasoning, etc.
        #      - These must be passed through intact so the API sees valid
        #        assistant→tool sequences (dropping tool_calls causes 500 errors)
        #
        # Telegram observed group context is handled structurally here:
        # observed=True transcript rows are withheld from replayable
        # history and attached to the current addressed message as
        # API-only context, so persisted history stores only the real
        # addressed user turn.
        agent_history, observed_group_context = _build_gateway_agent_history(
            history,
            channel_prompt=channel_prompt,
            inject_timestamps=_message_timestamps_enabled(_load_gateway_config()),
        )

        # FTS write-corruption guard (#50502): when message persistence
        # fails silently through corrupt FTS triggers, the reloaded
        # transcript above is stale/empty even though the SAME cached agent
        # still holds the full live conversation in `_session_messages`.
        # Replacing the live transcript with that shorter copy causes
        # immediate same-session amnesia. Only applies when we reused a
        # cached agent bound to this exact session_id.
        if reused_cached_agent and getattr(agent, "session_id", None) == session_id:
            _selected = _select_cached_agent_history(
                agent_history, getattr(agent, "_session_messages", None)
            )
            if _selected is not agent_history:
                logger.warning(
                    "Persisted transcript lagged live cached history for "
                    "session %s (disk=%d, memory=%d); preserving live "
                    "conversation context (possible FTS write corruption)",
                    session_key, len(agent_history), len(_selected),
                )
                # The live in-memory history bypassed the
                # _build_gateway_agent_history cleanup pipeline above —
                # re-apply the stale-confirmation expiry (#59607) so a
                # dangerous confirmation can't slip through this path
                # either. Idempotent; messages without timestamps are
                # untouched.
                agent_history = _strip_stale_dangerous_confirmations(
                    _selected, now=time.time()
                )

        # Collect MEDIA paths already in history so we can exclude them
        # from the current turn's extraction. This is compression-safe:
        # even if the message list shrinks, we know which paths are old.
        _history_media_paths: set = _collect_history_media_paths(agent_history)

        # Register per-session gateway approval callback so dangerous
        # command approval blocks the agent thread (mirrors CLI input()).
        # The callback bridges sync→async to send the approval request
        # to the user immediately.
        from tools.approval import (
            register_gateway_notify,
            reset_current_session_key,
            set_current_session_key,
            unregister_gateway_notify,
        )

        def _approval_notify_sync(approval_data: dict) -> None:
            """Send the approval request to the user from the agent thread.

            If the adapter supports interactive button-based approvals
            (e.g. Discord's ``send_exec_approval``), use that for a richer
            UX.  Otherwise fall back to a plain text message with
            ``/approve`` instructions.
            """
            # Pause the typing indicator while the agent waits for
            # user approval.  Critical for Slack's Assistant API where
            # assistant_threads_setStatus disables the compose box — the
            # user literally cannot type /approve while "is thinking..."
            # is active.  The approval message send auto-clears the Slack
            # status; pausing prevents _keep_typing from re-setting it.
            # Typing resumes in _handle_approve_command/_handle_deny_command.
            _status_adapter.pause_typing_for_chat(_status_chat_id)

            cmd = approval_data.get("command", "")
            desc = approval_data.get("description", "dangerous command")

            # Redact credentials from the command before displaying it in
            # the approval prompt — Tirith's findings are already redacted,
            # but the raw command string still leaks secrets to the chat
            # platform (#48456). Applied here so BOTH the button-based
            # (send_exec_approval) and plain-text fallback paths below use
            # the redacted value.
            cmd = _redact_approval_command(cmd)

            # Prefer button-based approval when the adapter supports it.
            # Check the *class* for the method, not the instance — avoids
            # false positives from MagicMock auto-attribute creation in tests.
            if getattr(type(_status_adapter), "send_exec_approval", None) is not None:
                try:
                    _approval_fut = safe_schedule_threadsafe(
                        _status_adapter.send_exec_approval(
                            chat_id=_status_chat_id,
                            command=cmd,
                            session_key=_approval_session_key,
                            description=desc,
                            metadata=_status_thread_metadata,
                            allow_permanent=approval_data.get("allow_permanent", True),
                            smart_denied=approval_data.get("smart_denied", False),
                        ),
                        _loop_for_step,
                        logger=logger,
                        log_message="send_exec_approval scheduling error",
                    )
                    if _approval_fut is None:
                        raise RuntimeError("send_exec_approval: loop unavailable")
                    _approval_result = _approval_fut.result(timeout=15)
                    if _approval_result.success:
                        return
                    logger.warning(
                        "Button-based approval failed (send returned error), falling back to text: %s",
                        _approval_result.error,
                    )
                except Exception as _e:
                    logger.warning(
                        "Button-based approval failed, falling back to text: %s", _e
                    )

            # Fallback: plain text approval prompt.  Use the adapter's
            # typed prefix so Slack/Matrix users are told the form they
            # can actually type (`!approve`) — typed "/" is blocked in
            # Slack threads and reserved by Matrix clients.
            _p = getattr(_status_adapter, "typed_command_prefix", "/")
            msg = _format_exec_approval_fallback(
                cmd,
                desc,
                _p,
                allow_permanent=approval_data.get("allow_permanent", True),
                smart_denied=approval_data.get("smart_denied", False),
            )
            try:
                _approval_send_fut = safe_schedule_threadsafe(
                    _status_adapter.send(
                        _status_chat_id,
                        msg,
                        metadata=_status_thread_metadata,
                    ),
                    _loop_for_step,
                    logger=logger,
                    log_message="Approval text-send scheduling error",
                )
                if _approval_send_fut is not None:
                    _approval_send_fut.result(timeout=15)
            except Exception as _e:
                logger.error("Failed to send approval request: %s", _e)

        # Keep real user text separate from API-only recovery guidance.  If
        # an auto-continue note is prepended below, persist the original
        # message so stale guidance never replays as user-authored text.
        _persist_user_message_override: Optional[Any] = persist_user_message
        _persist_user_timestamp_override: Optional[float] = persist_user_timestamp

        # Prepend pending model switch note so the model knows about the switch
        _pending_notes = getattr(runner, '_pending_model_notes', {})
        _msn = _pending_notes.pop(session_key, None) if session_key else None
        if _msn:
            message = _msn + "\n\n" + message

        # Auto-continue: if the loaded history ends with a tool result,
        # the previous agent turn was interrupted mid-work (gateway
        # restart, crash, SIGTERM).  Prepend a system note so the model
        # finishes processing the pending tool results before addressing
        # the user's new message.  (#4493)
        #
        # Session-level resume_pending (set on drain-timeout shutdown)
        # escalates the wording — the transcript's last role may be
        # anything (tool, assistant with unfinished work, etc.), so we
        # give a stronger, reason-aware instruction that subsumes the
        # tool-tail case.
        #
        # Freshness gate (#16802): both branches are gated on the age
        # of the last persisted transcript row.  That is the correct
        # "when did we last do anything here" signal for both the
        # resume_pending path (restart watchdog) and the tool-tail
        # path (in-flight tool loop killed).  We read ``history[-1]``
        # here because ``agent_history`` has already stripped the
        # ``timestamp`` field off tool/tool_call rows for API purity
        # (see the `k != "timestamp"` filter above).  Rows without a
        # timestamp (legacy transcripts) are treated as fresh so the
        # historical auto-continue behaviour is preserved.
        _freshness_window = _auto_continue_freshness_window()
        _interruption_is_fresh = _is_fresh_gateway_interruption(
            _last_transcript_timestamp(history),
            window_secs=_freshness_window,
        )

        _resume_entry = None
        if session_key:
            try:
                _resume_entry = runner.session_store._entries.get(session_key)
            except Exception:
                _resume_entry = None

        # resume_pending freshness uses a SECOND signal in addition to the
        # transcript clock above.  The restart watchdog stamps the session
        # with ``last_resume_marked_at`` at interrupt time — that is the
        # correct "when were we interrupted" signal.  The transcript clock
        # (_interruption_is_fresh) can be far older: an active thread you
        # return to may have its last persisted row hours back, even though
        # the interruption itself just happened.  Gating resume_pending on
        # the transcript clock alone makes the recovery note silently drop,
        # and because the startup auto-resume turn carries empty text
        # (_schedule_resume_pending_sessions), the model then receives a
        # blank user message and replies with confused "the message came
        # through blank" noise.  Treat the marker as fresh when
        # EITHER signal is fresh so the two freshness checks agree.
        _resume_mark_is_fresh = False
        if _resume_entry is not None and getattr(_resume_entry, "resume_pending", False):
            _resume_mark_is_fresh = _is_fresh_gateway_interruption(
                getattr(_resume_entry, "last_resume_marked_at", None),
                window_secs=_freshness_window,
            )
        _is_resume_pending = bool(
            _resume_entry is not None
            and getattr(_resume_entry, "resume_pending", False)
            and (_interruption_is_fresh or _resume_mark_is_fresh)
        )
        _has_fresh_tool_tail = bool(
            agent_history
            and agent_history[-1].get("role") == "tool"
            and _interruption_is_fresh
        )

        if _is_resume_pending:
            _reason = getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
            _persist_user_message_override = message
            # The empty-message case is the auto-resume startup turn
            # synthesized by _schedule_resume_pending_sessions — there is
            # no NEW user message to address.  Guidance is adapter-aware:
            # interactive platforms report the restore and ask what next;
            # non-interactive event platforms (webhook, API server)
            # continue the interrupted work instead, because nobody is
            # present to answer and an acknowledgement would silently
            # abandon the task (#57056).
            _resume_adapter = runner._adapter_for_source(source)
            _interactive_resume = bool(
                getattr(_resume_adapter, "interactive_resume", True)
            )
            message = build_resume_recovery_note(
                _reason, message, interactive=_interactive_resume,
            )
        elif _has_fresh_tool_tail:
            _persist_user_message_override = message
            message = (
                "[System note: A new message has arrived. The conversation "
                "history contains pending tool outputs from an interrupted turn. "
                "IGNORE those pending results. Address the user's NEW message "
                "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
                + message
            )

        # Consume one-shot /reload-skills note (if the user ran
        # /reload-skills since their last turn in this session). Same
        # queue pattern as CLI: prepend to the NEXT user message, then
        # clear. Nothing was written to the transcript out-of-band, so
        # message alternation stays intact.
        _pending_notes = getattr(runner, "_pending_skills_reload_notes", None)
        if _pending_notes and session_key and session_key in _pending_notes:
            _srn = _pending_notes.pop(session_key, None)
            if _srn:
                message = _srn + "\n\n" + message

        # Safety net: a startup auto-resume event carries empty
        # text and relies on the resume_pending branch above to supply the
        # recovery note.  If that branch did not fire for any reason (e.g.
        # both freshness signals disagreed, or the marker was cleared
        # between scheduling and dispatch) we must NOT hand the model a
        # blank user turn — it responds with confused "the message came
        # through blank" noise.  Restricted to resume_pending sessions so
        # legitimately empty user turns (e.g. an image with no caption,
        # wrapped as native content below) are untouched.
        if (
            isinstance(message, str)
            and not message.strip()
            and _resume_entry is not None
            and getattr(_resume_entry, "resume_pending", False)
        ):
            _sn_reason = (
                getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
            )
            _sn_adapter = runner._adapter_for_source(source)
            message = build_resume_recovery_note(
                _sn_reason,
                "",
                interactive=bool(
                    getattr(_sn_adapter, "interactive_resume", True)
                ),
            )

        _approval_session_key = session_key or ""
        _approval_session_token = set_current_session_key(_approval_session_key)
        register_gateway_notify(_approval_session_key, _approval_notify_sync)
        try:
            # If _prepare_inbound_message_text buffered image paths for native
            # attachment, wrap the user turn as an OpenAI-style multimodal
            # content list. Consume-and-clear so subsequent turns on the same
            # runner instance don't re-attach stale images.
            _native_imgs = runner._consume_pending_native_image_paths(session_key)
            if _native_imgs:
                try:
                    from agent.image_routing import build_native_content_parts
                    _parts, _skipped = build_native_content_parts(
                        message,
                        _native_imgs,
                    )
                    if _skipped:
                        logger.warning(
                            "Native image attachment: skipped %d unreadable path(s): %s",
                            len(_skipped), _skipped,
                        )
                    if any(p.get("type") == "image_url" for p in _parts):
                        _run_message: Any = _parts
                    else:
                        # All images failed to read — fall back to plain text.
                        _run_message = message
                except Exception as _img_exc:
                    logger.warning(
                        "Native image attachment failed, falling back to text: %s",
                        _img_exc,
                    )
                    _run_message = message
            else:
                _run_message = message

            _api_run_message = _wrap_current_message_with_observed_context(
                _run_message,
                observed_group_context,
            )
            _conversation_kwargs = {
                "conversation_history": agent_history,
                "task_id": session_id,
            }
            if _persist_user_message_override is not None:
                _conversation_kwargs["persist_user_message"] = _persist_user_message_override
            elif observed_group_context:
                _conversation_kwargs["persist_user_message"] = message
            if moa_config is not None:
                _conversation_kwargs["moa_config"] = moa_config
            if _persist_user_timestamp_override is not None:
                _conversation_kwargs["persist_user_timestamp"] = _persist_user_timestamp_override
            result = agent.run_conversation(_api_run_message, **_conversation_kwargs)
        finally:
            unregister_gateway_notify(_approval_session_key)
            # Cancel any pending clarify entries so blocked agent
            # threads don't hang past the end of the run (interrupt,
            # completion, gateway shutdown).  Idempotent.
            try:
                from tools.clarify_gateway import clear_session as _clear_clarify_session
                _clear_clarify_session(_approval_session_key)
            except Exception:
                pass
            reset_current_session_key(_approval_session_token)
        result_holder[0] = result

        # Signal the stream consumer that the agent is done
        if _stream_consumer is not None:
            _stream_consumer.finish()

        # Return final response, or a message if something went wrong
        final_response = result.get("final_response")

        # Extract actual token counts from the agent instance used for this run
        _last_prompt_toks = 0
        _input_toks = 0
        _output_toks = 0
        _context_length = 0
        _agent = agent_holder[0]
        if _agent and hasattr(_agent, "context_compressor"):
            _last_prompt_toks = getattr(_agent.context_compressor, "last_prompt_tokens", 0)
            _input_toks = getattr(_agent, "session_prompt_tokens", 0)
            _output_toks = getattr(_agent, "session_completion_tokens", 0)
            _context_length = getattr(_agent.context_compressor, "context_length", 0) or 0
        _resolved_model = getattr(_agent, "model", None) if _agent else None

        # Sync session_id immediately after run_conversation(). Compression
        # can rotate before a follow-up model call fails; the failure return
        # below must still point the gateway at the compressed child.
        agent = agent_holder[0]
        _session_was_split = False
        # In-place compaction (compression.in_place / #38763) compacts the
        # transcript WITHOUT rotating the id, so the id-change diff below
        # can't detect it. compress_context() sets this rotation-independent
        # flag on the agent; the gateway uses it to re-baseline transcript
        # handling (history_offset=0 + rewrite the JSONL transcript) the
        # same way a split would, even though the session_id is unchanged.
        _compacted_in_place = bool(getattr(agent, "_last_compaction_in_place", False)) if agent else False
        agent_session_id = getattr(agent, 'session_id', session_id) if agent else session_id
        if agent and session_key and agent_session_id != session_id:
            _session_was_split = True
            logger.info(
                "Session split detected: %s → %s (compression)",
                session_id, agent_session_id,
            )
            entry = runner.session_store._entries.get(session_key)
            _session_split_entry_persisted = False
            if entry:
                entry_session_id = getattr(entry, "session_id", None)
                if not _run_still_current():
                    logger.info(
                        "Skipping session split sync for stale run %s — "
                        "generation %s is no longer current",
                        session_key or "?",
                        run_generation,
                    )
                elif entry_session_id == agent_session_id:
                    _session_split_entry_persisted = True
                elif entry_session_id != session_id:
                    logger.info(
                        "Skipping session split sync for %s because the "
                        "session binding moved from %s to %s before "
                        "compression finished",
                        session_key or "?",
                        session_id,
                        entry_session_id,
                    )
                else:
                    entry.session_id = agent_session_id
                    runner.session_store._save()
                    runner.session_store._record_gateway_session_peer(
                        agent_session_id,
                        session_key,
                        source,
                    )
                    _session_split_entry_persisted = True

            # If this is a Telegram DM and source.thread_id was lost during
            # the session split (synthetic / recovered event), restore it
            # from the binding so _thread_metadata_for_source produces the
            # correct message_thread_id instead of routing to the General
            # thread.  Failure here is non-fatal — we log and continue;
            # worst case the message lands in General, which is the
            # pre-fix behaviour. Only do this after this run successfully
            # published its session split; a stale /stop→/new predecessor
            # must not mutate routing/binding state for the fresh session.
            if _session_split_entry_persisted and (
                getattr(source, "platform", None) == Platform.TELEGRAM
                and getattr(source, "chat_type", None) == "dm"
                and getattr(source, "thread_id", None) is None
                and runner._session_db is not None
            ):
                try:
                    # run_sync is off-loop (executor); sync DB is fine.
                    _binding = runner._session_db._db.get_telegram_topic_binding_by_session(
                        session_id=agent_session_id,
                    )
                    if _binding and _binding.get("thread_id"):
                        source.thread_id = str(_binding["thread_id"])
                        logger.debug(
                            "Restored source.thread_id=%s from binding after session split %s → %s",
                            source.thread_id,
                            session_id,
                            agent_session_id,
                        )
                except Exception:
                    logger.debug(
                        "Failed to restore thread_id from binding after session split",
                        exc_info=True,
                    )
            if _session_split_entry_persisted:
                runner._sync_telegram_topic_binding(
                    source, entry, reason="agent-run-compression",
                )

        effective_session_id = agent_session_id
        runner._sync_session_model_from_agent(effective_session_id, agent)
        # history_offset=0 whenever the agent's message list no longer has
        # the original history prefix — i.e. on rotation (split) OR in-place
        # compaction. In both cases the returned `messages` is the compacted
        # set, so the gateway must persist all of it (offset 0), not slice
        # past the pre-compaction length (which would drop everything).
        _effective_history_offset = (
            0 if (_session_was_split or _compacted_in_place) else len(agent_history)
        )

        if not final_response:
            final_response = _normalize_empty_agent_response(
                result, final_response or "", history_len=len(agent_history),
            )
            final_response = _sanitize_gateway_final_response(source.platform, final_response)
            if not final_response:
                final_response = f"⚠️ {result['error']}" if result.get("error") else ""
            return {
                "final_response": final_response,
                "messages": result.get("messages", []),
                "api_calls": result.get("api_calls", 0),
                "failed": result.get("failed", False),
                "partial": result.get("partial", False),
                "completed": result.get("completed"),
                "interrupted": result.get("interrupted", False),
                "interrupt_message": result.get("interrupt_message"),
                "error": result.get("error"),
                "compression_exhausted": result.get("compression_exhausted", False),
                "tools": tools_holder[0] or [],
                "history_offset": _effective_history_offset,
                "compacted_in_place": _compacted_in_place,
                "session_id": effective_session_id,
                "last_prompt_tokens": _last_prompt_toks,
                "input_tokens": _input_toks,
                "output_tokens": _output_toks,
                "model": _resolved_model,
                "context_length": _context_length,
            }

        # Scan tool results for MEDIA:<path> tags that need to be delivered
        # as native audio/file attachments.  The TTS tool embeds MEDIA: tags
        # in its JSON response, but the model's final text reply usually
        # doesn't include them.  We collect unique tags from tool results and
        # append any that aren't already present in the final response, so the
        # adapter's extract_media() can find and deliver the files exactly once.
        #
        # Scope the scan to THIS turn's tool results only. ``agent_history``
        # was passed into run_conversation as ``conversation_history``, so the
        # agent's returned ``messages`` list is ``agent_history`` followed by
        # the messages produced this turn. Slicing at ``len(agent_history)``
        # isolates the current turn precisely, so a stale MEDIA: path emitted
        # by a tool several turns earlier (still present in the full message
        # list) can never leak onto a later text-only reply. (Fixes #34608)
        #
        # Path-based deduplication against _history_media_paths (collected
        # before run_conversation) is retained as a secondary guard. It is
        # also the sole guard on the fallback branch taken when mid-run
        # context compression shrinks the message list below the original
        # history length, preserving the compression-safe behaviour of #160.
        if "MEDIA:" not in final_response:
            media_tags, has_voice_directive = _collect_auto_append_media_tags(
                result.get("messages", []),
                history_offset=len(agent_history),
                history_media_paths=_history_media_paths,
            )

            if media_tags:
                seen = set()
                unique_tags = []
                for tag in media_tags:
                    if tag not in seen:
                        seen.add(tag)
                        unique_tags.append(tag)
                if has_voice_directive:
                    unique_tags.insert(0, "[[audio_as_voice]]")
                final_response = final_response + "\n" + "\n".join(unique_tags)

        # Auto-generate session title after first exchange (non-blocking)
        if final_response and runner._session_db:
            try:
                from agent.title_generator import maybe_auto_title
                all_msgs = result_holder[0].get("messages", []) if result_holder[0] else []
                # In Gateway mode, auto-title failures must NOT be
                # surfaced as user-visible messages (fixes #23246).
                # Log them at debug level only — they are not actionable
                # to the end user. CLI mode keeps the existing behaviour
                # via the agent's _emit_auxiliary_failure path.
                def _title_failure_cb(task: str, exc: BaseException) -> None:
                    logger.debug(
                        "Gateway auto-title failure suppressed (not user-visible): %s: %s",
                        task, exc,
                    )
                # Snapshot the runtime identity; the validator lets the
                # background titler skip its LLM call if the session's
                # model changed before it fires (a stale request would
                # reload an unloaded Ollama model, #19027).
                _title_model = getattr(agent, "model", None) if agent else None
                _title_provider = getattr(agent, "provider", None) if agent else None
                maybe_auto_title_kwargs = {
                    "failure_callback": _title_failure_cb,
                    "main_runtime": {
                        "model": getattr(agent, "model", None),
                        "provider": getattr(agent, "provider", None),
                        "base_url": getattr(agent, "base_url", None),
                        "api_key": getattr(agent, "api_key", None),
                        "api_mode": getattr(agent, "api_mode", None),
                    } if agent else None,
                    "runtime_validator": (lambda: (
                        getattr(agent, "model", None) == _title_model
                        and getattr(agent, "provider", None) == _title_provider
                    )) if agent else None,
                }
                if runner._is_telegram_topic_lane(source):
                    maybe_auto_title_kwargs["title_callback"] = lambda title: runner._schedule_telegram_topic_title_rename(
                        source,
                        effective_session_id,
                        title,
                    )
                elif runner._is_discord_auto_thread_lane(source):
                    maybe_auto_title_kwargs["title_callback"] = lambda title: runner._schedule_discord_semantic_thread_rename(
                        source,
                        effective_session_id,
                        title,
                    )
                maybe_auto_title(
                    getattr(runner._session_db, "_db", runner._session_db),
                    effective_session_id,
                    message,
                    final_response,
                    all_msgs,
                    **maybe_auto_title_kwargs,
                )
            except Exception:
                pass

        return {
            "final_response": final_response,
            "last_reasoning": result.get("last_reasoning"),
            "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
            "api_calls": result_holder[0].get("api_calls", 0) if result_holder[0] else 0,
            "completed": result_holder[0].get("completed") if result_holder[0] else None,
            "interrupted": result_holder[0].get("interrupted", False) if result_holder[0] else False,
            "partial": result_holder[0].get("partial", False) if result_holder[0] else False,
            "error": result_holder[0].get("error") if result_holder[0] else None,
            "interrupt_message": result_holder[0].get("interrupt_message") if result_holder[0] else None,
            "tools": tools_holder[0] or [],
            "history_offset": _effective_history_offset,
            "compacted_in_place": _compacted_in_place,
            "last_prompt_tokens": _last_prompt_toks,
            "input_tokens": _input_toks,
            "output_tokens": _output_toks,
            "model": _resolved_model,
            "context_length": _context_length,
            "session_id": effective_session_id,
            "response_previewed": result.get("response_previewed", False),
            "response_transformed": result.get("response_transformed", False),
            # Pass through the agent_persisted flag so the persistence block
            # above can correctly determine whether the codex app-server path
            # self-persisted (it didn't — see codex_runtime.py).  Default
            # True preserves the skip-db behaviour for the standard runtime.
            "agent_persisted": (result_holder[0].get("agent_persisted", True) if result_holder[0] else True),
        }

    # Start progress message sender if enabled. Gate on needs_progress_queue
    # (tool_progress OR thinking_progress), not tool_progress alone: the
    # sender drains BOTH tool-progress lines and _thinking scratch bubbles.
    # With the old tool_progress-only gate, a thinking_progress:true /
    # tool_progress:off user had the callback queue _thinking messages that
    # no task ever drained — so they silently never appeared.
    progress_task = None
    if needs_progress_queue:
        progress_task = asyncio.create_task(send_progress_messages())

    # Start the tool-call log writer when tool_progress == "log".
    log_task = None
    if log_mode_enabled:
        log_task = asyncio.create_task(write_tool_log())

    # Start stream consumer task — polls for consumer creation since it
    # happens inside run_sync (thread pool) after the agent is constructed.
    stream_task = None

    async def _start_stream_consumer():
        """Wait for the stream consumer to be created, then run it."""
        for _ in range(200):  # Up to 10s wait
            if stream_consumer_holder[0] is not None:
                await stream_consumer_holder[0].run()
                return
            await asyncio.sleep(0.05)

    stream_task = asyncio.create_task(_start_stream_consumer())

    # Track this agent as running for this session (for interrupt support)
    # We do this in a callback after the agent is created
    async def track_agent():
        # Wait for agent to be created
        while agent_holder[0] is None:
            await asyncio.sleep(0.05)
        if not session_key:
            return
        # Only promote the sentinel to the real agent if this run is still
        # current.  If /stop or /new bumped the generation while we were
        # spinning up, leave the newer run's slot alone — we'll be
        # discarded by the stale-result check in _handle_message_with_agent.
        if run_generation is not None and not runner._is_session_run_current(
            session_key, run_generation
        ):
            logger.info(
                "Skipping stale agent promotion for %s — generation %s is no longer current",
                session_key or "",
                run_generation,
            )
            return
        runner._running_agents[session_key] = agent_holder[0]
        if runner._draining:
            runner._update_runtime_status("draining")

    tracking_task = asyncio.create_task(track_agent())

    # Monitor for interrupts from the adapter (new messages arriving).
    # This is the PRIMARY interrupt path for regular text messages —
    # Level 1 (base.py) catches them before _handle_message() is reached,
    # so the Level 2 running_agent.interrupt() path never fires.
    # The inactivity poll loop below has a BACKUP check in case this
    # task dies (no error handling = silent death = lost interrupts).
    _interrupt_detected = asyncio.Event()  # shared with backup check

    async def monitor_for_interrupt():
        if not session_key:
            return

        while True:
            await asyncio.sleep(0.2)  # Check every 200ms
            try:
                # Re-resolve adapter each iteration so reconnects don't
                # leave us holding a stale reference.
                _adapter = runner._adapter_for_source(source)
                if not _adapter:
                    continue
                # Check if adapter has a pending interrupt for this session.
                # Must use session_key (build_session_key output) — NOT
                # source.chat_id — because the adapter stores interrupt events
                # under the full session key.
                if hasattr(_adapter, 'has_pending_interrupt') and _adapter.has_pending_interrupt(session_key):
                    agent = agent_holder[0]
                    if agent:
                        # Peek at the pending message text WITHOUT consuming it.
                        # The message must remain in _pending_messages so the
                        # post-run dequeue at _dequeue_pending_event() can
                        # retrieve the full MessageEvent (with media metadata).
                        # If we pop here, a race exists: the agent may finish
                        # before checking _interrupt_requested, and the message
                        # is lost — neither the interrupt path nor the dequeue
                        # path finds it.
                        _peek_event = _adapter._pending_messages.get(session_key)
                        pending_text = None
                        if _peek_event is not None:
                            pending_text = _peek_event.text or ""
                            # Transcribe audio media BEFORE signaling the
                            # agent, so voice messages interrupt with the
                            # real transcript instead of an empty string
                            # (or file-path placeholder). Matches the UX
                            # of fresh voice messages including the
                            # optional 🎙️ echo back to the user.
                            _media_urls = getattr(_peek_event, "media_urls", None) or []
                            if runner._pending_event_audio_paths(_peek_event):
                                pending_text, _ = await runner._transcribe_and_echo_pending_voice(
                                    _peek_event,
                                    _adapter,
                                    source,
                                    pending_text,
                                    log_context="Voice-interrupt",
                                    metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                )
                            elif not pending_text and _media_urls:
                                pending_text = _build_media_placeholder(_peek_event)
                        logger.debug("Interrupt detected from adapter, signaling agent...")
                        agent.interrupt(pending_text)
                        _interrupt_detected.set()
                        break
            except asyncio.CancelledError:
                raise
            except Exception as _mon_err:
                logger.debug("monitor_for_interrupt error (will retry): %s", _mon_err)

    interrupt_monitor = asyncio.create_task(monitor_for_interrupt())

    # Periodic "still working" notifications for long-running tasks.
    # Fires every N seconds so the user knows the agent hasn't died.
    # Config: agent.gateway_notify_interval in config.yaml, or
    # HERMES_AGENT_NOTIFY_INTERVAL env var.  Default 180s (3 min).
    # 0 = disable notifications.
    _NOTIFY_INTERVAL_RAW = _float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180)
    _NOTIFY_INTERVAL = _NOTIFY_INTERVAL_RAW if _NOTIFY_INTERVAL_RAW > 0 else None
    _long_running_mode = _display_surface_mode(
        "long_running_notifications",
        default=True,
        allow_generic=True,
    )
    if _long_running_mode == "off":
        _NOTIFY_INTERVAL = None
    _notify_start = time.time()

    async def _notify_long_running():
        if _NOTIFY_INTERVAL is None:
            return  # Notifications disabled (gateway_notify_interval: 0)
        _notify_adapter = runner._adapter_for_source(source)
        if not _notify_adapter:
            return
        # Track the heartbeat message id so we can edit-in-place on
        # platforms that support it (Telegram, Discord, Slack, etc.)
        # instead of spamming a new "Still working" bubble every
        # interval. Falls back to send-new when edit fails or isn't
        # supported by the adapter.
        _heartbeat_msg_id: Optional[str] = None
        while True:
            await asyncio.sleep(_NOTIFY_INTERVAL)
            # Stop heartbeating once this run no longer owns the session
            # slot or the executor has finished — otherwise a stale
            # "running: delegate_task" bubble can outlive the run that
            # spawned it (#12029). _executor_task is a closure var bound
            # just after this task is scheduled; tolerate the brief window
            # before then (the first wake is _NOTIFY_INTERVAL away anyway).
            try:
                _exec_ref = _executor_task
            except NameError:
                _exec_ref = None
            if not runner._should_emit_long_running_notification(
                session_key, agent_holder[0], _exec_ref
            ):
                break
            _elapsed_mins = int((time.time() - _notify_start) // 60)
            # Include agent activity context if available. Default
            # heartbeat is terse: elapsed + current tool. Verbose
            # iteration counter is gated on busy_ack_detail so users
            # who want it can opt in per platform.
            _agent_ref = agent_holder[0]
            _status_detail = ""
            _want_iteration_detail = bool(
                resolve_display_setting(
                    user_config,
                    platform_key,
                    "busy_ack_detail",
                    True,
                )
            )
            if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                try:
                    _a = _agent_ref.get_activity_summary()
                    _parts = []
                    if _want_iteration_detail:
                        _parts.append(
                            f"iteration {_a['api_call_count']}/{_a['max_iterations']}"
                        )
                    _action = _a.get("current_tool") or _a.get("last_activity_desc")
                    if _action:
                        _parts.append(str(_action))
                    if _parts:
                        _status_detail = " — " + ", ".join(_parts)
                except Exception:
                    pass
            _heartbeat_text = (
                _generic_status_phrase("status")
                if _long_running_mode == "generic"
                else f"⏳ Working — {_elapsed_mins} min{_status_detail}"
            )
            try:
                _notify_res = None
                if _heartbeat_msg_id:
                    try:
                        _notify_res = await _notify_adapter.edit_message(
                            source.chat_id,
                            _heartbeat_msg_id,
                            _heartbeat_text,
                        )
                    except Exception as _ee:
                        logger.debug("Heartbeat edit failed: %s", _ee)
                        _notify_res = None
                if not (_notify_res and getattr(_notify_res, "success", False)):
                    _notify_res = await _notify_adapter.send(
                        source.chat_id,
                        _heartbeat_text,
                        metadata=_non_conversational_metadata(_status_thread_metadata, platform=source.platform),
                    )
                    if getattr(_notify_res, "success", False) and getattr(
                        _notify_res, "message_id", None
                    ):
                        _heartbeat_msg_id = str(_notify_res.message_id)
                        if _cleanup_progress:
                            _cleanup_msg_ids.append(_heartbeat_msg_id)
            except Exception as _ne:
                logger.debug("Long-running notification error: %s", _ne)

    _notify_task = asyncio.create_task(_notify_long_running())

    def _stream_confirmed_final_delivery(
        consumer,
        final_text: str,
        *,
        previewed: bool = False,
    ) -> bool:
        """Return True only when the actual final reply reached the user."""
        if consumer is None:
            return False
        if getattr(consumer, "final_response_sent", False):
            return True
        if previewed:
            has_delivered_text = getattr(consumer, "has_delivered_text", None)
            if callable(has_delivered_text):
                try:
                    return bool(has_delivered_text(final_text))
                except Exception:
                    return False
        return False

    try:
        # Run in thread pool to not block.  Use an *inactivity*-based
        # timeout instead of a wall-clock limit: the agent can run for
        # hours if it's actively calling tools / receiving stream tokens,
        # but a hung API call or stuck tool with no activity for the
        # configured duration is caught and killed.  (#4815)
        #
        # Config: agent.gateway_timeout in config.yaml, or
        # HERMES_AGENT_TIMEOUT env var (env var takes precedence).
        # Default 1800s (30 min inactivity).  0 = unlimited.
        _agent_timeout_raw = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        _agent_timeout = _agent_timeout_raw if _agent_timeout_raw > 0 else None
        _agent_warning_raw = _float_env("HERMES_AGENT_TIMEOUT_WARNING", 900)
        _agent_warning = _agent_warning_raw if _agent_warning_raw > 0 else None
        _warning_fired = False
        _executor_task = asyncio.ensure_future(
            runner._run_in_executor_with_context(run_sync)
        )

        _inactivity_timeout = False
        _POLL_INTERVAL = 5.0

        if _agent_timeout is None:
            # Unlimited — still poll periodically for backup interrupt
            # detection in case monitor_for_interrupt() silently died.
            response = None
            while True:
                done, _ = await asyncio.wait(
                    {_executor_task}, timeout=_POLL_INTERVAL
                )
                if done:
                    response = _executor_task.result()
                    break
                # Backup interrupt check: if the monitor task died or
                # missed the interrupt, catch it here.
                if not _interrupt_detected.is_set() and session_key:
                    _backup_adapter = runner._adapter_for_source(source)
                    _backup_agent = agent_holder[0]
                    if (_backup_adapter and _backup_agent
                            and hasattr(_backup_adapter, 'has_pending_interrupt')
                            and _backup_adapter.has_pending_interrupt(session_key)):
                        _bp_event = _backup_adapter._pending_messages.get(session_key)
                        _bp_text = _bp_event.text if _bp_event else None
                        if _bp_event is not None:
                            _bp_media_urls = getattr(_bp_event, "media_urls", None) or []
                            if runner._pending_event_audio_paths(_bp_event):
                                _bp_text, _ = await runner._transcribe_and_echo_pending_voice(
                                    _bp_event,
                                    _backup_adapter,
                                    source,
                                    _bp_text or "",
                                    log_context="Voice-backup-interrupt",
                                    metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                )
                            elif not _bp_text and _bp_media_urls:
                                _bp_text = _build_media_placeholder(_bp_event)
                        logger.info(
                            "Backup interrupt detected for session %s "
                            "(monitor task state: %s)",
                            session_key,
                            "done" if interrupt_monitor.done() else "running",
                        )
                        _backup_agent.interrupt(_bp_text)
                        _interrupt_detected.set()
        else:
            # Poll loop: check the agent's built-in activity tracker
            # (updated by _touch_activity() on every tool call, API
            # call, and stream delta) every few seconds.
            response = None
            while True:
                done, _ = await asyncio.wait(
                    {_executor_task}, timeout=_POLL_INTERVAL
                )
                if done:
                    response = _executor_task.result()
                    break
                # Agent still running — check inactivity.
                _agent_ref = agent_holder[0]
                _idle_secs = 0.0
                if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                    try:
                        _act = _agent_ref.get_activity_summary()
                        _idle_secs = _act.get("seconds_since_activity", 0.0)
                    except Exception:
                        pass
                # Staged warning: fire once before escalating to full timeout.
                if (not _warning_fired and _agent_warning is not None
                        and _idle_secs >= _agent_warning):
                    _warning_fired = True
                    _warn_adapter = runner._adapter_for_source(source)
                    if _warn_adapter:
                        _elapsed_warn = int(_agent_warning // 60) or 1
                        _remaining_mins = int((_agent_timeout - _agent_warning) // 60) or 1
                        try:
                            await _warn_adapter.send(
                                source.chat_id,
                                f"⚠️ No activity for {_elapsed_warn} min. "
                                f"If the agent does not respond soon, it will "
                                f"be timed out in {_remaining_mins} min. "
                                f"You can continue waiting or use /reset.",
                                metadata=_status_thread_metadata,
                            )
                        except Exception as _warn_err:
                            logger.debug("Inactivity warning send error: %s", _warn_err)
                if _idle_secs >= _agent_timeout:
                    _inactivity_timeout = True
                    break
                # Backup interrupt check (same as unlimited path).
                if not _interrupt_detected.is_set() and session_key:
                    _backup_adapter = runner._adapter_for_source(source)
                    _backup_agent = agent_holder[0]
                    if (_backup_adapter and _backup_agent
                            and hasattr(_backup_adapter, 'has_pending_interrupt')
                            and _backup_adapter.has_pending_interrupt(session_key)):
                        _bp_event = _backup_adapter._pending_messages.get(session_key)
                        _bp_text = _bp_event.text if _bp_event else None
                        if _bp_event is not None:
                            _bp_media_urls = getattr(_bp_event, "media_urls", None) or []
                            if runner._pending_event_audio_paths(_bp_event):
                                _bp_text, _ = await runner._transcribe_and_echo_pending_voice(
                                    _bp_event,
                                    _backup_adapter,
                                    source,
                                    _bp_text or "",
                                    log_context="Voice-backup-interrupt",
                                    metadata={"thread_id": source.thread_id} if source.thread_id else None,
                                )
                            elif not _bp_text and _bp_media_urls:
                                _bp_text = _build_media_placeholder(_bp_event)
                        logger.info(
                            "Backup interrupt detected for session %s "
                            "(monitor task state: %s)",
                            session_key,
                            "done" if interrupt_monitor.done() else "running",
                        )
                        _backup_agent.interrupt(_bp_text)
                        _interrupt_detected.set()

        if _inactivity_timeout:
            # Build a diagnostic summary from the agent's activity tracker.
            _timed_out_agent = agent_holder[0]
            _activity = {}
            if _timed_out_agent and hasattr(_timed_out_agent, "get_activity_summary"):
                try:
                    _activity = _timed_out_agent.get_activity_summary()
                except Exception:
                    pass

            _last_desc = _activity.get("last_activity_desc", "unknown")
            _secs_ago = _activity.get("seconds_since_activity", 0)
            _cur_tool = _activity.get("current_tool")
            _iter_n = _activity.get("api_call_count", 0)
            _iter_max = _activity.get("max_iterations", 0)

            logger.error(
                "Agent idle for %.0fs (timeout %.0fs) in session %s "
                "| last_activity=%s | iteration=%s/%s | tool=%s",
                _secs_ago, _agent_timeout, session_key,
                _last_desc, _iter_n, _iter_max,
                _cur_tool or "none",
            )

            # Interrupt the agent if it's still running so the thread
            # pool worker is freed.
            if _timed_out_agent and hasattr(_timed_out_agent, "interrupt"):
                _timed_out_agent.interrupt(_INTERRUPT_REASON_TIMEOUT)

            _timeout_mins = int(_agent_timeout // 60) or 1

            # Construct a user-facing message with diagnostic context.
            _diag_lines = [
                f"⏱️ Agent inactive for {_timeout_mins} min — no tool calls "
                f"or API responses."
            ]
            if _cur_tool:
                _diag_lines.append(
                    f"The agent appears stuck on tool `{_cur_tool}` "
                    f"({_secs_ago:.0f}s since last activity, "
                    f"iteration {_iter_n}/{_iter_max})."
                )
            else:
                _diag_lines.append(
                    f"Last activity: {_last_desc} ({_secs_ago:.0f}s ago, "
                    f"iteration {_iter_n}/{_iter_max}). "
                    "The agent may have been waiting on an API response."
                )
            _diag_lines.append(
                "To increase the limit, set agent.gateway_timeout in config.yaml "
                "(value in seconds, 0 = no limit) and restart the gateway.\n"
                "Try again, or use /reset to start fresh."
            )

            response = {
                "final_response": "\n".join(_diag_lines),
                "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                "api_calls": _iter_n,
                "tools": tools_holder[0] or [],
                "history_offset": 0,
                "failed": True,
            }

        # Track fallback model state: if the agent switched to a
        # fallback model during this run, persist it so /model shows
        # the actually-active model instead of the config default.
        # Skip eviction when the run failed — evicting a failed agent
        # forces MCP reinit on the next message for no benefit (the
        # same error will recur).  This was the root cause of #7130:
        # a bad model ID triggered fallback → eviction → recreation →
        # MCP reinit → same 400 → loop, burning 91% CPU for hours.
        _agent = agent_holder[0]
        _result_for_fb = result_holder[0]
        _run_failed = _result_for_fb.get("failed") if _result_for_fb else False
        # Skip eviction when the run failed — see #7130 (fallback →
        # eviction → MCP reinit loop).  Decision is unified in
        # _should_evict_cached_agent_after_turn (normalize + intentional
        # switch + pinned fallback).
        if not _run_failed and runner._should_evict_cached_agent_after_turn(
            _agent,
            _resolve_gateway_model(),
            session_key=session_key,
        ):
            runner._evict_cached_agent(session_key)

        # Check if we were interrupted OR have a queued message (/queue).
        result = result_holder[0]
        adapter = runner._adapter_for_source(source)

        # Get pending message from adapter.
        # Use session_key (not source.chat_id) to match adapter's storage keys.
        pending_event = None
        pending = None
        if result and adapter and session_key:
            pending_event = _dequeue_pending_event(adapter, session_key)
            # /queue overflow: after consuming the adapter's "next-up"
            # slot, promote the next queued event into it so the
            # recursive run's drain will see it.  This keeps the slot
            # occupied for the full FIFO chain, which (a) preserves
            # order, and (b) causes any mid-chain /queue to correctly
            # route to overflow rather than jumping the queue.
            pending_event = runner._promote_queued_event(session_key, adapter, pending_event)
            if result.get("interrupted") and not pending_event and result.get("interrupt_message"):
                interrupt_message = result.get("interrupt_message")
                if _is_control_interrupt_message(interrupt_message):
                    logger.info(
                        "Ignoring control interrupt message for session %s: %s",
                        session_key or "?",
                        interrupt_message,
                    )
                else:
                    pending = interrupt_message
            elif pending_event:
                # Transcribe audio media on the dequeued event BEFORE it is
                # handed back as the next user turn, so queued/interrupting
                # voice messages drain with the real transcript instead of
                # a file-path placeholder. When configured, echo each
                # transcript back to the user in the same 🎙️ format as
                # fresh voice messages.
                _pending_text = pending_event.text or ""
                _media_urls = getattr(pending_event, "media_urls", None) or []
                if runner._pending_event_audio_paths(pending_event):
                    pending, _ = await runner._transcribe_and_echo_pending_voice(
                        pending_event,
                        adapter,
                        source,
                        _pending_text,
                        log_context="Voice-drain",
                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                    )
                    if not pending:
                        pending = _build_media_placeholder(pending_event)
                else:
                    pending = _pending_text or _build_media_placeholder(pending_event)
                if pending:
                    logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])

        # Leftover /steer: if a steer arrived after the last tool batch
        # (e.g. during the final API call), the agent couldn't inject it
        # and returned it in result["pending_steer"]. Deliver it as the
        # next user turn so it isn't silently dropped.
        if result and not pending and not pending_event:
            _leftover_steer = result.get("pending_steer")
            if _leftover_steer:
                pending = _leftover_steer
                logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])

        # Safety net: if the pending text is a slash command (e.g. "/stop",
        # "/new"), discard it — commands should never be passed to the agent
        # as user input.  The primary fix is in base.py (commands bypass the
        # active-session guard), but this catches edge cases where command
        # text leaks through the interrupt_message fallback.
        if pending and pending.strip().startswith("/"):
            _pending_parts = pending.strip().split(None, 1)
            _pending_cmd_word = _pending_parts[0][1:].lower() if _pending_parts else ""
            if _pending_cmd_word:
                try:
                    from hermes_cli.commands import resolve_command as _rc_pending
                    if _rc_pending(_pending_cmd_word):
                        logger.info(
                            "Discarding command '/%s' from pending queue — "
                            "commands must not be passed as agent input",
                            _pending_cmd_word,
                        )
                        pending_event = None
                        pending = None
                except Exception:
                    pass

        if runner._draining and (pending_event or pending):
            logger.info(
                "Discarding pending follow-up for session %s during gateway %s",
                session_key or "?",
                runner._status_action_label(),
            )
            pending_event = None
            pending = None

        if pending_event or pending:
            logger.debug("Processing pending message: '%s...'", pending[:40])

            # Clear the adapter's interrupt event so the next _run_agent call
            # doesn't immediately re-trigger the interrupt before the new agent
            # even makes its first API call (this was causing an infinite loop).
            if adapter and hasattr(adapter, '_active_sessions') and session_key and session_key in adapter._active_sessions:
                adapter._active_sessions[session_key].clear()

            # Cap recursion depth to prevent resource exhaustion when the
            # user sends multiple messages while the agent keeps failing. (#816)
            if _interrupt_depth >= runner._MAX_INTERRUPT_DEPTH:
                logger.warning(
                    "Interrupt recursion depth %d reached for session %s — "
                    "queueing message instead of recursing.",
                    _interrupt_depth, session_key,
                )
                adapter = runner._adapter_for_source(source)
                if adapter and pending_event:
                    merge_pending_message_event(adapter._pending_messages, session_key, pending_event)
                elif adapter and hasattr(adapter, 'queue_message'):
                    adapter.queue_message(session_key, pending)
                return result_holder[0] or {"final_response": response, "messages": history}

            was_interrupted = result.get("interrupted")
            if not was_interrupted:
                # Queued message after normal completion — deliver the first
                # response before processing the queued follow-up.
                # Skip if streaming already delivered it.
                _sc = stream_consumer_holder[0]
                if _sc and stream_task:
                    try:
                        await asyncio.wait_for(stream_task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        stream_task.cancel()
                        try:
                            await stream_task
                        except asyncio.CancelledError:
                            pass
                    except Exception as e:
                        logger.debug("Stream consumer wait before queued message failed: %s", e)
                _previewed = bool(result.get("response_previewed"))
                first_response = result.get("final_response", "")
                _already_streamed = _stream_confirmed_final_delivery(
                    _sc,
                    first_response,
                    previewed=_previewed,
                )
                if first_response and not _already_streamed:
                    try:
                        logger.info(
                            "Queued follow-up for session %s: final stream delivery not confirmed; sending first response before continuing.",
                            session_key or "?",
                        )
                        await adapter.send(
                            source.chat_id,
                            first_response,
                            metadata=_status_thread_metadata,
                        )
                    except Exception as e:
                        logger.warning("Failed to send first response before queued message: %s", e)
                elif first_response:
                    logger.info(
                        "Queued follow-up for session %s: skipping resend because final streamed delivery was confirmed.",
                        session_key or "?",
                    )
                # Release deferred bg-review notifications now that the
                # first response has been delivered.  Pop from the
                # adapter's callback dict (prevents double-fire in
                # base.py's finally block) and call it.
                if getattr(type(adapter), "pop_post_delivery_callback", None) is not None:
                    _bg_cb = adapter.pop_post_delivery_callback(
                        session_key,
                        generation=run_generation,
                    )
                    if callable(_bg_cb):
                        try:
                            _bg_result = _bg_cb()
                            if inspect.isawaitable(_bg_result):
                                await _bg_result
                        except Exception:
                            pass
                elif adapter and hasattr(adapter, "_post_delivery_callbacks"):
                    _bg_cb = adapter._post_delivery_callbacks.pop(session_key, None)
                    if callable(_bg_cb):
                        try:
                            _bg_result = _bg_cb()
                            if inspect.isawaitable(_bg_result):
                                await _bg_result
                        except Exception:
                            pass
            # else: interrupted — discard the interrupted response ("Operation
            # interrupted." is just noise; the user already knows they sent a
            # new message).

            updated_history = result.get("messages", history)
            next_source = source
            next_message = pending
            next_message_id = None
            next_channel_prompt = None
            next_session_key = session_key
            if pending_event is not None:
                next_source = getattr(pending_event, "source", None) or source
                if runner._is_goal_continuation_event(pending_event) and not runner._goal_still_active_for_session(session_id):
                    logger.info(
                        "Discarding stale goal continuation for session %s — goal is no longer active",
                        session_key or "?",
                    )
                    return result
                # Resolve the follow-up's session key BEFORE preparing the
                # inbound text: _prepare_inbound_message_text buffers native
                # image paths under the key it is given, and the recursive
                # _run_agent below consumes them under next_session_key.
                # The write and consume keys must match or the images drop.
                try:
                    next_session_key = runner._session_key_for_source(next_source)
                except Exception:
                    logger.debug(
                        "Queued follow-up session-key resolution failed; reusing %s",
                        session_key or "?",
                        exc_info=True,
                    )
                next_message = await runner._prepare_profile_scoped_inbound_message_text(
                    event=pending_event,
                    source=next_source,
                    history=updated_history,
                    session_key=next_session_key,
                )
                if next_message is None:
                    return result
                next_message_id = runner._reply_anchor_for_event(pending_event)
                next_channel_prompt = getattr(pending_event, "channel_prompt", None)

            # Restart typing indicator so the user sees activity while
            # the follow-up turn runs.  The outer _process_message_background
            # typing task is still alive but may be stale.
            _followup_adapter = runner._adapter_for_source(source)
            if _followup_adapter:
                try:
                    await _followup_adapter.send_typing(
                        source.chat_id,
                        metadata=_status_thread_metadata,
                    )
                except Exception:
                    pass

            # Re-baseline the cached agent's message_count snapshot before
            # recursing into the in-band queued (/queue) follow-up turn.
            # The first turn has completed and flushed its own user +
            # assistant rows to the SessionDB, so the cross-process
            # coherence guard (#45966) — which this recursive _run_agent
            # call re-enters — would otherwise see the grown on-disk count
            # against the stale build-time snapshot and rebuild the agent
            # on THIS process's OWN writes, destroying the prompt-cache
            # prefix #46237 was merged to preserve.  The existing
            # re-baseline in _handle_message_with_agent only runs after the
            # whole _run_agent chain unwinds — too late for the in-band
            # follow-up.  Use the same (session_key, session_id) the
            # recursive call runs under so the snapshot matches exactly
            # what the follow-up's guard will consult.  Fail-safe in helper.
            await runner._refresh_agent_cache_message_count(session_key, session_id)

            followup_result = await runner._run_agent(
                message=next_message,
                context_prompt=context_prompt,
                history=updated_history,
                source=next_source,
                session_id=session_id,
                session_key=next_session_key,
                run_generation=run_generation,
                _interrupt_depth=_interrupt_depth + 1,
                event_message_id=next_message_id,
                channel_prompt=next_channel_prompt,
            )
            return _preserve_queued_followup_history_offset(result, followup_result)
    finally:
        # Stop progress sender, interrupt monitor, and notification task
        if progress_task:
            progress_task.cancel()
        if log_task:
            log_task.cancel()
        interrupt_monitor.cancel()
        _notify_task.cancel()

        # Wait for stream consumer to finish its final edit
        if stream_task:
            # If the agent never created a stream consumer (e.g. non-
            # streaming code path, or a test stub returning synchronously)
            # there is nothing to flush — cancel immediately instead of
            # waiting out the 5s timeout on a task that's just polling for
            # a consumer that will never arrive.  This was a 5-second
            # cost per non-streaming test run.
            _has_stream_consumer = (
                stream_consumer_holder
                and stream_consumer_holder[0] is not None
            )
            if not _has_stream_consumer:
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass
            else:
                try:
                    await asyncio.wait_for(stream_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass

        # Clean up tracking
        tracking_task.cancel()
        if session_key:
            # Only release the slot if this run's generation still owns
            # it.  A /stop or /new that bumped the generation while we
            # were unwinding has already installed its own state; this
            # guard prevents an old run from clobbering it on the way
            # out.
            runner._release_running_agent_state(
                session_key, run_generation=run_generation
            )
        if runner._draining:
            runner._update_runtime_status("draining")

        # Wait for cancelled tasks
        for task in [progress_task, log_task, interrupt_monitor, tracking_task, _notify_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # If streaming already delivered the response, mark it so the
    # caller's send() is skipped (avoiding duplicate messages).
    # BUT: never suppress delivery when the agent failed — the error
    # message is new content the user hasn't seen, and it must reach
    # them even if streaming had sent earlier partial output.
    #
    # Also never suppress when the final response is "(empty)" — this
    # means the model failed to produce content after tool calls (common
    # with mimo-v2-pro, GLM-5, etc.).  The stream consumer may have
    # sent intermediate text ("Let me search for that…") alongside the
    # tool call, setting already_sent=True, but that text is NOT the
    # final answer.  Suppressing delivery here leaves the user staring
    # at silence.  (#10xxx — "agent stops after web search")
    _sc = stream_consumer_holder[0]
    if isinstance(response, dict) and not response.get("failed"):
        _final = response.get("final_response") or ""
        _is_empty_sentinel = not _final or _final == "(empty)"
        # response_previewed means the interim_assistant_callback already
        # saw the final text, but only suppress the normal send if that
        # exact final text was delivered. Unrelated commentary/progress
        # must not be mistaken for the final response (#14238).
        _previewed = bool(response.get("response_previewed"))
        _content_delivered = bool(
            _sc and getattr(_sc, "final_content_delivered", False)
        )
        # Plugin hooks (e.g. transform_llm_output) may have appended content
        # after streaming finished — when the response was transformed, always
        # send the final version so the appended content reaches the client.
        _transformed = bool(response.get("response_transformed"))
        # Only suppress the normal send when the actual final reply reached
        # the user: the stream consumer streamed it (final_response_sent /
        # final_content_delivered), or the interim preview delivered that
        # *exact* final text. Unrelated commentary/progress shown during a
        # compression/session split must not be mistaken for the final
        # response (#14238).
        _streamed = _stream_confirmed_final_delivery(
            _sc,
            _final,
            previewed=_previewed,
        )
        if not _is_empty_sentinel and not _transformed and (_streamed or _content_delivered):
            logger.info(
                "Suppressing normal final send for session %s: final delivery already confirmed (streamed=%s previewed=%s content_delivered=%s).",
                session_key or "?",
                _streamed,
                _previewed,
                _content_delivered,
            )
            response["already_sent"] = True
        elif not _is_empty_sentinel and _transformed and _sc is not None:
            # Plugin hooks transformed the response after streaming — edit the
            # existing streamed message instead of sending a duplicate.
            _sc_msg_id = _sc.message_id
            if _sc_msg_id:
                try:
                    await _sc.adapter.edit_message(
                        chat_id=source.chat_id,
                        message_id=_sc_msg_id,
                        content=response["final_response"],
                        finalize=True,
                    )
                    response["already_sent"] = True
                    logger.info(
                        "Edited streamed message %s for session %s to include plugin-transformed content.",
                        _sc_msg_id, session_key or "?",
                    )
                except Exception as _edit_err:
                    logger.warning(
                        "Failed to edit streamed message for session %s: %s",
                        session_key or "?", _edit_err,
                    )

    # Schedule deletion of tracked temporary progress bubbles after the
    # final response lands. Failed runs skip this so bubbles remain as
    # breadcrumbs for the user to see what work happened. Only fires on
    # adapters that support ``delete_message`` (see init above); failures
    # are swallowed — deletion is best-effort.
    if (
        _cleanup_progress
        and _cleanup_adapter is not None
        and _cleanup_msg_ids
        and session_key
        and isinstance(response, dict)
        and not response.get("failed")
        and hasattr(_cleanup_adapter, "register_post_delivery_callback")
    ):
        _ids_snapshot = list(_cleanup_msg_ids)
        _chat_id_snapshot = source.chat_id
        _adapter_snapshot = _cleanup_adapter
        _loop_snapshot = asyncio.get_running_loop()

        def _cleanup_temp_bubbles() -> None:
            async def _delete_all() -> None:
                for _mid in _ids_snapshot:
                    try:
                        await _adapter_snapshot.delete_message(
                            _chat_id_snapshot, _mid
                        )
                    except Exception:
                        pass
            try:
                safe_schedule_threadsafe(
                    _delete_all(), _loop_snapshot,
                    logger=logger,
                    log_message="Temp bubble cleanup scheduling error",
                )
            except Exception:
                pass

        try:
            _cleanup_adapter.register_post_delivery_callback(
                session_key,
                _cleanup_temp_bubbles,
                generation=run_generation,
            )
        except Exception as _rpe:
            logger.debug("Post-delivery cleanup registration failed: %s", _rpe)

    return response
