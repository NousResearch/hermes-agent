"""
Gateway runner - entry point for messaging platform integrations.

This module provides:
- start_gateway(): Start all configured platform adapters
- GatewayRunner: Main class managing the gateway lifecycle

Usage:
    # Start the gateway
    python -m gateway.run
    
    # Or from CLI
    python cli.py --gateway
"""

import asyncio
import json
import logging
import os
import re
import shlex
import sys
import signal
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

# ---------------------------------------------------------------------------
# SSL certificate auto-detection for NixOS and other non-standard systems.
# Must run BEFORE any HTTP library (discord, aiohttp, etc.) is imported.
# ---------------------------------------------------------------------------
def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE if the system doesn't expose CA certs to Python."""
    if "SSL_CERT_FILE" in os.environ:
        return  # user already configured it

    import ssl

    # 1. Python's compiled-in defaults
    paths = ssl.get_default_verify_paths()
    for candidate in (paths.cafile, paths.openssl_cafile):
        if candidate and os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

    # 2. certifi (ships its own Mozilla bundle)
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        return
    except ImportError:
        pass

    # 3. Common distro / macOS locations
    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",               # Debian/Ubuntu/Gentoo
        "/etc/pki/tls/certs/ca-bundle.crt",                 # RHEL/CentOS 7
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", # RHEL/CentOS 8+
        "/etc/ssl/ca-bundle.pem",                            # SUSE/OpenSUSE
        "/etc/ssl/cert.pem",                                 # Alpine / macOS
        "/etc/pki/tls/cert.pem",                             # Fedora
        "/usr/local/etc/openssl@1.1/cert.pem",               # macOS Homebrew Intel
        "/opt/homebrew/etc/openssl@1.1/cert.pem",            # macOS Homebrew ARM
    ):
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

_ensure_ssl_certs()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Resolve Hermes home directory (respects HERMES_HOME override)
from hermes_constants import get_hermes_home
from utils import atomic_yaml_write
_hermes_home = get_hermes_home()

# Load environment variables from ~/.hermes/.env first.
# User-managed env files should override stale shell exports on restart.
from dotenv import load_dotenv  # backward-compat for tests that monkeypatch this symbol
from hermes_cli.env_loader import load_hermes_dotenv
_env_path = _hermes_home / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).resolve().parents[1] / '.env')

# Bridge config.yaml values into the environment so os.getenv() picks them up.
# config.yaml is authoritative for terminal settings — overrides .env.
_config_path = _hermes_home / 'config.yaml'
if _config_path.exists():
    try:
        import yaml as _yaml
        with open(_config_path, encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f) or {}
        # Expand ${ENV_VAR} references before bridging to env vars.
        from hermes_cli.config import _expand_env_vars, bridge_auxiliary_config_to_env
        _cfg = _expand_env_vars(_cfg)
        # Top-level simple values (fallback only — don't override .env)
        for _key, _val in _cfg.items():
            if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
                os.environ[_key] = str(_val)
        # Terminal config is nested — bridge to TERMINAL_* env vars.
        # config.yaml overrides .env for these since it's the documented config path.
        _terminal_cfg = _cfg.get("terminal", {})
        if _terminal_cfg and isinstance(_terminal_cfg, dict):
            _terminal_env_map = {
                "backend": "TERMINAL_ENV",
                "cwd": "TERMINAL_CWD",
                "timeout": "TERMINAL_TIMEOUT",
                "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
                "docker_image": "TERMINAL_DOCKER_IMAGE",
                "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
                "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
                "modal_image": "TERMINAL_MODAL_IMAGE",
                "daytona_image": "TERMINAL_DAYTONA_IMAGE",
                "ssh_host": "TERMINAL_SSH_HOST",
                "ssh_user": "TERMINAL_SSH_USER",
                "ssh_port": "TERMINAL_SSH_PORT",
                "ssh_key": "TERMINAL_SSH_KEY",
                "container_cpu": "TERMINAL_CONTAINER_CPU",
                "container_memory": "TERMINAL_CONTAINER_MEMORY",
                "container_disk": "TERMINAL_CONTAINER_DISK",
                "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
                "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
                "sandbox_dir": "TERMINAL_SANDBOX_DIR",
                "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
            }
            for _cfg_key, _env_var in _terminal_env_map.items():
                if _cfg_key in _terminal_cfg:
                    _val = _terminal_cfg[_cfg_key]
                    if isinstance(_val, list):
                        os.environ[_env_var] = json.dumps(_val)
                    else:
                        os.environ[_env_var] = str(_val)
        # Compression config is read directly from config.yaml by run_agent.py
        # and auxiliary_client.py — no env var bridging needed.
        # Auxiliary model/direct-endpoint overrides (vision, web_extract).
        # Each task has provider/model/base_url/api_key; bridge non-default values to env vars.
        _auxiliary_cfg = _cfg.get("auxiliary", {})
        bridge_auxiliary_config_to_env(_auxiliary_cfg)
        _agent_cfg = _cfg.get("agent", {})
        if _agent_cfg and isinstance(_agent_cfg, dict):
            if "max_turns" in _agent_cfg:
                os.environ["HERMES_MAX_ITERATIONS"] = str(_agent_cfg["max_turns"])
            # Bridge agent.gateway_timeout → HERMES_AGENT_TIMEOUT env var.
            # Env var from .env takes precedence (already in os.environ).
            if "gateway_timeout" in _agent_cfg and "HERMES_AGENT_TIMEOUT" not in os.environ:
                os.environ["HERMES_AGENT_TIMEOUT"] = str(_agent_cfg["gateway_timeout"])
            # Bridge agent.stream_stale_timeout → HERMES_STREAM_STALE_TIMEOUT.
            # Controls how long a streaming provider may stay silent before
            # the gateway tears down the connection and retries/fails over.
            if "stream_stale_timeout" in _agent_cfg and "HERMES_STREAM_STALE_TIMEOUT" not in os.environ:
                os.environ["HERMES_STREAM_STALE_TIMEOUT"] = str(_agent_cfg["stream_stale_timeout"])
        # Timezone: bridge config.yaml → HERMES_TIMEZONE env var.
        # HERMES_TIMEZONE from .env takes precedence (already in os.environ).
        _tz_cfg = _cfg.get("timezone", "")
        if _tz_cfg and isinstance(_tz_cfg, str) and "HERMES_TIMEZONE" not in os.environ:
            os.environ["HERMES_TIMEZONE"] = _tz_cfg.strip()
        # Security settings
        _security_cfg = _cfg.get("security", {})
        if isinstance(_security_cfg, dict):
            _redact = _security_cfg.get("redact_secrets")
            if _redact is not None:
                os.environ["HERMES_REDACT_SECRETS"] = str(_redact).lower()
    except Exception:
        pass  # Non-fatal; gateway can still run with .env values

# Validate config structure early — log warnings so gateway operators see problems
try:
    from hermes_cli.config import print_config_warnings
    print_config_warnings()
except Exception:
    pass

# Gateway runs in quiet mode - suppress debug output and use cwd directly (no temp dirs)
os.environ["HERMES_QUIET"] = "1"

# Enable interactive exec approval for dangerous commands on messaging platforms
os.environ["HERMES_EXEC_ASK"] = "1"

# Set terminal working directory for messaging platforms.
# If the user set an explicit path in config.yaml (not "." or "auto"),
# respect it. Otherwise use MESSAGING_CWD or default to home directory.
_configured_cwd = os.environ.get("TERMINAL_CWD", "")
if not _configured_cwd or _configured_cwd in (".", "auto", "cwd"):
    messaging_cwd = os.getenv("MESSAGING_CWD") or str(Path.home())
    os.environ["TERMINAL_CWD"] = messaging_cwd

from gateway.config import (
    Platform,
    GatewayConfig,
    load_gateway_config,
    _coerce_list,
)
from gateway.agent_execution_service import (
    create_gateway_agent,
    execute_gateway_sync_turn as shared_execute_gateway_sync_turn,
)
from gateway.agent_followup_runtime_service import (
    process_gateway_pending_followup as shared_process_gateway_pending_followup,
)
from gateway.agent_lifecycle_runtime_service import (
    cleanup_gateway_agent_runtime_tasks as shared_cleanup_gateway_agent_runtime_tasks,
    mark_gateway_streaming_delivery_state as shared_mark_gateway_streaming_delivery_state,
    resolve_gateway_effective_model_state as shared_resolve_gateway_effective_model_state,
    start_gateway_agent_runtime_tasks as shared_start_gateway_agent_runtime_tasks,
    wait_for_gateway_agent_result as shared_wait_for_gateway_agent_result,
)
from gateway.agent_progress_runtime_service import (
    build_gateway_progress_runtime as shared_build_gateway_progress_runtime,
)
from gateway.agent_turn_runtime_service import (
    prepare_gateway_cached_turn_agent as shared_prepare_gateway_cached_turn_agent,
)
from gateway.agent_runtime import (
    agent_config_signature as shared_agent_config_signature,
    load_ephemeral_system_prompt as shared_load_ephemeral_system_prompt,
    load_fallback_model as shared_load_fallback_model,
    load_gateway_user_config as shared_load_gateway_user_config,
    load_prefill_messages as shared_load_prefill_messages,
    load_provider_routing as shared_load_provider_routing,
    load_reasoning_config as shared_load_reasoning_config,
    load_show_reasoning as shared_load_show_reasoning,
    load_smart_model_routing as shared_load_smart_model_routing,
    platform_config_key as shared_platform_config_key,
    prepare_gateway_sync_turn_runtime as shared_prepare_gateway_sync_turn_runtime,
    resolve_gateway_model as shared_resolve_gateway_model,
    resolve_runtime_agent_kwargs as shared_resolve_runtime_agent_kwargs,
    resolve_turn_agent_config as shared_resolve_turn_agent_config,
)
from gateway.background_delivery_service import (
    background_completion_should_stay_silent as shared_background_completion_should_stay_silent,
    background_job_delivery_poller as shared_background_job_delivery_poller,
    build_background_delivery_header as shared_build_background_delivery_header,
    deliver_background_job_updates_once as shared_deliver_background_job_updates_once,
    recover_stale_background_jobs_once as shared_recover_stale_background_jobs_once,
    sanitize_background_visible_text as shared_sanitize_background_visible_text,
)
from gateway.background_job_start_service import (
    start_background_job as shared_start_background_job,
)
from gateway.auto_background_runtime_service import (
    format_auto_background_ack as shared_format_auto_background_ack,
    resolve_auto_background_dispatch as shared_resolve_auto_background_dispatch,
    resolve_employee_background_dispatch as shared_resolve_employee_background_dispatch,
)
from gateway.attachment_message_runtime_service import (
    collect_audio_paths as shared_collect_audio_paths,
    has_visible_image_attachments as shared_has_visible_image_attachments,
    prepend_document_context_notes as shared_prepend_document_context_notes,
)
from gateway.auto_vision_runtime_service import (
    auto_vision_cache_key as shared_auto_vision_cache_key,
    auto_vision_cooldown_remaining as shared_auto_vision_cooldown_remaining,
    auto_vision_degraded_note as shared_auto_vision_degraded_note,
    classify_auto_vision_failure as shared_classify_auto_vision_failure,
    clear_auto_vision_cooldown as shared_clear_auto_vision_cooldown,
    ensure_auto_vision_state as shared_ensure_auto_vision_state,
    get_auto_vision_cache_entry as shared_get_auto_vision_cache_entry,
    image_vision_inputs_from_event as shared_image_vision_inputs_from_event,
    mark_auto_vision_cooldown as shared_mark_auto_vision_cooldown,
    media_ref_suffix as shared_media_ref_suffix,
    prune_auto_vision_state as shared_prune_auto_vision_state,
    should_prefer_remote_auto_vision_source as shared_should_prefer_remote_auto_vision_source,
    should_skip_auto_vision_media as shared_should_skip_auto_vision_media,
)
from gateway.direct_control_router import (
    DIRECT_CONTROL_ROUTER_METHODS as SHARED_DIRECT_CONTROL_ROUTER_METHODS,
    DirectControlRouter,
)
from gateway.direct_shortcuts import run_direct_shortcut_handlers
from gateway.direct_shortcut_runtime_service import (
    get_direct_control_router as shared_get_direct_control_router,
    try_handle_direct_gateway_shortcuts as shared_try_handle_direct_gateway_shortcuts,
)
from gateway.employee_routes import get_employee_routes
from gateway.runtime_status_service import (
    build_runtime_status_summary as shared_build_runtime_status_summary,
    render_status_command as shared_render_status_command,
)
from gateway.runtime_shortcuts_service import (
    format_background_job_short_status as shared_format_background_job_short_status,
    format_running_session_short_status as shared_format_running_session_short_status,
    try_handle_background_job_status_shortcut as shared_try_handle_background_job_status_shortcut,
    try_handle_runtime_status_shortcut as shared_try_handle_runtime_status_shortcut,
)
from gateway.group_control_intents import (
    looks_like_group_listen_disable_request,
    looks_like_group_listen_enable_request,
    looks_like_group_runtime_status_query as looks_like_shared_group_runtime_status_query,
)
from gateway.group_runtime_status_service import (
    unique_report_targets as shared_unique_report_targets,
    worker_report_targets as shared_worker_report_targets,
)
from gateway.session_hygiene_runtime_service import (
    maybe_auto_compress_session_history as shared_maybe_auto_compress_session_history,
)
from gateway.message_preprocessing_runtime_service import (
    is_shared_thread_session as shared_is_shared_thread_session,
    prepend_reply_context_if_missing as shared_prepend_reply_context_if_missing,
    prepend_shared_thread_sender as shared_prepend_shared_thread_sender,
)
from gateway.agent_prelude_runtime_service import (
    append_discord_voice_channel_context as shared_append_discord_voice_channel_context,
    build_agent_start_hook_context as shared_build_agent_start_hook_context,
    run_gateway_agent_prelude as shared_run_gateway_agent_prelude,
)
from gateway.agent_completion_runtime_service import (
    prepare_gateway_agent_completion as shared_prepare_gateway_agent_completion,
    drain_pending_process_watchers as shared_drain_pending_process_watchers,
    stop_gateway_typing_indicator as shared_stop_gateway_typing_indicator,
)
from gateway.agent_response_runtime_service import (
    build_gateway_exception_response as shared_build_gateway_exception_response,
)
from gateway.agent_delivery_runtime_service import (
    finalize_gateway_agent_delivery as shared_finalize_gateway_agent_delivery,
)
from gateway.transcript_persistence_runtime_service import (
    persist_gateway_agent_transcript as shared_persist_gateway_agent_transcript,
)
from gateway.context_reference_runtime_service import (
    GatewayContextReferenceOutcome,
    expand_gateway_context_references as shared_expand_gateway_context_references,
)
from gateway.onboarding_runtime_service import (
    append_first_message_onboarding_note as shared_append_first_message_onboarding_note,
    build_home_channel_prompt as shared_build_home_channel_prompt,
    home_channel_env_var_name as shared_home_channel_env_var_name,
    should_prompt_for_home_channel as shared_should_prompt_for_home_channel,
)
from gateway.shared_group_history_runtime_service import (
    DEFAULT_SHARED_GROUP_VISIBLE_HISTORY_LIMIT as SHARED_DEFAULT_GROUP_VISIBLE_HISTORY_LIMIT,
    is_shared_group_internal_artifact as shared_is_shared_group_internal_artifact,
    prepare_history_for_agent as shared_prepare_history_for_agent,
    simplify_shared_group_history_for_agent as shared_simplify_shared_group_history_for_agent,
)
from gateway.group_target_intents import (
    extract_qq_group_target,
    extract_weixin_group_target,
)
from gateway.qq_group_archive import QqGroupArchiveStore
from gateway.qq_group_policies import list_group_policies
from gateway.qq_intel_assignments import list_intel_workers
from gateway.qq_intents import (
    _QQ_BACKGROUND_STATUS_QUERY_TERMS,
    _QQ_GROUP_REQUEST_HINT_TERMS,
    _QQ_JOINED_GROUP_LIST_TERMS,
    _QQ_MEDIA_PLACEHOLDER_MARKERS,
    _QQ_RUNTIME_STATUS_QUERY_TERMS,
    _QQ_RUNTIME_STATUS_SHORT_TERMS,
    _QQ_VISIBLE_NAME_ALIASES,
    _looks_like_qq_active_session_inline_candidate,
    _looks_like_qq_background_status_query,
    _looks_like_qq_group_moderation_candidate,
    _looks_like_qq_group_request_text,
    _looks_like_qq_joined_group_list_query,
    _looks_like_qq_media_message,
    _looks_like_qq_runtime_short_query,
    _looks_like_qq_runtime_status_query,
    _looks_like_qq_social_policy_candidate,
    _looks_like_qq_social_request_list_query,
    _qq_group_has_visible_bot_address,
)
from gateway.session import (
    SessionEntry,
    SessionStore,
    SessionSource,
    SessionContext,
    build_session_context,
    build_session_context_prompt,
    build_session_key,
)
from gateway.background_jobs import (
    BackgroundJobStore,
    background_job_chat_key as durable_background_job_chat_key,
    background_job_scope_key as durable_background_job_scope_key,
    launch_background_worker,
    stop_background_worker,
)
from gateway.delivery import DeliveryRouter
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType


_DIRECT_CONTROL_ROUTER_METHODS = SHARED_DIRECT_CONTROL_ROUTER_METHODS


def _normalize_whatsapp_identifier(value: str) -> str:
    """Strip WhatsApp JID/LID syntax down to its stable numeric identifier."""
    return (
        str(value or "")
        .strip()
        .replace("+", "", 1)
        .split(":", 1)[0]
        .split("@", 1)[0]
    )


def _expand_whatsapp_auth_aliases(identifier: str) -> set:
    """Resolve WhatsApp phone/LID aliases using bridge session mapping files."""
    normalized = _normalize_whatsapp_identifier(identifier)
    if not normalized:
        return set()

    session_dir = _hermes_home / "whatsapp" / "session"
    resolved = set()
    queue = [normalized]

    while queue:
        current = queue.pop(0)
        if not current or current in resolved:
            continue

        resolved.add(current)
        for suffix in ("", "_reverse"):
            mapping_path = session_dir / f"lid-mapping-{current}{suffix}.json"
            if not mapping_path.exists():
                continue
            try:
                mapped = _normalize_whatsapp_identifier(
                    json.loads(mapping_path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
            if mapped and mapped not in resolved:
                queue.append(mapped)

    return resolved

logger = logging.getLogger(__name__)

# Sentinel placed into _running_agents immediately when a session starts
# processing, *before* any await.  Prevents a second message for the same
# session from bypassing the "already running" guard during the async gap
# between the guard check and actual agent creation.
_AGENT_PENDING_SENTINEL = object()
_SHARED_GROUP_VISIBLE_HISTORY_LIMIT = SHARED_DEFAULT_GROUP_VISIBLE_HISTORY_LIMIT
_AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS = 8.0
_AUTO_VISION_ANALYSIS_TIMEOUT_CAP_SECONDS = 45.0
_AUTO_VISION_ANALYSIS_FAILURE_COOLDOWN_SECONDS = 300.0
_AUTO_VISION_TRANSIENT_COOLDOWN_SECONDS = 5.0
_AUTO_VISION_INLINE_WAIT_SECONDS = 0.75
_AUTO_VISION_GROUP_INLINE_WAIT_SECONDS = 0.25
_AUTO_VISION_IMAGE_ONLY_INLINE_WAIT_SECONDS = 8.0
_AUTO_VISION_CACHE_TTL_SECONDS = 3600.0
_AUTO_VISION_MAX_INFLIGHT_TASKS = 4
_AUTO_VISION_MAX_CACHE_ENTRIES = 256
def _is_shared_group_internal_artifact(content: Any) -> bool:
    return shared_is_shared_group_internal_artifact(content)


def _should_forward_agent_status(
    source: SessionSource,
    event_type: str,
    message: str,
) -> bool:
    """Return whether a run_agent status event should be shown to the user.

    QQ chats are especially sensitive to internal lifecycle chatter because the
    adapter operates in conversational contexts with high message volume.  The
    user-facing value of raw fallback/retry/context-pressure telemetry is low,
    and forwarding it into QQ groups creates visible spam that looks like the
    bot is malfunctioning instead of working.
    """
    del message  # Reserved for future content-aware filtering.

    if source.platform == Platform.QQ_NAPCAT and event_type in {
        "lifecycle",
        "context_pressure",
    }:
        return False
    return True


def _image_vision_inputs_from_event(event: MessageEvent) -> List[str]:
    return shared_image_vision_inputs_from_event(event)


def _should_prefer_remote_auto_vision_source(image_ref: str) -> bool:
    return shared_should_prefer_remote_auto_vision_source(image_ref)


def _should_skip_auto_vision_media(
    *,
    path: str,
    media_type: str,
    preferred_source: str = "",
) -> bool:
    return shared_should_skip_auto_vision_media(
        path=path,
        media_type=media_type,
        preferred_source=preferred_source,
    )


def _media_ref_suffix(ref: str) -> str:
    return shared_media_ref_suffix(ref)


def _simplify_shared_group_history_for_agent(
    history: List[Dict[str, Any]],
    *,
    visible_limit: int = _SHARED_GROUP_VISIBLE_HISTORY_LIMIT,
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper for shared-group history cleanup."""
    return shared_simplify_shared_group_history_for_agent(
        history,
        visible_limit=visible_limit,
    )


def _resolve_runtime_agent_kwargs() -> dict:
    """Resolve provider credentials for gateway-created AIAgent instances."""
    return shared_resolve_runtime_agent_kwargs()


def _is_shared_thread_session(
    *,
    source: SessionSource,
    thread_sessions_per_user: bool,
) -> bool:
    return shared_is_shared_thread_session(
        source=source,
        thread_sessions_per_user=thread_sessions_per_user,
    )


def _prepend_reply_context_if_missing(
    *,
    message_text: str,
    event: MessageEvent,
    history: List[Dict[str, Any]],
) -> str:
    return shared_prepend_reply_context_if_missing(
        message_text=message_text,
        reply_to_text=getattr(event, "reply_to_text", None),
        reply_to_message_id=getattr(event, "reply_to_message_id", None),
        history=history,
    )


def _prepend_shared_thread_sender(
    *,
    message_text: str,
    source: SessionSource,
    thread_sessions_per_user: bool,
) -> str:
    return shared_prepend_shared_thread_sender(
        message_text=message_text,
        user_name=source.user_name,
        shared_thread=_is_shared_thread_session(
            source=source,
            thread_sessions_per_user=thread_sessions_per_user,
        ),
    )


def _has_visible_image_attachments(attachments: List[Any]) -> bool:
    return shared_has_visible_image_attachments(attachments)


def _collect_audio_paths(
    attachments: List[Any],
    *,
    message_type: MessageType,
) -> List[str]:
    return shared_collect_audio_paths(
        attachments,
        message_type=message_type,
        voice_type=MessageType.VOICE,
        audio_type=MessageType.AUDIO,
    )


def _prepend_document_context_notes(
    message_text: str,
    *,
    attachments: List[Any],
    message_type: MessageType,
) -> str:
    return shared_prepend_document_context_notes(
        message_text,
        attachments=attachments,
        message_type=message_type,
        document_type=MessageType.DOCUMENT,
    )


def _append_first_message_onboarding_note(
    context_prompt: str,
    *,
    history: List[Dict[str, Any]],
    has_any_sessions: bool,
) -> str:
    return shared_append_first_message_onboarding_note(
        context_prompt,
        history=history,
        has_any_sessions=has_any_sessions,
    )


def _should_prompt_for_home_channel(
    *,
    history: List[Dict[str, Any]],
    platform: Optional[Platform],
) -> bool:
    env_key = shared_home_channel_env_var_name(platform)
    configured = bool(env_key and os.getenv(env_key))
    return shared_should_prompt_for_home_channel(
        history=history,
        platform=platform,
        home_channel_configured=configured,
    )


def _append_discord_voice_channel_context(
    context_prompt: str,
    *,
    source: SessionSource,
    guild_id: Optional[int],
    adapter: Any,
) -> str:
    return shared_append_discord_voice_channel_context(
        context_prompt,
        platform=source.platform,
        guild_id=guild_id,
        adapter=adapter,
    )


def _build_agent_start_hook_context(
    *,
    source: SessionSource,
    session_id: str,
    message_text: str,
) -> Dict[str, Any]:
    return shared_build_agent_start_hook_context(
        platform=source.platform,
        user_id=source.user_id,
        session_id=session_id,
        message_text=message_text,
    )


async def _expand_gateway_context_references(
    message_text: str,
    *,
    runner: Any,
    logger,
) -> GatewayContextReferenceOutcome:
    try:
        model = runner._model
        base_url = runner._base_url or ""
    except Exception as exc:
        logger.debug("@ context reference expansion failed: %s", exc)
        return GatewayContextReferenceOutcome(message_text=message_text)

    return await shared_expand_gateway_context_references(
        message_text,
        model=model,
        base_url=base_url,
        messaging_cwd=os.environ.get("MESSAGING_CWD"),
        logger=logger,
    )


def _load_gateway_flush_memory_store():
    """Load a live MemoryStore snapshot for background gateway flushes."""
    try:
        from hermes_cli.config import load_config as _load_config
        mem_cfg = (_load_config().get("memory") or {})
    except Exception:
        mem_cfg = {}

    from tools.memory_tool import MemoryStore

    store = MemoryStore(
        memory_char_limit=mem_cfg.get("memory_char_limit", 2200),
        user_char_limit=mem_cfg.get("user_char_limit", 1375),
    )
    store.load_from_disk()
    return store


def _extract_gateway_flush_tool_calls(response) -> list:
    """Return tool calls from an auxiliary chat response."""
    try:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return []
        message = getattr(choices[0], "message", None)
        if message is None:
            return []
        return list(getattr(message, "tool_calls", None) or [])
    except Exception:
        return []


def _execute_gateway_flush_tool_call(tool_call, *, memory_store) -> None:
    """Execute a single tool call emitted by the quiet gateway flush turn."""
    function = getattr(tool_call, "function", None)
    if function is None:
        return
    tool_name = str(getattr(function, "name", "") or "").strip()
    if not tool_name:
        return

    raw_args = getattr(function, "arguments", "{}")
    if isinstance(raw_args, str):
        args = json.loads(raw_args or "{}")
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}

    if tool_name == "memory":
        from tools.memory_tool import memory_tool as _memory_tool

        _memory_tool(
            action=args.get("action"),
            target=args.get("target", "memory"),
            content=args.get("content"),
            old_text=args.get("old_text"),
            store=memory_store,
        )
        return

    if tool_name == "skill_manage":
        from tools.skill_manager_tool import skill_manage as _skill_manage

        _skill_manage(
            action=args.get("action", ""),
            name=args.get("name", ""),
            content=args.get("content"),
            category=args.get("category"),
            file_path=args.get("file_path"),
            file_content=args.get("file_content"),
            old_string=args.get("old_string"),
            new_string=args.get("new_string"),
            replace_all=bool(args.get("replace_all", False)),
        )


def _build_media_placeholder(event) -> str:
    """Build a text placeholder for media-only events so they aren't dropped.

    When a photo/document is queued during active processing and later
    dequeued, only .text is extracted.  If the event has no caption,
    the media would be silently lost.  This builds a placeholder that
    the vision enrichment pipeline will replace with a real description.
    """
    parts = []
    media_urls = getattr(event, "media_urls", None) or []
    media_types = getattr(event, "media_types", None) or []
    for i, url in enumerate(media_urls):
        mtype = media_types[i] if i < len(media_types) else ""
        if mtype.startswith("image/") or getattr(event, "message_type", None) == MessageType.PHOTO:
            parts.append(f"[User sent an image: {url}]")
        elif mtype.startswith("audio/"):
            parts.append(f"[User sent audio: {url}]")
        else:
            parts.append(f"[User sent a file: {url}]")
    return "\n".join(parts)


def _sync_visible_final_response_into_messages(
    messages: List[Dict[str, Any]],
    *,
    raw_final_response: Any,
    visible_final_response: Any,
) -> List[Dict[str, Any]]:
    """Align persisted assistant transcript text with the actual user-facing reply."""
    visible = str(visible_final_response or "").strip()
    if not visible:
        return messages

    raw = str(raw_final_response or "").strip()
    if raw == visible:
        return messages

    synced = [dict(msg) if isinstance(msg, dict) else msg for msg in messages]
    for idx in range(len(synced) - 1, -1, -1):
        msg = synced[idx]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        current = str(msg.get("content") or "").strip()
        if current == visible:
            return synced
        if current in {"", "(empty)", "[[NO_REPLY]]", raw}:
            msg["content"] = visible
            return synced
        break

    synced.append({"role": "assistant", "content": visible})
    return synced


def _dequeue_pending_event_text(adapter, session_key: str) -> tuple[MessageEvent | None, str | None]:
    """Consume and return the pending queued event plus normalized text.

    Preserves media context for captionless photo/document events by
    building a placeholder so the message isn't silently dropped.
    """
    event = adapter.get_pending_message(session_key)
    if not event:
        return None, None
    text = event.text
    if not text and getattr(event, "media_urls", None):
        text = _build_media_placeholder(event)
    return event, text


def _qq_group_latest_admin_turn(raw_message: Any) -> bool:
    """Return True when QQ group metadata marks the latest turn as admin-owned."""
    if not isinstance(raw_message, dict):
        return False
    if not bool(raw_message.get("qq_group_batch")):
        return False
    return bool(raw_message.get("latest_is_admin"))


def _qq_group_no_reply_fallback(
    message: str,
    *,
    is_admin_user: bool = False,
    raw_message: Any = None,
) -> str:
    """Return a QQ-group fallback for explicit-address no-reply turns."""
    body = str(message or "").strip()
    if not body:
        return ""
    if _qq_group_latest_admin_turn(raw_message):
        return "刚才这轮接口空转了，消息我收到了。你再发一遍，或者我继续接着干。"
    if any(name in body for name in _QQ_VISIBLE_NAME_ALIASES):
        return "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"
    if _looks_like_qq_runtime_short_query(body):
        return "刚才我这轮空转了，但我还在。你再发一遍，或者我继续接着干。"
    if is_admin_user and _looks_like_qq_group_request_text(body):
        return "刚才这轮接口空转了，消息我收到了。你再发一遍，或者我继续接着干。"
    return ""


def _qq_group_empty_response_fallback(
    message: str,
    *,
    is_admin_user: bool = False,
    explicit_addressed: bool = False,
) -> str:
    """Return a QQ-group fallback when a provider/tool turn yielded no final text."""
    body = str(message or "").strip()
    if explicit_addressed and not body:
        return "我在，你继续说。"
    if not body:
        return ""
    if explicit_addressed:
        if _looks_like_qq_runtime_short_query(body):
            return "刚才我这轮空转了，但我还在。你再发一遍，或者我继续接着干。"
        return "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"
    if _qq_group_has_visible_bot_address(body):
        return "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"
    if _looks_like_qq_runtime_short_query(body):
        return "刚才我这轮空转了，但我还在。你再发一遍，或者我继续接着干。"
    if _looks_like_qq_media_message(body):
        return "刚才这条带图/附件的消息我这轮没读出来。你再发一次，或者补一句文字我继续接。"
    if _looks_like_qq_group_request_text(body):
        return "刚才这轮接口空转了，但消息我收到了。你再发一遍，或者我继续接着干。"
    if is_admin_user:
        return "刚才这轮接口空转了，消息我收到了。你再发一遍，或者我继续接着干。"
    return ""


def _explicit_group_trigger_label(event: MessageEvent | None) -> str:
    source = getattr(event, "source", None)
    if getattr(source, "chat_type", "") != "group":
        return ""

    metadata = getattr(event, "metadata", None) or {}
    explicit_trigger = bool(
        metadata.get("explicit_addressed")
        or metadata.get("requires_reply")
        or metadata.get("explicit_group_trigger")
    )
    explicit_reason = str(
        metadata.get("address_reason")
        or metadata.get("explicit_group_trigger_reason")
        or ""
    ).strip()
    trigger_reason = str(metadata.get("group_trigger_reason") or "").strip()
    if explicit_trigger:
        return explicit_reason or trigger_reason or "explicit_address"
    if trigger_reason in {"bot_mention", "reply_to_bot", "name_trigger"}:
        return trigger_reason
    return ""


def _qq_busy_followup_ack(source: SessionSource, message: str = "") -> str:
    """Return a short visible QQ acknowledgement for queued follow-ups."""
    if getattr(source, "platform", None) != Platform.QQ_NAPCAT:
        return ""
    if getattr(source, "chat_type", "") == "dm":
        return "收到，这条我先排队，上一轮忙完马上接着回你。"

    body = str(message or "").strip()
    if any(name in body for name in _QQ_VISIBLE_NAME_ALIASES):
        return "收到，这条我先排队，上一轮忙完接着回你。"
    return ""


def _truncate_status_preview(value: Any, *, limit: int = 120) -> str:
    """Return a single-line preview for status/approval messages."""
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _build_long_running_status_detail(agent_ref: Any, session_key: str = "") -> str:
    """Build the detail suffix for periodic long-running gateway updates."""
    parts: list[str] = []
    if agent_ref and hasattr(agent_ref, "get_activity_summary"):
        try:
            activity = agent_ref.get_activity_summary() or {}
        except Exception:
            activity = {}
        if isinstance(activity, dict) and activity:
            parts.append(
                f"iteration {activity.get('api_call_count', 0)}/{activity.get('max_iterations', 0)}"
            )
            current_tool = str(activity.get("current_tool") or "").strip()
            if current_tool:
                parts.append(f"running: {current_tool}")
            else:
                last_desc = str(activity.get("last_activity_desc") or "").strip()
                if last_desc:
                    parts.append(last_desc)

    if session_key:
        try:
            from tools.approval import has_blocking_approval, peek_blocking_approval

            if has_blocking_approval(session_key):
                approval = peek_blocking_approval(session_key) or {}
                command_preview = _truncate_status_preview(approval.get("command", ""))
                if command_preview:
                    parts.append(f"waiting for approval: {command_preview}")
                else:
                    parts.append("waiting for approval")
        except Exception:
            pass

    return f" — {', '.join(parts)}" if parts else ""


def _empty_response_fallback(
    source: SessionSource,
    message: str = "",
    *,
    empty_kind: str = "empty",
    is_admin_user: bool = False,
    raw_message: Any = None,
    event: MessageEvent | None = None,
) -> str:
    """Return a user-facing fallback when the model yields no final text."""
    if getattr(source, "chat_type", "") == "dm":
        if getattr(source, "platform", None) == Platform.QQ_NAPCAT:
            return "刚才接口抽了，没吐出正文。你再发一条，或者我继续接着刚才的话题说。"
        return "I didn't get a usable response just now. Please send that again."
    if getattr(source, "chat_type", "") == "group":
        explicit_group_trigger = bool(_explicit_group_trigger_label(event))
        if getattr(source, "platform", None) == Platform.QQ_NAPCAT:
            if empty_kind == "no_reply":
                if explicit_group_trigger:
                    return "收到，你继续说。"
                return _qq_group_no_reply_fallback(
                    message,
                    is_admin_user=is_admin_user,
                    raw_message=raw_message,
                )
            return _qq_group_empty_response_fallback(
                message,
                is_admin_user=is_admin_user,
                explicit_addressed=explicit_group_trigger,
            )
        if explicit_group_trigger:
            if empty_kind == "no_reply":
                return "收到，你继续说。"
            if not str(message or "").strip():
                return "我在，你继续说。"
            return "刚才这轮没吐出正文，但消息我收到了。你再发一遍，或者我继续接着干。"
    return ""


def _explicit_group_reply_context_note(event: MessageEvent) -> str:
    label = _explicit_group_trigger_label(event)
    if not label:
        return ""
    return (
        "[Current group turn note: This message explicitly addressed you "
        f"(trigger reason: {label}). You must reply briefly to this turn. "
        "Do not return [[NO_REPLY]] for this turn.]"
    )


_NON_INTERRUPTIBLE_RUNNING_TOOLS = frozenset({
    "qq_group_moderation",
})
_GATEWAY_SIGNAL_FORCE_EXIT_SECONDS = 20.0
def _safe_float(value: Any, default: float) -> float:
    """Best-effort float conversion that never raises for mocks or bad values."""
    try:
        if value is None or isinstance(value, bool):
            raise TypeError("invalid numeric value")
        return float(value)
    except Exception:
        return default


def _load_gateway_signal_force_exit_seconds() -> float:
    """Return the forced-exit grace period after SIGTERM/SIGINT.

    Set ``HERMES_GATEWAY_FORCE_EXIT_SECONDS=0`` to disable the watchdog.
    """
    raw = str(os.getenv("HERMES_GATEWAY_FORCE_EXIT_SECONDS", "") or "").strip()
    if not raw:
        return _GATEWAY_SIGNAL_FORCE_EXIT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid HERMES_GATEWAY_FORCE_EXIT_SECONDS=%r, using %.1fs",
            raw,
            _GATEWAY_SIGNAL_FORCE_EXIT_SECONDS,
        )
        return _GATEWAY_SIGNAL_FORCE_EXIT_SECONDS
    return max(0.0, value)


def _make_gateway_signal_handler(runner: "GatewayRunner", *, force_exit_after: float):
    """Create a signal handler that triggers graceful stop with a hard-exit watchdog."""
    force_exit_timer = None

    def _force_exit():
        logger.error(
            "Gateway shutdown exceeded %.1fs after signal; forcing process exit.",
            force_exit_after,
        )
        try:
            from gateway.status import remove_pid_file, write_runtime_status
            remove_pid_file()
            write_runtime_status(
                gateway_state="stopped",
                exit_reason="forced_signal_exit",
                runtime_summary={},
            )
        except Exception:
            pass
        os._exit(0)

    def _cancel_force_exit_timer() -> None:
        nonlocal force_exit_timer
        if force_exit_timer is None:
            return
        try:
            force_exit_timer.cancel()
        except Exception:
            pass
        force_exit_timer = None

    def _signal_handler() -> None:
        nonlocal force_exit_timer
        if force_exit_after > 0 and force_exit_timer is None:
            force_exit_timer = threading.Timer(force_exit_after, _force_exit)
            force_exit_timer.daemon = True
            force_exit_timer.start()
        asyncio.create_task(runner.stop())

    return _signal_handler, _cancel_force_exit_timer


def _check_unavailable_skill(command_name: str) -> str | None:
    """Check if a command matches a known-but-inactive skill.

    Returns a helpful message if the skill exists but is disabled or only
    available as an optional install. Returns None if no match found.
    """
    # Normalize: command uses hyphens, skill names may use hyphens or underscores
    normalized = command_name.lower().replace("_", "-")
    try:
        from tools.skills_tool import _get_disabled_skill_names
        from agent.skill_utils import get_all_skills_dirs
        disabled = _get_disabled_skill_names()

        # Check disabled skills across all dirs (local + external)
        for skills_dir in get_all_skills_dirs():
            if not skills_dir.exists():
                continue
            for skill_md in skills_dir.rglob("SKILL.md"):
                if any(part in ('.git', '.github', '.hub') for part in skill_md.parts):
                    continue
                name = skill_md.parent.name.lower().replace("_", "-")
                if name == normalized and name in disabled:
                    return (
                        f"The **{command_name}** skill is installed but disabled.\n"
                        f"Enable it with: `hermes skills config`"
                    )

        # Check optional skills (shipped with repo but not installed)
        from hermes_constants import get_optional_skills_dir
        repo_root = Path(__file__).resolve().parent.parent
        optional_dir = get_optional_skills_dir(repo_root / "optional-skills")
        if optional_dir.exists():
            for skill_md in optional_dir.rglob("SKILL.md"):
                name = skill_md.parent.name.lower().replace("_", "-")
                if name == normalized:
                    # Build install path: official/<category>/<name>
                    rel = skill_md.parent.relative_to(optional_dir)
                    parts = list(rel.parts)
                    install_path = f"official/{'/'.join(parts)}"
                    return (
                        f"The **{command_name}** skill is available but not installed.\n"
                        f"Install it with: `hermes skills install {install_path}`"
                    )
    except Exception:
        pass
    return None


def _platform_config_key(platform: "Platform") -> str:
    """Map a Platform enum to its config.yaml key (LOCAL→"cli", rest→enum value)."""
    return shared_platform_config_key(platform)


def _sync_shared_gateway_home() -> None:
    """Keep shared gateway helpers aligned with gateway.run test overrides."""
    try:
        import gateway.agent_runtime as _agent_runtime

        _agent_runtime._hermes_home = _hermes_home
    except Exception:
        pass


def _load_gateway_config() -> dict:
    """Load and parse ~/.hermes/config.yaml, returning {} on any error."""
    _sync_shared_gateway_home()
    return shared_load_gateway_user_config()


def _resolve_gateway_model(config: dict | None = None) -> str:
    """Read model from config.yaml — single source of truth.

    Without this, temporary AIAgent instances (memory flush, /compress) fall
    back to the hardcoded default which fails when the active provider is
    openai-codex.
    """
    return shared_resolve_gateway_model(config)


def _resolve_hermes_bin() -> Optional[list[str]]:
    """Resolve the Hermes update command as argv parts.

    Tries in order:
    1. ``shutil.which("hermes")`` — standard PATH lookup
    2. ``sys.executable -m hermes_cli.main`` — fallback when Hermes is running
       from a venv/module invocation and the ``hermes`` shim is not on PATH

    Returns argv parts ready for quoting/joining, or ``None`` if neither works.
    """
    import shutil

    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin]

    try:
        import importlib.util

        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass

    return None


class GatewayRunner:
    """
    Main gateway controller.

    Manages the lifecycle of all platform adapters and routes
    messages to/from the agent.
    """

    # Class-level defaults so partial construction in tests doesn't
    # blow up on attribute access.
    _running_agents_ts: Dict[str, float] = {}
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or load_gateway_config()
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}

        # Load ephemeral config from config.yaml / env vars.
        # Both are injected at API-call time only and never persisted.
        self._prefill_messages = self._load_prefill_messages()
        self._ephemeral_system_prompt = self._load_ephemeral_system_prompt()
        self._reasoning_config = self._load_reasoning_config()
        self._show_reasoning = self._load_show_reasoning()
        self._provider_routing = self._load_provider_routing()
        self._fallback_model = self._load_fallback_model()
        self._smart_model_routing = self._load_smart_model_routing()

        # Wire process registry into session store for reset protection
        from tools.process_registry import process_registry
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(key),
        )
        self.delivery_router = DeliveryRouter(self.config)
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._exit_cleanly = False
        self._exit_with_failure = False
        self._exit_reason: Optional[str] = None
        
        # Track running agents per session for interrupt support
        # Key: session_key, Value: AIAgent instance
        self._running_agents: Dict[str, Any] = {}
        self._running_agents_ts: Dict[str, float] = {}  # start timestamp per session
        self._pending_messages: Dict[str, str] = {}  # Queued messages during interrupt

        # Cache AIAgent instances per session to preserve prompt caching.
        # Without this, a new AIAgent is created per message, rebuilding the
        # system prompt (including memory) every turn — breaking prefix cache
        # and costing ~10x more on providers with prompt caching (Anthropic).
        # Key: session_key, Value: (AIAgent, config_signature_str)
        import threading as _threading
        self._agent_cache: Dict[str, tuple] = {}
        self._agent_cache_lock = _threading.Lock()

        # Track active fallback model/provider when primary is rate-limited.
        # Set after an agent run where fallback was activated; cleared when
        # the primary model succeeds again or the user switches via /model.
        self._effective_model: Optional[str] = None
        self._effective_provider: Optional[str] = None

        # Per-session model overrides from /model command.
        # Key: session_key, Value: dict with model/provider/api_key/base_url/api_mode
        self._session_model_overrides: Dict[str, Dict[str, str]] = {}
        # Track pending exec approvals per session
        # Key: session_key, Value: {"command": str, "pattern_key": str, ...}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

        # Track platforms that failed to connect for background reconnection.
        # Key: Platform enum, Value: {"config": platform_config, "attempts": int, "next_retry": float}
        self._failed_platforms: Dict[Platform, Dict[str, Any]] = {}

        # Track pending /update prompt responses per session.
        # Key: session_key, Value: True when a prompt is waiting for user input.
        self._update_prompt_pending: Dict[str, bool] = {}

        # Persistent Honcho managers keyed by gateway session key.
        # This preserves write_frequency="session" semantics across short-lived
        # per-message AIAgent instances.



        # Ensure tirith security scanner is available (downloads if needed)
        try:
            from tools.tirith_security import ensure_installed
            ensure_installed(log_failures=False)
        except Exception:
            pass  # Non-fatal — fail-open at scan time if unavailable
        
        # Initialize session database for session_search tool support
        self._session_db = None
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            logger.debug("SQLite session store not available: %s", e)
        
        # DM pairing store for code-based user authorization
        from gateway.pairing import PairingStore
        self.pairing_store = PairingStore()
        
        # Event hook system
        from gateway.hooks import HookRegistry
        self.hooks = HookRegistry()

        # Per-chat voice reply mode: "off" | "voice_only" | "all"
        self._voice_mode: Dict[str, str] = self._load_voice_modes()

        # Track background tasks to prevent garbage collection mid-execution
        self._background_tasks: set = set()
        self._background_job_store = BackgroundJobStore()
        self._direct_control_router = DirectControlRouter(self)
        self._auto_vision_cache: Dict[str, Dict[str, Any]] = {}
        self._auto_vision_tasks: Dict[str, asyncio.Task] = {}
        self._auto_vision_unhealthy_until = 0.0
        self._auto_vision_unhealthy_reason = ""




    # -- Setup skill availability ----------------------------------------

    def _has_setup_skill(self) -> bool:
        """Check if the hermes-agent-setup skill is installed."""
        try:
            from tools.skill_manager_tool import _find_skill
            return _find_skill("hermes-agent-setup") is not None
        except Exception:
            return False

    def _persist_gateway_direct_reply(
        self,
        *,
        session_id: str,
        source: SessionSource,
        history: list[dict[str, Any]],
        user_message: str,
        assistant_message: str,
    ) -> None:
        ts = datetime.now().isoformat()
        if not history:
            self.session_store.append_to_transcript(
                session_id,
                {
                    "role": "session_meta",
                    "tools": [],
                    "model": _resolve_gateway_model(),
                    "platform": source.platform.value if source.platform else "",
                    "timestamp": ts,
                },
            )
        self.session_store.append_to_transcript(
            session_id,
            {"role": "user", "content": user_message, "timestamp": ts},
        )
        self.session_store.append_to_transcript(
            session_id,
            {"role": "assistant", "content": assistant_message, "timestamp": ts},
        )

    # -- Voice mode persistence ------------------------------------------

    _VOICE_MODE_PATH = _hermes_home / "gateway_voice_mode.json"

    def _load_voice_modes(self) -> Dict[str, str]:
        try:
            data = json.loads(self._VOICE_MODE_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}

        valid_modes = {"off", "voice_only", "all"}
        return {
            str(chat_id): mode
            for chat_id, mode in data.items()
            if mode in valid_modes
        }

    def _save_voice_modes(self) -> None:
        try:
            self._VOICE_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._VOICE_MODE_PATH.write_text(
                json.dumps(self._voice_mode, indent=2)
            )
        except OSError as e:
            logger.warning("Failed to save voice modes: %s", e)

    def _set_adapter_auto_tts_disabled(self, adapter, chat_id: str, disabled: bool) -> None:
        """Update an adapter's in-memory auto-TTS suppression set if present."""
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if not isinstance(disabled_chats, set):
            return
        if disabled:
            disabled_chats.add(chat_id)
        else:
            disabled_chats.discard(chat_id)

    def _sync_voice_mode_state_to_adapter(self, adapter) -> None:
        """Restore persisted /voice off state into a live platform adapter."""
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if not isinstance(disabled_chats, set):
            return
        disabled_chats.clear()
        disabled_chats.update(
            chat_id for chat_id, mode in self._voice_mode.items() if mode == "off"
        )

    # -----------------------------------------------------------------

    def _flush_memories_for_session(
        self,
        old_session_id: str,
    ):
        """Prompt the agent to save memories/skills before context is lost.

        Synchronous worker — meant to be called via run_in_executor from
        an async context so it doesn't block the event loop.
        """
        # Skip cron sessions — they run headless with no meaningful user
        # conversation to extract memories from.
        if old_session_id and old_session_id.startswith("cron_"):
            logger.debug("Skipping memory flush for cron session: %s", old_session_id)
            return

        try:
            history = self.session_store.load_transcript(old_session_id)
            if not history or len(history) < 4:
                return

            from agent.auxiliary_client import call_llm
            from model_tools import get_tool_definitions

            runtime_kwargs = _resolve_runtime_agent_kwargs()
            if not runtime_kwargs.get("api_key"):
                return

            # Resolve model from config — AIAgent's default is OpenRouter-
            # formatted ("anthropic/claude-opus-4.6") which fails when the
            # active provider is openai-codex.
            model = _resolve_gateway_model()

            # Build conversation history from transcript
            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            if len(msgs) < 4:
                return

            # Read live memory state from disk so the flush agent can see
            # what's already saved and avoid overwriting newer entries.
            _current_memory = ""
            try:
                from tools.memory_tool import get_memory_dir
                _mem_dir = get_memory_dir()
                for fname, label in [
                    ("MEMORY.md", "MEMORY (your personal notes)"),
                    ("USER.md", "USER PROFILE (who the user is)"),
                ]:
                    fpath = _mem_dir / fname
                    if fpath.exists():
                        content = fpath.read_text(encoding="utf-8").strip()
                        if content:
                            _current_memory += f"\n\n## Current {label}:\n{content}"
            except Exception:
                pass  # Non-fatal — flush still works, just without the guard

            # Give the agent a real turn to think about what to save
            flush_prompt = (
                "[System: This session is about to be automatically reset due to "
                "inactivity or a scheduled daily reset. The conversation context "
                "will be cleared after this turn.\n\n"
                "Review the conversation above and:\n"
                "1. Save any important facts, preferences, or decisions to memory "
                "(user profile or your notes) that would be useful in future sessions.\n"
                "2. If you discovered a reusable workflow or solved a non-trivial "
                "problem, consider saving it as a skill.\n"
                "3. If nothing is worth saving, that's fine — just skip.\n\n"
            )

            if _current_memory:
                flush_prompt += (
                    "IMPORTANT — here is the current live state of memory. Other "
                    "sessions, cron jobs, or the user may have updated it since this "
                    "conversation ended. Do NOT overwrite or remove entries unless "
                    "the conversation above reveals something that genuinely "
                    "supersedes them. Only add new information that is not already "
                    "captured below."
                    f"{_current_memory}\n\n"
                )

            flush_prompt += (
                "Do NOT respond to the user. Just use the memory and skill_manage "
                "tools if needed, then stop.]"
            )

            tool_defs = [
                td
                for td in get_tool_definitions(
                    enabled_toolsets=["memory", "skills"],
                    quiet_mode=True,
                )
                if td.get("function", {}).get("name") in {"memory", "skill_manage"}
            ]
            if not tool_defs:
                return

            response = call_llm(
                task="flush_memories",
                provider=runtime_kwargs.get("provider"),
                model=model,
                base_url=runtime_kwargs.get("base_url"),
                api_key=runtime_kwargs.get("api_key"),
                messages=msgs + [{"role": "user", "content": flush_prompt}],
                tools=tool_defs,
                temperature=0.3,
                max_tokens=5120,
                timeout=30.0,
            )
            tool_calls = _extract_gateway_flush_tool_calls(response)
            if tool_calls:
                memory_store = _load_gateway_flush_memory_store()
                for tool_call in tool_calls:
                    _execute_gateway_flush_tool_call(
                        tool_call,
                        memory_store=memory_store,
                    )
            logger.info("Pre-reset memory flush completed for session %s", old_session_id)
        except Exception as e:
            logger.debug("Pre-reset memory flush failed for session %s: %s", old_session_id, e)

    async def _async_flush_memories(
        self,
        old_session_id: str,
    ):
        """Run the sync memory flush in a thread pool so it won't block the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._flush_memories_for_session,
            old_session_id,
        )

    @property
    def should_exit_cleanly(self) -> bool:
        return self._exit_cleanly

    @property
    def should_exit_with_failure(self) -> bool:
        return self._exit_with_failure

    @property
    def exit_reason(self) -> Optional[str]:
        return self._exit_reason

    def _session_key_for_source(self, source: SessionSource) -> str:
        """Resolve the current session key for a source, honoring gateway config when available."""
        if hasattr(self, "session_store") and self.session_store is not None:
            try:
                session_key = self.session_store._generate_session_key(source)
                if isinstance(session_key, str) and session_key:
                    return session_key
            except Exception:
                pass
        config = getattr(self, "config", None)
        group_sessions_per_user = getattr(config, "group_sessions_per_user", True)
        thread_sessions_per_user = getattr(config, "thread_sessions_per_user", False)
        if config is not None and hasattr(config, "get_session_isolation"):
            try:
                group_sessions_per_user, thread_sessions_per_user = config.get_session_isolation(
                    source.platform
                )
            except Exception:
                pass
        return build_session_key(
            source,
            group_sessions_per_user=group_sessions_per_user,
            thread_sessions_per_user=thread_sessions_per_user,
        )

    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        return shared_resolve_turn_agent_config(
            user_message,
            model,
            runtime_kwargs,
            getattr(self, "_smart_model_routing", {}),
        )

    async def _handle_adapter_fatal_error(self, adapter: BasePlatformAdapter) -> None:
        """React to an adapter failure after startup.

        If the error is retryable (e.g. network blip, DNS failure), queue the
        platform for background reconnection instead of giving up permanently.
        """
        logger.error(
            "Fatal %s adapter error (%s): %s",
            adapter.platform.value,
            adapter.fatal_error_code or "unknown",
            adapter.fatal_error_message or "unknown error",
        )

        existing = self.adapters.get(adapter.platform)
        if existing is adapter:
            try:
                await adapter.disconnect()
            finally:
                self.adapters.pop(adapter.platform, None)
                self.delivery_router.adapters = self.adapters

        # Queue retryable failures for background reconnection
        if adapter.fatal_error_retryable:
            platform_config = self.config.platforms.get(adapter.platform)
            if platform_config and adapter.platform not in self._failed_platforms:
                self._failed_platforms[adapter.platform] = {
                    "config": platform_config,
                    "attempts": 0,
                    "next_retry": time.monotonic() + 30,
                }
                logger.info(
                    "%s queued for background reconnection",
                    adapter.platform.value,
                )

        if not self.adapters and not self._failed_platforms:
            self._exit_reason = adapter.fatal_error_message or "All messaging adapters disconnected"
            if adapter.fatal_error_retryable:
                self._exit_with_failure = True
                logger.error("No connected messaging platforms remain. Shutting down gateway for service restart.")
            else:
                logger.error("No connected messaging platforms remain. Shutting down gateway cleanly.")
            await self.stop()
        elif not self.adapters and self._failed_platforms:
            # All platforms are down and queued for background reconnection.
            # If the error is retryable, exit with failure so systemd Restart=on-failure
            # can restart the process. Otherwise stay alive and keep retrying in background.
            if adapter.fatal_error_retryable:
                self._exit_reason = adapter.fatal_error_message or "All messaging platforms failed with retryable errors"
                self._exit_with_failure = True
                logger.error(
                    "All messaging platforms failed with retryable errors. "
                    "Shutting down gateway for service restart (systemd will retry)."
                )
                await self.stop()
            else:
                logger.warning(
                    "No connected messaging platforms remain, but %d platform(s) queued for reconnection",
                    len(self._failed_platforms),
                )

    def _request_clean_exit(self, reason: str) -> None:
        self._exit_cleanly = True
        self._exit_reason = reason
        self._shutdown_event.set()
    
    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.
        
        Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
        the prefill_messages_file key in ~/.hermes/config.yaml.
        Relative paths are resolved from ~/.hermes/.
        """
        _sync_shared_gateway_home()
        return shared_load_prefill_messages()

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """Load ephemeral system prompt from config or env var.
        
        Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then falls back to
        agent.system_prompt in ~/.hermes/config.yaml.
        """
        _sync_shared_gateway_home()
        return shared_load_ephemeral_system_prompt()

    @staticmethod
    def _load_reasoning_config() -> dict | None:
        """Load reasoning effort from config.yaml.

        Reads agent.reasoning_effort from config.yaml. Valid: "xhigh",
        "high", "medium", "low", "minimal", "none". Returns None to use
        default (medium).
        """
        _sync_shared_gateway_home()
        return shared_load_reasoning_config()

    @staticmethod
    def _load_show_reasoning() -> bool:
        """Load show_reasoning toggle from config.yaml display section."""
        _sync_shared_gateway_home()
        return shared_load_show_reasoning()

    @staticmethod
    def _load_background_notifications_mode() -> str:
        """Load background process notification mode from config or env var.

        Modes:
          - ``all``    — push running-output updates *and* the final message (default)
          - ``result`` — only the final completion message (regardless of exit code)
          - ``error``  — only the final message when exit code is non-zero
          - ``off``    — no watcher messages at all
        """
        mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
        if not mode:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as _f:
                        cfg = _y.safe_load(_f) or {}
                    raw = cfg.get("display", {}).get("background_process_notifications")
                    if raw is False:
                        mode = "off"
                    elif raw not in (None, ""):
                        mode = str(raw)
            except Exception:
                pass
        mode = (mode or "all").strip().lower()
        valid = {"all", "result", "error", "off"}
        if mode not in valid:
            logger.warning(
                "Unknown background_process_notifications '%s', defaulting to 'all'",
                mode,
            )
            return "all"
        return mode

    @staticmethod
    def _load_provider_routing() -> dict:
        """Load OpenRouter provider routing preferences from config.yaml."""
        _sync_shared_gateway_home()
        return shared_load_provider_routing()

    @staticmethod
    def _load_fallback_model() -> list | dict | None:
        """Load fallback provider chain from config.yaml.

        Returns a list of provider dicts (``fallback_providers``), a single
        dict (legacy ``fallback_model``), or None if not configured.
        AIAgent.__init__ normalizes both formats into a chain.
        """
        _sync_shared_gateway_home()
        return shared_load_fallback_model()

    @staticmethod
    def _load_smart_model_routing() -> dict:
        """Load optional smart cheap-vs-strong model routing config."""
        _sync_shared_gateway_home()
        return shared_load_smart_model_routing()

    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.
        
        Returns True if at least one adapter connected successfully.
        """
        logger.info("Starting Hermes Gateway...")
        logger.info("Session storage: %s", self.config.sessions_dir)
        try:
            from hermes_cli.profiles import get_active_profile_name
            _profile = get_active_profile_name()
            if _profile and _profile != "default":
                logger.info("Active profile: %s", _profile)
        except Exception:
            pass
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                gateway_state="starting",
                exit_reason=None,
                reset_platforms=True,
            )
        except Exception:
            pass
        
        # Warn if no user allowlists are configured and open access is not opted in
        _any_allowlist = any(
            os.getenv(v)
            for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
                       "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
                       "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS",
                       "QQ_NAPCAT_ALLOWED_USERS",
                       "EMAIL_ALLOWED_USERS",
                       "SMS_ALLOWED_USERS", "MATTERMOST_ALLOWED_USERS",
                       "MATRIX_ALLOWED_USERS", "DINGTALK_ALLOWED_USERS",
                       "FEISHU_ALLOWED_USERS",
                       "WECOM_ALLOWED_USERS",
                       "WECOM_CALLBACK_ALLOWED_USERS",
                       "WEIXIN_ALLOWED_USERS",
                       "GATEWAY_ALLOWED_USERS")
        )
        _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes") or any(
            os.getenv(v, "").lower() in ("true", "1", "yes")
            for v in ("TELEGRAM_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS",
                       "WHATSAPP_ALLOW_ALL_USERS", "SLACK_ALLOW_ALL_USERS",
                       "SIGNAL_ALLOW_ALL_USERS", "EMAIL_ALLOW_ALL_USERS",
                       "QQ_NAPCAT_ALLOW_ALL_USERS",
                       "SMS_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS",
                       "MATRIX_ALLOW_ALL_USERS", "DINGTALK_ALLOW_ALL_USERS",
                       "FEISHU_ALLOW_ALL_USERS",
                       "WECOM_ALLOW_ALL_USERS",
                       "WECOM_CALLBACK_ALLOW_ALL_USERS",
                       "WEIXIN_ALLOW_ALL_USERS")
        )
        if not _any_allowlist and not _allow_all:
            logger.warning(
                "No user allowlists configured. All unauthorized users will be denied. "
                "Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access, "
                "or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id)."
            )
        
        # Discover and load event hooks
        self.hooks.discover_and_load()
        
        # Recover background processes from checkpoint (crash recovery)
        try:
            from tools.process_registry import process_registry
            recovered = process_registry.recover_from_checkpoint()
            if recovered:
                logger.info("Recovered %s background process(es) from previous run", recovered)
        except Exception as e:
            logger.warning("Process checkpoint recovery: %s", e)
        
        connected_count = 0
        enabled_platform_count = 0
        startup_nonretryable_errors: list[str] = []
        startup_retryable_errors: list[str] = []
        
        # Initialize and connect each configured platform
        for platform, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
            enabled_platform_count += 1
            
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                logger.warning("No adapter available for %s", platform.value)
                continue
            
            # Set up message + fatal error handlers
            adapter.set_message_handler(self._handle_message)
            adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
            adapter.set_session_store(self.session_store)
            if hasattr(adapter, "set_busy_input_mode"):
                adapter.set_busy_input_mode(self._get_busy_input_mode(platform))
            
            # Try to connect
            logger.info("Connecting to %s...", platform.value)
            try:
                success = await adapter.connect()
                if success:
                    self.adapters[platform] = adapter
                    self._sync_voice_mode_state_to_adapter(adapter)
                    connected_count += 1
                    logger.info("✓ %s connected", platform.value)
                else:
                    logger.warning("✗ %s failed to connect", platform.value)
                    if adapter.has_fatal_error:
                        target = (
                            startup_retryable_errors
                            if adapter.fatal_error_retryable
                            else startup_nonretryable_errors
                        )
                        target.append(
                            f"{platform.value}: {adapter.fatal_error_message}"
                        )
                        # Queue for reconnection if the error is retryable
                        if adapter.fatal_error_retryable:
                            self._failed_platforms[platform] = {
                                "config": platform_config,
                                "attempts": 1,
                                "next_retry": time.monotonic() + 30,
                            }
                    else:
                        startup_retryable_errors.append(
                            f"{platform.value}: failed to connect"
                        )
                        # No fatal error info means likely a transient issue — queue for retry
                        self._failed_platforms[platform] = {
                            "config": platform_config,
                            "attempts": 1,
                            "next_retry": time.monotonic() + 30,
                        }
            except Exception as e:
                logger.error("✗ %s error: %s", platform.value, e)
                startup_retryable_errors.append(f"{platform.value}: {e}")
                # Unexpected exceptions are typically transient — queue for retry
                self._failed_platforms[platform] = {
                    "config": platform_config,
                    "attempts": 1,
                    "next_retry": time.monotonic() + 30,
                }
        
        if connected_count == 0:
            if startup_nonretryable_errors:
                reason = "; ".join(startup_nonretryable_errors)
                logger.error("Gateway hit a non-retryable startup conflict: %s", reason)
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(
                        gateway_state="startup_failed",
                        exit_reason=reason,
                        runtime_summary={},
                    )
                except Exception:
                    pass
                self._request_clean_exit(reason)
                return True
            if enabled_platform_count > 0:
                reason = "; ".join(startup_retryable_errors) or "all configured messaging platforms failed to connect"
                logger.error("Gateway failed to connect any configured messaging platform: %s", reason)
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(
                        gateway_state="startup_failed",
                        exit_reason=reason,
                        runtime_summary={},
                    )
                except Exception:
                    pass
                return False
            logger.warning("No messaging platforms enabled.")
            logger.info("Gateway will continue running for cron job execution.")
        
        # Update delivery router with adapters
        self.delivery_router.adapters = self.adapters
        
        self._running = True
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                gateway_state="running",
                exit_reason=None,
                runtime_summary=self._build_runtime_status_summary(),
            )
        except Exception:
            pass
        
        # Emit gateway:startup hook
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in self.adapters.keys()],
        })
        
        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)
        
        # Build initial channel directory for send_message name resolution
        try:
            from gateway.channel_directory import build_channel_directory
            directory = build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        except Exception as e:
            logger.warning("Channel directory build failed: %s", e)
        
        # Check if we're restarting after a /update command. If the update is
        # still running, keep watching so we notify once it actually finishes.
        notified = await self._send_update_notification()
        if not notified and any(
            path.exists()
            for path in (
                _hermes_home / ".update_pending.json",
                _hermes_home / ".update_pending.claimed.json",
            )
        ):
            self._schedule_update_notification_watch()

        # Drain any recovered process watchers (from crash recovery checkpoint)
        try:
            from tools.process_registry import process_registry
            shared_drain_pending_process_watchers(
                process_registry=process_registry,
                run_process_watcher=self._run_process_watcher,
                create_task=asyncio.create_task,
                logger=logger,
                resumed_log_template="Resumed watcher for recovered process %s",
            )
        except Exception as e:
            logger.error("Recovered watcher setup error: %s", e)

        # Start background session expiry watcher for proactive memory flushing
        asyncio.create_task(self._session_expiry_watcher())

        # Start background reconnection watcher for platforms that failed at startup
        if self._failed_platforms:
            logger.info(
                "Starting reconnection watcher for %d failed platform(s): %s",
                len(self._failed_platforms),
                ", ".join(p.value for p in self._failed_platforms),
            )
        asyncio.create_task(self._platform_reconnect_watcher())
        _bg_delivery_task = asyncio.create_task(self._background_job_delivery_poller())
        self._background_tasks.add(_bg_delivery_task)
        _bg_delivery_task.add_done_callback(self._background_tasks.discard)
        _runtime_status_task = asyncio.create_task(self._runtime_status_heartbeat())
        self._background_tasks.add(_runtime_status_task)
        _runtime_status_task.add_done_callback(self._background_tasks.discard)

        logger.info("Press Ctrl+C to stop")
        
        return True
    
    async def _session_expiry_watcher(self, interval: int = 300):
        """Background task that proactively flushes memories for expired sessions.
        
        Runs every `interval` seconds (default 5 min).  For each session that
        has expired according to its reset policy, flushes memories in a thread
        pool and marks the session so it won't be flushed again.

        This means memories are already saved by the time the user sends their
        next message, so there's no blocking delay.
        """
        await asyncio.sleep(60)  # initial delay — let the gateway fully start
        _flush_failures: dict[str, int] = {}  # session_id -> consecutive failure count
        _MAX_FLUSH_RETRIES = 3
        while self._running:
            try:
                self.session_store._ensure_loaded()
                # Collect expired sessions first, then log a single summary.
                _expired_entries = []
                for key, entry in list(self.session_store._entries.items()):
                    if entry.memory_flushed:
                        continue
                    if not self.session_store._is_session_expired(entry):
                        continue
                    _expired_entries.append((key, entry))

                if _expired_entries:
                    # Extract platform names from session keys for a compact summary.
                    # Keys look like "agent:main:telegram:dm:12345" — platform is field [2].
                    _platforms: dict[str, int] = {}
                    for _k, _e in _expired_entries:
                        _parts = _k.split(":")
                        _plat = _parts[2] if len(_parts) > 2 else "unknown"
                        _platforms[_plat] = _platforms.get(_plat, 0) + 1
                    _plat_summary = ", ".join(
                        f"{p}:{c}" for p, c in sorted(_platforms.items())
                    )
                    logger.info(
                        "Session expiry: %d sessions to flush (%s)",
                        len(_expired_entries), _plat_summary,
                    )

                for key, entry in _expired_entries:
                    try:
                        await self._async_flush_memories(entry.session_id)
                        # Shut down memory provider on the cached agent
                        cached_agent = self._running_agents.get(key)
                        if cached_agent and cached_agent is not _AGENT_PENDING_SENTINEL:
                            try:
                                if hasattr(cached_agent, 'shutdown_memory_provider'):
                                    cached_agent.shutdown_memory_provider()
                            except Exception:
                                pass
                        # Mark as flushed and persist to disk so the flag
                        # survives gateway restarts.
                        with self.session_store._lock:
                            entry.memory_flushed = True
                            self.session_store._save()
                        logger.debug(
                            "Memory flush completed for session %s",
                            entry.session_id,
                        )
                        _flush_failures.pop(entry.session_id, None)
                    except Exception as e:
                        failures = _flush_failures.get(entry.session_id, 0) + 1
                        _flush_failures[entry.session_id] = failures
                        if failures >= _MAX_FLUSH_RETRIES:
                            logger.warning(
                                "Memory flush gave up after %d attempts for %s: %s. "
                                "Marking as flushed to prevent infinite retry loop.",
                                failures, entry.session_id, e,
                            )
                            with self.session_store._lock:
                                entry.memory_flushed = True
                                self.session_store._save()
                            _flush_failures.pop(entry.session_id, None)
                        else:
                            logger.debug(
                                "Memory flush failed (%d/%d) for %s: %s",
                                failures, _MAX_FLUSH_RETRIES, entry.session_id, e,
                            )

                if _expired_entries:
                    _flushed = sum(
                        1 for _, e in _expired_entries if e.memory_flushed
                    )
                    _failed = len(_expired_entries) - _flushed
                    if _failed:
                        logger.info(
                            "Session expiry done: %d flushed, %d pending retry",
                            _flushed, _failed,
                        )
                    else:
                        logger.info(
                            "Session expiry done: %d flushed", _flushed,
                        )
            except Exception as e:
                logger.debug("Session expiry watcher error: %s", e)
            # Sleep in small increments so we can stop quickly
            for _ in range(interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

    async def _platform_reconnect_watcher(self) -> None:
        """Background task that periodically retries connecting failed platforms.

        Uses exponential backoff: 30s → 60s → 120s → 240s → 300s (cap).
        Stops retrying a platform after 20 failed attempts or if the error
        is non-retryable (e.g. bad auth token).
        """
        _MAX_ATTEMPTS = 20
        _BACKOFF_CAP = 300  # 5 minutes max between retries

        await asyncio.sleep(10)  # initial delay — let startup finish
        while self._running:
            if not self._failed_platforms:
                # Nothing to reconnect — sleep and check again
                for _ in range(30):
                    if not self._running:
                        return
                    await asyncio.sleep(1)
                continue

            now = time.monotonic()
            for platform in list(self._failed_platforms.keys()):
                if not self._running:
                    return
                info = self._failed_platforms[platform]
                if now < info["next_retry"]:
                    continue  # not time yet

                if info["attempts"] >= _MAX_ATTEMPTS:
                    logger.warning(
                        "Giving up reconnecting %s after %d attempts",
                        platform.value, info["attempts"],
                    )
                    del self._failed_platforms[platform]
                    continue

                platform_config = info["config"]
                attempt = info["attempts"] + 1
                logger.info(
                    "Reconnecting %s (attempt %d/%d)...",
                    platform.value, attempt, _MAX_ATTEMPTS,
                )

                try:
                    adapter = self._create_adapter(platform, platform_config)
                    if not adapter:
                        logger.warning(
                            "Reconnect %s: adapter creation returned None, removing from retry queue",
                            platform.value,
                        )
                        del self._failed_platforms[platform]
                        continue

                    adapter.set_message_handler(self._handle_message)
                    adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
                    adapter.set_session_store(self.session_store)
                    if hasattr(adapter, "set_busy_input_mode"):
                        adapter.set_busy_input_mode(self._get_busy_input_mode(platform))

                    success = await adapter.connect()
                    if success:
                        self.adapters[platform] = adapter
                        self._sync_voice_mode_state_to_adapter(adapter)
                        self.delivery_router.adapters = self.adapters
                        del self._failed_platforms[platform]
                        logger.info("✓ %s reconnected successfully", platform.value)

                        # Rebuild channel directory with the new adapter
                        try:
                            from gateway.channel_directory import build_channel_directory
                            build_channel_directory(self.adapters)
                        except Exception:
                            pass
                    else:
                        # Check if the failure is non-retryable
                        if adapter.has_fatal_error and not adapter.fatal_error_retryable:
                            logger.warning(
                                "Reconnect %s: non-retryable error (%s), removing from retry queue",
                                platform.value, adapter.fatal_error_message,
                            )
                            del self._failed_platforms[platform]
                        else:
                            backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                            info["attempts"] = attempt
                            info["next_retry"] = time.monotonic() + backoff
                            logger.info(
                                "Reconnect %s failed, next retry in %ds",
                                platform.value, backoff,
                            )
                except Exception as e:
                    backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                    info["attempts"] = attempt
                    info["next_retry"] = time.monotonic() + backoff
                    logger.warning(
                        "Reconnect %s error: %s, next retry in %ds",
                        platform.value, e, backoff,
                    )

            # Check every 10 seconds for platforms that need reconnection
            for _ in range(10):
                if not self._running:
                    return
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the gateway and disconnect all adapters."""
        logger.info("Stopping gateway...")
        self._running = False

        for session_key, agent in list(self._running_agents.items()):
            if agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                agent.interrupt("Gateway shutting down")
                logger.debug("Interrupted running agent for session %s during shutdown", session_key[:20])
            except Exception as e:
                logger.debug("Failed interrupting agent during shutdown: %s", e)
            # Fire plugin on_session_finalize hook before memory shutdown
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook("on_session_finalize",
                             session_id=getattr(agent, 'session_id', None),
                             platform="gateway")
            except Exception:
                pass
            # Shut down memory provider at actual session boundary
            try:
                if hasattr(agent, 'shutdown_memory_provider'):
                    agent.shutdown_memory_provider()
            except Exception:
                pass

        for platform, adapter in list(self.adapters.items()):
            try:
                await adapter.cancel_background_tasks()
            except Exception as e:
                logger.debug("✗ %s background-task cancel error: %s", platform.value, e)
            try:
                await adapter.disconnect()
                logger.info("✓ %s disconnected", platform.value)
            except Exception as e:
                logger.error("✗ %s disconnect error: %s", platform.value, e)

        # Cancel any pending background tasks
        for _task in list(self._background_tasks):
            _task.cancel()
        self._background_tasks.clear()

        self.adapters.clear()
        self._running_agents.clear()
        self._pending_messages.clear()
        self._pending_approvals.clear()
        self._shutdown_event.set()
        
        from gateway.status import remove_pid_file, write_runtime_status
        remove_pid_file()
        try:
            write_runtime_status(
                gateway_state="stopped",
                exit_reason=self._exit_reason,
                runtime_summary={},
            )
        except Exception:
            pass
        
        logger.info("Gateway stopped")
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()
    
    def _create_adapter(
        self, 
        platform: Platform, 
        config: Any
    ) -> Optional[BasePlatformAdapter]:
        """Create the appropriate adapter for a platform."""
        if hasattr(config, "extra") and isinstance(config.extra, dict):
            config.extra.setdefault(
                "group_sessions_per_user",
                self.config.group_sessions_per_user,
            )
            config.extra.setdefault(
                "thread_sessions_per_user",
                getattr(self.config, "thread_sessions_per_user", False),
            )

        if platform == Platform.TELEGRAM:
            from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements
            if not check_telegram_requirements():
                logger.warning("Telegram: python-telegram-bot not installed")
                return None
            return TelegramAdapter(config)
        
        elif platform == Platform.DISCORD:
            from gateway.platforms.discord import DiscordAdapter, check_discord_requirements
            if not check_discord_requirements():
                logger.warning("Discord: discord.py not installed")
                return None
            return DiscordAdapter(config)
        
        elif platform == Platform.WHATSAPP:
            from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements
            if not check_whatsapp_requirements():
                logger.warning("WhatsApp: Node.js not installed or bridge not configured")
                return None
            return WhatsAppAdapter(config)
        
        elif platform == Platform.SLACK:
            from gateway.platforms.slack import SlackAdapter, check_slack_requirements
            if not check_slack_requirements():
                logger.warning("Slack: slack-bolt not installed. Run: pip install 'hermes-agent[slack]'")
                return None
            return SlackAdapter(config)

        elif platform == Platform.SIGNAL:
            from gateway.platforms.signal import SignalAdapter, check_signal_requirements
            if not check_signal_requirements():
                logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
                return None
            return SignalAdapter(config)

        elif platform == Platform.QQ_NAPCAT:
            from gateway.platforms.qq_napcat import QqNapCatAdapter, check_qq_napcat_requirements
            if not check_qq_napcat_requirements():
                logger.warning("QQ NapCat: QQ_NAPCAT_WS_URL not configured or aiohttp missing")
                return None
            return QqNapCatAdapter(config)

        elif platform == Platform.HOMEASSISTANT:
            from gateway.platforms.homeassistant import HomeAssistantAdapter, check_ha_requirements
            if not check_ha_requirements():
                logger.warning("HomeAssistant: aiohttp not installed or HASS_TOKEN not set")
                return None
            return HomeAssistantAdapter(config)

        elif platform == Platform.EMAIL:
            from gateway.platforms.email import EmailAdapter, check_email_requirements
            if not check_email_requirements():
                logger.warning("Email: EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, or EMAIL_SMTP_HOST not set")
                return None
            return EmailAdapter(config)

        elif platform == Platform.SMS:
            from gateway.platforms.sms import SmsAdapter, check_sms_requirements
            if not check_sms_requirements():
                logger.warning("SMS: aiohttp not installed or TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set")
                return None
            return SmsAdapter(config)

        elif platform == Platform.DINGTALK:
            from gateway.platforms.dingtalk import DingTalkAdapter, check_dingtalk_requirements
            if not check_dingtalk_requirements():
                logger.warning("DingTalk: dingtalk-stream not installed or DINGTALK_CLIENT_ID/SECRET not set")
                return None
            return DingTalkAdapter(config)

        elif platform == Platform.FEISHU:
            from gateway.platforms.feishu import FeishuAdapter, check_feishu_requirements
            if not check_feishu_requirements():
                logger.warning("Feishu: lark-oapi not installed or FEISHU_APP_ID/SECRET not set")
                return None
            return FeishuAdapter(config)

        elif platform == Platform.WECOM_CALLBACK:
            from gateway.platforms.wecom_callback import (
                WecomCallbackAdapter,
                check_wecom_callback_requirements,
            )
            if not check_wecom_callback_requirements():
                logger.warning("WeCom Callback: aiohttp/httpx not installed or callback credentials not set")
                return None
            return WecomCallbackAdapter(config)

        elif platform == Platform.WECOM:
            from gateway.platforms.wecom import WeComAdapter, check_wecom_requirements
            if not check_wecom_requirements():
                logger.warning("WeCom: aiohttp not installed or WECOM_BOT_ID/SECRET not set")
                return None
            return WeComAdapter(config)

        elif platform == Platform.WEIXIN:
            from gateway.platforms.weixin import WeixinAdapter, check_weixin_requirements
            if not check_weixin_requirements():
                logger.warning("Weixin: aiohttp/cryptography not installed or WEIXIN credentials missing")
                return None
            return WeixinAdapter(config)

        elif platform == Platform.MATTERMOST:
            from gateway.platforms.mattermost import MattermostAdapter, check_mattermost_requirements
            if not check_mattermost_requirements():
                logger.warning("Mattermost: MATTERMOST_TOKEN or MATTERMOST_URL not set, or aiohttp missing")
                return None
            return MattermostAdapter(config)

        elif platform == Platform.MATRIX:
            from gateway.platforms.matrix import MatrixAdapter, check_matrix_requirements
            if not check_matrix_requirements():
                logger.warning("Matrix: matrix-nio not installed or credentials not set. Run: pip install 'matrix-nio[e2e]'")
                return None
            return MatrixAdapter(config)

        elif platform == Platform.API_SERVER:
            from gateway.platforms.api_server import APIServerAdapter, check_api_server_requirements
            if not check_api_server_requirements():
                logger.warning("API Server: aiohttp not installed")
                return None
            return APIServerAdapter(config)

        elif platform == Platform.WEBHOOK:
            from gateway.platforms.webhook import WebhookAdapter, check_webhook_requirements
            if not check_webhook_requirements():
                logger.warning("Webhook: aiohttp not installed")
                return None
            adapter = WebhookAdapter(config)
            adapter.gateway_runner = self  # For cross-platform delivery
            return adapter

        return None
    
    def _is_user_authorized(self, source: SessionSource) -> bool:
        """
        Check if a user is authorized to use the bot.
        
        Checks in order:
        1. Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        2. Environment variable allowlists (TELEGRAM_ALLOWED_USERS, etc.)
        3. DM pairing approved list
        4. Global allow-all (GATEWAY_ALLOW_ALL_USERS=true)
        5. Default: deny
        """
        # Home Assistant events are system-generated (state changes), not
        # user-initiated messages.  The HASS_TOKEN already authenticates the
        # connection, so HA events are always authorized.
        # Webhook events are authenticated via HMAC signature validation in
        # the adapter itself — no user allowlist applies.
        if source.platform in (Platform.HOMEASSISTANT, Platform.WEBHOOK):
            return True

        user_id = source.user_id
        if not user_id:
            return False

        if self._is_admin_user(source):
            return True

        platform_env_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
            Platform.DISCORD: "DISCORD_ALLOWED_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
            Platform.SLACK: "SLACK_ALLOWED_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOWED_USERS",
            Platform.QQ_NAPCAT: "QQ_NAPCAT_ALLOWED_USERS",
            Platform.EMAIL: "EMAIL_ALLOWED_USERS",
            Platform.SMS: "SMS_ALLOWED_USERS",
            Platform.MATTERMOST: "MATTERMOST_ALLOWED_USERS",
            Platform.MATRIX: "MATRIX_ALLOWED_USERS",
            Platform.DINGTALK: "DINGTALK_ALLOWED_USERS",
            Platform.FEISHU: "FEISHU_ALLOWED_USERS",
            Platform.WECOM: "WECOM_ALLOWED_USERS",
            Platform.WECOM_CALLBACK: "WECOM_CALLBACK_ALLOWED_USERS",
            Platform.WEIXIN: "WEIXIN_ALLOWED_USERS",
        }
        platform_allow_all_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOW_ALL_USERS",
            Platform.DISCORD: "DISCORD_ALLOW_ALL_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOW_ALL_USERS",
            Platform.SLACK: "SLACK_ALLOW_ALL_USERS",
            Platform.SIGNAL: "SIGNAL_ALLOW_ALL_USERS",
            Platform.QQ_NAPCAT: "QQ_NAPCAT_ALLOW_ALL_USERS",
            Platform.EMAIL: "EMAIL_ALLOW_ALL_USERS",
            Platform.SMS: "SMS_ALLOW_ALL_USERS",
            Platform.MATTERMOST: "MATTERMOST_ALLOW_ALL_USERS",
            Platform.MATRIX: "MATRIX_ALLOW_ALL_USERS",
            Platform.DINGTALK: "DINGTALK_ALLOW_ALL_USERS",
            Platform.FEISHU: "FEISHU_ALLOW_ALL_USERS",
            Platform.WECOM: "WECOM_ALLOW_ALL_USERS",
            Platform.WECOM_CALLBACK: "WECOM_CALLBACK_ALLOW_ALL_USERS",
            Platform.WEIXIN: "WEIXIN_ALLOW_ALL_USERS",
        }

        # Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        platform_allow_all_var = platform_allow_all_map.get(source.platform, "")
        if platform_allow_all_var and os.getenv(platform_allow_all_var, "").lower() in ("true", "1", "yes"):
            return True

        # Check pairing store (always checked, regardless of allowlists)
        platform_name = source.platform.value if source.platform else ""
        if self.pairing_store.is_approved(platform_name, user_id):
            return True

        # Check platform-specific and global allowlists
        platform_allowlist = os.getenv(platform_env_map.get(source.platform, ""), "").strip()
        global_allowlist = os.getenv("GATEWAY_ALLOWED_USERS", "").strip()

        if not platform_allowlist and not global_allowlist:
            # No allowlists configured -- check global allow-all flag
            return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")

        # Check if user is in any allowlist
        allowed_ids = set()
        if platform_allowlist:
            allowed_ids.update(uid.strip() for uid in platform_allowlist.split(",") if uid.strip())
        if global_allowlist:
            allowed_ids.update(uid.strip() for uid in global_allowlist.split(",") if uid.strip())

        # "*" in any allowlist means allow everyone (consistent with
        # SIGNAL_GROUP_ALLOWED_USERS precedent)
        if "*" in allowed_ids:
            return True

        check_ids = {user_id}
        if "@" in user_id:
            check_ids.add(user_id.split("@")[0])

        # WhatsApp: resolve phone↔LID aliases from bridge session mapping files
        if source.platform == Platform.WHATSAPP:
            normalized_allowed_ids = set()
            for allowed_id in allowed_ids:
                normalized_allowed_ids.update(_expand_whatsapp_auth_aliases(allowed_id))
            if normalized_allowed_ids:
                allowed_ids = normalized_allowed_ids

            check_ids.update(_expand_whatsapp_auth_aliases(user_id))
            normalized_user_id = _normalize_whatsapp_identifier(user_id)
            if normalized_user_id:
                check_ids.add(normalized_user_id)

        return bool(check_ids & allowed_ids)

    def _configured_admin_user_ids(self, platform: Optional[Platform]) -> list[str]:
        """Return configured administrator user IDs for a platform."""
        admin_ids: list[str] = []
        seen: set[str] = set()

        def _add(values) -> None:
            for value in values:
                normalized = str(value or "").strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                admin_ids.append(normalized)

        _add(_coerce_list(os.getenv("GATEWAY_ADMIN_USERS")))

        if platform:
            _add(_coerce_list(os.getenv(f"{platform.value.upper()}_ADMIN_USERS")))
            platform_cfg = self.config.platforms.get(platform) if getattr(self, "config", None) else None
            extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
            if isinstance(extra, dict):
                _add(_coerce_list(extra.get("admin_users")))

        return admin_ids

    def _source_identity_candidates(self, source: SessionSource) -> set[str]:
        """Return normalized sender identifiers for auth/admin matching."""
        candidates: set[str] = set()
        for raw in (source.user_id, source.user_id_alt):
            value = str(raw or "").strip()
            if not value:
                continue
            candidates.add(value)
            if "@" in value:
                candidates.add(value.split("@", 1)[0])

        if source.platform == Platform.WHATSAPP:
            expanded: set[str] = set()
            for candidate in list(candidates):
                expanded.update(_expand_whatsapp_auth_aliases(candidate))
                normalized = _normalize_whatsapp_identifier(candidate)
                if normalized:
                    expanded.add(normalized)
            candidates.update(expanded)

        return candidates

    def _is_admin_user(self, source: SessionSource) -> bool:
        """Return True when the source user is explicitly configured as an admin."""
        admin_ids = set(self._configured_admin_user_ids(source.platform))
        if not admin_ids:
            return False
        return bool(self._source_identity_candidates(source) & admin_ids)

    def _admin_only_message(self, source: SessionSource, action: str) -> Optional[str]:
        """Return an admin-only rejection message when the user is not an admin."""
        admin_ids = self._configured_admin_user_ids(source.platform)
        if not admin_ids or self._is_admin_user(source):
            return None

        if source.platform == Platform.QQ_NAPCAT:
            ids_text = "、".join(admin_ids)
            return f"这事得董事长拍板。当前只有 QQ {ids_text} 能授权这类操作。"

        noun = "user ID" if len(admin_ids) == 1 else "user IDs"
        ids_text = ", ".join(admin_ids)
        return f"Only administrator {noun} {ids_text} can {action}."

    def _get_unauthorized_dm_behavior(self, platform: Optional[Platform]) -> str:
        """Return how unauthorized DMs should be handled for a platform."""
        config = getattr(self, "config", None)
        if config and hasattr(config, "get_unauthorized_dm_behavior"):
            return config.get_unauthorized_dm_behavior(platform)
        return "pair"

    def _get_busy_input_mode(self, platform: Optional[Platform]) -> str:
        """Return how follow-up text should behave while the agent is active."""
        config = getattr(self, "config", None)
        if config and hasattr(config, "get_busy_input_mode"):
            return config.get_busy_input_mode(platform)
        return "interrupt"

    def _get_auto_background_work(self, platform: Optional[Platform]) -> bool:
        """Return whether obvious long-running work should detach to background."""
        config = getattr(self, "config", None)
        if config and hasattr(config, "get_auto_background_work"):
            return bool(config.get_auto_background_work(platform))
        return False

    def _busy_followup_force_queue_reason(self, session_key: str, running_agent: Any) -> str:
        """Return a reason when the active run must not be interrupted."""
        try:
            from tools.approval import has_blocking_approval

            if has_blocking_approval(session_key):
                return "approval_pending"
        except Exception:
            pass
        try:
            if self._get_background_job_store().has_pending_approval_requests(session_key):
                return "approval_pending"
        except Exception:
            pass

        if running_agent in (None, _AGENT_PENDING_SENTINEL):
            return ""
        if not hasattr(running_agent, "get_activity_summary"):
            return ""

        try:
            activity = running_agent.get_activity_summary()
        except Exception:
            return ""
        if not isinstance(activity, dict):
            return ""

        current_tool = str(activity.get("current_tool") or "").strip()
        if current_tool in _NON_INTERRUPTIBLE_RUNNING_TOOLS:
            return f"critical_tool:{current_tool}"
        return ""

    def _ensure_background_job_state(self) -> None:
        """Initialize background-job runtime state for tests and older runner state."""
        if not hasattr(self, "_background_tasks"):
            self._background_tasks = set()
        if not hasattr(self, "_background_job_store") or self._background_job_store is None:
            self._background_job_store = BackgroundJobStore()

    def _get_background_job_store(self) -> BackgroundJobStore:
        self._ensure_background_job_state()
        return self._background_job_store

    def _background_job_chat_key(self, source: SessionSource) -> str:
        """Return a stable chat-scoped key for managed background jobs."""
        return durable_background_job_chat_key(source)

    def _background_job_scope_key(
        self,
        source: SessionSource,
        *,
        session_key: str = "",
    ) -> str:
        """Return the session-scoped key used to isolate background jobs."""
        resolved = str(session_key or "").strip()
        if not resolved:
            try:
                resolved = str(self._session_key_for_source(source) or "").strip()
            except Exception:
                resolved = ""
        return durable_background_job_scope_key(source, session_key=resolved)

    def _background_jobs_for_source(
        self,
        source: SessionSource,
        *,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return managed background jobs associated with the source chat."""
        chat_key = self._background_job_chat_key(source)
        scope_key = self._background_job_scope_key(source)
        try:
            durable_jobs = self._get_background_job_store().list_jobs(
                chat_key=chat_key,
                scope_key=scope_key,
                active_only=active_only,
            )
        except Exception:
            durable_jobs = []
        return sorted(
            durable_jobs,
            key=lambda item: _safe_float(item.get("created_at"), 0.0),
        )

    def _launch_background_worker(self, task_id: str) -> Dict[str, Any]:
        """Launch an external worker for a durable background job."""
        return launch_background_worker(task_id=task_id)

    def _stop_background_worker(self, job: Dict[str, Any]) -> bool:
        """Stop an external worker for a durable background job."""
        return stop_background_worker(job)

    def _format_background_job_age(self, job: Dict[str, Any]) -> str:
        """Return a short human-readable elapsed time for a background job."""
        started_at = _safe_float(job.get("started_at"), 0.0)
        created_at = _safe_float(job.get("created_at"), time.time())
        finished_at = _safe_float(job.get("finished_at"), time.time())
        anchor = finished_at if job.get("status") in {"completed", "failed", "cancelled"} else time.time()
        base = started_at or created_at
        elapsed = max(0, int(anchor - base))
        if elapsed >= 3600:
            return f"{elapsed // 3600}h{(elapsed % 3600) // 60:02d}m"
        if elapsed >= 60:
            return f"{elapsed // 60}m{elapsed % 60:02d}s"
        return f"{elapsed}s"

    @staticmethod
    def _looks_like_qq_group_listen_disable_request(message_text: str) -> bool:
        return looks_like_group_listen_disable_request(message_text)

    @staticmethod
    def _looks_like_qq_group_listen_enable_request(message_text: str) -> bool:
        return looks_like_group_listen_enable_request(message_text)

    @staticmethod
    def _looks_like_background_status_query(message_text: str) -> bool:
        return _looks_like_qq_background_status_query(message_text)

    @staticmethod
    def _looks_like_runtime_status_query(message_text: str) -> bool:
        return _looks_like_qq_runtime_status_query(message_text) or _looks_like_qq_runtime_short_query(
            message_text
        )

    @staticmethod
    def _looks_like_joined_group_list_query(message_text: str) -> bool:
        return _looks_like_qq_joined_group_list_query(message_text)

    @staticmethod
    def _looks_like_group_runtime_status_query(message_text: str) -> bool:
        return looks_like_shared_group_runtime_status_query(message_text)

    @staticmethod
    def _format_intel_worker_status_label(status: str) -> str:
        return {
            "awaiting_group_approval": "等待入群通过",
            "active_collecting": "正在潜伏采集",
            "paused": "已暂停",
            "stopped": "已停止",
            "failed": "任务失联",
            "rejected": "已拒绝",
        }.get(str(status or "").strip().lower(), str(status or "").strip() or "unknown")

    def _format_background_job_short_status(self, job: dict[str, Any]) -> str:
        return shared_format_background_job_short_status(self, job)

    def _format_running_session_short_status(self, session_key: str, agent_ref: Any) -> str:
        return shared_format_running_session_short_status(
            session_key,
            agent_ref,
            detail_builder=_build_long_running_status_detail,
        )

    @staticmethod
    def _unique_report_targets(values: list[Any]) -> list[str]:
        return shared_unique_report_targets(values)

    @classmethod
    def _worker_report_targets(
        cls,
        workers: list[dict[str, Any]],
        key: str,
        *,
        require_daily_enabled: bool = False,
    ) -> list[str]:
        del cls
        return shared_worker_report_targets(
            workers,
            key,
            require_daily_enabled=require_daily_enabled,
        )

    @staticmethod
    def _runtime_session_metadata(session_key: str) -> dict[str, str]:
        parts = str(session_key or "").split(":")
        return {
            "platform": parts[2] if len(parts) > 2 else "",
            "chat_type": parts[3] if len(parts) > 3 else "",
            "chat_id": parts[4] if len(parts) > 4 else "",
        }

    def _build_runtime_model_summary(self) -> dict[str, Any]:
        configured_model = str(_resolve_gateway_model() or "").strip()
        configured_base_url = ""
        try:
            configured_runtime = _resolve_runtime_agent_kwargs() or {}
            configured_provider = str(configured_runtime.get("provider") or "").strip()
            configured_base_url = str(configured_runtime.get("base_url") or "").strip()
        except Exception:
            configured_provider = ""
        active_model = str(getattr(self, "_effective_model", None) or configured_model).strip()
        active_provider = str(getattr(self, "_effective_provider", None) or configured_provider).strip()
        fallback_pinned = False

        candidate_agents: list[Any] = []
        for agent_ref in getattr(self, "_running_agents", {}).values():
            if agent_ref not in (None, _AGENT_PENDING_SENTINEL):
                candidate_agents.append(agent_ref)
        for cached in getattr(self, "_agent_cache", {}).values():
            agent_ref = cached[0] if isinstance(cached, tuple) and cached else cached
            if agent_ref in (None, _AGENT_PENDING_SENTINEL):
                continue
            if agent_ref not in candidate_agents:
                candidate_agents.append(agent_ref)

        for agent_ref in candidate_agents:
            raw_model = getattr(agent_ref, "model", "")
            raw_provider = getattr(agent_ref, "provider", "")
            model = raw_model.strip() if isinstance(raw_model, str) else ""
            provider = raw_provider.strip() if isinstance(raw_provider, str) else ""
            if model and (not active_model or model != configured_model):
                active_model = model
            if provider and (model == active_model or not active_provider):
                active_provider = provider
            has_pinned_fallback = getattr(agent_ref, "_has_pinned_fallback", None)
            if callable(has_pinned_fallback):
                try:
                    if has_pinned_fallback():
                        fallback_pinned = True
                        if model:
                            active_model = model
                        if provider:
                            active_provider = provider
                except Exception:
                    logger.debug("Could not evaluate fallback pin state", exc_info=True)

        fallback_active = bool(
            active_model and configured_model and active_model != configured_model
        )
        degraded_runtime_count = 0
        degraded_runtimes: list[dict[str, Any]] = []
        primary_degraded = False
        primary_degraded_reason = ""
        primary_degraded_cooldown_seconds = 0
        try:
            from run_agent import get_provider_health_snapshot, _runtime_targets_match

            degraded_snapshot = get_provider_health_snapshot(limit=5)
            degraded_runtime_count = int(degraded_snapshot.get("count") or 0)
            degraded_runtimes = list(degraded_snapshot.get("runtimes") or [])
            for runtime in degraded_runtimes:
                if _runtime_targets_match(
                    configured_provider,
                    configured_model,
                    configured_base_url,
                    runtime.get("provider"),
                    runtime.get("model"),
                    runtime.get("base_url"),
                ):
                    primary_degraded = True
                    primary_degraded_reason = str(runtime.get("reason") or "").strip()
                    primary_degraded_cooldown_seconds = int(
                        max(0.0, float(runtime.get("cooldown_seconds") or 0.0))
                    )
                    break
        except Exception:
            logger.debug("Could not load provider health snapshot", exc_info=True)
        return {
            "configured_model": configured_model,
            "configured_provider": configured_provider,
            "configured_base_url": configured_base_url,
            "active_model": active_model or configured_model,
            "active_provider": active_provider,
            "fallback_active": fallback_active,
            "fallback_pinned": fallback_pinned,
            "primary_degraded": primary_degraded,
            "primary_degraded_reason": primary_degraded_reason,
            "primary_degraded_cooldown_seconds": primary_degraded_cooldown_seconds,
            "degraded_runtime_count": degraded_runtime_count,
            "degraded_runtimes": degraded_runtimes,
        }

    def _build_runtime_approval_summary(self) -> dict[str, Any]:
        store = self._get_background_job_store()
        try:
            pending_count = store.count_all_pending_approval_requests()
        except Exception:
            pending_count = 0

        live_sessions: set[str] = set(
            str(session_key or "").strip()
            for session_key in getattr(self, "_pending_approvals", {})
            if str(session_key or "").strip()
        )
        pending_sessions = set(live_sessions)
        live_sessions.update(
            str(session_key or "").strip()
            for session_key in getattr(self, "_running_agents", {})
            if str(session_key or "").strip()
        )
        for session_key in pending_sessions:
            try:
                already_counted = store.has_pending_approval_requests(session_key)
            except Exception:
                already_counted = False
            if not already_counted:
                pending_count += 1
        try:
            from tools.approval import has_blocking_approval

            for session_key in live_sessions:
                if not has_blocking_approval(session_key):
                    continue
                try:
                    already_counted = store.has_pending_approval_requests(session_key)
                except Exception:
                    already_counted = False
                if not already_counted:
                    pending_count += 1
        except Exception:
            pass

        return {
            "pending_count": int(max(pending_count, 0)),
        }

    def _build_runtime_qq_monitoring_summary(self) -> dict[str, Any]:
        groups_by_id: dict[str, dict[str, Any]] = {}
        try:
            policy_groups = list_group_policies()
        except Exception as exc:
            logger.debug("Failed to load QQ group policy snapshot: %s", exc)
            policy_groups = []

        for policy in policy_groups:
            if not isinstance(policy, dict):
                continue
            if str(policy.get("mode") or "").strip().lower() != "collect_only":
                continue
            group_id = str(policy.get("group_id") or "").strip()
            if not group_id:
                continue
            groups_by_id.setdefault(
                group_id,
                {
                    "group_id": group_id,
                    "group_name": str(policy.get("group_name") or group_id).strip(),
                    "mode": "collect_only",
                    "worker_names": [],
                    "daily_report_enabled": bool(policy.get("daily_report_enabled")),
                },
            )

        try:
            workers = list_intel_workers(status="active_collecting")
        except Exception as exc:
            logger.debug("Failed to collect QQ monitoring runtime stats: %s", exc)
            workers = []

        for worker in workers:
            if not isinstance(worker, dict):
                continue
            group_id = str(worker.get("target_group_id") or "").strip()
            group_name = str(worker.get("target_group_name") or group_id).strip()
            group_ref = str(worker.get("target_group_ref") or "").strip()
            if not group_id and group_ref.startswith("group:"):
                group_id = group_ref.split(":", 1)[1]
            if not group_id and not group_name:
                continue
            key = group_id or group_name
            entry = groups_by_id.setdefault(
                key,
                {
                    "group_id": group_id,
                    "group_name": group_name,
                    "mode": "collect_only",
                    "worker_names": [],
                    "daily_report_enabled": False,
                },
            )
            worker_name = str(worker.get("worker_name") or "").strip()
            if worker_name and worker_name not in entry["worker_names"]:
                entry["worker_names"].append(worker_name)
            if bool(worker.get("daily_report_enabled")):
                entry["daily_report_enabled"] = True

        groups = sorted(
            groups_by_id.values(),
            key=lambda item: (
                str(item.get("group_name") or "").strip(),
                str(item.get("group_id") or "").strip(),
            ),
        )
        active_worker_count = 0
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            if str(worker.get("status") or "").strip().lower() == "active_collecting":
                active_worker_count += 1
        return {
            "active_collect_only_groups": len(groups),
            "active_worker_count": active_worker_count,
            "groups": groups[:8],
        }

    @staticmethod
    def _load_runtime_qq_archive_stats() -> dict[str, Any]:
        return QqGroupArchiveStore().get_runtime_stats()

    def _build_runtime_status_summary(self) -> dict[str, Any]:
        return shared_build_runtime_status_summary(
            self,
            pending_sentinel=_AGENT_PENDING_SENTINEL,
        )

    def _write_runtime_status_snapshot(self) -> None:
        try:
            from gateway.status import write_runtime_status

            write_runtime_status(
                gateway_state="running" if self._running else None,
                runtime_summary=self._build_runtime_status_summary(),
            )
            for platform in getattr(self, "adapters", {}) or {}:
                try:
                    platform_name = platform.value if hasattr(platform, "value") else str(platform)
                    write_runtime_status(
                        platform=platform_name,
                        platform_state="connected",
                        error_code=None,
                        error_message=None,
                    )
                except Exception:
                    logger.debug("Failed to refresh runtime status for platform %s", platform, exc_info=True)
        except Exception as exc:
            logger.debug("Failed to write runtime status snapshot: %s", exc)

    async def _runtime_status_heartbeat(self, interval: float = 15.0) -> None:
        self._write_runtime_status_snapshot()
        while self._running:
            for _ in range(max(1, int(interval))):
                if not self._running:
                    return
                await asyncio.sleep(1)
            self._write_runtime_status_snapshot()

    def _try_handle_background_job_status_shortcut(self, event: MessageEvent) -> str | None:
        return shared_try_handle_background_job_status_shortcut(self, event)

    def _try_handle_runtime_status_shortcut(self, event: MessageEvent) -> str | None:
        return shared_try_handle_runtime_status_shortcut(
            self,
            event,
            pending_sentinel=_AGENT_PENDING_SENTINEL,
        )

    def _get_direct_control_router(self) -> DirectControlRouter:
        return shared_get_direct_control_router(self, router_cls=DirectControlRouter)

    def _try_handle_direct_gateway_shortcuts(
        self,
        event: MessageEvent,
        *,
        prepare_session_env: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str | None:
        return shared_try_handle_direct_gateway_shortcuts(
            self,
            event,
            prepare_session_env=prepare_session_env,
            conversation_history=conversation_history,
            logger=logger,
        )

    @staticmethod
    def _sanitize_background_visible_text(text: str) -> str:
        return shared_sanitize_background_visible_text(text)

    @staticmethod
    def _background_completion_should_stay_silent(*, job_kind: str, worker_name: str = "") -> bool:
        return shared_background_completion_should_stay_silent(
            job_kind=job_kind,
            worker_name=worker_name,
        )

    @staticmethod
    def _build_background_delivery_header(
        *,
        task_id: str,
        preview: str = "",
        worker_name: str = "",
        state: str = "completed",
    ) -> str:
        return shared_build_background_delivery_header(
            task_id=task_id,
            preview=preview,
            worker_name=worker_name,
            state=state,
        )

    def _start_background_job(
        self,
        prompt: str,
        source: SessionSource,
        *,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        context_prompt: str = "",
        session_key: str = "",
        job_kind: str = "manual",
        worker_name: str = "",
        preloaded_skills: Optional[List[str]] = None,
        admin_user_ids: Optional[List[str]] = None,
        is_admin_user: Optional[bool] = None,
    ) -> str:
        return shared_start_background_job(
            store=self._get_background_job_store(),
            launch_worker=self._launch_background_worker,
            prompt=prompt,
            source=source,
            conversation_history=conversation_history,
            context_prompt=context_prompt,
            session_key=session_key,
            job_kind=job_kind,
            worker_name=worker_name,
            preloaded_skills=list(preloaded_skills or []),
            admin_user_ids=list(admin_user_ids or []),
            is_admin_user=is_admin_user,
            logger=logger,
        )

    async def _maybe_auto_compress_session_history(
        self,
        *,
        history: Optional[List[Dict[str, Any]]],
        session_entry: SessionEntry,
    ) -> List[Dict[str, Any]]:
        return await shared_maybe_auto_compress_session_history(
            history=history,
            session_entry=session_entry,
            session_store=self.session_store,
            hermes_home=_hermes_home,
            runtime_agent_kwargs_loader=_resolve_runtime_agent_kwargs,
            logger=logger,
        )

    def _prepare_history_for_agent(
        self,
        *,
        history: List[Dict[str, Any]],
        context: SessionContext,
        session_entry: SessionEntry,
    ) -> List[Dict[str, Any]]:
        return shared_prepare_history_for_agent(
            history,
            shared_session_kind=getattr(context, "shared_session_kind", None),
            session_id=session_entry.session_id,
            logger=logger,
            visible_limit=_SHARED_GROUP_VISIBLE_HISTORY_LIMIT,
        )

    def _resolve_background_job_for_stop(
        self,
        source: SessionSource,
        raw_task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a background job for /stop within the current chat."""
        active_jobs = self._background_jobs_for_source(source, active_only=True)
        if not active_jobs:
            return None

        task_id = str(raw_task_id or "").strip()
        if task_id:
            for job in active_jobs:
                if job.get("task_id") == task_id or str(job.get("task_id", "")).startswith(task_id):
                    return job
            return None

        if len(active_jobs) == 1:
            return active_jobs[0]
        return {"ambiguous": True, "jobs": active_jobs}
    
    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """
        Handle an incoming message from any platform.
        
        This is the core message processing pipeline:
        1. Check user authorization
        2. Check for commands (/new, /reset, etc.)
        3. Check for running agent and interrupt if needed
        4. Get or create session
        5. Build context for agent
        6. Run agent conversation
        7. Return response
        """
        source = event.source

        # Check if user is authorized
        if not self._is_user_authorized(source):
            logger.warning("Unauthorized user: %s (%s) on %s", source.user_id, source.user_name, source.platform.value)
            # In DMs: offer pairing code. In groups: silently ignore.
            if source.chat_type == "dm" and self._get_unauthorized_dm_behavior(source.platform) == "pair":
                platform_name = source.platform.value if source.platform else "unknown"
                # Rate-limit ALL pairing responses (code or rejection) to
                # prevent spamming the user with repeated messages when
                # multiple DMs arrive in quick succession.
                if self.pairing_store._is_rate_limited(platform_name, source.user_id):
                    return None
                code = self.pairing_store.generate_code(
                    platform_name, source.user_id, source.user_name or ""
                )
                if code:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            f"Hi~ I don't recognize you yet!\n\n"
                            f"Here's your pairing code: `{code}`\n\n"
                            f"Ask the bot owner to run:\n"
                            f"`hermes pairing approve {platform_name} {code}`"
                        )
                else:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            "Too many pairing requests right now~ "
                            "Please try again later!"
                        )
                    # Record rate limit so subsequent messages are silently ignored
                    self.pairing_store._record_rate_limit(platform_name, source.user_id)
            return None
        
        # Intercept messages that are responses to a pending /update prompt.
        # The update process (detached) wrote .update_prompt.json; the watcher
        # forwarded it to the user; now the user's reply goes back via
        # .update_response so the update process can continue.
        _quick_key = self._session_key_for_source(source)
        _update_prompts = getattr(self, "_update_prompt_pending", {})
        if _update_prompts.get(_quick_key):
            raw = (event.text or "").strip()
            # Accept /approve and /deny as shorthand for yes/no
            cmd = event.get_command()
            if cmd in ("approve", "yes"):
                response_text = "y"
            elif cmd in ("deny", "no"):
                response_text = "n"
            else:
                response_text = raw
            if response_text:
                response_path = _hermes_home / ".update_response"
                try:
                    tmp = response_path.with_suffix(".tmp")
                    tmp.write_text(response_text)
                    tmp.replace(response_path)
                except OSError as e:
                    logger.warning("Failed to write update response: %s", e)
                    return f"✗ Failed to send response to update process: {e}"
                _update_prompts.pop(_quick_key, None)
                label = response_text if len(response_text) <= 20 else response_text[:20] + "…"
                return f"✓ Sent `{label}` to the update process."

        raw_text = (event.text or "").strip()
        if raw_text and not raw_text.startswith("/") and self._is_runtime_identity_query(raw_text):
            return self._format_runtime_identity_response(source)

        # PRIORITY handling when an agent is already running for this session.
        # Default behavior is to interrupt immediately so user text/stop messages
        # are handled with minimal latency.
        #
        # Special case: Telegram/photo bursts often arrive as multiple near-
        # simultaneous updates. Do NOT interrupt for photo-only follow-ups here;
        # let the adapter-level batching/queueing logic absorb them.

        # Staleness eviction: detect leaked locks from hung/crashed handlers.
        # With inactivity-based timeout, active tasks can run for hours, so
        # wall-clock age alone isn't sufficient.  Evict only when the agent
        # has been *idle* beyond the inactivity threshold (or when the agent
        # object has no activity tracker and wall-clock age is extreme).
        _raw_stale_timeout = float(os.getenv("HERMES_AGENT_TIMEOUT", 1800))
        _stale_ts = self._running_agents_ts.get(_quick_key, 0)
        if _quick_key in self._running_agents and _stale_ts:
            _stale_age = time.time() - _stale_ts
            _stale_agent = self._running_agents.get(_quick_key)
            # Never evict the pending sentinel — it was just placed moments
            # ago during the async setup phase before the real agent is
            # created.  Sentinels have no get_activity_summary(), so the
            # idle check below must stay conservative or tests/mocks and
            # partially-constructed agents get evicted spuriously.
            _stale_idle = 0.0
            _stale_detail = ""
            if _stale_agent and hasattr(_stale_agent, "get_activity_summary"):
                try:
                    _sa = _stale_agent.get_activity_summary()
                    if isinstance(_sa, dict):
                        _stale_idle = _safe_float(
                            _sa.get("seconds_since_activity"),
                            float("inf"),
                        )
                        _stale_detail = (
                            f" | last_activity={_sa.get('last_activity_desc', 'unknown')} "
                            f"({_stale_idle:.0f}s ago) "
                            f"| iteration={_sa.get('api_call_count', 0)}/{_sa.get('max_iterations', 0)}"
                        )
                except Exception:
                    pass
            # Evict if: agent is idle beyond timeout, OR wall-clock age is
            # extreme (10x timeout or 2h, whichever is larger — catches
            # cases where the agent object was garbage-collected).
            _wall_ttl = max(_raw_stale_timeout * 10, 7200) if _raw_stale_timeout > 0 else float("inf")
            _should_evict = (
                _stale_agent is not _AGENT_PENDING_SENTINEL
                and (
                    (_raw_stale_timeout > 0 and _stale_idle >= _raw_stale_timeout)
                    or _stale_age > _wall_ttl
                )
            )
            if _should_evict:
                logger.warning(
                    "Evicting stale _running_agents entry for %s "
                    "(age: %.0fs, idle: %.0fs, timeout: %.0fs)%s",
                    _quick_key[:30], _stale_age, _stale_idle,
                    _raw_stale_timeout, _stale_detail,
                )
                del self._running_agents[_quick_key]
                self._running_agents_ts.pop(_quick_key, None)

        if _quick_key in self._running_agents:
            if event.get_command() == "status":
                return await self._handle_status_command(event)
            shortcut_history = None
            try:
                shortcut_session = self.session_store.get_or_create_session(source)
                shortcut_history = self.session_store.load_transcript(shortcut_session.session_id)
            except Exception:
                shortcut_history = None
            direct_shortcut_response = self._try_handle_direct_gateway_shortcuts(
                event,
                prepare_session_env=True,
                conversation_history=list(shortcut_history or []),
            )
            if direct_shortcut_response is not None:
                return direct_shortcut_response
            _busy_input_mode = self._get_busy_input_mode(source.platform)

            # Resolve the command once for all early-intercept checks below.
            from hermes_cli.commands import resolve_command as _resolve_cmd_inner
            _evt_cmd = event.get_command()
            _cmd_def_inner = _resolve_cmd_inner(_evt_cmd) if _evt_cmd else None

            # /stop must hard-kill the session when an agent is running.
            # A soft interrupt (agent.interrupt()) doesn't help when the agent
            # is truly hung — the executor thread is blocked and never checks
            # _interrupt_requested.  Force-clean _running_agents so the session
            # is unlocked and subsequent messages are processed normally.
            if _cmd_def_inner and _cmd_def_inner.name == "stop":
                running_agent = self._running_agents.get(_quick_key)
                if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
                    running_agent.interrupt("Stop requested")
                # Force-clean: remove the session lock regardless of agent state
                adapter = self.adapters.get(source.platform)
                if adapter and hasattr(adapter, 'get_pending_message'):
                    adapter.get_pending_message(_quick_key)  # consume and discard
                self._pending_messages.pop(_quick_key, None)
                if _quick_key in self._running_agents:
                    del self._running_agents[_quick_key]
                logger.info("HARD STOP for session %s — session lock released", _quick_key[:20])
                return "⚡ Force-stopped. The session is unlocked — you can send a new message."

            # /reset and /new must bypass the running-agent guard so they
            # actually dispatch as commands instead of being queued as user
            # text (which would be fed back to the agent with the same
            # broken history — #2170).  Interrupt the agent first, then
            # clear the adapter's pending queue so the stale "/reset" text
            # doesn't get re-processed as a user message after the
            # interrupt completes.
            if _cmd_def_inner and _cmd_def_inner.name == "new":
                running_agent = self._running_agents.get(_quick_key)
                if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
                    running_agent.interrupt("Session reset requested")
                # Clear any pending messages so the old text doesn't replay
                adapter = self.adapters.get(source.platform)
                if adapter and hasattr(adapter, 'get_pending_message'):
                    adapter.get_pending_message(_quick_key)  # consume and discard
                self._pending_messages.pop(_quick_key, None)
                # Clean up the running agent entry so the reset handler
                # doesn't think an agent is still active.
                if _quick_key in self._running_agents:
                    del self._running_agents[_quick_key]
                return await self._handle_reset_command(event)

            # /queue <prompt> — queue without interrupting
            if event.get_command() in ("queue", "q"):
                queued_text = event.get_command_args().strip()
                if not queued_text:
                    return "Usage: /queue <prompt>"
                adapter = self.adapters.get(source.platform)
                if adapter:
                    from gateway.platforms.base import MessageEvent as _ME, MessageType as _MT
                    queued_event = _ME(
                        text=queued_text,
                        message_type=_MT.TEXT,
                        source=event.source,
                        message_id=event.message_id,
                    )
                    if hasattr(adapter, "queue_message"):
                        adapter.queue_message(_quick_key, queued_event)
                    else:
                        adapter._pending_messages[_quick_key] = queued_event
                return "Queued for the next turn."

            # /model must not be used while the agent is running.
            if _cmd_def_inner and _cmd_def_inner.name == "model":
                return "Agent is running — wait or /stop first, then switch models."

            # /approve and /deny must bypass the running-agent interrupt path.
            # The agent thread is blocked on a threading.Event inside
            # tools/approval.py — sending an interrupt won't unblock it.
            # Route directly to the approval handler so the event is signalled.
            if _cmd_def_inner and _cmd_def_inner.name in ("approve", "deny"):
                if _cmd_def_inner.name == "approve":
                    return await self._handle_approve_command(event)
                return await self._handle_deny_command(event)

            adapter = self.adapters.get(source.platform)
            _explicit_followup = getattr(source, "chat_type", "") == "dm"
            if (
                not _explicit_followup
                and adapter
                and hasattr(adapter, "_is_explicit_busy_followup")
            ):
                try:
                    _explicit_followup = bool(adapter._is_explicit_busy_followup(event))
                except Exception:
                    _explicit_followup = False

            if event.message_type == MessageType.TEXT and not _evt_cmd and _explicit_followup:
                try:
                    from tools.approval import (
                        has_blocking_approval,
                        peek_blocking_approval,
                        resolve_gateway_approval,
                    )

                    if has_blocking_approval(_quick_key):
                        admin_only_message = self._admin_only_message(
                            source,
                            "deny dangerous commands",
                        )
                        if admin_only_message is None:
                            current_approval = peek_blocking_approval(_quick_key) or {}
                            resolved = resolve_gateway_approval(
                                _quick_key,
                                "deny",
                                resolve_all=True,
                            )
                            if resolved:
                                if adapter:
                                    adapter.resume_typing_for_chat(source.chat_id)
                                running_agent = self._running_agents.get(_quick_key)
                                if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
                                    running_agent.interrupt(event.text)

                                followup_text = str(event.text or "").strip()
                                if followup_text:
                                    if _quick_key in self._pending_messages:
                                        self._pending_messages[_quick_key] += "\n" + followup_text
                                    else:
                                        self._pending_messages[_quick_key] = followup_text

                                cmd_preview = _truncate_status_preview(
                                    current_approval.get("command", "")
                                )
                                if resolved > 1:
                                    return (
                                        f"刚才挂起的 {resolved} 条危险命令我先给你拒了。"
                                        "你这条我接着处理。"
                                    )
                                if cmd_preview:
                                    return (
                                        f"刚才那条危险命令我先给你拒了：{cmd_preview}。"
                                        "你这条我接着处理。"
                                    )
                                return "刚才那条危险命令我先给你拒了。你这条我接着处理。"
                except Exception:
                    pass

            if event.message_type == MessageType.PHOTO:
                logger.debug("PRIORITY photo follow-up for session %s — queueing without interrupt", _quick_key[:20])
                if adapter:
                    if hasattr(adapter, "queue_message"):
                        adapter.queue_message(_quick_key, event)
                    else:
                        adapter._pending_messages[_quick_key] = event
                return None

            running_agent = self._running_agents.get(_quick_key)
            if running_agent is _AGENT_PENDING_SENTINEL:
                # Agent is being set up but not ready yet.
                if event.get_command() == "stop":
                    # Force-clean the sentinel so the session is unlocked.
                    if _quick_key in self._running_agents:
                        del self._running_agents[_quick_key]
                    logger.info("HARD STOP (pending) for session %s — sentinel cleared", _quick_key[:20])
                    return "⚡ Force-stopped. The agent was still starting — session unlocked."
                # Queue the message so it will be picked up after the
                # agent starts.
                adapter = self.adapters.get(source.platform)
                if adapter:
                    if hasattr(adapter, "queue_message"):
                        adapter.queue_message(_quick_key, event)
                    else:
                        adapter._pending_messages[_quick_key] = event
                busy_ack = ""
                if adapter and hasattr(adapter, "_busy_followup_ack"):
                    busy_ack = adapter._busy_followup_ack(event, interrupting=False)
                elif _busy_input_mode == "queue":
                    busy_ack = _qq_busy_followup_ack(source, event.text)
                if busy_ack:
                    logger.info(
                        "queued follow-up while session pending: platform=%s chat=%s session=%s",
                        source.platform.value if getattr(source, "platform", None) else "unknown",
                        source.chat_id or "unknown",
                        _quick_key[:32],
                    )
                    return busy_ack
                return None
            _force_queue_reason = self._busy_followup_force_queue_reason(
                _quick_key,
                running_agent,
            )
            if _force_queue_reason:
                logger.info(
                    "PRIORITY force-queue for session %s — preserving active run (%s)",
                    _quick_key[:20],
                    _force_queue_reason,
                )
                if adapter:
                    if hasattr(adapter, "queue_message"):
                        adapter.queue_message(_quick_key, event)
                    else:
                        adapter._pending_messages[_quick_key] = event
                busy_ack = ""
                if adapter and hasattr(adapter, "_busy_followup_ack"):
                    busy_ack = adapter._busy_followup_ack(event, interrupting=False)
                elif _busy_input_mode == "queue":
                    busy_ack = _qq_busy_followup_ack(source, event.text)
                if busy_ack:
                    return busy_ack
                return None
            if _busy_input_mode == "queue":
                logger.debug(
                    "PRIORITY queue for session %s — deferring follow-up without interrupt",
                    _quick_key[:20],
                )
                if adapter:
                    if hasattr(adapter, "queue_message"):
                        adapter.queue_message(_quick_key, event)
                    else:
                        adapter._pending_messages[_quick_key] = event
                busy_ack = ""
                if adapter and hasattr(adapter, "_busy_followup_ack"):
                    busy_ack = adapter._busy_followup_ack(event, interrupting=False)
                else:
                    busy_ack = _qq_busy_followup_ack(source, event.text)
                if busy_ack:
                    logger.info(
                        "queued follow-up for active session: platform=%s chat=%s session=%s",
                        source.platform.value if getattr(source, "platform", None) else "unknown",
                        source.chat_id or "unknown",
                        _quick_key[:32],
                    )
                    return busy_ack
                return None

            if _busy_input_mode == "smart":
                should_interrupt = False
                if adapter and hasattr(adapter, "_should_interrupt_busy_followup"):
                    try:
                        if (
                            hasattr(adapter, "_active_session_started_at")
                            and _quick_key not in adapter._active_session_started_at
                            and _quick_key in self._running_agents_ts
                        ):
                            adapter._active_session_started_at[_quick_key] = self._running_agents_ts[_quick_key]
                        should_interrupt = bool(adapter._should_interrupt_busy_followup(_quick_key, event))
                    except Exception as exc:
                        logger.debug("smart busy follow-up decision failed for %s: %s", _quick_key[:20], exc)
                if not should_interrupt:
                    logger.debug(
                        "PRIORITY smart-queue for session %s — deferring follow-up during grace window",
                        _quick_key[:20],
                    )
                    if adapter:
                        if hasattr(adapter, "queue_message"):
                            adapter.queue_message(_quick_key, event)
                        else:
                            adapter._pending_messages[_quick_key] = event
                    busy_ack = ""
                    if adapter and hasattr(adapter, "_busy_followup_ack"):
                        busy_ack = adapter._busy_followup_ack(event, interrupting=False)
                    if busy_ack:
                        logger.info(
                            "smart-queued follow-up for active session: platform=%s chat=%s session=%s",
                            source.platform.value if getattr(source, "platform", None) else "unknown",
                            source.chat_id or "unknown",
                            _quick_key[:32],
                        )
                        return busy_ack
                    return None

                logger.info(
                    "PRIORITY smart-interrupt for session %s — switching to fresher follow-up",
                    _quick_key[:20],
                )
                if adapter:
                    if hasattr(adapter, "queue_message"):
                        adapter.queue_message(_quick_key, event)
                    else:
                        adapter._pending_messages[_quick_key] = event
                running_agent.interrupt(event.text)
                busy_ack = ""
                if adapter and hasattr(adapter, "_busy_followup_ack"):
                    busy_ack = adapter._busy_followup_ack(event, interrupting=True)
                if busy_ack:
                    return busy_ack
                return None

            logger.debug("PRIORITY interrupt for session %s", _quick_key[:20])
            running_agent.interrupt(event.text)
            if _quick_key in self._pending_messages:
                self._pending_messages[_quick_key] += "\n" + event.text
            else:
                self._pending_messages[_quick_key] = event.text
            return None

        # Check for commands
        command = event.get_command()
        
        # Emit command:* hook for any recognized slash command.
        # GATEWAY_KNOWN_COMMANDS is derived from the central COMMAND_REGISTRY
        # in hermes_cli/commands.py — no hardcoded set to maintain here.
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command as _resolve_cmd
        if command and command in GATEWAY_KNOWN_COMMANDS:
            await self.hooks.emit(f"command:{command}", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "command": command,
                "args": event.get_command_args().strip(),
            })

        # Resolve aliases to canonical name so dispatch only checks canonicals.
        _cmd_def = _resolve_cmd(command) if command else None
        canonical = _cmd_def.name if _cmd_def else command

        if canonical == "new":
            return await self._handle_reset_command(event)
        
        if canonical == "help":
            return await self._handle_help_command(event)

        if canonical == "commands":
            return await self._handle_commands_command(event)
        
        if canonical == "profile":
            return await self._handle_profile_command(event)

        if canonical == "status":
            return await self._handle_status_command(event)
        
        if canonical == "stop":
            return await self._handle_stop_command(event)
        
        if canonical == "reasoning":
            return await self._handle_reasoning_command(event)

        if canonical == "verbose":
            return await self._handle_verbose_command(event)

        if canonical == "yolo":
            return await self._handle_yolo_command(event)

        if canonical == "model":
            return await self._handle_model_command(event)

        if canonical == "provider":
            return await self._handle_provider_command(event)
        
        if canonical == "personality":
            return await self._handle_personality_command(event)

        if canonical == "plan":
            try:
                from agent.skill_commands import build_plan_path, build_skill_invocation_message

                user_instruction = event.get_command_args().strip()
                plan_path = build_plan_path(user_instruction)
                event.text = build_skill_invocation_message(
                    "/plan",
                    user_instruction,
                    task_id=_quick_key,
                    runtime_note=(
                        "Save the markdown plan with write_file to this exact relative path "
                        f"inside the active workspace/backend cwd: {plan_path}"
                    ),
                )
                if not event.text:
                    return "Failed to load the bundled /plan skill."
                canonical = None
            except Exception as e:
                logger.exception("Failed to prepare /plan command")
                return f"Failed to enter plan mode: {e}"
        
        if canonical == "retry":
            return await self._handle_retry_command(event)
        
        if canonical == "undo":
            return await self._handle_undo_command(event)
        
        if canonical == "sethome":
            return await self._handle_set_home_command(event)

        if canonical == "compress":
            return await self._handle_compress_command(event)

        if canonical == "usage":
            return await self._handle_usage_command(event)

        if canonical == "insights":
            return await self._handle_insights_command(event)

        if canonical == "reload-mcp":
            return await self._handle_reload_mcp_command(event)

        if canonical == "approve":
            return await self._handle_approve_command(event)

        if canonical == "deny":
            return await self._handle_deny_command(event)

        if canonical == "update":
            return await self._handle_update_command(event)

        if canonical == "title":
            return await self._handle_title_command(event)

        if canonical == "resume":
            return await self._handle_resume_command(event)

        if canonical == "branch":
            return await self._handle_branch_command(event)

        if canonical == "rollback":
            return await self._handle_rollback_command(event)

        if canonical == "background":
            return await self._handle_background_command(event)

        if canonical == "btw":
            return await self._handle_btw_command(event)

        if canonical == "voice":
            return await self._handle_voice_command(event)

        # User-defined quick commands (bypass agent loop, no LLM call)
        if command:
            if isinstance(self.config, dict):
                quick_commands = self.config.get("quick_commands", {}) or {}
            else:
                quick_commands = getattr(self.config, "quick_commands", {}) or {}
            if not isinstance(quick_commands, dict):
                quick_commands = {}
            if command in quick_commands:
                qcmd = quick_commands[command]
                if qcmd.get("type") == "exec":
                    exec_cmd = qcmd.get("command", "")
                    if exec_cmd:
                        try:
                            proc = await asyncio.create_subprocess_shell(
                                exec_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                            output = (stdout or stderr).decode().strip()
                            return output if output else "Command returned no output."
                        except asyncio.TimeoutError:
                            return "Quick command timed out (30s)."
                        except Exception as e:
                            return f"Quick command error: {e}"
                    else:
                        return f"Quick command '/{command}' has no command defined."
                elif qcmd.get("type") == "alias":
                    target = qcmd.get("target", "").strip()
                    if target:
                        target = target if target.startswith("/") else f"/{target}"
                        target_command = target.lstrip("/")
                        user_args = event.get_command_args().strip()
                        event.text = f"{target} {user_args}".strip()
                        command = target_command
                        # Fall through to normal command dispatch below
                    else:
                        return f"Quick command '/{command}' has no target defined."
                else:
                    return f"Quick command '/{command}' has unsupported type (supported: 'exec', 'alias')."

        # Plugin-registered slash commands
        if command:
            try:
                from hermes_cli.plugins import get_plugin_command_handler
                # Normalize underscores to hyphens so Telegram's underscored
                # autocomplete form matches plugin commands registered with
                # hyphens. See hermes_cli/commands.py:_build_telegram_menu.
                plugin_handler = get_plugin_command_handler(command.replace("_", "-"))
                if plugin_handler:
                    user_args = event.get_command_args().strip()
                    import asyncio as _aio
                    result = plugin_handler(user_args)
                    if _aio.iscoroutine(result):
                        result = await result
                    return str(result) if result else None
            except Exception as e:
                logger.debug("Plugin command dispatch failed (non-fatal): %s", e)

        # Skill slash commands: /skill-name loads the skill and sends to agent.
        # resolve_skill_command_key() handles the Telegram underscore/hyphen
        # round-trip so /claude_code from Telegram autocomplete still resolves
        # to the claude-code skill.
        if command:
            try:
                from agent.skill_commands import (
                    get_skill_commands,
                    build_skill_invocation_message,
                    resolve_skill_command_key,
                )
                skill_cmds = get_skill_commands()
                cmd_key = resolve_skill_command_key(command)
                if cmd_key is not None:
                    # Check per-platform disabled status before executing.
                    # get_skill_commands() only applies the *global* disabled
                    # list at scan time; per-platform overrides need checking
                    # here because the cache is process-global across platforms.
                    _skill_name = skill_cmds[cmd_key].get("name", "")
                    _plat = source.platform.value if source.platform else None
                    if _plat and _skill_name:
                        from agent.skill_utils import get_disabled_skill_names as _get_plat_disabled
                        if _skill_name in _get_plat_disabled(platform=_plat):
                            return (
                                f"The **{_skill_name}** skill is disabled for {_plat}.\n"
                                f"Enable it with: `hermes skills config`"
                            )
                    user_instruction = event.get_command_args().strip()
                    msg = build_skill_invocation_message(
                        cmd_key, user_instruction, task_id=_quick_key
                    )
                    if msg:
                        event.text = msg
                        # Fall through to normal message processing with skill content
                else:
                    # Not an active skill — check if it's a known-but-disabled or
                    # uninstalled skill and give actionable guidance.
                    _unavail_msg = _check_unavailable_skill(command)
                    if _unavail_msg:
                        return _unavail_msg
                    # Genuinely unrecognized /command: not a built-in, not a
                    # plugin, not a skill, not a known-inactive skill. Warn
                    # the user instead of silently forwarding it to the LLM
                    # as free text (which leads to silent-failure behavior
                    # like the model inventing a delegate_task call).
                    # Normalize to hyphenated form before checking known
                    # built-ins (command may be an alias target set by the
                    # quick-command block above, so _cmd_def can be stale).
                    if command.replace("_", "-") not in GATEWAY_KNOWN_COMMANDS:
                        logger.warning(
                            "Unrecognized slash command /%s from %s — "
                            "replying with unknown-command notice",
                            command,
                            source.platform.value if source.platform else "?",
                        )
                        return (
                            f"Unknown command `/{command}`. "
                            f"Type /commands to see what's available, "
                            f"or resend without the leading slash to send "
                            f"as a regular message."
                        )
            except Exception as e:
                logger.debug("Skill command check failed (non-fatal): %s", e)
        
        # Pending exec approvals are handled by /approve and /deny commands above.
        # No bare text matching — "yes" in normal conversation must not trigger
        # execution of a dangerous command.

        # ── Claim this session before any await ───────────────────────
        # Between here and _run_agent registering the real AIAgent, there
        # are numerous await points (hooks, vision enrichment, STT,
        # session hygiene compression).  Without this sentinel a second
        # message arriving during any of those yields would pass the
        # "already running" guard and spin up a duplicate agent for the
        # same session — corrupting the transcript.
        self._running_agents[_quick_key] = _AGENT_PENDING_SENTINEL
        self._running_agents_ts[_quick_key] = time.time()

        try:
            return await self._handle_message_with_agent(event, source, _quick_key)
        finally:
            # If _run_agent replaced the sentinel with a real agent and
            # then cleaned it up, this is a no-op.  If we exited early
            # (exception, command fallthrough, etc.) the sentinel must
            # not linger or the session would be permanently locked out.
            if self._running_agents.get(_quick_key) is _AGENT_PENDING_SENTINEL:
                del self._running_agents[_quick_key]
            self._running_agents_ts.pop(_quick_key, None)

    async def _handle_message_with_agent(self, event, source, _quick_key: str):
        """Inner handler that runs under the _running_agents sentinel guard."""
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        _msg_preview = (event.text or "")[:80].replace("\n", " ")
        logger.info(
            "inbound message: platform=%s user=%s chat=%s msg=%r",
            _platform_name, source.user_name or source.user_id or "unknown",
            source.chat_id or "unknown", _msg_preview,
        )

        # Get or create session
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        # Emit session:start for new or auto-reset sessions
        _is_new_session = (
            session_entry.created_at == session_entry.updated_at
            or getattr(session_entry, "was_auto_reset", False)
        )
        if _is_new_session:
            await self.hooks.emit("session:start", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            })
        
        # Build session context
        admin_user_ids = self._configured_admin_user_ids(source.platform)
        is_admin_user = self._is_admin_user(source) if admin_user_ids else None
        context = build_session_context(
            source,
            self.config,
            session_entry,
            admin_user_ids=admin_user_ids,
            is_admin_user=is_admin_user,
        )
        
        # Set environment variables for tools
        self._set_session_env(context)

        history = self.session_store.load_transcript(session_entry.session_id)

        direct_shortcut_response = self._try_handle_direct_gateway_shortcuts(
            event,
            conversation_history=list(history or []),
        )
        if direct_shortcut_response is not None:
            return direct_shortcut_response
        
        # Read privacy.redact_pii from config (re-read per message)
        _redact_pii = False
        try:
            import yaml as _pii_yaml
            with open(_config_path, encoding="utf-8") as _pf:
                _pcfg = _pii_yaml.safe_load(_pf) or {}
            _redact_pii = bool((_pcfg.get("privacy") or {}).get("redact_pii", False))
        except Exception:
            pass

        # Build the context prompt to inject
        context_prompt = build_session_context_prompt(context, redact_pii=_redact_pii)
        explicit_group_reply_note = _explicit_group_reply_context_note(event)
        if explicit_group_reply_note:
            context_prompt = f"{context_prompt}\n\n{explicit_group_reply_note}"
        
        # If the previous session expired and was auto-reset, prepend a notice
        # so the agent knows this is a fresh conversation (not an intentional /reset).
        if getattr(session_entry, 'was_auto_reset', False):
            reset_reason = getattr(session_entry, 'auto_reset_reason', None) or 'idle'
            if reset_reason == "daily":
                context_note = "[System note: The user's session was automatically reset by the daily schedule. This is a fresh conversation with no prior context.]"
            else:
                context_note = "[System note: The user's previous session expired due to inactivity. This is a fresh conversation with no prior context.]"
            context_prompt = context_note + "\n\n" + context_prompt

            # Send a user-facing notification explaining the reset, unless:
            # - notifications are disabled in config
            # - the platform is excluded (e.g. api_server, webhook)
            # - the expired session had no activity (nothing was cleared)
            try:
                policy = self.session_store.config.get_reset_policy(
                    platform=source.platform,
                    session_type=getattr(source, 'chat_type', 'dm'),
                )
                platform_name = source.platform.value if source.platform else ""
                had_activity = getattr(session_entry, 'reset_had_activity', False)
                should_notify = (
                    policy.notify
                    and had_activity
                    and platform_name not in policy.notify_exclude_platforms
                )
                if should_notify:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        if reset_reason == "daily":
                            reason_text = f"daily schedule at {policy.at_hour}:00"
                        else:
                            hours = policy.idle_minutes // 60
                            mins = policy.idle_minutes % 60
                            duration = f"{hours}h" if not mins else f"{hours}h {mins}m" if hours else f"{mins}m"
                            reason_text = f"inactive for {duration}"
                        notice = (
                            f"◐ Session automatically reset ({reason_text}). "
                            f"Conversation history cleared.\n"
                            f"Use /resume to browse and restore a previous session.\n"
                            f"Adjust reset timing in config.yaml under session_reset."
                        )
                        try:
                            session_info = self._format_session_info()
                            if session_info:
                                notice = f"{notice}\n\n{session_info}"
                        except Exception:
                            pass
                        await adapter.send(
                            source.chat_id, notice,
                            metadata=getattr(event, 'metadata', None),
                        )
            except Exception as e:
                logger.debug("Auto-reset notification failed (non-fatal): %s", e)

            session_entry.was_auto_reset = False
            session_entry.auto_reset_reason = None

        # Auto-load skill for DM topic bindings (e.g., Telegram Private Chat Topics)
        # Only inject on NEW sessions — for ongoing conversations the skill content
        # is already in the conversation history from the first message.
        if _is_new_session and getattr(event, "auto_skill", None):
            try:
                from agent.skill_commands import _load_skill_payload, _build_skill_message
                _skill_name = event.auto_skill
                _loaded = _load_skill_payload(_skill_name, task_id=_quick_key)
                if _loaded:
                    _loaded_skill, _skill_dir, _display_name = _loaded
                    _activation_note = (
                        f'[SYSTEM: This conversation is in a topic with the "{_display_name}" skill '
                        f"auto-loaded. Follow its instructions for the duration of this session.]"
                    )
                    _skill_msg = _build_skill_message(
                        _loaded_skill, _skill_dir, _activation_note,
                        user_instruction=event.text,
                    )
                    if _skill_msg:
                        event.text = _skill_msg
                        logger.info(
                            "[Gateway] Auto-loaded skill '%s' for DM topic session %s",
                            _skill_name, session_key,
                        )
                else:
                    logger.warning(
                        "[Gateway] DM topic skill '%s' not found in available skills",
                        _skill_name,
                    )
            except Exception as e:
                logger.warning("[Gateway] Failed to auto-load topic skill '%s': %s", event.auto_skill, e)

        # Load conversation history from transcript
        history_for_agent = self._prepare_history_for_agent(
            history=history,
            context=context,
            session_entry=session_entry,
        )

        background_message_text = event.text or ""
        background_message_text = _prepend_shared_thread_sender(
            message_text=background_message_text,
            source=source,
            thread_sessions_per_user=bool(
                getattr(self.config, "thread_sessions_per_user", False)
            ),
        )
        background_message_text = _prepend_reply_context_if_missing(
            message_text=background_message_text,
            event=event,
            history=history,
        )
        background_dispatch = shared_resolve_auto_background_dispatch(
            event,
            background_message_text,
            auto_background_work_enabled=self._get_auto_background_work(
                getattr(event.source, "platform", None)
            ),
            employee_routes=get_employee_routes(
                self.config,
                platform=getattr(event.source, "platform", Platform.QQ_NAPCAT),
            ),
            conversation_history=list(history_for_agent or []),
        )
        if background_dispatch:
            task_id = self._start_background_job(
                background_message_text,
                source,
                conversation_history=list(history_for_agent or []),
                context_prompt=context_prompt,
                session_key=session_key,
                job_kind="auto",
                worker_name=str(background_dispatch.get("worker_name") or ""),
                preloaded_skills=list(background_dispatch.get("preloaded_skills") or []),
                admin_user_ids=context.admin_user_ids,
                is_admin_user=context.is_admin_user,
            )
            return shared_format_auto_background_ack(
                background_message_text,
                task_id,
                worker_name=str(background_dispatch.get("worker_name") or ""),
            )

        history = await self._maybe_auto_compress_session_history(
            history=history,
            session_entry=session_entry,
        )

        # First-message onboarding -- only on the very first interaction ever
        context_prompt = _append_first_message_onboarding_note(
            context_prompt,
            history=history,
            has_any_sessions=self.session_store.has_any_sessions(),
        )
        
        # One-time prompt if no home channel is set for this platform
        # Skip for webhooks - they deliver directly to configured targets (github_comment, etc.)
        if _should_prompt_for_home_channel(
            history=history,
            platform=source.platform,
        ):
            adapter = self.adapters.get(source.platform)
            if adapter and source.platform is not None:
                await adapter.send(
                    source.chat_id,
                    shared_build_home_channel_prompt(source.platform),
                )
        
        # -----------------------------------------------------------------
        # Voice channel awareness — inject current voice channel state
        # into context so the agent knows who is in the channel and who
        # is speaking, without needing a separate tool call.
        # -----------------------------------------------------------------
        if source.platform == Platform.DISCORD:
            adapter = self.adapters.get(Platform.DISCORD)
            guild_id = self._get_guild_id(event)
            context_prompt = _append_discord_voice_channel_context(
                context_prompt,
                source=source,
                guild_id=guild_id,
                adapter=adapter,
            )

        # -----------------------------------------------------------------
        # Auto-analyze images sent by the user
        #
        # If the user attached image(s), we run the vision tool eagerly so
        # the conversation model always receives a text description.  The
        # local file path is also included so the model can re-examine the
        # image later with a more targeted question via vision_analyze.
        #
        # We filter to image paths only (by media_type) so that non-image
        # attachments (documents, audio, etc.) are not sent to the vision
        # tool even when they appear in the same message.
        # -----------------------------------------------------------------
        raw_message_text = event.text or ""
        message_text = raw_message_text

        # -----------------------------------------------------------------
        # Sender attribution for shared thread sessions.
        #
        # When multiple users share a single thread session (the default for
        # threads), prefix each message with [sender name] so the agent can
        # tell participants apart.  Skip for DMs (single-user by nature) and
        # when per-user thread isolation is explicitly enabled.
        # -----------------------------------------------------------------
        _is_shared_thread = _is_shared_thread_session(
            source=source,
            thread_sessions_per_user=bool(
                getattr(self.config, "thread_sessions_per_user", False)
            ),
        )
        attachments = event.ensure_attachments()

        if _has_visible_image_attachments(attachments):
            image_paths = _image_vision_inputs_from_event(event)
            if image_paths:
                message_text = await self._enrich_message_with_vision(
                    raw_message_text,
                    image_paths,
                    source=source,
                )
            elif not raw_message_text.strip():
                message_text = self._auto_vision_degraded_note("", pending=False)
        
        # -----------------------------------------------------------------
        # Auto-transcribe voice/audio messages sent by the user
        # -----------------------------------------------------------------
        if attachments:
            audio_paths = _collect_audio_paths(
                attachments,
                message_type=event.message_type,
            )
            if audio_paths:
                message_text = await self._enrich_message_with_transcription(
                    message_text, audio_paths
                )
                # If STT failed, send a direct message to the user so they
                # know voice isn't configured — don't rely on the agent to
                # relay the error clearly.
                _stt_fail_markers = (
                    "No STT provider",
                    "STT is disabled",
                    "can't listen",
                    "VOICE_TOOLS_OPENAI_KEY",
                )
                if any(m in message_text for m in _stt_fail_markers):
                    _stt_adapter = self.adapters.get(source.platform)
                    _stt_meta = {"thread_id": source.thread_id} if source.thread_id else None
                    if _stt_adapter:
                        try:
                            _stt_msg = (
                                "🎤 I received your voice message but can't transcribe it — "
                                "no speech-to-text provider is configured.\n\n"
                                "To enable voice: install faster-whisper "
                                "(`pip install faster-whisper` in the Hermes venv) "
                                "and set `stt.enabled: true` in config.yaml, "
                                "then /restart the gateway."
                            )
                            # Point to setup skill if it's installed
                            if self._has_setup_skill():
                                _stt_msg += "\n\nFor full setup instructions, type: `/skill hermes-agent-setup`"
                            await _stt_adapter.send(
                                source.chat_id, _stt_msg,
                                metadata=_stt_meta,
                            )
                        except Exception:
                            pass

        # -----------------------------------------------------------------
        # Enrich document messages with context notes for the agent
        # -----------------------------------------------------------------
        message_text = _prepend_document_context_notes(
            message_text,
            attachments=attachments,
            message_type=event.message_type,
        )

        # -----------------------------------------------------------------
        # Inject reply context when user replies to a message not in history.
        # Telegram (and other platforms) let users reply to specific messages,
        # but if the quoted message is from a previous session, cron delivery,
        # or background task, the agent has no context about what's being
        # referenced. Prepend the quoted text so the agent understands. (#1594)
        # -----------------------------------------------------------------
        message_text = _prepend_reply_context_if_missing(
            message_text=message_text,
            event=event,
            history=history,
        )

        message_text = shared_prepend_shared_thread_sender(
            message_text=message_text,
            user_name=source.user_name,
            shared_thread=_is_shared_thread,
        )

        try:
            hook_ctx = _build_agent_start_hook_context(
                source=source,
                session_id=session_entry.session_id,
                message_text=message_text,
            )
            _adapter = self.adapters.get(source.platform)
            prelude = await shared_run_gateway_agent_prelude(
                hooks=self.hooks,
                hook_ctx=hook_ctx,
                message_text=message_text,
                should_expand_context_references="@" in message_text,
                expand_context_references=lambda: _expand_gateway_context_references(
                    message_text,
                    runner=self,
                    logger=logger,
                ),
                send_blocked_warning=(
                    (lambda warning: _adapter.send(source.chat_id, warning))
                    if _adapter
                    else None
                ),
            )
            if prelude.blocked:
                return
            message_text = prelude.message_text

            # Run the agent
            agent_result = await self._run_agent(
                message=message_text,
                context_prompt=context_prompt,
                history=history_for_agent,
                source=source,
                session_id=session_entry.session_id,
                session_key=session_key,
                event_message_id=event.message_id,
                event=event,
                admin_user_ids=context.admin_user_ids,
                is_admin_user=context.is_admin_user,
                raw_message=event.raw_message,
            )

            await shared_stop_gateway_typing_indicator(
                adapters=self.adapters,
                platform=source.platform,
                chat_id=source.chat_id,
            )

            _process_registry = None
            try:
                from tools.process_registry import process_registry as _process_registry
            except Exception as e:
                logger.error("Process watcher setup error: %s", e)

            prepared_completion = await shared_prepare_gateway_agent_completion(
                agent_result=agent_result,
                history_len=len(history),
                empty_response_fallback=lambda empty_kind: _empty_response_fallback(
                    source,
                    message_text,
                    empty_kind=empty_kind,
                    is_admin_user=bool(context.is_admin_user),
                    raw_message=event.raw_message,
                    event=event,
                ),
                session_entry=session_entry,
                show_reasoning=bool(getattr(self, "_show_reasoning", False)),
                hook_ctx=hook_ctx,
                hooks=self.hooks,
                logger=logger,
                platform_name=_platform_name,
                chat_id=source.chat_id,
                msg_start_time=_msg_start_time,
                process_registry=_process_registry,
                run_process_watcher=self._run_process_watcher,
                create_task=asyncio.create_task,
            )
            response = prepared_completion.response
            suppress_reply = prepared_completion.suppress_reply
            agent_messages = prepared_completion.agent_messages

            # NOTE: Dangerous command approvals are now handled inline by the
            # blocking gateway approval mechanism in tools/approval.py.  The agent
            # thread blocks until the user responds with /approve or /deny, so by
            # the time we reach here the approval has already been resolved.  The
            # old post-loop pop_pending + approval_hint code was removed in favour
            # of the blocking approach that mirrors CLI's synchronous input().
            
            # Save the full conversation to the transcript, including tool calls.
            # This preserves the complete agent loop (tool_calls, tool results,
            # intermediate reasoning) so sessions can be resumed with full context
            # and transcripts are useful for debugging and training data.
            #
            # IMPORTANT: When the agent failed before producing any response
            # (e.g. context-overflow 400), do NOT persist the user's message.
            # Persisting it would make the session even larger, causing the
            # same failure on the next attempt — an infinite loop. (#1630)
            shared_persist_gateway_agent_transcript(
                session_store=self.session_store,
                session_id=session_entry.session_id,
                session_key=session_entry.session_key,
                platform=source.platform.value if source.platform else "",
                history=history,
                agent_result=agent_result,
                agent_messages=agent_messages,
                message_text=message_text,
                visible_final_response=response,
                resolve_gateway_model=_resolve_gateway_model,
                sync_visible_final_response=_sync_visible_final_response_into_messages,
                session_db_present=self._session_db is not None,
                logger=logger,
            )

            return await shared_finalize_gateway_agent_delivery(
                agent_result=agent_result,
                suppress_reply=suppress_reply,
                response=response,
                agent_messages=agent_messages,
                event=event,
                platform=source.platform,
                adapters=self.adapters,
                should_send_voice_reply=self._should_send_voice_reply,
                send_voice_reply=self._send_voice_reply,
                deliver_media_from_response=self._deliver_media_from_response,
            )
            
        except Exception as e:
            await shared_stop_gateway_typing_indicator(
                adapters=self.adapters,
                platform=source.platform,
                chat_id=source.chat_id,
            )
            logger.exception("Agent error in session %s", session_key)
            _hist_len = len(history) if 'history' in locals() else 0
            return shared_build_gateway_exception_response(
                error=e,
                history_len=_hist_len,
            )
        finally:
            # Clear session env
            self._clear_session_env()
    
    def _format_session_info(self) -> str:
        """Resolve current model config and return a formatted info block.

        Surfaces model, provider, context length, and endpoint so gateway
        users can immediately see if context detection went wrong (e.g.
        local models falling to the 128K default).
        """
        from agent.model_metadata import get_model_context_length, DEFAULT_FALLBACK_CONTEXT

        model = _resolve_gateway_model()
        config_context_length = None
        provider = None
        base_url = None
        api_key = None

        try:
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                import yaml as _info_yaml
                with open(cfg_path, encoding="utf-8") as f:
                    data = _info_yaml.safe_load(f) or {}
                model_cfg = data.get("model", {})
                if isinstance(model_cfg, dict):
                    raw_ctx = model_cfg.get("context_length")
                    if raw_ctx is not None:
                        try:
                            config_context_length = int(raw_ctx)
                        except (TypeError, ValueError):
                            pass
                    provider = model_cfg.get("provider") or None
                    base_url = model_cfg.get("base_url") or None
        except Exception:
            pass

        # Resolve runtime credentials for probing
        try:
            runtime = _resolve_runtime_agent_kwargs()
            provider = provider or runtime.get("provider")
            base_url = base_url or runtime.get("base_url")
            api_key = runtime.get("api_key")
        except Exception:
            pass

        context_length = get_model_context_length(
            model,
            base_url=base_url or "",
            api_key=api_key or "",
            config_context_length=config_context_length,
            provider=provider or "",
        )

        # Format context source hint
        if config_context_length is not None:
            ctx_source = "config"
        elif context_length == DEFAULT_FALLBACK_CONTEXT:
            ctx_source = "default — set model.context_length in config to override"
        else:
            ctx_source = "detected"

        # Format context length for display
        if context_length >= 1_000_000:
            ctx_display = f"{context_length / 1_000_000:.1f}M"
        elif context_length >= 1_000:
            ctx_display = f"{context_length // 1_000}K"
        else:
            ctx_display = str(context_length)

        lines = [
            f"◆ Model: `{model}`",
            f"◆ Provider: {provider or 'openrouter'}",
            f"◆ Context: {ctx_display} tokens ({ctx_source})",
        ]

        # Show endpoint for local/custom setups
        if base_url and ("localhost" in base_url or "127.0.0.1" in base_url or "0.0.0.0" in base_url):
            lines.append(f"◆ Endpoint: {base_url}")

        return "\n".join(lines)

    @staticmethod
    def _is_runtime_identity_query(text: str) -> bool:
        """Detect short natural-language questions about the current model."""
        normalized = re.sub(r"\s+", "", (text or "").strip().lower())
        if not normalized:
            return False

        patterns = (
            r"^你(现[在]?|当前)?(是什么|是啥|啥)?模型.*$",
            r"^你(现[在]?|当前)?用的什么模型.*$",
            r"^你.*什么模型.*$",
            r"^你.*模型.*现在.*$",
            r"^你还是gpt[- ]?5(\.4)?[吗?？!！。,.看]*.*$",
            r"^whatmodelareyou.*$",
            r"^whichmodelareyou.*$",
            r"^whatproviderareyouusing.*$",
        )
        return any(re.match(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _endpoint_host(base_url: str) -> str:
        cleaned = str(base_url or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"^https?://", "", cleaned, flags=re.IGNORECASE)
        return cleaned.rstrip("/")

    def _format_runtime_identity_response(self, source: SessionSource) -> str:
        """Return a deterministic response for current model/provider questions."""
        model = _resolve_gateway_model()
        provider = None
        base_url = None
        scope = "当前主模型配置"

        try:
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                import yaml as _identity_yaml
                with open(cfg_path, encoding="utf-8") as f:
                    data = _identity_yaml.safe_load(f) or {}
                model_cfg = data.get("model", {})
                if isinstance(model_cfg, dict):
                    provider = model_cfg.get("provider") or None
                    base_url = model_cfg.get("base_url") or None
        except Exception:
            pass

        try:
            runtime = _resolve_runtime_agent_kwargs()
            provider = provider or runtime.get("provider")
            base_url = base_url or runtime.get("base_url")
        except Exception:
            pass

        session_key = self._session_key_for_source(source)
        override = getattr(self, "_session_model_overrides", {}).get(session_key, {})
        if override:
            scope = "当前这个会话"
            model = override.get("model") or model
            provider = override.get("provider") or provider
            base_url = override.get("base_url") or base_url

        provider = provider or "openrouter"
        lines = [
            f"{scope}是 `{model or 'unknown'}`。",
            f"Provider 是 `{provider}`。",
        ]
        if base_url and provider == "custom":
            lines.append(f"端点是 `{self._endpoint_host(base_url)}`。")
        return "\n".join(lines)

    async def _handle_reset_command(self, event: MessageEvent) -> str:
        """Handle /new or /reset command."""
        source = event.source
        
        # Get existing session key
        session_key = self._session_key_for_source(source)
        
        # Flush memories in the background (fire-and-forget) so the user
        # gets the "Session reset!" response immediately.
        try:
            old_entry = self.session_store._entries.get(session_key)
            if old_entry:
                _flush_task = asyncio.create_task(
                    self._async_flush_memories(old_entry.session_id)
                )
                self._background_tasks.add(_flush_task)
                _flush_task.add_done_callback(self._background_tasks.discard)
        except Exception as e:
            logger.debug("Gateway memory flush on reset failed: %s", e)
        self._evict_cached_agent(session_key)
        
        try:
            from tools.env_passthrough import clear_env_passthrough
            clear_env_passthrough()
        except Exception:
            pass

        try:
            from tools.credential_files import clear_credential_files
            clear_credential_files()
        except Exception:
            pass

        # Reset the session
        new_entry = self.session_store.reset_session(session_key)

        # Clear any session-scoped model override so the next agent picks up
        # the configured default instead of the previously switched model.
        self._session_model_overrides.pop(session_key, None)

        # Fire plugin on_session_finalize hook (session boundary)
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _old_sid = old_entry.session_id if old_entry else None
            _invoke_hook("on_session_finalize", session_id=_old_sid,
                         platform=source.platform.value if source.platform else "")
        except Exception:
            pass

        # Emit session:end hook (session is ending)
        await self.hooks.emit("session:end", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })

        # Emit session:reset hook
        await self.hooks.emit("session:reset", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })

        # Resolve session config info to surface to the user
        try:
            session_info = self._format_session_info()
        except Exception:
            session_info = ""

        if new_entry:
            header = "✨ Session reset! Starting fresh."
        else:
            # No existing session, just create one
            new_entry = self.session_store.get_or_create_session(source, force_new=True)
            header = "✨ New session started!"

        # Fire plugin on_session_reset hook (new session guaranteed to exist)
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _new_sid = new_entry.session_id if new_entry else None
            _invoke_hook("on_session_reset", session_id=_new_sid,
                         platform=source.platform.value if source.platform else "")
        except Exception:
            pass

        if session_info:
            return f"{header}\n\n{session_info}"
        return header
    
    async def _handle_profile_command(self, event: MessageEvent) -> str:
        """Handle /profile — show active profile name and home directory."""
        from hermes_constants import get_hermes_home, display_hermes_home
        from pathlib import Path

        home = get_hermes_home()
        display = display_hermes_home()

        # Detect profile name from HERMES_HOME path
        # Profile paths look like: ~/.hermes/profiles/<name>
        profiles_parent = Path.home() / ".hermes" / "profiles"
        try:
            rel = home.relative_to(profiles_parent)
            profile_name = str(rel).split("/")[0]
        except ValueError:
            profile_name = None

        if profile_name:
            lines = [
                f"👤 **Profile:** `{profile_name}`",
                f"📂 **Home:** `{display}`",
            ]
        else:
            lines = [
                "👤 **Profile:** default",
                f"📂 **Home:** `{display}`",
            ]

        return "\n".join(lines)

    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command."""
        return shared_render_status_command(
            self,
            event,
            pending_sentinel=_AGENT_PENDING_SENTINEL,
        )
    
    async def _handle_stop_command(self, event: MessageEvent) -> str:
        """Handle /stop command - interrupt a running agent.

        When an agent is truly hung (blocked thread that never checks
        _interrupt_requested), the early intercept in _handle_message()
        handles /stop before this method is reached.  This handler fires
        only through normal command dispatch (no running agent) or as a
        fallback.  Force-clean the session lock in all cases for safety.
        """
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        agent = self._running_agents.get(session_key)
        if agent is _AGENT_PENDING_SENTINEL:
            # Force-clean the sentinel so the session is unlocked.
            if session_key in self._running_agents:
                del self._running_agents[session_key]
            logger.info("HARD STOP (pending) for session %s — sentinel cleared", session_key[:20])
            return "⚡ Force-stopped. The agent was still starting — session unlocked."
        if agent:
            agent.interrupt("Stop requested")
            # Force-clean the session lock so a truly hung agent doesn't
            # keep it locked forever.
            if session_key in self._running_agents:
                del self._running_agents[session_key]
            return "⚡ Force-stopped. The session is unlocked — you can send a new message."

        raw_job_id = event.get_command_args().strip()
        job = self._resolve_background_job_for_stop(source, raw_job_id)
        if isinstance(job, dict) and job.get("ambiguous"):
            active_ids = ", ".join(f"`{item['task_id']}`" for item in job.get("jobs", []))
            return f"Multiple background jobs are running here. Use `/stop <task_id>`: {active_ids}"
        if job:
            task_id = str(job.get("task_id") or "")
            try:
                self._stop_background_worker(job)
                job = self._get_background_job_store().mark_job_cancelled(
                    task_id,
                    reason="stop requested",
                ) or job
            except Exception as exc:
                logger.warning("Failed to stop external background job %s: %s", task_id, exc)
                job = self._get_background_job_store().mark_job_cancelling(task_id) or job
            return f"⏹️ Requested stop for background job `{task_id}`."

        return "No active task to stop."
    
    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        from hermes_cli.commands import gateway_help_lines
        lines = [
            "📖 **Hermes Commands**\n",
            *gateway_help_lines(),
        ]
        try:
            from agent.skill_commands import get_skill_commands
            skill_cmds = get_skill_commands()
            if skill_cmds:
                lines.append(f"\n⚡ **Skill Commands** ({len(skill_cmds)} active):")
                # Show first 10, then point to /commands for the rest
                sorted_cmds = sorted(skill_cmds)
                for cmd in sorted_cmds[:10]:
                    lines.append(f"`{cmd}` — {skill_cmds[cmd]['description']}")
                if len(sorted_cmds) > 10:
                    lines.append(f"\n... and {len(sorted_cmds) - 10} more. Use `/commands` for the full paginated list.")
        except Exception:
            pass
        return "\n".join(lines)

    async def _handle_commands_command(self, event: MessageEvent) -> str:
        """Handle /commands [page] - paginated list of all commands and skills."""
        from hermes_cli.commands import gateway_help_lines

        raw_args = event.get_command_args().strip()
        if raw_args:
            try:
                requested_page = int(raw_args)
            except ValueError:
                return "Usage: `/commands [page]`"
        else:
            requested_page = 1

        # Build combined entry list: built-in commands + skill commands
        entries = list(gateway_help_lines())
        try:
            from agent.skill_commands import get_skill_commands
            skill_cmds = get_skill_commands()
            if skill_cmds:
                entries.append("")
                entries.append("⚡ **Skill Commands**:")
                for cmd in sorted(skill_cmds):
                    desc = skill_cmds[cmd].get("description", "").strip() or "Skill command"
                    entries.append(f"`{cmd}` — {desc}")
        except Exception:
            pass

        if not entries:
            return "No commands available."

        from gateway.config import Platform
        page_size = 15 if event.source.platform == Platform.TELEGRAM else 20
        total_pages = max(1, (len(entries) + page_size - 1) // page_size)
        page = max(1, min(requested_page, total_pages))
        start = (page - 1) * page_size
        page_entries = entries[start:start + page_size]

        lines = [
            f"📚 **Commands** ({len(entries)} total, page {page}/{total_pages})",
            "",
            *page_entries,
        ]
        if total_pages > 1:
            nav_parts = []
            if page > 1:
                nav_parts.append(f"`/commands {page - 1}` ← prev")
            if page < total_pages:
                nav_parts.append(f"next → `/commands {page + 1}`")
            lines.extend(["", " | ".join(nav_parts)])
        if page != requested_page:
            lines.append(f"_(Requested page {requested_page} was out of range, showing page {page}.)_")
        return "\n".join(lines)
    
    async def _handle_model_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /model command — switch model for this session.

        Supports:
          /model                              — interactive picker (Telegram/Discord) or text list
          /model <name>                       — switch for this session only
          /model <name> --global              — switch and persist to config.yaml
          /model <name> --provider <provider> — switch provider + model
          /model --provider <provider>        — switch to provider, auto-detect model
        """
        import yaml
        from hermes_cli.model_switch import (
            switch_model as _switch_model, parse_model_flags,
            list_authenticated_providers,
        )
        from hermes_cli.providers import get_label

        raw_args = event.get_command_args().strip()

        # Parse --provider and --global flags
        model_input, explicit_provider, persist_global = parse_model_flags(raw_args)

        # Read current model/provider from config
        current_model = ""
        current_provider = "openrouter"
        current_base_url = ""
        current_api_key = ""
        user_provs = None
        config_path = _hermes_home / "config.yaml"
        try:
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    current_model = model_cfg.get("default", "")
                    current_provider = model_cfg.get("provider", current_provider)
                    current_base_url = model_cfg.get("base_url", "")
                user_provs = cfg.get("providers")
        except Exception:
            pass

        # Check for session override
        source = event.source
        session_key = self._session_key_for_source(source)
        override = getattr(self, "_session_model_overrides", {}).get(session_key, {})
        if override:
            current_model = override.get("model", current_model)
            current_provider = override.get("provider", current_provider)
            current_base_url = override.get("base_url", current_base_url)
            current_api_key = override.get("api_key", current_api_key)

        # No args: show interactive picker (Telegram/Discord) or text list
        if not model_input and not explicit_provider:
            # Try interactive picker if the platform supports it
            adapter = self.adapters.get(source.platform)
            has_picker = (
                adapter is not None
                and getattr(type(adapter), "send_model_picker", None) is not None
            )

            if has_picker:
                try:
                    providers = list_authenticated_providers(
                        current_provider=current_provider,
                        user_providers=user_provs,
                        max_models=50,
                    )
                except Exception:
                    providers = []

                if providers:
                    # Build a callback closure for when the user picks a model.
                    # Captures self + locals needed for the switch logic.
                    _self = self
                    _session_key = session_key
                    _cur_model = current_model
                    _cur_provider = current_provider
                    _cur_base_url = current_base_url
                    _cur_api_key = current_api_key

                    async def _on_model_selected(
                        _chat_id: str, model_id: str, provider_slug: str
                    ) -> str:
                        """Perform the model switch and return confirmation text."""
                        result = _switch_model(
                            raw_input=model_id,
                            current_provider=_cur_provider,
                            current_model=_cur_model,
                            current_base_url=_cur_base_url,
                            current_api_key=_cur_api_key,
                            is_global=False,
                            explicit_provider=provider_slug,
                        )
                        if not result.success:
                            return f"Error: {result.error_message}"

                        # Update cached agent in-place
                        cached_entry = None
                        _cache_lock = getattr(_self, "_agent_cache_lock", None)
                        _cache = getattr(_self, "_agent_cache", None)
                        if _cache_lock and _cache is not None:
                            with _cache_lock:
                                cached_entry = _cache.get(_session_key)
                        if cached_entry and cached_entry[0] is not None:
                            try:
                                cached_entry[0].switch_model(
                                    new_model=result.new_model,
                                    new_provider=result.target_provider,
                                    api_key=result.api_key,
                                    base_url=result.base_url,
                                    api_mode=result.api_mode,
                                )
                            except Exception as exc:
                                logger.warning("Picker model switch failed for cached agent: %s", exc)

                        # Store model note + session override
                        if not hasattr(_self, "_pending_model_notes"):
                            _self._pending_model_notes = {}
                        _self._pending_model_notes[_session_key] = (
                            f"[Note: model was just switched from {_cur_model} to {result.new_model} "
                            f"via {result.provider_label or result.target_provider}. "
                            f"Adjust your self-identification accordingly.]"
                        )
                        if not hasattr(_self, "_session_model_overrides"):
                            _self._session_model_overrides = {}
                        _self._session_model_overrides[_session_key] = {
                            "model": result.new_model,
                            "provider": result.target_provider,
                            "api_key": result.api_key,
                            "base_url": result.base_url,
                            "api_mode": result.api_mode,
                        }

                        # Build confirmation text
                        plabel = result.provider_label or result.target_provider
                        lines = [f"Model switched to `{result.new_model}`"]
                        lines.append(f"Provider: {plabel}")
                        mi = result.model_info
                        if mi:
                            if mi.context_window:
                                lines.append(f"Context: {mi.context_window:,} tokens")
                            if mi.max_output:
                                lines.append(f"Max output: {mi.max_output:,} tokens")
                            if mi.has_cost_data():
                                lines.append(f"Cost: {mi.format_cost()}")
                            lines.append(f"Capabilities: {mi.format_capabilities()}")
                        lines.append("_(session only — use `/model <name> --global` to persist)_")
                        return "\n".join(lines)

                    metadata = {"thread_id": source.thread_id} if source.thread_id else None
                    result = await adapter.send_model_picker(
                        chat_id=source.chat_id,
                        providers=providers,
                        current_model=current_model,
                        current_provider=current_provider,
                        session_key=session_key,
                        on_model_selected=_on_model_selected,
                        metadata=metadata,
                    )
                    if result.success:
                        return None  # Picker sent — adapter handles the response

            # Fallback: text list (for platforms without picker or if picker failed)
            provider_label = get_label(current_provider)
            lines = [f"Current: `{current_model or 'unknown'}` on {provider_label}", ""]

            try:
                providers = list_authenticated_providers(
                    current_provider=current_provider,
                    user_providers=user_provs,
                    max_models=5,
                )
                for p in providers:
                    tag = " (current)" if p["is_current"] else ""
                    lines.append(f"**{p['name']}** `--provider {p['slug']}`{tag}:")
                    if p["models"]:
                        model_strs = ", ".join(f"`{m}`" for m in p["models"])
                        extra = f" (+{p['total_models'] - len(p['models'])} more)" if p["total_models"] > len(p["models"]) else ""
                        lines.append(f"  {model_strs}{extra}")
                    elif p.get("api_url"):
                        lines.append(f"  `{p['api_url']}`")
                    lines.append("")
            except Exception:
                pass

            lines.append("`/model <name>` — switch model")
            lines.append("`/model <name> --provider <slug>` — switch provider")
            lines.append("`/model <name> --global` — persist")
            return "\n".join(lines)

        # Perform the switch
        result = _switch_model(
            raw_input=model_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            current_api_key=current_api_key,
            is_global=persist_global,
            explicit_provider=explicit_provider,
        )

        if not result.success:
            return f"Error: {result.error_message}"

        # If there's a cached agent, update it in-place
        cached_entry = None
        _cache_lock = getattr(self, "_agent_cache_lock", None)
        _cache = getattr(self, "_agent_cache", None)
        if _cache_lock and _cache is not None:
            with _cache_lock:
                cached_entry = _cache.get(session_key)

        if cached_entry and cached_entry[0] is not None:
            try:
                cached_entry[0].switch_model(
                    new_model=result.new_model,
                    new_provider=result.target_provider,
                    api_key=result.api_key,
                    base_url=result.base_url,
                    api_mode=result.api_mode,
                )
            except Exception as exc:
                logger.warning("In-place model switch failed for cached agent: %s", exc)

        # Store a note to prepend to the next user message so the model
        # knows about the switch (avoids system messages mid-history).
        if not hasattr(self, "_pending_model_notes"):
            self._pending_model_notes = {}
        self._pending_model_notes[session_key] = (
            f"[Note: model was just switched from {current_model} to {result.new_model} "
            f"via {result.provider_label or result.target_provider}. "
            f"Adjust your self-identification accordingly.]"
        )

        # Store session override so next agent creation uses the new model
        if not hasattr(self, "_session_model_overrides"):
            self._session_model_overrides = {}
        self._session_model_overrides[session_key] = {
            "model": result.new_model,
            "provider": result.target_provider,
            "api_key": result.api_key,
            "base_url": result.base_url,
            "api_mode": result.api_mode,
        }

        # Persist to config if --global
        if persist_global:
            try:
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                else:
                    cfg = {}
                model_cfg = cfg.setdefault("model", {})
                model_cfg["default"] = result.new_model
                model_cfg["provider"] = result.target_provider
                if result.base_url:
                    model_cfg["base_url"] = result.base_url
                from hermes_cli.config import save_config
                save_config(cfg)
            except Exception as e:
                logger.warning("Failed to persist model switch: %s", e)

        # Build confirmation message with full metadata
        provider_label = result.provider_label or result.target_provider
        lines = [f"Model switched to `{result.new_model}`"]
        lines.append(f"Provider: {provider_label}")

        # Rich metadata from models.dev
        mi = result.model_info
        if mi:
            if mi.context_window:
                lines.append(f"Context: {mi.context_window:,} tokens")
            if mi.max_output:
                lines.append(f"Max output: {mi.max_output:,} tokens")
            if mi.has_cost_data():
                lines.append(f"Cost: {mi.format_cost()}")
            lines.append(f"Capabilities: {mi.format_capabilities()}")
        else:
            try:
                from agent.model_metadata import get_model_context_length
                ctx = get_model_context_length(
                    result.new_model,
                    base_url=result.base_url or current_base_url,
                    api_key=result.api_key or current_api_key,
                    provider=result.target_provider,
                )
                lines.append(f"Context: {ctx:,} tokens")
            except Exception:
                pass

        # Cache notice
        cache_enabled = (
            ("openrouter" in (result.base_url or "").lower() and "claude" in result.new_model.lower())
            or result.api_mode == "anthropic_messages"
        )
        if cache_enabled:
            lines.append("Prompt caching: enabled")

        if result.warning_message:
            lines.append(f"Warning: {result.warning_message}")

        if persist_global:
            lines.append("Saved to config.yaml (`--global`)")
        else:
            lines.append("_(session only -- add `--global` to persist)_")

        return "\n".join(lines)

    async def _handle_provider_command(self, event: MessageEvent) -> str:
        """Handle /provider command - show available providers."""
        import yaml
        from hermes_cli.models import (
            list_available_providers,
            normalize_provider,
            _PROVIDER_LABELS,
        )

        # Resolve current provider from config
        current_provider = "openrouter"
        config_path = _hermes_home / 'config.yaml'
        try:
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    current_provider = model_cfg.get("provider", current_provider)
        except Exception:
            pass

        current_provider = normalize_provider(current_provider)
        if current_provider == "auto":
            try:
                from hermes_cli.auth import resolve_provider as _resolve_provider
                current_provider = _resolve_provider(current_provider)
            except Exception:
                current_provider = "openrouter"

        # Detect custom endpoint from config base_url
        if current_provider == "openrouter":
            _cfg_base = model_cfg.get("base_url", "") if isinstance(model_cfg, dict) else ""
            if _cfg_base and "openrouter.ai" not in _cfg_base:
                current_provider = "custom"

        current_label = _PROVIDER_LABELS.get(current_provider, current_provider)

        lines = [
            f"🔌 **Current provider:** {current_label} (`{current_provider}`)",
            "",
            "**Available providers:**",
        ]

        providers = list_available_providers()
        for p in providers:
            marker = " ← active" if p["id"] == current_provider else ""
            auth = "✅" if p["authenticated"] else "❌"
            aliases = f"  _(also: {', '.join(p['aliases'])})_" if p["aliases"] else ""
            lines.append(f"{auth} `{p['id']}` — {p['label']}{aliases}{marker}")

        lines.append("")
        lines.append("Switch: `/model provider:model-name`")
        lines.append("Setup: `hermes setup`")
        return "\n".join(lines)
    
    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality command - list or set a personality."""
        import yaml

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / 'config.yaml'

        try:
            if config_path.exists():
                with open(config_path, 'r', encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                personalities = config.get("agent", {}).get("personalities", {})
            else:
                config = {}
                personalities = {}
        except Exception:
            config = {}
            personalities = {}

        if not personalities:
            return "No personalities configured in `~/.hermes/config.yaml`"

        if not args:
            lines = ["🎭 **Available Personalities**\n"]
            lines.append("• `none` — (no personality overlay)")
            for name, prompt in personalities.items():
                if isinstance(prompt, dict):
                    preview = prompt.get("description") or prompt.get("system_prompt", "")[:50]
                else:
                    preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
                lines.append(f"• `{name}` — {preview}")
            lines.append("\nUsage: `/personality <name>`")
            return "\n".join(lines)

        def _resolve_prompt(value):
            if isinstance(value, dict):
                parts = [value.get("system_prompt", "")]
                if value.get("tone"):
                    parts.append(f'Tone: {value["tone"]}')
                if value.get("style"):
                    parts.append(f'Style: {value["style"]}')
                return "\n".join(p for p in parts if p)
            return str(value)

        if args in ("none", "default", "neutral"):
            try:
                if "agent" not in config or not isinstance(config.get("agent"), dict):
                    config["agent"] = {}
                config["agent"]["system_prompt"] = ""
                atomic_yaml_write(config_path, config)
            except Exception as e:
                return f"⚠️ Failed to save personality change: {e}"
            self._ephemeral_system_prompt = ""
            return "🎭 Personality cleared — using base agent behavior.\n_(takes effect on next message)_"
        elif args in personalities:
            new_prompt = _resolve_prompt(personalities[args])

            # Write to config.yaml, same pattern as CLI save_config_value.
            try:
                if "agent" not in config or not isinstance(config.get("agent"), dict):
                    config["agent"] = {}
                config["agent"]["system_prompt"] = new_prompt
                atomic_yaml_write(config_path, config)
            except Exception as e:
                return f"⚠️ Failed to save personality change: {e}"

            # Update in-memory so it takes effect on the very next message.
            self._ephemeral_system_prompt = new_prompt

            return f"🎭 Personality set to **{args}**\n_(takes effect on next message)_"

        available = "`none`, " + ", ".join(f"`{n}`" for n in personalities)
        return f"Unknown personality: `{args}`\n\nAvailable: {available}"
    
    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # Find the last user message
        last_user_msg = None
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_msg = history[i].get("content", "")
                last_user_idx = i
                break
        
        if not last_user_msg:
            return "No previous message to retry."
        
        # Truncate history to before the last user message and persist
        truncated = history[:last_user_idx]
        self.session_store.rewrite_transcript(session_entry.session_id, truncated)
        # Reset stored token count — transcript was truncated
        session_entry.last_prompt_tokens = 0
        
        # Re-send by creating a fake text event with the old message
        retry_event = MessageEvent(
            text=last_user_msg,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event.raw_message,
        )
        
        # Let the normal message handler process it
        return await self._handle_message(retry_event)
    
    async def _handle_undo_command(self, event: MessageEvent) -> str:
        """Handle /undo command - remove the last user/assistant exchange."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # Find the last user message and remove everything from it onward
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            return "Nothing to undo."
        
        removed_msg = history[last_user_idx].get("content", "")
        removed_count = len(history) - last_user_idx
        self.session_store.rewrite_transcript(session_entry.session_id, history[:last_user_idx])
        # Reset stored token count — transcript was truncated
        session_entry.last_prompt_tokens = 0
        
        preview = removed_msg[:40] + "..." if len(removed_msg) > 40 else removed_msg
        return f"↩️ Undid {removed_count} message(s).\nRemoved: \"{preview}\""
    
    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command -- set the current chat as the platform's home channel."""
        source = event.source
        platform_name = source.platform.value if source.platform else "unknown"
        chat_id = source.chat_id
        chat_name = source.chat_name or chat_id
        
        env_key = f"{platform_name.upper()}_HOME_CHANNEL"
        
        # Save to config.yaml
        try:
            import yaml
            config_path = _hermes_home / 'config.yaml'
            user_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
            user_config[env_key] = chat_id
            atomic_yaml_write(config_path, user_config)
            # Also set in the current environment so it takes effect immediately
            os.environ[env_key] = str(chat_id)
        except Exception as e:
            return f"Failed to save home channel: {e}"
        
        return (
            f"✅ Home channel set to **{chat_name}** (ID: {chat_id}).\n"
            f"Cron jobs and cross-platform messages will be delivered here."
        )
    
    @staticmethod
    def _get_guild_id(event: MessageEvent) -> Optional[int]:
        """Extract Discord guild_id from the raw message object."""
        raw = getattr(event, "raw_message", None)
        if raw is None:
            return None
        # Slash command interaction
        if hasattr(raw, "guild_id") and raw.guild_id:
            return int(raw.guild_id)
        # Regular message
        if hasattr(raw, "guild") and raw.guild:
            return raw.guild.id
        return None

    async def _handle_voice_command(self, event: MessageEvent) -> str:
        """Handle /voice [on|off|tts|channel|leave|status] command."""
        args = event.get_command_args().strip().lower()
        chat_id = event.source.chat_id

        adapter = self.adapters.get(event.source.platform)

        if args in ("on", "enable"):
            self._voice_mode[chat_id] = "voice_only"
            self._save_voice_modes()
            if adapter:
                self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=False)
            return (
                "Voice mode enabled.\n"
                "I'll reply with voice when you send voice messages.\n"
                "Use /voice tts to get voice replies for all messages."
            )
        elif args in ("off", "disable"):
            self._voice_mode[chat_id] = "off"
            self._save_voice_modes()
            if adapter:
                self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
            return "Voice mode disabled. Text-only replies."
        elif args == "tts":
            self._voice_mode[chat_id] = "all"
            self._save_voice_modes()
            if adapter:
                self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=False)
            return (
                "Auto-TTS enabled.\n"
                "All replies will include a voice message."
            )
        elif args in ("channel", "join"):
            return await self._handle_voice_channel_join(event)
        elif args == "leave":
            return await self._handle_voice_channel_leave(event)
        elif args == "status":
            mode = self._voice_mode.get(chat_id, "off")
            labels = {
                "off": "Off (text only)",
                "voice_only": "On (voice reply to voice messages)",
                "all": "TTS (voice reply to all messages)",
            }
            # Append voice channel info if connected
            adapter = self.adapters.get(event.source.platform)
            guild_id = self._get_guild_id(event)
            if guild_id and hasattr(adapter, "get_voice_channel_info"):
                info = adapter.get_voice_channel_info(guild_id)
                if info:
                    lines = [
                        f"Voice mode: {labels.get(mode, mode)}",
                        f"Voice channel: #{info['channel_name']}",
                        f"Participants: {info['member_count']}",
                    ]
                    for m in info["members"]:
                        status = " (speaking)" if m.get("is_speaking") else ""
                        lines.append(f"  - {m['display_name']}{status}")
                    return "\n".join(lines)
            return f"Voice mode: {labels.get(mode, mode)}"
        else:
            # Toggle: off → on, on/all → off
            current = self._voice_mode.get(chat_id, "off")
            if current == "off":
                self._voice_mode[chat_id] = "voice_only"
                self._save_voice_modes()
                if adapter:
                    self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=False)
                return "Voice mode enabled."
            else:
                self._voice_mode[chat_id] = "off"
                self._save_voice_modes()
                if adapter:
                    self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
                return "Voice mode disabled."

    async def _handle_voice_channel_join(self, event: MessageEvent) -> str:
        """Join the user's current Discord voice channel."""
        adapter = self.adapters.get(event.source.platform)
        if not hasattr(adapter, "join_voice_channel"):
            return "Voice channels are not supported on this platform."

        guild_id = self._get_guild_id(event)
        if not guild_id:
            return "This command only works in a Discord server."

        voice_channel = await adapter.get_user_voice_channel(
            guild_id, event.source.user_id
        )
        if not voice_channel:
            return "You need to be in a voice channel first."

        # Wire callbacks BEFORE join so voice input arriving immediately
        # after connection is not lost.
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = self._handle_voice_channel_input
        if hasattr(adapter, "_on_voice_disconnect"):
            adapter._on_voice_disconnect = self._handle_voice_timeout_cleanup

        try:
            success = await adapter.join_voice_channel(voice_channel)
        except Exception as e:
            logger.warning("Failed to join voice channel: %s", e)
            adapter._voice_input_callback = None
            err_lower = str(e).lower()
            if "pynacl" in err_lower or "nacl" in err_lower or "davey" in err_lower:
                return (
                    "Voice dependencies are missing (PyNaCl / davey). "
                    "Install or reinstall Hermes with the messaging extra, e.g. "
                    "`pip install hermes-agent[messaging]`."
                )
            return f"Failed to join voice channel: {e}"

        if success:
            adapter._voice_text_channels[guild_id] = int(event.source.chat_id)
            self._voice_mode[event.source.chat_id] = "all"
            self._save_voice_modes()
            self._set_adapter_auto_tts_disabled(adapter, event.source.chat_id, disabled=False)
            return (
                f"Joined voice channel **{voice_channel.name}**.\n"
                f"I'll speak my replies and listen to you. Use /voice leave to disconnect."
            )
        # Join failed — clear callback
        adapter._voice_input_callback = None
        return "Failed to join voice channel. Check bot permissions (Connect + Speak)."

    async def _handle_voice_channel_leave(self, event: MessageEvent) -> str:
        """Leave the Discord voice channel."""
        adapter = self.adapters.get(event.source.platform)
        guild_id = self._get_guild_id(event)

        if not guild_id or not hasattr(adapter, "leave_voice_channel"):
            return "Not in a voice channel."

        if not hasattr(adapter, "is_in_voice_channel") or not adapter.is_in_voice_channel(guild_id):
            return "Not in a voice channel."

        try:
            await adapter.leave_voice_channel(guild_id)
        except Exception as e:
            logger.warning("Error leaving voice channel: %s", e)
        # Always clean up state even if leave raised an exception
        self._voice_mode[event.source.chat_id] = "off"
        self._save_voice_modes()
        self._set_adapter_auto_tts_disabled(adapter, event.source.chat_id, disabled=True)
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = None
        return "Left voice channel."

    def _handle_voice_timeout_cleanup(self, chat_id: str) -> None:
        """Called by the adapter when a voice channel times out.

        Cleans up runner-side voice_mode state that the adapter cannot reach.
        """
        self._voice_mode[chat_id] = "off"
        self._save_voice_modes()
        adapter = self.adapters.get(Platform.DISCORD)
        self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)

    async def _handle_voice_channel_input(
        self, guild_id: int, user_id: int, transcript: str
    ):
        """Handle transcribed voice from a user in a voice channel.

        Creates a synthetic MessageEvent and processes it through the
        adapter's full message pipeline (session, typing, agent, TTS reply).
        """
        adapter = self.adapters.get(Platform.DISCORD)
        if not adapter:
            return

        text_ch_id = adapter._voice_text_channels.get(guild_id)
        if not text_ch_id:
            return

        # Check authorization before processing voice input
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id=str(text_ch_id),
            user_id=str(user_id),
            user_name=str(user_id),
            chat_type="channel",
        )
        if not self._is_user_authorized(source):
            logger.debug("Unauthorized voice input from user %d, ignoring", user_id)
            return

        # Show transcript in text channel (after auth, with mention sanitization)
        try:
            channel = adapter._client.get_channel(text_ch_id)
            if channel:
                safe_text = transcript[:2000].replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
                await channel.send(f"**[Voice]** <@{user_id}>: {safe_text}")
        except Exception:
            pass

        # Build a synthetic MessageEvent and feed through the normal pipeline
        # Use SimpleNamespace as raw_message so _get_guild_id() can extract
        # guild_id and _send_voice_reply() plays audio in the voice channel.
        from types import SimpleNamespace
        event = MessageEvent(
            source=source,
            text=transcript,
            message_type=MessageType.VOICE,
            raw_message=SimpleNamespace(guild_id=guild_id, guild=None),
        )

        await adapter.handle_message(event)

    def _should_send_voice_reply(
        self,
        event: MessageEvent,
        response: str,
        agent_messages: list,
        already_sent: bool = False,
    ) -> bool:
        """Decide whether the runner should send a TTS voice reply.

        Returns False when:
        - voice_mode is off for this chat
        - response is empty or an error
        - agent already called text_to_speech tool (dedup)
        - voice input and base adapter auto-TTS already handled it (skip_double)
          UNLESS streaming already consumed the response (already_sent=True),
          in which case the base adapter won't have text for auto-TTS so the
          runner must handle it.
        """
        if not response or response.startswith("Error:"):
            return False

        chat_id = event.source.chat_id
        voice_mode = self._voice_mode.get(chat_id, "off")
        is_voice_input = (event.message_type == MessageType.VOICE)

        should = (
            (voice_mode == "all")
            or (voice_mode == "voice_only" and is_voice_input)
        )
        if not should:
            return False

        # Dedup: agent already called TTS tool
        has_agent_tts = any(
            msg.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name") == "text_to_speech"
                for tc in (msg.get("tool_calls") or [])
            )
            for msg in agent_messages
        )
        if has_agent_tts:
            return False

        # Dedup: base adapter auto-TTS already handles voice input
        # (play_tts plays in VC when connected, so runner can skip).
        # When streaming already delivered the text (already_sent=True),
        # the base adapter will receive None and can't run auto-TTS,
        # so the runner must take over.
        if is_voice_input and not already_sent:
            return False

        return True

    async def _send_voice_reply(self, event: MessageEvent, text: str) -> None:
        """Generate TTS audio and send as a voice message before the text reply."""
        import uuid as _uuid
        audio_path = None
        actual_path = None
        try:
            from tools.tts_tool import text_to_speech_tool, _strip_markdown_for_tts

            tts_text = _strip_markdown_for_tts(text[:4000])
            if not tts_text:
                return

            # Use .mp3 extension so edge-tts conversion to opus works correctly.
            # The TTS tool may convert to .ogg — use file_path from result.
            audio_path = os.path.join(
                tempfile.gettempdir(), "hermes_voice",
                f"tts_reply_{_uuid.uuid4().hex[:12]}.mp3",
            )
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            result_json = await asyncio.to_thread(
                text_to_speech_tool, text=tts_text, output_path=audio_path
            )
            result = json.loads(result_json)

            # Use the actual file path from result (may differ after opus conversion)
            actual_path = result.get("file_path", audio_path)
            if not result.get("success") or not os.path.isfile(actual_path):
                logger.warning("Auto voice reply TTS failed: %s", result.get("error"))
                return

            adapter = self.adapters.get(event.source.platform)

            # If connected to a voice channel, play there instead of sending a file
            guild_id = self._get_guild_id(event)
            if (guild_id
                    and hasattr(adapter, "play_in_voice_channel")
                    and hasattr(adapter, "is_in_voice_channel")
                    and adapter.is_in_voice_channel(guild_id)):
                await adapter.play_in_voice_channel(guild_id, actual_path)
            elif adapter and hasattr(adapter, "send_voice"):
                send_kwargs: Dict[str, Any] = {
                    "chat_id": event.source.chat_id,
                    "audio_path": actual_path,
                    "reply_to": event.message_id,
                }
                if event.source.thread_id:
                    send_kwargs["metadata"] = {"thread_id": event.source.thread_id}
                await adapter.send_voice(**send_kwargs)
        except Exception as e:
            logger.warning("Auto voice reply failed: %s", e, exc_info=True)
        finally:
            for p in {audio_path, actual_path} - {None}:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    async def _deliver_media_from_response(
        self,
        response: str,
        event: MessageEvent,
        adapter,
    ) -> None:
        """Extract MEDIA: tags and local file paths from a response and deliver them.

        Called after streaming has already sent the text to the user, so the
        text itself is already delivered — this only handles file attachments
        that the normal _process_message_background path would have caught.
        """
        from pathlib import Path

        try:
            media_files, _ = adapter.extract_media(response)
            _, cleaned = adapter.extract_images(response)
            local_files, _ = adapter.extract_local_files(cleaned)

            _thread_meta = {"thread_id": event.source.thread_id} if event.source.thread_id else None

            _AUDIO_EXTS = {'.ogg', '.opus', '.mp3', '.wav', '.m4a'}
            _VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
            _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

            for media_path, is_voice in media_files:
                try:
                    ext = Path(media_path).suffix.lower()
                    if ext in _AUDIO_EXTS:
                        await adapter.send_voice(
                            chat_id=event.source.chat_id,
                            audio_path=media_path,
                            metadata=_thread_meta,
                        )
                    elif ext in _VIDEO_EXTS:
                        await adapter.send_video(
                            chat_id=event.source.chat_id,
                            video_path=media_path,
                            metadata=_thread_meta,
                        )
                    elif ext in _IMAGE_EXTS:
                        await adapter.send_image_file(
                            chat_id=event.source.chat_id,
                            image_path=media_path,
                            metadata=_thread_meta,
                        )
                    else:
                        await adapter.send_document(
                            chat_id=event.source.chat_id,
                            file_path=media_path,
                            metadata=_thread_meta,
                        )
                except Exception as e:
                    logger.warning("[%s] Post-stream media delivery failed: %s", adapter.name, e)

            for file_path in local_files:
                try:
                    ext = Path(file_path).suffix.lower()
                    if ext in _IMAGE_EXTS:
                        await adapter.send_image_file(
                            chat_id=event.source.chat_id,
                            image_path=file_path,
                            metadata=_thread_meta,
                        )
                    else:
                        await adapter.send_document(
                            chat_id=event.source.chat_id,
                            file_path=file_path,
                            metadata=_thread_meta,
                        )
                except Exception as e:
                    logger.warning("[%s] Post-stream file delivery failed: %s", adapter.name, e)

        except Exception as e:
            logger.warning("Post-stream media extraction failed: %s", e)

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — list or restore filesystem checkpoints."""
        from tools.checkpoint_manager import CheckpointManager, format_checkpoint_list

        # Read checkpoint config from config.yaml
        cp_cfg = {}
        try:
            import yaml as _y
            _cfg_path = _hermes_home / "config.yaml"
            if _cfg_path.exists():
                with open(_cfg_path, encoding="utf-8") as _f:
                    _data = _y.safe_load(_f) or {}
                cp_cfg = _data.get("checkpoints", {})
                if isinstance(cp_cfg, bool):
                    cp_cfg = {"enabled": cp_cfg}
        except Exception:
            pass

        if not cp_cfg.get("enabled", False):
            return (
                "Checkpoints are not enabled.\n"
                "Enable in config.yaml:\n```\ncheckpoints:\n  enabled: true\n```"
            )

        mgr = CheckpointManager(
            enabled=True,
            max_snapshots=cp_cfg.get("max_snapshots", 50),
        )

        cwd = os.getenv("MESSAGING_CWD", str(Path.home()))
        arg = event.get_command_args().strip()

        if not arg:
            checkpoints = mgr.list_checkpoints(cwd)
            return format_checkpoint_list(checkpoints, cwd)

        # Restore by number or hash
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            return f"No checkpoints found for {cwd}"

        target_hash = None
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(checkpoints):
                target_hash = checkpoints[idx]["hash"]
            else:
                return f"Invalid checkpoint number. Use 1-{len(checkpoints)}."
        except ValueError:
            target_hash = arg

        result = mgr.restore(cwd, target_hash)
        if result["success"]:
            return (
                f"✅ Restored to checkpoint {result['restored_to']}: {result['reason']}\n"
                f"A pre-rollback snapshot was saved automatically."
            )
        return f"❌ {result['error']}"

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /background <prompt> — run a prompt in a separate background session.

        Spawns a new AIAgent in a background thread with its own session.
        When it completes, sends the result back to the same chat without
        modifying the active session's conversation history.
        """
        prompt = event.get_command_args().strip()
        if not prompt:
            return (
                "Usage: /background <prompt>\n"
                "Example: /background Summarize the top HN stories today\n\n"
                "Runs the prompt in a separate session. "
                "You can keep chatting — the result will appear here when done."
            )

        source = event.source
        admin_user_ids = self._configured_admin_user_ids(source.platform)
        is_admin_user = self._is_admin_user(source) if admin_user_ids else None
        dispatch = shared_resolve_employee_background_dispatch(
            prompt,
            employee_routes=get_employee_routes(
                self.config,
                platform=getattr(source, "platform", Platform.QQ_NAPCAT),
            ),
        )
        task_id = self._start_background_job(
            prompt,
            source,
            session_key=self._session_key_for_source(source),
            job_kind="manual",
            worker_name=str((dispatch or {}).get("worker_name") or ""),
            preloaded_skills=list((dispatch or {}).get("preloaded_skills") or []),
            admin_user_ids=admin_user_ids,
            is_admin_user=is_admin_user,
        )

        preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
        worker_name = str((dispatch or {}).get("worker_name") or "")
        if worker_name:
            return (
                f"🔄 已交给{worker_name}后台处理：\"{preview}\"\n"
                f"任务ID：{task_id}\n"
                "你可以继续聊天，结果回来我会发到这里。"
            )
        return f'🔄 Background task started: "{preview}"\nTask ID: {task_id}\nYou can keep chatting — results will appear when done.'

    async def _deliver_background_job_updates_once(self) -> None:
        """Deliver one polling pass of durable background job completions and approvals."""
        await shared_deliver_background_job_updates_once(self)

    async def _background_job_delivery_poller(self, interval: float = 2.0) -> None:
        """Background poller for durable job completions and approval prompts."""
        await shared_background_job_delivery_poller(self, interval=interval)

    async def _recover_stale_background_jobs_once(
        self,
        *,
        queued_grace_seconds: float = 120.0,
        heartbeat_stale_seconds: float = 120.0,
        now_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        return await shared_recover_stale_background_jobs_once(
            self,
            now_ts=now_ts,
            queued_grace_seconds=queued_grace_seconds,
            heartbeat_stale_seconds=heartbeat_stale_seconds,
        )

    async def _handle_btw_command(self, event: MessageEvent) -> str:
        """Handle /btw <question> — ephemeral side question in the same chat."""
        question = event.get_command_args().strip()
        if not question:
            return (
                "Usage: /btw <question>\n"
                "Example: /btw what module owns session title sanitization?\n\n"
                "Answers using session context in a detached background turn."
            )

        source = event.source
        session_key = self._session_key_for_source(source)

        existing = [
            job
            for job in self._background_jobs_for_source(source, active_only=True)
            if str(job.get("kind") or "").strip().lower() == "btw"
        ]
        if existing:
            return "A /btw is already running for this chat. Wait for it to finish."

        running_agent = self._running_agents.get(session_key)
        if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            history_snapshot = list(getattr(running_agent, "_session_messages", []) or [])
        else:
            session_entry = self.session_store.get_or_create_session(source)
            history_snapshot = self.session_store.load_transcript(session_entry.session_id)

        task_id = self._start_background_job(
            question,
            source,
            conversation_history=list(history_snapshot or []),
            session_key=session_key,
            job_kind="btw",
        )

        preview = question[:60] + ("..." if len(question) > 60 else "")
        return f'💬 /btw: "{preview}"\nReply will appear here shortly.'

    async def _handle_reasoning_command(self, event: MessageEvent) -> str:
        """Handle /reasoning command — manage reasoning effort and display toggle.

        Usage:
            /reasoning              Show current effort level and display state
            /reasoning <level>      Set reasoning effort (none, low, medium, high, xhigh)
            /reasoning show|on      Show model reasoning in responses
            /reasoning hide|off     Hide model reasoning from responses
        """
        import yaml

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / "config.yaml"
        self._reasoning_config = self._load_reasoning_config()
        self._show_reasoning = self._load_show_reasoning()

        def _save_config_key(key_path: str, value):
            """Save a dot-separated key to config.yaml."""
            try:
                user_config = {}
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                keys = key_path.split(".")
                current = user_config
                for k in keys[:-1]:
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                atomic_yaml_write(config_path, user_config)
                return True
            except Exception as e:
                logger.error("Failed to save config key %s: %s", key_path, e)
                return False

        if not args:
            # Show current state
            rc = self._reasoning_config
            if rc is None:
                level = "medium (default)"
            elif rc.get("enabled") is False:
                level = "none (disabled)"
            else:
                level = rc.get("effort", "medium")
            display_state = "on ✓" if self._show_reasoning else "off"
            return (
                "🧠 **Reasoning Settings**\n\n"
                f"**Effort:** `{level}`\n"
                f"**Display:** {display_state}\n\n"
                "_Usage:_ `/reasoning <none|low|medium|high|xhigh|show|hide>`"
            )

        # Display toggle
        if args in ("show", "on"):
            self._show_reasoning = True
            _save_config_key("display.show_reasoning", True)
            return "🧠 ✓ Reasoning display: **ON**\nModel thinking will be shown before each response."

        if args in ("hide", "off"):
            self._show_reasoning = False
            _save_config_key("display.show_reasoning", False)
            return "🧠 ✓ Reasoning display: **OFF**"

        # Effort level change
        effort = args.strip()
        if effort == "none":
            parsed = {"enabled": False}
        elif effort in ("xhigh", "high", "medium", "low", "minimal"):
            parsed = {"enabled": True, "effort": effort}
        else:
            return (
                f"⚠️ Unknown argument: `{effort}`\n\n"
                "**Valid levels:** none, low, minimal, medium, high, xhigh\n"
                "**Display:** show, hide"
            )

        self._reasoning_config = parsed
        if _save_config_key("agent.reasoning_effort", effort):
            return f"🧠 ✓ Reasoning effort set to `{effort}` (saved to config)\n_(takes effect on next message)_"
        else:
            return f"🧠 ✓ Reasoning effort set to `{effort}` (this session only)"

    async def _handle_yolo_command(self, event: MessageEvent) -> str:
        """Handle /yolo — toggle dangerous command approval bypass."""
        admin_only_message = self._admin_only_message(
            event.source,
            "toggle YOLO mode",
        )
        if admin_only_message:
            return admin_only_message

        current = bool(os.environ.get("HERMES_YOLO_MODE"))
        if current:
            os.environ.pop("HERMES_YOLO_MODE", None)
            return "⚠️ YOLO mode **OFF** — dangerous commands will require approval."
        else:
            os.environ["HERMES_YOLO_MODE"] = "1"
            return "⚡ YOLO mode **ON** — all commands auto-approved. Use with caution."

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        """Handle /verbose command — cycle tool progress display mode.

        Gated by ``display.tool_progress_command`` in config.yaml (default off).
        When enabled, cycles the tool progress mode through off → new → all →
        verbose → off, same as the CLI.
        """
        import yaml

        config_path = _hermes_home / "config.yaml"

        # --- check config gate ------------------------------------------------
        try:
            user_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
            gate_enabled = user_config.get("display", {}).get("tool_progress_command", False)
        except Exception:
            gate_enabled = False

        if not gate_enabled:
            return (
                "The `/verbose` command is not enabled for messaging platforms.\n\n"
                "Enable it in `config.yaml`:\n```yaml\n"
                "display:\n  tool_progress_command: true\n```"
            )

        # --- cycle mode -------------------------------------------------------
        cycle = ["off", "new", "all", "verbose"]
        descriptions = {
            "off": "⚙️ Tool progress: **OFF** — no tool activity shown.",
            "new": "⚙️ Tool progress: **NEW** — shown when tool changes (preview length: `display.tool_preview_length`, default 40).",
            "all": "⚙️ Tool progress: **ALL** — every tool call shown (preview length: `display.tool_preview_length`, default 40).",
            "verbose": "⚙️ Tool progress: **VERBOSE** — every tool call with full arguments.",
        }

        raw_progress = user_config.get("display", {}).get("tool_progress", "all")
        # YAML 1.1 parses bare "off" as boolean False — normalise back
        if raw_progress is False:
            current = "off"
        elif raw_progress is True:
            current = "all"
        else:
            current = str(raw_progress).lower()
        if current not in cycle:
            current = "all"
        idx = (cycle.index(current) + 1) % len(cycle)
        new_mode = cycle[idx]

        # Save to config.yaml
        try:
            if "display" not in user_config or not isinstance(user_config.get("display"), dict):
                user_config["display"] = {}
            user_config["display"]["tool_progress"] = new_mode
            atomic_yaml_write(config_path, user_config)
            return f"{descriptions[new_mode]}\n_(saved to config — takes effect on next message)_"
        except Exception as e:
            logger.warning("Failed to save tool_progress mode: %s", e)
            return f"{descriptions[new_mode]}\n_(could not save to config: {e})_"

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Handle /compress command -- manually compress conversation context."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)

        if not history or len(history) < 4:
            return "Not enough conversation to compress (need at least 4 messages)."

        try:
            from run_agent import AIAgent
            from agent.model_metadata import estimate_messages_tokens_rough

            runtime_kwargs = _resolve_runtime_agent_kwargs()
            if not runtime_kwargs.get("api_key"):
                return "No provider configured -- cannot compress."

            # Resolve model from config (same reason as memory flush above).
            model = _resolve_gateway_model()

            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            original_count = len(msgs)
            approx_tokens = estimate_messages_tokens_rough(msgs)

            tmp_agent = AIAgent(
                **runtime_kwargs,
                model=model,
                max_iterations=4,
                quiet_mode=True,
                enabled_toolsets=["memory"],
                session_id=session_entry.session_id,
            )
            tmp_agent._print_fn = lambda *a, **kw: None

            loop = asyncio.get_event_loop()
            compressed, _ = await loop.run_in_executor(
                None,
                lambda: tmp_agent._compress_context(msgs, "", approx_tokens=approx_tokens)
            )

            # _compress_context already calls end_session() on the old session
            # (preserving its full transcript in SQLite) and creates a new
            # session_id for the continuation.  Write the compressed messages
            # into the NEW session so the original history stays searchable.
            new_session_id = tmp_agent.session_id
            if new_session_id != session_entry.session_id:
                session_entry.session_id = new_session_id
                self.session_store._save()

            self.session_store.rewrite_transcript(new_session_id, compressed)
            # Reset stored token count — transcript changed, old value is stale
            self.session_store.update_session(
                session_entry.session_key, last_prompt_tokens=0
            )
            new_count = len(compressed)
            new_tokens = estimate_messages_tokens_rough(compressed)

            return (
                f"🗜️ Compressed: {original_count} → {new_count} messages\n"
                f"~{approx_tokens:,} → ~{new_tokens:,} tokens"
            )
        except Exception as e:
            logger.warning("Manual compress failed: %s", e)
            return f"Compression failed: {e}"

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_id = session_entry.session_id

        if not self._session_db:
            return "Session database not available."

        # Ensure session exists in SQLite DB (it may only exist in session_store
        # if this is the first command in a new session)
        existing_title = self._session_db.get_session_title(session_id)
        if existing_title is None:
            # Session doesn't exist in DB yet — create it
            try:
                self._session_db.create_session(
                    session_id=session_id,
                    source=source.platform.value if source.platform else "unknown",
                    user_id=source.user_id,
                )
            except Exception:
                pass  # Session might already exist, ignore errors

        title_arg = event.get_command_args().strip()
        if title_arg:
            # Sanitize the title before setting
            try:
                sanitized = self._session_db.sanitize_title(title_arg)
            except ValueError as e:
                return f"⚠️ {e}"
            if not sanitized:
                return "⚠️ Title is empty after cleanup. Please use printable characters."
            # Set the title
            try:
                if self._session_db.set_session_title(session_id, sanitized):
                    return f"✏️ Session title set: **{sanitized}**"
                else:
                    return "Session not found in database."
            except ValueError as e:
                return f"⚠️ {e}"
        else:
            # Show the current title and session ID
            title = self._session_db.get_session_title(session_id)
            if title:
                return f"📌 Session: `{session_id}`\nTitle: **{title}**"
            else:
                return f"📌 Session: `{session_id}`\nNo title set. Usage: `/title My Session Name`"

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command — switch to a previously-named session."""
        if not self._session_db:
            return "Session database not available."

        source = event.source
        session_key = self._session_key_for_source(source)
        name = event.get_command_args().strip()

        if not name:
            # List recent titled sessions for this user/platform
            try:
                user_source = source.platform.value if source.platform else None
                sessions = self._session_db.list_sessions_rich(
                    source=user_source, limit=10
                )
                titled = [s for s in sessions if s.get("title")]
                if not titled:
                    return (
                        "No named sessions found.\n"
                        "Use `/title My Session` to name your current session, "
                        "then `/resume My Session` to return to it later."
                    )
                lines = ["📋 **Named Sessions**\n"]
                for s in titled[:10]:
                    title = s["title"]
                    preview = s.get("preview", "")[:40]
                    preview_part = f" — _{preview}_" if preview else ""
                    lines.append(f"• **{title}**{preview_part}")
                lines.append("\nUsage: `/resume <session name>`")
                return "\n".join(lines)
            except Exception as e:
                logger.debug("Failed to list titled sessions: %s", e)
                return f"Could not list sessions: {e}"

        # Resolve the name to a session ID
        target_id = self._session_db.resolve_session_by_title(name)
        if not target_id:
            return (
                f"No session found matching '**{name}**'.\n"
                "Use `/resume` with no arguments to see available sessions."
            )

        # Check if already on that session
        current_entry = self.session_store.get_or_create_session(source)
        if current_entry.session_id == target_id:
            return f"📌 Already on session **{name}**."

        # Flush memories for current session before switching
        try:
            _flush_task = asyncio.create_task(
                self._async_flush_memories(current_entry.session_id)
            )
            self._background_tasks.add(_flush_task)
            _flush_task.add_done_callback(self._background_tasks.discard)
        except Exception as e:
            logger.debug("Memory flush on resume failed: %s", e)

        # Clear any running agent for this session key
        if session_key in self._running_agents:
            del self._running_agents[session_key]

        # Switch the session entry to point at the old session
        new_entry = self.session_store.switch_session(session_key, target_id)
        if not new_entry:
            return "Failed to switch session."

        # Get the title for confirmation
        title = self._session_db.get_session_title(target_id) or name

        # Count messages for context
        history = self.session_store.load_transcript(target_id)
        msg_count = len([m for m in history if m.get("role") == "user"]) if history else 0
        msg_part = f" ({msg_count} message{'s' if msg_count != 1 else ''})" if msg_count else ""

        return f"↻ Resumed session **{title}**{msg_part}. Conversation restored."

    async def _handle_branch_command(self, event: MessageEvent) -> str:
        """Handle /branch [name] — fork the current session into a new independent copy.

        Copies conversation history to a new session so the user can explore
        a different approach without losing the original.
        Inspired by Claude Code's /branch command.
        """
        import uuid as _uuid

        if not self._session_db:
            return "Session database not available."

        source = event.source
        session_key = self._session_key_for_source(source)

        # Load the current session and its transcript
        current_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(current_entry.session_id)
        if not history:
            return "No conversation to branch — send a message first."

        branch_name = event.get_command_args().strip()

        # Generate the new session ID
        from datetime import datetime as _dt
        now = _dt.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        short_uuid = _uuid.uuid4().hex[:6]
        new_session_id = f"{timestamp_str}_{short_uuid}"

        # Determine branch title
        if branch_name:
            branch_title = branch_name
        else:
            current_title = self._session_db.get_session_title(current_entry.session_id)
            base = current_title or "branch"
            branch_title = self._session_db.get_next_title_in_lineage(base)

        parent_session_id = current_entry.session_id

        # Create the new session with parent link
        try:
            self._session_db.create_session(
                session_id=new_session_id,
                source=source.platform.value if source.platform else "gateway",
                model=(self.config.get("model", {}) or {}).get("default") if isinstance(self.config, dict) else None,
                parent_session_id=parent_session_id,
            )
        except Exception as e:
            logger.error("Failed to create branch session: %s", e)
            return f"Failed to create branch: {e}"

        # Copy conversation history to the new session
        for msg in history:
            try:
                self._session_db.append_message(
                    session_id=new_session_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content"),
                    tool_name=msg.get("tool_name") or msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    reasoning=msg.get("reasoning"),
                )
            except Exception:
                pass  # Best-effort copy

        # Set title
        try:
            self._session_db.set_session_title(new_session_id, branch_title)
        except Exception:
            pass

        # Switch the session store entry to the new session
        new_entry = self.session_store.switch_session(session_key, new_session_id)
        if not new_entry:
            return "Branch created but failed to switch to it."

        # Evict any cached agent for this session
        self._evict_cached_agent(session_key)

        msg_count = len([m for m in history if m.get("role") == "user"])
        return (
            f"⑂ Branched to **{branch_title}**"
            f" ({msg_count} message{'s' if msg_count != 1 else ''} copied)\n"
            f"Original: `{parent_session_id}`\n"
            f"Branch: `{new_session_id}`\n"
            f"Use `/resume` to switch back to the original."
        )

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the session's last agent run."""
        source = event.source
        session_key = self._session_key_for_source(source)

        agent = self._running_agents.get(session_key)
        if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
            lines = [
                "📊 **Session Token Usage**",
                f"Prompt (input): {agent.session_prompt_tokens:,}",
                f"Completion (output): {agent.session_completion_tokens:,}",
                f"Total: {agent.session_total_tokens:,}",
                f"API calls: {agent.session_api_calls}",
            ]
            ctx = agent.context_compressor
            if ctx.last_prompt_tokens:
                pct = min(100, ctx.last_prompt_tokens / ctx.context_length * 100) if ctx.context_length else 0
                lines.append(f"Context: {ctx.last_prompt_tokens:,} / {ctx.context_length:,} ({pct:.0f}%)")
            if ctx.compression_count:
                lines.append(f"Compressions: {ctx.compression_count}")
            return "\n".join(lines)

        # No running agent -- check session history for a rough count
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough
            msgs = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
            approx = estimate_messages_tokens_rough(msgs)
            return (
                f"📊 **Session Info**\n"
                f"Messages: {len(msgs)}\n"
                f"Estimated context: ~{approx:,} tokens\n"
                f"_(Detailed usage available during active conversations)_"
            )
        return "No usage data available for this session."

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        import asyncio as _asyncio

        args = event.get_command_args().strip()
        days = 30
        source = None

        # Parse simple args: /insights 7  or  /insights --days 7
        if args:
            parts = args.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--days" and i + 1 < len(parts):
                    try:
                        days = int(parts[i + 1])
                    except ValueError:
                        return f"Invalid --days value: {parts[i + 1]}"
                    i += 2
                elif parts[i] == "--source" and i + 1 < len(parts):
                    source = parts[i + 1]
                    i += 2
                elif parts[i].isdigit():
                    days = int(parts[i])
                    i += 1
                else:
                    i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            loop = _asyncio.get_event_loop()

            def _run_insights():
                db = SessionDB()
                engine = InsightsEngine(db)
                report = engine.generate(days=days, source=source)
                result = engine.format_gateway(report)
                db.close()
                return result

            return await loop.run_in_executor(None, _run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return f"Error generating insights: {e}"

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> str:
        """Handle /reload-mcp command -- disconnect and reconnect all MCP servers."""
        loop = asyncio.get_event_loop()
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _load_mcp_config, _servers, _lock

            # Capture old server names before shutdown
            with _lock:
                old_servers = set(_servers.keys())

            # Read new config before shutting down, so we know what will be added/removed
            # Shutdown existing connections
            await loop.run_in_executor(None, shutdown_mcp_servers)

            # Reconnect by discovering tools (reads config.yaml fresh)
            new_tools = await loop.run_in_executor(None, discover_mcp_tools)

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            lines = ["🔄 **MCP Servers Reloaded**\n"]
            if reconnected:
                lines.append(f"♻️ Reconnected: {', '.join(sorted(reconnected))}")
            if added:
                lines.append(f"➕ Added: {', '.join(sorted(added))}")
            if removed:
                lines.append(f"➖ Removed: {', '.join(sorted(removed))}")
            if not connected_servers:
                lines.append("No MCP servers connected.")
            else:
                lines.append(f"\n🔧 {len(new_tools)} tool(s) available from {len(connected_servers)} server(s)")

            # Inject a message at the END of the session history so the
            # model knows tools changed on its next turn.  Appended after
            # all existing messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[SYSTEM: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            try:
                session_entry = self.session_store.get_or_create_session(event.source)
                self.session_store.append_to_transcript(
                    session_entry.session_id, reload_msg
                )
            except Exception:
                pass  # Best-effort; don't fail the reload over a transcript write

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return f"❌ MCP reload failed: {e}"

    # ------------------------------------------------------------------
    # /approve & /deny — explicit dangerous-command approval
    # ------------------------------------------------------------------

    _APPROVAL_TIMEOUT_SECONDS = 300  # 5 minutes

    async def _handle_approve_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /approve command — unblock waiting agent thread(s).

        The agent thread(s) are blocked inside tools/approval.py waiting for
        the user to respond.  This handler signals the event so the agent
        resumes and the terminal_tool executes the command inline — the same
        flow as the CLI's synchronous input() approval.

        Supports multiple concurrent approvals (parallel subagents,
        execute_code).  ``/approve`` resolves the oldest pending command;
        ``/approve all`` resolves every pending command at once.

        Usage:
            /approve              — approve oldest pending command once
            /approve all          — approve ALL pending commands at once
            /approve session      — approve oldest + remember for session
            /approve all session  — approve all + remember for session
            /approve always       — approve oldest + remember permanently
            /approve all always   — approve all + remember permanently
        """
        source = event.source
        admin_only_message = self._admin_only_message(
            source,
            "approve dangerous commands",
        )
        if admin_only_message:
            return admin_only_message
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            has_blocking_approval, peek_blocking_approval, resolve_gateway_approval,
        )
        store = self._get_background_job_store()

        if not has_blocking_approval(session_key):
            external_approval = store.peek_pending_approval_request(session_key)
            if external_approval:
                current_approval = external_approval
            else:
                if session_key in self._pending_approvals:
                    self._pending_approvals.pop(session_key)
                    return "⚠️ Approval expired (agent is no longer waiting). Ask the agent to try again."
                return "No pending command to approve."
        else:
            current_approval = peek_blocking_approval(session_key) or {}

        allow_persistence = bool(current_approval.get("allow_persistence", True))

        # Parse args: support "all", "all session", "all always", "session", "always"
        args = event.get_command_args().strip().lower().split()
        resolve_all = "all" in args
        remaining = [a for a in args if a != "all"]

        if any(a in ("always", "permanent", "permanently") for a in remaining):
            choice = "always"
            scope_msg = " (pattern approved permanently)"
        elif any(a in ("session", "ses") for a in remaining):
            choice = "session"
            scope_msg = " (pattern approved for this session)"
        else:
            choice = "once"
            scope_msg = ""

        if not resolve_all and not allow_persistence and choice in {"session", "always"}:
            choice = "once"
            scope_msg = " (approved for this action only)"

        if has_blocking_approval(session_key):
            count = resolve_gateway_approval(session_key, choice, resolve_all=resolve_all)
        else:
            count = store.resolve_approval_requests(
                session_key=session_key,
                choice=choice,
                resolve_all=resolve_all,
            )
        if not count:
            return "No pending command to approve."

        # Resume typing indicator — agent is about to continue processing.
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        count_msg = f" ({count} commands)" if count > 1 else ""
        logger.info("User approved %d dangerous command(s) via /approve%s", count, scope_msg)
        return f"✅ Command{'s' if count > 1 else ''} approved{scope_msg}{count_msg}. The agent is resuming..."


    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command — reject pending dangerous command(s).

        Signals blocked agent thread(s) with a 'deny' result so they receive
        a definitive BLOCKED message, same as the CLI deny flow.

        ``/deny`` denies the oldest; ``/deny all`` denies everything.
        """
        source = event.source
        admin_only_message = self._admin_only_message(
            source,
            "deny dangerous commands",
        )
        if admin_only_message:
            return admin_only_message
        session_key = self._session_key_for_source(source)

        from tools.approval import (
            resolve_gateway_approval, has_blocking_approval,
        )
        store = self._get_background_job_store()

        if not has_blocking_approval(session_key):
            if not store.has_pending_approval_requests(session_key):
                if session_key in self._pending_approvals:
                    self._pending_approvals.pop(session_key)
                    return "❌ Command denied (approval was stale)."
                return "No pending command to deny."

        args = event.get_command_args().strip().lower()
        resolve_all = "all" in args

        if has_blocking_approval(session_key):
            count = resolve_gateway_approval(session_key, "deny", resolve_all=resolve_all)
        else:
            count = store.resolve_approval_requests(
                session_key=session_key,
                choice="deny",
                resolve_all=resolve_all,
            )
        if not count:
            return "No pending command to deny."

        # Resume typing indicator — agent continues (with BLOCKED result).
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            _adapter.resume_typing_for_chat(source.chat_id)

        count_msg = f" ({count} commands)" if count > 1 else ""
        logger.info("User denied %d dangerous command(s) via /deny", count)
        return f"❌ Command{'s' if count > 1 else ''} denied{count_msg}."

    # Platforms where /update is allowed.  ACP, API server, and webhooks are
    # programmatic interfaces that should not trigger system updates.
    _UPDATE_ALLOWED_PLATFORMS = frozenset({
        Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK, Platform.WHATSAPP,
        Platform.SIGNAL, Platform.MATTERMOST, Platform.MATRIX,
        Platform.HOMEASSISTANT, Platform.EMAIL, Platform.SMS, Platform.DINGTALK,
        Platform.FEISHU, Platform.WECOM, Platform.WECOM_CALLBACK,
        Platform.WEIXIN, Platform.LOCAL,
    })

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update command — update Hermes Agent to the latest version.

        Spawns ``hermes update`` in a detached session (via ``setsid``) so it
        survives the gateway restart that ``hermes update`` may trigger. Marker
        files are written so either the current gateway process or the next one
        can notify the user when the update finishes.
        """
        import json
        import shutil
        import subprocess
        from datetime import datetime
        from hermes_cli.config import is_managed, format_managed_message

        # Block non-messaging platforms (API server, webhooks, ACP)
        platform = event.source.platform
        if platform not in self._UPDATE_ALLOWED_PLATFORMS:
            return "✗ /update is only available from messaging platforms. Run `hermes update` from the terminal."

        if is_managed():
            return f"✗ {format_managed_message('update Hermes Agent')}"

        project_root = Path(__file__).parent.parent.resolve()
        git_dir = project_root / '.git'

        if not git_dir.exists():
            return "✗ Not a git repository — cannot update."

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            return (
                "✗ Could not locate the `hermes` command. "
                "Hermes is running, but the update command could not find the "
                "executable on PATH or via the current Python interpreter. "
                "Try running `hermes update` manually in your terminal."
            )

        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        session_key = self._session_key_for_source(event.source)
        pending = {
            "platform": event.source.platform.value,
            "chat_id": event.source.chat_id,
            "user_id": event.source.user_id,
            "session_key": session_key,
            "timestamp": datetime.now().isoformat(),
        }
        _tmp_pending = pending_path.with_suffix(".tmp")
        _tmp_pending.write_text(json.dumps(pending))
        _tmp_pending.replace(pending_path)
        exit_code_path.unlink(missing_ok=True)

        # Spawn `hermes update --gateway` detached so it survives gateway restart.
        # --gateway enables file-based IPC for interactive prompts (stash
        # restore, config migration) so the gateway can forward them to the
        # user instead of silently skipping them.
        # Use setsid for portable session detach (works under system services
        # where systemd-run --user fails due to missing D-Bus session).
        # PYTHONUNBUFFERED ensures output is flushed line-by-line so the
        # gateway can stream it to the messenger in near-real-time.
        hermes_cmd_str = " ".join(shlex.quote(part) for part in hermes_cmd)
        update_cmd = (
            f"PYTHONUNBUFFERED=1 {hermes_cmd_str} update --gateway"
            f" > {shlex.quote(str(output_path))} 2>&1; "
            f"status=$?; printf '%s' \"$status\" > {shlex.quote(str(exit_code_path))}"
        )
        try:
            setsid_bin = shutil.which("setsid")
            if setsid_bin:
                # Preferred: setsid creates a new session, fully detached
                subprocess.Popen(
                    [setsid_bin, "bash", "-c", update_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                # Fallback: start_new_session=True calls os.setsid() in child
                subprocess.Popen(
                    ["bash", "-c", update_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except Exception as e:
            pending_path.unlink(missing_ok=True)
            exit_code_path.unlink(missing_ok=True)
            return f"✗ Failed to start update: {e}"

        self._schedule_update_notification_watch()
        return "⚕ Starting Hermes update… I'll stream progress here."

    def _schedule_update_notification_watch(self) -> None:
        """Ensure a background task is watching for update completion."""
        existing_task = getattr(self, "_update_notification_task", None)
        if existing_task and not existing_task.done():
            return

        try:
            self._update_notification_task = asyncio.create_task(
                self._watch_update_progress()
            )
        except RuntimeError:
            logger.debug("Skipping update notification watcher: no running event loop")

    async def _watch_update_progress(
        self,
        poll_interval: float = 2.0,
        stream_interval: float = 4.0,
        timeout: float = 1800.0,
    ) -> None:
        """Watch ``hermes update --gateway``, streaming output + forwarding prompts.

        Polls ``.update_output.txt`` for new content and sends chunks to the
        user periodically.  Detects ``.update_prompt.json`` (written by the
        update process when it needs user input) and forwards the prompt to
        the messenger.  The user's next message is intercepted by
        ``_handle_message`` and written to ``.update_response``.
        """
        import json
        import re as _re

        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        prompt_path = _hermes_home / ".update_prompt.json"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        # Resolve the adapter and chat_id for sending messages
        adapter = None
        chat_id = None
        session_key = None
        for path in (claimed_path, pending_path):
            if path.exists():
                try:
                    pending = json.loads(path.read_text())
                    platform_str = pending.get("platform")
                    chat_id = pending.get("chat_id")
                    session_key = pending.get("session_key")
                    if platform_str and chat_id:
                        platform = Platform(platform_str)
                        adapter = self.adapters.get(platform)
                        # Fallback session key if not stored (old pending files)
                        if not session_key:
                            session_key = f"{platform_str}:{chat_id}"
                    break
                except Exception:
                    pass

        if not adapter or not chat_id:
            logger.warning("Update watcher: cannot resolve adapter/chat_id, falling back to completion-only")
            # Fall back to old behavior: wait for exit code and send final notification
            while (pending_path.exists() or claimed_path.exists()) and loop.time() < deadline:
                if exit_code_path.exists():
                    await self._send_update_notification()
                    return
                await asyncio.sleep(poll_interval)
            if (pending_path.exists() or claimed_path.exists()) and not exit_code_path.exists():
                exit_code_path.write_text("124")
                await self._send_update_notification()
            return

        def _strip_ansi(text: str) -> str:
            return _re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)

        bytes_sent = 0
        last_stream_time = loop.time()
        buffer = ""

        async def _flush_buffer() -> None:
            """Send buffered output to the user."""
            nonlocal buffer, last_stream_time
            if not buffer.strip():
                buffer = ""
                return
            # Chunk to fit message limits (Telegram: 4096, others: generous)
            clean = _strip_ansi(buffer).strip()
            buffer = ""
            last_stream_time = loop.time()
            if not clean:
                return
            # Split into chunks if too long
            max_chunk = 3500
            chunks = [clean[i:i + max_chunk] for i in range(0, len(clean), max_chunk)]
            for chunk in chunks:
                try:
                    await adapter.send(chat_id, f"```\n{chunk}\n```")
                except Exception as e:
                    logger.debug("Update stream send failed: %s", e)

        while loop.time() < deadline:
            # Check for completion
            if exit_code_path.exists():
                # Read any remaining output
                if output_path.exists():
                    try:
                        content = output_path.read_text()
                        if len(content) > bytes_sent:
                            buffer += content[bytes_sent:]
                            bytes_sent = len(content)
                    except OSError:
                        pass
                await _flush_buffer()

                # Send final status
                try:
                    exit_code_raw = exit_code_path.read_text().strip() or "1"
                    exit_code = int(exit_code_raw)
                    if exit_code == 0:
                        await adapter.send(chat_id, "✅ Hermes update finished.")
                    else:
                        await adapter.send(chat_id, "❌ Hermes update failed (exit code {}).".format(exit_code))
                    logger.info("Update finished (exit=%s), notified %s", exit_code, session_key)
                except Exception as e:
                    logger.warning("Update final notification failed: %s", e)

                # Cleanup
                for p in (pending_path, claimed_path, output_path,
                          exit_code_path, prompt_path):
                    p.unlink(missing_ok=True)
                (_hermes_home / ".update_response").unlink(missing_ok=True)
                self._update_prompt_pending.pop(session_key, None)
                return

            # Check for new output
            if output_path.exists():
                try:
                    content = output_path.read_text()
                    if len(content) > bytes_sent:
                        buffer += content[bytes_sent:]
                        bytes_sent = len(content)
                except OSError:
                    pass

            # Flush buffer periodically
            if buffer.strip() and (loop.time() - last_stream_time) >= stream_interval:
                await _flush_buffer()

            # Check for prompts
            if prompt_path.exists() and session_key:
                try:
                    prompt_data = json.loads(prompt_path.read_text())
                    prompt_text = prompt_data.get("prompt", "")
                    default = prompt_data.get("default", "")
                    if prompt_text:
                        # Flush any buffered output first so the user sees
                        # context before the prompt
                        await _flush_buffer()
                        # Try platform-native buttons first (Discord, Telegram)
                        sent_buttons = False
                        if getattr(type(adapter), "send_update_prompt", None) is not None:
                            try:
                                await adapter.send_update_prompt(
                                    chat_id=chat_id,
                                    prompt=prompt_text,
                                    default=default,
                                    session_key=session_key,
                                )
                                sent_buttons = True
                            except Exception as btn_err:
                                logger.debug("Button-based update prompt failed: %s", btn_err)
                        if not sent_buttons:
                            default_hint = f" (default: {default})" if default else ""
                            await adapter.send(
                                chat_id,
                                f"⚕ **Update needs your input:**\n\n"
                                f"{prompt_text}{default_hint}\n\n"
                                f"Reply `/approve` (yes) or `/deny` (no), "
                                f"or type your answer directly."
                            )
                        self._update_prompt_pending[session_key] = True
                        logger.info("Forwarded update prompt to %s: %s", session_key, prompt_text[:80])
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Failed to read update prompt: %s", e)

            await asyncio.sleep(poll_interval)

        # Timeout
        if not exit_code_path.exists():
            logger.warning("Update watcher timed out after %.0fs", timeout)
            exit_code_path.write_text("124")
            await _flush_buffer()
            try:
                await adapter.send(chat_id, "❌ Hermes update timed out after 30 minutes.")
            except Exception:
                pass
            for p in (pending_path, claimed_path, output_path,
                      exit_code_path, prompt_path):
                p.unlink(missing_ok=True)
            (_hermes_home / ".update_response").unlink(missing_ok=True)
            self._update_prompt_pending.pop(session_key, None)

    async def _send_update_notification(self) -> bool:
        """If an update finished, notify the user.

        Returns False when the update is still running so a caller can retry
        later. Returns True after a definitive send/skip decision.

        This is the legacy notification path used when the streaming watcher
        cannot resolve the adapter (e.g. after a gateway restart where the
        platform hasn't reconnected yet).
        """
        import json
        import re as _re

        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"

        if not pending_path.exists() and not claimed_path.exists():
            return False

        cleanup = True
        active_pending_path = claimed_path
        try:
            if pending_path.exists():
                try:
                    pending_path.replace(claimed_path)
                except FileNotFoundError:
                    if not claimed_path.exists():
                        return True
            elif not claimed_path.exists():
                return True

            pending = json.loads(claimed_path.read_text())
            platform_str = pending.get("platform")
            chat_id = pending.get("chat_id")

            if not exit_code_path.exists():
                logger.info("Update notification deferred: update still running")
                cleanup = False
                active_pending_path = pending_path
                claimed_path.replace(pending_path)
                return False

            exit_code_raw = exit_code_path.read_text().strip() or "1"
            exit_code = int(exit_code_raw)

            # Read the captured update output
            output = ""
            if output_path.exists():
                output = output_path.read_text()

            # Resolve adapter
            platform = Platform(platform_str)
            adapter = self.adapters.get(platform)

            if adapter and chat_id:
                # Strip ANSI escape codes for clean display
                output = _re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
                if output:
                    if len(output) > 3500:
                        output = "…" + output[-3500:]
                    if exit_code == 0:
                        msg = f"✅ Hermes update finished.\n\n```\n{output}\n```"
                    else:
                        msg = f"❌ Hermes update failed.\n\n```\n{output}\n```"
                else:
                    if exit_code == 0:
                        msg = "✅ Hermes update finished successfully."
                    else:
                        msg = "❌ Hermes update failed. Check the gateway logs or run `hermes update` manually for details."
                await adapter.send(chat_id, msg)
                logger.info(
                    "Sent post-update notification to %s:%s (exit=%s)",
                    platform_str,
                    chat_id,
                    exit_code,
                )
        except Exception as e:
            logger.warning("Post-update notification failed: %s", e)
        finally:
            if cleanup:
                active_pending_path.unlink(missing_ok=True)
                claimed_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                exit_code_path.unlink(missing_ok=True)

        return True

    def _set_session_env(self, context: SessionContext) -> None:
        """Set environment variables for the current session."""
        os.environ["HERMES_SESSION_PLATFORM"] = context.source.platform.value
        os.environ["HERMES_SESSION_CHAT_ID"] = context.source.chat_id
        if context.source.chat_name:
            os.environ["HERMES_SESSION_CHAT_NAME"] = context.source.chat_name
        if context.source.chat_type:
            os.environ["HERMES_SESSION_CHAT_TYPE"] = context.source.chat_type
        if context.source.thread_id:
            os.environ["HERMES_SESSION_THREAD_ID"] = str(context.source.thread_id)
        if context.source.user_id:
            os.environ["HERMES_SESSION_USER_ID"] = str(context.source.user_id)
        if context.source.user_name:
            os.environ["HERMES_SESSION_USER_NAME"] = str(context.source.user_name)
        if context.admin_user_ids:
            os.environ["HERMES_SESSION_ADMIN_USER_IDS"] = ",".join(context.admin_user_ids)
        if context.is_admin_user is not None:
            os.environ["HERMES_SESSION_IS_ADMIN"] = str(context.is_admin_user).lower()
    
    def _clear_session_env(self) -> None:
        """Clear session environment variables."""
        for var in [
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
            "HERMES_SESSION_CHAT_TYPE",
            "HERMES_SESSION_THREAD_ID",
            "HERMES_SESSION_USER_ID",
            "HERMES_SESSION_USER_NAME",
            "HERMES_SESSION_ADMIN_USER_IDS",
            "HERMES_SESSION_IS_ADMIN",
        ]:
            if var in os.environ:
                del os.environ[var]
    
    async def _enrich_message_with_vision(
        self,
        user_text: str,
        image_paths: List[str],
        *,
        source: Optional[SessionSource] = None,
    ) -> str:
        """
        Opportunistically analyze user-attached images without blocking the
        foreground chat path.

        Successful analyses are cached per local image file so later turns can
        reuse the description instantly.  Cache misses warm in a background
        task; the foreground waits only a very short grace period and then
        degrades to a non-blocking placeholder note.

        Args:
            user_text:   The user's original caption / message text.
            image_paths: List of local file paths to cached images.
            source:      Optional session source used for chat-aware latency caps.

        Returns:
            The enriched message string with vision descriptions prepended.
        """
        self._ensure_auto_vision_state()
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )
        analysis_timeout = self._auto_vision_analysis_timeout_seconds()
        inline_wait = self._auto_vision_inline_wait_seconds(
            source,
            has_user_text=bool(str(user_text or "").strip()),
        )
        inline_deadline = time.monotonic() + inline_wait if inline_wait > 0 else None

        enriched_parts = []
        pending_tasks: List[tuple[str, str, asyncio.Task]] = []
        for path in image_paths:
            cache_key = self._auto_vision_cache_key(path)
            cached = self._get_auto_vision_cache_entry(cache_key)
            if cached:
                if cached.get("status") == "success":
                    description = str(cached.get("analysis") or "").strip()
                    if description:
                        enriched_parts.append(
                            f"[The user sent an image~ Here's what I can see:\n{description}]"
                        )
                        continue
                elif cached.get("status") == "error":
                    enriched_parts.append(self._auto_vision_degraded_note(path, pending=False))
                    continue

            remaining, reason = self._auto_vision_cooldown_remaining()
            if remaining > 0:
                logger.debug(
                    "Skipping vision auto-analysis for %.1fs after %s (image=%s)",
                    remaining,
                    reason or "recent_failure",
                    path,
                )
                enriched_parts.append(self._auto_vision_degraded_note(path, pending=False))
                continue

            task = self._start_auto_vision_task(
                cache_key=cache_key,
                path=path,
                analysis_prompt=analysis_prompt,
                analysis_timeout=analysis_timeout,
            )
            if task is None:
                enriched_parts.append(self._auto_vision_degraded_note(path, pending=False))
                continue
            pending_tasks.append((path, cache_key, task))

        completed_tasks: set[asyncio.Task] = set()
        if pending_tasks:
            if inline_deadline is not None:
                remaining_inline = max(0.0, inline_deadline - time.monotonic())
            else:
                remaining_inline = 0.0
            if remaining_inline > 0:
                done, _pending = await asyncio.wait(
                    [task for _, _, task in pending_tasks],
                    timeout=remaining_inline,
                )
                completed_tasks = set(done)

        for path, cache_key, task in pending_tasks:
            task_entry: Optional[Dict[str, Any]] = None
            if task in completed_tasks or task.done():
                try:
                    task_entry = task.result()
                except Exception as exc:
                    logger.debug("Auto vision background task failed for %s: %s", path, exc)
                    task_entry = self._get_auto_vision_cache_entry(cache_key)

            if task_entry and task_entry.get("status") == "success":
                description = str(task_entry.get("analysis") or "").strip()
                if description:
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]"
                    )
                    continue

            if task_entry and task_entry.get("status") == "error":
                enriched_parts.append(self._auto_vision_degraded_note(path, pending=False))
                continue

            enriched_parts.append(self._auto_vision_degraded_note(path, pending=True))

        # Combine: vision descriptions first, then the user's original text
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    def _ensure_auto_vision_state(self) -> None:
        shared_ensure_auto_vision_state(self)

    def _prune_auto_vision_state(self) -> None:
        shared_prune_auto_vision_state(
            self,
            max_cache_entries=int(_AUTO_VISION_MAX_CACHE_ENTRIES),
        )

    def _auto_vision_cache_key(self, path: str) -> str:
        return shared_auto_vision_cache_key(path)

    def _get_auto_vision_cache_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return shared_get_auto_vision_cache_entry(
            self,
            cache_key,
            max_cache_entries=int(_AUTO_VISION_MAX_CACHE_ENTRIES),
        )

    def _auto_vision_inline_wait_seconds(
        self,
        source: Optional[SessionSource] = None,
        *,
        has_user_text: bool = True,
    ) -> float:
        wait_seconds = float(_AUTO_VISION_INLINE_WAIT_SECONDS)
        image_only_wait = float(_AUTO_VISION_IMAGE_ONLY_INLINE_WAIT_SECONDS)
        try:
            cfg = _load_gateway_config()
            vision_cfg = (((cfg or {}).get("auxiliary") or {}).get("vision") or {})
            explicit = vision_cfg.get("auto_inline_wait")
            if explicit is not None:
                configured_wait = float(explicit)
                if configured_wait >= 0:
                    wait_seconds = configured_wait
            image_only_explicit = vision_cfg.get("image_only_inline_wait")
            if image_only_explicit is not None:
                configured_image_only_wait = float(image_only_explicit)
                if configured_image_only_wait >= 0:
                    image_only_wait = configured_image_only_wait
        except Exception:
            pass
        if not has_user_text:
            return max(0.0, min(image_only_wait, self._auto_vision_analysis_timeout_seconds()))
        chat_type = str(getattr(source, "chat_type", "") or "").strip().lower()
        if chat_type and chat_type != "dm":
            wait_seconds = min(wait_seconds, float(_AUTO_VISION_GROUP_INLINE_WAIT_SECONDS))
        return max(0.0, wait_seconds)

    def _auto_vision_degraded_note(self, path: str, *, pending: bool) -> str:
        return shared_auto_vision_degraded_note(path, pending=pending)

    def _classify_auto_vision_failure(self, error_text: str) -> str:
        return shared_classify_auto_vision_failure(error_text)

    async def _run_auto_vision_task(
        self,
        *,
        cache_key: str,
        path: str,
        analysis_prompt: str,
        analysis_timeout: float,
    ) -> Dict[str, Any]:
        from tools.vision_tools import vision_analyze_tool

        self._ensure_auto_vision_state()
        try:
            logger.debug("Auto-analyzing user image in background: %s", path)
            result_json = await asyncio.wait_for(
                vision_analyze_tool(
                    image_url=path,
                    user_prompt=analysis_prompt,
                ),
                timeout=analysis_timeout,
            )
            result = json.loads(result_json)
            if result.get("success"):
                description = str(result.get("analysis") or "").strip()
                if description:
                    now = time.time()
                    entry = {
                        "status": "success",
                        "analysis": description,
                        "updated_at": now,
                        "expires_at": now + _AUTO_VISION_CACHE_TTL_SECONDS,
                    }
                    self._auto_vision_cache[cache_key] = entry
                    self._clear_auto_vision_cooldown()
                    return entry
            error_text = str(result.get("error") or result.get("analysis") or "auto vision failed").strip()
            classification = self._classify_auto_vision_failure(error_text)
            cooldown_seconds = _AUTO_VISION_ANALYSIS_FAILURE_COOLDOWN_SECONDS
            if classification == "transient":
                cooldown_seconds = _AUTO_VISION_TRANSIENT_COOLDOWN_SECONDS
            if classification in {"deterministic", "transient"}:
                self._mark_auto_vision_cooldown(
                    reason=classification,
                    seconds=cooldown_seconds,
                )
            entry = {
                "status": "error",
                "error": error_text or "auto vision failed",
            }
            if classification == "deterministic":
                now = time.time()
                entry["updated_at"] = now
                entry["expires_at"] = now + cooldown_seconds
                self._auto_vision_cache[cache_key] = entry
            return entry
        except asyncio.TimeoutError:
            self._mark_auto_vision_cooldown(
                reason="timeout",
                seconds=_AUTO_VISION_TRANSIENT_COOLDOWN_SECONDS,
            )
            logger.warning(
                "Vision auto-analysis timed out after %.2fs for %s; "
                "skipping auto-analysis for %.1fs",
                analysis_timeout,
                path,
                _AUTO_VISION_TRANSIENT_COOLDOWN_SECONDS,
            )
            entry = {
                "status": "error",
                "error": "Vision auto-analysis timed out",
            }
            return entry
        except Exception as exc:
            error_text = str(exc)
            classification = self._classify_auto_vision_failure(error_text)
            cooldown_seconds = _AUTO_VISION_ANALYSIS_FAILURE_COOLDOWN_SECONDS
            if classification == "transient":
                cooldown_seconds = _AUTO_VISION_TRANSIENT_COOLDOWN_SECONDS
            if classification in {"deterministic", "transient"}:
                self._mark_auto_vision_cooldown(
                    reason=classification,
                    seconds=cooldown_seconds,
                )
            logger.error("Vision auto-analysis error: %s", exc)
            entry = {
                "status": "error",
                "error": error_text or "Vision auto-analysis error",
            }
            if classification == "deterministic":
                now = time.time()
                entry["updated_at"] = now
                entry["expires_at"] = now + cooldown_seconds
                self._auto_vision_cache[cache_key] = entry
            return entry
        finally:
            current = asyncio.current_task()
            existing = self._auto_vision_tasks.get(cache_key)
            if existing is current:
                self._auto_vision_tasks.pop(cache_key, None)

    def _start_auto_vision_task(
        self,
        *,
        cache_key: str,
        path: str,
        analysis_prompt: str,
        analysis_timeout: float,
    ) -> Optional[asyncio.Task]:
        self._prune_auto_vision_state()
        existing = self._auto_vision_tasks.get(cache_key)
        if existing and not existing.done():
            return existing
        inflight_count = 0
        for task in self._auto_vision_tasks.values():
            try:
                if not task.done():
                    inflight_count += 1
            except Exception:
                continue
        if inflight_count >= int(_AUTO_VISION_MAX_INFLIGHT_TASKS):
            logger.debug(
                "Skipping auto vision warmup for %s; inflight limit reached (%d)",
                path,
                inflight_count,
            )
            return None
        task = asyncio.create_task(
            self._run_auto_vision_task(
                cache_key=cache_key,
                path=path,
                analysis_prompt=analysis_prompt,
                analysis_timeout=analysis_timeout,
            )
        )
        self._auto_vision_tasks[cache_key] = task
        self._background_tasks.add(task)
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)
        return task

    def _auto_vision_analysis_timeout_seconds(self) -> float:
        """Return the timeout for opportunistic auto vision analysis."""
        timeout = float(_AUTO_VISION_ANALYSIS_TIMEOUT_SECONDS)
        try:
            cfg = _load_gateway_config()
            vision_cfg = (((cfg or {}).get("auxiliary") or {}).get("vision") or {})
            explicit = vision_cfg.get("auto_timeout")
            if explicit is not None:
                explicit_timeout = float(explicit)
                if explicit_timeout > 0:
                    return explicit_timeout

            configured = vision_cfg.get("timeout")
            if configured is not None:
                configured_timeout = float(configured)
                if configured_timeout > 0:
                    return min(
                        configured_timeout,
                        _AUTO_VISION_ANALYSIS_TIMEOUT_CAP_SECONDS,
                    )
        except Exception:
            pass
        return timeout

    def _auto_vision_cooldown_remaining(self) -> tuple[float, str]:
        return shared_auto_vision_cooldown_remaining(self)

    def _mark_auto_vision_cooldown(self, *, reason: str, seconds: float) -> None:
        shared_mark_auto_vision_cooldown(self, reason=reason, seconds=seconds)

    def _clear_auto_vision_cooldown(self) -> None:
        shared_clear_auto_vision_cooldown(self)

    async def _enrich_message_with_transcription(
        self,
        user_text: str,
        audio_paths: List[str],
    ) -> str:
        """
        Auto-transcribe user voice/audio messages using the configured STT provider
        and prepend the transcript to the message text.

        Args:
            user_text:   The user's original caption / message text.
            audio_paths: List of local file paths to cached audio files.

        Returns:
            The enriched message string with transcriptions prepended.
        """
        if not getattr(self.config, "stt_enabled", True):
            disabled_note = "[The user sent voice message(s), but transcription is disabled in config."
            if self._has_setup_skill():
                disabled_note += (
                    " You have a skill called hermes-agent-setup that can help "
                    "users configure Hermes features including voice, tools, and more."
                )
            disabled_note += "]"
            if user_text:
                return f"{disabled_note}\n\n{user_text}"
            return disabled_note

        from tools.transcription_tools import transcribe_audio, get_stt_model_from_config
        import asyncio

        stt_model = get_stt_model_from_config()

        enriched_parts = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(transcribe_audio, path, model=stt_model)
                if result["success"]:
                    transcript = result["transcript"]
                    enriched_parts.append(
                        f'[The user sent a voice message~ '
                        f'Here\'s what they said: "{transcript}"]'
                    )
                else:
                    error = result.get("error", "unknown error")
                    if (
                        "No STT provider" in error
                        or error.startswith("Neither VOICE_TOOLS_OPENAI_KEY nor OPENAI_API_KEY is set")
                    ):
                        _no_stt_note = (
                            "[The user sent a voice message but I can't listen "
                            "to it right now — no STT provider is configured. "
                            "A direct message has already been sent to the user "
                            "with setup instructions."
                        )
                        if self._has_setup_skill():
                            _no_stt_note += (
                                " You have a skill called hermes-agent-setup "
                                "that can help users configure Hermes features "
                                "including voice, tools, and more."
                            )
                        _no_stt_note += "]"
                        enriched_parts.append(_no_stt_note)
                    else:
                        enriched_parts.append(
                            "[The user sent a voice message but I had trouble "
                            f"transcribing it~ ({error})]"
                        )
            except Exception as e:
                logger.error("Transcription error: %s", e)
                enriched_parts.append(
                    "[The user sent a voice message but something went wrong "
                    "when I tried to listen to it~ Let them know!]"
                )

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            # Strip the empty-content placeholder from the Discord adapter
            # when we successfully transcribed the audio — it's redundant.
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _run_process_watcher(self, watcher: dict) -> None:
        """
        Periodically check a background process and push updates to the user.

        Runs as an asyncio task. Stays silent when nothing changed.
        Auto-removes when the process exits or is killed.

        Notification mode (from ``display.background_process_notifications``):
          - ``all``    — running-output updates + final message
          - ``result`` — final completion message only
          - ``error``  — final message only when exit code != 0
          - ``off``    — no messages at all
        """
        from tools.process_registry import process_registry

        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        session_key = watcher.get("session_key", "")
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")
        thread_id = watcher.get("thread_id", "")
        chat_type = str(watcher.get("chat_type", "") or "").lower()
        agent_notify = watcher.get("notify_on_complete", False)
        notify_mode = self._load_background_notifications_mode()

        logger.debug("Process watcher started: %s (every %ss, notify=%s, agent_notify=%s)",
                      session_id, interval, notify_mode, agent_notify)

        if notify_mode == "off" and not agent_notify:
            # Still wait for the process to exit so we can log it, but don't
            # push any messages to the user.
            while True:
                await asyncio.sleep(interval)
                session = process_registry.get(session_id)
                if session is None or session.exited:
                    break
            logger.debug("Process watcher ended (silent): %s", session_id)
            return

        last_output_len = 0
        while True:
            await asyncio.sleep(interval)

            session = process_registry.get(session_id)
            if session is None:
                break

            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len

            if session.exited:
                # --- Agent-triggered completion: inject synthetic message ---
                if agent_notify:
                    from tools.ansi_strip import strip_ansi
                    _out = strip_ansi(session.output_buffer[-2000:]) if session.output_buffer else ""
                    synth_text = (
                        f"[SYSTEM: Background process {session_id} completed "
                        f"(exit code {session.exit_code}).\n"
                        f"Command: {session.command}\n"
                        f"Output:\n{_out}]"
                    )
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter and chat_id:
                        try:
                            from gateway.platforms.base import MessageEvent, MessageType
                            from gateway.session import SessionSource
                            from gateway.config import Platform
                            if (
                                platform_name == "qq_napcat"
                                and chat_id
                                and chat_type in {"group", "dm", "private"}
                                and hasattr(adapter, "_chat_types")
                            ):
                                adapter._chat_types[str(chat_id)] = (
                                    "group" if chat_type == "group" else "private"
                                )
                            _platform_enum = Platform(platform_name)
                            _source = SessionSource(
                                platform=_platform_enum,
                                chat_id=chat_id,
                                chat_type=chat_type or "dm",
                                thread_id=thread_id or None,
                            )
                            synth_event = MessageEvent(
                                text=synth_text,
                                message_type=MessageType.TEXT,
                                source=_source,
                            )
                            logger.info(
                                "Process %s finished — injecting agent notification for session %s",
                                session_id, session_key,
                            )
                            await adapter.handle_message(synth_event)
                        except Exception as e:
                            logger.error("Agent notify injection error: %s", e)
                    break

                # --- Normal text-only notification ---
                # Decide whether to notify based on mode
                should_notify = (
                    notify_mode in ("all", "result")
                    or (notify_mode == "error" and session.exit_code not in (0, None))
                )
                if should_notify:
                    new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                    message_text = (
                        f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                        f"Here's the final output:\n{new_output}]"
                    )
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter and chat_id:
                        try:
                            if (
                                platform_name == "qq_napcat"
                                and chat_id
                                and chat_type in {"group", "dm", "private"}
                                and hasattr(adapter, "_chat_types")
                            ):
                                adapter._chat_types[str(chat_id)] = (
                                    "group" if chat_type == "group" else "private"
                                )
                            send_meta = {"thread_id": thread_id} if thread_id else None
                            await adapter.send(chat_id, message_text, metadata=send_meta)
                        except Exception as e:
                            logger.error("Watcher delivery error: %s", e)
                break

            elif has_new_output and notify_mode == "all" and not agent_notify:
                # New output available -- deliver status update (only in "all" mode)
                # Skip periodic updates for agent_notify watchers (they only care about completion)
                new_output = session.output_buffer[-500:] if session.output_buffer else ""
                message_text = (
                    f"[Background process {session_id} is still running~ "
                    f"New output:\n{new_output}]"
                )
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        if (
                            platform_name == "qq_napcat"
                            and chat_id
                            and chat_type in {"group", "dm", "private"}
                            and hasattr(adapter, "_chat_types")
                        ):
                            adapter._chat_types[str(chat_id)] = (
                                "group" if chat_type == "group" else "private"
                            )
                        send_meta = {"thread_id": thread_id} if thread_id else None
                        await adapter.send(chat_id, message_text, metadata=send_meta)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)

    _MAX_INTERRUPT_DEPTH = 3  # Cap recursive interrupt handling (#816)

    @staticmethod
    def _agent_config_signature(
        model: str,
        runtime: dict,
        enabled_toolsets: list,
        ephemeral_prompt: str,
    ) -> str:
        """Compute a stable string key from agent config values.

        When this signature changes between messages, the cached AIAgent is
        discarded and rebuilt.  When it stays the same, the cached agent is
        reused — preserving the frozen system prompt and tool schemas for
        prompt cache hits.
        """
        return shared_agent_config_signature(
            model,
            runtime,
            enabled_toolsets,
            ephemeral_prompt,
        )

    def _evict_cached_agent(self, session_key: str) -> None:
        """Remove a cached agent for a session (called on /new, /model, etc)."""
        _lock = getattr(self, "_agent_cache_lock", None)
        if _lock:
            with _lock:
                self._agent_cache.pop(session_key, None)

    @staticmethod
    def _should_evict_cached_agent_after_turn(agent, configured_model: str) -> bool:
        """Return True when a post-turn cached agent should be discarded."""
        if agent is None or not hasattr(agent, "model"):
            return False
        if agent.model == configured_model:
            return False

        has_pinned_fallback = getattr(agent, "_has_pinned_fallback", None)
        if callable(has_pinned_fallback):
            try:
                if has_pinned_fallback():
                    return False
            except Exception:
                logger.debug("Could not evaluate fallback pin state", exc_info=True)

        return True

    async def _run_agent(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        event: MessageEvent | None = None,
        admin_user_ids: Optional[List[str]] = None,
        is_admin_user: Optional[bool] = None,
        raw_message: Any = None,
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
        user_config = _load_gateway_config()
        platform_key = _platform_config_key(source.platform)

        from hermes_cli.tools_config import _get_platform_tools
        enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))

        # Apply tool preview length config (0 = no limit)
        try:
            from agent.display import set_tool_preview_max_len
            _tpl = user_config.get("display", {}).get("tool_preview_length", 0)
            set_tool_preview_max_len(int(_tpl) if _tpl else 0)
        except Exception:
            pass

        _loop_for_step = asyncio.get_event_loop()
        _hooks_ref = self.hooks
        progress_runtime = shared_build_gateway_progress_runtime(
            user_config=user_config,
            source=source,
            event_message_id=event_message_id,
            adapter=self.adapters.get(source.platform),
            hooks_ref=_hooks_ref,
            loop_for_step=_loop_for_step,
            session_id=session_id,
            logger=logger,
            should_forward_status=_should_forward_agent_status,
        )
        tool_progress_enabled = progress_runtime.tool_progress_enabled
        _progress_thread_id = progress_runtime.thread_id
        _status_adapter = progress_runtime.status_adapter
        _status_chat_id = progress_runtime.status_chat_id
        _status_thread_metadata = progress_runtime.thread_metadata

        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        stream_consumer_holder = [None]  # Mutable container for stream consumer

        def run_sync():
            # The conditional re-assignment of `message` further below
            # (prepending model-switch notes) makes Python treat it as a
            # local variable in the entire function.  `nonlocal` lets us
            # read *and* reassign the outer `_run_agent` parameter without
            # triggering an UnboundLocalError during runtime resolution.
            nonlocal message

            # Pass session_key to process registry via env var so background
            # processes can be mapped back to this gateway session
            os.environ["HERMES_SESSION_KEY"] = session_key or ""

            try:
                prepared_runtime = shared_prepare_gateway_sync_turn_runtime(
                    env_path=_env_path,
                    load_dotenv_fn=load_dotenv,
                    resolve_runtime_agent_kwargs_fn=_resolve_runtime_agent_kwargs,
                    load_reasoning_config_fn=self._load_reasoning_config,
                    source=source,
                    user_message=message,
                    context_prompt=context_prompt,
                    gateway_ephemeral_system_prompt=getattr(self, "_ephemeral_system_prompt", ""),
                    provider_routing=getattr(self, "_provider_routing", {}),
                    fallback_model=getattr(self, "_fallback_model", None),
                    smart_model_routing=getattr(self, "_smart_model_routing", {}),
                    user_config=user_config,
                    model=_resolve_gateway_model(user_config),
                    enabled_toolsets=enabled_toolsets,
                )
            except Exception as exc:
                return {
                    "final_response": f"⚠️ Provider authentication failed: {exc}",
                    "messages": [],
                    "api_calls": 0,
                    "tools": [],
                }

            runtime_spec = prepared_runtime.runtime_spec
            reasoning_config = prepared_runtime.reasoning_config
            max_iterations = prepared_runtime.max_iterations
            self._reasoning_config = reasoning_config

            prepared_agent = shared_prepare_gateway_cached_turn_agent(
                runtime_spec=runtime_spec,
                session_key=session_key,
                session_id=session_id,
                source=source,
                progress_runtime=progress_runtime,
                reasoning_config=reasoning_config,
                streaming_config=getattr(getattr(self, "config", None), "streaming", None),
                adapter=self.adapters.get(source.platform),
                thread_metadata={"thread_id": _progress_thread_id} if _progress_thread_id else None,
                stream_consumer_holder=stream_consumer_holder,
                cache=getattr(self, "_agent_cache", None),
                cache_lock=getattr(self, "_agent_cache_lock", None),
                create_agent=lambda: create_gateway_agent(
                    runtime_spec=runtime_spec,
                    session_id=session_id,
                    source=source,
                    session_db=self._session_db,
                    prefill_messages=self._prefill_messages or None,
                    max_iterations=max_iterations,
                    enabled_toolsets=runtime_spec.enabled_toolsets,
                    quiet_mode=True,
                    verbose_logging=False,
                ),
                status_adapter=_status_adapter,
                status_chat_id=_status_chat_id,
                status_thread_metadata=_status_thread_metadata,
                loop_for_step=_loop_for_step,
                logger=logger,
            )
            agent = prepared_agent.agent
            _stream_consumer = prepared_agent.stream_consumer

            # Store agent reference for interrupt support
            agent_holder[0] = agent
            # Capture the full tool definitions for transcript logging
            tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None

            outcome = shared_execute_gateway_sync_turn(
                agent=agent,
                message=message,
                history=history,
                session_id=session_id,
                session_key=session_key,
                admin_user_ids=admin_user_ids,
                is_admin_user=is_admin_user,
                status_adapter=_status_adapter,
                status_chat_id=_status_chat_id,
                status_thread_metadata=_status_thread_metadata,
                loop_for_step=_loop_for_step,
                logger=logger,
                admin_only_message_builder=lambda action: self._admin_only_message(
                    source,
                    action,
                ),
                stream_consumer=_stream_consumer,
                session_store=getattr(self, "session_store", None),
                session_db=self._session_db,
                empty_response_fallback=lambda empty_kind: _empty_response_fallback(
                    source,
                    message,
                    empty_kind=empty_kind,
                    is_admin_user=bool(is_admin_user),
                    raw_message=raw_message,
                    event=event,
                ),
                pending_model_notes=getattr(self, "_pending_model_notes", {}),
            )
            result_holder[0] = outcome.result
            return outcome.final_result
        
        runtime_tasks = shared_start_gateway_agent_runtime_tasks(
            tool_progress_enabled=tool_progress_enabled,
            send_progress_messages=progress_runtime.send_progress_messages,
            stream_consumer_holder=stream_consumer_holder,
            agent_holder=agent_holder,
            running_agents=self._running_agents,
            session_key=session_key,
            adapter=self.adapters.get(source.platform),
            chat_id=source.chat_id,
            notify_metadata=_status_thread_metadata,
            long_running_detail_builder=_build_long_running_status_detail,
            logger=logger,
        )

        try:
            response = await shared_wait_for_gateway_agent_result(
                run_sync=run_sync,
                agent_holder=agent_holder,
                result_holder=result_holder,
                tools_holder=tools_holder,
                session_key=session_key,
                logger=logger,
            )

            effective_model_state = shared_resolve_gateway_effective_model_state(
                agent=agent_holder[0],
                configured_model=_resolve_gateway_model(),
                should_evict_cached_agent_after_turn=self._should_evict_cached_agent_after_turn,
            )
            self._effective_model = effective_model_state.effective_model
            self._effective_provider = effective_model_state.effective_provider
            if effective_model_state.should_evict_cached_agent:
                # Unpinned fallback: next message should try the primary again.
                self._evict_cached_agent(session_key)

            result = result_holder[0]
            adapter = self.adapters.get(source.platform)

            async def _continue_pending_followup(pending_followup, updated_history):
                return await self._run_agent(
                    message=pending_followup.text,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=source,
                    session_id=session_id,
                    session_key=session_key,
                    _interrupt_depth=_interrupt_depth + 1,
                    event_message_id=getattr(pending_followup.event, "message_id", None),
                    event=pending_followup.event,
                    admin_user_ids=admin_user_ids,
                    is_admin_user=is_admin_user,
                    raw_message=getattr(pending_followup.event, "raw_message", None),
                )

            followup_result = await shared_process_gateway_pending_followup(
                result=result,
                adapter=adapter,
                session_key=session_key,
                dequeue_pending_event_text=_dequeue_pending_event_text,
                logger=logger,
                interrupt_depth=_interrupt_depth,
                max_interrupt_depth=self._MAX_INTERRUPT_DEPTH,
                source=source,
                fallback_event=event,
                chat_id=source.chat_id,
                stream_consumer=stream_consumer_holder[0],
                history=history,
                current_response_fallback=result_holder[0] or {
                    "final_response": response,
                    "messages": history,
                },
                recurse_followup=_continue_pending_followup,
            )
            if followup_result is not None:
                return followup_result
        finally:
            await shared_cleanup_gateway_agent_runtime_tasks(
                tasks=runtime_tasks,
                session_key=session_key,
                running_agents=self._running_agents,
                running_agents_ts=self._running_agents_ts,
            )

        return shared_mark_gateway_streaming_delivery_state(
            response=response,
            stream_consumer=stream_consumer_holder[0],
        )


def _start_cron_ticker(stop_event: threading.Event, adapters=None, loop=None, interval: int = 60):
    """
    Background thread that ticks the cron scheduler at a regular interval.
    
    Runs inside the gateway process so cronjobs fire automatically without
    needing a separate `hermes cron daemon` or system cron entry.

    When ``adapters`` and ``loop`` are provided, passes them through to the
    cron delivery path so live adapters can be used for E2EE rooms.

    Also refreshes the channel directory every 5 minutes and prunes the
    image/audio/document cache once per hour.
    """
    from cron.scheduler import tick as cron_tick
    from gateway.platforms.base import cleanup_image_cache, cleanup_document_cache

    IMAGE_CACHE_EVERY = 60   # ticks — once per hour at default 60s interval
    CHANNEL_DIR_EVERY = 5    # ticks — every 5 minutes

    logger.info("Cron ticker started (interval=%ds)", interval)
    tick_count = 0
    while not stop_event.is_set():
        try:
            cron_tick(verbose=False, adapters=adapters, loop=loop)
        except Exception as e:
            logger.debug("Cron tick error: %s", e)

        tick_count += 1

        if tick_count % CHANNEL_DIR_EVERY == 0 and adapters:
            try:
                from gateway.channel_directory import build_channel_directory
                build_channel_directory(adapters)
            except Exception as e:
                logger.debug("Channel directory refresh error: %s", e)

        if tick_count % IMAGE_CACHE_EVERY == 0:
            try:
                removed = cleanup_image_cache(max_age_hours=24)
                if removed:
                    logger.info("Image cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Image cache cleanup error: %s", e)
            try:
                removed = cleanup_document_cache(max_age_hours=24)
                if removed:
                    logger.info("Document cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Document cache cleanup error: %s", e)

        stop_event.wait(timeout=interval)
    logger.info("Cron ticker stopped")


async def start_gateway(config: Optional[GatewayConfig] = None, replace: bool = False, verbosity: Optional[int] = 0) -> bool:
    """
    Start the gateway and run until interrupted.
    
    This is the main entry point for running the gateway.
    Returns True if the gateway ran successfully, False if it failed to start.
    A False return causes a non-zero exit code so systemd can auto-restart.
    
    Args:
        config: Optional gateway configuration override.
        replace: If True, kill any existing gateway instance before starting.
                 Useful for systemd services to avoid restart-loop deadlocks
                 when the previous process hasn't fully exited yet.
    """
    # ── Duplicate-instance guard ──────────────────────────────────────
    # Prevent two gateways from running under the same HERMES_HOME.
    # The PID file is scoped to HERMES_HOME, so future multi-profile
    # setups (each profile using a distinct HERMES_HOME) will naturally
    # allow concurrent instances without tripping this guard.
    import time as _time
    from gateway.status import get_running_pid, remove_pid_file
    existing_pid = get_running_pid()
    if existing_pid is not None and existing_pid != os.getpid():
        if replace:
            logger.info(
                "Replacing existing gateway instance (PID %d) with --replace.",
                existing_pid,
            )
            try:
                os.kill(existing_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # Already gone
            except PermissionError:
                logger.error(
                    "Permission denied killing PID %d. Cannot replace.",
                    existing_pid,
                )
                return False
            # Wait up to 10 seconds for the old process to exit
            for _ in range(20):
                try:
                    os.kill(existing_pid, 0)
                    _time.sleep(0.5)
                except (ProcessLookupError, PermissionError):
                    break  # Process is gone
            else:
                # Still alive after 10s — force kill
                logger.warning(
                    "Old gateway (PID %d) did not exit after SIGTERM, sending SIGKILL.",
                    existing_pid,
                )
                try:
                    os.kill(existing_pid, signal.SIGKILL)
                    _time.sleep(0.5)
                except (ProcessLookupError, PermissionError):
                    pass
            remove_pid_file()
            # Also release all scoped locks left by the old process.
            # Stopped (Ctrl+Z) processes don't release locks on exit,
            # leaving stale lock files that block the new gateway from starting.
            try:
                from gateway.status import release_all_scoped_locks
                _released = release_all_scoped_locks()
                if _released:
                    logger.info("Released %d stale scoped lock(s) from old gateway.", _released)
            except Exception:
                pass
        else:
            hermes_home = str(get_hermes_home())
            logger.error(
                "Another gateway instance is already running (PID %d, HERMES_HOME=%s). "
                "Use 'hermes gateway restart' to replace it, or 'hermes gateway stop' first.",
                existing_pid, hermes_home,
            )
            print(
                f"\n❌ Gateway already running (PID {existing_pid}).\n"
                f"   Use 'hermes gateway restart' to replace it,\n"
                f"   or 'hermes gateway stop' to kill it first.\n"
                f"   Or use 'hermes gateway run --replace' to auto-replace.\n"
            )
            return False

    # Sync bundled skills on gateway start (fast -- skips unchanged)
    try:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)
    except Exception:
        pass

    # Centralized logging — agent.log (INFO+) and errors.log (WARNING+).
    # Idempotent, so repeated calls from AIAgent.__init__ won't duplicate.
    from hermes_logging import setup_logging
    log_dir = setup_logging(hermes_home=_hermes_home, mode="gateway")

    # Gateway-specific rotating log — captures all gateway-level messages
    # (session management, platform adapters, slash commands, etc.).
    from agent.redact import RedactingFormatter
    from hermes_logging import _add_rotating_handler
    _add_rotating_handler(
        logging.getLogger(),
        log_dir / 'gateway.log',
        level=logging.INFO,
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
        formatter=RedactingFormatter('%(asctime)s %(levelname)s %(name)s: %(message)s'),
    )

    # Optional stderr handler — level driven by -v/-q flags on the CLI.
    # verbosity=None (-q/--quiet): no stderr output
    # verbosity=0    (default):    WARNING and above
    # verbosity=1    (-v):         INFO and above
    # verbosity=2+   (-vv/-vvv):   DEBUG
    if verbosity is not None:
        _stderr_level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
        _stderr_handler = logging.StreamHandler()
        _stderr_handler.setLevel(_stderr_level)
        _stderr_handler.setFormatter(RedactingFormatter('%(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(_stderr_handler)
        # Lower root logger level if needed so DEBUG records can reach the handler
        if _stderr_level < logging.getLogger().level:
            logging.getLogger().setLevel(_stderr_level)

    runner = GatewayRunner(config)
    
    # Set up signal handlers
    signal_handler, cancel_force_exit_timer = _make_gateway_signal_handler(
        runner,
        force_exit_after=_load_gateway_signal_force_exit_seconds(),
    )
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    # Start the gateway
    success = await runner.start()
    if not success:
        return False
    if runner.should_exit_cleanly:
        if runner.exit_reason:
            logger.error("Gateway exiting cleanly: %s", runner.exit_reason)
        return True
    
    # Write PID file so CLI can detect gateway is running
    import atexit
    from gateway.status import write_pid_file, remove_pid_file
    write_pid_file()
    atexit.register(remove_pid_file)
    
    # Start background cron ticker so scheduled jobs fire automatically.
    # Pass the event loop so cron delivery can use live adapters (E2EE support).
    cron_stop = threading.Event()
    cron_thread = threading.Thread(
        target=_start_cron_ticker,
        args=(cron_stop,),
        kwargs={"adapters": runner.adapters, "loop": asyncio.get_running_loop()},
        daemon=True,
        name="cron-ticker",
    )
    cron_thread.start()
    
    # Wait for shutdown
    await runner.wait_for_shutdown()
    cancel_force_exit_timer()

    if runner.should_exit_with_failure:
        if runner.exit_reason:
            logger.error("Gateway exiting with failure: %s", runner.exit_reason)
        return False
    
    # Stop cron ticker cleanly
    cron_stop.set()
    cron_thread.join(timeout=5)

    # Close MCP server connections
    try:
        from tools.mcp_tool import shutdown_mcp_servers
        shutdown_mcp_servers()
    except Exception:
        pass

    return True


def main():
    """CLI entry point for the gateway."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        import json
        with open(args.config, encoding="utf-8") as f:
            data = json.load(f)
            config = GatewayConfig.from_dict(data)
    
    # Run the gateway - exit with code 1 if no platforms connected,
    # so systemd Restart=on-failure will retry on transient errors (e.g. DNS)
    success = asyncio.run(start_gateway(config))
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
