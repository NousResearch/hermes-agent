"""
Session management utilities for the Gateway.

Provides standalone functions for session lifecycle management including:
- Session key resolution
- Agent runtime resolution
- Pending event queue management
- Active session drain/interrupt/shutdown
- Busy message handling
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple

from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key

logger = logging.getLogger(__name__)

# Sentinel placed into _running_agents immediately when a session starts
# processing, *before* any await.  Prevents a second message for the same
# session from bypassing the "already running" guard during the async gap
# between the guard check and actual agent creation.
AGENT_PENDING_SENTINEL = object()


def session_key_for_source(
    source: SessionSource,
    session_store=None,
    config=None,
) -> str:
    """Resolve the current session key for a source, honoring gateway config when available."""
    if session_store is not None:
        try:
            session_key = session_store._generate_session_key(source)
            if isinstance(session_key, str) and session_key:
                return session_key
        except Exception:
            pass
    return build_session_key(
        source,
        group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
        thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
    )


def resolve_session_agent_runtime(
    *,
    source: Optional[SessionSource] = None,
    session_key: Optional[str] = None,
    user_config: Optional[dict] = None,
    session_model_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    resolve_gateway_model_fn=None,
    resolve_runtime_agent_kwargs_fn=None,
    apply_session_model_override_fn=None,
) -> Tuple[str, dict]:
    """Resolve model/runtime for a session, honoring session-scoped /model overrides.

    If the session override already contains a complete provider bundle
    (provider/api_key/base_url/api_mode), prefer it directly instead of
    resolving fresh global runtime state first.
    """
    resolved_session_key = session_key
    if not resolved_session_key and source is not None:
        try:
            resolved_session_key = session_key_for_source(source)
        except Exception:
            resolved_session_key = None

    model = resolve_gateway_model_fn(user_config) if resolve_gateway_model_fn else _resolve_gateway_model_fallback(user_config)
    overrides = session_model_overrides.get(resolved_session_key) if resolved_session_key and session_model_overrides else None
    if overrides:
        override_model = overrides.get("model", model)
        override_runtime = {
            "provider": overrides.get("provider"),
            "api_key": overrides.get("api_key"),
            "base_url": overrides.get("base_url"),
            "api_mode": overrides.get("api_mode"),
        }
        if override_runtime.get("api_key"):
            logger.debug(
                "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                (resolved_session_key or "")[:30], model, override_model,
                override_runtime.get("provider"),
            )
            return override_model, override_runtime
        logger.debug(
            "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
            (resolved_session_key or "")[:30], model, override_model,
        )
    else:
        logger.debug(
            "No session model override: session=%s config_model=%s override_keys=%s",
            (resolved_session_key or "")[:30], model,
            list(session_model_overrides.keys())[:5] if session_model_overrides else "[]",
        )

    runtime_kwargs = resolve_runtime_agent_kwargs_fn() if resolve_runtime_agent_kwargs_fn else _resolve_runtime_agent_kwargs_fallback()
    if overrides and resolved_session_key and apply_session_model_override_fn:
        model, runtime_kwargs = apply_session_model_override_fn(
            resolved_session_key, model, runtime_kwargs
        )

    # When the config has no model.default but a provider was resolved
    # (e.g. user ran `hermes auth add openai-codex` without `hermes model`),
    # fall back to the provider's first catalog model so the API call
    # doesn't fail with "model must be a non-empty string".
    if not model and runtime_kwargs.get("provider"):
        try:
            from hermes_cli.models import get_default_model_for_provider
            model = get_default_model_for_provider(runtime_kwargs["provider"])
            if model:
                logger.info(
                    "No model configured — defaulting to %s for provider %s",
                    model, runtime_kwargs["provider"],
                )
        except Exception:
            pass

    return model, runtime_kwargs


def _resolve_gateway_model_fallback(config=None) -> str:
    """Fallback resolver for gateway model."""
    from pathlib import Path
    _hermes_home = Path(__file__).resolve().parent.parent / "hermes_home"
    try:
        from hermes_cli.env_loader import get_hermes_home
        _hermes_home = get_hermes_home()
    except Exception:
        pass

    try:
        config_path = _hermes_home / 'config.yaml'
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = config or {}
    except Exception:
        cfg = config or {}

    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""


def _resolve_runtime_agent_kwargs_fallback() -> dict:
    """Fallback resolver for runtime agent kwargs."""
    import os
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        format_runtime_provider_error,
    )
    try:
        runtime = resolve_runtime_provider(
            requested=os.getenv("HERMES_INFERENCE_PROVIDER"),
        )
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
    }


def queue_or_replace_pending_event(
    adapters: Dict,
    session_key: str,
    event: MessageEvent,
) -> None:
    """Queue or replace a pending event for a session."""
    adapter = adapters.get(event.source.platform)
    if not adapter:
        return
    from gateway.platforms.base import merge_pending_message_event
    merge_pending_message_event(adapter._pending_messages, session_key, event)


async def handle_active_session_busy_message(
    event: MessageEvent,
    session_key: str,
    adapters: Dict,
    running_agents: Dict[str, Any],
    running_agents_ts: Dict[str, float],
    busy_ack_ts: Dict[str, float],
    draining: bool,
    restart_requested: bool,
    busy_input_mode: str,
    agent_pending_sentinel=AGENT_PENDING_SENTINEL,
) -> bool:
    """Handle a message arriving while the session's agent is busy.

    Returns True if the message was handled (interrupted/queued), False otherwise.
    """
    # --- Draining case (gateway restarting/stopping) ---
    if draining:
        adapter = adapters.get(event.source.platform)
        if not adapter:
            return True

        thread_meta = {"thread_id": event.source.thread_id} if event.source.thread_id else None
        if restart_requested and busy_input_mode == "queue":
            queue_or_replace_pending_event(adapters, session_key, event)
            message = f"⏳ Gateway restarting — queued for the next turn after it comes back."
        else:
            message = f"⏳ Gateway is shutting down and is not accepting another turn right now."

        await adapter._send_with_retry(
            chat_id=event.source.chat_id,
            content=message,
            reply_to=event.message_id,
            metadata=thread_meta,
        )
        return True

    # --- Normal busy case (agent actively running a task) ---
    adapter = adapters.get(event.source.platform)
    if not adapter:
        return False  # let default path handle it

    # Store the message so it's processed as the next turn after the
    # interrupt causes the current run to exit.
    from gateway.platforms.base import merge_pending_message_event
    merge_pending_message_event(adapter._pending_messages, session_key, event)

    # Interrupt the running agent
    running_agent = running_agents.get(session_key)
    if running_agent and running_agent is not agent_pending_sentinel:
        try:
            running_agent.interrupt(event.text)
        except Exception:
            pass  # don't let interrupt failure block the ack

    # Debounce: only send an acknowledgment once every 30 seconds per session
    _BUSY_ACK_COOLDOWN = 30
    now = time.time()
    last_ack = busy_ack_ts.get(session_key, 0)
    if now - last_ack < _BUSY_ACK_COOLDOWN:
        return True  # interrupt sent, ack already delivered recently

    busy_ack_ts[session_key] = now

    # Build a status-rich acknowledgment
    status_parts = []
    if running_agent and running_agent is not agent_pending_sentinel:
        try:
            summary = running_agent.get_activity_summary()
            iteration = summary.get("api_call_count", 0)
            max_iter = summary.get("max_iterations", 0)
            current_tool = summary.get("current_tool")
            start_ts = running_agents_ts.get(session_key, 0)
            if start_ts:
                elapsed_min = int((now - start_ts) / 60)
                if elapsed_min > 0:
                    status_parts.append(f"{elapsed_min} min elapsed")
            if max_iter:
                status_parts.append(f"iteration {iteration}/{max_iter}")
            if current_tool:
                status_parts.append(f"running: {current_tool}")
        except Exception:
            pass

    status_detail = f" ({', '.join(status_parts)})" if status_parts else ""
    message = (
        f"⚡ Interrupting current task{status_detail}. "
        f"I'll respond to your message shortly."
    )

    thread_meta = {"thread_id": event.source.thread_id} if event.source.thread_id else None
    try:
        await adapter._send_with_retry(
            chat_id=event.source.chat_id,
            content=message,
            reply_to=event.message_id,
            metadata=thread_meta,
        )
    except Exception as e:
        logger.debug("Failed to send busy-ack: %s", e)

    return True


async def drain_active_agents(
    running_agents: Dict[str, Any],
    snapshot_running_agents_fn,
    running_agent_count_fn,
    update_runtime_status_fn,
    timeout: float,
) -> Tuple[Dict[str, Any], bool]:
    """Drain all active agents, waiting for them to complete up to the given timeout.

    Returns a tuple of (snapshot of agents, timed_out boolean).
    """
    snapshot = snapshot_running_agents_fn()
    last_active_count = running_agent_count_fn()
    last_status_at = 0.0

    def _maybe_update_status(force: bool = False) -> None:
        nonlocal last_active_count, last_status_at
        now = asyncio.get_running_loop().time()
        active_count = running_agent_count_fn()
        if force or active_count != last_active_count or (now - last_status_at) >= 1.0:
            update_runtime_status_fn("draining")
            last_active_count = active_count
            last_status_at = now

    if not running_agents:
        _maybe_update_status(force=True)
        return snapshot, False

    _maybe_update_status(force=True)
    if timeout <= 0:
        return snapshot, True

    deadline = asyncio.get_running_loop().time() + timeout
    while running_agents and asyncio.get_running_loop().time() < deadline:
        _maybe_update_status()
        await asyncio.sleep(0.1)
    timed_out = bool(running_agents)
    _maybe_update_status(force=True)
    return snapshot, timed_out


def interrupt_running_agents(
    running_agents: Dict[str, Any],
    reason: str,
    agent_pending_sentinel=AGENT_PENDING_SENTINEL,
) -> None:
    """Interrupt all running agents with the given reason."""
    for session_key, agent in list(running_agents.items()):
        if agent is agent_pending_sentinel:
            continue
        try:
            agent.interrupt(reason)
            logger.debug("Interrupted running agent for session %s during shutdown", session_key[:20])
        except Exception as e:
            logger.debug("Failed interrupting agent during shutdown: %s", e)


async def notify_active_sessions_of_shutdown(
    snapshot_running_agents_fn,
    adapters: Dict,
    restart_requested: bool,
    parse_session_key_fn,
) -> None:
    """Send a notification to every chat with an active agent.

    Called at the very start of stop() — adapters are still connected so
    messages can be delivered.  Best-effort: individual send failures are
    logged and swallowed so they never block the shutdown sequence.
    """
    from gateway.config import Platform

    active = snapshot_running_agents_fn()
    if not active:
        return

    action = "restarting" if restart_requested else "shutting down"
    hint = (
        "Your current task will be interrupted. "
        "Send any message after restart to resume where it left off."
        if restart_requested
        else "Your current task will be interrupted."
    )
    msg = f"⚠️ Gateway {action} — {hint}"

    notified: set = set()
    for session_key in active:
        # Parse platform + chat_id from the session key.
        _parsed = parse_session_key_fn(session_key)
        if not _parsed:
            continue
        platform_str = _parsed["platform"]
        chat_id = _parsed["chat_id"]

        # Deduplicate: one notification per chat, even if multiple
        # sessions (different users/threads) share the same chat.
        dedup_key = (platform_str, chat_id)
        if dedup_key in notified:
            continue

        try:
            platform = Platform(platform_str)
            adapter = adapters.get(platform)
            if not adapter:
                continue

            # Include thread_id if present so the message lands in the
            # correct forum topic / thread.
            thread_id = _parsed.get("thread_id")
            metadata = {"thread_id": thread_id} if thread_id else None

            await adapter.send(chat_id, msg, metadata=metadata)
            notified.add(dedup_key)
            logger.info(
                "Sent shutdown notification to %s:%s",
                platform_str, chat_id,
            )
        except Exception as e:
            logger.debug(
                "Failed to send shutdown notification to %s:%s: %s",
                platform_str, chat_id, e,
            )
