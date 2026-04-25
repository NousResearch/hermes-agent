"""
Display and formatting utilities for the Gateway.

Provides standalone functions for formatting and notification including:
- Session info formatting
- Gateway process notification formatting
- Update notification sending
- Restart notification sending
- Media placeholder building
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from gateway.platforms.base import MessageEvent, MessageType

logger = logging.getLogger(__name__)

# --- Standalone formatting helpers -------------------------------------------


def build_media_placeholder(event) -> str:
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


def format_gateway_process_notification(evt: dict) -> Optional[str]:
    """Format a watch pattern event from completion_queue into a [SYSTEM:] message."""
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")

    if evt_type == "watch_disabled":
        return f"[SYSTEM: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _pat = evt.get("pattern", "?")
        _out = evt.get("output", "")
        _sup = evt.get("suppressed", 0)
        text = (
            f"[SYSTEM: Background process {_sid} matched "
            f"watch pattern \"{_pat}\".\n"
            f"Command: {_cmd}\n"
            f"Matched output:\n{_out}"
        )
        if _sup:
            text += f"\n({_sup} earlier matches were suppressed by rate limit)"
        text += "]"
        return text

    return None


def format_session_info(
    resolve_gateway_model_fn=None,
    resolve_runtime_agent_kwargs_fn=None,
) -> str:
    """Resolve current model config and return a formatted info block.

    Surfaces model, provider, context length, and endpoint so gateway
    users can immediately see if context detection went wrong (e.g.
    local models falling to the 128K default).
    """
    from agent.model_metadata import get_model_context_length, DEFAULT_FALLBACK_CONTEXT

    # Try to get hermes_home
    try:
        from hermes_cli.env_loader import get_hermes_home
        _hermes_home = get_hermes_home()
    except Exception:
        _hermes_home = Path.home() / ".hermes"

    model = resolve_gateway_model_fn() if resolve_gateway_model_fn else _default_resolve_gateway_model()
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
        runtime = resolve_runtime_agent_kwargs_fn() if resolve_runtime_agent_kwargs_fn else _default_resolve_runtime_agent_kwargs()
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


def _default_resolve_gateway_model() -> str:
    """Default model resolution for use when no override is available."""
    try:
        from hermes_cli.env_loader import get_hermes_home
        _hermes_home = get_hermes_home()
    except Exception:
        _hermes_home = Path.home() / ".hermes"

    try:
        config_path = _hermes_home / 'config.yaml'
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
    except Exception:
        cfg = {}

    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""


def _default_resolve_runtime_agent_kwargs() -> dict:
    """Default runtime kwargs resolution."""
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


# --- Notification helpers -----------------------------------------------------


async def send_update_notification(adapters: dict, platform_config: dict) -> bool:
    """If an update finished, notify the user.

    Returns False when the update is still running so a caller can retry
    later. Returns True after a definitive send/skip decision.

    This is the legacy notification path used when the streaming watcher
    cannot resolve the adapter (e.g. after a gateway restart where the
    platform hasn't reconnected yet).
    """
    from gateway.config import Platform

    # Try to get hermes_home
    try:
        from hermes_cli.env_loader import get_hermes_home
        _hermes_home = get_hermes_home()
    except Exception:
        _hermes_home = Path.home() / ".hermes"

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
        adapter = adapters.get(platform)

        if adapter and chat_id:
            # Strip ANSI escape codes for clean display
            output = re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
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


async def send_restart_notification(adapters: dict) -> None:
    """Notify the chat that initiated /restart that the gateway is back."""
    from gateway.config import Platform

    # Try to get hermes_home
    try:
        from hermes_cli.env_loader import get_hermes_home
        _hermes_home = get_hermes_home()
    except Exception:
        _hermes_home = Path.home() / ".hermes"

    notify_path = _hermes_home / ".restart_notify.json"
    if not notify_path.exists():
        return

    try:
        data = json.loads(notify_path.read_text())
        platform_str = data.get("platform")
        chat_id = data.get("chat_id")
        thread_id = data.get("thread_id")

        if not platform_str or not chat_id:
            return

        platform = Platform(platform_str)
        adapter = adapters.get(platform)
        if not adapter:
            logger.debug(
                "Restart notification skipped: %s adapter not connected",
                platform_str,
            )
            return

        metadata = {"thread_id": thread_id} if thread_id else None
        await adapter.send(
            chat_id,
            "♻ Gateway restarted successfully. Your session continues.",
            metadata=metadata,
        )
        logger.info(
            "Sent restart notification to %s:%s",
            platform_str,
            chat_id,
        )
    except Exception as e:
        logger.warning("Restart notification failed: %s", e)
    finally:
        notify_path.unlink(missing_ok=True)
