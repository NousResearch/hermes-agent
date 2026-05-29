"""Gateway delivery helpers for MCP custom notification passthrough."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from gateway.config import HomeChannel, Platform, load_gateway_config
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

_DEFAULT_RETRY_ATTEMPTS = 3


def _truncate_log_text(text: Any, max_chars: int = 240) -> str:
    """Return a single-line truncated string for log payload previews."""
    rendered = str(text or "").replace("\n", "\\n")
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 3] + "..."


def build_mcp_passthrough_transcript_content(server_name: str, payload_json: str) -> str:
    """Build the assistant transcript entry stored for future replay."""
    return f"[MCP passthrough from {server_name}] {payload_json}"


def _get_live_runner() -> Any | None:
    try:
        from gateway.run import _gateway_runner_ref

        return _gateway_runner_ref()
    except Exception:
        return None


def _resolve_home(platform: Platform, runner: Any | None) -> tuple[Any, Any | None, HomeChannel | None]:
    config = runner.config if runner is not None else load_gateway_config()
    return config, config.platforms.get(platform), config.get_home_channel(platform)


def _resolve_live_runner_target_entries(runner: Any, platform: Platform, home: HomeChannel) -> list[Any]:
    thread_id = str(home.thread_id) if home.thread_id else None
    matches: list[Any] = []
    for entry in runner.session_store.list_sessions():
        origin = getattr(entry, "origin", None)
        entry_platform = getattr(origin, "platform", None) or getattr(entry, "platform", None)
        entry_chat_id = str(getattr(origin, "chat_id", "") or "")
        entry_thread_id = getattr(origin, "thread_id", None)
        if entry_platform != platform:
            continue
        if entry_chat_id != str(home.chat_id):
            continue
        if str(entry_thread_id or "") != str(thread_id or ""):
            continue
        matches.append(entry)
    if not matches:
        matches = [runner.session_store.get_or_create_session(_default_home_source(platform, home))]

    deduped: list[Any] = []
    seen_session_ids: set[str] = set()
    for entry in matches:
        session_id = getattr(entry, "session_id", None)
        if not session_id or session_id in seen_session_ids:
            continue
        seen_session_ids.add(session_id)
        deduped.append(entry)
    return deduped


def _default_home_source(platform: Platform, home: HomeChannel) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=str(home.chat_id),
        chat_name=home.name,
        chat_type="group" if home.thread_id else "dm",
        thread_id=str(home.thread_id) if home.thread_id else None,
    )


def _persist_to_live_runner_sessions(
    runner: Any,
    platform: Platform,
    home: HomeChannel,
    server_name: str,
    payload_json: str,
) -> int:
    message = {
        "role": "assistant",
        "content": build_mcp_passthrough_transcript_content(server_name, payload_json),
    }
    entries = _resolve_live_runner_target_entries(runner, platform, home)

    written = 0
    for entry in entries:
        session_id = getattr(entry, "session_id", None)
        session_key = getattr(entry, "session_key", None)
        if not session_id:
            continue
        runner.session_store.append_to_transcript(session_id, message)
        if session_key:
            runner.session_store.update_session(session_key)
        written += 1
    return written


def _build_memory_manager_for_entry(entry: Any, platform: Platform) -> Any | None:
    try:
        from agent.memory_manager import MemoryManager
        from hermes_cli.config import load_config
        from hermes_constants import get_hermes_home
        from plugins.memory import load_memory_provider

        config = load_config() or {}
        mem_cfg = config.get("memory") or {}
        provider_name = str(mem_cfg.get("provider") or "").strip()
        if not provider_name:
            return None

        provider = load_memory_provider(provider_name)
        if provider is None or not provider.is_available():
            return None

        manager = MemoryManager()
        manager.add_provider(provider)
        if not manager.providers:
            return None

        origin = getattr(entry, "origin", None)
        init_kwargs = {
            "platform": platform.value,
            "hermes_home": str(get_hermes_home()),
            "agent_context": "primary",
        }
        if origin is not None:
            if getattr(origin, "user_id", None):
                init_kwargs["user_id"] = origin.user_id
            if getattr(origin, "user_id_alt", None):
                init_kwargs["user_id_alt"] = origin.user_id_alt
            if getattr(origin, "user_name", None):
                init_kwargs["user_name"] = origin.user_name
            if getattr(origin, "chat_id", None):
                init_kwargs["chat_id"] = origin.chat_id
            if getattr(origin, "chat_name", None):
                init_kwargs["chat_name"] = origin.chat_name
            if getattr(origin, "chat_type", None):
                init_kwargs["chat_type"] = origin.chat_type
            if getattr(origin, "thread_id", None):
                init_kwargs["thread_id"] = origin.thread_id
        session_key = getattr(entry, "session_key", None)
        if session_key:
            init_kwargs["gateway_session_key"] = session_key
        try:
            from hermes_cli.profiles import get_active_profile_name

            init_kwargs["agent_identity"] = get_active_profile_name()
            init_kwargs["agent_workspace"] = "hermes"
        except Exception:
            pass

        session_id = str(getattr(entry, "session_id", "") or "").strip()
        if not session_id:
            return None
        manager.initialize_all(session_id=session_id, **init_kwargs)
        return manager
    except Exception:
        logger.exception(
            "MCP passthrough: failed to initialize memory provider for %s",
            platform.value,
        )
        return None


def _sync_entries_to_memory(entries: list[Any], platform: Platform, server_name: str, payload_json: str) -> int:
    synced = 0
    for entry in entries:
        session_id = str(getattr(entry, "session_id", "") or "").strip()
        if not session_id:
            continue
        manager = _build_memory_manager_for_entry(entry, platform)
        if manager is None:
            continue
        try:
            manager.sync_passive_event_all(
                payload_json,
                session_id=session_id,
                source_label=f"mcp:{server_name}",
                metadata={
                    "kind": "mcp_passthrough",
                    "server_name": server_name,
                    "platform": platform.value,
                },
            )
            synced += 1
        finally:
            manager.shutdown_all()
    return synced


def _persist_without_live_runner(
    platform: Platform,
    home: HomeChannel,
    server_name: str,
    payload_json: str,
) -> int:
    from gateway.mirror import mirror_to_session

    mirrored = mirror_to_session(
        platform=platform.value,
        chat_id=str(home.chat_id),
        message_text=build_mcp_passthrough_transcript_content(server_name, payload_json),
        source_label=f"mcp:{server_name}",
        thread_id=str(home.thread_id) if home.thread_id else None,
    )
    return 1 if mirrored else 0


async def _deliver_once(
    platform: Platform,
    pconfig: Any | None,
    home: HomeChannel,
    payload_json: str,
    runner: Any | None,
) -> tuple[bool, str | None]:
    thread_id = str(home.thread_id) if home.thread_id else None
    if runner is not None:
        adapter = runner.adapters.get(platform)
        if adapter is not None:
            try:
                metadata = {"thread_id": thread_id} if thread_id else None
                result = await adapter.send(
                    chat_id=str(home.chat_id),
                    content=payload_json,
                    metadata=metadata,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return False, str(exc)
            if result.success:
                return True, None
            return False, result.error or "adapter send failed"

    if pconfig is None:
        return False, "platform is not configured in gateway config"

    from tools.send_message_tool import _send_to_platform

    result = await _send_to_platform(
        platform,
        pconfig,
        str(home.chat_id),
        payload_json,
        thread_id=thread_id,
        media_files=[],
        force_document=False,
    )
    if result.get("success"):
        return True, None
    return False, result.get("error") or "delivery failed"


async def forward_mcp_passthrough_notification(
    *,
    server_name: str,
    payload_json: str,
    targets: list[str],
    retry_attempts: int = _DEFAULT_RETRY_ATTEMPTS,
) -> None:
    """Forward an MCP custom notification to configured gateway home channels."""
    if not targets:
        return

    runner = _get_live_runner()
    logger.info(
        "MCP server '%s': dispatching passthrough payload to targets=%s live_runner=%s payload_bytes=%d payload_preview=%s",
        server_name,
        ",".join(targets),
        bool(runner),
        len(payload_json.encode("utf-8")),
        _truncate_log_text(payload_json),
    )
    for target in targets:
        try:
            platform = Platform(target)
        except Exception:
            logger.warning(
                "MCP server '%s': unknown passthrough target '%s'",
                server_name,
                target,
            )
            continue

        _config, pconfig, home = _resolve_home(platform, runner)
        if home is None or not home.chat_id:
            logger.warning(
                "MCP server '%s': passthrough target '%s' has no home channel configured",
                server_name,
                platform.value,
            )
            continue

        adapter = runner.adapters.get(platform) if runner is not None else None
        delivery_path = "live_adapter" if adapter is not None else "send_message_tool"
        logger.info(
            "MCP server '%s': resolved passthrough target=%s chat_id=%s thread_id=%s path=%s",
            server_name,
            platform.value,
            home.chat_id,
            home.thread_id or "-",
            delivery_path,
        )

        delivered = False
        last_error: str | None = None
        attempts = max(1, retry_attempts)
        for attempt in range(1, attempts + 1):
            logger.info(
                "MCP server '%s': passthrough delivery attempt %d/%d to %s",
                server_name,
                attempt,
                attempts,
                platform.value,
            )
            ok, error = await _deliver_once(platform, pconfig, home, payload_json, runner)
            if ok:
                delivered = True
                logger.info(
                    "MCP server '%s': passthrough delivered to %s on attempt %d/%d",
                    server_name,
                    platform.value,
                    attempt,
                    attempts,
                )
                break
            last_error = error or "delivery failed"
            if attempt < attempts:
                logger.warning(
                    "MCP server '%s': passthrough to %s failed (attempt %d/%d), retrying: %s",
                    server_name,
                    platform.value,
                    attempt,
                    attempts,
                    last_error,
                )
                await asyncio.sleep(min(2 ** (attempt - 1), 2.0))

        if not delivered:
            logger.warning(
                "MCP server '%s': passthrough to %s failed after %d attempts: %s",
                server_name,
                platform.value,
                attempts,
                last_error or "delivery failed",
            )
            continue

        memory_synced = 0
        if runner is not None:
            memory_entries = _resolve_live_runner_target_entries(runner, platform, home)
            persisted = _persist_to_live_runner_sessions(
                runner,
                platform,
                home,
                server_name,
                payload_json,
            )
            memory_synced = _sync_entries_to_memory(
                memory_entries,
                platform,
                server_name,
                payload_json,
            )
        else:
            persisted = _persist_without_live_runner(
                platform,
                home,
                server_name,
                payload_json,
            )
        logger.info(
            "MCP server '%s': passthrough target=%s persisted=%d memory_synced=%d",
            server_name,
            platform.value,
            persisted,
            memory_synced,
        )
        if persisted == 0:
            logger.warning(
                "MCP server '%s': delivered passthrough to %s but no matching session transcript was updated",
                server_name,
                platform.value,
            )
        if runner is not None and memory_synced == 0:
            logger.warning(
                "MCP server '%s': delivered passthrough to %s but no memory provider session was synced",
                server_name,
                platform.value,
            )