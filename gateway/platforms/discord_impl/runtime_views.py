"""Discord-native runtime status, help, command, and identity renderers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from typing import Any, Optional

from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_context, build_session_context_prompt
from hermes_cli.commands import (
    CommandDef,
    format_gateway_command_signature,
    gateway_command_defs,
    gateway_commands_by_category,
)
from hermes_cli.models import provider_label


@dataclass(frozen=True)
class DiscordStatusSnapshot:
    session_id: str
    session_key: str
    created_at: datetime
    updated_at: datetime
    source: SessionSource
    target_source: SessionSource
    configured_model: str
    configured_provider: str
    active_model: str
    active_provider: str
    is_fallback: bool
    runtime_provider: str
    api_mode: str
    base_url: str
    credentials_configured: bool
    credential_source: Optional[str]
    transport_command: Optional[str]
    runtime_error: Optional[str]
    context_length: Optional[int]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_tokens: int
    last_prompt_tokens: int
    estimated_cost_usd: float
    cost_status: str
    is_running: bool
    has_pending_message: bool
    has_pending_approval: bool
    approval_command_preview: Optional[str]
    has_background_process: bool
    voice_mode: str
    send_policy: str
    dock_target: Optional[str]
    activation_mode: Optional[str]
    focused_thread_summary: Optional[str]
    connected_platforms: tuple[str, ...]


@dataclass(frozen=True)
class DiscordWhoamiSnapshot:
    source: SessionSource
    target_source: SessionSource
    session_key: str


@dataclass(frozen=True)
class DiscordContextSnapshot:
    session_id: str
    session_key: str
    source: SessionSource
    target_source: SessionSource
    message_count: int
    role_counts: dict[str, int]
    transcript_tokens: int
    context_prompt_tokens: int
    context_prompt_chars: int
    current_model: str
    current_provider: str
    connected_platforms: tuple[str, ...]
    context_files: tuple[str, ...]
    skill_commands: tuple[str, ...]
    has_soul: bool
    context_prompt: str


def _format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def _short_id(value: str, limit: int = 12) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _yes_no(value: bool, yes: str = "Yes", no: str = "No") -> str:
    return yes if value else no


def _alias_note(cmd: CommandDef) -> str:
    visible_aliases = [
        f"`/{alias}`"
        for alias in cmd.aliases
        if alias.replace("-", "_") != cmd.name.replace("-", "_")
    ]
    if not visible_aliases:
        return ""
    return f" (alias: {', '.join(visible_aliases)})"


def _command_lookup() -> dict[str, CommandDef]:
    return {cmd.name: cmd for cmd in gateway_command_defs()}


def _load_skill_commands() -> dict[str, str]:
    try:
        from agent.skill_commands import get_skill_commands

        skill_cmds = get_skill_commands()
    except Exception:
        return {}

    result: dict[str, str] = {}
    for name, payload in sorted(skill_cmds.items()):
        result[name] = str(payload.get("description") or "Skill command")
    return result


def _resolve_target_source(event: MessageEvent, session_source: SessionSource) -> SessionSource:
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        target = metadata.get("command_target_source")
        if isinstance(target, SessionSource):
            return target
    return event.source if isinstance(event.source, SessionSource) else session_source


def _resolve_context_length(active_model: str, runtime: dict[str, Any]) -> Optional[int]:
    if not active_model:
        return None
    try:
        from agent.model_metadata import get_model_context_length

        return get_model_context_length(
            active_model,
            base_url=str(runtime.get("base_url") or ""),
            api_key=str(runtime.get("api_key") or ""),
        )
    except Exception:
        return None


def collect_discord_status_snapshot(runner: Any, event: MessageEvent) -> DiscordStatusSnapshot:
    """Collect Discord-native status data without rendering concerns."""
    session_source = runner._session_source_for_event(event)
    target_source = _resolve_target_source(event, session_source)
    session_entry = runner.session_store.get_or_create_session(session_source)
    session_key = runner._session_key_for_source(session_source)
    state = runner._load_current_model_selection()
    configured_model = str(state.get("current_model") or "")
    configured_provider = str(state.get("current_provider") or "openrouter")
    active_model = str(getattr(runner, "_effective_model", None) or configured_model)
    active_provider = str(getattr(runner, "_effective_provider", None) or configured_provider)
    is_fallback = bool(getattr(runner, "_effective_model", None))

    runtime_requested = active_provider or configured_provider
    runtime, runtime_error = runner._resolve_model_runtime_details(runtime_requested)
    runtime_provider = str(runtime.get("provider") or runtime_requested)
    api_mode = str(runtime.get("api_mode") or "unknown")
    base_url = str(runtime.get("base_url") or "unknown")
    transport_command = str(runtime.get("command") or "").strip() or None
    credential_source = str(runtime.get("source") or "").strip() or None
    credentials_configured = bool(str(runtime.get("api_key") or "").strip() or transport_command)
    context_length = _resolve_context_length(active_model, runtime)

    try:
        from tools.process_registry import process_registry

        has_background_process = process_registry.has_active_for_session(session_key)
    except Exception:
        has_background_process = False

    pending_approval = getattr(runner, "_pending_approvals", {}).get(session_key)
    approval_command_preview = None
    if isinstance(pending_approval, dict):
        approval_command = str(pending_approval.get("command") or "").strip()
        if approval_command:
            approval_command_preview = approval_command[:80]
            if len(approval_command) > 80:
                approval_command_preview += "..."

    connected_platforms = tuple(sorted(platform.value for platform in getattr(runner, "adapters", {}).keys()))
    send_policy = "inherit"
    if hasattr(runner, "_session_send_policy"):
        try:
            send_policy = str(runner._session_send_policy(session_key) or "inherit")
        except Exception:
            send_policy = "inherit"

    dock_target = None
    if hasattr(runner, "_dock_target_for_session"):
        try:
            dock_target_state = runner._dock_target_for_session(session_key)
        except Exception:
            dock_target_state = None
        if dock_target_state and hasattr(runner, "_format_dock_target"):
            try:
                dock_target = runner._format_dock_target(dock_target_state)
            except Exception:
                dock_target = None

    activation_mode = None
    focused_thread_summary = None
    if target_source.platform.value == "discord":
        adapter = getattr(runner, "adapters", {}).get(target_source.platform)
        if adapter is not None and target_source.chat_type != "dm":
            if hasattr(adapter, "get_activation_mode"):
                try:
                    activation_mode = adapter.get_activation_mode(target_source.chat_id)
                except Exception:
                    activation_mode = None
            if activation_mode is None and hasattr(adapter, "_get_discord_policy"):
                try:
                    policy = adapter._get_discord_policy()
                    activation_mode = "mention" if getattr(policy, "require_mention", True) else "always"
                except Exception:
                    activation_mode = None
        if adapter is not None and target_source.thread_id and hasattr(adapter, "get_thread_binding"):
            try:
                binding = adapter.get_thread_binding(target_source.thread_id)
            except Exception:
                binding = None
            if binding is not None:
                idle = getattr(binding, "idle_timeout_minutes", 0) or 0
                max_age = getattr(binding, "max_age_minutes", 0) or 0
                focused_thread_summary = (
                    f"yes ({getattr(binding, 'chat_name', '') or target_source.thread_id}; "
                    f"idle {idle}m; max-age {max_age}m)"
                )

    return DiscordStatusSnapshot(
        session_id=session_entry.session_id,
        session_key=session_key,
        created_at=session_entry.created_at,
        updated_at=session_entry.updated_at,
        source=session_source,
        target_source=target_source,
        configured_model=configured_model,
        configured_provider=configured_provider,
        active_model=active_model,
        active_provider=active_provider,
        is_fallback=is_fallback,
        runtime_provider=runtime_provider,
        api_mode=api_mode,
        base_url=base_url,
        credentials_configured=credentials_configured,
        credential_source=credential_source,
        transport_command=transport_command,
        runtime_error=runtime_error,
        context_length=context_length,
        input_tokens=session_entry.input_tokens,
        output_tokens=session_entry.output_tokens,
        cache_read_tokens=session_entry.cache_read_tokens,
        cache_write_tokens=session_entry.cache_write_tokens,
        total_tokens=session_entry.total_tokens,
        last_prompt_tokens=session_entry.last_prompt_tokens,
        estimated_cost_usd=session_entry.estimated_cost_usd,
        cost_status=session_entry.cost_status,
        is_running=session_key in getattr(runner, "_running_agents", {}),
        has_pending_message=bool(getattr(runner, "_pending_messages", {}).get(session_key)),
        has_pending_approval=bool(pending_approval),
        approval_command_preview=approval_command_preview,
        has_background_process=has_background_process,
        voice_mode=getattr(runner, "_voice_mode", {}).get(target_source.chat_id, "off"),
        send_policy=send_policy,
        dock_target=dock_target,
        activation_mode=activation_mode,
        focused_thread_summary=focused_thread_summary,
        connected_platforms=connected_platforms,
    )


def _estimate_tokens(messages: list[dict[str, str]]) -> int:
    try:
        from agent.model_metadata import estimate_messages_tokens_rough

        return int(estimate_messages_tokens_rough(messages) or 0)
    except Exception:
        return 0


def _extract_context_file_names(context_prompt: str) -> tuple[str, ...]:
    names: list[str] = []
    for line in context_prompt.splitlines():
        if line.startswith("## "):
            names.append(line[3:].strip())
    return tuple(names)


def collect_discord_context_snapshot(runner: Any, event: MessageEvent) -> DiscordContextSnapshot:
    """Collect a Discord-native snapshot of the current transcript and prompt context."""
    target_source = runner._command_target_source_for_event(event)
    session_entry = runner.session_store.get_or_create_session(target_source)
    history = runner.session_store.load_transcript(session_entry.session_id)
    role_counts: dict[str, int] = {}
    transcript_messages: list[dict[str, str]] = []
    for item in history:
        role = str(item.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        content = str(item.get("content") or "")
        if content:
            transcript_messages.append({"role": role, "content": content})

    current_state = runner._load_current_model_selection()
    current_model = str(current_state.get("current_model") or "")
    current_provider = str(current_state.get("current_provider") or "")
    connected_platforms = tuple(sorted(platform.value for platform in getattr(runner, "adapters", {}).keys()))

    cwd = os.environ.get("TERMINAL_CWD") or os.environ.get("MESSAGING_CWD") or os.getcwd()
    try:
        from agent.prompt_builder import (
            build_context_files_prompt,
            build_skills_system_prompt,
            load_soul_md,
        )

        soul = load_soul_md() or ""
        context_files_prompt = build_context_files_prompt(cwd, skip_soul=bool(soul))
        skills_prompt = build_skills_system_prompt()
    except Exception:
        soul = ""
        context_files_prompt = ""
        skills_prompt = ""

    session_context = build_session_context(target_source, runner.config, session_entry)
    session_context_prompt = build_session_context_prompt(session_context)
    prompt_parts = [part for part in (soul, skills_prompt, context_files_prompt, session_context_prompt) if part]
    context_prompt = "\n\n".join(prompt_parts)
    skill_commands = tuple(sorted(_load_skill_commands().keys()))

    return DiscordContextSnapshot(
        session_id=session_entry.session_id,
        session_key=session_entry.session_key,
        source=runner._session_source_for_event(event),
        target_source=target_source,
        message_count=len(history),
        role_counts=role_counts,
        transcript_tokens=_estimate_tokens(transcript_messages),
        context_prompt_tokens=_estimate_tokens([{"role": "system", "content": context_prompt}]) if context_prompt else 0,
        context_prompt_chars=len(context_prompt),
        current_model=current_model,
        current_provider=current_provider,
        connected_platforms=connected_platforms,
        context_files=_extract_context_file_names(context_files_prompt),
        skill_commands=skill_commands,
        has_soul=bool(soul),
        context_prompt=context_prompt,
    )


def render_discord_status(snapshot: DiscordStatusSnapshot) -> str:
    """Render Discord status output in a multi-section, chat-readable format."""
    lines = [
        "📊 **Hermes Status**",
        "",
        "**Session**",
        f"• Session ID: `{_short_id(snapshot.session_id)}`",
        f"• Session Key: `{_short_id(snapshot.session_key, limit=18)}`",
        f"• Created: {_format_timestamp(snapshot.created_at)}",
        f"• Last Activity: {_format_timestamp(snapshot.updated_at)}",
        f"• Chat: {snapshot.target_source.chat_name or snapshot.target_source.chat_id}",
        f"• Chat Type: `{snapshot.target_source.chat_type}`",
    ]
    if snapshot.target_source.thread_id:
        lines.append(f"• Thread ID: `{snapshot.target_source.thread_id}`")
    if snapshot.target_source.chat_topic:
        lines.append(f"• Chat Topic: {snapshot.target_source.chat_topic}")
    if snapshot.source.session_namespace:
        lines.append(f"• Session Namespace: `{snapshot.source.session_namespace}`")

    lines.extend(
        [
            "",
            "**Model**",
            f"• Configured Model: `{snapshot.configured_model}`",
            f"• Configured Provider: {provider_label(snapshot.configured_provider)} (`{snapshot.configured_provider}`)",
            f"• Active Model: `{snapshot.active_model}`{' (fallback)' if snapshot.is_fallback else ''}",
            f"• Active Provider: {provider_label(snapshot.active_provider)} (`{snapshot.active_provider}`)",
        ]
    )
    if snapshot.runtime_error:
        lines.append(f"• Runtime Resolution: failed ({snapshot.runtime_error})")
    else:
        lines.extend(
            [
                f"• Runtime Provider: {provider_label(snapshot.runtime_provider)} (`{snapshot.runtime_provider}`)",
                f"• API Mode: `{snapshot.api_mode}`",
                f"• Base URL: `{snapshot.base_url}`",
                f"• Credentials: {'configured ✓' if snapshot.credentials_configured else 'missing ⚠️'}",
            ]
        )
        if snapshot.credential_source:
            lines.append(f"• Credential Source: `{snapshot.credential_source}`")
        if snapshot.transport_command:
            lines.append(f"• Transport Command: `{snapshot.transport_command}`")

    lines.extend(
        [
            "",
            "**Usage & Context**",
            f"• Total Tokens: {snapshot.total_tokens:,}",
            f"• Last Prompt Tokens: {snapshot.last_prompt_tokens:,}",
            f"• Input / Output: {snapshot.input_tokens:,} / {snapshot.output_tokens:,}",
            f"• Cache Read / Write: {snapshot.cache_read_tokens:,} / {snapshot.cache_write_tokens:,}",
        ]
    )
    if snapshot.context_length is not None:
        lines.append(f"• Context Window: {snapshot.context_length:,} tokens")
    if snapshot.estimated_cost_usd:
        lines.append(
            f"• Estimated Cost: ${snapshot.estimated_cost_usd:.4f} ({snapshot.cost_status or 'unknown'})"
        )
    elif snapshot.cost_status and snapshot.cost_status != "unknown":
        lines.append(f"• Cost Status: `{snapshot.cost_status}`")

    lines.extend(
        [
            "",
            "**Runtime**",
            f"• Agent Running: {_yes_no(snapshot.is_running, 'Yes ⚡', 'No')}",
            f"• Pending Interrupt: {_yes_no(snapshot.has_pending_message)}",
            f"• Pending Approval: {_yes_no(snapshot.has_pending_approval)}",
            f"• Background Processes: {_yes_no(snapshot.has_background_process)}",
            f"• Voice Mode: `{snapshot.voice_mode}`",
            f"• Send Policy: `{snapshot.send_policy}`",
        ]
    )
    if snapshot.approval_command_preview:
        lines.append(f"• Approval Command: `{snapshot.approval_command_preview}`")
    if snapshot.dock_target:
        lines.append(f"• Dock Target: {snapshot.dock_target}")
    if snapshot.activation_mode:
        lines.append(f"• Activation: `{snapshot.activation_mode}`")
    if snapshot.focused_thread_summary:
        lines.append(f"• Focused Thread: {snapshot.focused_thread_summary}")

    lines.extend(
        [
            "",
            "**Platforms**",
            f"• Connected: {', '.join(snapshot.connected_platforms) if snapshot.connected_platforms else 'none'}",
        ]
    )
    return "\n".join(lines)


def render_discord_context(snapshot: DiscordContextSnapshot, mode: str = "list") -> str:
    """Render Discord context output in list, detail, or JSON form."""
    mode = str(mode or "list").strip().lower()
    if mode == "json":
        payload = {
            "session_id": snapshot.session_id,
            "session_key": snapshot.session_key,
            "source": snapshot.source.to_dict(),
            "target_source": snapshot.target_source.to_dict(),
            "message_count": snapshot.message_count,
            "role_counts": snapshot.role_counts,
            "transcript_tokens": snapshot.transcript_tokens,
            "context_prompt_tokens": snapshot.context_prompt_tokens,
            "context_prompt_chars": snapshot.context_prompt_chars,
            "current_model": snapshot.current_model,
            "current_provider": snapshot.current_provider,
            "connected_platforms": list(snapshot.connected_platforms),
            "context_files": list(snapshot.context_files),
            "skill_commands": list(snapshot.skill_commands),
            "has_soul": snapshot.has_soul,
        }
        return f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```"

    lines = [
        "🧠 **Hermes Context**",
        "",
        f"• Session ID: `{_short_id(snapshot.session_id)}`",
        f"• Session Key: `{_short_id(snapshot.session_key, limit=18)}`",
        f"• Chat: {snapshot.target_source.chat_name or snapshot.target_source.chat_id}",
        f"• Messages: {snapshot.message_count}",
        f"• Transcript Tokens: ~{snapshot.transcript_tokens:,}",
        f"• Context Prompt: ~{snapshot.context_prompt_tokens:,} tokens / {snapshot.context_prompt_chars:,} chars",
        f"• Model: `{snapshot.current_model or 'unknown'}`",
    ]
    if snapshot.current_provider:
        lines.append(f"• Provider: `{snapshot.current_provider}`")
    if snapshot.connected_platforms:
        lines.append(f"• Connected Platforms: {', '.join(snapshot.connected_platforms)}")
    if snapshot.role_counts:
        role_summary = ", ".join(f"{role}={count}" for role, count in sorted(snapshot.role_counts.items()))
        lines.append(f"• Roles: {role_summary}")

    if mode != "detail":
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "**Prompt Sources**",
            f"• SOUL.md Loaded: {_yes_no(snapshot.has_soul)}",
            f"• Context Files: {', '.join(snapshot.context_files) if snapshot.context_files else 'none'}",
            f"• Skill Commands Indexed: {', '.join(snapshot.skill_commands) if snapshot.skill_commands else 'none'}",
            "",
            "**Prompt Preview**",
            "```text",
            snapshot.context_prompt[:3500] + ("..." if len(snapshot.context_prompt) > 3500 else ""),
            "```",
        ]
    )
    return "\n".join(lines)


def _render_command_lines(command_names: tuple[str, ...]) -> list[str]:
    lookup = _command_lookup()
    lines: list[str] = []
    for name in command_names:
        cmd = lookup.get(name)
        if cmd is None:
            continue
        lines.append(
            f"`{format_gateway_command_signature(cmd)}` — {cmd.description}{_alias_note(cmd)}"
        )
    return lines


def render_discord_help() -> str:
    """Render concise Discord help with clear separation from /commands."""
    skill_cmds = _load_skill_commands()
    lines = [
        "📖 **Hermes Help**",
        "",
        "Use `/commands` for the full command catalog.",
        "",
        "**Quick Start**",
        *(
            _render_command_lines(
                ("help", "commands", "status", "whoami", "model", "models")
            )
        ),
        "",
        "**Session**",
        *(_render_command_lines(("new", "retry", "undo", "thread", "resume", "title", "compact", "context", "stop"))),
        "",
        "**Configuration**",
        *(_render_command_lines(("model", "models", "provider", "reasoning", "personality", "voice", "allowlist", "config"))),
        "",
        "**Info & Maintenance**",
        *(_render_command_lines(("usage", "insights", "export-session", "reload-mcp", "update"))),
    ]
    if skill_cmds:
        lines.extend(
            [
                "",
                f"**Skill Commands** ({len(skill_cmds)} installed)",
            ]
        )
        for name, description in skill_cmds.items():
            lines.append(f"`{name}` — {description}")
    return "\n".join(lines)


def render_discord_commands() -> str:
    """Render the full grouped Discord command catalog."""
    skill_cmds = _load_skill_commands()
    lines = [
        "🧭 **Hermes Command Catalog**",
        "",
        "Use `/help` for the guided overview.",
    ]
    for category, commands in gateway_commands_by_category():
        lines.extend(["", f"**{category}**"])
        for cmd in commands:
            lines.append(
                f"`{format_gateway_command_signature(cmd)}` — {cmd.description}{_alias_note(cmd)}"
            )
    if skill_cmds:
        lines.extend(["", "**Skill Commands**"])
        for name, description in skill_cmds.items():
            lines.append(f"`{name}` — {description}")
    return "\n".join(lines)


def collect_discord_whoami_snapshot(runner: Any, event: MessageEvent) -> DiscordWhoamiSnapshot:
    """Collect Discord identity and routing details."""
    session_source = runner._session_source_for_event(event)
    target_source = _resolve_target_source(event, session_source)
    session_key = runner._session_key_for_source(session_source)
    return DiscordWhoamiSnapshot(
        source=session_source,
        target_source=target_source,
        session_key=session_key,
    )


def render_discord_whoami(snapshot: DiscordWhoamiSnapshot) -> str:
    """Render Discord sender and routing identity."""
    source = snapshot.source
    target = snapshot.target_source
    lines = [
        "👤 **Hermes Sees You As**",
        "",
        "**Identity**",
        f"• Platform: {source.platform.value if source.platform else 'unknown'}",
        f"• User ID: `{source.user_id or 'unknown'}`",
        f"• User Name: {source.user_name or 'unknown'}",
        "",
        "**Routing**",
        f"• Chat ID: `{target.chat_id}`",
        f"• Chat Type: `{target.chat_type}`",
        f"• Chat Name: {target.chat_name or 'unknown'}",
    ]
    if target.thread_id:
        lines.append(f"• Thread ID: `{target.thread_id}`")
    if target.chat_topic:
        lines.append(f"• Chat Topic: {target.chat_topic}")
    if source.session_namespace:
        lines.append(f"• Session Namespace: `{source.session_namespace}`")
    lines.append(f"• Session Key: `{_short_id(snapshot.session_key, limit=18)}`")
    return "\n".join(lines)
