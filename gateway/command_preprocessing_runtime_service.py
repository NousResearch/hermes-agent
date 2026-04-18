"""Shared runtime helpers for gateway command preprocessing and dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GatewayCommandPreprocessResult:
    """Result of preprocessing one gateway slash command."""

    command: str | None
    canonical: str | None
    handled: bool = False
    response: str | None = None


_BUILTIN_HANDLER_ATTRS = {
    "new": "_handle_reset_command",
    "help": "_handle_help_command",
    "commands": "_handle_commands_command",
    "profile": "_handle_profile_command",
    "status": "_handle_status_command",
    "stop": "_handle_stop_command",
    "reasoning": "_handle_reasoning_command",
    "verbose": "_handle_verbose_command",
    "yolo": "_handle_yolo_command",
    "model": "_handle_model_command",
    "provider": "_handle_provider_command",
    "personality": "_handle_personality_command",
    "retry": "_handle_retry_command",
    "undo": "_handle_undo_command",
    "sethome": "_handle_set_home_command",
    "compress": "_handle_compress_command",
    "usage": "_handle_usage_command",
    "insights": "_handle_insights_command",
    "reload-mcp": "_handle_reload_mcp_command",
    "approve": "_handle_approve_command",
    "deny": "_handle_deny_command",
    "update": "_handle_update_command",
    "title": "_handle_title_command",
    "resume": "_handle_resume_command",
    "branch": "_handle_branch_command",
    "rollback": "_handle_rollback_command",
    "background": "_handle_background_command",
    "btw": "_handle_btw_command",
    "voice": "_handle_voice_command",
}


async def preprocess_gateway_command(
    *,
    runner: Any,
    event: Any,
    source: Any,
    session_key: str,
    logger: Any,
) -> GatewayCommandPreprocessResult:
    """Resolve, emit, and dispatch built-in slash commands before agent execution."""

    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    command = event.get_command()
    if command and command in GATEWAY_KNOWN_COMMANDS:
        await runner.hooks.emit(
            f"command:{command}",
            {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "command": command,
                "args": event.get_command_args().strip(),
            },
        )

    cmd_def = resolve_command(command) if command else None
    canonical = cmd_def.name if cmd_def else command

    if canonical == "plan":
        try:
            from agent.skill_commands import (
                build_plan_path,
                build_skill_invocation_message,
            )

            user_instruction = event.get_command_args().strip()
            plan_path = build_plan_path(user_instruction)
            event.text = build_skill_invocation_message(
                "/plan",
                user_instruction,
                task_id=session_key,
                runtime_note=(
                    "Save the markdown plan with write_file to this exact relative path "
                    f"inside the active workspace/backend cwd: {plan_path}"
                ),
            )
            if not event.text:
                return GatewayCommandPreprocessResult(
                    command=command,
                    canonical=canonical,
                    handled=True,
                    response="Failed to load the bundled /plan skill.",
                )
            return GatewayCommandPreprocessResult(
                command=command,
                canonical=None,
                handled=False,
                response=None,
            )
        except Exception as exc:
            logger.exception("Failed to prepare /plan command")
            return GatewayCommandPreprocessResult(
                command=command,
                canonical=canonical,
                handled=True,
                response=f"Failed to enter plan mode: {exc}",
            )

    handler_attr = _BUILTIN_HANDLER_ATTRS.get(str(canonical or "").strip())
    if not handler_attr:
        return GatewayCommandPreprocessResult(
            command=command,
            canonical=canonical,
            handled=False,
            response=None,
        )

    handler = getattr(runner, handler_attr)
    response = await handler(event)
    return GatewayCommandPreprocessResult(
        command=command,
        canonical=canonical,
        handled=True,
        response=response,
    )
