"""Helpers for resolving non-built-in gateway slash commands."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from gateway.command_preprocessing_runtime_service import (
    preprocess_gateway_command,
)


@dataclass(slots=True)
class GatewayCommandResolutionResult:
    """Result after resolving non-built-in gateway slash commands."""

    command: str | None
    handled: bool = False
    response: str | None = None


def _load_quick_commands(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        quick_commands = config.get("quick_commands", {}) or {}
    else:
        quick_commands = getattr(config, "quick_commands", {}) or {}
    return quick_commands if isinstance(quick_commands, dict) else {}


async def resolve_gateway_non_builtin_command(
    *,
    runner: Any,
    event: Any,
    source: Any,
    session_key: str,
    command: str | None,
    logger: Any,
    unavailable_skill_checker: Callable[[str], str | None],
) -> GatewayCommandResolutionResult:
    """Resolve quick/plugin/skill slash commands after built-in preprocessing."""

    if not command:
        return GatewayCommandResolutionResult(command=None, handled=False, response=None)

    quick_commands = _load_quick_commands(runner.config)
    if command in quick_commands:
        qcmd = quick_commands[command]
        if qcmd.get("type") == "exec":
            exec_cmd = qcmd.get("command", "")
            if not exec_cmd:
                return GatewayCommandResolutionResult(
                    command=command,
                    handled=True,
                    response=f"Quick command '/{command}' has no command defined.",
                )
            try:
                proc = await asyncio.create_subprocess_shell(
                    exec_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                output = (stdout or stderr).decode().strip()
                return GatewayCommandResolutionResult(
                    command=command,
                    handled=True,
                    response=output if output else "Command returned no output.",
                )
            except asyncio.TimeoutError:
                return GatewayCommandResolutionResult(
                    command=command,
                    handled=True,
                    response="Quick command timed out (30s).",
                )
            except Exception as exc:
                return GatewayCommandResolutionResult(
                    command=command,
                    handled=True,
                    response=f"Quick command error: {exc}",
                )

        if qcmd.get("type") == "alias":
            target = str(qcmd.get("target", "") or "").strip()
            if not target:
                return GatewayCommandResolutionResult(
                    command=command,
                    handled=True,
                    response=f"Quick command '/{command}' has no target defined.",
                )

            target = target if target.startswith("/") else f"/{target}"
            target_command = target.lstrip("/")
            user_args = event.get_command_args().strip()
            event.text = f"{target} {user_args}".strip()
            command = target_command

            preprocessed_command = await preprocess_gateway_command(
                runner=runner,
                event=event,
                source=source,
                session_key=session_key,
                logger=logger,
            )
            if preprocessed_command.handled:
                return GatewayCommandResolutionResult(
                    command=preprocessed_command.command,
                    handled=True,
                    response=preprocessed_command.response,
                )
            command = preprocessed_command.command
        else:
            return GatewayCommandResolutionResult(
                command=command,
                handled=True,
                response=(
                    f"Quick command '/{command}' has unsupported type "
                    "(supported: 'exec', 'alias')."
                ),
            )

    try:
        from hermes_cli.plugins import get_plugin_command_handler

        plugin_handler = get_plugin_command_handler(command.replace("_", "-"))
        if plugin_handler:
            user_args = event.get_command_args().strip()
            result = plugin_handler(user_args)
            if asyncio.iscoroutine(result):
                result = await result
            return GatewayCommandResolutionResult(
                command=command,
                handled=True,
                response=str(result) if result else None,
            )
    except Exception as exc:
        logger.debug("Plugin command dispatch failed (non-fatal): %s", exc)

    try:
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        from agent.skill_commands import (
            build_skill_invocation_message,
            get_skill_commands,
            resolve_skill_command_key,
        )

        skill_cmds = get_skill_commands()
        cmd_key = resolve_skill_command_key(command)
        if cmd_key is not None:
            skill_name = skill_cmds[cmd_key].get("name", "")
            platform_name = source.platform.value if source.platform else None
            if platform_name and skill_name:
                from agent.skill_utils import get_disabled_skill_names as _get_plat_disabled

                if skill_name in _get_plat_disabled(platform=platform_name):
                    return GatewayCommandResolutionResult(
                        command=command,
                        handled=True,
                        response=(
                            f"The **{skill_name}** skill is disabled for {platform_name}.\n"
                            f"Enable it with: `hermes skills config`"
                        ),
                    )

            user_instruction = event.get_command_args().strip()
            msg = build_skill_invocation_message(
                cmd_key,
                user_instruction,
                task_id=session_key,
            )
            if msg:
                event.text = msg
            return GatewayCommandResolutionResult(
                command=command,
                handled=False,
                response=None,
            )

        unavailable_skill_message = unavailable_skill_checker(command)
        if unavailable_skill_message:
            return GatewayCommandResolutionResult(
                command=command,
                handled=True,
                response=unavailable_skill_message,
            )

        if command.replace("_", "-") not in GATEWAY_KNOWN_COMMANDS:
            logger.warning(
                "Unrecognized slash command /%s from %s — replying with unknown-command notice",
                command,
                source.platform.value if source.platform else "?",
            )
            return GatewayCommandResolutionResult(
                command=command,
                handled=True,
                response=(
                    f"Unknown command `/{command}`. "
                    f"Type /commands to see what's available, "
                    f"or resend without the leading slash to send "
                    f"as a regular message."
                ),
            )
    except Exception as exc:
        logger.debug("Skill command check failed (non-fatal): %s", exc)

    return GatewayCommandResolutionResult(
        command=command,
        handled=False,
        response=None,
    )
