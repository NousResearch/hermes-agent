# gateway/daimon/admin_commands.py
"""Admin command handlers for /daimon slash command."""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from gateway.daimon.session_manager import DaimonSessionManager

logger = logging.getLogger(__name__)

CONTAINER_NAME = "daimon-sandbox"


@dataclass
class CommandResult:
    """Result of an admin command."""
    success: bool
    message: str


def handle_daimon_command(
    subcommand: str,
    args: str,
    session_manager: DaimonSessionManager,
    banned_users: set[str],
) -> CommandResult:
    """Dispatch a /daimon subcommand.

    Args:
        subcommand: One of "restart", "status", "kill", "ban", "limits"
        args: Remaining arguments after the subcommand
        session_manager: The DaimonSessionManager instance
        banned_users: Mutable set of banned user IDs (persisted by caller)

    Returns:
        CommandResult with success flag and formatted message.
    """
    handlers = {
        "restart": _handle_restart,
        "status": _handle_status,
        "kill": _handle_kill,
        "ban": _handle_ban,
        "limits": _handle_limits,
    }

    handler = handlers.get(subcommand)
    if handler is None:
        available = ", ".join(sorted(handlers.keys()))
        return CommandResult(
            success=False,
            message=f"Unknown subcommand: `{subcommand}`\nAvailable: {available}",
        )

    return handler(args, session_manager, banned_users)


def _handle_restart(
    args: str, mgr: DaimonSessionManager, banned: set[str]
) -> CommandResult:
    """Restart the sandbox container."""
    docker = shutil.which("docker") or "docker"
    try:
        result = subprocess.run(
            [docker, "restart", CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return CommandResult(
                success=True,
                message=(
                    f"✅ Container `{CONTAINER_NAME}` restarted.\n"
                    f"⚠️ All active sessions ({mgr.active_sessions}) were terminated."
                ),
            )
        else:
            return CommandResult(
                success=False,
                message=f"❌ Restart failed: {result.stderr.strip()}",
            )
    except subprocess.TimeoutExpired:
        return CommandResult(success=False, message="❌ Restart timed out (60s).")
    except Exception as e:
        return CommandResult(success=False, message=f"❌ Restart error: {e}")


def _handle_status(
    args: str, mgr: DaimonSessionManager, banned: set[str]
) -> CommandResult:
    """Show container and session status."""
    docker = shutil.which("docker") or "docker"

    # Get container stats
    container_info = "unavailable"
    try:
        result = subprocess.run(
            [docker, "stats", CONTAINER_NAME, "--no-stream", "--format",
             "CPU: {{.CPUPerc}}, Mem: {{.MemUsage}}, PIDs: {{.PIDs}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            container_info = result.stdout.strip()
    except Exception:
        pass

    # Get container uptime
    uptime = "unknown"
    try:
        result = subprocess.run(
            [docker, "inspect", CONTAINER_NAME, "--format", "{{.State.StartedAt}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            uptime = f"since {result.stdout.strip()[:19]}"
    except Exception:
        pass

    msg = (
        f"**Daimon Status**\n"
        f"Container: `{CONTAINER_NAME}` ({uptime})\n"
        f"Resources: {container_info}\n"
        f"Active sessions: {mgr.active_sessions}/{mgr.config.max_active_sessions}\n"
        f"Queue: {mgr.queue_length}\n"
        f"Banned users: {len(banned)}"
    )
    return CommandResult(success=True, message=msg)


def _handle_kill(
    args: str, mgr: DaimonSessionManager, banned: set[str]
) -> CommandResult:
    """Kill a specific session by thread ID."""
    thread_id = args.strip()
    if not thread_id:
        return CommandResult(success=False, message="Usage: `/daimon kill <thread_id>`")

    promoted = mgr.end_session(thread_id)
    msg = f"✅ Session `{thread_id}` terminated."
    if promoted:
        msg += f"\n↪ Promoted queued session: `{promoted}`"
    return CommandResult(success=True, message=msg)


def _handle_ban(
    args: str, mgr: DaimonSessionManager, banned: set[str]
) -> CommandResult:
    """Ban a user by Discord user ID."""
    user_id = args.strip()
    if not user_id:
        return CommandResult(success=False, message="Usage: `/daimon ban <user_id>`")

    banned.add(user_id)
    return CommandResult(
        success=True,
        message=f"✅ Banned user `{user_id}`. They can no longer create Daimon sessions.",
    )


def _handle_limits(
    args: str, mgr: DaimonSessionManager, banned: set[str]
) -> CommandResult:
    """Display current user limits."""
    cfg = mgr.config

    # Format tool limits (only show non-unlimited ones)
    tool_lines = []
    for tool, limit in sorted(cfg.tool_limits.items()):
        if limit == 0:
            tool_lines.append(f"  {tool}: ❌ disabled")
        elif limit > 0:
            tool_lines.append(f"  {tool}: {limit}/session")
        # Skip -1 (unlimited) — not interesting to show

    msg = (
        f"**Daimon User Limits**\n"
        f"Model: `{cfg.user_model}`\n"
        f"Iterations/thread: {cfg.max_iterations}\n"
        f"Threads/day/user: {cfg.max_threads_per_day}\n"
        f"Timeout: {cfg.gateway_timeout}s\n"
        f"Concurrency: {cfg.max_active_sessions}\n"
        f"**Tool limits:**\n" + "\n".join(tool_lines)
    )
    return CommandResult(success=True, message=msg)
