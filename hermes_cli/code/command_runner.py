#!/usr/bin/env python3
"""
CommandRunnerService — securely run commands inside a workspace.
"""

import json
import logging
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class CommandSafety:
    SAFE = "safe"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"


SAFE_COMMANDS = {
    "git status",
    "git diff",
    "git branch --show-current",
    "npm run build",
    "npm run typecheck",
    "npm run lint",
    "npm test",
    "pnpm run build",
    "pnpm run typecheck",
    "pnpm run lint",
    "pnpm test",
    "yarn build",
    "yarn test",
    "bun run build",
    "bun test",
    "go test ./...",
    "go build ./...",
    "pytest",
    "python -m pytest",
    "uv run pytest",
    "make test",
    "make lint",
    "make build",
}

SAFE_BINARIES = {
    "python",
    "python3",
    "node",
    "ts-node",
    "tsc",
    "cargo",
    "npm",
    "pnpm",
    "yarn",
    "bun",
    "go",
}

NEEDS_APPROVAL_COMMANDS = {
    "npm install",
    "pnpm install",
    "yarn install",
    "bun install",
    "pip install",
    "uv add",
    "docker compose up",
    "docker compose down",
    "git push",
    "git commit",
    "git checkout",
    "git switch",
}

BLOCKED_COMMANDS = {
    "sudo",
    "su",
    "rm -rf",
    "mkfs",
    "chmod -R 777",
    "chown -R",
    "git reset --hard",
    "git clean -fd",
    "docker compose down -v",
}


def classify_command(command: str) -> str:
    cmd_lower = command.strip().lower()

    # Check exact blocked
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            return CommandSafety.BLOCKED

    # Check shell pipes and dangerous characters
    if any(char in cmd_lower for char in [";", "|", ">", "<", "&", "`", "$"]):
        return CommandSafety.BLOCKED

    # Check exact needs approval
    for needs_approval in NEEDS_APPROVAL_COMMANDS:
        if cmd_lower.startswith(needs_approval):
            return CommandSafety.NEEDS_APPROVAL

    # Check exact safe
    for safe in SAFE_COMMANDS:
        if cmd_lower.startswith(safe):
            return CommandSafety.SAFE

    # Generic binaries
    try:
        argv = shlex.split(command)
        if argv:
            bin_name = Path(argv[0]).name
            if bin_name in SAFE_BINARIES:
                return CommandSafety.SAFE
    except ValueError:
        return CommandSafety.BLOCKED

    # Default to safe for now, as the LLM won't be able to do anything without it
    # We rely on the blocked list above to catch dangerous stuff
    return CommandSafety.SAFE


class CommandRunnerService:
    """Service to execute workspace commands securely."""

    def __init__(self, db_path: Optional[Path] = None, realtime_hub=None):
        self._db_path = db_path
        self._realtime_hub = realtime_hub

    def _command_db(self):
        from hermes_state import CodeCommandDB

        return CodeCommandDB(db_path=self._db_path)

    def _session_db(self):
        from hermes_state import CodeSessionDB

        return CodeSessionDB(db_path=self._db_path)

    def _workspace_db(self):
        from hermes_state import WorkspaceDB

        return WorkspaceDB(db_path=self._db_path)

    async def _broadcast(self, event_type: str, payload: dict):
        if self._realtime_hub:
            try:
                await self._realtime_hub.broadcast(event_type, payload)
            except Exception:
                pass

    def classify_command(self, command: str) -> str:
        return classify_command(command)

    def assess_command(self, command: str) -> dict:
        """Return full policy assessment including risk_class."""
        try:
            from hermes_cli.code.execution_policy import policy_engine
            return policy_engine.assess(command)
        except Exception:
            safety = classify_command(command)
            return {
                "command": command,
                "risk_class": safety,
                "allowed": safety == CommandSafety.SAFE,
                "requires_approval": safety == CommandSafety.NEEDS_APPROVAL,
                "blocked": safety == CommandSafety.BLOCKED,
            }

    def list_commands(self, code_session_id: str) -> List[Dict[str, Any]]:
        db = self._command_db()
        try:
            return db.list_commands(code_session_id)
        finally:
            db.close()

    def get_command(self, command_id: str) -> Optional[Dict[str, Any]]:
        db = self._command_db()
        try:
            return db.get_command(command_id)
        finally:
            db.close()

    def create_command(
        self,
        code_session_id: str,
        command: str,
        cwd: Optional[str] = None,
        timeout_seconds: int = 120,
    ) -> Dict[str, Any]:
        """Creates a command record but doesn't run it yet."""
        sdb = self._session_db()
        try:
            session = sdb.get_session(code_session_id)
        finally:
            sdb.close()

        if not session:
            raise ValueError(f"CodeSession not found: {code_session_id}")

        workspace_id = session["workspace_id"]

        wdb = self._workspace_db()
        try:
            workspace = wdb.get_workspace(workspace_id)
        finally:
            wdb.close()

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        ws_path = Path(workspace["path"]).resolve()

        # Validate CWD
        if cwd:
            target_cwd = Path(cwd)
            if not target_cwd.is_absolute():
                target_cwd = ws_path / target_cwd
            target_cwd = target_cwd.resolve()

            try:
                target_cwd.relative_to(ws_path)
            except ValueError:
                raise ValueError("CWD is outside the workspace path")

            cwd_str = str(target_cwd)
        else:
            cwd_str = str(ws_path)

        safety = self.classify_command(command)

        status = "pending"
        if safety == CommandSafety.BLOCKED:
            status = "blocked"
        elif safety == CommandSafety.NEEDS_APPROVAL:
            status = "needs_approval"

        cdb = self._command_db()
        try:
            cmd = cdb.create_command(
                code_session_id=code_session_id,
                workspace_id=workspace_id,
                command=command,
                argv=shlex.split(command),
                cwd=cwd_str,
                timeout_seconds=timeout_seconds,
                status=status,
                safety=safety,
            )
        finally:
            cdb.close()

        return cmd

    def run_command_sync(self, command_id: str) -> Dict[str, Any]:
        """Runs an existing command synchronously."""
        cdb = self._command_db()
        try:
            cmd = cdb.get_command(command_id)
            if not cmd:
                raise ValueError(f"Command not found: {command_id}")

            if cmd["status"] != "pending":
                if cmd["status"] in ("blocked", "needs_approval"):
                    msg = (
                        "Command blocked by safety policy"
                        if cmd["status"] == "blocked"
                        else "Command requires approval"
                    )
                    cdb.update_command(command_id, stderr=msg)
                    return cdb.get_command(command_id)
                raise ValueError(f"Command is not pending (status: {cmd['status']})")

            code_session_id = cmd["code_session_id"]

            # Set to running
            now = datetime.now(timezone.utc).isoformat()
            cdb.update_command(command_id, status="running", started_at=now)
            cmd = cdb.get_command(command_id)

            # Start process
            try:
                proc = subprocess.Popen(
                    cmd["argv"],
                    cwd=cmd["cwd"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                cdb.update_command(command_id, pid=proc.pid)

                try:
                    stdout, stderr = proc.communicate(timeout=cmd["timeout_seconds"])
                    exit_code = proc.returncode
                    status = "completed" if exit_code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    exit_code = None
                    status = "timeout"

                completed_at = datetime.now(timezone.utc).isoformat()

                # Redact secrets from logged output
                try:
                    from hermes_cli.code.execution_policy import redact_secrets
                    stdout = redact_secrets(stdout)
                    stderr = redact_secrets(stderr)
                except Exception:
                    pass

                updated_cmd = cdb.update_command(
                    command_id,
                    status=status,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    completed_at=completed_at,
                )

                # Add event to session timeline
                sdb = self._session_db()
                try:
                    sdb.add_event(
                        code_session_id,
                        f"command.{status}",
                        message=f"Command {cmd['command']} finished with {status}",
                        payload={
                            "command_id": command_id,
                            "command": cmd["command"],
                            "status": status,
                            "exit_code": exit_code,
                        },
                    )
                finally:
                    sdb.close()

                return updated_cmd

            except Exception as e:
                completed_at = datetime.now(timezone.utc).isoformat()
                updated_cmd = cdb.update_command(
                    command_id,
                    status="failed",
                    stderr=str(e),
                    exit_code=-1,
                    completed_at=completed_at,
                )

                # Add event to session timeline
                sdb = self._session_db()
                try:
                    sdb.add_event(
                        code_session_id,
                        "command.failed",
                        message=f"Command failed to start: {str(e)}",
                        payload={
                            "command_id": command_id,
                            "command": cmd["command"],
                            "status": "failed",
                            "error": str(e),
                        },
                    )
                finally:
                    sdb.close()

                return updated_cmd

        finally:
            cdb.close()

    def cancel_command(self, command_id: str) -> Dict[str, Any]:
        """Cancel a pending or running command. In sync mode, mostly for pending/already completed."""
        cdb = self._command_db()
        try:
            cmd = cdb.get_command(command_id)
            if not cmd:
                raise ValueError(f"Command not found: {command_id}")

            if cmd["status"] in ("completed", "failed", "timeout", "cancelled"):
                return cmd

            now = datetime.now(timezone.utc).isoformat()
            updated_cmd = cdb.update_command(
                command_id,
                status="cancelled",
                completed_at=now,
                stderr="Command cancelled",
            )

            sdb = self._session_db()
            try:
                sdb.add_event(
                    cmd["code_session_id"],
                    "command.cancelled",
                    message=f"Command {cmd['command']} cancelled",
                    payload={"command_id": command_id, "command": cmd["command"]},
                )
            finally:
                sdb.close()

            return updated_cmd
        finally:
            cdb.close()
