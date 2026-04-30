#!/usr/bin/env python3
"""GitHub ChatOps parser and orchestration bridge (P1)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli.code.agent_orchestrator import AgentOrchestrator, OrchestratorState
from hermes_cli.code.artifact_ledger import ArtifactLedger
from hermes_cli.code.execution_policy import policy_engine
from hermes_cli.code.github_integration import GitHubIntegrationStore
from hermes_cli.code.repo_knowledge import RepoKnowledgeService
from hermes_state import SessionDB

SUPPORTED_COMMANDS = frozenset({"plan", "review", "fix", "explain", "status"})
_COMMAND_RE = re.compile(r"(?im)^\s*@hermes(?:\s+|:)(plan|review|fix|explain|status)\b[ \t]*(.*)$")


@dataclass(frozen=True)
class ChatOpsCommand:
    command: str
    args: str = ""


def parse_chatops_commands(text: str) -> list[ChatOpsCommand]:
    if not text:
        return []
    commands: list[ChatOpsCommand] = []
    for match in _COMMAND_RE.finditer(text):
        command = match.group(1).lower()
        if command in SUPPORTED_COMMANDS:
            commands.append(ChatOpsCommand(command=command, args=match.group(2).strip()))
    return commands


def _task_title(command: dict[str, Any]) -> str:
    repo = command.get("repo_full_name") or "GitHub"
    number = command.get("pr_number") or command.get("issue_number")
    suffix = f" #{number}" if number else ""
    return f"GitHub @{command.get('command')} request: {repo}{suffix}"


def _task_description(command: dict[str, Any]) -> str:
    lines = [
        f"GitHub ChatOps command: @hermes {command.get('command')}",
        f"Repository: {command.get('repo_full_name')}",
        f"Sender: {command.get('sender_login') or 'unknown'}",
    ]
    if command.get("issue_number"):
        lines.append(f"Issue: #{command.get('issue_number')}")
    if command.get("pr_number"):
        lines.append(f"Pull request: #{command.get('pr_number')}")
    if command.get("comment_id"):
        lines.append(f"Comment: {command.get('comment_id')}")
    if command.get("args"):
        lines.extend(["", str(command["args"])])
    return "\n".join(lines)


class GitHubChatOpsService:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path

    def _store(self) -> GitHubIntegrationStore:
        return GitHubIntegrationStore(db_path=self._db_path)

    def _orchestrator(self) -> AgentOrchestrator:
        return AgentOrchestrator(db_path=self._db_path)

    def _ledger(self) -> ArtifactLedger:
        return ArtifactLedger(db_path=self._db_path)

    def create_commands_from_comment(
        self,
        *,
        delivery_id: Optional[str],
        repo_full_name: str,
        issue_number: Optional[int],
        pr_number: Optional[int],
        comment_id: Optional[int],
        sender_login: Optional[str],
        body: str,
    ) -> list[dict[str, Any]]:
        commands = parse_chatops_commands(body)
        if not commands:
            return []
        store = self._store()
        try:
            return [
                store.create_chatops_command(
                    delivery_id=delivery_id,
                    repo_full_name=repo_full_name,
                    issue_number=issue_number,
                    pr_number=pr_number,
                    comment_id=comment_id,
                    sender_login=sender_login,
                    command=item.command,
                    args=item.args,
                )
                for item in commands
            ]
        finally:
            store.close()

    def _workspace_for_repo(self, repo_full_name: str) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        db = SessionDB(db_path=self._db_path) if self._db_path else SessionDB()
        try:
            for workspace in db.list_code_workspaces(limit=500):
                repo_url = str(workspace.get("repo_url") or "").removesuffix(".git")
                if repo_url.endswith(f"github.com/{repo_full_name}") or repo_url.endswith(f":{repo_full_name}"):
                    return workspace.get("id"), workspace
        finally:
            db.close()
        return None, None

    def _repo_guidance(self, workspace: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not workspace or not workspace.get("path"):
            return None
        try:
            return RepoKnowledgeService().detect(Path(workspace["path"]))
        except Exception:
            return None

    def _link_latest_run_for_status(self, command: dict[str, Any]) -> Optional[dict[str, Any]]:
        store = self._store()
        try:
            recent = store.list_chatops_commands(limit=200)
            for item in recent:
                if item.get("id") == command.get("id"):
                    continue
                if item.get("repo_full_name") != command.get("repo_full_name"):
                    continue
                if item.get("orchestrated_run_id") and (
                    item.get("issue_number") == command.get("issue_number")
                    or item.get("pr_number") == command.get("pr_number")
                ):
                    run = self._orchestrator().get_run(item["orchestrated_run_id"])
                    if run:
                        store.update_chatops_command(
                            command["id"],
                            status="linked_existing_run",
                            orchestrated_run_id=item["orchestrated_run_id"],
                            code_session_id=run.get("code_session_id"),
                        )
                        return run
        finally:
            store.close()
        return None

    def run_command(self, command_id: str) -> dict[str, Any]:
        store = self._store()
        try:
            command = store.get_chatops_command(command_id)
        finally:
            store.close()
        if not command:
            raise ValueError(f"GitHub ChatOps command not found: {command_id}")

        if command.get("orchestrated_run_id"):
            run = self._orchestrator().get_run(command["orchestrated_run_id"])
            return {"command": command, "run": run, "resumed": True}

        if str(command.get("command")) == "status":
            linked = self._link_latest_run_for_status(command)
            if linked is not None:
                refreshed_store = self._store()
                try:
                    updated = refreshed_store.get_chatops_command(command_id)
                finally:
                    refreshed_store.close()
                return {"command": updated, "run": linked, "resumed": True}

        workspace_id, workspace = self._workspace_for_repo(str(command.get("repo_full_name") or ""))
        guidance = self._repo_guidance(workspace)
        title = _task_title(command)
        description = _task_description(command)

        metadata = {
            "source": "github_chatops",
            "github": {
                "repo_full_name": command.get("repo_full_name"),
                "issue_number": command.get("issue_number"),
                "pr_number": command.get("pr_number"),
                "comment_id": command.get("comment_id"),
                "sender_login": command.get("sender_login"),
                "command": command.get("command"),
                "args": command.get("args"),
            },
            "repo_guidance": guidance,
        }

        run = self._orchestrator().create_run(
            title=title,
            goal=description,
            workspace_id=workspace_id,
            metadata=metadata,
            create_intake_artifact=True,
        )

        command_name = str(command.get("command") or "")
        if command_name == "plan":
            self._ledger().create_artifact(
                "implementation_plan",
                f"Planning requested from GitHub.\n\n{description}",
                title="GitHub Planning Request",
                workspace_id=workspace_id,
                code_session_id=run.get("code_session_id"),
                orchestrated_run_id=run["id"],
            )
        elif command_name == "review":
            self._ledger().create_artifact(
                "review_report",
                f"Review requested from GitHub.\n\n{description}",
                title="GitHub Review Request",
                workspace_id=workspace_id,
                code_session_id=run.get("code_session_id"),
                orchestrated_run_id=run["id"],
            )
        elif command_name == "fix":
            run = self._orchestrator().transition_run(run["id"], OrchestratorState.DISCOVERY, reason="GitHub fix request intake")
            run = self._orchestrator().transition_run(run["id"], OrchestratorState.PLANNING, reason="Plan fix before implementation")
            run = self._orchestrator().transition_run(
                run["id"],
                OrchestratorState.APPROVAL,
                reason="Fix request requires approval before implementation",
                metadata={"risk": policy_engine.assess_command("git commit -m fix")},
            )
        elif command_name == "explain":
            self._ledger().create_artifact(
                "architecture_note",
                f"Explanation requested from GitHub.\n\n{description}",
                title="GitHub Explanation Request",
                workspace_id=workspace_id,
                code_session_id=run.get("code_session_id"),
                orchestrated_run_id=run["id"],
            )

        updated_store = self._store()
        try:
            updated = updated_store.update_chatops_command(
                command_id,
                status="run_created",
                orchestrated_run_id=run["id"],
                code_session_id=run.get("code_session_id"),
            )
        finally:
            updated_store.close()
        return {"command": updated, "run": run, "resumed": False}
