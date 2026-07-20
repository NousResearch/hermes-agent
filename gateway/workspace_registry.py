"""Profile-local workspace registry for gateway channel/project bindings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from hermes_constants import get_hermes_home

from .session import SessionSource


@dataclass(frozen=True)
class WorkspaceBinding:
    """Authoritative project binding resolved from a gateway channel."""

    slug: str
    name: str
    repo_path: Optional[str] = None
    canonical_repo_url: Optional[str] = None
    default_branch: Optional[str] = None
    response_policy: Optional[str] = None
    source: str = "workspaces.yaml"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceBinding":
        return cls(
            slug=str(data["slug"]),
            name=str(data.get("name") or data["slug"]),
            repo_path=data.get("repo_path"),
            canonical_repo_url=data.get("canonical_repo_url"),
            default_branch=data.get("default_branch"),
            response_policy=data.get("response_policy"),
            source=str(data.get("source") or "workspaces.yaml"),
        )


class WorkspaceRegistry:
    """Resolve platform channel/thread IDs to profile-local project metadata."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path is not None else default_workspace_registry_path()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        loaded = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        workspaces = loaded.get("workspaces", loaded)
        return workspaces if isinstance(workspaces, dict) else {}

    def resolve_source(self, source: SessionSource) -> Optional[WorkspaceBinding]:
        platform = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        chat_id = str(source.chat_id)
        thread_id = str(source.thread_id) if source.thread_id else None
        source_scope = source.scope_id or source.guild_id
        workspace_scope = str(source_scope) if source_scope else None

        for slug, workspace in self._data.items():
            if not isinstance(workspace, dict):
                continue
            for channel in workspace.get("channels", []) or []:
                if not isinstance(channel, dict):
                    continue
                if str(channel.get("platform", "")) != platform:
                    continue
                if not _channel_matches(channel, chat_id, thread_id, workspace_scope):
                    continue
                return WorkspaceBinding(
                    slug=str(slug),
                    name=str(workspace.get("name") or slug),
                    repo_path=workspace.get("repo_path"),
                    canonical_repo_url=workspace.get("canonical_repo_url"),
                    default_branch=workspace.get("default_branch"),
                    response_policy=channel.get("response_policy"),
                    source=str(self.config_path),
                )
        return None


def default_workspace_registry_path() -> Path:
    """Return the default profile-local workspaces.yaml path."""

    return get_hermes_home() / "workspaces.yaml"


def resolve_workspace_binding(source: SessionSource, config_path: str | Path | None = None) -> Optional[WorkspaceBinding]:
    """Resolve a source using the default profile-local workspace registry."""

    return WorkspaceRegistry(config_path).resolve_source(source)


def _channel_matches(
    channel: dict[str, Any],
    chat_id: str,
    thread_id: Optional[str],
    workspace_scope: Optional[str],
) -> bool:
    channel_ids = (
        channel.get("chat_id"),
        channel.get("channel_id"),
        channel.get("room_id"),
        channel.get("conversation_id"),
    )
    if chat_id not in {str(value) for value in channel_ids if value is not None}:
        return False

    configured_thread = channel.get("thread_id") or channel.get("topic_id")
    if configured_thread is not None and str(configured_thread) != (thread_id or ""):
        return False

    configured_scope = (
        channel.get("scope_id")
        or channel.get("guild_id")
        or channel.get("workspace_id")
        or channel.get("team_id")
    )
    if configured_scope is not None and str(configured_scope) != (workspace_scope or ""):
        return False

    return True
