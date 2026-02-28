"""Honcho client initialization and configuration.

Reads the global ~/.honcho/config.json when available, falling back
to environment variables.

Resolution order for host-specific settings:
  1. Explicit host block fields (always win)
  2. Flat/global fields from config root
  3. Defaults (host name as workspace/peer)
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from honcho import Honcho

logger = logging.getLogger(__name__)

GLOBAL_CONFIG_PATH = Path.home() / ".honcho" / "config.json"
HOST = "hermes"


@dataclass
class HonchoClientConfig:
    """Configuration for Honcho client, resolved for a specific host."""

    host: str = HOST
    workspace_id: str = "hermes"
    api_key: str | None = None
    environment: str = "production"
    # Identity
    peer_name: str | None = None
    ai_peer: str = "hermes"
    linked_hosts: list[str] = field(default_factory=list)
    # Toggles
    enabled: bool = False
    save_messages: bool = True
    # Prefetch budget
    context_tokens: int | None = None
    # Session resolution
    session_strategy: str = "per-directory"
    session_peer_prefix: bool = False
    sessions: dict[str, str] = field(default_factory=dict)
    # Raw global config for anything else consumers need
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, workspace_id: str = "hermes") -> HonchoClientConfig:
        """Create config from environment variables (fallback)."""
        return cls(
            workspace_id=workspace_id,
            api_key=os.environ.get("HONCHO_API_KEY"),
            environment=os.environ.get("HONCHO_ENVIRONMENT", "production"),
            enabled=True,
        )

    @classmethod
    def from_global_config(
        cls,
        host: str = HOST,
        config_path: Path | None = None,
    ) -> HonchoClientConfig:
        """Create config from ~/.honcho/config.json.

        Falls back to environment variables if the file doesn't exist.
        """
        path = config_path or GLOBAL_CONFIG_PATH
        if not path.exists():
            logger.debug("No global Honcho config at %s, falling back to env", path)
            return cls.from_env()

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s, falling back to env", path, e)
            return cls.from_env()

        host_block = (raw.get("hosts") or {}).get(host, {})

        # Explicit host block fields win, then flat/global, then defaults
        workspace = (
            host_block.get("workspace")
            or raw.get("workspace")
            or host
        )
        ai_peer = (
            host_block.get("aiPeer")
            or raw.get("aiPeer")
            or host
        )
        linked_hosts = host_block.get("linkedHosts", [])

        return cls(
            host=host,
            workspace_id=workspace,
            api_key=raw.get("apiKey") or os.environ.get("HONCHO_API_KEY"),
            environment=raw.get("environment", "production"),
            peer_name=raw.get("peerName"),
            ai_peer=ai_peer,
            linked_hosts=linked_hosts,
            enabled=raw.get("enabled", False),
            save_messages=raw.get("saveMessages", True),
            context_tokens=raw.get("contextTokens") or host_block.get("contextTokens"),
            session_strategy=raw.get("sessionStrategy", "per-directory"),
            session_peer_prefix=raw.get("sessionPeerPrefix", False),
            sessions=raw.get("sessions", {}),
            raw=raw,
        )

    def resolve_session_name(self, cwd: str | None = None) -> str | None:
        """Resolve session name for a directory.

        Checks manual overrides first, then derives from directory name.
        """
        if not cwd:
            cwd = os.getcwd()

        # Manual override
        manual = self.sessions.get(cwd)
        if manual:
            return manual

        # Derive from directory basename
        base = Path(cwd).name
        if self.session_peer_prefix and self.peer_name:
            return f"{self.peer_name}-{base}"
        return base

    def get_linked_workspaces(self) -> list[str]:
        """Resolve linked host keys to workspace names."""
        hosts = self.raw.get("hosts", {})
        workspaces = []
        for host_key in self.linked_hosts:
            block = hosts.get(host_key, {})
            ws = block.get("workspace") or host_key
            if ws != self.workspace_id:
                workspaces.append(ws)
        return workspaces


_honcho_client: Honcho | None = None


def get_honcho_client(config: HonchoClientConfig | None = None) -> Honcho:
    """Get or create the Honcho client singleton.

    When no config is provided, attempts to load ~/.honcho/config.json
    first, falling back to environment variables.
    """
    global _honcho_client

    if _honcho_client is not None:
        return _honcho_client

    if config is None:
        config = HonchoClientConfig.from_global_config()

    if not config.api_key:
        raise ValueError(
            "Honcho API key not found. Set it in ~/.honcho/config.json "
            "or the HONCHO_API_KEY environment variable. "
            "Get an API key from https://app.honcho.dev"
        )

    try:
        from honcho import Honcho
    except ImportError:
        raise ImportError(
            "honcho-ai is required for Honcho integration. "
            "Install it with: pip install honcho-ai"
        )

    logger.info("Initializing Honcho client (host: %s, workspace: %s)", config.host, config.workspace_id)

    _honcho_client = Honcho(
        workspace_id=config.workspace_id,
        api_key=config.api_key,
        environment=config.environment,
    )

    return _honcho_client


def reset_honcho_client() -> None:
    """Reset the Honcho client singleton (useful for testing)."""
    global _honcho_client
    _honcho_client = None
