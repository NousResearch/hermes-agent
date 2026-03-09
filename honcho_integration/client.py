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


def _resolve_memory_mode(
    global_val: str | dict,
    host_val: str | dict | None,
) -> dict:
    """Parse memoryMode (string or object) into memory_mode + peer_memory_modes.

    Resolution order: host-level wins over global.
    String form:  applies as the default for all peers.
    Object form:  { "default": "hybrid", "hermes": "honcho", ... }
                  "default" key sets the fallback; other keys are per-peer overrides.
    """
    # Pick the winning value (host beats global)
    val = host_val if host_val is not None else global_val

    if isinstance(val, dict):
        default = val.get("default", "hybrid")
        overrides = {k: v for k, v in val.items() if k != "default"}
    else:
        default = str(val) if val else "hybrid"
        overrides = {}

    return {"memory_mode": default, "peer_memory_modes": overrides}


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
    # memoryMode: default for all peers. "hybrid" / "honcho" / "local"
    memory_mode: str = "hybrid"
    # Per-peer overrides — any named Honcho peer. Override memory_mode when set.
    # Config object form: "memoryMode": { "default": "hybrid", "hermes": "honcho" }
    peer_memory_modes: dict[str, str] = field(default_factory=dict)

    def peer_memory_mode(self, peer_name: str) -> str:
        """Return the effective memory mode for a named peer.

        Resolution: per-peer override → global memory_mode default.
        """
        return self.peer_memory_modes.get(peer_name, self.memory_mode)
    # Write frequency: "async" (background thread), "turn" (sync per turn),
    # "session" (flush on session end), or int (every N turns)
    write_frequency: str | int = "async"
    # Prefetch budget
    context_tokens: int | None = None
    # Dialectic (peer.chat) settings
    # reasoning_level: "minimal" | "low" | "medium" | "high" | "max"
    # Used as the default; prefetch_dialectic may bump it dynamically.
    dialectic_reasoning_level: str = "low"
    # Max chars of dialectic result to inject into Hermes system prompt
    dialectic_max_chars: int = 600
    # Recall mode: how memory retrieval works when Honcho is active.
    # "hybrid"  — pre-warmed context + memory tools available (model decides)
    # "context" — pre-warmed context only, honcho memory tools removed
    # "tools"   — no pre-loaded context, rely on tool calls only
    recall_mode: str = "hybrid"
    # Session resolution
    session_strategy: str = "per-session"
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

        api_key = raw.get("apiKey") or os.environ.get("HONCHO_API_KEY")

        # Auto-enable when API key is present (unless explicitly disabled)
        # This matches user expectations: setting an API key should activate the feature.
        explicit_enabled = raw.get("enabled")
        if explicit_enabled is None:
            # Not explicitly set in config -> auto-enable if API key exists
            enabled = bool(api_key)
        else:
            # Respect explicit setting
            enabled = explicit_enabled

        # write_frequency: accept int or string
        raw_wf = (
            host_block.get("writeFrequency")
            or raw.get("writeFrequency")
            or "async"
        )
        try:
            write_frequency: str | int = int(raw_wf)
        except (TypeError, ValueError):
            write_frequency = str(raw_wf)

        return cls(
            host=host,
            workspace_id=workspace,
            api_key=api_key,
            environment=raw.get("environment", "production"),
            peer_name=raw.get("peerName"),
            ai_peer=ai_peer,
            linked_hosts=linked_hosts,
            enabled=enabled,
            save_messages=raw.get("saveMessages", True),
            **_resolve_memory_mode(
                raw.get("memoryMode", "hybrid"),
                host_block.get("memoryMode"),
            ),
            write_frequency=write_frequency,
            context_tokens=host_block.get("contextTokens") or raw.get("contextTokens"),
            dialectic_reasoning_level=(
                host_block.get("dialecticReasoningLevel")
                or raw.get("dialecticReasoningLevel")
                or "low"
            ),
            dialectic_max_chars=int(
                host_block.get("dialecticMaxChars")
                or raw.get("dialecticMaxChars")
                or 600
            ),
            recall_mode=(
                host_block.get("recallMode")
                or raw.get("recallMode")
                or "hybrid"
            ),
            session_strategy=raw.get("sessionStrategy", "per-session"),
            session_peer_prefix=raw.get("sessionPeerPrefix", False),
            sessions=raw.get("sessions", {}),
            raw=raw,
        )

    @staticmethod
    def _git_repo_name(cwd: str) -> str | None:
        """Return the git repo root directory name, or None if not in a repo."""
        import subprocess

        try:
            root = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, cwd=cwd, timeout=5,
            )
            if root.returncode == 0:
                return Path(root.stdout.strip()).name
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    def resolve_session_name(
        self,
        cwd: str | None = None,
        session_title: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        """Resolve Honcho session name.

        Resolution order:
          1. Manual directory override from sessions map
          2. Hermes session title (from /title command)
          3. per-session strategy — Hermes session_id ({timestamp}_{hex})
          4. per-repo strategy — git repo root directory name
          5. per-directory strategy — directory basename
          6. global strategy — workspace name
        """
        import re

        if not cwd:
            cwd = os.getcwd()

        # Manual override always wins
        manual = self.sessions.get(cwd)
        if manual:
            return manual

        # /title mid-session remap
        if session_title:
            sanitized = re.sub(r'[^a-zA-Z0-9_-]', '-', session_title).strip('-')
            if sanitized:
                if self.session_peer_prefix and self.peer_name:
                    return f"{self.peer_name}-{sanitized}"
                return sanitized

        # per-session: inherit Hermes session_id (new Honcho session each run)
        if self.session_strategy == "per-session" and session_id:
            if self.session_peer_prefix and self.peer_name:
                return f"{self.peer_name}-{session_id}"
            return session_id

        # per-repo: one Honcho session per git repository
        if self.session_strategy == "per-repo":
            base = self._git_repo_name(cwd) or Path(cwd).name
            if self.session_peer_prefix and self.peer_name:
                return f"{self.peer_name}-{base}"
            return base

        # per-directory: one Honcho session per working directory
        if self.session_strategy in ("per-directory", "per-session"):
            base = Path(cwd).name
            if self.session_peer_prefix and self.peer_name:
                return f"{self.peer_name}-{base}"
            return base

        # global: single session across all directories
        return self.workspace_id

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
            "Honcho API key not found. "
            "Get your API key at https://app.honcho.dev, "
            "then run 'hermes honcho setup' or set HONCHO_API_KEY."
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
