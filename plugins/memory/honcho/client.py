"""Honcho client initialization and configuration.

Resolution order for config file:
  1. $HERMES_HOME/honcho.json  (instance-local, enables isolated Hermes instances)
  2. ~/.honcho/config.json     (global, shared across all Honcho-enabled apps)
  3. Environment variables     (HONCHO_API_KEY, HONCHO_ENVIRONMENT)

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

from hermes_constants import get_hermes_home
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from honcho import Honcho

logger = logging.getLogger(__name__)

GLOBAL_CONFIG_PATH = Path.home() / ".honcho" / "config.json"
HOST = "hermes"


def resolve_active_host() -> str:
    """Derive the Honcho host key from the active Hermes profile.

    Resolution order:
      1. HERMES_HONCHO_HOST env var (explicit override)
      2. Active profile name via profiles system -> ``hermes.<profile>``
      3. Fallback: ``"hermes"`` (default profile)
    """
    explicit = os.environ.get("HERMES_HONCHO_HOST", "").strip()
    if explicit:
        return explicit

    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name()
        if profile and profile not in ("default", "custom"):
            return f"{HOST}.{profile}"
    except Exception:
        pass
    return HOST


def resolve_config_path() -> Path:
    """Return the active Honcho config path.

    Resolution order:
      1. $HERMES_HOME/honcho.json      (profile-local, if it exists)
      2. ~/.hermes/honcho.json          (default profile — shared host blocks live here)
      3. ~/.honcho/config.json          (global, cross-app interop)

    Returns the global path if none exist (for first-time setup writes).
    """
    local_path = get_hermes_home() / "honcho.json"
    if local_path.exists():
        return local_path

    # Default profile's config — host blocks accumulate here via setup/clone
    default_path = Path.home() / ".hermes" / "honcho.json"
    if default_path != local_path and default_path.exists():
        return default_path

    return GLOBAL_CONFIG_PATH


_RECALL_MODE_ALIASES = {"auto": "hybrid"}
_VALID_RECALL_MODES = {"hybrid", "context", "tools"}


def _normalize_recall_mode(val: str) -> str:
    """Normalize legacy recall mode values (e.g. 'auto' → 'hybrid')."""
    val = _RECALL_MODE_ALIASES.get(val, val)
    return val if val in _VALID_RECALL_MODES else "hybrid"


def _resolve_bool(host_val, root_val, *, default: bool) -> bool:
    """Resolve a bool config field: host wins, then root, then default."""
    if host_val is not None:
        return bool(host_val)
    if root_val is not None:
        return bool(root_val)
    return default


_VALID_OBSERVATION_MODES = {"unified", "directional"}
_OBSERVATION_MODE_ALIASES = {"shared": "unified", "separate": "directional", "cross": "directional"}


def _normalize_observation_mode(val: str) -> str:
    """Normalize observation mode values."""
    val = _OBSERVATION_MODE_ALIASES.get(val, val)
    return val if val in _VALID_OBSERVATION_MODES else "directional"


# Observation presets — granular booleans derived from legacy string mode.
# Explicit per-peer config always wins over presets.
_OBSERVATION_PRESETS = {
    "directional": {
        "user_observe_me": True, "user_observe_others": True,
        "ai_observe_me": True, "ai_observe_others": True,
    },
    "unified": {
        "user_observe_me": True, "user_observe_others": False,
        "ai_observe_me": False, "ai_observe_others": True,
    },
}


def _resolve_observation(
    mode: str,
    observation_obj: dict | None,
) -> dict:
    """Resolve per-peer observation booleans.

    Config forms:
      String shorthand:  ``"observationMode": "directional"``
      Granular object:   ``"observation": {"user": {"observeMe": true, "observeOthers": true},
                                           "ai": {"observeMe": true, "observeOthers": false}}``

    Granular fields override preset defaults.
    """
    preset = _OBSERVATION_PRESETS.get(mode, _OBSERVATION_PRESETS["directional"])
    if not observation_obj or not isinstance(observation_obj, dict):
        return dict(preset)

    user_block = observation_obj.get("user") or {}
    ai_block = observation_obj.get("ai") or {}

    return {
        "user_observe_me": user_block.get("observeMe", preset["user_observe_me"]),
        "user_observe_others": user_block.get("observeOthers", preset["user_observe_others"]),
        "ai_observe_me": ai_block.get("observeMe", preset["ai_observe_me"]),
        "ai_observe_others": ai_block.get("observeOthers", preset["ai_observe_others"]),
    }





@dataclass
class HonchoClientConfig:
    """Configuration for Honcho client, resolved for a specific host."""

    host: str = HOST
    workspace_id: str = "hermes"
    api_key: str | None = None
    environment: str = "production"
    # Optional base URL for self-hosted Honcho (overrides environment mapping)
    base_url: str | None = None
    # Identity
    peer_name: str | None = None
    ai_peer: str = "hermes"
    # Toggles
    enabled: bool = False
    save_messages: bool = True
    # Write frequency: "async" (background thread), "turn" (sync per turn),
    # "session" (flush on session end), or int (every N turns)
    write_frequency: str | int = "async"
    # Prefetch budget
    context_tokens: int | None = None
    # Dialectic (peer.chat) settings
    # reasoning_level: "minimal" | "low" | "medium" | "high" | "max"
    dialectic_reasoning_level: str = "low"
    # dynamic: auto-bump reasoning level based on query length
    #   true  — low->medium (120+ chars), low->high (400+ chars), capped at "high"
    #   false — always use dialecticReasoningLevel as-is
    dialectic_dynamic: bool = True
    # Max chars of dialectic result to inject into Hermes system prompt
    dialectic_max_chars: int = 600
    # Honcho API limits — configurable for self-hosted instances
    # Max chars per message sent via add_messages() (Honcho cloud: 25000)
    message_max_chars: int = 25000
    # Max chars for dialectic query input to peer.chat() (Honcho cloud: 10000)
    dialectic_max_input_chars: int = 10000
    # Recall mode: how memory retrieval works when Honcho is active.
    # "hybrid"  — auto-injected context + Honcho tools available (model decides)
    # "context" — auto-injected context only, Honcho tools removed
    # "tools"   — Honcho tools only, no auto-injected context
    recall_mode: str = "hybrid"
    # Observation mode: legacy string shorthand ("directional" or "unified").
    # Kept for backward compat; granular per-peer booleans below are preferred.
    observation_mode: str = "directional"
    # Per-peer observation booleans — maps 1:1 to Honcho's SessionPeerConfig.
    # Resolved from "observation" object in config, falling back to observation_mode preset.
    user_observe_me: bool = True
    user_observe_others: bool = True
    ai_observe_me: bool = True
    ai_observe_others: bool = True
    # Session resolution
    session_strategy: str = "per-directory"
    session_peer_prefix: bool = False
    sessions: dict[str, str] = field(default_factory=dict)
    # Raw global config for anything else consumers need
    raw: dict[str, Any] = field(default_factory=dict)
    # True when Honcho was explicitly configured for this host (hosts.hermes
    # block exists or enabled was set explicitly), vs auto-enabled from a
    # stray HONCHO_API_KEY env var.
    explicitly_configured: bool = False

    @classmethod
    def from_env(
        cls,
        workspace_id: str = "hermes",
        host: str | None = None,
    ) -> HonchoClientConfig:
        """Create config from environment variables (fallback)."""
        resolved_host = host or resolve_active_host()
        api_key = os.environ.get("HONCHO_API_KEY")
        base_url = os.environ.get("HONCHO_BASE_URL", "").strip() or None
        return cls(
            host=resolved_host,
            workspace_id=workspace_id,
            api_key=api_key,
            environment=os.environ.get("HONCHO_ENVIRONMENT", "production"),
            base_url=base_url,
            ai_peer=resolved_host,
            enabled=bool(api_key or base_url),
        )

    @classmethod
    def from_global_config(
        cls,
        host: str | None = None,
        config_path: Path | None = None,
    ) -> HonchoClientConfig:
        """Create config from the resolved Honcho config path.

        Resolution: $HERMES_HOME/honcho.json -> ~/.honcho/config.json -> env vars.
        When host is None, derives it from the active Hermes profile.
        """
        resolved_host = host or resolve_active_host()
        path = config_path or resolve_config_path()
        if not path.exists():
            logger.debug("No global Honcho config at %s, falling back to env", path)
            return cls.from_env(host=resolved_host)

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s, falling back to env", path, e)
            return cls.from_env(host=resolved_host)

        host_block = (raw.get("hosts") or {}).get(resolved_host, {})
        # A hosts.hermes block or explicit enabled flag means the user
        # intentionally configured Honcho for this host.
        _explicitly_configured = bool(host_block) or raw.get("enabled") is True

        # Explicit host block fields win, then flat/global, then defaults
        workspace = (
            host_block.get("workspace")
            or raw.get("workspace")
            or resolved_host
        )
        ai_peer = (
            host_block.get("aiPeer")
            or raw.get("aiPeer")
            or resolved_host
        )
        api_key = (
            host_block.get("apiKey")
            or raw.get("apiKey")
            or os.environ.get("HONCHO_API_KEY")
        )

        environment = (
            host_block.get("environment")
            or raw.get("environment", "production")
        )

        base_url = (
            raw.get("baseUrl")
            or raw.get("base_url")
            or os.environ.get("HONCHO_BASE_URL", "").strip()
            or None
        )

        # Auto-enable when API key or base_url is present (unless explicitly disabled)
        # Host-level enabled wins, then root-level, then auto-enable if key/url exists.
        host_enabled = host_block.get("enabled")
        root_enabled = raw.get("enabled")
        if host_enabled is not None:
            enabled = host_enabled
        elif root_enabled is not None:
            enabled = root_enabled
        else:
            # Not explicitly set anywhere -> auto-enable if API key or base_url exists
            enabled = bool(api_key or base_url)

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

        # saveMessages: host wins (None-aware since False is valid)
        host_save = host_block.get("saveMessages")
        save_messages = host_save if host_save is not None else raw.get("saveMessages", True)

        # sessionStrategy / sessionPeerPrefix: host first, root fallback
        session_strategy = (
            host_block.get("sessionStrategy")
            or raw.get("sessionStrategy", "per-directory")
        )
        host_prefix = host_block.get("sessionPeerPrefix")
        session_peer_prefix = (
            host_prefix if host_prefix is not None
            else raw.get("sessionPeerPrefix", False)
        )

        return cls(
            host=resolved_host,
            workspace_id=workspace,
            api_key=api_key,
            environment=environment,
            base_url=base_url,
            peer_name=host_block.get("peerName") or raw.get("peerName"),
            ai_peer=ai_peer,
            enabled=enabled,
            save_messages=save_messages,
            write_frequency=write_frequency,
            context_tokens=host_block.get("contextTokens") or raw.get("contextTokens"),
            dialectic_reasoning_level=(
                host_block.get("dialecticReasoningLevel")
                or raw.get("dialecticReasoningLevel")
                or "low"
            ),
            dialectic_dynamic=_resolve_bool(
                host_block.get("dialecticDynamic"),
                raw.get("dialecticDynamic"),
                default=True,
            ),
            dialectic_max_chars=int(
                host_block.get("dialecticMaxChars")
                or raw.get("dialecticMaxChars")
                or 600
            ),
            message_max_chars=int(
                host_block.get("messageMaxChars")
                or raw.get("messageMaxChars")
                or 25000
            ),
            dialectic_max_input_chars=int(
                host_block.get("dialecticMaxInputChars")
                or raw.get("dialecticMaxInputChars")
                or 10000
            ),
            recall_mode=_normalize_recall_mode(
                host_block.get("recallMode")
                or raw.get("recallMode")
                or "hybrid"
            ),
            observation_mode=_normalize_observation_mode(
                host_block.get("observationMode")
                or raw.get("observationMode")
                or "directional"
            ),
            **_resolve_observation(
                _normalize_observation_mode(
                    host_block.get("observationMode")
                    or raw.get("observationMode")
                    or "directional"
                ),
                host_block.get("observation") or raw.get("observation"),
            ),
            session_strategy=session_strategy,
            session_peer_prefix=session_peer_prefix,
            sessions=raw.get("sessions", {}),
            raw=raw,
            explicitly_configured=_explicitly_configured,
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

        # per-directory: one Honcho session per working directory (default)
        if self.session_strategy in ("per-directory", "per-session"):
            base = Path(cwd).name
            if self.session_peer_prefix and self.peer_name:
                return f"{self.peer_name}-{base}"
            return base

        # global: single session across all directories
        return self.workspace_id


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

    if not config.api_key and not config.base_url:
        raise ValueError(
            "Honcho API key not found. "
            "Get your API key at https://app.honcho.dev, "
            "then run 'hermes honcho setup' or set HONCHO_API_KEY. "
            "For local instances, set HONCHO_BASE_URL instead."
        )

    try:
        from honcho import Honcho
    except ImportError:
        raise ImportError(
            "honcho-ai is required for Honcho integration. "
            "Install it with: pip install honcho-ai"
        )

    # Allow config.yaml honcho.base_url to override the SDK's environment
    # mapping, enabling remote self-hosted Honcho deployments without
    # requiring the server to live on localhost.
    resolved_base_url = config.base_url
    if not resolved_base_url:
        try:
            from hermes_cli.config import load_config
            hermes_cfg = load_config()
            honcho_cfg = hermes_cfg.get("honcho", {})
            if isinstance(honcho_cfg, dict):
                resolved_base_url = honcho_cfg.get("base_url", "").strip() or None
        except Exception:
            pass

    if resolved_base_url:
        logger.info("Initializing Honcho client (base_url: %s, workspace: %s)", resolved_base_url, config.workspace_id)
    else:
        logger.info("Initializing Honcho client (host: %s, workspace: %s)", config.host, config.workspace_id)

    # Local Honcho instances don't require an API key, but the SDK
    # expects a non-empty string.  Use a placeholder for local URLs.
    # For local: only use config.api_key if the host block explicitly
    # sets apiKey (meaning the user wants local auth). Otherwise skip
    # the stored key -- it's likely a cloud key that would break local.
    _is_local = resolved_base_url and (
        "localhost" in resolved_base_url
        or "127.0.0.1" in resolved_base_url
        or "::1" in resolved_base_url
    )
    if _is_local:
        # Check if the host block has its own apiKey (explicit local auth)
        _raw = config.raw or {}
        _host_block = (_raw.get("hosts") or {}).get(config.host, {})
        _host_has_key = bool(_host_block.get("apiKey"))
        effective_api_key = config.api_key if _host_has_key else "local"
    else:
        effective_api_key = config.api_key

    kwargs: dict = {
        "workspace_id": config.workspace_id,
        "api_key": effective_api_key,
        "environment": config.environment,
    }
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url

    _honcho_client = Honcho(**kwargs)

    return _honcho_client


def reset_honcho_client() -> None:
    """Reset the Honcho client singleton (useful for testing)."""
    global _honcho_client
    _honcho_client = None
