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
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.profiles import _get_default_hermes_home
from plugins.plugin_utils import SingletonSlot
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from honcho import Honcho

logger = logging.getLogger(__name__)

HOST = "hermes"


def profile_host_key(profile: str | None) -> str:
    """Return the safe Honcho host key for a Hermes profile."""
    if not profile or profile in {"default", "custom"}:
        return HOST
    sanitized = "".join(c if c.isalnum() or c in "_-" else "_" for c in profile).strip("_")
    return f"{HOST}_{sanitized or 'profile'}"


def _host_block(raw: dict, host: str) -> dict:
    """Return host config, accepting legacy dot-form profile host keys."""
    hosts = raw.get("hosts") or {}
    block = hosts.get(host, {})
    if block or not host.startswith(f"{HOST}_"):
        return block
    legacy = f"{HOST}.{host[len(HOST) + 1:]}"
    return hosts.get(legacy, {})


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
        return profile_host_key(profile)
    except Exception:
        pass
    return HOST


def resolve_global_config_path() -> Path:
    """Return the shared Honcho config path for the current HOME."""
    return Path.home() / ".honcho" / "config.json"


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
    default_path = _get_default_hermes_home() / "honcho.json"
    if default_path != local_path and default_path.exists():
        return default_path

    return resolve_global_config_path()


_RECALL_MODE_ALIASES = {"auto": "hybrid"}
_VALID_RECALL_MODES = {"hybrid", "context", "tools"}


def _normalize_recall_mode(val: str) -> str:
    """Normalize legacy recall mode values (e.g. 'auto' → 'hybrid')."""
    val = _RECALL_MODE_ALIASES.get(val, val)
    return val if val in _VALID_RECALL_MODES else "hybrid"


def _resolve_bool(*vals, default: bool) -> bool:
    """Resolve a bool config field: first non-None wins, else default.

    Variadic to support aliased keys (e.g. ``pinUserPeer`` shadowing
    ``pinPeerName`` for backwards compatibility).  Pass values in
    precedence order: caller's preferred alias first, then fallback
    aliases, in (host, root) interleaving as needed.
    """
    for val in vals:
        if val is not None:
            return bool(val)
    return default


def _parse_context_tokens(host_val, root_val) -> int | None:
    """Parse contextTokens: host wins, then root, then None (uncapped)."""
    for val in (host_val, root_val):
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return None


def _parse_int_config(host_val, root_val, default: int) -> int:
    """Parse an integer config: host wins, then root, then default."""
    for val in (host_val, root_val):
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return default


def _parse_string_map(host_obj: dict, root_obj: dict, key: str) -> dict[str, str]:
    """Parse a string-to-string map with host-level whole-map override."""
    source = host_obj[key] if key in host_obj else root_obj.get(key)
    if not isinstance(source, dict):
        return {}

    result: dict[str, str] = {}
    for raw_key, raw_value in source.items():
        alias_key = str(raw_key).strip()
        alias_value = str(raw_value).strip() if raw_value is not None else ""
        if alias_key and alias_value:
            result[alias_key] = alias_value
    return result


def _parse_optional_string(
    host_obj: dict, root_obj: dict, key: str, default: str = ""
) -> str:
    """Parse a string field where host-level empty string can override root."""
    if key in host_obj:
        value = host_obj.get(key)
    else:
        value = root_obj.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _parse_dialectic_depth(host_val, root_val) -> int:
    """Parse dialecticDepth: host wins, then root, then 1. Clamped to 1-3."""
    for val in (host_val, root_val):
        if val is not None:
            try:
                return max(1, min(int(val), 3))
            except (ValueError, TypeError):
                pass
    return 1


_VALID_REASONING_LEVELS = ("minimal", "low", "medium", "high", "max")


def _parse_dialectic_depth_levels(host_val, root_val, depth: int) -> list[str] | None:
    """Parse dialecticDepthLevels: optional array of reasoning levels per pass.

    Returns None when not configured (use proportional defaults).
    When configured, validates each level and truncates/pads to match depth.
    """
    for val in (host_val, root_val):
        if val is not None and isinstance(val, list):
            levels = [
                lvl if lvl in _VALID_REASONING_LEVELS else "low"
                for lvl in val[:depth]
            ]
            # Pad with "low" if array is shorter than depth
            while len(levels) < depth:
                levels.append("low")
            return levels
    return None


# Default HTTP timeout (seconds) applied when no explicit timeout is
# configured via HonchoClientConfig.timeout, honcho.timeout / requestTimeout,
# or HONCHO_TIMEOUT. Honcho calls happen on the post-response path of
# run_conversation; without a cap the agent can block indefinitely when
# the Honcho backend is unreachable, preventing the gateway from
# delivering the already-generated response.
_DEFAULT_HTTP_TIMEOUT = 30.0


def _resolve_optional_float(*values: Any) -> float | None:
    """Return the first non-empty value coerced to a positive float."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


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
    # Optional request timeout in seconds for Honcho SDK HTTP calls
    timeout: float | None = None
    # Identity
    peer_name: str | None = None
    ai_peer: str = "hermes"
    # When True, ``peer_name`` wins over any gateway-supplied runtime
    # identity (Telegram UID, Discord ID, …) when resolving the user peer.
    # This keeps memory unified across platforms for single-user deployments
    # where Honcho's one peer-name is an unambiguous identity — otherwise
    # each platform would fork memory into its own peer (#14984).  Default
    # ``False`` preserves existing multi-user behaviour.
    pin_peer_name: bool = False
    # Map gateway runtime user IDs to stable Honcho user peers. Host-level
    # config replaces the root map as a whole so profiles can intentionally
    # own their identity mappings.
    user_peer_aliases: dict[str, str] = field(default_factory=dict)
    # Optional prefix for unknown gateway runtime user IDs, e.g. "telegram_".
    runtime_peer_prefix: str = ""
    # Toggles
    enabled: bool = False
    save_messages: bool = True
    # Write frequency: "async" (background thread), "turn" (sync per turn),
    # "session" (flush on session end), or int (every N turns)
    write_frequency: str | int = "async"
    # Prefetch budget (None = no cap; set to an integer to bound auto-injected context)
    context_tokens: int | None = None
    # Dialectic (peer.chat) settings
    # reasoning_level: "minimal" | "low" | "medium" | "high" | "max"
    dialectic_reasoning_level: str = "low"
    # When true, the model can override reasoning_level per-call via the
    # honcho_reasoning tool param (agentic). When false, always uses
    # dialecticReasoningLevel and ignores model-provided overrides.
    dialectic_dynamic: bool = True
    # Max chars of dialectic result to inject into Hermes system prompt
    dialectic_max_chars: int = 600
    # Dialectic depth: how many .chat() calls per dialectic cycle (1-3).
    # Depth 1: single call. Depth 2: self-audit + targeted synthesis.
    # Depth 3: self-audit + synthesis + reconciliation.
    dialectic_depth: int = 1
    # Optional per-pass reasoning level override. Array of reasoning levels
    # matching dialectic_depth length. When None, uses proportional defaults
    # derived from dialectic_reasoning_level.
    dialectic_depth_levels: list[str] | None = None
    # When true, the auto-injected dialectic scales reasoning level up on
    # longer queries. See HonchoMemoryProvider for thresholds.
    reasoning_heuristic: bool = True
    # Ceiling for the heuristic-selected reasoning level.
    reasoning_level_cap: str = "high"
    # Honcho API limits — configurable for self-hosted instances
    # Max chars per message sent via add_messages() (Honcho cloud: 25000)
    message_max_chars: int = 25000
    # Max chars for dialectic query input to peer.chat() (Honcho cloud: 10000)
    dialectic_max_input_chars: int = 10000
    # Cost-awareness cadence fields (host-scoped, host-first precedence).
    # These control how often dialectic/context injection fires and the
    # injection frequency mode.  Previously read directly from root-level
    # ``cfg.raw``, bypassing the host-block resolution that the setup
    # wizard writes to — see #35359.
    injection_frequency: str = "every-turn"
    context_cadence: int = 1
    dialectic_cadence: int = 1
    # Recall mode: how memory retrieval works when Honcho is active.
    # "hybrid"  — auto-injected context + Honcho tools available (model decides)
    # "context" — auto-injected context only, Honcho tools removed
    # "tools"   — Honcho tools only, no auto-injected context
    recall_mode: str = "hybrid"
    # Eager init in tools mode — when true, initializes session during
    # initialize() instead of deferring to first tool call
    init_on_session_start: bool = False
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
        timeout = _resolve_optional_float(os.environ.get("HONCHO_TIMEOUT"))
        return cls(
            host=resolved_host,
            workspace_id=workspace_id,
            api_key=api_key,
            environment=os.environ.get("HONCHO_ENVIRONMENT", "production"),
            base_url=base_url,
            timeout=timeout,
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

        host_block = _host_block(raw, resolved_host)
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
        timeout = _resolve_optional_float(
            raw.get("timeout"),
            raw.get("requestTimeout"),
            os.environ.get("HONCHO_TIMEOUT"),
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
            timeout=timeout,
            peer_name=host_block.get("peerName") or raw.get("peerName"),
            ai_peer=ai_peer,
            pin_peer_name=_resolve_bool(
                # ``pinUserPeer`` is the clearer name (the resolver pins
                # the user-side peer to ``peerName``, ignoring runtime
                # identity).  ``pinPeerName`` is the original key from
                # #14984 and stays accepted for backward compatibility.
                # Host-level keys win over root-level; among same-level
                # keys, ``pinUserPeer`` wins over ``pinPeerName``.
                host_block.get("pinUserPeer"),
                host_block.get("pinPeerName"),
                raw.get("pinUserPeer"),
                raw.get("pinPeerName"),
                default=False,
            ),
            user_peer_aliases=_parse_string_map(
                host_block,
                raw,
                "userPeerAliases",
            ),
            runtime_peer_prefix=_parse_optional_string(
                host_block,
                raw,
                "runtimePeerPrefix",
            ),
            enabled=enabled,
            save_messages=save_messages,
            write_frequency=write_frequency,
            context_tokens=_parse_context_tokens(
                host_block.get("contextTokens"),
                raw.get("contextTokens"),
            ),
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
            dialectic_max_chars=_parse_int_config(
                host_block.get("dialecticMaxChars"),
                raw.get("dialecticMaxChars"),
                default=600,
            ),
            dialectic_depth=_parse_dialectic_depth(
                host_block.get("dialecticDepth"),
                raw.get("dialecticDepth"),
            ),
            dialectic_depth_levels=_parse_dialectic_depth_levels(
                host_block.get("dialecticDepthLevels"),
                raw.get("dialecticDepthLevels"),
                depth=_parse_dialectic_depth(host_block.get("dialecticDepth"), raw.get("dialecticDepth")),
            ),
            reasoning_heuristic=_resolve_bool(
                host_block.get("reasoningHeuristic"),
                raw.get("reasoningHeuristic"),
                default=True,
            ),
            reasoning_level_cap=(
                host_block.get("reasoningLevelCap")
                or raw.get("reasoningLevelCap")
                or "high"
            ),
            message_max_chars=_parse_int_config(
                host_block.get("messageMaxChars"),
                raw.get("messageMaxChars"),
                default=25000,
            ),
            dialectic_max_input_chars=_parse_int_config(
                host_block.get("dialecticMaxInputChars"),
                raw.get("dialecticMaxInputChars"),
                default=10000,
            ),
            # Cost-awareness cadence — host-first precedence (#35359).
            # Setup wizard writes these to the host block; previously the
            # provider and status display read only root-level raw,
            # silently ignoring host-scoped overrides.
            injection_frequency=(
                host_block.get("injectionFrequency")
                or raw.get("injectionFrequency")
                or "every-turn"
            ),
            context_cadence=_parse_int_config(
                host_block.get("contextCadence"),
                raw.get("contextCadence"),
                default=1,
            ),
            dialectic_cadence=_parse_int_config(
                host_block.get("dialecticCadence"),
                raw.get("dialecticCadence"),
                default=1,
            ),
            recall_mode=_normalize_recall_mode(
                host_block.get("recallMode")
                or raw.get("recallMode")
                or "hybrid"
            ),
            init_on_session_start=_resolve_bool(
                host_block.get("initOnSessionStart"),
                raw.get("initOnSessionStart"),
                default=False,
            ),
            # Migration guard: existing configs without an explicit
            # observationMode keep the old "unified" default so users
            # aren't silently switched to full bidirectional observation.
            # New installations (no host block, no credentials) get
            # "directional" (all observations on) as the new default.
            observation_mode=_normalize_observation_mode(
                host_block.get("observationMode")
                or raw.get("observationMode")
                or ("unified" if _explicitly_configured else "directional")
            ),
            **_resolve_observation(
                _normalize_observation_mode(
                    host_block.get("observationMode")
                    or raw.get("observationMode")
                    or ("unified" if _explicitly_configured else "directional")
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
                stdin=subprocess.DEVNULL,
            )
            if root.returncode == 0:
                return Path(root.stdout.strip()).name
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    # Honcho enforces a 100-char limit on session IDs. Long gateway session keys
    # (Matrix "!room:server" + thread event IDs, Telegram supergroup reply
    # chains, Slack thread IDs with long workspace prefixes) can overflow this
    # limit after sanitization; the Honcho API then rejects every call for that
    # session with "session_id too long". See issue #13868.
    _HONCHO_SESSION_ID_MAX_LEN = 100
    _HONCHO_SESSION_ID_HASH_LEN = 8

    @classmethod
    def _enforce_session_id_limit(cls, sanitized: str, original: str) -> str:
        """Truncate a sanitized session ID to Honcho's 100-char limit.

        The common case (short keys) short-circuits with no modification.
        For over-limit keys, keep a prefix of the sanitized ID and append a
        deterministic ``-<sha256 prefix>`` suffix so two distinct long keys
        that share a leading segment don't collide onto the same truncated ID.
        The hash is taken over the *original* pre-sanitization key, so two
        inputs that sanitize to the same string still collide intentionally
        (same logical session), but two inputs that only share a prefix do not.
        """
        max_len = cls._HONCHO_SESSION_ID_MAX_LEN
        if len(sanitized) <= max_len:
            return sanitized

        hash_len = cls._HONCHO_SESSION_ID_HASH_LEN
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:hash_len]
        # max_len - hash_len - 1 (for the '-' separator) chars of the sanitized
        # prefix, then '-<hash>'. Strip any trailing hyphen from the prefix so
        # the result doesn't double up on separators.
        prefix_len = max_len - hash_len - 1
        prefix = sanitized[:prefix_len].rstrip("-")
        return f"{prefix}-{digest}"

    def resolve_session_name(
        self,
        cwd: str | None = None,
        session_title: str | None = None,
        session_id: str | None = None,
        gateway_session_key: str | None = None,
    ) -> str | None:
        """Resolve Honcho session name.

        Resolution order:
          1. Gateway session key (stable per-chat identifier from gateway platforms)
          2. per-session strategy — Hermes session_id ({timestamp}_{hex}); authoritative,
             so a generated title never remaps a live conversation
          3. Manual directory override from sessions map
          4. Hermes session title (from /title command; non-per-session)
          5. per-repo strategy — git repo root directory name
          6. per-directory strategy — directory basename
          7. global strategy — workspace name
        """
        import re

        if not cwd:
            cwd = os.getcwd()

        # Gateway per-chat key wins everywhere — gateways (telegram/discord/…)
        # need per-chat isolation no cwd/strategy name can provide.
        if gateway_session_key:
            sanitized = re.sub(r'[^a-zA-Z0-9_-]+', '-', gateway_session_key).strip('-')
            if sanitized:
                return self._enforce_session_id_limit(sanitized, gateway_session_key)

        # per-session: the run's session_id IS the identity — resolve before the
        # cwd map / title so an auto-generated title can't remap a live
        # conversation onto a second Honcho session mid-stream.
        if self.session_strategy == "per-session" and session_id:
            if self.session_peer_prefix and self.peer_name:
                return f"{self.peer_name}-{session_id}"
            return session_id

        # Manual override (cwd → name), for non-per-session strategies.
        manual = self.sessions.get(cwd)
        if manual:
            return manual

        # /title mid-session remap (non-per-session).
        if session_title:
            sanitized = re.sub(r'[^a-zA-Z0-9_-]+', '-', session_title).strip('-')
            if sanitized:
                if self.session_peer_prefix and self.peer_name:
                    return f"{self.peer_name}-{sanitized}"
                return sanitized

        # per-repo: one Honcho session per git repository
        if self.session_strategy == "per-repo":
            base = self._git_repo_name(cwd) or Path(cwd).name
            if self.session_peer_prefix and self.peer_name:
                return f"{self.peer_name}-{base}"
            return base

        # per-directory: one Honcho session per working directory (default)
        if self.session_strategy in {"per-directory", "per-session"}:
            base = Path(cwd).name
            if self.session_peer_prefix and self.peer_name:
                return f"{self.peer_name}-{base}"
            return base

        # global: single session across all directories
        return self.workspace_id


_honcho_client_slot: SingletonSlot = SingletonSlot()


def _apply_fresh_oauth_token(config: HonchoClientConfig) -> None:
    """Refresh a near-expiry OAuth grant and point ``config.api_key`` at it.

    No-op for static API keys or when refresh fails (fail-open: the stale token
    is left in place and the existing 401 handling degrades gracefully).
    """
    try:
        from plugins.memory.honcho import oauth

        token, _ = oauth.ensure_fresh_token(resolve_config_path(), config.host)
        if token:
            config.api_key = token
    except Exception:
        logger.warning("Honcho OAuth pre-build refresh failed", exc_info=True)


def _refresh_cached_oauth(client: "Honcho", config: HonchoClientConfig | None) -> None:
    """Rotate the cached client's Bearer in place when its OAuth token is stale.

    If the SDK shape changed and the in-place rotation can't apply, the slot is
    reset so the next acquisition rebuilds with the fresh token.
    """
    try:
        from plugins.memory.honcho import oauth

        host = config.host if config is not None else resolve_active_host()
        token, refreshed = oauth.ensure_fresh_token(resolve_config_path(), host)
        if refreshed and token and not oauth.apply_token_to_client(client, token):
            _honcho_client_slot.reset()
    except Exception:
        logger.warning("Honcho OAuth cached refresh failed", exc_info=True)


def get_honcho_client(config: HonchoClientConfig | None = None) -> Honcho:
    """Get or create the Honcho client singleton.

    When no config is provided, attempts to load ~/.honcho/config.json
    first, falling back to environment variables.

    Thread-safe: the client is built exactly once even under concurrent
    first calls (double-checked locking via ``SingletonSlot``), so racing
    threads can't each construct a client and leak the loser's connection.
    """
    cached = _honcho_client_slot.peek()
    if cached is not None:
        _refresh_cached_oauth(cached, config)
        return cached

    if config is None:
        config = HonchoClientConfig.from_global_config()

    # Refresh a near-expiry OAuth grant before the first build so the client
    # starts with a live access token rather than 401ing an hour in.
    _apply_fresh_oauth_token(config)

    if not config.api_key and not config.base_url:
        raise ValueError(
            "Honcho API key not found. "
            "Get your API key at https://app.honcho.dev, "
            "then run 'hermes honcho setup' or set HONCHO_API_KEY. "
            "For local instances, set HONCHO_BASE_URL instead."
        )

    # Everything below is the expensive part the issue flags: lazy SDK
    # install, config resolution, and client construction. Run it inside the
    # slot's factory so it executes exactly once even when several threads
    # race the first call — the slot's double-checked lock serializes them and
    # the losers get the winner's client instead of building their own.
    def _build() -> "Honcho":
        # Lazy-install the honcho SDK on demand. ensure() honors
        # security.allow_lazy_installs (default true). On failure we surface
        # the original ImportError-shape message so existing callers still get
        # the "go run hermes honcho setup" hint they used to.
        try:
            from tools.lazy_deps import FeatureUnavailable, ensure as _lazy_ensure
            _lazy_ensure("memory.honcho", prompt=False)
        except ImportError:
            # lazy_deps module missing — fall through to the raw import below.
            pass
        except Exception:
            # FeatureUnavailable or unexpected error. Don't crash here; let the
            # actual import attempt produce the canonical error message.
            pass

        try:
            from honcho import Honcho
        except ImportError:
            raise ImportError(
                "honcho-ai is required for Honcho integration. "
                "Install it with: pip install honcho-ai  "
                "(or run `hermes honcho setup` to configure)."
            )

        # Allow config.yaml honcho.base_url to override the SDK's environment
        # mapping, enabling remote self-hosted Honcho deployments without
        # requiring the server to live on localhost.
        resolved_base_url = config.base_url
        resolved_timeout = config.timeout
        if not resolved_base_url or resolved_timeout is None:
            try:
                from hermes_cli.config import load_config
                hermes_cfg = load_config()
                honcho_cfg = hermes_cfg.get("honcho", {})
                if isinstance(honcho_cfg, dict):
                    if not resolved_base_url:
                        resolved_base_url = honcho_cfg.get("base_url", "").strip() or None
                    if resolved_timeout is None:
                        resolved_timeout = _resolve_optional_float(
                            honcho_cfg.get("timeout"),
                            honcho_cfg.get("request_timeout"),
                        )
            except Exception:
                pass

        # Fall back to the default so an unconfigured install cannot hang
        # indefinitely on a stalled Honcho request.
        if resolved_timeout is None:
            resolved_timeout = _DEFAULT_HTTP_TIMEOUT

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
            # Check if the host block has its own apiKey (explicit local auth).
            # Auth-skipping is loopback-only: a stored key is likely a cloud key
            # that would break a no-auth local server, so we substitute the SDK's
            # required-non-empty placeholder unless the host block opts in.
            _raw = config.raw or {}
            _host_block = (_raw.get("hosts") or {}).get(config.host, {})
            _host_has_key = bool(_host_block.get("apiKey"))
            effective_api_key = config.api_key if _host_has_key else "local"
        else:
            effective_api_key = config.api_key

        # The Honcho SDK's route builders (e.g. routes.workspaces()) already
        # include the version prefix (e.g. "/v3/workspaces").  When a user-supplied
        # base_url already ends in a version segment (e.g.
        # "http://localhost:38000/v3", "https://honcho.my.ts.net/v3"), concatenating
        # the two produces "/v3/v3/workspaces" → 404 on every call.  This is a pure
        # routing concern independent of host, so strip a trailing version segment
        # from ANY base_url — loopback, LAN, custom domain, or cloud alike.  The
        # SDK then appends its own versioned paths correctly.
        if resolved_base_url:
            import re as _re
            resolved_base_url = _re.sub(r"/v\d+/*$", "", resolved_base_url).rstrip("/")

        kwargs: dict = {
            "workspace_id": config.workspace_id,
            "api_key": effective_api_key,
            "environment": config.environment,
        }
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if resolved_timeout is not None:
            kwargs["timeout"] = resolved_timeout

        return Honcho(**kwargs)

    return _honcho_client_slot.get(_build)


def reset_honcho_client() -> None:
    """Reset the Honcho client singleton (useful for testing)."""
    _honcho_client_slot.reset()
