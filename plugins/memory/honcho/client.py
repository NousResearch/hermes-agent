"""Honcho client construction and ``HonchoClientConfig`` resolution.

Config file resolution: $HERMES_HOME/honcho.json -> ~/.honcho/config.json -> env vars
(HONCHO_API_KEY, HONCHO_ENVIRONMENT). Within a file, host-block fields win over
flat/global fields, which win over defaults.
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json
import logging
import os
# --- per-identity client cache ------------------------------------------- One slot per client identity,
# replacing the single process-wide slot that pinned the first profile's workspace and bearer for every
# later profile in multi-profile processes (#69123 multiplexed gateway, #74065 dashboard). The legacy names
# above are retained only for reset bookkeeping.
import threading as _threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

from agent.secret_scope import get_secret
from hermes_cli.profiles import _get_default_hermes_home
from hermes_constants import get_hermes_home

from plugins.memory.honcho.client_cache import (
    _DEFAULT_HTTP_TIMEOUT, _client_cache_key, _client_slots, _client_slots_lock,
    _honcho_json_timeout_memo, _refresh_oauth, _slot_for,
)

if TYPE_CHECKING:
    from honcho import Honcho

logger = logging.getLogger(__name__)

HOST = "hermes"


def _sanitize_url(url: str | None) -> str | None:
    """``url``, or None (with a warning) if it carries non-printable ASCII: a stray terminal
    escape in a pasted URL otherwise makes the SDK raise at client construction."""
    if url is None or all(0x20 <= ord(c) < 0x7F for c in url):
        return url
    logger.warning("Honcho base_url contains non-printable characters and will be ignored: %r", url)
    return None


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
    return hosts.get(f"{HOST}.{host[len(HOST) + 1:]}", {})


def resolve_active_host() -> str:
    """Honcho host key: HERMES_HONCHO_HOST env, else the active profile. The config's
    ``defaultHost`` is honored only for the default profile so named profiles stay isolated."""
    explicit = os.environ.get("HERMES_HONCHO_HOST", "").strip()
    if explicit:
        return explicit
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile_host = profile_host_key(get_active_profile_name())
    except Exception:
        profile_host = HOST
    if profile_host == HOST:
        try:
            default_host = str(_read_config(resolve_config_path()).get("defaultHost", "")).strip()
        except Exception:
            default_host = ""
        if default_host:
            return default_host
    return profile_host


def _read_config(path: Path) -> dict:
    """Parse a honcho.json; {} when absent (parse/OS errors propagate)."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def resolve_global_config_path() -> Path:
    """Return the shared Honcho config path for the current HOME."""
    return Path.home() / ".honcho" / "config.json"


def resolve_config_path() -> Path:
    """Active Honcho config path: $HERMES_HOME/honcho.json -> default profile's honcho.json
    (host blocks accumulate there via setup/clone) -> ~/.honcho/config.json (also the
    first-time-setup write target when nothing exists)."""
    local_path = get_hermes_home() / "honcho.json"
    if local_path.exists():
        return local_path
    default_path = _get_default_hermes_home() / "honcho.json"
    if default_path != local_path and default_path.exists():
        return default_path
    return resolve_global_config_path()


# --- config coercion -------------------------------------------------------

_RECALL_MODES = ({"auto": "hybrid"}, {"hybrid", "context", "tools"}, "hybrid")
_OBSERVATION_MODES = (
    {"shared": "unified", "separate": "directional", "cross": "directional"},
    {"unified", "directional"},
    "directional",
)
_VALID_REASONING_LEVELS = ("minimal", "low", "medium", "high", "max")

# Granular observation booleans derived from the legacy string mode; explicit
# per-peer config always wins over these presets.
_OBSERVATION_PRESETS = {
    "directional": {"user_observe_me": True, "user_observe_others": True, "ai_observe_me": True, "ai_observe_others": True},
    "unified": {"user_observe_me": True, "user_observe_others": False, "ai_observe_me": False, "ai_observe_others": True},
}


def _normalize_choice(val: str, spec: tuple[dict, set, str]) -> str:
    """Map aliases, then fall back to the default for unknown values."""
    aliases, valid, default = spec
    val = aliases.get(val, val)
    return val if val in valid else default


def _first_set(*vals, default):
    """First non-None value, else default."""
    return next((val for val in vals if val is not None), default)


def _first_parsed(vals, caster: Callable[[Any], Any], default):
    """First non-None value that ``caster`` accepts, else default."""
    for val in (v for v in vals if v is not None):
        try:
            return caster(val)
        except (ValueError, TypeError):
            pass
    return default


def _positive_float(value: Any) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(value)
    return parsed


def _resolve_optional_float(*values: Any) -> float | None:
    """First value (blank strings included) that parses as a positive float, else None."""
    return _first_parsed(values, _positive_float, None)


def _parse_dialectic_depth_levels(vals, depth: int) -> list[str] | None:
    """First list value, validated and padded/truncated to ``depth``; None if unset."""
    for val in vals:
        if isinstance(val, list):
            levels = [lvl if lvl in _VALID_REASONING_LEVELS else "low" for lvl in val[:depth]]
            return levels + ["low"] * (depth - len(levels))
    return None


def _resolve_observation(mode: str, observation_obj: dict | None) -> dict:
    """Per-peer observation booleans: ``observation`` object fields override the mode preset."""
    preset = _OBSERVATION_PRESETS.get(mode, _OBSERVATION_PRESETS["directional"])
    if not isinstance(observation_obj, dict) or not observation_obj:
        return dict(preset)
    return {
        f"{kind}_observe_{who}": (observation_obj.get(kind) or {}).get(json_key, preset[f"{kind}_observe_{who}"])
        for kind in ("user", "ai")
        for who, json_key in (("me", "observeMe"), ("others", "observeOthers"))
    }


class _HostLookup:
    """Host-block-over-root field lookups for one honcho.json."""

    def __init__(self, host_block: dict, raw: dict):
        self.host, self.raw = host_block, raw

    def pick(self, key: str, default=None):
        """Truthy host value, else ``raw.get(key, default)``."""
        return self.host.get(key) or self.raw.get(key, default)

    def pick_set(self, key: str, default=None):
        """Non-None host value, else ``raw.get(key, default)``."""
        return _first_set(self.host.get(key), default=self.raw.get(key, default))

    def vals(self, *keys: str) -> list:
        """Host values then root values for the aliased ``keys`` (lookup order)."""
        return [self.host.get(k) for k in keys] + [self.raw.get(k) for k in keys]

    def flag(self, *keys: str, default: bool) -> bool:
        """First non-None value across host-then-root for the aliased ``keys``, as bool."""
        return bool(_first_set(*self.vals(*keys), default=default))

    def parsed(self, key: str, caster: Callable[[Any], Any], default):
        """First host-then-root value ``caster`` accepts, else default."""
        return _first_parsed(self.vals(key), caster, default)

    def present(self, key: str, default=None):
        """Host value if the key is PRESENT there (even empty/None), else root."""
        return self.host[key] if key in self.host else self.raw.get(key, default)

    def string(self, key: str, default: str = "") -> str:
        """String field where a host-level empty string can override root."""
        value = self.present(key, default)
        return default if value is None else str(value).strip()

    def string_map(self, key: str) -> dict[str, str]:
        """String-to-string map; a host-level map replaces the root map wholesale."""
        source = self.present(key)
        if not isinstance(source, dict):
            return {}
        pairs = ((str(k).strip(), str(v).strip() if v is not None else "") for k, v in source.items())
        return {k: v for k, v in pairs if k and v}


def _is_local_base_url(base_url: str | None) -> bool:
    """True for loopback/RFC1918/link-local/ULA/CGNAT self-hosted Honcho URLs. Local
    deployments can run without auth but the SDK needs a non-empty api_key, so LAN/VPN
    URLs get the same placeholder-key treatment as localhost."""
    if not base_url:
        return False
    try:
        host = (urlparse(base_url).hostname or "").strip().lower()
    except Exception:
        host = ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # Tailscale/other VPN setups often sit in carrier-grade NAT space (100.64.0.0/10).
    cgnat = ip.version == 4 and ipaddress.ip_address("100.64.0.0") <= ip <= ipaddress.ip_address("100.127.255.255")
    return ip.is_loopback or ip.is_private or ip.is_link_local or cgnat


def _env_base_url() -> str | None:
    """HONCHO_BASE_URL / HONCHO_URL (the SDK's own var); a deployment setting, so plain os.environ."""
    return os.environ.get("HONCHO_BASE_URL", "").strip() or os.environ.get("HONCHO_URL", "").strip() or None


def _connection_fields(look: _HostLookup, host: str, path: Path) -> dict[str, Any]:
    """Resolve identity/credential/transport fields (host block -> root -> env)."""
    raw, host_block = look.raw, look.host
    api_key = look.pick("apiKey") or get_secret("HONCHO_API_KEY")
    # Named-profile host blocks do NOT inherit the default host's apiKey (profiles
    # are credential-isolated); the failure is silent 401s, so warn loudly.
    if not api_key and host_block and host != HOST and _host_block(raw, HOST).get("apiKey"):
        logger.warning("Honcho host block '%s' has no apiKey; the default '%s' host's key "
                       "is NOT inherited (profiles are credential-isolated). Set apiKey on "
                       "hosts.%s in %s or this profile runs unauthenticated.", host, HOST, host, path)
    # The SDK's native format (and Claude Desktop) nests the URL at endpoint.baseUrl;
    # read it before the flat Hermes spellings.
    endpoint_block = raw.get("endpoint")
    native_base_url = endpoint_block.get("baseUrl") if isinstance(endpoint_block, dict) else None
    base_url = _sanitize_url(host_block.get("baseUrl") or host_block.get("base_url") or native_base_url
                             or raw.get("baseUrl") or raw.get("base_url") or _env_base_url())
    return {
        "workspace_id": look.pick("workspace") or host,
        "ai_peer": look.pick("aiPeer") or host,
        "api_key": api_key,
        "environment": look.pick("environment", "production"),
        "base_url": base_url,
        "timeout": _resolve_optional_float(*look.vals("timeout", "requestTimeout"), os.environ.get("HONCHO_TIMEOUT")),
        # Explicit enabled (host, then root) wins; else auto-enable on key/url.
        "enabled": _first_set(*look.vals("enabled"), default=bool(api_key or base_url)),
    }


def _behavior_fields(look: _HostLookup, explicitly_configured: bool) -> dict[str, Any]:
    """Resolve memory-behavior tuning fields (host block -> root -> defaults)."""
    raw_wf = look.pick("writeFrequency") or "async"
    write_frequency: str | int = _first_parsed([raw_wf], int, str(raw_wf))
    depth = look.parsed("dialecticDepth", lambda v: max(1, min(int(v), 3)), 1)
    # Migration guard: configs that predate observationMode keep the old
    # "unified" default; fresh installs get "directional" (all observations on).
    observation_mode = _normalize_choice(
        look.pick("observationMode") or ("unified" if explicitly_configured else "directional"), _OBSERVATION_MODES)
    return {
        "peer_name": look.pick("peerName"),
        # pinUserPeer is the clearer name; the original pinPeerName stays accepted.
        "pin_peer_name": look.flag("pinUserPeer", "pinPeerName", default=False),
        "user_peer_aliases": look.string_map("userPeerAliases"),
        "runtime_peer_prefix": look.string("runtimePeerPrefix"),
        "save_messages": look.pick_set("saveMessages", True),
        "write_frequency": write_frequency,
        "context_tokens": look.parsed("contextTokens", int, None),
        "dialectic_reasoning_level": look.pick("dialecticReasoningLevel") or "low",
        "dialectic_dynamic": look.flag("dialecticDynamic", default=True),
        "dialectic_max_chars": look.parsed("dialecticMaxChars", int, 600),
        "dialectic_depth": depth,
        "dialectic_depth_levels": _parse_dialectic_depth_levels(look.vals("dialecticDepthLevels"), depth),
        "reasoning_heuristic": look.flag("reasoningHeuristic", default=True),
        "reasoning_level_cap": look.pick("reasoningLevelCap") or "high",
        "message_max_chars": look.parsed("messageMaxChars", int, 25000),
        "dialectic_max_input_chars": look.parsed("dialecticMaxInputChars", int, 10000),
        "recall_mode": _normalize_choice(look.pick("recallMode") or "hybrid", _RECALL_MODES),
        "init_on_session_start": look.flag("initOnSessionStart", default=False),
        "injection_frequency": look.pick("injectionFrequency", "every-turn"),
        "context_cadence": look.parsed("contextCadence", int, 1),
        "dialectic_cadence": look.parsed("dialecticCadence", int, 1),
        "query_rewrite": look.flag("queryRewrite", default=False),
        "first_turn_base_wait": look.parsed("firstTurnBaseWait", lambda v: max(0.0, float(v)), 3.0),
        "first_turn_dialectic_wait": look.parsed("firstTurnDialecticWait", lambda v: max(0.0, float(v)), 2.0),
        "observation_mode": observation_mode,
        **_resolve_observation(observation_mode, look.pick("observation")),
        "session_strategy": look.pick("sessionStrategy", "per-directory"),
        "session_peer_prefix": look.pick_set("sessionPeerPrefix", False),
        "session_ai_peer_prefix": look.pick_set("sessionAiPeerPrefix", False),
    }


@dataclass
class HonchoClientConfig:
    """Configuration for Honcho client, resolved for a specific host."""

    host: str = HOST
    workspace_id: str = "hermes"
    api_key: str | None = None
    environment: str = "production"
    base_url: str | None = None  # self-hosted override of the environment mapping
    timeout: float | None = None  # SDK HTTP timeout, seconds
    # Identity
    peer_name: str | None = None
    ai_peer: str = "hermes"
    # True: peer_name wins over gateway runtime identity (Telegram UID, ...), so a
    # single-user deployment keeps one memory across platforms.
    # This keeps memory unified across platforms for single-user deployments where Honcho's one peer-name is
    # an unambiguous identity — otherwise each platform would fork memory into its own peer (#14984).
    # Default ``False`` preserves existing multi-user behaviour.
    pin_peer_name: bool = False
    # Gateway runtime user id -> stable Honcho peer; host map replaces root map.
    user_peer_aliases: dict[str, str] = field(default_factory=dict)
    runtime_peer_prefix: str = ""  # prefix for unknown runtime user ids, e.g. "telegram_"
    # Toggles
    enabled: bool = False
    save_messages: bool = True
    write_frequency: str | int = "async"  # "async" | "turn" | "session" | every-N-turns int
    context_tokens: int | None = None  # prefetch budget; None = uncapped
    # Dialectic (peer.chat) settings
    dialectic_reasoning_level: str = "low"  # minimal | low | medium | high | max
    dialectic_dynamic: bool = True  # model may override the level via honcho_reasoning
    dialectic_max_chars: int = 600  # auto-injection cap; explicit calls bypass it
    dialectic_depth: int = 1  # .chat() passes per cycle (1-3): audit / synthesis / reconcile
    dialectic_depth_levels: list[str] | None = None  # per-pass levels; None = proportional defaults
    reasoning_heuristic: bool = True  # scale auto-injected level up on longer queries
    reasoning_level_cap: str = "high"
    # Honcho API limits (Honcho cloud: 25000 / 10000) — configurable for self-hosts
    message_max_chars: int = 25000
    dialectic_max_input_chars: int = 10000
    # "hybrid" (context + tools) | "context" (no tools) | "tools" (no auto context)
    recall_mode: str = "hybrid"
    init_on_session_start: bool = False  # tools mode: init eagerly instead of on first tool call
    injection_frequency: str = "every-turn"  # or "first-turn"
    context_cadence: int = 1  # min turns between peer.context() calls
    dialectic_cadence: int = 1  # min turns between dialectic prefetches
    query_rewrite: bool = False  # rewrite the user message into a retrieval query (one aux LLM call)
    # Bounded synchronous waits on turn 1, seconds; 0 = fully async first turn.
    first_turn_base_wait: float = 3.0
    first_turn_dialectic_wait: float = 2.0
    # Legacy string shorthand; the granular per-peer booleans below are preferred
    # and map 1:1 to Honcho's SessionPeerConfig.
    observation_mode: str = "directional"
    user_observe_me: bool = True
    user_observe_others: bool = True
    ai_observe_me: bool = True
    ai_observe_others: bool = True
    # Session resolution
    session_strategy: str = "per-directory"
    session_peer_prefix: bool = False
    # Prefixes every resolved session name with ``{ai_peer}-``. The gateway_session_key branch is
    # AI-peer-agnostic, so several AI peers sharing one workspace + peerName + chat key would collide.
    session_ai_peer_prefix: bool = False
    sessions: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    # A hosts.<host> block or explicit enabled flag, vs auto-enabled from a stray env key.
    explicitly_configured: bool = False
    # Provenance captured at resolution time; bound consumers use these instead of
    # re-resolving (the resolvers read a ContextVar background threads can't see).
    # Provenance: WHERE this config was resolved from, captured at resolution time (inside the caller's
    # profile scope). Bound consumers (session manager, OAuth refresh paths) use these instead of
    # re-resolving resolve_config_path()/get_hermes_home() later — those resolvers read a ContextVar that
    # background threads cannot see, so re-resolution from a daemon thread silently lands on the DEFAULT
    # profile (#69123, #74065).
    config_path: Path | None = None
    hermes_home: Path | None = None

    def bound_config_path(self) -> Path:
        """Config path this was resolved from; ambient fallback for hand-built configs."""
        return self.config_path if self.config_path is not None else resolve_config_path()

    @classmethod
    def from_env(cls, workspace_id: str = "hermes", host: str | None = None) -> HonchoClientConfig:
        """Create config from environment variables (fallback)."""
        resolved_host = host or resolve_active_host()
        api_key = get_secret("HONCHO_API_KEY")
        base_url = _sanitize_url(_env_base_url())
        return cls(
            host=resolved_host, workspace_id=workspace_id, api_key=api_key, base_url=base_url,
            environment=os.environ.get("HONCHO_ENVIRONMENT", "production"),
            timeout=_resolve_optional_float(os.environ.get("HONCHO_TIMEOUT")),
            ai_peer=resolved_host, enabled=bool(api_key or base_url),
            config_path=resolve_config_path(), hermes_home=get_hermes_home(),
        )

    @classmethod
    def from_global_config(cls, host: str | None = None, config_path: Path | None = None) -> HonchoClientConfig:
        """Config from the resolved Honcho config path, falling back to env. ``host=None``
        derives it from the active Hermes profile."""
        resolved_host = host or resolve_active_host()
        path = config_path or resolve_config_path()
        if not path.exists():
            logger.debug("No global Honcho config at %s, falling back to env", path)
            return cls.from_env(host=resolved_host)
        try:
            raw = _read_config(path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s, falling back to env", path, e)
            return cls.from_env(host=resolved_host)

        host_block = _host_block(raw, resolved_host)
        explicitly_configured = bool(host_block) or raw.get("enabled") is True
        look = _HostLookup(host_block, raw)
        return cls(
            host=resolved_host, **_connection_fields(look, resolved_host, path), **_behavior_fields(look, explicitly_configured),
            sessions=raw.get("sessions", {}), raw=raw, explicitly_configured=explicitly_configured,
            config_path=path, hermes_home=get_hermes_home(),
        )

    @staticmethod
    def _git_repo_name(cwd: str) -> str | None:
        """Return the git repo root directory name, or None if not in a repo."""
        import subprocess

        try:
            root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, encoding='utf-8',
                                  errors='replace', cwd=cwd, timeout=5, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return Path(root.stdout.strip()).name if root.returncode == 0 else None

    # Honcho rejects session IDs over 100 chars; long gateway keys (Matrix
    # rooms + thread ids, Slack threads) overflow after sanitization.
    _HONCHO_SESSION_ID_MAX_LEN, _HONCHO_SESSION_ID_HASH_LEN = 100, 8

    @classmethod
    def _enforce_session_id_limit(cls, sanitized: str, original: str) -> str:
        """Truncate to the limit with a ``-<sha256 prefix>`` suffix hashed over the ORIGINAL key:
        two long keys sharing a prefix stay distinct; keys that sanitize identically still
        collide intentionally (same logical session)."""
        max_len, hash_len = cls._HONCHO_SESSION_ID_MAX_LEN, cls._HONCHO_SESSION_ID_HASH_LEN
        if len(sanitized) <= max_len:
            return sanitized
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:hash_len]
        prefix = sanitized[: max_len - hash_len - 1].rstrip("-")
        return f"{prefix}-{digest}"

    def _with_peer_prefix(self, name: str) -> str:
        return f"{self.peer_name}-{name}" if self.session_peer_prefix and self.peer_name else name

    def resolve_session_name(
        self, cwd: str | None = None, session_title: str | None = None,
        session_id: str | None = None, gateway_session_key: str | None = None,
    ) -> str | None:
        """Resolve the Honcho session name; with ``session_ai_peer_prefix`` the result is prefixed
        ``{ai_peer}-`` on every path, including the AI-peer-agnostic gateway session key."""
        import re

        result = self._resolve_session_name_base(cwd=cwd, session_title=session_title,
                                                 session_id=session_id, gateway_session_key=gateway_session_key)
        if result and self.session_ai_peer_prefix and self.ai_peer:
            ai = re.sub(r'[^a-zA-Z0-9_-]+', '-', self.ai_peer).strip('-')
            if ai:
                prefixed = f"{ai}-{result}"
                return self._enforce_session_id_limit(prefixed, prefixed)
        return result

    def _resolve_session_name_base(
        self, cwd: str | None = None, session_title: str | None = None,
        session_id: str | None = None, gateway_session_key: str | None = None,
    ) -> str | None:
        """Order: gateway session key (per-chat isolation no cwd/strategy gives) -> per-session
        strategy's session_id (authoritative, so a generated title never remaps a live conversation)
        -> sessions map override -> /title -> per-repo (git root name) -> per-directory (basename)
        -> global (workspace)."""
        import re

        def _slug(text: str) -> str:
            return re.sub(r'[^a-zA-Z0-9_-]+', '-', text).strip('-')

        cwd = cwd or os.getcwd()
        if gateway_session_key and _slug(gateway_session_key):
            return self._enforce_session_id_limit(_slug(gateway_session_key), gateway_session_key)
        if self.session_strategy == "per-session" and session_id:
            return self._with_peer_prefix(session_id)
        manual = self.sessions.get(cwd)
        if manual:
            return manual
        if session_title and _slug(session_title):
            return self._with_peer_prefix(_slug(session_title))
        if self.session_strategy == "per-repo":
            return self._with_peer_prefix(self._git_repo_name(cwd) or Path(cwd).name)
        if self.session_strategy in {"per-directory", "per-session"}:
            return self._with_peer_prefix(Path(cwd).name)
        return self.workspace_id


def spawn_context_thread(target, *, name: str, daemon: bool = True, args: tuple = ()) -> "_threading.Thread":
    """Thread that inherits the caller's contextvars: profile isolation is a ContextVar
    (set_hermes_home_override) and a plain Thread starts EMPTY, so ambient resolution on it
    would silently land on the default profile."""
    import contextvars

    ctx = contextvars.copy_context()
    return _threading.Thread(target=lambda: ctx.run(target, *args), name=name, daemon=daemon)


def get_honcho_client(config: HonchoClientConfig | None = None) -> Honcho:
    """Get or create the Honcho client for this config's identity. Clients are cached PER
    IDENTITY (host, workspace, provenance paths, credential fingerprint, timeout) so
    multi-profile processes don't share a first-config-wins client. With no config the active
    honcho.json is resolved — correct only on threads that see the profile ContextVar; pass a
    bound config elsewhere. Each identity's client is built once under concurrent first calls.

    See #69123, #74065.
    """
    key = _client_cache_key(config)
    slot = _slot_for(key)
    cached = slot.peek()
    if cached is not None:
        _refresh_oauth(config, cached, slot)
        refreshed = slot.peek()
        if refreshed is not None:
            return refreshed
        # Slot was reset by a failed in-place rotation — rebuild below.

    if config is None:
        config = HonchoClientConfig.from_global_config()

    # Start with a live access token rather than 401ing an hour in.
    _refresh_oauth(config)

    if not config.api_key and not config.base_url:
        raise ValueError("Honcho API key not found. Get your API key at https://app.honcho.dev, "
                         "then run 'hermes honcho setup' or set HONCHO_API_KEY. "
                         "For local instances, set HONCHO_BASE_URL instead.")

    return slot.get(lambda: _build_client(config))


def _build_client(config: HonchoClientConfig) -> "Honcho":
    """Construct the SDK client (runs inside the slot factory so racing callers share one)."""
    with contextlib.suppress(Exception):  # lazy-dep failures fall through to the canonical import error below
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("memory.honcho", prompt=False)
    try:
        from honcho import Honcho
    except ImportError:
        raise ImportError("honcho-ai is required for Honcho integration. Install it with: pip install honcho-ai  "
                          "(or run `hermes honcho setup` to configure).")

    # config.yaml honcho.base_url / timeout fill whatever honcho.json left unset.
    base_url, timeout = config.base_url, config.timeout
    if not base_url or timeout is None:
        with contextlib.suppress(Exception):
            from hermes_cli.config import load_config
            honcho_cfg = load_config().get("honcho", {})
            if isinstance(honcho_cfg, dict):
                base_url = base_url or _sanitize_url(honcho_cfg.get("base_url", "").strip() or None)
                if timeout is None:
                    timeout = _resolve_optional_float(honcho_cfg.get("timeout"), honcho_cfg.get("request_timeout"))
    if timeout is None:
        timeout = _DEFAULT_HTTP_TIMEOUT  # an unconfigured install must not hang on a stalled request

    if base_url:
        logger.info("Initializing Honcho client (base_url: %s, workspace: %s)", base_url, config.workspace_id)
    else:
        # Name the SDK's environment fallback at INFO so a self-hosted user whose
        # config wasn't picked up notices they're talking to the public cloud.
        logger.info("Initializing Honcho client (host: %s, workspace: %s, base_url unset — SDK will resolve from environment=%s)",
                    config.host, config.workspace_id, config.environment)

    # Local instances need no key but the SDK wants a non-empty string: honor a
    # key set EXPLICITLY in honcho.json (host block or root) and treat an
    # env-sourced key as likely-cloud, substituting the placeholder.
    raw = config.raw or {}
    explicit_key = _host_block(raw, config.host).get("apiKey") or raw.get("apiKey")
    api_key = "local" if _is_local_base_url(base_url) and not explicit_key else config.api_key
    kwargs: dict = {"workspace_id": config.workspace_id, "api_key": api_key, "environment": config.environment, "timeout": timeout}
    if base_url:
        # The SDK's route builders already carry the version prefix ("/v3/..."), so
        # strip a trailing version segment from any base_url to avoid "/v3/v3/...".
        import re
        kwargs["base_url"] = re.sub(r"/v\d+/*$", "", base_url).rstrip("/")
    return Honcho(**kwargs)


def reset_honcho_client() -> None:
    """Reset all cached Honcho clients (tests, OAuth re-login)."""
    with _client_slots_lock:
        _client_slots.clear()
    _honcho_json_timeout_memo.clear()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'SingletonSlot': ('plugins.plugin_utils', 'SingletonSlot'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
