"""Configuration resolution for the teams_voice bridge.

Values come from (in priority order): the plugin's ``config.extra`` block in
``config.yaml`` (when wired through the gateway), then environment variables,
then safe defaults. Secrets are never logged.

The wire contract is fixed by the companion .NET media worker (AzureBot /
OpenClawBridge), so the header names, HMAC payload shape, and default path mirror
that worker exactly — see ``protocol.py`` and ``hmac_auth.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

# Audio wire format — single source of truth, mirrors the worker (PCM 16 kHz,
# 16-bit signed, mono, little-endian; 20 ms / 640-byte frames).
PCM_SAMPLE_RATE_HZ = 16_000
FRAME_DURATION_MS = 20
BYTES_PER_FRAME = PCM_SAMPLE_RATE_HZ * FRAME_DURATION_MS // 1000 * 2  # 640

# Default WebSocket path the worker connects to: ``/voice/msteams/stream/{callId}``.
DEFAULT_PATH = "/voice/msteams/stream"

# HMAC upgrade header names — fixed by the existing worker. The "OpenClaw" prefix
# is historical; Hermes reuses the headers verbatim so the worker needs no change.
HEADER_TIMESTAMP = "X-OpenClawTeamsBridge-Timestamp"
HEADER_SIGNATURE = "X-OpenClawTeamsBridge-Signature"


@dataclass(frozen=True)
class TeamsVoiceConfig:
    """Resolved bridge configuration."""

    shared_secret: str
    host: str = "127.0.0.1"
    port: int = 8443
    path: str = DEFAULT_PATH
    # Replay/clock-skew window for the HMAC handshake, in milliseconds.
    hmac_window_ms: int = 60_000
    # Connection caps (DoS guards) — mirror the TS driver's defaults.
    max_connections: int = 64
    max_connections_per_ip: int = 8
    # A connection must send ``session.start`` within this window or it is reaped.
    pre_start_timeout_s: float = 10.0
    require_recording_status: bool = True
    # Outbound "call me back": the worker's loopback HTTP endpoint + default tenant.
    worker_base_url: str = "http://127.0.0.1:9440"
    tenant_id: str = ""
    # Caller allowlist (AAD object ids). Empty = allow all. Display-name matching
    # is weaker (spoofable) and off unless ``allowlist_allow_names`` is set.
    allowlist: tuple[str, ...] = ()
    allowlist_allow_names: bool = False
    # Refuse outbound place-call to a non-loopback worker unless explicitly allowed
    # (the shared secret would otherwise be sent to that host).
    allow_remote_worker: bool = False
    # Per-call vision spend cap across look_at_screen + ambient push (0 = unlimited).
    max_vision_per_minute: int = 30
    # Agent session continuity: "per-call" | "per-thread" | "per-aad".
    session_scope: str = "per-call"
    # Group-call wake phrases (speak only when addressed).
    wake_phrases: tuple[str, ...] = ("assistant", "hermes")
    # Post end-of-call meeting minutes to the Teams chat (opt-in).
    meeting_recap: bool = False

    @property
    def configured(self) -> bool:
        """True when a shared secret is present (the bridge can authenticate)."""
        return bool(self.shared_secret)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def plugin_config_block() -> dict[str, Any]:
    """Return the ``plugins.entries.teams_voice.config`` block from config.yaml.

    Empty dict when unset or config can't be loaded. ``${VAR}`` references are
    already expanded by Hermes's config loader, so secrets can live in ``.env``
    and be referenced here (e.g. ``shared_secret: ${TEAMS_VOICE_SHARED_SECRET}``).
    """
    try:
        from hermes_cli.config import load_config

        config = load_config()
        node = (
            config.get("plugins", {})
            .get("entries", {})
            .get("teams_voice", {})
            .get("config", {})
        )
        return node if isinstance(node, dict) else {}
    except Exception:  # noqa: BLE001 — config is optional; fall back to env
        return {}


def resolve_config(extra: Mapping[str, Any] | None = None) -> TeamsVoiceConfig:
    """Build a :class:`TeamsVoiceConfig` from config.yaml + environment.

    ``extra`` is the per-plugin config block; when omitted it is read from
    ``plugins.entries.teams_voice.config`` in config.yaml. Environment variables
    are the fallback so the bridge still works with no config file.
    """
    extra = extra if extra is not None else plugin_config_block()

    shared_secret = (
        str(extra.get("shared_secret") or "").strip()
        or os.getenv("TEAMS_VOICE_SHARED_SECRET", "").strip()
    )
    host = (
        str(extra.get("host") or "").strip()
        or os.getenv("TEAMS_VOICE_HOST", "").strip()
        or "127.0.0.1"
    )
    port = _coerce_int(
        extra.get("port") or os.getenv("TEAMS_VOICE_PORT", ""), 8443
    )
    path = str(extra.get("path") or "").strip() or DEFAULT_PATH
    window = _coerce_int(
        extra.get("hmac_window_ms") or os.getenv("TEAMS_VOICE_HMAC_WINDOW_MS", ""),
        60_000,
    )
    worker_base_url = (
        str(extra.get("worker_base_url") or "").strip()
        or os.getenv("TEAMS_VOICE_WORKER_BASE_URL", "").strip()
        or "http://127.0.0.1:9440"
    )
    tenant_id = (
        str(extra.get("tenant_id") or "").strip()
        or os.getenv("TEAMS_VOICE_TENANT_ID", "").strip()
        or os.getenv("TEAMS_TENANT_ID", "").strip()
    )
    allowlist = _coerce_list(extra.get("allowlist"), os.getenv("TEAMS_VOICE_ALLOWLIST", ""))
    max_vision = _coerce_int(
        extra.get("max_vision_per_minute") or os.getenv("TEAMS_VOICE_MAX_VISION_PER_MINUTE", ""), 30
    )
    session_scope = (
        str(extra.get("session_scope") or "").strip()
        or os.getenv("TEAMS_VOICE_SESSION_SCOPE", "").strip()
        or "per-call"
    )
    wake = _coerce_list(extra.get("wake_phrases"), os.getenv("TEAMS_VOICE_WAKE_PHRASES", ""))
    meeting_recap = str(
        extra.get("meeting_recap", "") or os.getenv("TEAMS_VOICE_MEETING_RECAP", "")
    ).strip().lower() in ("1", "true", "yes", "on")

    return TeamsVoiceConfig(
        shared_secret=shared_secret,
        host=host,
        port=port,
        path=path,
        hmac_window_ms=window,
        worker_base_url=worker_base_url,
        tenant_id=tenant_id,
        allowlist=allowlist,
        max_vision_per_minute=max_vision,
        session_scope=session_scope,
        wake_phrases=wake or ("assistant", "hermes"),
        meeting_recap=meeting_recap,
        allowlist_allow_names=_coerce_bool(extra.get("allowlist_allow_names"), "TEAMS_VOICE_ALLOWLIST_ALLOW_NAMES"),
        allow_remote_worker=_coerce_bool(extra.get("allow_remote_worker"), "TEAMS_VOICE_ALLOW_REMOTE_WORKER"),
    )


def caller_allowed(config: "TeamsVoiceConfig", aad_id: str | None, display_name: str | None) -> bool:
    """Allowlist check: AAD id by default; display name only if opted in.

    Empty allowlist = allow all (backward-compatible)."""
    if not config.allowlist:
        return True
    if (aad_id or "").strip().lower() in config.allowlist:
        return True
    if config.allowlist_allow_names and (display_name or "").strip().lower() in config.allowlist:
        return True
    return False


def _coerce_bool(value: Any, env: str) -> bool:
    return str(value if value not in (None, "") else os.getenv(env, "")).strip().lower() in (
        "1", "true", "yes", "on",
    )


def _coerce_list(value: Any, env: str) -> tuple[str, ...]:
    """List from a config list or a comma-separated env string (lowercased, trimmed)."""
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        items = [p.strip() for p in (env or "").split(",")]
    return tuple(i.lower() for i in items if i)
