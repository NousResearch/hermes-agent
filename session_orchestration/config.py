"""
session_orchestration/config.py — typed accessor for the session_orchestration
config section in Hermes config.yaml.

All session-orchestration components (watcher, spawn, ingest, status, relay)
read their config through this module so the gate is checked consistently.

Config section (config.yaml):

    session_orchestration:
      enabled: false                       # master gate — default OFF
      feed_channel_id: ""                  # Discord channel id for the unified feed
      external_runs_channel_id: ""         # Discord channel for adopted external runs
      hang_stale_seconds: 300              # static stale threshold (seconds) before hang
      hang_idle_ticks: 3                   # N idle ticks before hang is declared

Disabled ⇒ byte-identical prior behaviour: no watcher tick runs, no ingest
route processes, no commands surface in the gateway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — hard-coded here so callers get sensible values without any
# config file present.  enabled=False is the safety invariant.
# ---------------------------------------------------------------------------

_DEFAULT_ENABLED: bool = False
_DEFAULT_FEED_CHANNEL_ID: Optional[str] = None
_DEFAULT_EXTERNAL_RUNS_CHANNEL_ID: Optional[str] = None
_DEFAULT_HANG_STALE_SECONDS: int = 300  # 5 min static stale threshold
_DEFAULT_HANG_IDLE_TICKS: int = 3       # N ticks of pane-hash unchanged before hang
_DEFAULT_DEAD_TMUX_REAP: bool = True    # reap sessions whose tmux pane died without a done marker
_DEFAULT_GC_AFTER_SECONDS: int = 86400  # 24 h retention for terminal rows
_DEFAULT_RENUDGE_AFTER_SECONDS: int = 1800  # 30 min before re-surfacing an unacted attention item


@dataclass(frozen=True)
class SessionOrchestrationConfig:
    """Typed view of the ``session_orchestration`` config section.

    All fields have safe defaults so callers never need ``None``-guards on
    the struct itself.  The only guarantee that matters for correctness is:
    ``enabled=False`` ⇒ no side effects (callers MUST check ``enabled``).
    """

    enabled: bool = _DEFAULT_ENABLED
    feed_channel_id: Optional[str] = _DEFAULT_FEED_CHANNEL_ID
    external_runs_channel_id: Optional[str] = _DEFAULT_EXTERNAL_RUNS_CHANNEL_ID
    hang_stale_seconds: int = _DEFAULT_HANG_STALE_SECONDS
    hang_idle_ticks: int = _DEFAULT_HANG_IDLE_TICKS
    # When True (default), the watcher proactively marks a session terminal
    # (ERROR/DONE) if its tmux pane has died without emitting a 'done' marker
    # and there are no recent heartbeats.  Set to false to revert to the prior
    # behaviour of waiting out the full hang-nudge ladder instead.
    dead_tmux_reap: bool = _DEFAULT_DEAD_TMUX_REAP
    # Retention window (seconds) for terminal rows (DONE / ERROR).  Rows with a
    # ``terminated_at`` stamp older than this are GC-ed once per watcher tick.
    # Set to 0 (or any value <= 0) to disable GC entirely.  Default: 86400 (24 h).
    gc_after_seconds: int = _DEFAULT_GC_AFTER_SECONDS
    # Re-nudge interval (seconds) for sessions that remain in an attention state
    # (WAITING_USER or PAUSED_HANDOFF) without user action.  The watcher sends a
    # DM at most once per interval to re-surface the waiting item.  Set to 0 (or
    # any value <= 0) to disable re-nudging entirely.  Default: 1800 (30 min).
    renudge_after_seconds: int = _DEFAULT_RENUDGE_AFTER_SECONDS
    # Manual repo alias overrides for the repo registry (alias → path or dict).
    # Stored as the raw config dict so repo_registry.build_repo_registry() can
    # parse it without the config module importing the registry module.
    repos: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionOrchestrationConfig":
        """Parse from a raw config dict (the ``session_orchestration`` sub-dict).

        Unknown keys are ignored silently so forward-compat is preserved.
        Malformed values fall back to defaults — never raises.
        """
        if not isinstance(data, dict):
            return cls()

        enabled = _coerce_bool(data.get("enabled"), _DEFAULT_ENABLED)
        feed_channel_id = _coerce_optional_str(data.get("feed_channel_id"))
        external_runs_channel_id = _coerce_optional_str(
            data.get("external_runs_channel_id")
            # Legacy alias used by ingest.py before T016
            or data.get("external_runs_thread_id")
        )
        hang_stale_seconds = _coerce_positive_int(
            data.get("hang_stale_seconds"), _DEFAULT_HANG_STALE_SECONDS
        )
        hang_idle_ticks = _coerce_positive_int(
            data.get("hang_idle_ticks"), _DEFAULT_HANG_IDLE_TICKS
        )
        dead_tmux_reap = _coerce_bool(data.get("dead_tmux_reap"), _DEFAULT_DEAD_TMUX_REAP)
        gc_after_seconds = _coerce_int(
            data.get("gc_after_seconds"), _DEFAULT_GC_AFTER_SECONDS
        )
        renudge_after_seconds = _coerce_int(
            data.get("renudge_after_seconds"), _DEFAULT_RENUDGE_AFTER_SECONDS
        )
        repos = data.get("repos")
        repos = repos if isinstance(repos, dict) else {}

        return cls(
            enabled=enabled,
            feed_channel_id=feed_channel_id,
            external_runs_channel_id=external_runs_channel_id,
            hang_stale_seconds=hang_stale_seconds,
            hang_idle_ticks=hang_idle_ticks,
            dead_tmux_reap=dead_tmux_reap,
            gc_after_seconds=gc_after_seconds,
            renudge_after_seconds=renudge_after_seconds,
            repos=repos,
        )


# ---------------------------------------------------------------------------
# Public loader — reads live Hermes config; safe to call at any time
# ---------------------------------------------------------------------------


def load_session_orchestration_config(
    _cfg: Optional[Dict[str, Any]] = None,
) -> SessionOrchestrationConfig:
    """Return the typed ``SessionOrchestrationConfig`` for the current process.

    Parameters
    ----------
    _cfg:
        Optional pre-loaded config dict (the full Hermes config, not just the
        ``session_orchestration`` sub-dict).  Used by callers that already
        hold a config dict (e.g. spawn.py) to avoid re-loading from disk.
        When ``None`` (the default), config is loaded via
        ``hermes_cli.config.load_config_readonly()``.

    Always returns a valid ``SessionOrchestrationConfig`` — never raises.
    Falls back to all-defaults (``enabled=False``) on any error.
    """
    try:
        if _cfg is None:
            from hermes_cli.config import load_config_readonly
            _cfg = load_config_readonly()

        so_raw = _cfg.get("session_orchestration") if isinstance(_cfg, dict) else None
        return SessionOrchestrationConfig.from_dict(so_raw or {})
    except Exception as exc:  # broad: config load must never crash a caller  # noqa: BLE001
        logger.debug("session_orchestration.config: load failed — using defaults: %s", exc)
        return SessionOrchestrationConfig()


def is_enabled(_cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Return True iff session_orchestration is enabled in Hermes config.

    Convenience one-liner for gate checks — equivalent to
    ``load_session_orchestration_config(_cfg).enabled``.
    """
    return load_session_orchestration_config(_cfg).enabled


# ---------------------------------------------------------------------------
# Coercion helpers (private — keep coercion logic local to this module)
# ---------------------------------------------------------------------------


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        pass
    return default


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _coerce_positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        n = int(value)
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    """Coerce *value* to int; fall back to *default* on parse failure.

    Unlike ``_coerce_positive_int`` this allows 0 and negative values so
    callers can use 0 as a sentinel (e.g. gc_after_seconds=0 disables GC).
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
