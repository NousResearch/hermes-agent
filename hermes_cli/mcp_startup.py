"""Shared CLI/TUI-safe helpers for background MCP discovery."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Optional

_mcp_discovery_lock = threading.Lock()
_mcp_discovery_started = False
_mcp_discovery_thread: Optional[threading.Thread] = None


def _has_configured_mcp_servers() -> bool:
    """Cheap config probe so non-MCP users avoid importing the MCP stack."""
    try:
        from hermes_cli.config import read_raw_config

        mcp_servers = (read_raw_config() or {}).get("mcp_servers")
        return isinstance(mcp_servers, dict) and len(mcp_servers) > 0
    except Exception:
        # Be conservative: if config probing fails, try discovery in the
        # background so startup still can't block.
        return True


def start_background_mcp_discovery(*, logger, thread_name: str) -> None:
    """Spawn one shared background MCP discovery thread for this process."""
    global _mcp_discovery_started, _mcp_discovery_thread

    with _mcp_discovery_lock:
        if _mcp_discovery_started:
            return
        _mcp_discovery_started = True
        if not _has_configured_mcp_servers():
            return

        def _discover() -> None:
            try:
                _discover_mcp_tools_without_interactive_oauth()
            except Exception:
                logger.debug("Background MCP tool discovery failed", exc_info=True)

        thread = threading.Thread(
            target=_discover,
            name=thread_name,
            daemon=True,
        )
        _mcp_discovery_thread = thread
        thread.start()


def _resolve_discovery_timeout(explicit: "float | None") -> float:
    """Resolve the MCP discovery wait bound: explicit arg > config > default.

    Reads ``mcp_discovery_timeout`` from config.yaml, defaulting to the value in
    ``DEFAULT_CONFIG`` (single source of truth) when the key is absent. Kept lazy
    and fail-safe — a missing/invalid value or a broken config falls back to a
    short safe bound so startup can never hang or crash.
    """
    if explicit is not None:
        return explicit
    try:
        from hermes_cli.config import load_config, DEFAULT_CONFIG

        default = float(DEFAULT_CONFIG.get("mcp_discovery_timeout", 1.5))
        raw = (load_config() or {}).get("mcp_discovery_timeout", default)
        val = float(raw)
        return val if val > 0 else default
    except Exception:
        return 1.5


def _discover_mcp_tools_without_interactive_oauth() -> None:
    """Run MCP discovery without letting OAuth read from the user's stdin."""
    try:
        from tools.mcp_oauth import suppress_interactive_oauth
    except Exception:
        suppress_interactive_oauth = nullcontext

    with suppress_interactive_oauth():
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()


def wait_for_mcp_discovery(timeout: "float | None" = None) -> None:
    """Wait for background MCP discovery before the first tool snapshot.

    ``thread.join(timeout)`` returns the INSTANT discovery completes, so this
    only ever blocks for the real connect time of a still-pending server —
    users with no MCP servers or fast servers pay ~0s.  The bound (from
    ``mcp_discovery_timeout`` in config) just caps the wait so a dead server
    can't freeze startup; servers that miss it are picked up by the automatic
    late-binding refresh.
    """
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout))


def mcp_discovery_in_flight() -> bool:
    """Return True if THIS module's background discovery thread is still running.

    Mirrors ``tui_gateway.entry.mcp_discovery_in_flight`` for the surfaces that
    start discovery through ``start_background_mcp_discovery`` here (the desktop
    app + dashboard WebSocket sidecar via ``tui_gateway/ws.py``, and
    ``hermes dashboard``).  Those processes populate THIS module's
    ``_mcp_discovery_thread``, not ``tui_gateway.entry``'s, so the late-refresh
    scheduler must consult both to decide whether a slow server's tools are
    still pending (see #51587).
    """
    thread = _mcp_discovery_thread
    return thread is not None and thread.is_alive()


def join_mcp_discovery(timeout: "float | None" = None) -> bool:
    """Block until THIS module's background discovery finishes, up to ``timeout``.

    Returns True if discovery has completed (thread absent or no longer alive),
    False if it is still running after the timeout.  Unlike
    ``wait_for_mcp_discovery`` this accepts an unbounded/long wait and reports
    the outcome, for the off-critical-path late-refresh waiter.
    """
    thread = _mcp_discovery_thread
    if thread is None:
        return True
    thread.join(timeout=timeout)
    return not thread.is_alive()


# ─── Config hot-reload change detection ──────────────────────────────────────

def detect_mcp_servers_change(
    cfg_path,
    prev_mtime: float,
    prev_servers: dict,
) -> "tuple[bool, float, dict]":
    """Detect whether config.yaml's ``mcp_servers`` section changed on disk.

    Shared by the interactive TUI (``cli.HermesCLI._check_config_mcp_changes``)
    and the long-running gateway
    (``gateway.run.GatewayRunner._check_config_mcp_changes``) so both surfaces
    auto-reload MCP connections on a config edit using one change-detection
    implementation, without a restart or a new session.

    Returns ``(changed, new_mtime, new_mcp_servers)``:

    * ``changed`` is True ONLY when the ``mcp_servers`` subtree differs from
      ``prev_servers`` — editing any OTHER config section never triggers a
      reload.
    * ``new_mtime`` is the file's current mtime, advanced even when the edit was
      to an unrelated section or the file was mid-write, so callers don't
      re-``stat``/re-parse the same bytes on every poll tick.
    * ``new_mcp_servers`` is the freshly-parsed ``mcp_servers`` map when
      ``changed`` is True, otherwise ``prev_servers`` unchanged.

    Never raises: a missing file, an ``OSError`` from ``stat()``, or
    invalid/partial YAML (e.g. an editor mid-save) is reported as "no change",
    so a background watcher can poll safely.
    """
    import yaml as _yaml

    try:
        mtime = cfg_path.stat().st_mtime
    except OSError:
        return False, prev_mtime, prev_servers

    # Fast path: file untouched since the last check — skip the parse entirely.
    if mtime == prev_mtime:
        return False, prev_mtime, prev_servers

    try:
        with open(cfg_path, encoding="utf-8") as f:
            new_cfg = _yaml.safe_load(f) or {}
    except Exception:
        # File changed but is unreadable/partial (mid-write) or invalid YAML.
        # Adopt the new mtime so we don't re-parse the same broken bytes every
        # tick, but report no change — the next good write bumps mtime again.
        return False, mtime, prev_servers

    new_mcp = new_cfg.get("mcp_servers") or {}
    if new_mcp == prev_servers:
        # Some OTHER config section was edited; mcp_servers is identical.
        return False, mtime, prev_servers

    return True, mtime, new_mcp
