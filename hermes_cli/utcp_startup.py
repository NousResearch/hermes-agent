"""Shared CLI/TUI-safe helpers for background UTCP provider discovery.

Mirrors :mod:`hermes_cli.mcp_startup`: one discovery slot per profile home, the
caller's context (HERMES_HOME override) carried into the worker thread, an
optional ``-t/--toolsets`` provider filter, and a bounded join before the agent
snapshots its tool list.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from contextvars import copy_context
from typing import Iterator, Optional

from hermes_constants import hermes_home_key

_utcp_discovery_lock = threading.Lock()
# Discovery slot per profile home: a shared desktop/dashboard backend serving
# several profiles runs one UTCP discovery per profile instead of the first
# profile claiming the slot for everybody (same contract as MCP, #67605).
_utcp_discovery_started: set[str] = set()
_utcp_discovery_thread: dict[str, threading.Thread] = {}
# Process-wide provider-name allowlist derived from ``-t/--toolsets``.
_utcp_provider_filter: Optional[list[str]] = None

# Bursty CPU work (SDK import, schema parsing, registration) rides the GIL
# convoy at the default 5 ms switch interval; slicing it finer keeps the
# agent-build thread and the main event loop responsive while discovery runs.
_DISCOVERY_SWITCH_INTERVAL_S = 0.0005


def set_utcp_provider_filter(toolsets: object) -> Optional[list[str]]:
    """Derive the UTCP provider allowlist from a ``-t/--toolsets`` value.

    Built-in toolset names never match a provider name, so `-t terminal` simply
    discovers nothing. ``all``/``*`` or an empty value clears the filter.
    """
    global _utcp_provider_filter
    names: list[str] = []
    if isinstance(toolsets, str):
        names = [item.strip() for item in toolsets.split(",") if item.strip()]
    elif isinstance(toolsets, (list, tuple, set)):
        for item in toolsets:
            names.extend(part.strip() for part in str(item).split(",") if part.strip())
    if not names or "all" in names or "*" in names:
        _utcp_provider_filter = None
    else:
        _utcp_provider_filter = names
    return _utcp_provider_filter


def get_utcp_provider_filter() -> Optional[list[str]]:
    return _utcp_provider_filter


def configured_utcp_provider_names() -> set[str]:
    """Cheap raw-config probe: names of enabled providers (no SDK import)."""
    try:
        from hermes_cli.config import read_raw_config

        section = (read_raw_config() or {}).get("utcp")
        if not isinstance(section, dict) or not section.get("enabled"):
            return set()
        providers = section.get("providers")
        if not isinstance(providers, (list, tuple)):
            return set()
        names = set()
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            enabled = entry.get("enabled", True)
            if isinstance(enabled, str) and enabled.strip().lower() in {"false", "0", "no", "off"}:
                continue
            if enabled is False:
                continue
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
        return names
    except Exception:
        # Conservative: let discovery run and fail loudly in its own thread.
        return {"<probe-failed>"}


def _has_configured_utcp_providers() -> bool:
    return bool(configured_utcp_provider_names())


def _any_utcp_connected() -> bool:
    from tools.utcp_tool import get_utcp_status

    return any(row.get("connected") for row in (get_utcp_status() or []))


@contextmanager
def _discovery_gil_budget() -> Iterator[None]:
    previous = sys.getswitchinterval()
    sys.setswitchinterval(_DISCOVERY_SWITCH_INTERVAL_S)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _run_discovery() -> None:
    with _discovery_gil_budget():
        from tools.utcp_tool import discover_utcp_tools

        discover_utcp_tools(allowed_names=_utcp_provider_filter)


def start_background_utcp_discovery(*, logger, thread_name: str = "cli-utcp-discovery") -> None:
    """Spawn one background UTCP discovery thread per profile home.

    A run that finished without connecting any provider is retried by the next
    call (a provider added later, or a transient failure), so the slot never
    pins the profile in a permanently-empty state.
    """
    home_key = hermes_home_key()
    with _utcp_discovery_lock:
        if home_key in _utcp_discovery_started:
            thread = _utcp_discovery_thread.get(home_key)
            if thread is not None and thread.is_alive():
                return
            try:
                if _any_utcp_connected():
                    return
            except Exception:
                return
            logger.info("Previous UTCP discovery left no connected provider; retrying")
            _utcp_discovery_started.discard(home_key)
            _utcp_discovery_thread.pop(home_key, None)

        _utcp_discovery_started.add(home_key)
        if not _has_configured_utcp_providers():
            return

        def _discover() -> None:
            try:
                _run_discovery()
                try:
                    if not _any_utcp_connected():
                        logger.warning("Background UTCP discovery completed with no connected provider")
                except Exception:
                    logger.debug("Failed to inspect UTCP status after discovery", exc_info=True)
            except Exception:
                logger.debug("Background UTCP tool discovery failed", exc_info=True)
            finally:
                with _utcp_discovery_lock:
                    _utcp_discovery_thread.pop(home_key, None)

        thread = threading.Thread(
            target=copy_context().run, args=(_discover,), name=thread_name, daemon=True
        )
        _utcp_discovery_thread[home_key] = thread
        thread.start()


def _resolve_discovery_timeout(explicit: Optional[float], *, single_query: bool = False) -> float:
    """Resolve the join bound: explicit arg > ``utcp.*`` config > default."""
    if explicit is not None:
        return float(explicit)
    key = "single_query_discovery_timeout" if single_query else "discovery_timeout"
    fallback = 15.0 if single_query else 1.5
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("utcp")
        raw = section.get(key) if isinstance(section, dict) else None
        value = float(raw) if raw is not None else fallback
        return value if value > 0 else fallback
    except Exception:
        return fallback


def _current_home_thread() -> Optional[threading.Thread]:
    return _utcp_discovery_thread.get(hermes_home_key())


def wait_for_utcp_discovery(timeout: Optional[float] = None, *, single_query: bool = False) -> None:
    """Join in-flight UTCP discovery for the current profile, bounded by config."""
    thread = _current_home_thread()
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout, single_query=single_query))


def utcp_discovery_in_flight() -> bool:
    thread = _current_home_thread()
    return thread is not None and thread.is_alive()


def join_utcp_discovery(timeout: Optional[float] = None) -> bool:
    """Block up to *timeout*; True once discovery finished, False if still running."""
    thread = _current_home_thread()
    if thread is None:
        return True
    thread.join(timeout=timeout)
    return not thread.is_alive()


def ensure_utcp_discovery_before_agent_build(
    *,
    logger,
    timeout: Optional[float] = None,
    single_query: bool = False,
    thread_name: str = "cli-utcp-discovery",
) -> None:
    """Give configured UTCP tools a bounded chance to register before AIAgent.

    Same contract as ``ensure_mcp_discovery_before_agent_build``: start
    discovery if nothing did, then wait at most the configured bound. Failures
    are swallowed — a broken provider must never abort agent construction.
    """
    try:
        start_background_utcp_discovery(logger=logger, thread_name=thread_name)
        wait_for_utcp_discovery(timeout=timeout, single_query=single_query)
    except Exception:
        logger.debug("UTCP discovery readiness check failed before agent build", exc_info=True)
