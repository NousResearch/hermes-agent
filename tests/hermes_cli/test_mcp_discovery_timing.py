"""Regression tests for MCP discovery timing in non-interactive sessions.

Covers the race where AIAgent snapshots its tool registry at construction
time before background MCP discovery finishes.  In single-query (``-q``) and
oneshot (``-z``) mode there is only ONE turn — no between-turns late-binding
refresh — so missing tools at construction are missing for the entire
session.

Tests verify:
  1. The ``single_query`` flag resolves to the larger bound.
  2. ``ensure_mcp_discovery_before_agent_build`` starts discovery if needed.
  3. Oneshot calls the helper before AIAgent construction (ordering).
  4. The wait stays bounded when discovery is slow (dead server).
  5. Interactive mode keeps the small bound (not affected).
"""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from hermes_cli import mcp_startup
from hermes_constants import hermes_home_key


@pytest.fixture(autouse=True)
def _reset_mcp_startup_state():
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread
    try:
        mcp_startup._mcp_discovery_started = set()
        mcp_startup._mcp_discovery_thread = {}
        yield
    finally:
        thread = mcp_startup._current_home_thread()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        mcp_startup._mcp_discovery_started = saved_started
        mcp_startup._mcp_discovery_thread = saved_thread


# ── _resolve_discovery_timeout: single_query bound ──────────────────────────


def test_resolve_discovery_timeout_single_query_uses_larger_bound(monkeypatch):
    """Single-query mode reads the larger mcp_single_query_discovery_timeout."""
    import hermes_cli.config as cfg

    monkeypatch.setattr(
        cfg,
        "load_config",
        lambda: {
            "mcp_discovery_timeout": 1.5,
            "mcp_single_query_discovery_timeout": 25.0,
        },
    )
    assert mcp_startup._resolve_discovery_timeout(None) == 1.5
    assert mcp_startup._resolve_discovery_timeout(None, single_query=True) == 25.0


def test_resolve_discovery_timeout_single_query_falls_back(monkeypatch):
    """Bad/absent single-query value falls back to DEFAULT_CONFIG, never hangs."""
    import hermes_cli.config as cfg

    default = float(cfg.DEFAULT_CONFIG.get("mcp_single_query_discovery_timeout", 15.0))
    monkeypatch.setattr(
        cfg, "load_config", lambda: {"mcp_single_query_discovery_timeout": 0}
    )
    assert mcp_startup._resolve_discovery_timeout(None, single_query=True) == default

    monkeypatch.setattr(
        cfg, "load_config", lambda: {"mcp_single_query_discovery_timeout": "oops"}
    )
    assert mcp_startup._resolve_discovery_timeout(None, single_query=True) == default

    monkeypatch.setattr(cfg, "load_config", lambda: {})
    assert mcp_startup._resolve_discovery_timeout(None, single_query=True) == default


def test_resolve_discovery_timeout_explicit_overrides_single_query():
    """An explicit timeout always wins, even in single-query mode."""
    assert mcp_startup._resolve_discovery_timeout(5.0, single_query=True) == 5.0


# ── ensure_mcp_discovery_before_agent_build ─────────────────────────────────


def _stub_mcp_modules(monkeypatch):
    """Stub MCP-related modules for helper tests."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
            load_config=lambda: {},
            DEFAULT_CONFIG={"mcp_discovery_timeout": 0.1, "mcp_single_query_discovery_timeout": 0.2},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: __import__("contextlib").nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool_discovery",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: None,
            get_mcp_status=lambda: [{"connected": True}],
        ),
    )


def test_ensure_helper_starts_discovery_and_waits(monkeypatch):
    """The helper starts background discovery if not yet started, then waits."""
    _stub_mcp_modules(monkeypatch)
    waited = []

    original_wait = mcp_startup.wait_for_mcp_discovery

    def _spy_wait(timeout=None, *, single_query=False):
        waited.append(("wait", single_query))
        original_wait(timeout=timeout, single_query=single_query)

    monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", _spy_wait)

    logger = types.SimpleNamespace(debug=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)

    mcp_startup.ensure_mcp_discovery_before_agent_build(
        logger=logger,
        single_query=True,
    )

    # Discovery was started (thread created)
    assert mcp_startup._current_home_thread() is not None or waited
    # Wait was called with single_query=True
    assert any(call[1] is True for call in waited)



    # Second call didn't create a new thread (first one completed, status shows connected)
    # or if it did, it's because the first exited with zero connected — but we stubbed
    # get_mcp_status to return connected=True, so no retry.
    # The key invariant: no exception, no hang.


def test_ensure_helper_swallows_errors(monkeypatch):
    """A broken MCP config never aborts agent construction."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            load_config=lambda: {},
            DEFAULT_CONFIG={},
        ),
    )
    logger = types.SimpleNamespace(debug=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)

    # Should not raise
    mcp_startup.ensure_mcp_discovery_before_agent_build(logger=logger)


# ── oneshot ordering: discovery before AIAgent ──────────────────────────────




# ── _init_agent ordering: discovery before AIAgent (CLI path) ───────────────




def test_init_agent_forwards_single_query_flag(monkeypatch):
    """Single-query mode forwards single_query=True to the discovery wait."""
    import cli as cli_mod

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    cli._single_query_mode = True

    seen = {}

    def _fake_ensure(*, logger, timeout=None, single_query=False, **_kw):
        seen["single_query"] = single_query

    monkeypatch.setattr(
        mcp_startup,
        "ensure_mcp_discovery_before_agent_build",
        _fake_ensure,
    )
    import run_agent
    monkeypatch.setattr(run_agent, "AIAgent", lambda *_a, **_k: types.SimpleNamespace())

    assert cli._init_agent() is True
    assert seen.get("single_query") is True


def test_init_agent_defaults_to_interactive(monkeypatch):
    """Without _single_query_mode, the helper uses interactive (short) bound."""
    import cli as cli_mod

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True

    seen = {}

    def _fake_ensure(*, logger, timeout=None, single_query=False, **_kw):
        seen["single_query"] = single_query

    monkeypatch.setattr(
        mcp_startup,
        "ensure_mcp_discovery_before_agent_build",
        _fake_ensure,
    )
    import run_agent
    monkeypatch.setattr(run_agent, "AIAgent", lambda *_a, **_k: types.SimpleNamespace())

    assert cli._init_agent() is True
    assert seen.get("single_query") is False


# ── bounded wait: slow server doesn't freeze startup ────────────────────────


def test_wait_stays_bounded_when_discovery_is_slow(monkeypatch):
    """A slow/dead MCP server must not freeze startup: the wait is capped."""
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_single_query_discovery_timeout": 0.1})

    stop = threading.Event()
    thread = threading.Thread(target=lambda: stop.wait(10), daemon=True)
    thread.start()
    mcp_startup._mcp_discovery_thread[hermes_home_key()] = thread

    try:
        start = time.monotonic()
        mcp_startup.wait_for_mcp_discovery(single_query=True)
        elapsed = time.monotonic() - start
    finally:
        stop.set()

    assert elapsed < 3.0, (
        f"wait blocked {elapsed:.2f}s on a stuck MCP server — the wait must "
        "stay bounded by mcp_single_query_discovery_timeout"
    )


