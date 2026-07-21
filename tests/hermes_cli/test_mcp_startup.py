"""Regression tests for bounded/lazy CLI MCP startup."""

from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import sys
import threading
import time
import types

import pytest

import cli as cli_mod
from hermes_cli import main as main_mod
from hermes_cli import mcp_startup


@pytest.fixture(autouse=True)
def _reset_mcp_startup_state():
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread
    try:
        mcp_startup._mcp_discovery_started = False
        mcp_startup._mcp_discovery_thread = None
        yield
    finally:
        thread = mcp_startup._mcp_discovery_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        mcp_startup._mcp_discovery_started = saved_started
        mcp_startup._mcp_discovery_thread = saved_thread


def _agent_args(**overrides) -> Namespace:
    base = {
        "accept_hooks": False,
        "command": "chat",
        "cron_command": None,
        "gateway_command": None,
        "mcp_action": None,
        "tui": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_prepare_agent_startup_backgrounds_blocking_mcp_for_chat(monkeypatch):
    stop = threading.Event()
    calls = {"mcp": 0}

    def _blocking_discover():
        calls["mcp"] += 1
        stop.wait()

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
            load_config=lambda: {},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=lambda *_a, **_k: None),
    )
    # Stub mcp_oauth so the background thread doesn't pay the real (cold,
    # ~0.75s) ``tools.mcp_oauth`` import before calling discovery. This test
    # asserts the *backgrounding contract* (main thread returns fast, discovery
    # runs off-thread), not OAuth suppression — the unrelated import latency
    # would otherwise blow the polling deadline on a loaded CI runner.
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=_blocking_discover),
    )

    try:
        start = time.monotonic()
        main_mod._prepare_agent_startup(_agent_args())
        elapsed = time.monotonic() - start
        assert elapsed < 0.2
        deadline = time.monotonic() + 3.0
        while calls["mcp"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls["mcp"] == 1
        assert mcp_startup._mcp_discovery_thread is not None
        assert mcp_startup._mcp_discovery_thread.is_alive()
    finally:
        stop.set()


def test_background_mcp_discovery_suppresses_interactive_oauth(monkeypatch):
    state = {"active": False, "during_discover": None}

    class SuppressInteractiveOAuth:
        def __enter__(self):
            state["active"] = True

        def __exit__(self, *_exc):
            state["active"] = False

    def _discover():
        state["during_discover"] = state["active"]

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"url": "https://mcp.example.test/mcp"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(
            suppress_interactive_oauth=lambda: SuppressInteractiveOAuth(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=_discover),
    )

    mcp_startup.start_background_mcp_discovery(
        logger=types.SimpleNamespace(debug=lambda *_a, **_k: None),
        thread_name="test-mcp-discovery",
    )
    assert mcp_startup._mcp_discovery_thread is not None
    mcp_startup._mcp_discovery_thread.join(timeout=1.0)

    assert state["during_discover"] is True
    assert state["active"] is False


def test_prepare_agent_startup_skips_mcp_bootstrap_for_tui_chat(monkeypatch):
    calls = {"mcp": 0}

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(load_config=lambda: {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=lambda *_a, **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: calls.__setitem__("mcp", calls["mcp"] + 1)
        ),
    )

    main_mod._prepare_agent_startup(_agent_args(tui=True))

    assert calls["mcp"] == 0
    assert mcp_startup._mcp_discovery_thread is None


def test_cli_get_tool_definitions_briefly_waits_for_fast_mcp_thread(monkeypatch):
    thread = threading.Thread(target=lambda: time.sleep(0.05), daemon=True)
    thread.start()
    mcp_startup._mcp_discovery_thread = thread

    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(get_tool_definitions=lambda *_a, **_k: ["ok"]),
    )

    start = time.monotonic()
    result = cli_mod.get_tool_definitions(enabled_toolsets=["web"], quiet_mode=True)
    elapsed = time.monotonic() - start

    assert result == ["ok"]
    assert elapsed >= 0.04
    assert not thread.is_alive()


def test_init_agent_waits_for_mcp_discovery_before_agent_build(monkeypatch):
    waited = {"done": False}

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True

    monkeypatch.setattr(
        mcp_startup,
        "wait_for_mcp_discovery",
        lambda timeout=0.75: waited.__setitem__("done", True),
    )

    def _fake_agent(*_a, **_k):
        assert waited["done"] is True
        return types.SimpleNamespace()

    monkeypatch.setattr(cli_mod, "AIAgent", _fake_agent)

    assert cli._init_agent() is True


# ---------------------------------------------------------------------------
# First-turn readiness regressions for #38448 — hermes -z and hermes chat -q
# must wait (bounded) for in-flight background MCP discovery before they
# snapshot the tool registry at agent-construction time.
# ---------------------------------------------------------------------------


def test_oneshot_waits_for_inflight_mcp_discovery_before_agent_build(monkeypatch):
    """hermes -z race (#38448): background discovery started by main.py, then
    oneshot constructs its agent immediately.  _run_agent must join the
    discovery thread (bounded) BEFORE the AIAgent construction-time tool
    snapshot, so MCP tools are present on the first turn."""
    discovery_done = threading.Event()

    def _fast_discovery():
        time.sleep(0.05)
        discovery_done.set()

    thread = threading.Thread(target=_fast_discovery, daemon=True)
    thread.start()
    mcp_startup._mcp_discovery_thread = thread

    observed = []

    def _observe_at_construction():
        observed.append(discovery_done.is_set())

    import run_agent

    import hermes_cli.config as config_mod
    import hermes_cli.fallback_config as fallback_mod
    import hermes_cli.oneshot as oneshot_mod
    import hermes_cli.runtime_provider as runtime_mod
    import hermes_cli.tools_config as tools_config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {"mcp_discovery_timeout": 2.0})
    monkeypatch.setattr(runtime_mod, "resolve_runtime_provider", lambda **_k: {})
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *_a, **_k: ["terminal"])
    monkeypatch.setattr(fallback_mod, "get_fallback_chain", lambda *_a, **_k: [])
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)

    class _FakeAgent:
        def __init__(self, **_kwargs):
            _observe_at_construction()

        def run_conversation(self, _prompt):
            return {"final_response": "ok", "messages": []}

    monkeypatch.setattr(run_agent, "AIAgent", _FakeAgent)

    response, _result = oneshot_mod._run_agent("hi")

    assert response == "ok"
    assert observed == [True], (
        "AIAgent was constructed before background MCP discovery finished — "
        "the tool snapshot would miss MCP tools (#38448)"
    )


def test_oneshot_mcp_wait_is_bounded_when_discovery_is_slow(monkeypatch):
    """A slow/dead MCP server must not freeze hermes -z: the wait is capped by
    ``mcp_discovery_timeout`` (same bounded startup contract as the interactive
    CLI), and the agent is still constructed when the bound expires."""
    stop = threading.Event()

    def _stuck_discovery():
        stop.wait(10)

    thread = threading.Thread(target=_stuck_discovery, daemon=True)
    thread.start()
    mcp_startup._mcp_discovery_thread = thread

    import run_agent

    import hermes_cli.config as config_mod
    import hermes_cli.fallback_config as fallback_mod
    import hermes_cli.oneshot as oneshot_mod
    import hermes_cli.runtime_provider as runtime_mod
    import hermes_cli.tools_config as tools_config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {"mcp_discovery_timeout": 0.1})
    monkeypatch.setattr(runtime_mod, "resolve_runtime_provider", lambda **_k: {})
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *_a, **_k: ["terminal"])
    monkeypatch.setattr(fallback_mod, "get_fallback_chain", lambda *_a, **_k: [])
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)

    built = {"n": 0}

    class _FakeAgent:
        def __init__(self, **_kwargs):
            built["n"] += 1

        def run_conversation(self, _prompt):
            return {"final_response": "ok", "messages": []}

    monkeypatch.setattr(run_agent, "AIAgent", _FakeAgent)

    try:
        start = time.monotonic()
        response, _result = oneshot_mod._run_agent("hi")
        elapsed = time.monotonic() - start
    finally:
        stop.set()

    assert response == "ok"
    assert built["n"] == 1
    assert elapsed < 3.0, (
        f"oneshot blocked {elapsed:.2f}s on a stuck MCP server — the wait must "
        "stay bounded by mcp_discovery_timeout, not join unboundedly"
    )


def test_quiet_chat_init_agent_mcp_wait_is_bounded_when_discovery_is_slow(monkeypatch):
    """hermes chat -q constructs its agent through HermesCLI._init_agent; its
    MCP wait must likewise stay bounded so a dead server can't freeze the
    single-query path (#38448 discussion)."""
    stop = threading.Event()

    thread = threading.Thread(target=lambda: stop.wait(10), daemon=True)
    thread.start()
    mcp_startup._mcp_discovery_thread = thread

    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {"mcp_discovery_timeout": 0.1})

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True

    monkeypatch.setattr(cli_mod, "AIAgent", lambda *_a, **_k: types.SimpleNamespace())

    try:
        start = time.monotonic()
        ok = cli._init_agent()
        elapsed = time.monotonic() - start
    finally:
        stop.set()

    assert ok is True
    assert elapsed < 3.0, (
        f"quiet-chat agent init blocked {elapsed:.2f}s on a stuck MCP server — "
        "wait_for_mcp_discovery must keep its configured bound"
    )
