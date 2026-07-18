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
        lambda timeout=0.75, single_query=False: waited.__setitem__("done", True),
    )

    def _fake_agent(*_a, **_k):
        assert waited["done"] is True
        return types.SimpleNamespace()

    monkeypatch.setattr(cli_mod, "AIAgent", _fake_agent)

    assert cli._init_agent() is True


def test_init_agent_passes_single_query_flag_to_discovery_wait(monkeypatch):
    """Single-query mode forwards single_query=True so the larger MCP cold-start
    bound is used — the one tool snapshot must wait for slow servers (#51316)."""
    seen = {}

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    cli._single_query_mode = True

    monkeypatch.setattr(
        mcp_startup,
        "wait_for_mcp_discovery",
        lambda timeout=0.75, single_query=False: seen.__setitem__(
            "single_query", single_query
        ),
    )
    monkeypatch.setattr(cli_mod, "AIAgent", lambda *_a, **_k: types.SimpleNamespace())

    assert cli._init_agent() is True
    assert seen.get("single_query") is True


def test_oneshot_run_agent_waits_single_query_mcp_before_aiagent(monkeypatch):
    """Top-level `hermes -z` routes to oneshot; MCP wait must precede AIAgent (#51316)."""
    import inspect

    import hermes_cli.oneshot as oneshot_mod

    order: list[str] = []

    def _wait(*, timeout=None, single_query=False):
        order.append(f"wait:{single_query}")

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            order.append("aiagent")
            self.suppress_status_output = False
            self.stream_delta_callback = None
            self.tool_gen_callback = None

        def run_conversation(self, prompt):
            return {"final_response": "ok"}

    monkeypatch.setattr(
        oneshot_mod, "wait_for_mcp_discovery", _wait, raising=False
    )
    # Patch the imported symbol used inside _run_agent.
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.wait_for_mcp_discovery",
        _wait,
    )
    monkeypatch.setattr(oneshot_mod, "AIAgent", _FakeAgent, raising=False)
    monkeypatch.setattr(
        "run_agent.AIAgent",
        _FakeAgent,
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "test-model"}},
    )
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **k: {
            "api_key": "k",
            "base_url": "http://x",
            "provider": "test",
            "api_mode": "chat",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda *a, **k: set(),
    )
    monkeypatch.setattr(
        oneshot_mod,
        "_create_session_db_for_oneshot",
        lambda: None,
    )
    monkeypatch.setattr(
        oneshot_mod,
        "get_fallback_chain",
        lambda cfg: None,
    )

    text, _ = oneshot_mod._run_agent("hello", use_config_toolsets=False)
    assert text == "ok"
    assert order[:2] == ["wait:True", "aiagent"]

    # Dispatch: -z / --oneshot exits via run_oneshot (not cli single-query).
    main_src = inspect.getsource(
        __import__("hermes_cli.main", fromlist=["*"])
    )
    assert "run_oneshot" in main_src
    assert 'getattr(args, "oneshot", None)' in main_src


def test_top_level_z_dispatches_to_oneshot_not_cli_query(monkeypatch):
    """Regression: -z must not rely on cli.py `_single_query_mode` (#51322 review)."""
    import inspect
    import hermes_cli.main as main_mod
    import hermes_cli.oneshot as oneshot_mod

    src = inspect.getsource(oneshot_mod._run_agent)
    wait_idx = src.find("wait_for_mcp_discovery(single_query=True)")
    agent_idx = src.find("AIAgent(")
    assert wait_idx != -1 and agent_idx != -1 and wait_idx < agent_idx

    main_src = inspect.getsource(main_mod)
    z_block = main_src
    assert "from hermes_cli.oneshot import run_oneshot" in z_block
