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
        "tools.mcp_tool_discovery",
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


def test_prepare_agent_startup_skips_discovery_when_chat_resolves_to_tui(
    monkeypatch,
):
    """Bare ``hermes`` / ``hermes chat`` on a TTY with ``display.interface:
    tui`` resolves to the TUI via ``_resolve_use_tui``, but does NOT pass
    ``--tui`` or ``HERMES_TUI``. Discovery must be skipped in the wrapper:
    the TUI gateway owns it, and the wrapper would otherwise hold a dead
    MCP server for the entire session (3 copies per TUI instance).
    """
    calls = {"background": 0, "inline": 0}

    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: True)
    monkeypatch.setattr(
        mcp_startup,
        "start_background_mcp_discovery",
        lambda **_kwargs: calls.__setitem__("background", calls["background"] + 1),
    )
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
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool_discovery",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: calls.__setitem__("inline", calls["inline"] + 1),
        ),
    )

    main_mod._prepare_agent_startup(_agent_args(command=None))

    assert calls["background"] == 0
    assert calls["inline"] == 0
    assert mcp_startup._mcp_discovery_thread is None


def test_prepare_agent_startup_keeps_discovery_for_non_chat_commands(
    monkeypatch,
):
    """Non-chat commands never launch the TUI, so they must keep their own
    MCP discovery even when the ambient display config resolves to TUI —
    ``_is_tui_chat_launch`` must not consult ``_resolve_use_tui`` there."""
    calls = {"inline": 0}

    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: True)
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
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool_discovery",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: calls.__setitem__("inline", calls["inline"] + 1),
        ),
    )

    main_mod._prepare_agent_startup(_agent_args(command="mcp", mcp_action="serve"))

    assert calls["inline"] == 1


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
        "tools.mcp_tool_discovery",
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


def test_portable_only_mcp_configuration_opens_startup_gate(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(read_raw_config=lambda: {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.agent_plugins",
        types.SimpleNamespace(
            has_enabled_agent_plugin_mcp=lambda _config: True,
        ),
    )

    assert mcp_startup._has_configured_mcp_servers() is True








def _retry_logger():
    return types.SimpleNamespace(
        debug=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
    )


def _install_retry_stubs(monkeypatch, *, connected: bool, calls: dict):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool_discovery",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: calls.__setitem__("mcp", calls["mcp"] + 1),
            get_mcp_status=lambda: [{"connected": connected}],
        ),
    )




# --- -t/--toolsets MCP spawn filter (#19000) --------------------------------


@pytest.fixture
def _reset_mcp_server_filter():
    saved = mcp_startup._mcp_server_filter
    try:
        yield
    finally:
        mcp_startup._mcp_server_filter = saved


@pytest.mark.parametrize(
    ("toolsets", "expected"),
    [
        (None, None),
        ("", None),
        ("all", None),
        (["*"], None),
        ("terminal,web", ["terminal", "web"]),
        (["terminal", "code-mcp,web"], ["terminal", "code-mcp", "web"]),
    ],
)
def test_set_mcp_server_filter_normalizes(_reset_mcp_server_filter, toolsets, expected):
    assert mcp_startup.set_mcp_server_filter(toolsets) == expected
    assert mcp_startup.get_mcp_server_filter() == expected


def test_discover_mcp_tools_spawns_only_allowed_servers(monkeypatch):
    """The filter must narrow the spawn set before any server is connected;
    built-in toolset names in the list are ignored."""
    from tools import mcp_tool
    from tools import mcp_tool_config as _mcp_config
    from tools import mcp_tool_discovery as _mcp_discovery
    from tools import mcp_tool_loop as _mcp_loop

    servers = {
        "code-mcp": {"command": "true"},
        "docs-mcp": {"command": "true"},
    }
    seen: dict[str, dict] = {}

    sdk_probes = {"n": 0}

    def _fake_ensure_sdk():
        sdk_probes["n"] += 1
        return True

    monkeypatch.setattr(_mcp_config, "_load_mcp_config", lambda: dict(servers))
    monkeypatch.setattr(mcp_tool, "_ensure_mcp_sdk", _fake_ensure_sdk)
    monkeypatch.setattr(_mcp_loop, "_try_acquire_mcp_discovery_lock", lambda: mcp_tool._LOCK_UNAVAILABLE)
    monkeypatch.setattr(mcp_tool, "_release_mcp_discovery_lock", lambda *_a, **_k: None, raising=False)

    def _fake_register(cfgs):
        seen.update(cfgs)
        return []

    monkeypatch.setattr(_mcp_discovery, "register_mcp_servers", _fake_register)
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_server_connecting", set())

    # Everything (no filter) — both would be registered.
    _mcp_discovery.discover_mcp_tools()
    assert set(seen) == {"code-mcp", "docs-mcp"}

    # `-t terminal,code-mcp` — only the matching server; "terminal" is a no-op.
    seen.clear()
    _mcp_discovery.discover_mcp_tools(allowed_mcp_names=["terminal", "code-mcp"])
    assert set(seen) == {"code-mcp"}

    # `-t terminal` — no MCP server in the filter: skip the whole MCP load,
    # including the ~260ms `mcp` SDK import.
    seen.clear()
    sdk_probes["n"] = 0
    assert _mcp_discovery.discover_mcp_tools(allowed_mcp_names=["terminal"]) == []
    assert seen == {}
    assert sdk_probes["n"] == 0


def test_background_discovery_honors_server_filter(monkeypatch, _reset_mcp_server_filter):
    calls: list = []
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool_discovery",
        types.SimpleNamespace(discover_mcp_tools=lambda allowed_mcp_names=None: calls.append(allowed_mcp_names)),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=nullcontext),
    )
    mcp_startup.set_mcp_server_filter("terminal,code-mcp")
    mcp_startup._discover_mcp_tools_without_interactive_oauth()
    assert calls == [["terminal", "code-mcp"]]


def test_prepare_agent_startup_installs_server_filter(monkeypatch, _reset_mcp_server_filter):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setattr(main_mod, "_should_background_mcp_startup", lambda args: False)
    monkeypatch.setattr(main_mod, "_command_has_dedicated_mcp_startup", lambda args: True)
    main_mod._prepare_agent_startup(_agent_args(toolsets="terminal,code-mcp"))
    assert mcp_startup.get_mcp_server_filter() == ["terminal", "code-mcp"]
