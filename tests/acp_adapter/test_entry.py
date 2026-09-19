"""Tests for acp_adapter.entry startup wiring."""

import sys

import acp
import pytest

from acp_adapter import entry


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {"order": []}

    class FakeAgent:
        def __init__(self):
            calls["order"].append("agent")

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs
        calls["order"].append("run")

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(entry, "_preload_stdin_sensitive_dependencies", lambda: calls["order"].append("preload"))
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **_kwargs: calls["order"].append("mcp"),
    )
    monkeypatch.setattr("acp_adapter.server.HermesACPAgent", FakeAgent)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True
    assert calls["order"] == ["preload", "mcp", "agent", "run"]


def test_preloads_numpy_only_for_holographic_memory(monkeypatch):
    imported = []
    monkeypatch.setattr("plugins.memory._get_active_memory_provider", lambda: "holographic")
    monkeypatch.setattr(entry.importlib, "import_module", imported.append)

    entry._preload_stdin_sensitive_dependencies()

    assert imported == ["numpy"]


def test_preload_preserves_holographic_fallback_without_numpy(monkeypatch):
    def missing_numpy(_name):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    monkeypatch.setattr("plugins.memory._get_active_memory_provider", lambda: "holographic")
    monkeypatch.setattr(entry.importlib, "import_module", missing_numpy)

    entry._preload_stdin_sensitive_dependencies()


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool_discovery.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []










def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1
