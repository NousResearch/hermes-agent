"""Tests for acp_adapter.entry startup wiring."""

import sys
from unittest.mock import MagicMock, patch

import acp
import pytest

from acp_adapter import entry


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool.discover_mcp_tools",
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


def test_skip_configured_mcp_is_honored_at_the_session_build_site(monkeypatch):
    """The flag must gate BOTH discovery sites, not just the spawn-time one.

    ``entry.py`` checks it before the JSON-RPC loop, but ``_make_agent`` used
    to call ``ensure_mcp_discovery_before_agent_build`` unconditionally — and
    that helper *starts* discovery, not just joins it. A metadata-only host
    that set the flag and supplied its servers through ``session/new`` got the
    configured-MCP startup anyway, inside session construction (block/buzz#4098,
    the recurrence of block/buzz#3355 one protocol step on).
    """
    import acp_adapter.session as session_mod

    calls: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **kwargs: calls.append("started"),
    )
    manager = session_mod.SessionManager(agent_factory=None)
    monkeypatch.setattr(
        session_mod.SessionManager,
        "_build_agent_kwargs",
        lambda self, **kwargs: {},
        raising=False,
    )

    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    with patch("run_agent.AIAgent", MagicMock()):
        manager._make_agent(session_id="s1", cwd=".")
    assert calls == [], "flag set — discovery must not be (re)started here"

    monkeypatch.delenv("HERMES_ACP_SKIP_CONFIGURED_MCP", raising=False)
    with patch("run_agent.AIAgent", MagicMock()):
        manager._make_agent(session_id="s2", cwd=".")
    assert calls == ["started"], "flag unset — normal behavior must be unchanged"
