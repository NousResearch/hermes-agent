"""Tests for acp_adapter.entry startup wiring."""

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
    import pm

    def refuse(name, **kwargs):
        raise pm.InstallError(name, "download failed")

    monkeypatch.setattr(pm, "ensure", refuse)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1


def test_setup_browser_is_one_explicit_package_request(monkeypatch):
    import pm

    calls = []
    monkeypatch.setattr(pm, "ensure", lambda name, **kwargs: calls.append((name, kwargs)))

    entry.main(["--setup-browser", "--yes"])

    assert calls == [("agent-browser", {"explicit": True})]


def test_client_disconnect_stops_the_running_turn(monkeypatch):
    """stdio EOF ends ``acp.run_agent`` while a turn still runs on the executor thread. Nobody can
    receive its updates or answer its approvals, so the turn must be stopped, not finished headless."""
    import asyncio
    import threading

    from acp.schema import TextContentBlock

    class BlockingAgent:
        session_id = "busy"

        def __init__(self):
            self.started, self.stopped = threading.Event(), threading.Event()

        def run_conversation(self, **kwargs):
            self.started.set()
            self.stopped.wait(30)  # a long foreground tool
            return {"final_response": "", "messages": []}

        def interrupt(self, message=None):
            self.stopped.set()

    turn_agent = BlockingAgent()

    async def fake_run_agent(agent, **kwargs):
        agent.session_manager._install_state("busy", turn_agent, ".", "m", [], persist=False)
        asyncio.get_running_loop().create_task(agent.prompt([TextContentBlock(type="text", text="go")], "busy"))
        await asyncio.to_thread(turn_agent.started.wait, 10)
        # Returning here is the client hanging up mid-turn.

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert turn_agent.started.is_set()
    assert turn_agent.stopped.is_set()
