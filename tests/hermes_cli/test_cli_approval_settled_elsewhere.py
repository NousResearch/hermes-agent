"""An approval or clarify another viewer answers closes the classic CLI's panel.

The CLI's gateway agent (``GatewayAgent``) receives ``approval.request`` / ``clarify.request`` and
blocks its pump inside the CLI's real panel (``HermesCLI._approval_callback`` /
``HermesCLI._clarify_callback``). When Desktop or the TUI answers the same prompt first, the gateway
publishes ``approval.settled`` / ``clarify.settled`` (gateway/session_pending_controls.py,
``PendingControls.respond``); the panel must close then, not at its timeout. The gateway is scripted
with the frames the real one publishes; the CLI side is real.
"""

import threading

import pytest

from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_connection import GatewayConnection
from tests.hermes_cli.test_gateway_agent import SID, FakeClient, connect_with, frame


@pytest.fixture
def panel(monkeypatch):
    import cli as cli_module
    from cli import HermesCLI

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    monkeypatch.setitem(cli_module.CLI_CONFIG, "approvals", {"timeout": 60})
    monkeypatch.setitem(cli_module.CLI_CONFIG, "clarify", {"timeout": 60})
    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture",
                    base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    return cli


def _run_turn(script, panel, deadline_attr, **agent_kw):
    """Play ``script`` through a real GatewayAgent wired to the CLI panel on an agent thread (the
    approval callback is registered per thread, as the CLI registers it for the turn)."""
    from tools.terminal_tool import set_approval_callback

    client = FakeClient(script + [frame("message.complete", {"text": "ok", "outcome": "completed"})])
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    conn.state(SID)
    agent = GatewayAgent(connection=conn, session_id=SID, modal_withdraw_callback=panel._withdraw_gateway_modal,
                         **agent_kw)

    def run():
        set_approval_callback(panel._approval_callback)
        try:
            agent.run_conversation("clean the build")
        finally:
            set_approval_callback(None)
    worker = threading.Thread(target=run, daemon=True)
    try:
        worker.start()
        worker.join(10)
        assert not worker.is_alive(), "the CLI's panel stayed open after another viewer answered"
    finally:
        setattr(panel, deadline_attr, 0.1)  # release a panel left open so the worker thread ends
        worker.join(5)
        conn.close()


def test_an_approval_answered_by_another_viewer_closes_the_cli_panel(panel):
    request = frame("approval.request", {"kind": "approval", "prompt_id": "apr-1", "execution_generation": 7,
                                         "command": "rm -rf build", "description": "recursive delete",
                                         "choices": ["once", "deny", "session", "always"]})
    settled = frame("approval.settled", {"prompt_id": "apr-1", "execution_generation": 7}, admission=None)
    _run_turn([request, "wait", settled], panel, "_approval_deadline")
    assert panel._approval_state is None


def test_a_clarify_answered_by_another_viewer_closes_the_cli_panel(panel):
    request = frame("clarify.request", {"kind": "clarify", "prompt_id": "clr-1", "execution_generation": 7,
                                        "question": "Which build?", "choices": ["debug", "release"],
                                        "multi_select": False})
    settled = frame("clarify.settled", {"prompt_id": "clr-1", "execution_generation": 7}, admission=None)
    _run_turn([request, "wait", settled], panel, "_clarify_deadline", clarify_callback=panel._clarify_callback)
    assert panel._clarify_state is None
