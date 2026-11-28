"""A consent question that is not a shell command keeps its own header in the classic CLI panel.

MCP elicitation, MCP skill loads and card-fill consent pass ``title=`` to
``request_elicitation_consent``; the panel used to ignore it and always read "Dangerous Command",
both when the agent runs in-process and when the classic CLI is a gateway client (the gateway's
pending-prompt projection dropped the key).
"""
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI
from tools.approval_prompt import prompt_dangerous_approval


def _bare_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_lock = threading.Lock()
    cli._sudo_state = cli._secret_state = cli._slash_confirm_state = cli._connection_state = None
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._command_running = False
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._app = SimpleNamespace(invalidate=MagicMock())
    cli._ring_bell = lambda **_k: None
    cli._paint_now = lambda: None
    cli._persist_prompt_summary = lambda *a, **k: None
    return cli


def _panel_while_asking(monkeypatch, ask):
    """Run *ask(cli)* on a worker thread, render the panel it opens, answer ``once``."""
    monkeypatch.setattr("cli.shutil.get_terminal_size", lambda *_a, **_k: os.terminal_size((100, 40)))
    cli, result = _bare_cli(), {}
    worker = threading.Thread(target=lambda: result.update(choice=ask(cli)), daemon=True)
    worker.start()
    deadline = time.monotonic() + 5
    while cli._approval_state is None and time.monotonic() < deadline:
        time.sleep(0.01)
    state = cli._approval_state
    assert state is not None, "the approval panel never opened"
    text = "".join(fragment for _style, fragment in cli._get_approval_display_fragments())
    state["response_queue"].put("once")
    worker.join(5)
    return result.get("choice"), text


def test_in_process_consent_title_replaces_the_dangerous_command_header(monkeypatch):
    choice, text = _panel_while_asking(monkeypatch, lambda cli: prompt_dangerous_approval(
        "Load MCP skill 'review'?", "Untrusted server input.", timeout_seconds=10,
        approval_callback=cli._approval_callback, allow_session=False, allow_permanent=False,
        title="Load remote skill?"))
    assert choice == "once"
    assert "Load remote skill?" in text and "Dangerous Command" not in text


def test_gateway_projected_consent_keeps_its_title_in_the_cli_panel(monkeypatch):
    """Gateway queue entry -> pending-prompt projection -> CLI gateway client -> panel."""
    from gateway.session_pending_controls import PendingControls
    from hermes_cli.gateway_agent import GatewayAgent
    from tools.terminal_tool import set_approval_callback
    from tui_gateway.contracts.canonical_sessions import CanonicalPendingPrompt

    published = []
    events = SimpleNamespace(lock=threading.RLock(),
                             publish=lambda sid, prompt, event_type: published.append(prompt))
    PendingControls(events).register("s", SimpleNamespace(), 1, {
        "request_id": "r1", "command": "Load MCP skill 'review'?", "description": "Untrusted server input.",
        "title": "Load remote skill?", "allow_session": False, "allow_permanent": False})
    prompt = CanonicalPendingPrompt.model_validate(published[0]).model_dump(exclude_none=True)
    assert prompt["title"] == "Load remote skill?"

    client = SimpleNamespace(call=MagicMock())
    agent = SimpleNamespace(connection=client, session_id="s", _generation=1)

    def answer(cli):
        set_approval_callback(cli._approval_callback)
        try:
            GatewayAgent._on_approval(agent, prompt, None, None)
        finally:
            set_approval_callback(None)
        return client.call.call_args.kwargs["choice"]

    choice, text = _panel_while_asking(monkeypatch, answer)
    assert choice == "once"
    assert "Load remote skill?" in text and "Dangerous Command" not in text
