"""A held turn (``hold_stream=True`` transform) never lets the original text out by a side path.

Through a real ``AIAgent`` against the loopback model: the transform still runs when a plugin
reload is mid-flight at the final; a transform that raises withholds the reply instead of passing
the original; commentary the turn never showed live stays out of every history load while the
model's own context keeps it byte-for-byte; a Codex turn that fails after its agentMessage
completed is rewritten too.
"""

import os
import sys
import threading
import types
from pathlib import Path

from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.fakes.redactor_plugin import COMMENTARY, RAW, REWRITE, redactor_home, redactor_turn


def _agent(server, **kwargs):
    from run_agent import AIAgent

    return AIAgent(base_url=server.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                   quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["todo"], **kwargs)


def _write_plugin(home: Path, body: str) -> None:
    (home / "plugins" / "redactor" / "__init__.py").write_text(body, encoding="utf-8")


def test_reload_during_a_held_turn_cannot_skip_the_final_rewrite(monkeypatch):
    from hermes_cli import plugins

    home = Path(os.environ["HERMES_HOME"])
    gate = types.ModuleType("hold_reload_gate")
    gate.reload, gate.entered, gate.release = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setitem(sys.modules, gate.__name__, gate)
    holder = {}

    def reply(_request):
        # The model answers while a force reload sits inside the plugin's register().
        gate.reload.set()
        holder["reload"] = threading.Thread(target=plugins.get_plugin_manager().discover_and_load,
                                            kwargs={"force": True})
        holder["reload"].start()
        assert gate.entered.wait(10)
        return Text(RAW, chunk_chars=6)

    with FakeLLMServer(reply) as server:
        redactor_home(home, server.base_url, hold=True)
        _write_plugin(home, (
            "import re\nimport hold_reload_gate as gate\n"
            "def redact(response_text, **_):\n"
            "    return re.sub(r'SECRET-\\d+', '[REDACTED]', response_text)\n"
            "def register(ctx):\n"
            "    if gate.reload.is_set():\n"
            "        gate.entered.set()\n"
            "        gate.release.wait(30)\n"
            "    ctx.register_hook('transform_llm_output', redact, hold_stream=True)\n"))
        plugins._reset_plugin_managers_for_tests()
        shown = []
        try:
            agent = _agent(server)
            agent.stream_delta_callback = shown.append
            result = agent.run_conversation("show config")
            assert not gate.release.is_set()  # the reload was still in flight at the final
            assert not [text for text in shown if text]
            assert result["final_response"] == REWRITE
        finally:
            gate.release.set()
            if holder.get("reload"):
                holder["reload"].join(15)
            plugins._reset_plugin_managers_for_tests()


def test_held_transform_that_raises_withholds_the_reply():
    from hermes_cli import plugins

    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer([Text(RAW)]) as server:
        redactor_home(home, server.base_url, hold=True)
        _write_plugin(home, (
            "def redact(response_text, **_):\n    raise RuntimeError('redactor down')\n"
            "def register(ctx):\n    ctx.register_hook('transform_llm_output', redact, hold_stream=True)\n"))
        plugins._reset_plugin_managers_for_tests()
        try:
            agent = _agent(server)
            shown = []
            agent.stream_delta_callback = shown.append
            result = agent.run_conversation("show config")
        finally:
            plugins._reset_plugin_managers_for_tests()
    assert RAW not in result["final_response"] and RAW not in str(result["pre_transform_response"])
    assert RAW not in str(result["messages"][-1].get("content"))
    assert not [text for text in shown if text]
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY
    assert result["final_response"] == HELD_TRANSFORM_FAILED_REPLY


def _assistant_commentary(request: dict) -> list:
    return [m for m in request["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]


def test_held_commentary_stays_out_of_history_but_in_the_model_context():
    from gateway.session_controls import display_projection
    from hermes_cli import plugins
    from hermes_state import SessionDB

    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer(redactor_turn() + [Text("ok")]) as server:
        redactor_home(home, server.base_url, hold=True)
        plugins._reset_plugin_managers_for_tests()
        db = SessionDB(db_path=home / "held-history.db")
        try:
            agent = _agent(server, session_id="held-history", session_db=db)
            shown = []
            agent.stream_delta_callback = shown.append
            agent.interim_assistant_callback = lambda text, **_: shown.append(text)
            assert agent.run_conversation("show config")["final_response"] == REWRITE
            assert not [text for text in shown if text]

            displayed = display_projection(db.get_messages("held-history"), home)
            assert COMMENTARY not in str(displayed)
            assert [row["text"] for row in displayed if row["role"] == "assistant"] == [REWRITE]
            # The tool card of the hidden row still renders, labelled from its call.
            assert [row["name"] for row in displayed if row["role"] == "tool"] == ["todo_list"]

            # Resume: the model gets the commentary exactly as it got it live (prompt-cache prefix).
            resumed = _agent(server, session_id="held-history", session_db=db)
            resumed.run_conversation("thanks", conversation_history=db.get_messages_as_conversation("held-history"))
            live, replayed = server.main_requests()[1], server.main_requests()[2]
            assert _assistant_commentary(replayed) == _assistant_commentary(live)
            assert _assistant_commentary(live)[0]["content"] == COMMENTARY
        finally:
            db.close()
            plugins._reset_plugin_managers_for_tests()


def test_codex_reply_that_failed_after_its_text_is_still_rewritten(monkeypatch):
    from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
    from gateway.session_controls import display_projection
    from hermes_cli import plugins
    from tests.agent.test_codex_app_server_integration import _make_codex_agent

    home = Path(os.environ["HERMES_HOME"])
    redactor_home(home, "http://127.0.0.1:9/v1", hold=True)
    plugins._reset_plugin_managers_for_tests()

    def run_turn(self, user_input, **kwargs):
        projected = [{"role": "assistant", "content": COMMENTARY}, {"role": "assistant", "content": RAW}]
        return TurnResult(final_text=RAW, projected_messages=projected, should_retire=True,
                          error="codex app-server subprocess exited unexpectedly",
                          thread_id="thread-held", turn_id="turn-held")

    monkeypatch.setattr(CodexAppServerSession, "run_turn", run_turn)
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-held")
    try:
        agent = _make_codex_agent(enabled_toolsets=["todo"])
        shown = []
        agent.stream_delta_callback = shown.append
        result = agent.run_conversation("show config")
    finally:
        plugins._reset_plugin_managers_for_tests()
    assert result["final_response"] == REWRITE
    assert result["pre_transform_response"] is None
    assert not [text for text in shown if text]
    displayed = display_projection(result["messages"], home)
    assert [row["text"] for row in displayed if row["role"] == "assistant"] == [REWRITE]
