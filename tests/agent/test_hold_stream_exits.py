"""Every exit of a held turn (``hold_stream=True`` transform) goes through the turn-start transform
snapshot.

Through a real ``AIAgent`` + ``SessionDB`` against the loopback model: a reply capped by the output
limit on every continuation, an iteration-budget summary, and a provider error that ends the turn
reach the user (and the stored row) only as the rewrite.
"""

import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall
from tests.fakes.redactor_plugin import RAW, REWRITE, redactor_home


def _agent(server, db, session_id: str, **kwargs):
    from run_agent import AIAgent

    return AIAgent(base_url=server.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                   quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["todo"],
                   session_id=session_id, session_db=db, **kwargs)


@pytest.fixture
def held(tmp_path):
    """``run(script, plugin_body=None, **agent_kwargs) -> (result, shown, displayed history, server, db)``
    for one held turn."""
    from gateway.session_controls import display_projection
    from hermes_cli import plugins
    from hermes_state import SessionDB

    home = Path(os.environ["HERMES_HOME"])
    opened = []

    def run(script, plugin_body=None, **kwargs):
        server = FakeLLMServer(script)
        server.start()
        opened.append(server)
        redactor_home(home, server.base_url, hold=True)
        if plugin_body:
            (home / "plugins" / "redactor" / "__init__.py").write_text(plugin_body, encoding="utf-8")
        plugins._reset_plugin_managers_for_tests()
        db = SessionDB(db_path=tmp_path / "held.db")
        opened.append(db)
        agent = _agent(server, db, "held-exit", **kwargs)
        shown = []
        agent.stream_delta_callback = shown.append
        agent.interim_assistant_callback = lambda text, **_: shown.append(text)
        result = agent.run_conversation("show config")
        displayed = display_projection(db.get_messages("held-exit"), home)
        return result, [text for text in shown if text], displayed, server, db

    yield run
    for item in opened:
        (item.stop if isinstance(item, FakeLLMServer) else item.close)()
    plugins._reset_plugin_managers_for_tests()


def _assistant_texts(displayed) -> list:
    return [row["text"] for row in displayed if row["role"] == "assistant"]


def test_reply_capped_on_every_continuation_is_still_rewritten(held):
    # Four finish_reason="length" responses: the ceiling exit keeps the stitched partial.
    result, shown, displayed, _, _ = held([Text(RAW, finish_reason="length")] * 4)
    assert result.get("partial") and "SECRET-" not in result["final_response"]
    assert "[REDACTED]" in result["final_response"]
    assert result.get("pre_transform_response") is None
    assert shown == []
    assert "SECRET-" not in str(displayed) and _assistant_texts(displayed) == [result["final_response"]]


def test_iteration_budget_summary_is_stored_as_the_rewrite(held):
    result, shown, displayed, _, _ = held([ToolCall("todo_list", {}), Text(RAW)], max_iterations=1)
    assert result["final_response"] == REWRITE
    assert shown == []
    assert "SECRET-" not in str(displayed) and REWRITE in _assistant_texts(displayed)


_TAGGING_PLUGIN = (
    "def tag(response_text, **_):\n    return '[filtered] ' + response_text\n"
    "def register(ctx):\n    ctx.register_hook('transform_llm_output', tag, hold_stream=True)\n"
)


def test_a_provider_error_exit_also_runs_the_held_transform(held):
    result, shown, _, _, _ = held([Error(status=400, message="bad request")], plugin_body=_TAGGING_PLUGIN)
    assert result.get("failed") and result["final_response"].startswith("[filtered] ")
    assert shown == []


def test_provider_error_after_a_capped_fragment_delivers_only_the_rewrite(held):
    # The error exit prepends the fragment the turn produced; held, the user never saw it raw.
    result, shown, displayed, _, _ = held([Text(RAW, finish_reason="length"), Error(status=400, message="bad request")])
    assert result.get("failed") and REWRITE in result["final_response"]
    assert "SECRET-" not in result["final_response"] and shown == []
    assert "SECRET-" not in str(displayed) and REWRITE in _assistant_texts(displayed)
