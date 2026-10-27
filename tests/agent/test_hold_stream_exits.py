"""Every exit of a held turn (``hold_stream=True`` transform) goes through the turn-start transform
snapshot, and every assistant row the user did not get stays out of history.

Through a real ``AIAgent`` + ``SessionDB`` against the loopback model: a reply capped by the output
limit on every continuation, a continuation that recovers, an iteration-budget summary, and a
provider error that ends the turn all reach the user (and the stored row) only as the rewrite; an
answer a late language-server verdict sends back for another try, and commentary on a call to an
unknown tool, are hidden in every history load while the model keeps them byte-for-byte.
"""

import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall
from tests.fakes.providers.openai_responses import FakeResponsesServer, Message, Turn
from tests.fakes.redactor_plugin import COMMENTARY, RAW, REWRITE, redactor_home

ATTEMPT = "Done, the deploy key SECRET-321 is in place."


def _agent(server, db, session_id: str, **kwargs):
    from run_agent import AIAgent

    kwargs.setdefault("enabled_toolsets", ["todo"])
    return AIAgent(base_url=server.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                   quiet_mode=True, skip_context_files=True, skip_memory=True,
                   session_id=session_id, session_db=db, **kwargs)


@pytest.fixture
def held(tmp_path):
    """``run(script, plugin_body=None, server_cls=FakeLLMServer, **agent_kwargs)
    -> (result, shown, displayed history, server, db)`` for one held turn."""
    from gateway.session_controls import display_projection
    from hermes_cli import plugins
    from hermes_state import SessionDB

    home = Path(os.environ["HERMES_HOME"])
    opened = []

    def run(script, plugin_body=None, server_cls=FakeLLMServer, **kwargs):
        server = server_cls(script).__enter__()
        opened.append(lambda: server.__exit__(None, None, None))
        redactor_home(home, server.base_url, hold=True)
        if plugin_body:
            (home / "plugins" / "redactor" / "__init__.py").write_text(plugin_body, encoding="utf-8")
        plugins._reset_plugin_managers_for_tests()
        db = SessionDB(db_path=tmp_path / "held.db")
        opened.append(db.close)
        agent = _agent(server, db, "held-exit", **kwargs)
        shown = []
        agent.stream_delta_callback = shown.append
        agent.interim_assistant_callback = lambda text, **_: shown.append(text)
        result = agent.run_conversation("show config")
        displayed = display_projection(db.get_messages("held-exit"), home)
        return result, [text for text in shown if text], displayed, server, db

    yield run
    for close in opened:
        close()
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


def test_recovered_continuation_shows_only_the_rewritten_whole(held):
    cut = RAW.index("the backup")  # at a space: the fragments join back byte-for-byte
    first, rest = RAW[:cut], RAW[cut:]
    result, shown, displayed, _, _ = held([Text(first, finish_reason="length"), Text(rest)])
    assert result["final_response"] == REWRITE
    assert shown == []
    assert _assistant_texts(displayed) == [REWRITE]


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


def test_answer_sent_back_by_a_late_lsp_verdict_stays_out_of_history(held, monkeypatch):
    from agent.lsp import deferred

    verdicts = [[("app.py", "app.py:3: error: name 'deploy' is not defined")]]
    monkeypatch.setattr(deferred, "settle_for_agent", lambda agent, **_: verdicts.pop() if verdicts else [])
    result, shown, displayed, server, db = held([Text(ATTEMPT), Text(RAW)])
    assert result["final_response"] == REWRITE
    assert shown == []
    assert "SECRET-" not in str(displayed) and _assistant_texts(displayed) == [REWRITE]

    # The model saw its attempted answer, and gets it byte-for-byte again on resume (cached prefix).
    live = [m for m in server.main_requests()[1]["messages"] if m["role"] == "assistant"]
    assert [m["content"] for m in live] == [ATTEMPT]
    resumed = _agent(server, db, "held-exit")
    resumed.run_conversation("thanks", conversation_history=db.get_messages_as_conversation("held-exit"))
    replayed = [m for m in server.main_requests()[2]["messages"] if m["role"] == "assistant"]
    assert replayed[0] == live[0]


def test_reply_reused_from_held_commentary_is_not_reported_as_already_shown(held):
    # Text beside a housekeeping call becomes the reply when the follow-up is empty. Held, that text
    # was never shown, so the rewrite must not be flagged as previewed (the CLI would print nothing).
    result, shown, displayed, _, _ = held([ToolCall("todo_list", {}, text=RAW), Text("")])
    assert result["final_response"] == REWRITE and shown == []
    assert not result["response_previewed"] and not result["response_reused"]
    assert "SECRET-" not in str(displayed) and REWRITE in _assistant_texts(displayed)


def test_commentary_on_an_unknown_tool_call_stays_out_of_history(held):
    result, shown, displayed, _, _ = held([ToolCall("no_such_tool", {}, text=COMMENTARY), Text(RAW)])
    assert result["final_response"] == REWRITE
    assert shown == []
    assert COMMENTARY not in str(displayed) and _assistant_texts(displayed) == [REWRITE]


_VERIFYING_PLUGIN = (
    "import re\n"
    "def redact(response_text, **_):\n    return re.sub(r'SECRET-\\d+', '[REDACTED]', response_text)\n"
    "def verify(**_):\n    return {'action': 'continue', 'message': 'Run the tests before you finish.'}\n"
    "def register(ctx):\n"
    "    ctx.register_hook('transform_llm_output', redact, hold_stream=True)\n"
    "    ctx.register_hook('pre_verify', verify)\n"
)


def test_answer_kept_as_the_budget_fallback_is_shown_in_history_as_the_rewrite(held, tmp_path):
    # pre_verify sends the answer back on the last allowed call: the turn ends on that saved answer.
    edit = ToolCall("write_file", {"path": str(tmp_path / "app.py"), "content": "x = 1\n"})
    result, shown, displayed, server, db = held(
        [edit, Text(ATTEMPT)], plugin_body=_VERIFYING_PLUGIN, enabled_toolsets=["file"], max_iterations=2)
    rewrite = ATTEMPT.replace("SECRET-321", "[REDACTED]")
    assert result["final_response"] == rewrite and shown == []
    assert "SECRET-" not in str(displayed) and _assistant_texts(displayed) == [rewrite]
    # The model keeps the answer it wrote: a resumed turn replays it unchanged.
    resumed = _agent(server, db, "held-exit", enabled_toolsets=["file"])
    resumed.run_conversation("thanks", conversation_history=db.get_messages_as_conversation("held-exit"))
    replayed = [m for m in server.main_requests()[-1]["messages"] if m["role"] == "assistant" and m.get("content")]
    assert [m["content"] for m in replayed] == [ATTEMPT]


def test_codex_commentary_riding_a_held_final_stays_out_of_history(held):
    turn = Turn([Message(COMMENTARY, phase="commentary"), Message(RAW, phase="final_answer")])
    result, shown, displayed, _, _ = held([turn], server_cls=FakeResponsesServer, api_mode="codex_responses")
    assert result["final_response"] == REWRITE and shown == []
    assert _assistant_texts(displayed) == [REWRITE]
    # The provider items stay on the row for replay; no rendered field carries the commentary.
    rendered = [row.get(key) for row in displayed for key in ("text", "display_content", "display_commentary")]
    assert COMMENTARY not in str(rendered)
