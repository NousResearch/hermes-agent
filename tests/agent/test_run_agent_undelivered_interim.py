"""Undelivered tool-call text: with interim messages off and the text not fully streamed,
the tool-call row's API sidecar carries a marker so the next call knows the user never saw it.

Split from test_run_agent.py and test_run_agent_codex_responses.py, both over their line caps.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_run_agent import (
    _make_tool_defs,
    _mock_assistant_msg,
    _mock_response,
    _mock_tool_call,
)


@pytest.fixture(autouse=True)
def _mock_plugin_discovery(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)


@pytest.fixture()
def agent():
    """test_run_agent's ``agent`` fixture plus ``TestRunConversation._setup_agent``."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = False
    a.save_trajectories = False
    return a


class TestUndeliveredInterimMarker:
    @staticmethod
    def _tool_call_rows(api_messages):
        return [
            msg for msg in api_messages
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        ]

    def test_material_tool_call_content_is_recovered_when_interim_disabled(
        self, agent, caplog
    ):
        agent.platform = "gateway"
        agent.interim_assistant_callback = None
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        hidden_quote = (
            "Unit price = $128\n"
            "4 x $128 = $512\n"
            "Delivery = $7\n"
            "Total = $519"
        )
        reply = "When do you need them by?"
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content=hidden_quote, finish_reason="tool_calls", tool_calls=[tc]),
            _mock_response(content=reply, finish_reason="stop"),
        ]

        with (
            patch("model_tools.handle_function_call", return_value="delivery checked"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            caplog.at_level(logging.INFO, logger="agent.conversation_loop"),
        ):
            result = agent.run_conversation("how much for 4 items?")

        # The model's own reply is final: no extra reconciliation call.
        assert agent.client.chat.completions.create.call_count == 2
        assert result["api_calls"] == 2
        assert result["final_response"] == reply
        assert result["messages"][-1]["content"] == reply

        # The second request tells the model the quote never reached the user.
        from agent.turn_tool_round import UNDELIVERED_INTERIM_MARKER

        second_request = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
        (quote_row,) = self._tool_call_rows(second_request)
        assert quote_row["content"] == f"{hidden_quote}\n\n{UNDELIVERED_INTERIM_MARKER}"
        # The stored, user-visible row stays clean.
        (stored_row,) = self._tool_call_rows(result["messages"])
        assert stored_row["content"] == hidden_quote
        assert [
            msg["content"] for msg in result["messages"] if msg.get("role") == "user"
        ] == ["how much for 4 items?"]
        # Logged with its length, never its text.
        marked_logs = [r for r in caplog.records if "marked it for the model" in r.getMessage()]
        assert [(r.levelno, r.getMessage()) for r in marked_logs] == [(
            logging.INFO,
            "Tool-call content was not delivered to the user; marked it for the model "
            f"({len(hidden_quote)} chars)",
        )]
        assert not any("Total = $519" in r.getMessage() for r in caplog.records)

    def test_undelivered_interim_marker_replays_byte_identical_across_turns(
        self, agent, tmp_path
    ):
        from hermes_state import SessionDB
        from agent.turn_tool_round import UNDELIVERED_INTERIM_MARKER

        agent.platform = "gateway"
        agent.interim_assistant_callback = None
        db = SessionDB(db_path=tmp_path / "state.db")
        agent._session_db = db
        hidden_quote = "Unit price = $128\nTotal for 4 items = $519"
        agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content=hidden_quote,
                finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")],
            ),
            _mock_response(content="When do you need them by?", finish_reason="stop"),
            _mock_response(content="Noted, delivery by Friday.", finish_reason="stop"),
        ]

        try:
            with (
                patch("model_tools.handle_function_call", return_value="delivery checked"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                agent.run_conversation("how much for 4 items?")
                stored = db.get_messages_as_conversation(agent.session_id)
                agent.run_conversation("by Friday", conversation_history=stored)

            (stored_row,) = self._tool_call_rows(stored)
            assert stored_row["content"] == hidden_quote
            assert UNDELIVERED_INTERIM_MARKER not in json.dumps(
                [msg.get("content") for msg in stored]
            )

            calls = agent.client.chat.completions.create.call_args_list
            same_turn = self._tool_call_rows(calls[1].kwargs["messages"])
            next_turn = self._tool_call_rows(calls[2].kwargs["messages"])
            assert len(same_turn) == len(next_turn) == 1
            assert same_turn[0]["content"] == f"{hidden_quote}\n\n{UNDELIVERED_INTERIM_MARKER}"
            assert json.dumps(next_turn[0], sort_keys=True) == json.dumps(
                same_turn[0], sort_keys=True
            )
        finally:
            db.close()

    def test_every_undelivered_tool_round_carries_the_marker(self, agent):
        from agent.turn_tool_round import UNDELIVERED_INTERIM_MARKER

        agent.platform = "gateway"
        agent.interim_assistant_callback = None
        first_quote = "Unit price = $128\n4 x $128 = $512"
        second_quote = "Delivery = $7\nTotal = $519"
        agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content=first_quote,
                finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")],
            ),
            _mock_response(
                content=second_quote,
                finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c2")],
            ),
            _mock_response(content="When do you need them by?", finish_reason="stop"),
        ]

        with (
            patch("model_tools.handle_function_call", return_value="checked"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("how much for 4 items?")

        assert agent.client.chat.completions.create.call_count == 3
        assert result["final_response"] == "When do you need them by?"
        calls = agent.client.chat.completions.create.call_args_list
        second_request = self._tool_call_rows(calls[1].kwargs["messages"])
        third_request = self._tool_call_rows(calls[2].kwargs["messages"])
        assert [row["content"] for row in third_request] == [
            f"{first_quote}\n\n{UNDELIVERED_INTERIM_MARKER}",
            f"{second_quote}\n\n{UNDELIVERED_INTERIM_MARKER}",
        ]
        # The first round's row is byte-identical on every later call in the turn.
        assert json.dumps(second_request[0], sort_keys=True) == json.dumps(
            third_request[0], sort_keys=True
        )
        assert [row["content"] for row in self._tool_call_rows(result["messages"])] == [
            first_quote, second_quote,
        ]

    def test_delivered_interim_gets_no_marker(self, agent):
        from agent.turn_tool_round import UNDELIVERED_INTERIM_MARKER

        agent.platform = "gateway"
        delivered = []
        agent.interim_assistant_callback = (
            lambda text, already_streamed=False: delivered.append(text)
        )
        hidden_quote = "Unit price = $128\nTotal for 4 items = $519"
        agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content=hidden_quote,
                finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")],
            ),
            _mock_response(content="When do you need them by?", finish_reason="stop"),
        ]

        with (
            patch("model_tools.handle_function_call", return_value="delivery checked"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("how much for 4 items?")

        assert delivered == [hidden_quote]
        assert agent.client.chat.completions.create.call_count == 2
        assert result["final_response"] == "When do you need them by?"
        second_request = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
        (quote_row,) = self._tool_call_rows(second_request)
        assert quote_row["content"] == hidden_quote
        assert UNDELIVERED_INTERIM_MARKER not in json.dumps(second_request)

    def test_marker_does_not_depend_on_interim_emission_dedup(self, agent):
        """A tool-call row that repeats the previous ``incomplete`` row's text is only
        deduplicated for emission; its undelivered text is still marked."""
        from agent import turn_tool_round
        from agent.turn_tool_round import UNDELIVERED_INTERIM_MARKER

        agent.platform = "gateway"
        agent.interim_assistant_callback = None
        agent.stream_delta_callback = None
        text = "Unit price = $128 and total price = $519 for four items."
        agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content=text,
                finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")],
            ),
            _mock_response(content="When do you need them by?", finish_reason="stop"),
        ]
        stage = turn_tool_round.stage_tool_call_message
        dedup_flags = []

        def _stage_as_duplicate(*args, **kwargs):
            # Same result as a round whose previous row is an ``incomplete`` row with
            # identical visible text.
            assistant_msg, _ = stage(*args, **kwargs)
            dedup_flags.append(True)
            return assistant_msg, True

        with (
            patch.object(turn_tool_round, "stage_tool_call_message", _stage_as_duplicate),
            patch.object(agent, "_emit_interim_assistant_message") as emit,
            patch("model_tools.handle_function_call", return_value="checked"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("how much for 4 items?")

        assert dedup_flags == [True]
        emit.assert_not_called()
        assert agent.client.chat.completions.create.call_count == 2
        assert result["final_response"] == "When do you need them by?"
        second_request = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
        (row,) = self._tool_call_rows(second_request)
        assert row["content"] == f"{text}\n\n{UNDELIVERED_INTERIM_MARKER}"

    @pytest.mark.parametrize(
        ("streamed", "marked"),
        [
            ("", True),
            ("Unit price = $128 and total", True),
            ("Unit price = $128 and total price = $519 for four items.", False),
        ],
        ids=["not-streamed", "partly-streamed", "fully-streamed"],
    )
    def test_marker_follows_the_full_stream_check(self, agent, streamed, marked):
        from agent.turn_tool_round import UNDELIVERED_INTERIM_MARKER, mark_undelivered_interim

        agent.interim_assistant_callback = None
        agent._current_streamed_assistant_text = streamed
        text = "Unit price = $128 and total price = $519 for four items."
        assistant_message = _mock_assistant_msg(
            content=text, tool_calls=[_mock_tool_call(call_id="c1")]
        )
        assistant_msg = agent._build_assistant_message(assistant_message, "tool_calls")

        assert mark_undelivered_interim(
            agent, assistant_message=assistant_message, assistant_msg=assistant_msg
        ) is marked
        assert assistant_msg["content"] == text
        if marked:
            assert assistant_msg["api_content"] == f"{text}\n\n{UNDELIVERED_INTERIM_MARKER}"
        else:
            assert "api_content" not in assistant_msg

    def test_marker_leaves_content_and_reasoning_unchanged(self, agent):
        from agent.turn_tool_round import UNDELIVERED_INTERIM_MARKER

        agent.platform = "gateway"
        agent.interim_assistant_callback = None
        text = "Unit price = $128\nTotal for 4 items = $519"
        agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content=text,
                finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")],
                reasoning="Work out the total first.",
                reasoning_content="Work out the total first.",
            ),
            _mock_response(content="When do you need them by?", finish_reason="stop"),
        ]

        with (
            patch("model_tools.handle_function_call", return_value="checked"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("how much for 4 items?")

        (stored,) = self._tool_call_rows(result["messages"])
        assert stored["content"] == text
        assert stored["reasoning"] == "Work out the total first."
        assert stored["reasoning_content"] == "Work out the total first."
        assert stored["api_content"] == f"{text}\n\n{UNDELIVERED_INTERIM_MARKER}"
        second_request = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
        (sent,) = self._tool_call_rows(second_request)
        assert sent["content"] == f"{text}\n\n{UNDELIVERED_INTERIM_MARKER}"


def test_codex_tool_call_repeating_incomplete_text_is_still_marked(monkeypatch):
    """incomplete(text) -> tool call(same text) -> final, with no interim or stream
    delivery: the tool-call row is deduplicated for emission only, and its undelivered
    text still carries the marker in its API sidecar."""
    from agent.turn_tool_round import UNDELIVERED_INTERIM_MARKER
    from tests.agent.test_run_agent_codex_responses import (
        _build_agent,
        _codex_incomplete_message_response,
        _codex_message_response,
    )

    agent = _build_agent(monkeypatch)
    agent.interim_assistant_callback = None
    agent.stream_delta_callback = None
    text = "Unit price = $128 and total price = $519 for four items."
    message_and_call = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=text)],
            ),
            SimpleNamespace(
                type="function_call", id="fc_1", call_id="call_1", name="terminal", arguments="{}",
            ),
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
        status="completed",
        model="gpt-5-codex",
    )
    responses = [
        _codex_incomplete_message_response(text),
        message_and_call,
        _codex_message_response("When do you need them by?"),
    ]
    requests = []

    def _call(api_kwargs):
        requests.append(api_kwargs)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_interruptible_api_call", _call)
    monkeypatch.setattr("model_tools.handle_function_call", lambda *args, **kwargs: "ok")

    result = agent.run_conversation("how much for 4 items?")

    assert len(requests) == 3
    assert result["final_response"] == "When do you need them by?"
    rows = [
        (msg.get("finish_reason"), msg.get("content"), msg.get("api_content"))
        for msg in result["messages"]
        if msg.get("role") == "assistant"
    ]
    assert rows == [
        ("incomplete", text, None),
        ("tool_calls", text, f"{text}\n\n{UNDELIVERED_INTERIM_MARKER}"),
        ("stop", "When do you need them by?", None),
    ]
