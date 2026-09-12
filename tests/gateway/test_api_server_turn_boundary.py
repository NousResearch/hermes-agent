"""Responses API turn-boundary contract regressions."""

import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.context_compressor import (
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
    COMPRESSED_SUMMARY_METADATA_KEY,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _SUMMARY_END_MARKER,
)
from agent.agent_runtime_helpers import repair_message_sequence
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
try:
    from gateway.response_turn_boundary import (
        build_response_conversation_history,
        response_messages_turn_start_index,
        semantic_prefix_matches,
    )
except ModuleNotFoundError:  # RED witness before the focused policy module exists.
    build_response_conversation_history = APIServerAdapter._build_response_conversation_history
    response_messages_turn_start_index = APIServerAdapter._response_messages_turn_start_index

    def semantic_prefix_matches(messages, expected):
        return messages[:len(expected)] == expected


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


class TestTurnStartRobustnessE2E:
    @pytest.mark.asyncio
    async def test_durability_stamps_no_history_doubling(self, adapter):
        first_history = [
            {"role": "user", "content": "search the web for X"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "function": {"name": "web_search", "arguments": '{"query": "X"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "results..."},
            {"role": "assistant", "content": "Here is what I found about X."},
        ]
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Here is what I found about X.", "messages": list(first_history), "api_calls": 2},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp1 = await cli.post("/v1/responses", json={"model": "hermes-agent", "input": "search the web for X"})
            assert resp1.status == 200
            resp1_data = await resp1.json()

            user2 = "thanks, now just say hi"
            reshaped_prior = [dict(m) for m in first_history]
            reshaped_prior[0].update(timestamp="2026-08-27T00:00:00Z", _db_persisted=True)
            reshaped_prior[1].update(
                timestamp="2026-08-27T00:00:01Z", _db_persisted=True,
                reasoning="searched deliberately", finish_reason="tool_calls",
                codex_message_items=[{"id": "provider-message-1"}],
                codex_reasoning_items=[{"id": "provider-reasoning-1"}],
            )
            second_transcript = reshaped_prior + [
                {"role": "user", "content": user2},
                {"role": "assistant", "content": "hi"},
            ]
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "hi", "messages": second_transcript, "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp2 = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": user2, "previous_response_id": resp1_data["id"]},
                )
            assert resp2.status == 200
            resp2_data = await resp2.json()
            assert response_messages_turn_start_index(first_history, user2, {"messages": second_transcript}) == len(first_history) + 1
            assert [it["type"] for it in resp2_data["output"]] == ["message"]
            stored = adapter._response_store.get(resp2_data["id"])["conversation_history"]
            assert stored == second_transcript
            assert sum(1 for m in stored if m.get("role") == "user" and m.get("content") == "search the web for X") == 1
            assert sum(1 for m in stored if m.get("role") == "user" and m.get("content") == user2) == 1

    @pytest.mark.asyncio
    async def test_client_supplied_leading_system_history_preserved(self, adapter):
        leading_system = {"role": "system", "content": "You are a pirate."}
        prior = [leading_system, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "Ahoy!"}]
        user_message = "say bye"
        transcript = [dict(m) for m in prior] + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "Farewell, matey!"},
        ]
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Farewell, matey!", "messages": transcript, "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": user_message, "conversation_history": prior},
                )
            assert resp.status == 200
            data = await resp.json()
            assert [it["type"] for it in data["output"]] == ["message"]
            stored = adapter._response_store.get(data["id"])["conversation_history"]
            assert stored == prior + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "Farewell, matey!"},
            ]
            assert sum(1 for m in stored if m.get("role") == "system") == 1

    @pytest.mark.asyncio
    async def test_prior_history_with_suffix_only_tool_turn_preserves_tools_and_history(self, adapter):
        prior = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        user_message = "calculate 6*7"
        suffix = [
            {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "call_2", "function": {"name": "calculator", "arguments": '{"expression": "6*7"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "42"},
            {"role": "assistant", "content": "42"},
        ]
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "42", "messages": suffix, "api_calls": 2},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": user_message, "conversation_history": prior},
                )
            assert resp.status == 200
            data = await resp.json()
        assert [item["type"] for item in data["output"]] == ["function_call", "function_call_output", "message"]
        assert data["output"][0]["call_id"] == "call_2"
        stored = adapter._response_store.get(data["id"])["conversation_history"]
        assert stored == prior + [{"role": "user", "content": user_message}, *suffix]

    def test_tool_identity_prevents_false_prefix_and_keeps_current_call(self, adapter):
        prior = [{"role": "assistant", "content": "", "tool_calls": [{"id": "old", "function": {"name": "old_tool", "arguments": "{}"}}]}]
        current = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "new", "function": {"name": "new_tool", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "new", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        result = {"messages": current, "final_response": "done"}
        assert semantic_prefix_matches(current, prior) is False
        assert response_messages_turn_start_index(prior, "run new", result) == 0
        output = adapter._extract_output_items(result)
        assert [item["type"] for item in output] == ["function_call", "function_call_output", "message"]
        assert output[0]["call_id"] == output[1]["call_id"] == "new"

    def test_repaired_current_user_full_transcript_keeps_current_tool_trace(self, adapter):
        prior = [{"role": "user", "content": "historical request"}]
        repaired_current = {
            "role": "user",
            "content": f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nUser asked: 'historical request'\n\n{_SUMMARY_END_MARKER}\n\ncurrent request",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
            COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: True,
        }
        transcript = [
            {"role": "user", "content": "historical request [repaired]"}, repaired_current,
            {"role": "assistant", "content": "", "tool_calls": [{"id": "current-call", "function": {"name": "calculator", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "current-call", "content": "42"},
            {"role": "assistant", "content": "done"},
        ]
        result = {"messages": transcript, "final_response": "done", "_transcript_mode": "full"}
        start = response_messages_turn_start_index(prior, "current request", result)
        assert start == 2
        output = adapter._extract_output_items(result, start_index=start)
        assert [item["type"] for item in output] == ["function_call", "function_call_output", "message"]
        assert output[0]["call_id"] == output[1]["call_id"] == "current-call"
        assert build_response_conversation_history(prior, "current request", result, "done") == transcript
        completed_messages = adapter._turn_transcript_messages(prior, "current request", result)
        assert [message["role"] for message in completed_messages] == ["assistant", "tool", "assistant"]
        assert completed_messages[0]["tool_calls"][0]["id"] == "current-call"
        assert completed_messages[1]["tool_call_id"] == "current-call"

    @pytest.mark.asyncio
    async def test_duplicate_user_text_after_pass2_merge_keeps_only_current_call(self, adapter):
        user_message = "repeat this request"
        prior = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "old-call", "function": {"name": "old_tool", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "old-call", "content": "old"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "unfinished redirect"},
        ]
        transcript = [
            *copy.deepcopy(prior), {"role": "user", "content": user_message},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "current-call", "function": {"name": "current_tool", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "current-call", "content": "current"},
            {"role": "assistant", "content": "current result"},
        ]
        owner = MagicMock()
        owner._persist_user_message_idx = len(prior)
        assert repair_message_sequence(owner, transcript) == 1
        assert transcript[4]["content"] == "unfinished redirect\n\nrepeat this request"
        assert owner._persist_user_message_idx == 4
        result = {
            "final_response": "current result", "messages": transcript,
            "_transcript_mode": "full", "_current_turn_user_idx": owner._persist_user_message_idx,
        }
        assert response_messages_turn_start_index(prior, user_message, result) == 5
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                response = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": user_message, "conversation_history": prior},
                )
                assert response.status == 200
                data = await response.json()
        assert [item["type"] for item in data["output"]] == ["function_call", "function_call_output", "message"]
        assert data["output"][0]["call_id"] == "current-call"
        assert data["output"][1]["call_id"] == "current-call"
        assert all(item.get("call_id") != "old-call" for item in data["output"])

    @pytest.mark.asyncio
    async def test_normal_agent_run_marks_list_messages_as_full(self, adapter):
        transcript = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        mock_agent = MagicMock()
        mock_agent.run_conversation = MagicMock(return_value={"final_response": "hi", "messages": transcript})
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0
        mock_agent._persist_user_message_idx = 0
        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            result, _ = await adapter._run_agent(
                user_message="hello", conversation_history=[], session_id="session-full-contract",
            )
        assert result["messages"] == transcript
        assert result["_transcript_mode"] == "full"
        assert result["_current_turn_user_idx"] == 0

    @pytest.mark.parametrize(
        "actual, expected",
        [
            (["same"], ["same"]),
            ([{"content": "same"}], [{"content": "same"}]),
            ([{"role": "future-role", "content": "same"}], [{"role": "future-role", "content": "same"}]),
        ],
    )
    def test_malformed_messages_never_form_semantic_prefix(self, adapter, actual, expected):
        assert semantic_prefix_matches(actual, expected) is False
