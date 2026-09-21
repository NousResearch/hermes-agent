"""Operation metadata survives the real request builders and SDK serialization."""

import copy
import json

import httpx
import pytest
from openai import OpenAI

from agent.portal_tags import reset_conversation_context, set_conversation_context


def _capture_request(request):
    return httpx.Response(200, json=json.loads(request.content))


@pytest.mark.parametrize("wire", ["chat_completions", "codex_responses", "anthropic_messages"])
def test_main_requests_describe_the_operation_without_copying_content(wire):
    from agent.codex_responses_adapter import _preflight_codex_api_kwargs
    from run_agent import AIAgent

    messages = [
        {"role": "user", "content": "private user text"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "private tool output"},
    ]
    original = copy.deepcopy(messages)
    agent = AIAgent(
        provider="nous", model="hermes-3-70b", api_mode=wire,
        api_key="test", base_url="https://portal.example.test/v1", session_id="rotated-session",
        quiet_mode=True, skip_context_files=True, skip_memory=True, max_tokens=1024,
        save_trajectories=False, enabled_toolsets=[],
    )
    token = set_conversation_context("conversation-root")
    try:
        for history in (messages[:1], messages):
            kwargs = agent._build_api_kwargs(history)
            if wire == "codex_responses":
                kwargs = _preflight_codex_api_kwargs(kwargs)
            with httpx.Client(transport=httpx.MockTransport(_capture_request)) as http:
                if wire == "anthropic_messages":
                    from anthropic import Anthropic
                    with Anthropic(api_key="test", base_url="https://portal.example.test", http_client=http) as client:
                        body = client.messages.create(**kwargs).model_dump()
                else:
                    with OpenAI(api_key="test", base_url="https://portal.example.test/v1", http_client=http) as client:
                        resource = client.responses if wire == "codex_responses" else client.chat.completions
                        body = resource.create(**kwargs).model_dump()
            metadata = body["metadata"]
            assert metadata["hermes_activity"] == "assistant_chat"
            assert metadata["hermes_activity_id"] == "conversation-root"
            assert ("tool results" in metadata["hermes_purpose"]) == (len(history) > 1)
            assert "private" not in json.dumps(metadata)
        assert messages == original
    finally:
        reset_conversation_context(token)
        agent.close()


@pytest.mark.parametrize("task", ["compression", "title_generation", "vision", "private plugin task"])
def test_auxiliary_requests_keep_task_and_conversation_across_wires(task):
    from agent.auxiliary_client import _build_call_kwargs, _CodexCompletionsAdapter
    from agent.codex_responses_adapter import _preflight_codex_api_kwargs

    token = set_conversation_context("conversation-root")
    try:
        kwargs = _build_call_kwargs(
            "nous", "hermes-3-70b", [{"role": "user", "content": "private input"}], task=task,
            extra_body={"metadata": {"caller_label": "retained"}},
        )
        with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(_capture_request))) as client:
            chat_body = client.chat.completions.create(**kwargs).model_dump()
            responses_kwargs, _, _ = _CodexCompletionsAdapter(client, "hermes-3-70b")._build_responses_kwargs(kwargs)
            responses_body = client.responses.create(**_preflight_codex_api_kwargs(responses_kwargs)).model_dump()
        metadata = chat_body["metadata"]
        assert metadata["caller_label"] == "retained"
        assert responses_body["metadata"] == metadata
        assert metadata["hermes_activity"] == ("auxiliary" if task.startswith("private") else task)
        assert metadata["hermes_activity_id"] == "conversation-root"
        assert metadata["hermes_purpose"] != "Respond to the user's latest message."
        assert "private" not in json.dumps(metadata)
        other_kwargs = _build_call_kwargs("openai", "gpt-4.1", [], task=task)
        assert "metadata" not in other_kwargs.get("extra_body", {})
    finally:
        reset_conversation_context(token)
