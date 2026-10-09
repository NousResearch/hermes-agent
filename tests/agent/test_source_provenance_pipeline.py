"""Published end-to-end provenance proof across request construction and tool loops."""
import json
from hashlib import sha256
import pytest
from tests.agent.test_run_agent import agent as _agent_fixture
agent = _agent_fixture


@pytest.mark.parametrize("terminator", ["\n", "\n\n", ""])
@pytest.mark.parametrize("local_sdk", [False, True])
def test_source_provenance_survives_build_and_rebinds_next_tool_loop(
    agent, tmp_path, monkeypatch, terminator, local_sdk
):
    import httpx
    from openai import OpenAI
    from agent.llm_egress_runtime import dispatch_provider_request as _dispatch_provider_request
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        source_provenance_activation,
    )
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1234abcd")
    source = tmp_path / "source.py"
    source.write_text("first = 1\nsecond = 2" + terminator, encoding="utf-8")
    agent.provider = "nous"
    agent.base_url = (
        "http://127.0.0.1:11434/v1" if local_sdk
        else "https://inference-api.nousresearch.com/v1"
    )
    agent._base_url_lower = agent.base_url.lower()
    agent.api_mode = "chat_completions"
    agent.session_id = "session-1"
    agent._current_turn_id = "turn-1"
    agent._current_api_request_id = "turn-1:api:1"
    agent._llm_egress_policy_digest = sha256(b"policy").hexdigest()
    agent._llm_egress_state_dir = tmp_path / "egress"

    with source_provenance_activation(agent, "read_file"):
        result = read_file_tool(str(source), task_id="build-wire-read")
    metadata = attach_trusted_source_provenance_metadata(
        agent, "read_file", content=result
    )
    assert metadata is not None
    tool_message = make_tool_result_message(
        "read_file",
        result,
        "call_read_1",
        source_provenance=metadata,
    )
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_read_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        tool_message,
    ]

    captured = []
    def final_sdk_io(request):
        assert str(request.url) == agent.base_url + "/chat/completions"
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "synthetic-completion", "object": "chat.completion",
            "created": 0, "model": agent.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })
    with OpenAI(
        api_key="synthetic-key", base_url=agent.base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(final_sdk_io)),
    ) as sdk:
        for request_id in ("turn-1:api:2", "turn-1:api:3"):
            agent._current_api_request_id = request_id
            kwargs = agent._build_api_kwargs(history, tools_for_api=[])
            sidecar_before = json.dumps(kwargs["_hermes_source_provenance"], sort_keys=True)
            assert "_source_provenance" not in kwargs["messages"][1]
            _dispatch_provider_request(
                agent, kwargs, lambda request: sdk.chat.completions.create(**request)
            )
            assert json.dumps(kwargs["_hermes_source_provenance"], sort_keys=True) == sidecar_before

    assert len(captured) == 2
    for request in captured:
        assert "_hermes_source_provenance" not in request
        assert "_source_provenance" not in request["messages"][1]
        assert request["messages"][1]["content"] == result
    receipt_path = tmp_path / "egress" / "llm-egress-receipts.jsonl"
    if local_sdk:
        assert not receipt_path.exists()
        return
    receipts = [
        json.loads(line)
        for line in receipt_path.read_text().splitlines()
    ]
    assert [receipt["source_grant_count"] for receipt in receipts] == [1, 1]
    assert [receipt["source_segment_count"] for receipt in receipts] == [1, 1]



def test_forged_build_sidecar_fails_closed(agent, tmp_path, monkeypatch):
    from agent.llm_egress_runtime import dispatch_provider_request as _dispatch_provider_request
    from agent.llm_egress_firewall import EgressBlocked
    from agent.source_provenance_tools import (
        attach_trusted_source_provenance_metadata,
        source_provenance_activation,
    )
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.file_tools import read_file_tool

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1234abcd")
    source = tmp_path / "source.py"
    source.write_text("safe = True\n", encoding="utf-8")
    agent.provider = "nous"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    agent._base_url_lower = agent.base_url.lower()
    agent.api_mode = "chat_completions"
    agent.session_id = "session-1"
    agent._current_turn_id = "turn-1"
    agent._current_api_request_id = "turn-1:api:1"
    agent._llm_egress_policy_digest = sha256(b"policy").hexdigest()
    agent._llm_egress_state_dir = tmp_path / "egress"
    with source_provenance_activation(agent, "read_file"):
        result = read_file_tool(str(source), task_id="build-wire-forged")
    metadata = attach_trusted_source_provenance_metadata(
        agent, "read_file", content=result
    )
    assert metadata is not None
    message = make_tool_result_message(
        "read_file", result, "call_read_1", source_provenance=metadata
    )
    agent._current_api_request_id = "turn-1:api:2"
    kwargs = agent._build_api_kwargs([message], tools_for_api=[])
    kwargs["_hermes_source_provenance"][0]["content_sha256"] = "0" * 64

    with pytest.raises(EgressBlocked) as exc_info:
        _dispatch_provider_request(agent, kwargs, lambda _: None)
    assert "untrusted_provenance" in exc_info.value.decision.reason_codes



class TestAnthropicInterruptHandler:
    """_interruptible_api_call must handle Anthropic mode when interrupted."""


    def test_interruptible_anthropic_interrupt_never_closes_shared_client(self, agent, tmp_path):
        """#67142: a non-streaming Anthropic interrupt must abort the
        request-local client from the poll thread, never close/rebuild the
        shared _anthropic_client (which raced a live SSL BIO and corrupted an
        unrelated SQLite DB via TLS-FD recycling).

        Replaces the former source-reading assertion (which asserted the old,
        now-removed rebuild-on-interrupt behavior) with a behavior test.
        """
        import time
        from unittest.mock import MagicMock
        from agent.chat_completion_helpers import interruptible_api_call

        agent.api_mode = "anthropic_messages"
        agent.session_id = "interrupt-session"
        agent._current_turn_id = "interrupt-turn"
        agent._current_api_request_id = "interrupt-turn:api:1"
        agent._llm_egress_state_dir = tmp_path / "egress"
        agent._interrupt_requested = False
        agent._anthropic_client = MagicMock()
        agent._rebuild_anthropic_client = MagicMock()
        request_client = MagicMock()
        agent._create_request_anthropic_client = MagicMock(return_value=request_client)
        agent._abort_request_anthropic_client = MagicMock()
        agent._close_request_anthropic_client = MagicMock()

        def _create(_api_kwargs, *, client):
            assert client is request_client
            agent._interrupt_requested = True
            time.sleep(0.5)
            raise RuntimeError("forced close would have happened")

        agent._anthropic_messages_create = MagicMock(side_effect=_create)

        t0 = time.time()
        with pytest.raises(InterruptedError):
            interruptible_api_call(agent, {"model": "x", "messages": []})
        elapsed = time.time() - t0

        assert elapsed < 3.0, f"interrupt took {elapsed:.1f}s — should be near-instant"
        # The shared client is never closed/rebuilt from the poll thread.
        agent._anthropic_client.close.assert_not_called()
        agent._rebuild_anthropic_client.assert_not_called()
        # The poll (stranger) thread aborts the request-local client's socket.
        agent._abort_request_anthropic_client.assert_called_once_with(
            request_client, reason="interrupt_abort"
        )



@pytest.mark.parametrize("raise_in_dispatch", [False, True])
def test_live_tool_worker_scope_resets_across_provider_switches(
    agent, monkeypatch, raise_in_dispatch
):
    from contextvars import copy_context
    from concurrent.futures import ThreadPoolExecutor
    from agent.delegation_context import delegated_child_context
    from agent.llm_egress_runtime import is_protected_worker, protected_worker_scope
    from tests.agent.test_run_agent import _mock_tool_call, _mock_assistant_msg

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1234abcd")
    observed = []
    def final_tool_io(*args, **kwargs):
        observed.append(is_protected_worker())
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(copy_context().run, is_protected_worker).result() == observed[-1]
        if raise_in_dispatch:
            raise RuntimeError("final tool I/O failed")
        return "ok"
    monkeypatch.setattr("model_tools.handle_function_call", final_tool_io)
    for provider in ("nous", "custom", "nous"):
        agent.provider = provider
        calls = [_mock_tool_call(name="web_search", arguments="{}", call_id="scope_call")]
        agent._execute_tool_calls(_mock_assistant_msg(content="", tool_calls=calls), [], "scope-task")
        assert is_protected_worker() is False
    assert observed == [True, False, True]
    with protected_worker_scope(agent):
        assert is_protected_worker() is True
        with delegated_child_context(), protected_worker_scope(agent):
            assert is_protected_worker() is False
        assert is_protected_worker() is True
    assert is_protected_worker() is False
