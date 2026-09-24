"""Request-boundary regressions for reloaded Skill resources."""
from __future__ import annotations

import json
import os
from pathlib import Path
from contextlib import suppress

import httpx
import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


MODEL = "gpt-5.4"
MAIN_BODY = "# Main skill instructions\n\n" + ("MAIN-RESOURCE-EXACT-CONTENT " * 900)
REFERENCE_BODY = "# Supporting file instructions\n\n" + ("REFERENCE-RESOURCE-EXACT-CONTENT " * 900)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "isolated-hermes-home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _contains_decoded(value, expected: str) -> bool:
    """Find exact text through the SDK JSON and nested JSON tool-result encodings."""
    if isinstance(value, str):
        if expected in value:
            return True
        try:
            return _contains_decoded(json.loads(value), expected)
        except (json.JSONDecodeError, TypeError):
            return False
    if isinstance(value, dict):
        return any(_contains_decoded(v, expected) for v in value.values())
    if isinstance(value, list):
        return any(_contains_decoded(v, expected) for v in value)
    return False


def _response(output, call_index: int, *, input_tokens: int = 70_000):
    return {
        "id": f"resp_skill_reload_{call_index}",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": MODEL,
        "output": output,
        # Synthetic usage selects the pressure scenario without contacting a provider.
        "usage": {"input_tokens": input_tokens, "output_tokens": 8, "total_tokens": input_tokens + 8},
    }


def _function_call(call_id: str, args: dict):
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "skill_view",
        "arguments": json.dumps(args),
    }


def _stream_response(response):
    events = [
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": response["output"][0],
        },
        {"type": "response.completed", "response": response},
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        content=content + "data: [DONE]\n\n",
    )


def _patch_synthetic_http(monkeypatch, AIAgent, auxiliary_client, respond):
    def http_client(*_args, async_mode=False, **_kwargs):
        cls = httpx.AsyncClient if async_mode else httpx.Client
        return cls(transport=httpx.MockTransport(respond))

    # Keep OpenAI SDK and Hermes transports real; substitute only outbound HTTP.
    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client", staticmethod(http_client))
    monkeypatch.setattr(
        auxiliary_client,
        "_openai_http_client_kwargs",
        lambda _url, *, async_mode=False: {"http_client": http_client(async_mode=async_mode)},
    )


def _make_agent(*, max_iterations=3):
    from run_agent import AIAgent

    return AIAgent(
        model=MODEL,
        provider="custom",
        api_mode="codex_responses",
        base_url="https://mock.openai.test/v1",
        api_key="synthetic-test-key",
        enabled_toolsets=["skills"],
        max_iterations=max_iterations,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _write_skill(hermes_home, *, with_reference=False):
    skill_name = "delivery-fixture"
    skill_dir = hermes_home / "skills" / skill_name
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: Request-boundary fixture.\n---\n{MAIN_BODY}",
        encoding="utf-8",
    )
    if with_reference:
        (skill_dir / "references" / "guide.md").write_text(REFERENCE_BODY, encoding="utf-8")
    return skill_name


def test_skill_reload_resource_reaches_request_without_pressure(hermes_home, tmp_path, monkeypatch):
    """Ordinary turns carry the exact successful reload through the SDK request."""
    from agent import auxiliary_client
    from run_agent import AIAgent

    skill_name = _write_skill(hermes_home)
    requests = []

    def respond(request: httpx.Request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/responses") and payload.get("tools"):
            requests.append(payload)
            output = (
                [_function_call("ordinary-main", {"name": skill_name})]
                if len(requests) == 1
                else [{"type": "message", "id": "msg_ordinary_final", "role": "assistant",
                       "status": "completed", "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}]
            )
            return _stream_response(_response(output, len(requests), input_tokens=128))
        raise AssertionError(f"unexpected synthetic provider endpoint: {request.url}")

    _patch_synthetic_http(monkeypatch, AIAgent, auxiliary_client, respond)
    agent = _make_agent(max_iterations=2)
    agent._persist_session = lambda *_args, **_kwargs: None
    agent._save_trajectory = lambda *_args, **_kwargs: None
    agent._cleanup_task_resources = lambda *_args, **_kwargs: None
    compressor = agent.context_compressor
    compressor.threshold_tokens = 60_000
    try:
        result = agent.run_conversation("Read the fixture skill.")
        pending_after_turn = compressor.pending_skill_view_results()
    finally:
        with suppress(Exception):
            agent.close()

    assert result.get("completed") is True, result
    assert len(requests) == 2
    assert _contains_decoded(requests[1], MAIN_BODY)
    assert not pending_after_turn


def test_unfittable_skill_reload_fails_closed_without_oversized_request_or_loop(
    hermes_home, tmp_path, monkeypatch
):
    """An impossible hard window returns a typed failure rather than a retry loop."""
    from agent import auxiliary_client
    from run_agent import AIAgent

    skill_name = _write_skill(hermes_home)
    requests = []
    compressor_holder = {}

    def respond(request: httpx.Request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/responses") and payload.get("tools"):
            requests.append(payload)
            if len(requests) == 1:
                # The initial request fits; discover an impossible ceiling before tool delivery.
                compressor_holder["compressor"].context_length = 128
                output = [_function_call("unfittable-main", {"name": skill_name})]
                return _stream_response(_response(output, 1))
            raise AssertionError("Agent sent a doomed/retried main request after the hard limit")
        raise AssertionError(f"unexpected synthetic provider endpoint: {request.url}")

    _patch_synthetic_http(monkeypatch, AIAgent, auxiliary_client, respond)
    agent = _make_agent(max_iterations=4)
    setattr(agent, "compression_enabled", False)
    agent._persist_session = lambda *_args, **_kwargs: None
    agent._save_trajectory = lambda *_args, **_kwargs: None
    agent._cleanup_task_resources = lambda *_args, **_kwargs: None
    compressor = getattr(agent, "context_compressor")
    compressor_holder["compressor"] = compressor
    try:
        result = agent.run_conversation("Read the fixture skill.")
        pending_after_turn = compressor.pending_skill_view_results()
    finally:
        with suppress(Exception):
            agent.close()

    assert result.get("completed") is False, result
    assert result.get("failed") is True, result
    assert result.get("turn_exit_reason") == "skill_reload_context_budget_exceeded", result
    assert "cannot fit within the model context window" in result.get("final_response", "")
    assert len(requests) == 1, f"expected no post-reload model request or loop, got {len(requests)}"
    assert pending_after_turn, "unfitted skill was forgotten despite no request carrying it"


@pytest.mark.parametrize("middleware_stage", ["llm_request", "llm_execution"])
@pytest.mark.parametrize("streaming", [True, False])
def test_execution_middleware_cannot_send_a_request_missing_reloaded_skill(
    hermes_home, tmp_path, monkeypatch, middleware_stage, streaming
):
    """Check the final middleware payload, not the earlier request-builder payload."""
    from copy import deepcopy

    from agent import auxiliary_client
    from hermes_cli.plugins import get_plugin_manager
    from run_agent import AIAgent

    skill_name = _write_skill(hermes_home)
    requests = []
    dropped = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        output = (
            [_function_call("middleware-main", {"name": skill_name})]
            if len(requests) == 1
            else [{"type": "message", "id": "msg_unprotected", "role": "assistant",
                   "status": "completed", "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}]
        )
        response = _response(output, len(requests), input_tokens=128)
        return _stream_response(response) if payload.get("stream") else httpx.Response(200, json=response)

    def drop_reloaded_body(*, request, next_call=None, **_context):
        payload = deepcopy(request)
        for item in payload.get("input", []):
            if item.get("type") == "function_call_output" and _contains_decoded(item, MAIN_BODY):
                item["output"] = "Resource removed by test execution middleware."
                dropped.append(True)
        return next_call(payload) if next_call is not None else {"request": payload}

    _patch_synthetic_http(monkeypatch, AIAgent, auxiliary_client, respond)
    agent = _make_agent(max_iterations=2)
    agent._persist_session = lambda *_args, **_kwargs: None
    agent._save_trajectory = lambda *_args, **_kwargs: None
    agent._cleanup_task_resources = lambda *_args, **_kwargs: None
    setattr(agent, "_disable_streaming", not streaming)
    monkeypatch.setitem(get_plugin_manager()._middleware, middleware_stage, [drop_reloaded_body])
    try:
        result = agent.run_conversation("Read the fixture skill.")
        pending_after_turn = getattr(agent, "context_compressor").pending_skill_view_results()
    finally:
        with suppress(Exception):
            agent.close()

    assert dropped, "fixture did not modify the actual execution middleware payload"
    assert len(requests) == 1, "a request missing the reloaded Skill escaped after middleware"
    assert result.get("completed") is False, result
    assert result.get("failed") is True, result
    assert result.get("turn_exit_reason") == "skill_reload_delivery_blocked", result
    assert result.get("api_calls") == 1, result
    assert "No model request was sent" in result.get("final_response", "")
    assert pending_after_turn, "undelivered skill was incorrectly acknowledged"


def test_skill_reload_resources_reach_distinct_requests_under_pressure_then_reclaim(
    hermes_home, tmp_path, monkeypatch
):
    """A main Skill and a file_path reload each reach a real serialized request once."""
    from agent import auxiliary_client
    from run_agent import AIAgent

    skill_name = _write_skill(hermes_home, with_reference=True)

    requests = []

    def respond(request: httpx.Request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/responses") and payload.get("tools"):
            requests.append(payload)
            if len(requests) == 1:
                output = [_function_call("main", {"name": skill_name})]
            elif len(requests) == 2:
                output = [_function_call("reference", {"name": skill_name, "file_path": "references/guide.md"})]
            else:
                output = [{
                    "type": "message",
                    "id": "msg_final",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
                }]
            return _stream_response(_response(output, len(requests)))
        if request.url.path.endswith("/responses"):
            summary = [{
                "type": "message",
                "id": "msg_summary",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Synthetic prior-context summary.", "annotations": []}],
            }]
            return _stream_response(_response(summary, 0))
        if request.url.path.endswith("/chat/completions"):
            model = payload.get("model", "synthetic-compressor")
            if payload.get("stream"):
                chunks = [
                    {"id": "chatcmpl-synthetic", "object": "chat.completion.chunk", "created": 0,
                     "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Synthetic prior-context summary."}, "finish_reason": None}]},
                    {"id": "chatcmpl-synthetic", "object": "chat.completion.chunk", "created": 0,
                     "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=content + "data: [DONE]\n\n")
            return httpx.Response(200, json={
                "id": "chatcmpl-synthetic", "object": "chat.completion", "created": 0, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Synthetic prior-context summary."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5},
            })
        raise AssertionError(f"unexpected synthetic provider endpoint: {request.url}")

    _patch_synthetic_http(monkeypatch, AIAgent, auxiliary_client, respond)
    agent = _make_agent(max_iterations=3)
    agent._persist_session = lambda *_args, **_kwargs: None
    agent._save_trajectory = lambda *_args, **_kwargs: None
    agent._cleanup_task_resources = lambda *_args, **_kwargs: None
    compressor = agent.context_compressor
    agent.compression_enabled = True
    compressor.threshold_tokens = 60_000
    compressor.tail_token_budget = 100
    compressor.protect_last_n = 20

    history = []
    for index in range(5):
        history.extend([
            {"role": "user", "content": f"prior question {index}"},
            {"role": "assistant", "content": f"prior answer {index}"},
        ])

    try:
        result = agent.run_conversation(
            "Read the fixture skill and its supporting guide.", conversation_history=history
        )
        pending_after_turn = compressor.pending_skill_view_results()
    finally:
        with suppress(Exception):
            agent.close()

    if capture_dir := os.environ.get("SKILL_RELOAD_CAPTURE_DIR"):
        capture_path = Path(capture_dir)
        capture_path.mkdir(parents=True, exist_ok=True)
        for index, payload in enumerate(requests, 1):
            (capture_path / f"codex-request-{index}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (capture_path / "pending-after-run.json").write_text(
            json.dumps(pending_after_turn, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    assert result.get("completed") is True, result
    assert len(requests) == 3
    assert _contains_decoded(requests[1], MAIN_BODY), "main SKILL.md body missing from the next serialized request"
    assert not _contains_decoded(requests[1], REFERENCE_BODY), "file_path resource was conflated with the main Skill"
    assert _contains_decoded(requests[2], REFERENCE_BODY), "file_path body missing from its next serialized request"
    assert not _contains_decoded(requests[2], MAIN_BODY), "delivered main Skill body was not reclaimed before the next request"
    assert not pending_after_turn, f"resource remained pending after a request carried it: {pending_after_turn}"

    reclaimed, reclaimed_count = compressor._prune_old_tool_results(
        result["messages"], protect_tail_count=5, protect_tail_tokens=100, min_prune_chars=8_000
    )
    assert reclaimed_count > 0
    assert not _contains_decoded(reclaimed, MAIN_BODY)
    assert not _contains_decoded(reclaimed, REFERENCE_BODY), "normal pressure reclamation still treated delivered file_path content as pending"
