"""End-to-end turn containment for required plugin lifecycle hooks."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

import hermes_cli.plugins as plugins_mod
from hermes_cli.required_lifecycle import (
    REQUIRED_LIFECYCLE_FAILURE_TEXT,
    quarantine_required_provider_fields,
)
from run_agent import AIAgent


REQUIRED = {
    "pre_llm_call": ["behavioral.pre_llm.v1"],
    "pre_tool_call": ["behavioral.pre_tool.v1"],
    "post_tool_call": ["behavioral.post_tool.v1"],
    "transform_llm_output": ["behavioral.output.v1"],
}


def _response(text: str, *, finish_reason: str = "stop", tool_calls=None):
    message = SimpleNamespace(
        content=text,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _install_required_plugin(
    home: Path,
    pre_llm_expr: str,
    output_expr: str,
    *,
    post_tool_expr: str = 'required_hook_result("behavioral.post_tool.v1", None)',
    pre_tool_expr: str = 'required_hook_result("behavioral.pre_tool.v1", {})',
) -> None:
    plugin = home / "plugins" / "atlas"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "atlas", "version": "1.0.0"}),
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "from hermes_cli.plugins import required_hook_result\n"
        "def register(ctx):\n"
        "    ctx.register_hook(\"pre_llm_call\", lambda **kw: "
        f"{pre_llm_expr}, registration_id=\"behavioral.pre_llm.v1\")\n"
        "    ctx.register_hook(\"pre_tool_call\", lambda **kw: "
        f"{pre_tool_expr}, "
        "registration_id=\"behavioral.pre_tool.v1\")\n"
        "    ctx.register_hook(\"post_tool_call\", lambda **kw: "
        f"{post_tool_expr}, "
        "registration_id=\"behavioral.post_tool.v1\")\n"
        "    ctx.register_hook(\"transform_llm_output\", lambda **kw: "
        f"{output_expr}, registration_id=\"behavioral.output.v1\")\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "enabled": ["atlas"],
                    "required_lifecycle_hooks": {"atlas": REQUIRED},
                }
            }
        ),
        encoding="utf-8",
    )


def _agent() -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            session_id="required-lifecycle-test",
            api_key="test-key-1234567890",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=2,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._fallback_chain = []
    agent._disable_streaming = True
    return agent


def test_required_pre_llm_failure_stops_before_provider_call(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        '(_ for _ in ()).throw(RuntimeError("raw secret"))',
        'required_hook_result("behavioral.output.v1", None)',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    calls = []
    agent._interruptible_api_call = lambda kwargs: calls.append(kwargs) or _response(
        "raw answer"
    )

    result = agent.run_conversation("do work", conversation_history=[], task_id="task-1")

    assert calls == []
    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assert "raw secret" not in str(result)


def test_required_output_is_buffered_transformed_then_persisted(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", "safe answer")',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    stream = []
    deltas = []
    delta_callback = deltas.append
    agent.stream_delta_callback = delta_callback

    def provider(_kwargs):
        assert agent._stream_callback is None
        assert agent.stream_delta_callback is None
        return _response("raw answer")

    agent._interruptible_api_call = provider
    flushed = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        flushed.append(copy.deepcopy(messages))
    )
    result = agent.run_conversation(
        "do work",
        conversation_history=[],
        task_id="task-1",
        stream_callback=stream.append,
    )

    assert stream == []
    assert deltas == []
    assert agent.stream_delta_callback is delta_callback
    assert result["final_response"] == "safe answer"
    assert result["pre_transform_response"] is None
    assert result["messages"][-1]["content"] == "safe answer"
    assert all(message.get("content") != "raw answer" for message in result["messages"])
    assert all(
        message.get("content") != "raw answer"
        for snapshot in flushed
        for message in snapshot
    )


def test_required_output_failure_replaces_raw_text_before_persistence(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        '(_ for _ in ()).throw(RuntimeError("raw transform secret"))',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent._interruptible_api_call = lambda _kwargs: _response("raw answer")

    result = agent.run_conversation(
        "do work",
        conversation_history=[],
        task_id="task-1",
    )

    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assert result["pre_transform_response"] is None
    assert all(message.get("content") != "raw answer" for message in result["messages"])
    assert "raw transform secret" not in str(result)


@pytest.mark.parametrize(
    ("output_expr", "expected", "failed"),
    (
        (
            'required_hook_result("behavioral.output.v1", "safe partial")',
            "safe partial",
            False,
        ),
        (
            '(_ for _ in ()).throw(RuntimeError("raw ceiling secret"))',
            REQUIRED_LIFECYCLE_FAILURE_TEXT,
            True,
        ),
    ),
)
def test_required_output_authorizes_length_ceiling_before_any_persistence(
    tmp_path, monkeypatch, output_expr, expected, failed
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        output_expr,
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.max_iterations = 6
    fragments = [
        "raw ceiling fragment one ",
        "raw ceiling fragment two ",
        "raw ceiling fragment three ",
        "raw ceiling fragment four",
    ]
    responses = iter(_response(part, finish_reason="length") for part in fragments)
    agent._interruptible_api_call = lambda _kwargs: next(responses)
    persisted = []
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )

    result = agent.run_conversation(
        "write a long response", conversation_history=[], task_id="task-1"
    )

    assert result["final_response"] == expected
    assert bool(result.get("failed")) is failed
    assert persisted
    for raw in fragments:
        assert raw.strip() not in str(result)
        assert all(raw.strip() not in str(snapshot) for snapshot in persisted)


def test_required_post_tool_failure_stops_before_next_provider_call(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
        post_tool_expr='(_ for _ in ()).throw(RuntimeError("raw post secret"))',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.valid_tool_names.add("web_search")
    tool_calls = [
        SimpleNamespace(
            id=f"call-{index}",
            type="function",
            function=SimpleNamespace(name="web_search", arguments="{}"),
        )
        for index in (1, 2)
    ]
    responses = iter(
        (
            _response("", finish_reason="tool_calls", tool_calls=tool_calls),
            _response("provider must not be called again"),
        )
    )
    calls = []

    def provider(kwargs):
        calls.append(kwargs)
        return next(responses)

    agent._interruptible_api_call = provider
    with patch(
        "model_tools.handle_function_call", return_value="search result"
    ) as execute:
        result = agent.run_conversation(
            "search",
            conversation_history=[],
            task_id="task-1",
        )

    assert len(calls) == 1
    assert execute.call_count == 1
    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assert "raw post secret" not in str(result)


def test_required_pre_tool_failure_blocks_sequential_dispatch(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
        pre_tool_expr='(_ for _ in ()).throw(RuntimeError("raw pre secret"))',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.valid_tool_names.add("web_search")
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    agent._interruptible_api_call = lambda _kwargs: _response(
        "", finish_reason="tool_calls", tool_calls=[tool_call]
    )

    with patch("model_tools.handle_function_call") as execute:
        result = agent.run_conversation(
            "search", conversation_history=[], task_id="task-1"
        )

    assert execute.call_count == 0
    assert result["failed"] is True
    assert result["final_response"] == REQUIRED_LIFECYCLE_FAILURE_TEXT
    assert "raw pre secret" not in str(result)


def test_required_pre_tool_failure_blocks_direct_invoke_tool(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", None)',
        pre_tool_expr='(_ for _ in ()).throw(RuntimeError("raw direct secret"))',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent._current_turn_id = "turn-direct"
    agent._current_api_request_id = "api-direct"
    agent.valid_tool_names.add("web_search")

    with patch("model_tools.handle_function_call") as execute:
        result = agent._invoke_tool(
            "web_search", {"query": "x"}, "task-1", tool_call_id="call-1"
        )

    assert execute.call_count == 0
    assert REQUIRED_LIFECYCLE_FAILURE_TEXT in result
    assert "raw direct secret" not in result


def test_required_output_hides_tool_narration_before_persist_and_interim_egress(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", "safe answer")',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    agent.valid_tool_names.add("web_search")
    raw_text = "raw narration before tool"
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        extra_content={"thought_signature": "private tool signature"},
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    first_response = _response(
        raw_text,
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    first_response.choices[0].message.reasoning_content = "private continuity"
    first_response.choices[0].message.reasoning_details = [
        {"type": "reasoning", "text": "private chain of thought"}
    ]
    responses = iter((first_response, _response("raw terminal answer")))
    provider_calls = []

    def provider(kwargs):
        provider_calls.append(copy.deepcopy(kwargs))
        return next(responses)

    agent._interruptible_api_call = provider
    persisted = []
    interim = []
    progress = []
    displayed = []
    post_api = []
    manager = plugins_mod._delivery_manager()
    manager._hooks.setdefault("post_api_request", []).append(
        lambda **payload: post_api.append(copy.deepcopy(payload))
    )
    agent._flush_messages_to_session_db = lambda messages, _history=None: (
        persisted.append(copy.deepcopy(messages)) or True
    )
    agent.interim_assistant_callback = lambda message: interim.append(
        copy.deepcopy(message)
    )
    agent.tool_progress_callback = lambda *parts: progress.append(parts)
    agent._vprint = lambda message, **_kwargs: displayed.append(message)

    with patch("model_tools.handle_function_call", return_value="search result"):
        result = agent.run_conversation(
            "search",
            conversation_history=[],
            task_id="task-1",
        )

    observable = str((result, persisted, interim, progress, displayed, post_api))
    assert result["final_response"] == "safe answer"
    assert raw_text not in observable
    assert "raw terminal answer" not in observable
    assert "private continuity" not in observable
    assert "private chain of thought" not in observable
    assert "private tool signature" not in observable
    assert interim == []
    assert post_api == []
    assert any(
        message.get("reasoning_details")
        == [{"type": "reasoning", "text": "private chain of thought"}]
        for message in provider_calls[1]["messages"]
        if isinstance(message, dict)
    ), provider_calls[1]["messages"]


def test_required_output_authorizes_content_filter_direct_return(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / "hermes"
    _install_required_plugin(
        home,
        'required_hook_result("behavioral.pre_llm.v1", None)',
        'required_hook_result("behavioral.output.v1", "safe refusal")',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    agent = _agent()
    raw = "private provider refusal"
    agent._interruptible_api_call = lambda _kwargs: _response(
        raw, finish_reason="content_filter"
    )

    result = agent.run_conversation(
        "do work", conversation_history=[], task_id="task-1"
    )

    assert result["final_response"] == "safe refusal"
    assert result["pre_transform_response"] is None
    assert raw not in str(result)
    assert raw not in caplog.text


def test_exceptional_turn_exit_scrubs_ephemeral_provider_continuity(monkeypatch):
    agent = _agent()

    def fail_after_quarantine(*_args, **_kwargs):
        message = {
            "role": "assistant",
            "timestamp": 1.0,
            "finish_reason": "tool_calls",
            "reasoning_content": "private continuity",
            "tool_calls": [{"id": "call-1", "call_id": "call-1"}],
        }
        quarantine_required_provider_fields(agent, message)
        raise RuntimeError("provider failed")

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fail_after_quarantine)

    with pytest.raises(RuntimeError, match="provider failed"):
        agent.run_conversation("hello", conversation_history=[])

    assert agent._required_provider_continuity == {}


def test_outer_turn_scope_restores_required_stream_state_after_direct_return():
    """A direct inner-loop return must not poison the cached agent's next turn."""
    agent = _agent()
    original_delta_callback = object()
    agent.stream_delta_callback = None
    agent._disable_streaming = True
    agent._required_lifecycle_stream_delta_callback = original_delta_callback
    agent._required_lifecycle_disable_streaming = False
    agent._current_turn_id = "turn-direct-return"

    with patch(
        "agent.conversation_loop.run_conversation",
        return_value={
            "final_response": "bounded failure",
            "messages": [],
            "api_calls": 0,
            "completed": False,
            "failed": True,
        },
    ):
        agent.run_conversation(
            "do work",
            conversation_history=[],
            task_id="task-direct-return",
        )

    assert agent.stream_delta_callback is original_delta_callback
    assert agent._disable_streaming is False
    assert not hasattr(agent, "_required_lifecycle_stream_delta_callback")
    assert not hasattr(agent, "_required_lifecycle_disable_streaming")
