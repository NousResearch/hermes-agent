"""apply_patch: GPT-5+ models write files as grammar-constrained patch text.

The model sees a freeform ``apply_patch`` custom tool in place of ``write_file``
and ``patch``; the rest of Hermes sees the existing ``patch`` tool in V4A mode.
"""

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import apply_patch_tool

PATCH = "*** Begin Patch\n*** Add File: /tmp/a.yaml\n+fruits:\n+  - apple\n*** End Patch\n"


def _function_tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _tool_names(tools):
    return [t.get("name") if t.get("type") == "custom" else t["function"]["name"] for t in tools]


class TestIsEnabled:
    @pytest.mark.parametrize("model", ["openai/gpt-5.6-luna", "openai/gpt-6-luna", "gpt-5", "openai/gpt-10-mini", "openai/gpt-6-luna@preset/omnio"])
    def test_should_enable_gpt5_and_later_on_openrouter_by_default(self, model):
        assert apply_patch_tool.is_enabled("auto", model, "openrouter", "chat_completions")

    @pytest.mark.parametrize(
        "model,provider",
        [
            ("openai/gpt-4.1", "openrouter"),
            ("openai/gpt-oss-120b", "openrouter"),
            ("openai/gpt-5-chat-latest", "openrouter"),
            ("anthropic/claude-sonnet-5", "openrouter"),
            ("@preset/omnio", "openrouter"),
            ("openai/gpt-6-luna", "custom"),
        ],
    )
    def test_should_stay_off_by_default_elsewhere(self, model, provider):
        assert not apply_patch_tool.is_enabled("auto", model, provider, "chat_completions")

    def test_should_add_listed_presets_to_auto_detection(self):
        listed = ["@preset/omnio-brand-setup"]
        assert apply_patch_tool.is_enabled(listed, "@preset/omnio-brand-setup", "openrouter", "chat_completions")
        assert apply_patch_tool.is_enabled(listed, "openai/gpt-5.6-luna@preset/omnio", "openrouter", "chat_completions")
        assert not apply_patch_tool.is_enabled(listed, "@preset/omnio", "openrouter", "chat_completions")
        assert not apply_patch_tool.is_enabled(listed, "deepseek/deepseek-v4", "openrouter", "chat_completions")

    def test_should_follow_an_explicit_switch(self):
        assert apply_patch_tool.is_enabled(True, "anything", "custom", "chat_completions")
        assert not apply_patch_tool.is_enabled(False, "openai/gpt-6-luna", "openrouter", "chat_completions")
        assert not apply_patch_tool.is_enabled("off", "openai/gpt-6-luna", "openrouter", "chat_completions")

    @pytest.mark.parametrize("api_mode", ["codex_responses", "anthropic_messages", "bedrock_converse"])
    def test_should_only_apply_to_chat_completions(self, api_mode):
        assert not apply_patch_tool.is_enabled(True, "openai/gpt-6-luna", "openrouter", api_mode)


class TestRewriteRequest:
    def test_should_offer_apply_patch_in_place_of_the_file_writing_tools(self):
        tools = [_function_tool("read_file"), _function_tool("write_file"), _function_tool("patch"), _function_tool("terminal")]

        rewritten, _ = apply_patch_tool.rewrite_request(tools, [])

        assert _tool_names(rewritten) == ["read_file", "terminal", "apply_patch"]
        custom = rewritten[-1]
        assert custom["type"] == "custom"
        assert custom["format"] == {"type": "grammar", "syntax": "lark", "definition": apply_patch_tool.GRAMMAR}

    def test_should_point_other_tools_at_apply_patch(self):
        terminal = {"type": "function", "function": {"name": "terminal", "description": "Do NOT use sed/awk to edit files — use patch instead. Do NOT use echo/cat heredoc to create files — use write_file instead."}}

        rewritten, _ = apply_patch_tool.rewrite_request([terminal, _function_tool("write_file")], [])

        assert rewritten[0]["function"]["description"] == (
            "Do NOT use sed/awk to edit files — use apply_patch instead. "
            "Do NOT use echo/cat heredoc to create files — use apply_patch instead."
        )
        assert "use write_file instead" in terminal["function"]["description"]

    def test_should_stop_execute_code_from_offering_file_writes(self):
        from tools.code_execution_tool import build_execute_code_schema

        execute_code = {"type": "function", "function": build_execute_code_schema({"read_file", "write_file", "patch", "terminal"}, mode="project")}
        assert "write_file(" in execute_code["function"]["description"]
        tools = [execute_code, _function_tool("read_file"), _function_tool("write_file"), _function_tool("patch"), _function_tool("terminal")]

        rewritten, _ = apply_patch_tool.rewrite_request(tools, [])

        description = rewritten[0]["function"]["description"]
        assert "write_file(" not in description
        assert "patch(" not in description
        assert "read_file(" in description
        assert "write_file(" in execute_code["function"]["description"]

    def test_should_leave_requests_without_file_writing_tools_alone(self):
        tools = [_function_tool("read_file")]
        messages = [{"role": "user", "content": "hi"}]

        assert apply_patch_tool.rewrite_request(tools, messages) == (tools, messages)
        assert apply_patch_tool.rewrite_request(None, messages) == (None, messages)

    def test_should_replay_v4a_patch_calls_as_apply_patch_text(self):
        call = {"id": "call_1", "type": "function", "function": {"name": "patch", "arguments": json.dumps({"mode": "patch", "patch": PATCH})}}
        replace_call = {"id": "call_2", "type": "function", "function": {"name": "patch", "arguments": json.dumps({"mode": "replace", "path": "a", "old_string": "x", "new_string": "y"})}}
        write_call = {"id": "call_3", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "a", "content": "b"})}}
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [call, replace_call, write_call]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        original = copy.deepcopy(messages)

        _, wire = apply_patch_tool.rewrite_request([_function_tool("write_file")], messages)

        calls = wire[1]["tool_calls"]
        assert calls[0]["function"] == {"name": "apply_patch", "arguments": PATCH}
        assert calls[1] == replace_call
        assert calls[2] == write_call
        assert wire[2] == messages[2]
        assert messages == original


class TestResponseTranslation:
    def test_should_detect_a_patch_cut_off_before_its_end_marker(self):
        assert apply_patch_tool.is_complete(PATCH)
        assert apply_patch_tool.is_complete(PATCH.rstrip())
        assert not apply_patch_tool.is_complete(PATCH[: PATCH.index("*** End")])

    def test_should_turn_a_completed_call_into_a_v4a_patch_call(self):
        fn = SimpleNamespace(name="apply_patch", arguments=PATCH)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[SimpleNamespace(function=fn)]))])

        apply_patch_tool.normalize_response(response)

        assert fn.name == "patch"
        assert json.loads(fn.arguments) == {"mode": "patch", "patch": PATCH, "add_overwrites": True}

    def test_should_keep_a_cut_off_call_unparseable_so_it_is_refused(self):
        cut = PATCH[:30]
        fn = SimpleNamespace(name="apply_patch", arguments=cut)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[SimpleNamespace(function=fn)]))])

        apply_patch_tool.normalize_response(response)

        assert fn.name == "patch"
        assert fn.arguments == cut
        with pytest.raises(json.JSONDecodeError):
            json.loads(fn.arguments)

    def test_should_ignore_other_tools_and_shapeless_responses(self):
        fn = SimpleNamespace(name="read_file", arguments='{"path": "a"}')
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[SimpleNamespace(function=fn)]))])

        apply_patch_tool.normalize_response(response)
        apply_patch_tool.normalize_response(SimpleNamespace(choices=[]))

        assert (fn.name, fn.arguments) == ("read_file", '{"path": "a"}')


def _chunk(tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=None, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)], model="openai/gpt-6-luna", usage=None)


def _tool_delta(name=None, arguments=None, tc_id=None):
    return SimpleNamespace(index=0, id=tc_id, function=SimpleNamespace(name=name, arguments=arguments))


def _agent(model="openai/gpt-6-luna"):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model=model,
        provider="openrouter",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


_OFFERED = {"tools": [apply_patch_tool.wire_tool()]}


class TestStreaming:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_should_deliver_a_streamed_patch_as_a_v4a_patch_call(self, mock_close, mock_create):
        half = len(PATCH) // 2
        chunks = [
            _chunk([_tool_delta(name="apply_patch", tc_id="call_1")]),
            _chunk([_tool_delta(arguments=PATCH[:half])]),
            _chunk([_tool_delta(arguments=PATCH[half:])]),
            _chunk(finish_reason="tool_calls"),
        ]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = client
        agent = _agent()
        started = []
        agent.tool_gen_callback = started.append

        response = agent._interruptible_streaming_api_call(dict(_OFFERED))

        call = response.choices[0].message.tool_calls[0]
        assert call.function.name == "patch"
        assert json.loads(call.function.arguments) == {"mode": "patch", "patch": PATCH, "add_overwrites": True}
        assert response.choices[0].finish_reason == "tool_calls"
        assert started == ["patch"]

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_should_refuse_a_patch_cut_off_by_the_output_cap(self, mock_close, mock_create):
        chunks = [
            _chunk([_tool_delta(name="apply_patch", tc_id="call_1")]),
            _chunk([_tool_delta(arguments="*** Begin Patch\n*** Add File: /tmp/a.yaml\n+- name: end\n")]),
            # OpenRouter reported tool_calls for the 65,536-token cut in production.
            _chunk(finish_reason="tool_calls"),
        ]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = client

        response = _agent()._interruptible_streaming_api_call(dict(_OFFERED))

        assert response.choices[0].finish_reason == "length"

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_should_leave_a_tool_named_apply_patch_alone_when_not_offered(self, mock_close, mock_create):
        args = json.dumps({"x": 1})
        chunks = [
            _chunk([_tool_delta(name="apply_patch", tc_id="call_1", arguments=args)]),
            _chunk(finish_reason="tool_calls"),
        ]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = client

        response = _agent()._interruptible_streaming_api_call({})

        call = response.choices[0].message.tool_calls[0]
        assert (call.function.name, call.function.arguments) == ("apply_patch", args)


class TestBuildApiKwargs:
    def test_should_send_apply_patch_to_gpt_models_on_openrouter(self):
        from agent.chat_completion_helpers import build_api_kwargs

        agent = _agent()
        agent.tools = [_function_tool("read_file"), _function_tool("write_file"), _function_tool("patch")]

        kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

        assert _tool_names(kwargs["tools"]) == ["read_file", "apply_patch"]

    def test_should_keep_json_file_tools_for_other_models(self):
        from agent.chat_completion_helpers import build_api_kwargs

        agent = _agent(model="anthropic/claude-sonnet-5")
        agent.tools = [_function_tool("read_file"), _function_tool("write_file"), _function_tool("patch")]

        kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

        assert _tool_names(kwargs["tools"]) == ["read_file", "write_file", "patch"]


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home
    from tools.file_tools import clear_file_ops_cache
    from tools.terminal_tool import _active_environments, _env_lock
    clear_file_ops_cache()
    with _env_lock:
        _active_environments.clear()


class TestPatchToolExecution:
    """apply_patch calls run through the real patch tool with Codex's Add File semantics."""

    def test_should_replace_an_existing_file_with_add_file(self, hermes_home, tmp_path):
        from tools.file_tools import _handle_patch

        target = tmp_path / "draft.yaml"
        target.write_text("brand: old\n")
        text = f"*** Begin Patch\n*** Add File: {target}\n+brand: new\n*** End Patch\n"

        result = json.loads(_handle_patch(json.loads(apply_patch_tool.internal_arguments(text)), task_id="apply_patch_add_1"))

        assert not result.get("error"), result
        assert target.read_text() == "brand: new"

    def test_should_create_a_new_file_with_add_file(self, hermes_home, tmp_path):
        from tools.file_tools import _handle_patch

        target = tmp_path / "new.yaml"
        text = f"*** Begin Patch\n*** Add File: {target}\n+brand: new\n*** End Patch\n"

        result = json.loads(_handle_patch(json.loads(apply_patch_tool.internal_arguments(text)), task_id="apply_patch_add_2"))

        assert not result.get("error"), result
        assert target.read_text() == "brand: new"
