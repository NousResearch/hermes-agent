import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.openmaus_screenshot_compat as screenshot_compat
from agent.openmaus_screenshot_compat import (
    ENABLE_ENV,
    MAX_BLOCK_BYTES,
    MODEL_ENV,
    SCREENSHOT_TOOL,
    is_openmaus_screenshot_compat_enabled,
    maybe_normalize_openmaus_screenshot_call,
)
from agent.transports.types import NormalizedResponse, ToolCall
from run_agent import AIAgent


MODEL = "local/screenshot-compat-model"
EXACT = (
    "<computer>\n"
    "call:default/ScreenshotTool()#description=Take one screenshot; do not click or type.\n"
    "</computer>"
)


def cua_call(*, user: str = "local-user", bot: str = "local-bot") -> str:
    """Build a structurally valid fixture without retaining live identities."""

    return (
        '<tool_code name="cuacall" code="#include '
        f'<C:/Users/{user}/.openmausbot/bots/{bot}/'
        'libraries/cuapm-cpp/dist/bin/xcua.h></code>\n'
        "    xc:0(1)\n"
        '    { "op":"screen_snapshot", "args":{ } }</tool_code>'
        '<tool_result op="exec">Succeeded</tool_result> WINDOWS_VM_SCREENSHOT_OK'
    )


EXACT_CUACALL = cua_call()


def bind_cuacall_digest(monkeypatch, content: str = EXACT_CUACALL) -> None:
    monkeypatch.setattr(
        screenshot_compat,
        "_OBSERVED_CUACALL_SHA256",
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def agent(model=MODEL, tools=(SCREENSHOT_TOOL,)):
    return SimpleNamespace(model=model, valid_tool_names=set(tools))


def response(content=EXACT, *, tool_calls=None, finish_reason="stop"):
    return NormalizedResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


def raw_response(content, *, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=MODEL, usage=None)


def screenshot_tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": SCREENSHOT_TOOL,
                "description": "Read the current desktop state.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.fixture(autouse=True)
def compat_env(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV, "1")
    monkeypatch.setenv(MODEL_ENV, MODEL)


def test_observed_cuacall_digest_is_pinned_without_retaining_identity():
    assert screenshot_compat._OBSERVED_CUACALL_SHA256 == (
        "9c50c64b19f16f0b0779685bb56967aac0d9cfc5e86e90dd3ec5f8f8a1057eaa"
    )


def assert_registered_screenshot_call(content):
    item = response(content)

    assert maybe_normalize_openmaus_screenshot_call(
        agent(), item, "stop", was_streaming=False
    )
    assert item.content is None
    assert item.finish_reason == "tool_calls"
    assert len(item.tool_calls) == 1
    assert item.tool_calls[0].id is None
    assert item.tool_calls[0].name == SCREENSHOT_TOOL
    assert item.tool_calls[0].arguments == "{}"


def test_exact_bound_nonstreaming_computer_form_becomes_registered_screenshot_call():
    assert_registered_screenshot_call(EXACT)


def test_structural_cuacall_with_matching_observed_digest_becomes_screenshot(monkeypatch):
    bind_cuacall_digest(monkeypatch)
    assert_registered_screenshot_call(EXACT_CUACALL)


def test_same_structure_with_wrong_path_is_rejected_by_full_message_digest(monkeypatch):
    bind_cuacall_digest(monkeypatch)
    wrong_path = cua_call(user="different-user")
    item = response(wrong_path)
    assert not maybe_normalize_openmaus_screenshot_call(
        agent(), item, "stop", was_streaming=False
    )


@pytest.mark.parametrize(
    "content",
    [
        EXACT_CUACALL.replace("C:/Users/", "C:/Profiles/"),
        EXACT_CUACALL.replace("/.openmausbot/", "/.other-app/"),
        EXACT_CUACALL.replace("/xcua.h", "/other.h"),
        cua_call(user=".."),
        cua_call(bot="."),
    ],
)
def test_noncanonical_header_path_rejects_even_with_matching_digest(monkeypatch, content):
    bind_cuacall_digest(monkeypatch, content)
    item = response(content)
    assert not maybe_normalize_openmaus_screenshot_call(
        agent(), item, "stop", was_streaming=False
    )


def test_exact_computer_crlf_form_is_accepted():
    item = response(EXACT.replace("\n", "\r\n"))
    assert maybe_normalize_openmaus_screenshot_call(
        agent(), item, "stop", was_streaming=False
    )


def test_provider_prefixed_runtime_model_is_not_the_exact_bound_model():
    assert not is_openmaus_screenshot_compat_enabled(
        agent(model=f"custom:ollama-windows:{MODEL}")
    )


@pytest.mark.parametrize(
    ("runtime_agent", "env_enabled", "env_model"),
    [
        (agent(), "0", MODEL),
        (agent(), "true", MODEL),
        (agent(), "1", ""),
        (agent(model="other/model"), "1", MODEL),
        (agent(tools=()), "1", MODEL),
        (agent(tools=("mcp__computer__click",)), "1", MODEL),
    ],
)
def test_activation_fails_closed(monkeypatch, runtime_agent, env_enabled, env_model):
    monkeypatch.setenv(ENABLE_ENV, env_enabled)
    monkeypatch.setenv(MODEL_ENV, env_model)
    assert not is_openmaus_screenshot_compat_enabled(runtime_agent)


@pytest.mark.parametrize(
    "content",
    [
        f"Here is the call:\n{EXACT}",
        f"{EXACT}\nDone.",
        EXACT.replace("<computer>", "<Computer>"),
        EXACT.replace("ScreenshotTool()", "ScreenshotTool(x=1)"),
        EXACT.replace("ScreenshotTool", "ClickTool"),
        EXACT.replace("#description=", "#Description="),
        f"{EXACT}\n{EXACT}",
        "<computer>\n<computer>\ncall:default/ScreenshotTool()#description=x\n</computer>\n</computer>",
        "<computer>\ncall:default/ScreenshotTool()#description=line one\nline two\n</computer>",
        "<computer>\ncall:default/ScreenshotTool()#description=   \n</computer>",
        "<computer>\ncall:default/ScreenshotTool()#description=x<y\n</computer>",
        "<computer>\ncall:default/ScreenshotTool()\n</computer>",
        "<computer>\ncall:default/ScreenshotTool()#description=x",
        f"Here is the call:\n{EXACT_CUACALL}",
        f"{EXACT_CUACALL}\nDone.",
        EXACT_CUACALL.replace('name="cuacall"', 'name="CuaCall"'),
        EXACT_CUACALL.replace('"args":{ }', '"args":{"button":"left"}'),
        EXACT_CUACALL.replace("screen_snapshot", "mouse_click"),
        EXACT_CUACALL.replace("Succeeded", "Failed"),
        EXACT_CUACALL.replace("WINDOWS_VM_SCREENSHOT_OK", "OTHER_MARKER"),
        EXACT_CUACALL.replace("C:/Users/local-user/", "C:/Users/other-user/"),
        EXACT_CUACALL.replace("    xc:0(1)", "xc:0(1)"),
        EXACT_CUACALL.replace("\n", "\r\n"),
        EXACT_CUACALL.replace("</tool_code><tool_result", "</tool_code> <tool_result"),
        EXACT_CUACALL.replace("</tool_result> ", "</tool_result>\n"),
        EXACT_CUACALL.replace('"args":{ }', '"args":{ }\n'),
        "",
    ],
)
def test_malformed_or_mixed_forms_remain_text(content):
    item = response(content)
    assert not maybe_normalize_openmaus_screenshot_call(
        agent(), item, "stop", was_streaming=False
    )
    assert item.content == content
    assert item.tool_calls is None
    assert item.finish_reason == "stop"


def test_oversized_form_remains_text():
    content = (
        "<computer>\ncall:default/ScreenshotTool()#description="
        + "x" * MAX_BLOCK_BYTES
        + "\n</computer>"
    )
    item = response(content)
    assert len(content.encode("utf-8")) > MAX_BLOCK_BYTES
    assert not maybe_normalize_openmaus_screenshot_call(
        agent(), item, "stop", was_streaming=False
    )


@pytest.mark.parametrize(
    ("was_streaming", "finish_reason"),
    [(True, "stop"), (False, "length"), (False, "content_filter")],
)
def test_incomplete_or_streamed_forms_remain_text(was_streaming, finish_reason):
    item = response(EXACT_CUACALL, finish_reason=finish_reason)
    assert not maybe_normalize_openmaus_screenshot_call(
        agent(), item, finish_reason, was_streaming=was_streaming
    )


def test_existing_structured_tool_call_is_never_rewritten():
    existing = ToolCall(id="call-1", name=SCREENSHOT_TOOL, arguments="{}")
    item = response(EXACT_CUACALL, tool_calls=[existing], finish_reason="tool_calls")
    assert not maybe_normalize_openmaus_screenshot_call(
        agent(), item, "tool_calls", was_streaming=False
    )
    assert item.tool_calls == [existing]


def test_conversation_loop_dispatches_real_tool_and_discards_fabricated_result(
    tmp_path, monkeypatch
):
    bind_cuacall_digest(monkeypatch)
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    with (
        patch("run_agent.get_tool_definitions", return_value=screenshot_tool_definitions()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        runtime_agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model=MODEL,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    runtime_agent.client = MagicMock()
    runtime_agent._cached_system_prompt = "You are helpful."
    runtime_agent._use_prompt_caching = False
    runtime_agent.compression_enabled = False
    runtime_agent.save_trajectories = False
    runtime_agent.client.chat.completions.create.side_effect = [
        raw_response(EXACT_CUACALL),
        raw_response("WINDOWS_VM_SCREENSHOT_OK"),
    ]

    with (
        patch(
            "run_agent.handle_function_call",
            return_value='{"ok":true,"image":"[screenshot omitted]"}',
        ) as dispatch,
        patch.object(runtime_agent, "_persist_session"),
        patch.object(runtime_agent, "_save_trajectory"),
        patch.object(runtime_agent, "_cleanup_task_resources"),
    ):
        result = runtime_agent.run_conversation("Take one screenshot")

    assert result["final_response"] == "WINDOWS_VM_SCREENSHOT_OK"
    assert result["api_calls"] == 2
    assert dispatch.call_count == 1
    assert dispatch.call_args.args[:2] == (SCREENSHOT_TOOL, {})
    assert runtime_agent.client.chat.completions.create.call_args_list[0].kwargs.get("stream") is not True

    second_messages = runtime_agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assistant_call = next(message for message in second_messages if message.get("tool_calls"))
    tool_result = next(message for message in second_messages if message.get("role") == "tool")
    call_id = assistant_call["tool_calls"][0]["id"]
    assert call_id
    assert tool_result["tool_call_id"] == call_id
    assert all(
        '<tool_result op="exec">Succeeded</tool_result>' not in str(message)
        for message in second_messages
    )
