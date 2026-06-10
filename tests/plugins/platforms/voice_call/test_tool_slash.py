"""voice_call tool, /voicecall slash command, and speech stripping."""

import asyncio
import json

import pytest
import pytest_asyncio

from plugins.platforms.voice_call import runtime as runtime_mod
from plugins.platforms.voice_call.responder import strip_for_speech
from plugins.platforms.voice_call.tool import slash_handler, voice_call_handler


async def _tool(action, **args):
    return json.loads(await voice_call_handler({"action": action, **args}))


async def _wait_listening(runtime, call_id, timeout=1.0):
    from plugins.platforms.voice_call.events import CallState

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        record = runtime.manager.get_call(call_id)
        if record is not None and record.state == CallState.LISTENING:
            return record
        await asyncio.sleep(0.01)
    raise AssertionError("never reached listening")


@pytest_asyncio.fixture
async def running_runtime(tmp_path, make_config):
    cfg = make_config()
    cfg.serve.port = 0
    runtime = await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    yield runtime
    await runtime_mod.stop_runtime()


@pytest.mark.asyncio
async def test_tool_errors_without_runtime():
    result = await _tool("get_status")
    assert result["success"] is False
    assert "gateway" in result["error"]


@pytest.mark.asyncio
async def test_tool_requires_action():
    result = json.loads(await voice_call_handler({}))
    assert result["success"] is False


@pytest.mark.asyncio
async def test_tool_initiate_status_speak_end(running_runtime):
    result = await _tool("initiate_call", to_number="+15555550001", message="hi")
    assert result["success"] is True
    call_id = result["call"]["call_id"]
    await _wait_listening(running_runtime, call_id)

    status = await _tool("get_status")
    assert status["success"] and len(status["active_calls"]) == 1
    assert status["provider"] == "mock"

    one = await _tool("get_status", call_id=call_id)
    assert one["call"]["call_id"] == call_id

    spoke = await _tool("speak_to_user", call_id=call_id, message="news for you")
    assert spoke["success"] is True

    ended = await _tool("end_call", call_id=call_id)
    assert ended["success"] is True
    gone = await _tool("get_status", call_id=call_id)
    assert gone["success"] is False


@pytest.mark.asyncio
async def test_tool_error_paths_return_json(running_runtime):
    assert (await _tool("initiate_call"))["success"] is False        # no number
    assert (await _tool("initiate_call", to_number="junk"))["success"] is False
    assert (await _tool("speak_to_user", call_id="nope", message="x"))[
        "success"
    ] is False
    assert (await _tool("end_call", call_id="nope"))["success"] is False
    assert (await _tool("explode"))["success"] is False              # unknown action


@pytest.mark.asyncio
async def test_tool_continue_call_roundtrip_and_timeout(running_runtime):
    from plugins.platforms.voice_call.events import EventType, NormalizedEvent

    result = await _tool("initiate_call", to_number="+15555550001")
    call_id = result["call"]["call_id"]
    record = await _wait_listening(running_runtime, call_id)

    async def reply():
        await asyncio.sleep(0.05)
        await running_runtime.manager.process_event(
            NormalizedEvent(
                type=EventType.CALL_SPEECH, provider="mock",
                provider_call_id=record.provider_call_id, text="sounds good",
            )
        )

    asyncio.get_running_loop().create_task(reply())
    result = await _tool("continue_call", call_id=call_id, message="ok to proceed?")
    assert result["success"] and result["reply"] == "sounds good"

    # No reply this time → JSON timeout error, not an exception.
    result = await _tool("continue_call", call_id=call_id, message="hello?")
    assert result["success"] is False
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_slash_usage_and_status(running_runtime):
    usage = await slash_handler("")
    assert "Usage" in usage
    out = await slash_handler("status")
    assert "no active calls" in out

    out = await slash_handler('call --to +15555550001 --message "hello"')
    assert "dialing +15555550001" in out
    call_id = out.rsplit(" ", 1)[-1]
    await _wait_listening(running_runtime, call_id)

    out = await slash_handler("status")
    assert call_id in out
    out = await slash_handler(f"end --call-id {call_id}")
    assert "ok" in out

    out = await slash_handler("bogus")
    assert "unknown subcommand" in out


# -- strip_for_speech ---------------------------------------------------------


def test_strip_for_speech_removes_markup():
    text = (
        "# Update\n"
        "Here is **bold** and _italic_ and `inline()` plus a "
        "[link](https://example.com) and https://raw.example.com/x.\n"
        "```python\nprint('hi')\n```\n"
        "- item one\n- item two"
    )
    out = strip_for_speech(text)
    assert "```" not in out and "**" not in out and "](" not in out
    assert "https://" not in out
    assert "bold" in out and "italic" in out and "inline()" in out
    assert "code omitted" in out
    assert "item one item two" in out


def test_strip_for_speech_caps_length():
    out = strip_for_speech("word. " * 500, max_chars=100)
    assert len(out) <= 101
    assert out.endswith(".")


def test_strip_for_speech_empty():
    assert strip_for_speech("") == ""
    assert strip_for_speech("```\nonly code\n```") == "code omitted."
