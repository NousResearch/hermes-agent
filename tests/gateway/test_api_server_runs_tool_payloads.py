"""/v1/runs tool events carry redacted full args and result next to the short preview."""
import asyncio
import json

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import api_server as _api_server
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_runs import _RunStream, _execute_run
from gateway.session_api_turn import admit_api_turn
from tests.gateway.test_api_canonical_projection import _events, _launch, _turn_runner

SECRET = "abcdefghijklmnopqrstuvwxyz0123"


async def _emit(adapter, event_type, *args, **kwargs):
    adapter._run_streams["run"] = _RunStream()
    adapter._make_run_event_callback("run", asyncio.get_running_loop())(event_type, *args, **kwargs)
    await asyncio.sleep(0)
    return adapter._run_streams["run"].backlog[-1][1]


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


@pytest.mark.asyncio
async def test_tool_started_carries_redacted_full_args(adapter):
    command = "cd /repo && " + "echo long; " * 40 + f"OPENAI_API_KEY=sk-{SECRET} run"
    event = await _emit(adapter, "tool.started", "terminal", "cd /repo && echo…", {"command": command, "timeout": 60})

    assert event["preview"] == "cd /repo && echo…"
    assert event["args"]["timeout"] == 60
    assert event["args"]["command"].startswith("cd /repo && echo long;")
    assert len(event["args"]["command"]) > len(event["preview"])
    assert SECRET not in event["args"]["command"]


@pytest.mark.asyncio
async def test_tool_completed_carries_redacted_full_result_beside_preview(adapter):
    output = "line\n" * 300 + f"token sk-{SECRET} end"
    event = await _emit(adapter, "tool.completed", "terminal", None, None, is_error=False,
                        result={"exit_code": 0, "output": output})

    assert len(event["preview"]) <= 500
    assert len(event["result"]) > 500
    assert json.loads(event["result"])["output"].endswith(" end")
    assert SECRET not in event["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    {"output": f"API_KEY=sk-{SECRET}", "return_code": 0},
    json.dumps({"output": f"API_KEY=sk-{SECRET}", "return_code": 0}),
])
async def test_tool_completed_result_stays_valid_json_when_secret_ends_a_string(adapter, result):
    event = await _emit(adapter, "tool.completed", "terminal", None, None, is_error=False, result=result)
    parsed = json.loads(event["result"])

    assert parsed["return_code"] == 0
    assert parsed["output"].startswith("API_KEY=") and SECRET not in parsed["output"]


@pytest.mark.asyncio
async def test_admitted_run_tool_completed_carries_result(api, owner):
    admitted = admit_api_turn(api, session_id="payloads", user_message="hello", conversation_history=[])
    launch, queue = _launch(api, admitted, run_id="run_admitted")

    async def handle(event):
        from gateway.session_results import execution_result
        turn = _turn_runner(owner, admitted[1])
        turn.combined_tool_start_callback("call-1", "read_file", {"path": "/tmp/x"})
        turn.combined_tool_complete_callback("call-1", "read_file", {"path": "/tmp/x"}, {"content": "body"})
        execution_result.get()["result"] = {"final_response": "ok"}
        return "ok"
    owner.runner._handle_message = handle
    await _execute_run(api, launch, _api_server=_api_server)
    await asyncio.sleep(0)
    completed = next(e for e in _events(queue) if e["event"] == "tool.completed")

    assert json.loads(completed["result"]) == {"content": "body"}
    assert "body" in completed["preview"]
