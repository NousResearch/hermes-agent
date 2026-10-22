"""Disposable wire-level probes of the July adapter on October Hermes.

Only the model endpoint is synthetic. The adapter, AIAgent, dispatch, approval
gate and HTTP/SSE exchange are real.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import shlex
import threading

from fastapi.testclient import TestClient
import pytest

from agui_adapter.server import create_app
from agui_adapter.session import AgentConfig
from hermes_cli.config import save_config
from tests.waiting import wait_until


class ModelPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if getattr(self.server, "capture_status", None) and self.path in (
            "/v1/messages",
            "/v1/responses",
        ):
            self.server.requests.append(request)
            self.server.request_paths.append(self.path)
            self.send_response(self.server.capture_status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error = getattr(
                self.server,
                "capture_error",
                b'{"error":{"type":"authentication_error","message":"capture complete"}}',
            )
            self.wfile.write(error)
            return
        if "messages" not in request:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.server.requests.append(request)
        if hasattr(self.server, "request_headers"):
            self.server.request_headers.append(dict(self.headers))
        messages = request.get("messages", [])
        returned = next((m for m in reversed(messages) if m["role"] == "tool"), None)
        if hasattr(self.server, "script"):
            message = self.server.script(request, len(self.server.requests) - 1)
            finish = "tool_calls" if message.get("tool_calls") else "stop"
        elif returned is not None:
            message = {
                "role": "assistant",
                "content": "PROBE_DONE " + returned["content"],
            }
            finish = "stop"
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "probe-call-1",
                        "type": "function",
                        "function": {
                            "name": self.server.tool_name,
                            "arguments": json.dumps(self.server.tool_args),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        frame = {
            "id": "probe-response",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": getattr(
                self.server,
                "usage",
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
        }
        if request.get("stream"):
            if "tool_calls" in message:
                for index, call in enumerate(message["tool_calls"]):
                    call["index"] = index
            frame["object"] = "chat.completion.chunk"
            frame["choices"][0]["delta"] = frame["choices"][0].pop("message")
            payload = ("data: " + json.dumps(frame) + "\n\ndata: [DONE]\n\n").encode()
            kind = "text/event-stream"
        else:
            payload, kind = json.dumps(frame).encode(), "application/json"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def peer():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelPeer)
    server.requests = []
    server.tool_name = "agui_probe_frontend_action"
    server.tool_args = {"color": "blue"}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def config_for(peer, toolsets):
    config = AgentConfig()
    config.base_url = f"http://127.0.0.1:{peer.server_port}/v1"
    config.api_key = "loopback-probe"
    config.provider = "custom"
    config.model = "gpt-4o"
    config.api_mode = "chat_completions"
    config.enabled_toolsets = toolsets
    return config


def events(response):
    assert response.status_code == 200, response.text
    result = [
        json.loads(line[5:].strip())
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    assert not [e for e in result if e["type"] == "RUN_ERROR"], result
    return result


def body(thread, tools=None):
    return {
        "threadId": thread,
        "runId": "leg-1",
        "state": {},
        "messages": [
            {"id": "user-1", "role": "user", "content": "Run the probe action"}
        ],
        "tools": tools or [],
        "context": [],
        "forwardedProps": {},
    }


@pytest.mark.parametrize("change_shared_state", [False, True])
def test_real_frontend_handoff_and_resume_preserve_wire_history(
    peer, change_shared_state, monkeypatch
):
    from agui_adapter import server

    agents = []
    real_build = server.build_run_agent

    def observe(*args, **kwargs):
        agent = real_build(*args, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(server, "build_run_agent", observe)
    save_config({"auxiliary": {"title_generation": {"enabled": False}}})
    request = body(
        "frontend-probe",
        [
            {
                "name": peer.tool_name,
                "description": "Set the browser color",
                "parameters": {
                    "type": "object",
                    "properties": {"color": {"type": "string"}},
                },
            }
        ],
    )
    request["state"] = {"color": "red"}
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        first = events(client.post("/", json=request))
        call = next(e for e in first if e["type"] == "TOOL_CALL_START")
        assert call["toolCallId"] == "probe-call-1"
        assert first[-1]["type"] == "RUN_FINISHED"
        assert not any(e["type"] == "TOOL_CALL_RESULT" for e in first)
        assert len(peer.requests) == 1, "model must wait for the client result"
        request["runId"] = "leg-2"
        if change_shared_state:
            request["state"] = {"color": "blue"}
        request["messages"] += [
            {
                "id": "assistant-1",
                "role": "assistant",
                "content": "",
                "toolCalls": [
                    {
                        "id": call["toolCallId"],
                        "type": "function",
                        "function": {
                            "name": peer.tool_name,
                            "arguments": json.dumps(peer.tool_args),
                        },
                    }
                ],
            },
            {
                "id": "result-1",
                "role": "tool",
                "toolCallId": call["toolCallId"],
                "content": "BROWSER_APPLIED_BLUE",
            },
        ]
        second = events(client.post("/", json=request))
        assert second[-1]["type"] == "RUN_FINISHED"
        assert "BROWSER_APPLIED_BLUE" in "".join(e.get("delta", "") for e in second)
    assert len(peer.requests) == 2
    assert len(agents) == 1, "frontend continuation rebuilt the agent"
    assert agents[0]._user_turn_count == 1, (
        "frontend continuation started another user turn"
    )
    first_wire, resumed_wire = (r["messages"] for r in peer.requests)
    assert resumed_wire[-1]["role"] == "tool", resumed_wire
    assert resumed_wire[-1]["tool_call_id"] == "probe-call-1"
    assert resumed_wire[: len(first_wire)] == first_wire, (
        "historical model prefix changed on resume"
    )
    assert peer.requests[0]["tools"] == peer.requests[1]["tools"]
    if change_shared_state:
        assert '"blue"' in resumed_wire[-1]["content"]


@pytest.mark.parametrize("approved", [True, False])
def test_real_agent_approval_park_and_resume(peer, tmp_path, approved, request):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.terminal_tool import _active_environments, _qualify_task_key

    owner = tmp_path / "owner"
    owner.mkdir()
    token = set_hermes_home_override(owner)
    request.addfinalizer(lambda: reset_hermes_home_override(token))
    target = tmp_path / "disposable-removal"
    target.mkdir()
    (target / "owned.txt").write_text("disposable test data")
    peer.tool_name = "terminal"
    peer.tool_args = {"command": "rm -r -- " + shlex.quote(str(target))}
    save_config(
        {
            "approvals": {"mode": "manual", "timeout": 10},
            "auxiliary": {"title_generation": {"enabled": False}},
            "terminal": {"env": "local"},
        }
    )
    run_input = body("approval-probe")
    with TestClient(
        create_app(config_for(peer, ["terminal"])), base_url="http://127.0.0.1"
    ) as client:
        first = events(client.post("/", json=run_input))
        record = client.app.state.executions.conversations.get(run_input["threadId"])
        terminal_key = record.profile_context.copy().run(
            _qualify_task_key, f"session:{record.approval_key}"
        )
        assert terminal_key != f"session:{record.approval_key}"
        outcome = first[-1].get("outcome", {})
        assert outcome["type"] == "interrupt", first
        started = [e["toolCallId"] for e in first if e["type"] == "TOOL_CALL_START"]
        ended = [e["toolCallId"] for e in first if e["type"] == "TOOL_CALL_END"]
        assert started and ended == started, "interrupt left tool arguments open"
        assert not any(e["type"] == "TOOL_CALL_RESULT" for e in first)
        assert target.exists(), "effect occurred before approval"
        assert len(peer.requests) == 1
        run_input["runId"] = "leg-2"
        run_input["resume"] = [
            {
                "interruptId": outcome["interrupts"][0]["id"],
                "status": "resolved",
                "payload": {"approved": approved, "scope": "once"},
            }
        ]
        second = events(client.post("/", json=run_input))
        assert second[-1]["outcome"]["type"] == "success", second
        assert not any(e["type"] == "TOOL_CALL_END" for e in second)
        assert [
            e["toolCallId"] for e in second if e["type"] == "TOOL_CALL_RESULT"
        ] == started
        assert terminal_key in _active_environments
    assert terminal_key not in _active_environments
    assert target.exists() is (not approved)
    assert len(peer.requests) == 2


@pytest.mark.parametrize("kind", ["frontend", "writer-list", "writer-map"])
def test_duplicate_declarations_rejected_before_conversation(peer, kind):
    peer.script = lambda *_: {"role": "assistant", "content": "unexpected model call"}
    declarations = [
        {
            "name": "agui_duplicate_" + kind.replace("-", "_"),
            "description": "Update shared state",
            "stateKey": key,
            "arg": key,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": value_type}},
            },
        }
        for key, value_type in [("notes", "array"), ("recipe", "object")]
    ]
    request = body("duplicate-declaration")
    if kind == "frontend":
        request["tools"] = declarations
    else:
        writers = (
            dict(zip(("notes", "recipe"), declarations))
            if kind == "writer-map"
            else declarations
        )
        request["forwardedProps"] = {"stateWriterTools": writers}
    with TestClient(
        create_app(config_for(peer, [])), base_url="http://127.0.0.1"
    ) as client:
        response = client.post("/", json=request)
    assert response.status_code == 200, response.text
    frames = [
        json.loads(line[5:])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    assert [frame["type"] for frame in frames] == ["RUN_STARTED", "RUN_ERROR"]
    assert frames[-1]["message"]
    assert not peer.requests


@pytest.mark.parametrize(
    "writers",
    [
        "unsupported",
        1,
        [None],
        {"x": None},
        [{}],
        *[{"x": {"name": name}} for name in ([], {}, "", " ", 1)],
        *[[{"name": name}] for name in (["x"], {"x": 1})],
        *[
            {"x": {key: value}}
            for key in ("stateKey", "state_key")
            for value in (None, [], {}, 1)
        ],
        {"x": {"stateKey": "ok", "state_key": []}},
        *[
            {"x": {key: value}}
            for key in ("arg", "description")
            for value in (False, [], {})
        ],
        {"x": {"parameters": False}},
    ],
)
def test_malformed_writers_rejected_before_conversation(peer, writers):
    peer.script = lambda *_: {"role": "assistant", "content": "unexpected model call"}
    request = body("invalid-writer")
    request["forwardedProps"] = {"stateWriterTools": writers}
    app = create_app(config_for(peer, []))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/", json=request)
        assert response.status_code == 200, response.text
        frames = [
            json.loads(line[5:])
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert [frame["type"] for frame in frames] == ["RUN_STARTED", "RUN_ERROR"]
        assert frames[-1]["message"]
        assert not peer.requests
        assert not app.state.executions.active
        assert not app.state.executions.conversations.records


@pytest.mark.parametrize("status", [401, 400])
@pytest.mark.parametrize("mode", ["anthropic_messages", "codex_responses"])
def test_initial_frontend_data_reaches_native_user_tail_with_hermes_prompt(
    peer, mode, status, monkeypatch, tmp_path
):
    from tests.agui_adapter.test_parked_frontend import error_events

    home = tmp_path / "native-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "SOUL.md").write_text("Identity: HERMES_NATIVE_SENTINEL.\n")
    save_config({"auxiliary": {"title_generation": {"enabled": False}}})
    config = config_for(peer, [])
    config.api_mode = mode
    config.model = "claude-sonnet-4-6" if mode == "anthropic_messages" else "gpt-4.1"
    peer.capture_status, peer.request_paths = status, []
    if status == 400:
        peer.capture_error = (
            b'{"error":{"type":"content_filter","message":"content_filter"}}'
        )
    request = body("native-" + mode)
    request["context"] = [{"description": "name", "value": "CONTEXT_SENTINEL"}]
    request["forwardedProps"] = {"tone": "PROPS_SENTINEL"}
    request["state"] = {"recipe": "STATE_SENTINEL"}
    with TestClient(create_app(config), base_url="http://127.0.0.1") as client:
        response = client.post("/", json=request)
        assert error_events(response)[-1]["message"] == "The agent run failed."
        wait_until(
            lambda: not client.app.state.executions.conversations.records,
            timeout=10,
            what="failed conversation retirement",
        )
    assert peer.request_paths == [
        "/v1/messages" if mode == "anthropic_messages" else "/v1/responses"
    ]
    (payload,) = peer.requests
    prompt = payload["system" if mode == "anthropic_messages" else "instructions"]
    assert "HERMES_NATIVE_SENTINEL" in str(prompt)
    (user,) = payload["messages" if mode == "anthropic_messages" else "input"]
    assert user["role"] == "user"
    assert request["messages"][0]["content"] in str(user["content"])
    for sentinel in ("CONTEXT_SENTINEL", "PROPS_SENTINEL", "STATE_SENTINEL"):
        assert sentinel in str(user["content"])
        assert sentinel not in str(prompt)
