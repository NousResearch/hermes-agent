"""GatewayAgent drives main's HermesCLI callbacks from a scripted gateway (no live gateway)."""
import asyncio
import threading
from contextlib import asynccontextmanager

import pytest

from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from hermes_cli.gateway_connection import DISCONNECTED, GatewayConnection

SID = "local-abc"


def frame(kind, payload, *, generation=7, admission="adm-1"):
    params = {"type": kind, "session_id": SID, "execution_generation": generation, "payload": payload}
    if admission:
        params["admission_id"] = admission
    return {"jsonrpc": "2.0", "method": "event", "params": params}


class FakeClient:
    """Answers RPCs; ``prompt.submit`` plays ``script`` (frames, or callables run on the loop)."""

    def __init__(self, script, result=None, usage=None):
        self.events = asyncio.Queue()
        self.calls = []
        self.script = script
        self.result = result if result is not None else {}
        self.usage = usage or {}
        self.on_interrupt = None
        self.mutate_replies = []

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "session.mutate" and self.mutate_replies:
            return self.mutate_replies.pop(0)
        if method == "prompt.submit":
            asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(self._play()))
            return {"admission_id": "adm-1", "status": "queued", "execution_generation": 7}
        if method == "prompt.receipt":
            return {"admission_id": "adm-1", "status": "terminal", "result": self.result, "usage": self.usage}
        if method == "session.resume":
            return {"session_id": params["session_id"], "revision": 1, "execution_generation": 7, "messages": []}
        if method == "session.interrupt":
            if self.on_interrupt:
                await self.events.put(self.on_interrupt)
            return {"execution_state": "running"}
        return {"status": "ok"}

    async def _play(self):
        for item in self.script:
            if item == "disconnect":
                await self.events.put(GatewayClientError("Gateway disconnected"))
                return
            if item == "wait":
                await asyncio.sleep(0.3)  # health: allow HX013 -- a slow stream frame gap IS the scripted behaviour
                continue
            await self.events.put(item)


def connect_with(client):
    @asynccontextmanager
    async def connect():
        yield client
    return connect


@pytest.fixture
def make_agent():
    made = []

    def build(script, **kw):
        result = kw.pop("result", None)
        client = FakeClient(script, result=result, usage=kw.pop("usage", None))
        conn = GatewayConnection(connect=connect_with(client))
        conn.start(5)
        conn.state(SID)
        made.append(conn)
        agent = GatewayAgent(connection=conn, session_id=SID, **kw)
        return agent, client
    yield build
    for conn in made:
        conn.close()


def test_streamed_reply_reaches_stream_delta_callback(make_agent):
    deltas = []
    agent, client = make_agent(
        [frame("message.delta", {"text": "Hel"}), frame("message.delta", {"text": "lo"}),
         frame("message.delta", {"text": None}),
         frame("message.complete", {"text": "Hello", "outcome": "completed", "status": "complete"})],
        result={"final_response": "Hello", "api_calls": 1, "completed": True,
                "messages": [{"role": "assistant", "content": "Hello"}], "_messages_are_turn_suffix": True},
        stream_delta_callback=deltas.append)
    result = agent.run_conversation("hi", conversation_history=[{"role": "user", "content": "old"}])
    assert deltas == ["Hel", "lo", None]
    assert result["final_response"] == "Hello"
    assert [m["content"] for m in result["messages"]] == ["old", "hi", "Hello"]
    submit = dict(client.calls)["prompt.submit"]
    assert submit["text"] == "hi" and submit["session_id"] == SID


def test_tool_events_drive_progress_and_complete_callbacks(make_agent):
    progress, completes = [], []
    agent, _ = make_agent(
        [frame("tool.generating", {"name": "read_file"}),
         frame("tool.start", {"tool_id": "c1", "name": "read_file", "args": {"path": "notes.txt"}, "context": ""}),
         frame("tool.complete", {"tool_id": "c1", "name": "read_file", "args": {"path": "notes.txt"},
                                 "result": "{}", "is_error": False, "duration_s": 0.25, "inline_diff": None}),
         frame("message.complete", {"text": "done", "outcome": "completed"})],
        tool_progress_callback=lambda *a, **k: progress.append((a, k)),
        tool_complete_callback=lambda *a: completes.append(a),
        tool_gen_callback=lambda name: progress.append((("gen", name), {})))
    agent.run_conversation("read it")
    assert progress[0] == (("gen", "read_file"), {})
    started = progress[1][0]
    assert started[0] == "tool.started" and started[1] == "read_file" and started[3] == {"path": "notes.txt"}
    done_args, done_kw = progress[2]
    assert done_args[:2] == ("tool.completed", "read_file")
    assert done_kw["duration"] == 0.25 and done_kw["is_error"] is False and done_kw["result"] == "{}"
    assert completes == [("c1", "read_file", {"path": "notes.txt"}, "{}")]


def test_approval_request_calls_cli_approval_callback_then_responds(make_agent):
    from tools.terminal_tool import set_approval_callback
    seen = []

    def approve(command, description, **kw):
        seen.append((command, description, kw))
        return "once"
    agent, client = make_agent(
        [frame("approval.request", {"kind": "approval", "prompt_id": "p1", "execution_generation": 7,
                                    "command": "rm -rf x", "description": "dangerous",
                                    "choices": ["once", "deny", "session", "always"]}),
         frame("message.complete", {"text": "ok", "outcome": "completed"})])
    set_approval_callback(approve)
    try:
        agent.run_conversation("do it")
    finally:
        set_approval_callback(None)
    assert seen and seen[0][0] == "rm -rf x" and seen[0][2]["allow_permanent"] is True
    respond = dict(client.calls)["approval.respond"]
    assert respond == {"session_id": SID, "execution_generation": 7, "prompt_id": "p1", "choice": "once"}


def test_interrupt_sends_session_interrupt_and_settles_interrupted(make_agent):
    agent, client = make_agent(
        [frame("message.delta", {"text": "w001 "})] + ["wait"] * 20,
        stream_delta_callback=lambda text: None)
    client.on_interrupt = frame("message.complete", {"text": "", "outcome": "cancelled", "status": "interrupted"})
    out = {}
    worker = threading.Thread(target=lambda: out.update(agent.run_conversation("story")))
    worker.start()
    from tests.waiting import wait_until
    wait_until(lambda: agent._generation is not None and agent._admission, timeout=5)
    agent.interrupt("stop")
    worker.join(5)
    assert not worker.is_alive()
    interrupt = dict(client.calls)["session.interrupt"]
    assert interrupt == {"session_id": SID, "execution_generation": 7}
    assert out["interrupted"] is True and out["completed"] is False


def test_result_dict_has_every_key_hermes_cli_reads(make_agent):
    agent, _ = make_agent([frame("message.complete", {"text": "x", "outcome": "completed"})],
                          result={"final_response": "x", "api_calls": 2, "input_tokens": 10, "output_tokens": 3},
                          usage={"last_prompt_tokens": 42, "context_length": 1000})
    result = agent.run_conversation("q")
    for key in ("final_response", "messages", "failed", "partial", "error", "failure_reason", "completed",
                "interrupted", "interrupt_message", "api_calls", "pending_steer", "last_reasoning",
                "response_previewed", "response_transformed", "pre_transform_response", "billing_block",
                "session_id", "auxiliary_usage"):
        assert key in result, key
    assert agent.context_compressor.last_prompt_tokens == 42
    assert agent.context_compressor.context_length == 1000
    assert agent.session_api_calls == 2


def test_disconnect_returns_failed_result(make_agent):
    agent, _ = make_agent([frame("message.delta", {"text": "par"}), "disconnect"],
                          stream_delta_callback=lambda text: None)
    result = agent.run_conversation("q")
    assert result["failed"] is True and result["completed"] is False
    assert "disconnected" in result["error"].lower()


def test_other_admissions_frames_are_ignored(make_agent):
    deltas = []
    agent, _ = make_agent([frame("message.delta", {"text": "theirs"}, admission="adm-9"),
                           frame("message.delta", {"text": "ours"}),
                           frame("message.complete", {"text": "ours", "outcome": "completed"})],
                          stream_delta_callback=deltas.append)
    agent.run_conversation("q")
    assert deltas == ["ours"]


def test_agent_print_routes_to_print_fn(make_agent):
    printed = []
    agent, _ = make_agent([frame("agent.print", {"text": "⚡ Interrupted during API call."}),
                           frame("message.complete", {"text": "", "outcome": "completed"})])
    agent._print_fn = printed.append
    agent.run_conversation("q")
    assert printed == ["⚡ Interrupted during API call."]


def test_hard_interrupt_is_a_real_method():
    import inspect
    assert inspect.getattr_static(GatewayAgent, "hard_interrupt")


def test_manual_compress_is_the_gateway_compress_mutation(make_agent):
    """Main's /compress (compress_now -> agent._compress_context) commits on the gateway and adopts
    the compacted transcript; the CLI prints main's \"Compressed: N -> M\" report from it."""
    from agent.conversation_compression_manual import CompressRequest, compress_now
    agent, client = make_agent([])
    compacted = [{"role": "user", "content": "summary", "row_id": 1}, {"role": "assistant", "content": "ok", "row_id": 2}]
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.resume":
            client.calls.append((method, params))
            done = any(m == "session.mutate" for m, _ in client.calls)
            return {"revision": 9, "execution_generation": 7, "messages": compacted if done else []}
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]
    result = compress_now(agent, history, CompressRequest(), task_id=SID)
    mutate = dict(client.calls)["session.mutate"]
    assert mutate["operation"] == "compress" and mutate["expected_revision"] == 9 and mutate["payload"] == {}
    assert mutate["expected_generation"] == 7
    assert result.status == "compressed"
    assert result.after_messages == [{"role": "user", "content": "summary"}, {"role": "assistant", "content": "ok"}]
    assert result.summary["headline"] == "Compressed: 6 → 2 messages"
    assert agent.context_compressor.compression_count == 1


SESSION_USAGE = {"input_tokens": 500, "output_tokens": 80, "cache_read_tokens": 0, "cache_write_tokens": 0,
                 "prompt_tokens": 500, "completion_tokens": 80, "total_tokens": 580, "reasoning_tokens": 0,
                 "api_calls": 5, "last_prompt_tokens": 229, "context_tokens": 229, "context_length": 128000,
                 "config_context_length": 128000, "context_source": "local_estimate", "compression_count": 0,
                 "avg_latency_s": 0.2, "tokens_per_s": 116.0, "last_turn_tokens": 30, "rate_limit": None,
                 "active_subagents": 0, "model": "fake-model", "provider": "custom", "fallback_model": None}


def test_session_usage_event_feeds_mains_status_bar_attributes(make_agent):
    """The wave_a2 `session.usage` frame (session-cumulative) lands on the attributes main's status
    bar, spinner and /usage read: counters, pin, estimate marker, latency/velocity, rate limits."""
    from agent.context_breakdown import context_display_source
    from agent.context_pin import is_context_pinned
    agent, _ = make_agent([frame("session.usage", {"usage": SESSION_USAGE}, admission=None),
                           frame("message.complete", {"text": "x", "outcome": "completed"})])
    agent.run_conversation("q")
    assert (agent.session_input_tokens, agent.session_output_tokens, agent.session_total_tokens) == (500, 80, 580)
    assert agent.session_prompt_tokens == 500 and agent.session_completion_tokens == 80
    assert agent.session_api_calls == 5
    comp = agent.context_compressor
    assert comp.last_prompt_tokens == 229 and comp.context_length == 128000
    assert is_context_pinned(comp.context_length, comp._config_context_length)
    assert is_context_pinned(comp.context_length, agent._config_context_length)
    assert context_display_source(comp) == "local_estimate"
    n = len(agent._api_latency_history)
    assert n == 1 and round(sum(agent._api_output_history) / sum(agent._api_latency_history)) == 116
    assert agent.get_rate_limit_state() is None


def test_gateway_session_usage_frame_feeds_the_status_bar_mid_turn(make_agent):
    """The frame the gateway really publishes (`usage_frame`: flat counters plus the nested TUI
    `usage` view) lands the flat counters, not the nested view, on main's attributes mid-turn."""
    from gateway.session_usage import usage_frame
    usage = dict(SESSION_USAGE, compression_count=2)
    seen = []
    agent, _ = make_agent([frame("session.usage", usage_frame(usage), admission=None),
                           frame("message.complete", {"text": "x", "outcome": "completed"})])
    agent.context_compressor.context_length = 0
    # Runs after the agent's own listener: what the status bar shows right after the mid-turn frame.
    agent._state().usage_listeners.append(lambda cached: seen.append((
        dict(cached), agent.session_api_calls, agent.context_compressor.context_length,
        agent.context_compressor.compression_count, list(agent._api_latency_history))))
    agent.run_conversation("q")
    cached, *bar = seen[0]
    assert bar == [5, 128000, 2, [0.2]]
    assert cached["api_calls"] == 5 and "context_max" not in cached


def test_turn_end_reads_session_stats_cumulative(make_agent):
    """/usage after a turn shows main's session-cumulative totals (session.stats), not the turn's."""
    agent, client = make_agent([frame("message.complete", {"text": "x", "outcome": "completed"})],
                               result={"final_response": "x", "api_calls": 1, "input_tokens": 100, "output_tokens": 10})
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.stats":
            client.calls.append((method, params))
            return dict(SESSION_USAGE, rate_limit={"requests_min": {"limit": 10, "remaining": 9}, "captured_at": 5.0})
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    agent.run_conversation("q")
    assert agent.session_input_tokens == 500 and agent.session_api_calls == 5
    assert agent.get_rate_limit_state().has_data and agent.get_rate_limit_state().requests_min.limit == 10


def test_status_bar_context_is_the_gateways_anchored_figure_with_mains_estimate_marker(make_agent):
    """Main's bar re-derives context_tokens from _turn_base_usage_anchor over _session_messages and
    shows `~` when rows follow the turn's first response; an interrupted turn (no response) is `~`."""
    from agent.usage_anchor import anchored_context_tokens
    usage = dict(SESSION_USAGE, context_tokens=700, api_calls=2)
    suffix = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
               "function": {"name": "read_file", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "x" * 400},
              {"role": "assistant", "content": "done"}]
    agent, client = make_agent([frame("message.complete", {"text": "done", "outcome": "completed"})],
                               result={"final_response": "done", "api_calls": 2, "messages": suffix})
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.stats":
            return usage
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    agent.run_conversation("read it")
    msgs = agent._session_messages
    assert anchored_context_tokens(msgs, agent._turn_base_usage_anchor, charge_stale_thinking=False) == 700
    assert len(msgs) > agent._turn_base_usage_anchor["base_count"] + 1  # rows after the first reply: `~`
    agent2, client2 = make_agent([frame("message.complete", {"text": "", "outcome": "cancelled", "status": "interrupted"})])
    original2 = client2.rpc

    async def rpc2(method, *, _timeout=None, **params):
        if method == "session.stats":
            return usage
        return await original2(method, _timeout=_timeout, **params)
    client2.rpc = rpc2
    agent2.session_api_calls = 2
    agent2.run_conversation("story", conversation_history=[{"role": "user", "content": "a"},
                                                           {"role": "assistant", "content": "b"}])
    assert agent2._turn_base_usage_anchor["base_count"] == 2  # the interrupted ask is estimated


def test_first_tool_of_a_batch_reads_session_stats_for_mains_spinner_token_flow(make_agent):
    """Main's `↓ N tok` beside a running tool/approval reads session_output_tokens right after the API
    call that asked for the tool; the stand-in refreshes session.stats once per tool batch."""
    seen = []
    agent, client = make_agent(
        [frame("tool.start", {"tool_id": "c1", "name": "terminal", "args": {"command": "ls"}}),
         frame("tool.start", {"tool_id": "c2", "name": "terminal", "args": {"command": "pwd"}}),
         frame("tool.complete", {"tool_id": "c1", "name": "terminal", "args": {}, "result": "{}"}),
         frame("tool.complete", {"tool_id": "c2", "name": "terminal", "args": {}, "result": "{}"}),
         frame("message.complete", {"text": "x", "outcome": "completed"})],
        tool_progress_callback=lambda kind, *a, **k: seen.append((kind, agent.session_output_tokens)))
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.stats":
            client.calls.append((method, params))
            return dict(SESSION_USAGE, output_tokens=10)
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    agent.run_conversation("q")
    assert seen[0] == ("tool.started", 10)
    assert [m for m, _ in client.calls].count("session.stats") == 2  # first tool.start + turn end


def test_moa_reference_maps_index_and_count_to_mains_kwargs(make_agent):
    seen = []
    agent, _ = make_agent([frame("moa.reference", {"label": "gpt-x", "text": "ref", "index": 1, "count": 2}),
                           frame("message.complete", {"text": "x", "outcome": "completed"})],
                          tool_progress_callback=lambda *a, **k: seen.append((a, k)))
    agent.run_conversation("q")
    assert seen[0][0][:3] == ("moa.reference", "gpt-x", "ref")
    assert seen[0][1] == {"moa_index": 1, "moa_count": 2}


def test_agent_follows_mains_session_id_switch(make_agent):
    """/new, /resume and /branch point ``agent.session_id`` at another session: the agent attaches to it
    (``session.resume``) and reads that session's state."""
    agent, client = make_agent([])
    agent.connection.state("local-other").info = {"model": "other-model"}
    agent.session_id = "local-other"
    assert agent.session_id == "local-other"
    assert ("session.resume", {"session_id": "local-other"}) in client.calls
    agent.connection.state("local-other").apply_usage({"api_calls": 3, "input_tokens": 9})
    assert agent.session_api_calls == 3


def test_turn_after_a_rewind_anchors_on_its_first_response(make_agent):
    """/undo and /retry bump the session's runtime generation, so the gateway rebuilds its agent and the
    cumulative counters restart below this viewer's turn base. The turn still completed an API call:
    the bar anchors on its reply (no `~`), as main's does after /retry."""
    usage = dict(SESSION_USAGE, context_tokens=120, api_calls=1)
    agent, client = make_agent([frame("message.complete", {"text": "done", "outcome": "completed"})],
                               result={"final_response": "done", "api_calls": 1})
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.stats":
            return usage
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    agent.session_api_calls = 4  # the previous agent's cumulative count
    agent.run_conversation("again", conversation_history=[{"role": "user", "content": "a"},
                                                          {"role": "assistant", "content": "b"}])
    msgs = agent._session_messages
    assert agent._turn_base_usage_anchor["base_count"] == len(msgs) - 1  # through the ask; reply follows


def test_verbose_mode_reaches_the_gateway_session(make_agent):
    """``/verbose`` / ``/focus`` write ``agent.tool_progress_mode``: the gateway renders tool events in
    that mode (``config.set verbose`` with the explicit mode, never ``cycle``)."""
    agent, client = make_agent([])
    agent.tool_progress_mode = "verbose"
    agent.tool_progress_mode = "verbose"
    agent.tool_progress_mode = "off"
    # Own key only: the stand-in also sends show_reasoning at build (the reasoning-sink test above).
    sets = [p for m, p in client.calls if m == "config.set" and p.get("key") == "verbose"]
    assert sets == [{"session_id": SID, "key": "verbose", "value": "verbose"},
                    {"session_id": SID, "key": "verbose", "value": "off"}]


def test_a_non_default_verbose_mode_is_sent_when_the_agent_is_built(make_agent):
    _agent, client = make_agent([], tool_progress_mode="new")
    assert {"session_id": SID, "key": "verbose", "value": "new"} in [p for m, p in client.calls if m == "config.set"]


def test_reasoning_level_change_is_the_session_reasoning_mutation(make_agent):
    """``/reasoning high`` sets ``reasoning_config`` and rebuilds the agent: the rebuilt agent commits
    the effort as ``session.mutate reasoning`` (session scope, as main's), once."""
    agent, client = make_agent([], reasoning_config={"enabled": True, "effort": "high"}, push_reasoning=True)
    mut = [p for m, p in client.calls if m == "session.mutate"]
    assert len(mut) == 1 and mut[0]["operation"] == "reasoning" and mut[0]["payload"] == {"effort": "high"}
    # A policy mutation without the generation it was read at is refused by the owner (invalid_params).
    assert mut[0]["expected_generation"] == 7
    agent.reasoning_config = {"enabled": False}
    mut = [p for m, p in client.calls if m == "session.mutate"]
    assert mut[-1]["payload"] == {"effort": "none"}
    agent.connection.state(SID).info["reasoning_effort"] = "none"
    agent2 = GatewayAgent(connection=agent.connection, session_id=SID, reasoning_config={"enabled": False})
    assert len([p for m, p in client.calls if m == "session.mutate"]) == 2  # already the session's effort
    assert agent2.reasoning_config == {"enabled": False}


def test_an_explicit_level_the_session_already_inherits_is_still_committed(make_agent):
    """``/reasoning high`` (the rebuilt agent) or a model pick's ``--reasoning high`` while the session runs
    high by inheritance: still the owner's reasoning mutation, so a later profile change can't move it. An
    implicit re-derivation (/new, a one-turn restore) of the same level commits nothing."""
    from types import SimpleNamespace
    from hermes_cli.cli_model_switch_mixin import _apply_reasoning_after_switch
    agent, client = make_agent([])
    agent.connection.state(SID).info["reasoning_effort"] = "high"
    GatewayAgent(connection=agent.connection, session_id=SID, reasoning_config={"enabled": True, "effort": "high"},
                 push_reasoning=True)
    _apply_reasoning_after_switch(SimpleNamespace(agent=agent), "high", persist_global=False)
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    assert [p["payload"] for m, p in client.calls if m == "session.mutate"] == [{"effort": "high"}] * 2


def test_a_config_resolved_effort_never_overrides_the_sessions_launch_policy(make_agent):
    """Building the agent with the CLI's config-resolved effort (no /reasoning pick) commits nothing:
    a resumed `-q --resume` must keep the session's frozen --reasoning, not the profile default."""
    agent, client = make_agent([], reasoning_config={"enabled": True, "effort": "none"})
    assert [p for m, p in client.calls if m == "session.mutate"] == []
    assert agent.reasoning_config == {"enabled": True, "effort": "none"}


def test_reasoning_sink_sets_the_sessions_show_reasoning_flag_on_attach_and_toggle(make_agent):
    """Main: a reasoning sink on the agent (show_reasoning, or verbose) makes the agent request and
    stream reasoning. The gateway session gets the same switch at construction and on every toggle."""
    shown = []
    agent, client = make_agent(
        [frame("reasoning.delta", {"text": "thinking hard"}),
         frame("message.complete", {"text": "ok", "outcome": "completed"})],
        reasoning_callback=shown.append)
    sets = [p for m, p in client.calls if m == "config.set"]
    assert sets == [{"session_id": SID, "key": "show_reasoning", "value": "1"}]
    agent.run_conversation("q")
    assert shown == ["thinking hard"]
    agent.reasoning_callback = None  # /reasoning hide
    agent.reasoning_callback = None  # unchanged: no second RPC
    agent.reasoning_callback = shown.append  # /reasoning show
    values = [p["value"] for m, p in client.calls if m == "config.set" and p.get("key") == "show_reasoning"]
    assert values == ["1", "0", "1"]


def test_gateway_sudo_request_opens_mains_sudo_panel_and_replies_with_the_value(make_agent):
    """The gateway's ``sudo`` server→client request (a JSON-RPC request frame with an ``srq-`` id)
    is answered by main's masked sudo panel (the turn's registered callback) as a response frame."""
    from tools.terminal_tool import set_sudo_password_callback
    request = {"jsonrpc": "2.0", "id": "srq-abc", "method": "sudo",
               "params": {"session_id": SID, "command": "sudo true"}}
    agent, _ = make_agent([request, frame("message.complete", {"text": "ok", "outcome": "completed"})])
    replies = []
    agent.connection.reply = lambda rid, result: replies.append((rid, result)) or True
    asked = []
    set_sudo_password_callback(lambda: asked.append(1) or "pw-from-panel")
    try:
        agent.run_conversation("run sudo")
    finally:
        set_sudo_password_callback(None)
    assert asked == [1]
    assert replies == [("srq-abc", {"value": "pw-from-panel"})]


def test_connection_reply_sends_a_jsonrpc_response_frame():
    sent = []

    class WS:
        async def send(self, raw):
            sent.append(raw)
    client = FakeClient([])
    client.websocket = WS()
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    try:
        assert conn.reply("srq-1", {"value": ""}) is True
    finally:
        conn.close()
    import json
    assert [json.loads(raw) for raw in sent] == [{"jsonrpc": "2.0", "id": "srq-1", "result": {"value": ""}}]


def test_gateway_secret_request_opens_mains_secret_panel_and_the_gateway_saves(make_agent, monkeypatch):
    """``secret`` (skill env var): main's panel collects the value and prints its lines, but the CLI
    process never writes ``.env`` — the value goes back to the gateway, which saves it."""
    from hermes_cli import callbacks
    from tools import skills_tool
    request = {"jsonrpc": "2.0", "id": "srq-sec", "method": "secret",
               "params": {"session_id": SID, "env_var": "HAB_KEY", "prompt": "Enter it",
                          "metadata": {"skill_name": "hab"}}}
    agent, _ = make_agent([request, frame("message.complete", {"text": "ok", "outcome": "completed"})])
    replies = []
    agent.connection.reply = lambda rid, result: replies.append((rid, result)) or True
    local_writes = []
    monkeypatch.setattr(callbacks, "save_env_value_secure", lambda k, v: local_writes.append((k, v)))
    monkeypatch.setattr(callbacks, "cprint", lambda *a, **k: None)
    seen = []

    class Cli:  # no prompt_toolkit app: main's prompt_for_secret reads the masked prompt
        pass
    monkeypatch.setattr(callbacks, "masked_secret_prompt", lambda prompt: seen.append(prompt) or "sekrit")
    monkeypatch.setattr(skills_tool, "_secret_capture_callback",
                        lambda var, prompt, metadata=None: callbacks.prompt_for_secret(Cli(), var, prompt, metadata))
    agent.run_conversation("view the skill")
    assert seen and seen[0].startswith("Enter it")
    assert replies == [("srq-sec", {"value": "sekrit"})]
    assert local_writes == []  # the gateway saves; this process never writes .env


def test_gateway_clarify_batch_request_opens_mains_batch_panel(make_agent):
    """All questions arrive in one ``clarify`` request; main's batch panel answers the set."""
    request = {"jsonrpc": "2.0", "id": "srq-clr", "method": "clarify", "params": {"session_id": SID, "questions": [
        {"qid": "q0", "question": "colour?", "choices": ["red", "blue"]},
        {"qid": "q1", "question": "size?", "choices": ["s", "l"], "multi_select": True}]}}
    shown = []

    def panel(questions):
        shown.append(questions)
        return {"answers": {"q0": "red", "q1": ["s", "l"]}, "outcome": "submitted"}
    agent, client = make_agent([request, frame("message.complete", {"text": "ok", "outcome": "completed"})],
                          clarify_callback=panel)
    replies = []
    agent.connection.reply = lambda rid, result: replies.append((rid, result)) or True
    agent.run_conversation("ask")
    assert [q["qid"] for q in shown[0]] == ["q0", "q1"] and shown[0][1]["multi_select"] is True
    assert replies == [("srq-clr", {"answers": {"q0": "red", "q1": '["s", "l"]'}, "outcome": "submitted"})]
    # The turn is marked as answered by main's batch panel: the gateway sends this viewer the batch.
    assert dict(client.calls)["prompt.submit"]["clarify_batch"] is True


def test_a_mid_turn_steer_row_is_estimated_delta_like_main(make_agent):
    """Main anchors the bar on the turn's first response priced over the history through the turn's
    own ask; a /steer delivered mid-turn is a later user-role row, so it is delta and the bar shows `~`."""
    usage = dict(SESSION_USAGE, context_tokens=900, api_calls=2)
    suffix = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
               "function": {"name": "terminal", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "ok"},
              {"role": "user", "content": "[USER STEER] also this", "display_kind": "steer"},
              {"role": "assistant", "content": "done"}]
    agent, client = make_agent([frame("message.complete", {"text": "done", "outcome": "completed"})],
                               result={"final_response": "done", "api_calls": 2, "messages": suffix})
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.stats":
            return usage
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    agent.run_conversation("slow", conversation_history=[{"role": "user", "content": "a"},
                                                          {"role": "assistant", "content": "b"}])
    assert agent._turn_base_usage_anchor["base_count"] == 3  # history + the ask, not the steer row


def test_first_turn_on_a_rebuilt_gateway_agent_keeps_the_exact_figure(make_agent):
    """/personality or /model rebuilds the gateway's agent, whose session counters restart: the turn
    result's own api_calls says the turn answered, so its reply is anchored (no `~`), as on main."""
    usage = dict(SESSION_USAGE, context_tokens=120, api_calls=1)  # restarted counters
    agent, client = make_agent([frame("message.complete", {"text": "arr", "outcome": "completed"})],
                               result={"final_response": "arr", "api_calls": 1,
                                       "messages": [{"role": "assistant", "content": "arr"}]})
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.stats":
            return usage
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    agent.session_api_calls = 9  # what the retired agent's counters had reached
    agent.run_conversation("who are you", conversation_history=[{"role": "user", "content": "a"},
                                                                 {"role": "assistant", "content": "b"}])
    assert agent._turn_base_usage_anchor["base_count"] == 3  # anchored after the ask: no estimated delta


def test_native_image_parts_are_staged_and_submitted_as_attachments(make_agent):
    """Main hands run_conversation OpenAI content parts for a vision model (images read locally);
    prompt.submit takes text + attachments, so each image is staged with image.attach_bytes first."""
    from agent.image_routing import build_native_content_parts
    import base64
    import tempfile
    from pathlib import Path
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNg+M8AAAICAQB7CYF4AAAAAElFTkSuQmCC")
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        path = Path(tmp) / "pixel.png"
        path.write_bytes(png)
        parts, skipped = build_native_content_parts("what is this?", [str(path)])
    assert not skipped
    agent, client = make_agent([frame("message.complete", {"text": "a pixel", "outcome": "completed"})])
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "image.attach_bytes":
            client.calls.append((method, params))
            return {"attached": True, "path": "/owner/cache/images/upload_1.png", "mime": "image/png"}
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    agent.run_conversation(parts, persist_user_message=None)
    staged = [p for m, p in client.calls if m == "image.attach_bytes"]
    assert len(staged) == 1 and staged[0]["content_base64"].startswith("data:image/png;base64,")
    submit = dict(client.calls)["prompt.submit"]
    assert submit["text"] == "what is this?"
    assert submit["attachments"] == [{"path": "/owner/cache/images/upload_1.png", "mime": "image/png"}]


def test_after_a_redirect_the_redirect_row_is_the_turns_ask_like_main(make_agent):
    """busy_input_mode=interrupt redirects the streaming call: main's first completed response answers
    the redirect row, so the reply is anchored (no `~`), unlike a /steer row."""
    usage = dict(SESSION_USAGE, context_tokens=300, api_calls=1)
    suffix = [{"role": "assistant", "content": "STEP8 story: w000 w001"},
              {"role": "user", "content": "[Context from the interrupted assistant response] redirect now"},
              {"role": "assistant", "content": "redirected"}]
    agent, client = make_agent([frame("message.delta", {"text": "w000 "}), "wait", "wait",
                                frame("message.complete", {"text": "redirected", "outcome": "completed"})],
                               result={"final_response": "redirected", "api_calls": 1, "messages": suffix},
                               stream_delta_callback=lambda text: None)
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.stats":
            return usage
        if method == "session.redirect":
            client.calls.append((method, params))
            return {"status": "redirected"}
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    out = {}
    worker = threading.Thread(target=lambda: out.update(agent.run_conversation("story")))
    worker.start()
    from tests.waiting import wait_until
    wait_until(lambda: agent._generation is not None, timeout=5)
    assert agent.redirect("redirect now") is True
    worker.join(5)
    assert agent._turn_base_usage_anchor["base_count"] == 3  # through the redirect row


def test_voice_turn_flag_rides_prompt_submit(make_agent):
    """Main sets agent._voice_turn_pending for a spoken turn (auxiliary.voice_chat route); the gateway
    gets it as prompt.submit voice_turn=true, and a typed turn carries no flag."""
    agent, client = make_agent([frame("message.complete", {"text": "ok", "outcome": "completed"})])
    agent._voice_turn_pending = True
    agent.run_conversation("[Voice input — respond concisely] hi", persist_user_message="hi")
    submit = dict(client.calls)["prompt.submit"]
    assert submit["voice_turn"] is True and submit["text"] == "hi"
    assert submit["api_text"].startswith("[Voice input")
    client.calls.clear()
    client.script = [frame("message.complete", {"text": "ok", "outcome": "completed"})]
    agent._voice_turn_pending = False
    agent.run_conversation("typed")
    assert "voice_turn" not in dict(client.calls)["prompt.submit"]


def test_fast_tier_reaches_the_session_on_each_rebuild_and_back_to_normal(make_agent):
    """/fast: main rebuilds AIAgent(service_tier=tier). Each rebuilt stand-in sends the tier as the
    session's override when it changed; /fast normal clears it; an unset tier never overrides."""
    agent, client = make_agent([])  # first build, no tier: the session keeps its own default
    assert not [p for m, p in client.calls if m == "config.set" and p.get("key") == "service_tier"]
    fast = GatewayAgent(connection=agent.connection, session_id=SID, service_tier="priority")
    again = GatewayAgent(connection=agent.connection, session_id=SID, service_tier="priority")
    normal = GatewayAgent(connection=agent.connection, session_id=SID, service_tier=None)
    assert fast and again and normal
    sets = [p["value"] for m, p in client.calls if m == "config.set" and p.get("key") == "service_tier"]
    assert sets == ["priority", "normal"]


class RefusingClient(FakeClient):
    """``prompt.submit`` raises ``error`` instead of admitting the turn."""

    def __init__(self, error):
        super().__init__([])
        self.error = error

    async def rpc(self, method, *, _timeout=None, **params):
        if method == "prompt.submit":
            self.calls.append((method, params))
            raise self.error
        return await super().rpc(method, _timeout=_timeout, **params)


def submit_against(error):
    conn = GatewayConnection(connect=connect_with(RefusingClient(error)))
    conn.start(5)
    try:
        return GatewayAgent(connection=conn, session_id=SID).run_conversation("q")
    finally:
        conn.close()


def test_a_session_busy_refusal_at_submit_is_the_session_busy_verdict():
    """The owner answered: the session is busy. That is the busy verdict (send again), not a
    lost gateway connection."""
    result = submit_against(GatewayRPCError("session_busy"))
    assert result["failed"] is True and result["completed"] is False
    assert result["failure_reason"] == "session_busy"
    assert result["failure_retryable"] is True


@pytest.mark.parametrize("reason", ["permission_denied", "runtime_draining", "invalid_params"])
def test_other_owner_refusals_at_submit_are_not_a_lost_connection(reason):
    """A received refusal keeps main's unclassified failed result: no gateway_connection label and
    no retryability verdict."""
    result = submit_against(GatewayRPCError(reason))
    assert result["failed"] is True and result["error"] == reason
    assert result["failure_reason"] is None
    assert "failure_retryable" not in result


def test_a_transport_loss_at_submit_is_still_gateway_connection():
    result = submit_against(GatewayClientError(DISCONNECTED))
    assert result["failed"] is True and result["error"] == DISCONNECTED
    assert result["failure_reason"] == "gateway_connection" and result["failure_retryable"] is False


def test_model_switch_is_the_session_model_mutation_and_a_rebuilt_agent_shows_it(make_agent):
    """A session-scoped ``/model`` calls ``agent.switch_model`` (#134889 M1). The gateway commits
    the target as a generation-fenced ``session.mutate model`` (no key or endpoint leaves the CLI),
    and the agent main rebuilds for the next turn reads the committed model from the session.
    ``/model --once`` is not this path: that pick is one admission's ``model_override``."""
    agent, client = make_agent([], model="old-model", provider="openrouter")
    agent.switch_model(new_model="new-model", new_provider="anthropic", api_key="sk-local",
                       base_url="https://example.invalid", api_mode="anthropic_messages", capabilities=None)
    mutations = [p for m, p in client.calls if m == "session.mutate"]
    assert len(mutations) == 1
    assert mutations[0]["operation"] == "model"
    assert mutations[0]["payload"] == {"model": "new-model", "provider": "anthropic"}
    assert (mutations[0]["expected_revision"], mutations[0]["expected_generation"]) == (1, 7)
    assert (agent.model, agent.provider) == ("new-model", "anthropic")
    rebuilt = GatewayAgent(connection=agent.connection, session_id=SID, model="old-model", provider="openrouter")
    assert (rebuilt.model, rebuilt.provider) == ("new-model", "anthropic")


def test_a_guarded_model_target_commits_only_on_the_users_yes_to_the_owners_warning(make_agent):
    """The owner's guard refusal (cost, data policy, large context) is the user's to answer: its
    warning goes to the CLI's prompt and only a yes re-sends with its token. A no commits nothing;
    with no prompt (-q) the token is never sent. A second refusal after a yes (the session moved)
    raises, so main rolls its fields back and the agent keeps the old model."""
    refusal = {"status": "confirmation_required", "confirm": "tok-1", "confirm_message": "Expensive model"}
    asked = []
    agent, client = make_agent([], model="old-model",
                               model_confirm_callback=lambda r: asked.append(r["confirm_message"]) or True)
    client.mutate_replies = [refusal, {"status": "committed", "model": "pricey-model"}]
    assert agent.switch_model(new_model="pricey-model", new_provider="openrouter") is True
    assert asked == ["Expensive model"]
    sent = [p["payload"] for m, p in client.calls if m == "session.mutate"]
    assert sent == [{"model": "pricey-model", "provider": "openrouter"},
                    {"model": "pricey-model", "provider": "openrouter", "confirm": "tok-1"}]
    assert agent.model == "pricey-model"
    declined, declined_client = make_agent([], model="old-model", model_confirm_callback=lambda r: False)
    unattended, unattended_client = make_agent([], model="old-model")
    declined_client.mutate_replies, unattended_client.mutate_replies = [refusal], [refusal]
    assert declined.switch_model(new_model="pricey-model", new_provider="openrouter") is False
    with pytest.raises(GatewayClientError, match="interactive session"):
        unattended.switch_model(new_model="pricey-model", new_provider="openrouter")
    for refused, refused_client in ((declined, declined_client), (unattended, unattended_client)):
        assert [p["payload"] for m, p in refused_client.calls if m == "session.mutate"] == [
            {"model": "pricey-model", "provider": "openrouter"}]
        assert refused.model == "old-model"
    other, other_client = make_agent([], model="old-model", model_confirm_callback=lambda r: True)
    other_client.mutate_replies = [refusal, dict(refusal, confirm="tok-2")]
    with pytest.raises(GatewayClientError, match="Model not switched"):
        other.switch_model(new_model="pricey-model", new_provider="openrouter")
    assert other.model == "old-model"


def test_moa_one_shot_preset_rides_this_turns_submit_as_its_model_override(make_agent):
    """``/moa <prompt>`` (#134889 CLI-18): main rebuilds its agent on the MoA virtual provider for
    one turn, but the session's info names the session's own model. The preset must reach the turn
    as its admission's ``model_override``; a regular rebuild sends none."""
    done = [frame("message.complete", {"text": "ok", "outcome": "completed"})]
    agent, client = make_agent(done)
    agent.connection.state(SID).info.update(model="claude-sonnet", provider="anthropic")
    moa = GatewayAgent(connection=agent.connection, session_id=SID, model="default", provider="moa")
    moa.run_conversation("compare these designs")
    submit = dict(client.calls)["prompt.submit"]
    assert submit["model_override"] == {"model": "default", "provider": "moa"}
    client.calls.clear()
    regular = GatewayAgent(connection=agent.connection, session_id=SID, model="claude-sonnet", provider="anthropic")
    regular.run_conversation("next")
    assert "model_override" not in dict(client.calls)["prompt.submit"]


def test_moa_one_shot_overrides_a_moa_session_running_another_preset(make_agent):
    """A session already on the MoA provider runs ITS preset; ``/moa`` with another preset must still
    send that preset as the turn's override. Only the same preset is the session's own route."""
    done = [frame("message.complete", {"text": "ok", "outcome": "completed"})]
    agent, client = make_agent(done)
    agent.connection.state(SID).info.update(model="fast", provider="moa")
    GatewayAgent(connection=agent.connection, session_id=SID, model="deep", provider="moa").run_conversation("go")
    assert dict(client.calls)["prompt.submit"]["model_override"] == {"model": "deep", "provider": "moa"}
    client.calls.clear()
    GatewayAgent(connection=agent.connection, session_id=SID, model="fast", provider="moa").run_conversation("again")
    assert "model_override" not in dict(client.calls)["prompt.submit"]


def test_a_committed_model_switch_stands_when_the_refresh_after_it_fails(make_agent):
    """Once ``session.mutate`` commits, the gateway runs the next turn on the new model. A failed
    ``session.resume`` refresh after that must not raise: main's caller would roll its fields back to
    a model the session no longer has. The agent shows the model and provider the gateway committed."""
    agent, client = make_agent([], model="old-model", provider="openrouter")
    client.mutate_replies = [{"status": "committed", "model": "claude-sonnet-4", "provider": "anthropic"}]
    rpc = client.rpc

    async def refresh_fails(method, **params):
        if method == "session.resume" and any(m == "session.mutate" for m, _ in client.calls):
            raise GatewayClientError(DISCONNECTED)
        return await rpc(method, **params)
    client.rpc = refresh_fails
    agent.switch_model(new_model="sonnet")
    assert len([m for m, _ in client.calls if m == "session.mutate"]) == 1
    assert (agent.model, agent.provider) == ("claude-sonnet-4", "anthropic")
    rebuilt = GatewayAgent(connection=agent.connection, session_id=SID, model="old-model", provider="openrouter")
    assert (rebuilt.model, rebuilt.provider) == ("claude-sonnet-4", "anthropic")


def test_a_background_review_summary_during_a_turn_is_printed_like_mains(make_agent):
    """``/refine`` (#134889 M3) and the post-turn review run in the gateway, which publishes their
    summary as a ``review.summary`` frame (no admission). One landing while a turn streams is printed
    as main's ``display.review.summary_cli`` line, never dropped."""
    printed = []
    agent, _ = make_agent([frame("review.summary", {"text": "💾 Self-improvement review: Memory updated"}, admission=None),
                           frame("message.complete", {"text": "ok", "outcome": "completed"})])
    agent._print_fn = printed.append
    agent.run_conversation("hi")
    assert printed == ["  💾 Self-improvement review: Memory updated"]


def test_another_viewer_completing_the_clarify_batch_releases_mains_panel(make_agent, monkeypatch):
    """Another viewer (Desktop/Ink) locks every answer of the batch the classic CLI is showing: the
    gateway settles it and withdraws the request (``request.cancel`` reason ``resolved``). main's real
    batch panel (``CLIModalMixin._clarify_callback``) blocks the agent thread, so the withdrawal must
    close it from the connection thread; the pump then reads the turn's completion instead of sitting
    in the modal until clarify_timeout."""
    import cli
    from tests.hermes_cli.test_cli_clarify_batch import _make_cli_stub
    monkeypatch.setattr(cli, "CLI_CONFIG", {"agent": {"clarify_timeout": 60}})
    panel = _make_cli_stub()
    panel._ring_bell = lambda **kw: None
    request = {"jsonrpc": "2.0", "id": "srq-clr", "method": "clarify", "params": {"session_id": SID, "questions": [
        {"qid": "q0", "question": "colour?", "choices": ["red", "blue"]},
        {"qid": "q1", "question": "size?", "choices": ["s", "l"]}]}}
    withdrawn = frame("request.cancel", {"id": "srq-clr", "method": "clarify", "reason": "resolved"}, admission=None)
    agent, _client = make_agent(
        [request, "wait", withdrawn, frame("message.complete", {"text": "ok", "outcome": "completed"})],
        clarify_callback=panel._clarify_callback, modal_withdraw_callback=panel._withdraw_gateway_modal)
    replies = []
    agent.connection.reply = lambda rid, result: replies.append((rid, result)) or True
    out = {}
    worker = threading.Thread(target=lambda: out.update(agent.run_conversation("ask")), daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "the CLI stayed blocked in its clarify panel"
    assert panel._clarify_state is None
    assert out["completed"] is True or out.get("final_response") is not None
    # Its late reply is a cancel the gateway drops (the request is already settled).
    assert replies == [("srq-clr", {"answers": {}, "outcome": "cancelled"})]


def test_a_withdrawal_for_another_request_leaves_the_panel_open(make_agent):
    agent, _client = make_agent([])
    closed = []
    agent.modal_withdraw_callback = closed.append
    agent._open_request = ("srq-mine", "clarify")
    agent._on_request_withdrawn({"id": "srq-other", "method": "clarify", "reason": "resolved"})
    assert closed == []
    agent._on_request_withdrawn({"id": "srq-mine", "method": "clarify", "reason": "resolved"})
    assert closed == ["clarify"]


def test_ctrl_c_during_the_first_api_call_keeps_mains_estimate_marker(make_agent):
    """T4: Ctrl+C while the turn's first request is in flight. The gateway agent counted the attempt
    (``api_calls: 1`` in the turn result) but no response priced the ask, so main's in-process bar kept
    the previous turn's anchor and estimated the ask (``~``). The stand-in must not anchor after the ask."""
    from agent.usage_anchor import anchored_context_tokens
    usage = dict(SESSION_USAGE, context_tokens=700, api_calls=2)  # no response: cumulative calls unchanged
    agent, client = make_agent([frame("message.complete", {"text": "", "outcome": "cancelled", "status": "interrupted"})],
                               result={"final_response": "", "api_calls": 1, "interrupted": True, "messages": []})
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.stats":
            return usage
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    agent.session_api_calls = 2
    agent.run_conversation("story", conversation_history=[{"role": "user", "content": "a"},
                                                          {"role": "assistant", "content": "b"}])
    anchor = agent._turn_base_usage_anchor
    assert anchor["base_count"] == 2  # the unanswered ask is the estimated delta: main's `~`
    assert anchored_context_tokens(agent._session_messages, anchor, charge_stale_thinking=False) == 700


def test_a_voice_stopped_reply_returns_without_waiting_for_a_turn():
    """TUI-26: the owner answers a typed stop phrase in a LOCAL voice chat (turned on by another
    viewer) with ``{voice_stopped: true}`` and admits nothing, so no ``message.complete`` will ever
    come: the classic CLI returns at once with the stop notice instead of pumping forever."""
    from agent.i18n import t

    class Client(FakeClient):
        async def rpc(self, method, *, _timeout=None, **params):
            if method == "prompt.submit":
                self.calls.append((method, params))
                return {"voice_stopped": True}
            return await super().rpc(method, _timeout=_timeout, **params)

    client = Client([])
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    conn.state(SID)
    try:
        printed = []
        agent = GatewayAgent(connection=conn, session_id=SID)
        agent._print_fn = printed.append
        done = {}
        worker = threading.Thread(target=lambda: done.setdefault("r", agent.run_conversation(
            "stop", conversation_history=[{"role": "user", "content": "hi"}])), daemon=True)
        worker.start()
        worker.join(8)
        assert not worker.is_alive(), "run_conversation hung waiting for a turn that was never admitted"
        result = done["r"]
        assert result["voice_stopped"] is True and not result["failed"] and result["final_response"] == ""
        assert result["messages"] == [{"role": "user", "content": "hi"}], "nothing was admitted"
        assert "prompt.receipt" not in [m for m, _ in client.calls]
        assert any(t("cli.voice.stop_phrase_typed") in line for line in printed), printed
    finally:
        conn.close()


@pytest.mark.parametrize("mode", ["steer", "interrupt"])
def test_a_voice_stopped_correction_is_consumed_not_requeued(make_agent, mode):
    """TUI-26 (Greptile): a bare stop typed while a turn runs reaches ``session.steer`` /
    ``session.redirect``; the owner ends the LOCAL voice chat and answers ``{voice_stopped: true}``.
    The classic CLI must take the text as consumed: re-queuing it would submit ``stop`` again once
    voice is already off, and that becomes an agent turn."""
    import queue

    from cli import HermesCLI

    agent, client = make_agent([])
    agent._generation = 7
    agent._supports_active_turn_redirect = True
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method in {"session.steer", "session.redirect"}:
            client.calls.append((method, params))
            return {"voice_stopped": True}
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    printed = []
    agent._print_fn = printed.append
    shell = object.__new__(HermesCLI)
    shell.busy_input_mode = mode
    shell.agent = agent
    shell._pending_input, shell._interrupt_queue = queue.Queue(), queue.Queue()
    agent._turn_ask_index = 3  # the running turn's ask; a consumed stop must leave it alone
    shell._tui_enter_while_busy("stop", [], "stop")
    assert shell._pending_input.empty() and shell._interrupt_queue.empty(), "consumed stop was re-queued"
    sent = [m for m, _ in client.calls if m.startswith("session.")]
    assert sent == [{"steer": "session.steer", "interrupt": "session.redirect"}[mode]]
    assert agent._turn_ask_index == 3, "the running turn's history changed for a consumed stop"


def test_a_session_switch_drops_the_previous_sessions_figures(make_agent):
    """/new, /resume and /branch repoint ``session_id`` then call ``reset_session_state``: the bar and
    /usage show the target's gateway figures (``session.stats``), never the session left behind, and
    a late frame of the old session does not repaint them."""
    agent, client = make_agent([])
    agent.connection.state(SID).apply_usage(SESSION_USAGE)
    assert agent.session_api_calls == 5 and agent._api_latency_history
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.stats" and params.get("session_id") == "local-other":
            return {"api_calls": 3, "input_tokens": 9}
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    agent.connection.state("local-other").messages = [{"role": "user", "content": "earlier"}]  # resumed
    agent.session_id = "local-other"
    agent.reset_session_state()
    assert (agent.session_api_calls, agent.session_input_tokens, agent.session_output_tokens) == (3, 9, 0)
    assert agent.context_compressor.last_prompt_tokens == 0 and agent._api_latency_history == []
    agent.connection.state(SID).apply_usage(dict(SESSION_USAGE, api_calls=6))
    assert agent.session_api_calls == 3
    assert agent._on_request_withdrawn not in agent.connection.state(SID).request_listeners


def test_new_session_reads_zero_without_a_stats_round_trip(make_agent):
    """/new lands on a session the gateway just created (no transcript, no usage): its figures are
    known zero, so the reset makes no ``session.stats`` call."""
    agent, client = make_agent([])
    agent.connection.state(SID).apply_usage(SESSION_USAGE)
    agent.session_id = "local-fresh"
    client.calls.clear()
    agent.reset_session_state()
    assert agent.session_api_calls == 0 and agent._api_latency_history == []
    assert [method for method, _ in client.calls] == []
