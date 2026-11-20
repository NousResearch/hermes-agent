"""Regression tests for memory/skill review nudge evaluation and counter resets (#13075).

Exercises the live extracted turn paths (``turn_context`` -> ``tool_executor`` ->
``turn_finalizer`` and ``codex_runtime``) via real ``AIAgent.run_conversation`` and
``finalize_turn`` / ``run_codex_app_server_turn`` calls, plus a loopback HTTP end-to-end
test that runs the real background review fork against a local OpenAI-compatible server:

1. Normal threshold trigger fires memory review on the Nth turn and resets after spawn.
2. A mid-turn ``memory`` tool call on the Nth turn prevents duplicate background review
   (while a hook-blocked ``memory`` call does NOT cancel the review).
3. Dual-trigger turns where ``memory`` runs mid-turn suppress only memory review while
   still running the due skill review.
4. Interrupted, empty-response, and ``skip_background_review=True`` turns preserve both
   ``_turns_since_memory`` and ``_iters_since_skill`` progress.
5. When ``_spawn_background_review()`` raises, both counters keep their progress and the
   next normal turn retries cleanly.
6. Single-trigger spawns reset only the counter that triggered; disabled intervals /
   missing memory stores never trigger.
7. Codex app-server runtime stays coherent across threshold triggers, interrupts,
   empty responses, ``skip_background_review=True``, and spawn exceptions.
8. End-to-end loopback HTTP: real ``AIAgent.run_conversation`` + real
   ``_spawn_background_review`` thread writing to ``MEMORY.md`` across threshold,
   mid-turn foreground ``memory``, and spawn-failure recovery turns.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent
from tools.memory_tool import MemoryStore


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    for name in ("memory", "skill_manage", "noop")
]


def _reply(text=None, tool_name=None, tool_args=None):
    calls = None
    if tool_name:
        calls = [
            SimpleNamespace(
                id=f"call_{tool_name}_{id(object())}",
                type="function",
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=json.dumps(tool_args or {}),
                ),
            )
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=calls),
                finish_reason="tool_calls" if calls else "stop",
            )
        ],
        model="test/model",
        usage=None,
    )


def _make_live_agent(
    monkeypatch,
    tmp_path,
    scripts,
    *,
    memory_interval: int = 3,
    skill_interval: int = 10,
    skip_background_review: bool = False,
    with_memory_store: bool = True,
):
    """Build a real AIAgent with a scripted OpenAI client and real MemoryStore."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)

    replies = [reply for turn in scripts for reply in turn]
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kw: replies.pop(0)

    with (
        patch("model_tools.get_tool_definitions", return_value=_TOOLS),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            skip_background_review=skip_background_review,
        )
    agent._cached_system_prompt = "You are helpful."
    if with_memory_store:
        store = MemoryStore(memory_enabled=True, user_profile_enabled=True)
        store.load_from_disk()
        agent._memory_store = store
    else:
        agent._memory_store = None
    agent._memory_nudge_interval = memory_interval
    agent._skill_nudge_interval = skill_interval
    agent._turns_since_memory = 0
    agent._iters_since_skill = 0
    return agent, replies


def _stub_agent_for_finalize(
    *,
    memory_interval: int = 3,
    turns_since_memory: int = 3,
    skill_interval: int = 5,
    iters_since_skill: int = 5,
    skip_background_review: bool = False,
) -> AIAgent:
    agent = AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=skip_background_review,
        platform="cli",
    )
    agent._spawn_background_review = MagicMock()
    agent._save_trajectory = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._persist_session = MagicMock()
    agent._session_messages = []
    agent._file_mutation_verifier_enabled = lambda: False
    agent._turn_completion_explainer_enabled = lambda: False
    agent.clear_interrupt = MagicMock()
    agent._stream_callback = None
    agent._sync_external_memory_for_turn = MagicMock()
    agent._memory_store = object()
    agent._memory_nudge_interval = memory_interval
    agent._turns_since_memory = turns_since_memory
    agent._skill_nudge_interval = skill_interval
    agent._iters_since_skill = iters_since_skill
    agent.valid_tool_names = {"memory", "skill_manage"}
    agent.iteration_budget = MagicMock(remaining=100, used=1, max_total=100)
    agent.max_iterations = 50
    agent._emit_status = MagicMock()
    agent._safe_print = MagicMock()
    agent._apply_persist_user_message_override = MagicMock()
    agent.context_compressor = None
    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False
    agent.model = "test-model"
    agent.session_id = "test-session"
    agent._turn_failed_file_mutations = {}
    agent._db_flush_scan_prefix = None
    return agent


def _call_finalize(agent: AIAgent, *, final_response="ok", interrupted=False, failed=False):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=failed,
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": final_response}],
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hi",
        original_user_message="hi",
        _turn_exit_reason="text_response(1)",
    )


def _run_sequence(agent: AIAgent, count: int) -> tuple[list[tuple[int, int]], list[dict]]:
    spawns: list[dict] = []
    agent._spawn_background_review = lambda **kw: spawns.append(kw)
    history: list = []
    snapshots: list[tuple[int, int]] = []
    for idx in range(count):
        result = agent.run_conversation(f"turn {idx + 1}", conversation_history=history)
        history = result["messages"]
        snapshots.append((len(spawns), agent._turns_since_memory))
    return snapshots, spawns


def test_normal_threshold_trigger_fires_on_nth_turn_and_resets(monkeypatch, tmp_path):
    """Memory review triggers on the Nth turn and resets _turns_since_memory after spawn."""
    scripts = [[_reply("t1")], [_reply("t2")], [_reply("t3")], [_reply("t4")]]
    agent, replies = _make_live_agent(monkeypatch, tmp_path, scripts, memory_interval=3, skill_interval=0)

    snapshots, spawns = _run_sequence(agent, 4)

    assert snapshots == [(0, 1), (0, 2), (1, 0), (1, 1)]
    assert spawns[0]["review_memory"] is True
    assert spawns[0]["review_skills"] is False
    assert replies == []


def test_mid_turn_memory_tool_call_prevents_duplicate_background_review(monkeypatch, tmp_path):
    """When the foreground model calls `memory` during the Nth turn, no duplicate memory review spawns."""
    memory_turn = [
        _reply(
            tool_name="memory",
            tool_args={"action": "add", "target": "memory", "content": "User prefers concise diffs."},
        ),
        _reply("Noted in memory."),
    ]
    scripts = [
        [_reply("t1")],
        [_reply("t2")],
        memory_turn,  # 3rd turn — would hit threshold 3 if memory weren't called
        [_reply("t4")],
        [_reply("t5")],
        [_reply("t6")],  # 3 turns after the manual memory write -> review fires here
    ]
    agent, replies = _make_live_agent(monkeypatch, tmp_path, scripts, memory_interval=3, skill_interval=0)

    snapshots, spawns = _run_sequence(agent, 6)

    assert snapshots == [(0, 1), (0, 2), (0, 0), (0, 1), (0, 2), (1, 0)]
    assert len(spawns) == 1
    assert spawns[0]["review_memory"] is True
    assert spawns[0]["review_skills"] is False
    assert replies == []


def test_dual_trigger_mid_turn_memory_suppresses_only_memory_review(monkeypatch, tmp_path):
    """When both triggers are due and `memory` runs mid-turn, only skill review spawns."""
    memory_turn = [
        _reply(
            tool_name="memory",
            tool_args={"action": "add", "target": "memory", "content": "Saved inline."},
        ),
        _reply("Saved."),
    ]
    scripts = [[_reply("t1")], memory_turn]
    agent, replies = _make_live_agent(monkeypatch, tmp_path, scripts, memory_interval=2, skill_interval=3)

    spawns: list[dict] = []
    agent._spawn_background_review = lambda **kw: spawns.append(kw)

    history = agent.run_conversation("turn 1")["messages"]
    assert spawns == []
    assert agent._turns_since_memory == 1
    assert agent._iters_since_skill == 1

    # Turn 2 hits memory threshold (2) AND skill threshold (1 + 2 iters = 3),
    # but the foreground `memory` call clears only the memory trigger.
    agent.run_conversation("turn 2", conversation_history=history)
    assert len(spawns) == 1
    assert spawns[0]["review_memory"] is False
    assert spawns[0]["review_skills"] is True
    assert agent._turns_since_memory == 0
    assert agent._iters_since_skill == 0
    assert replies == []


def test_blocked_memory_tool_call_on_nth_turn_still_triggers_review(monkeypatch, tmp_path):
    """A pre_tool_call-blocked `memory` call must not suppress the due memory review."""
    memory_turn = [
        _reply(
            tool_name="memory",
            tool_args={"action": "add", "target": "memory", "content": "blocked"},
        ),
        _reply("Could not save."),
    ]
    scripts = [[_reply("t1")], memory_turn]
    agent, replies = _make_live_agent(monkeypatch, tmp_path, scripts, memory_interval=2, skill_interval=0)

    monkeypatch.setattr(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
        lambda *args, **kwargs: ("Blocked by policy", None),
    )
    spawns: list[dict] = []
    agent._spawn_background_review = lambda **kw: spawns.append(kw)

    history = agent.run_conversation("turn 1")["messages"]
    assert spawns == []
    assert agent._turns_since_memory == 1

    agent.run_conversation("turn 2", conversation_history=history)
    assert len(spawns) == 1
    assert spawns[0]["review_memory"] is True
    assert agent._turns_since_memory == 0
    assert replies == []


@pytest.mark.parametrize(
    "case",
    ["interrupted", "empty_response", "skip_background_review"],
)
def test_non_spawning_finalizer_exits_preserve_counters(case):
    """Interrupted, empty-response, and skip_background_review turns preserve both counters."""
    agent = _stub_agent_for_finalize(
        memory_interval=3,
        turns_since_memory=3,
        skill_interval=5,
        iters_since_skill=6,
        skip_background_review=(case == "skip_background_review"),
    )

    _call_finalize(
        agent,
        final_response="" if case == "empty_response" else "ok",
        interrupted=(case == "interrupted"),
    )

    agent._spawn_background_review.assert_not_called()
    assert agent._turns_since_memory == 3
    assert agent._iters_since_skill == 6


def test_spawn_exception_preserves_counters_and_retries_next_turn():
    """If _spawn_background_review raises, counters must NOT be consumed."""
    agent = _stub_agent_for_finalize(
        memory_interval=3,
        turns_since_memory=3,
        skill_interval=5,
        iters_since_skill=5,
    )
    agent._spawn_background_review.side_effect = RuntimeError("transient spawn failure")

    result = _call_finalize(agent, final_response="answer")
    assert result["final_response"] == "answer"
    agent._spawn_background_review.assert_called_once()
    assert agent._turns_since_memory == 3
    assert agent._iters_since_skill == 5

    agent._spawn_background_review.side_effect = None
    _call_finalize(agent, final_response="second answer")
    assert agent._spawn_background_review.call_count == 2
    assert agent._turns_since_memory == 0
    assert agent._iters_since_skill == 0


@pytest.mark.parametrize(
    ("turns", "iters", "expect_mem", "expect_skill", "after_turns", "after_iters"),
    [
        (3, 2, True, False, 0, 2),
        (1, 5, False, True, 1, 0),
    ],
    ids=["memory_only", "skill_only"],
)
def test_spawn_resets_only_counters_that_triggered(
    turns, iters, expect_mem, expect_skill, after_turns, after_iters
):
    """A single-trigger review resets only its own counter, preserving the other."""
    agent = _stub_agent_for_finalize(
        memory_interval=3,
        turns_since_memory=turns,
        skill_interval=5,
        iters_since_skill=iters,
    )
    _call_finalize(agent, final_response="answer")
    agent._spawn_background_review.assert_called_once()
    kwargs = agent._spawn_background_review.call_args.kwargs
    assert kwargs["review_memory"] is expect_mem
    assert kwargs["review_skills"] is expect_skill
    assert agent._turns_since_memory == after_turns
    assert agent._iters_since_skill == after_iters


@pytest.mark.parametrize(
    ("memory_interval", "with_memory_store"),
    [(0, True), (2, False)],
    ids=["interval_zero", "no_memory_store"],
)
def test_disabled_memory_nudge_never_increments_or_triggers(
    monkeypatch, tmp_path, memory_interval, with_memory_store
):
    """nudge_interval == 0 or disabled memory store never advances counter or spawns."""
    scripts = [[_reply("t1")], [_reply("t2")], [_reply("t3")]]
    agent, replies = _make_live_agent(
        monkeypatch,
        tmp_path,
        scripts,
        memory_interval=memory_interval,
        skill_interval=0,
        with_memory_store=with_memory_store,
    )

    snapshots, spawns = _run_sequence(agent, 3)

    assert snapshots == [(0, 0), (0, 0), (0, 0)]
    assert spawns == []
    assert replies == []


def test_codex_runtime_memory_and_skill_counters(monkeypatch, tmp_path):
    """Codex app-server runtime evaluates memory/skill triggers at turn end and resets only on spawn success."""
    from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)

    turn_results: list[TurnResult] = [
        TurnResult(final_text="reply 1", projected_messages=[{"role": "assistant", "content": "reply 1"}],
                   tool_iterations=2, turn_id="t1", thread_id="th1"),
        TurnResult(final_text="reply 2", projected_messages=[{"role": "assistant", "content": "reply 2"}],
                   tool_iterations=3, turn_id="t2", thread_id="th1", interrupted=True),
        TurnResult(final_text="", projected_messages=[],
                   tool_iterations=0, turn_id="t3", thread_id="th1"),
        TurnResult(final_text="reply 4", projected_messages=[{"role": "assistant", "content": "reply 4"}],
                   tool_iterations=1, turn_id="t4", thread_id="th1"),
        TurnResult(final_text="reply 5", projected_messages=[{"role": "assistant", "content": "reply 5"}],
                   tool_iterations=1, turn_id="t5", thread_id="th1"),
    ]
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "th1")
    monkeypatch.setattr(CodexAppServerSession, "run_turn", lambda self, **kw: turn_results.pop(0))

    with (
        patch("model_tools.get_tool_definitions", return_value=_TOOLS),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            model="gpt-5.4",
            provider="openai-codex",
            api_key="sk-test",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_app_server",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "You are helpful."
    agent._memory_store = object()
    agent._memory_nudge_interval = 2
    agent._skill_nudge_interval = 4
    agent._turns_since_memory = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = {"memory", "skill_manage"}

    spawn = MagicMock()
    agent._spawn_background_review = spawn

    # Turn 1: below both thresholds -> no spawn.
    agent.run_conversation("one")
    spawn.assert_not_called()
    assert (agent._turns_since_memory, agent._iters_since_skill) == (1, 2)

    # Turn 2: crosses both thresholds, interrupted -> no spawn, counters preserved.
    agent.run_conversation("two")
    spawn.assert_not_called()
    assert (agent._turns_since_memory, agent._iters_since_skill) == (2, 5)

    # Turn 3: empty final_text -> no spawn, counters preserved.
    agent.run_conversation("three")
    spawn.assert_not_called()
    assert (agent._turns_since_memory, agent._iters_since_skill) == (3, 5)

    # Turn 4: spawn raises -> exception suppressed, counters preserved.
    spawn.side_effect = RuntimeError("codex spawn boom")
    agent.run_conversation("four")
    assert spawn.call_count == 1
    assert spawn.call_args.kwargs["review_memory"] is True
    assert spawn.call_args.kwargs["review_skills"] is True
    assert (agent._turns_since_memory, agent._iters_since_skill) == (4, 6)

    # Turn 5: spawn succeeds -> both counters reset to 0.
    spawn.side_effect = None
    agent.run_conversation("five")
    assert spawn.call_count == 2
    assert spawn.call_args.kwargs["review_memory"] is True
    assert spawn.call_args.kwargs["review_skills"] is True
    assert (agent._turns_since_memory, agent._iters_since_skill) == (0, 0)


def test_codex_runtime_skip_background_review_preserves_counters(monkeypatch, tmp_path):
    """Codex runtime respects skip_background_review=True and preserves both counters."""
    from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "th1")
    monkeypatch.setattr(
        CodexAppServerSession,
        "run_turn",
        lambda self, **kw: TurnResult(
            final_text="done",
            projected_messages=[{"role": "assistant", "content": "done"}],
            tool_iterations=4,
            turn_id="t1",
            thread_id="th1",
        ),
    )

    with (
        patch("model_tools.get_tool_definitions", return_value=_TOOLS),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            model="gpt-5.4",
            provider="openai-codex",
            api_key="sk-test",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_app_server",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            skip_background_review=True,
        )
    agent._cached_system_prompt = "You are helpful."
    agent._memory_store = object()
    agent._memory_nudge_interval = 1
    agent._skill_nudge_interval = 2
    agent._turns_since_memory = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = {"memory", "skill_manage"}

    spawn = MagicMock()
    agent._spawn_background_review = spawn

    agent.run_conversation("work")
    spawn.assert_not_called()
    assert (agent._turns_since_memory, agent._iters_since_skill) == (1, 4)


def _write_chat_response(handler: BaseHTTPRequestHandler, *, stream: bool, spec: dict) -> None:
    content = spec.get("content")
    tool_calls = spec.get("tool_calls")
    finish = "tool_calls" if tool_calls else "stop"
    if stream:
        delta: dict = {"role": "assistant"}
        if content is not None:
            delta["content"] = content
        if tool_calls:
            delta["tool_calls"] = [{"index": idx, **tc} for idx, tc in enumerate(tool_calls)]
        chunks = [
            {"id": "chatcmpl-loopback", "model": "test/loopback",
             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {"id": "chatcmpl-loopback", "model": "test/loopback",
             "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        ]
        payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode("utf-8") + b"data: [DONE]\n\n"
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
        return
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body = json.dumps({
        "id": "chatcmpl-loopback",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "test/loopback",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _memory_tool_call_spec(call_id: str, content: str) -> dict:
    return {
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "memory",
                    "arguments": json.dumps({"action": "add", "target": "memory", "content": content}),
                },
            }
        ]
    }


def _start_loopback_openai_server():
    """Local OpenAI-compatible HTTP server distinguishing live turns from background reviews."""
    state = {
        "lock": threading.Lock(),
        "fg_turns": [],
        "review_calls": 0,
        "review_done": threading.Event(),
    }

    def _select_spec(msgs: list[dict]) -> dict:
        last_role = msgs[-1].get("role") if msgs else ""
        last_user = next(
            (m["content"] for m in reversed(msgs) if m.get("role") == "user" and isinstance(m.get("content"), str)),
            "",
        )
        if last_user.startswith("Review the conversation above"):
            if last_role == "tool":
                state["review_done"].set()
                return {"content": "Nothing else to save."}
            state["review_calls"] += 1
            idx = state["review_calls"]
            return _memory_tool_call_spec(f"call_bg_mem_{idx}", f"Background note #{idx}")
        state["fg_turns"].append((last_user, last_role))
        if last_user == "B2" and last_role != "tool":
            return _memory_tool_call_spec("call_fg_mem_1", "Foreground inline note")
        return {"content": f"reply:{last_user}"}

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            with state["lock"]:
                spec = _select_spec(payload.get("messages") or [])
            _write_chat_response(self, stream=bool(payload.get("stream")), spec=spec)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


def test_e2e_loopback_http_background_memory_review_lifecycle(monkeypatch, tmp_path):
    """End-to-end loopback HTTP test covering real background review spawning,
    mid-turn foreground memory suppression, and spawn-failure recovery."""
    for env_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(env_key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)

    server, state = _start_loopback_openai_server()
    port = server.server_address[1]
    try:
        with (
            patch("model_tools.get_tool_definitions", return_value=_TOOLS),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        ):
            agent = AIAgent(
                api_key="sk-loopback",
                base_url=f"http://127.0.0.1:{port}/v1",
                model="test/loopback",
                provider="custom",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            agent._cached_system_prompt = "You are helpful."
            store = MemoryStore(memory_enabled=True, user_profile_enabled=True)
            store.load_from_disk()
            agent._memory_store = store
            agent._memory_enabled = True
            agent._user_profile_enabled = True
            agent._memory_nudge_interval = 2
            agent._skill_nudge_interval = 0

            # --- Cycle A: threshold reached -> real background review thread runs over HTTP ---
            history = agent.run_conversation("A1")["messages"]
            assert agent._turns_since_memory == 1
            assert state["review_calls"] == 0

            history = agent.run_conversation("A2", conversation_history=history)["messages"]
            assert agent._turns_since_memory == 0
            assert state["review_done"].wait(timeout=15), "background review fork did not finish"
            run_token = getattr(agent, "_background_review_run", None)
            if run_token is not None:
                assert run_token.request_done.wait(timeout=15)
            assert state["review_calls"] == 1
            memory_file = tmp_path / "memories" / "MEMORY.md"
            assert "Background note #1" in memory_file.read_text(encoding="utf-8")

            # --- Cycle B: foreground `memory` call on threshold turn suppresses background review ---
            history = agent.run_conversation("B1", conversation_history=history)["messages"]
            assert agent._turns_since_memory == 1

            history = agent.run_conversation("B2", conversation_history=history)["messages"]
            assert agent._turns_since_memory == 0
            assert state["review_calls"] == 1  # no duplicate background review spawned
            assert "Foreground inline note" in memory_file.read_text(encoding="utf-8")

            # --- Cycle C: spawn failure preserves counter; next completed turn spawns real review ---
            history = agent.run_conversation("C1", conversation_history=history)["messages"]
            assert agent._turns_since_memory == 1

            real_spawn = agent._spawn_background_review
            agent._spawn_background_review = MagicMock(side_effect=RuntimeError("transient spawn error"))
            history = agent.run_conversation("C2", conversation_history=history)["messages"]
            assert agent._turns_since_memory == 2  # preserved across spawn failure
            assert state["review_calls"] == 1

            state["review_done"].clear()
            agent._spawn_background_review = real_spawn
            agent.run_conversation("C3", conversation_history=history)
            assert agent._turns_since_memory == 0
            assert state["review_done"].wait(timeout=15), "recovery background review fork did not finish"
            run_token = getattr(agent, "_background_review_run", None)
            if run_token is not None:
                assert run_token.request_done.wait(timeout=15)
            assert state["review_calls"] == 2
            assert "Background note #2" in memory_file.read_text(encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
