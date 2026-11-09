"""Plugin side questions on the live session agent (``agent/side_question_request.py``).

A real parent ``AIAgent`` runs a turn against a capturing fake client; the side call then
replays the lease captured at finalization through the turn loop's own request phases. The
contracts: the side request extends the parent's cached prefix byte-for-byte (system, tools,
request parameters, messages), tool calls in its reply never run, nothing but one usage row is
written, and refusals send no request at all.
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import side_question_lease
from agent.side_question_lease import redeem_side_question
from hermes_state import SessionDB
from run_agent import AIAgent

SIDE_PROMPT = "Suggest the user's next prompt in one line."
# Plugin side questions bill per plugin, apart from ``/btw``'s ``side_question``.
PLUGIN_TASK = "plugin_side_question:next-prompt"


def _tool_defs(*names: str) -> list:
    return [{
        "type": "function",
        "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object", "properties": {}}},
    } for name in names]


def _usage(prompt=1200, completion=20, cached=1000):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _response(content="", tool_calls=None, finish_reason="stop", usage=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], model="test/model", usage=usage,
    )


def _tool_call(name="web_search", call_id="call_1", arguments="{}"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


class _FakeProvider:
    """One fake client shared by the parent and every fork (``OpenAI(...)`` returns it)."""

    def __init__(self):
        self.client = MagicMock()
        self.requests: list[dict] = []
        self.replies: list = []
        self.client.chat.completions.create.side_effect = self._create

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply() if callable(reply) else reply


@pytest.fixture
def provider(tmp_path, monkeypatch):
    fake = _FakeProvider()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    # Every agent built during the test (parent and fork) talks to the same fake.
    patches = [
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("web_search", "terminal")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=fake.client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        # The side call only runs where a consented plugin holds the capability; that gate has
        # its own tests (test_side_question_lease.py).
        patch.object(side_question_lease, "_consented_plugin_loaded", return_value=True),
    ]
    for p in patches:
        p.start()
    yield fake
    for p in reversed(patches):
        p.stop()
    side_question_lease._leases.clear()


def _make_parent(model="test/model", **kwargs) -> AIAgent:
    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1", model=model, quiet_mode=True,
        skip_context_files=True, skip_memory=True, **kwargs,
    )
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.skip_background_review = True
    return agent


@pytest.fixture
def issued(monkeypatch):
    """Tokens the finalizer leased, in order."""
    tokens: list = []
    real = side_question_lease.issue_side_question_token

    def _spy(*args, **kwargs):
        tokens.append(real(*args, **kwargs))
        return tokens[-1]

    monkeypatch.setattr(side_question_lease, "issue_side_question_token", _spy)
    return tokens


def _run_parent_turn(parent, provider, *, tool_round=True):
    if tool_round:
        provider.replies += [
            _response(tool_calls=[_tool_call()], finish_reason="tool_calls", usage=_usage()),
            _response("Found it: the answer is 42.", usage=_usage()),
        ]
    else:
        provider.replies.append(_response("Found it: the answer is 42.", usage=_usage()))
    with patch("model_tools.handle_function_call", return_value=json.dumps({"result": "42"})):
        return parent.run_conversation("look this up")


def _canon(value):
    """Drop cache_control markers (they legitimately move to the newest rows) and collapse the
    text-block form a marker forces back to the plain string it wraps."""
    if isinstance(value, list):
        items = [_canon(v) for v in value]
        if items and all(isinstance(i, dict) and set(i) == {"type", "text"} and i["type"] == "text" for i in items):
            return "".join(i["text"] for i in items)
        return items
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items() if k != "cache_control"}
    return value


# ── 1. prefix parity ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["test/model", "anthropic/claude-sonnet-4"])
def test_side_request_extends_the_parents_cached_prefix(provider, issued, model):
    parent = _make_parent(model=model)
    result = _run_parent_turn(parent, provider)
    assert result["final_response"] == "Found it: the answer is 42."
    parent_last = provider.requests[-1]

    provider.replies.append(_response("Ask: why 42?", usage=_usage()))
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") == "Ask: why 42?"

    side = provider.requests[-1]
    assert len(provider.requests) == 3
    # Everything but the messages — system/tools/reasoning/max_tokens/tool_choice — is verbatim.
    assert _canon({k: v for k, v in side.items() if k != "messages"}) == \
        _canon({k: v for k, v in parent_last.items() if k != "messages"})
    parent_msgs, side_msgs = _canon(parent_last["messages"]), _canon(side["messages"])
    assert side_msgs[:len(parent_msgs)] == parent_msgs
    assert len(side_msgs) == len(parent_msgs) + 2
    assert side_msgs[-2]["role"] == "assistant" and side_msgs[-2]["content"] == "Found it: the answer is 42."
    assert side_msgs[-1] == {"role": "user", "content": SIDE_PROMPT}


def test_prompt_caching_runtime_marks_the_side_request(provider, issued):
    """The Claude-on-OpenRouter arm really exercises cache markers (else the parity test above
    would compare two unmarked requests)."""
    parent = _make_parent(model="anthropic/claude-sonnet-4")
    assert parent._use_prompt_caching
    _run_parent_turn(parent, provider)
    provider.replies.append(_response("ok", usage=_usage()))
    redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt")
    assert "cache_control" in json.dumps(provider.requests[-1]["messages"])


# ── 2. tool calls are never dispatched ───────────────────────────────────────────────────


@pytest.fixture
def dispatch_spy():
    calls: list = []
    lock = threading.Lock()

    def _record(name):
        def _fn(*args, **kwargs):
            with lock:
                calls.append((name, threading.current_thread().name))
            return "{}"
        return _fn

    with (
        patch("model_tools.handle_function_call", side_effect=_record("handle_function_call")),
        patch.object(AIAgent, "_invoke_tool", side_effect=_record("_invoke_tool"), autospec=False),
        patch.object(AIAgent, "_execute_tool_calls", side_effect=_record("_execute_tool_calls"), autospec=False),
    ):
        yield calls


def _parent_without_tools(provider):
    parent = _make_parent()
    provider.replies.append(_response("Found it: the answer is 42.", usage=_usage()))
    parent.run_conversation("look this up")
    return parent


def test_tool_calls_in_the_side_reply_are_dropped(provider, issued, dispatch_spy):
    _parent = _parent_without_tools(provider)  # alive for the call: the lease holds its parent weakly
    provider.replies.append(_response(
        "Try: rm the cache", tool_calls=[_tool_call("terminal", arguments='{"command": "rm -rf /tmp/x"}')],
        finish_reason="tool_calls", usage=_usage()))
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") == "Try: rm the cache"
    assert dispatch_spy == []


def test_tool_call_only_side_reply_is_none(provider, issued, dispatch_spy):
    _parent = _parent_without_tools(provider)  # alive for the call: the lease holds its parent weakly
    provider.replies.append(_response("", tool_calls=[_tool_call("terminal")], finish_reason="tool_calls"))
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") is None
    assert dispatch_spy == []


def test_side_call_never_reaches_the_parents_ui_callbacks(provider, issued):
    """A side call is invisible in the live chat: no stream, status, progress or thinking events."""
    parent = _make_parent()
    _run_parent_turn(parent, provider, tool_round=False)
    seen: list = []
    for name in ("stream_delta_callback", "status_callback", "tool_progress_callback", "thinking_callback",
                 "reasoning_callback", "step_callback", "interim_assistant_callback", "background_review_callback"):
        setattr(parent, name, lambda *a, _n=name, **k: seen.append(_n))
    parent._stream_callback = lambda *a, **k: seen.append("_stream_callback")
    provider.replies.append(_response("<think>plan</think>Ask: why 42?", usage=_usage()))
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") == "Ask: why 42?"
    assert seen == []
    assert provider.requests[-1].get("stream") is not True


# ── 3. no writes besides usage ───────────────────────────────────────────────────────────


def _message_rows(db: SessionDB, session_id: str) -> list:
    with db._lock:
        return [tuple(r) for r in db._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()]


def _usage_rows(db: SessionDB, session_id: str) -> list[dict]:
    with db._lock:
        return [dict(r) for r in db._conn.execute(
            "SELECT * FROM session_model_usage WHERE session_id = ?", (session_id,)).fetchall()]


def test_side_call_writes_only_one_usage_row(provider, issued, tmp_path):
    from hermes_constants import get_hermes_home

    memories = Path(get_hermes_home()) / "memories"
    memories.mkdir(parents=True, exist_ok=True)
    (memories / "MEMORY.md").write_text("user likes tea\n", encoding="utf-8")
    db = SessionDB(db_path=tmp_path / "state.db")
    parent = _make_parent(session_db=db, session_id="sess-side")
    _run_parent_turn(parent, provider)
    rows_before = _message_rows(db, "sess-side")
    usage_before = [r for r in _usage_rows(db, "sess-side") if r["task"] == PLUGIN_TASK]
    assert rows_before and usage_before == []
    memory_before = {p.name: p.read_bytes() for p in memories.iterdir()}

    provider.replies.append(_response("Ask: why 42?", usage=_usage()))
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") == "Ask: why 42?"

    assert _message_rows(db, "sess-side") == rows_before
    side_rows = [r for r in _usage_rows(db, "sess-side") if r["task"] == PLUGIN_TASK]
    assert len(side_rows) == 1 and side_rows[0]["api_call_count"] == 1
    assert {p.name: p.read_bytes() for p in memories.iterdir()} == memory_before


def test_side_call_bills_the_session_the_token_was_issued_for(provider, issued, tmp_path):
    """The parent's session id can rotate between issue and redeem (``/compress``, a session
    reset): the side call's spend belongs to the session whose turn it replays."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent = _make_parent(session_db=db, session_id="sess-issued")
    _run_parent_turn(parent, provider, tool_round=False)
    parent.session_id = "sess-rotated"

    provider.replies.append(_response("Ask: why 42?", usage=_usage()))
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") == "Ask: why 42?"

    assert [r["api_call_count"] for r in _usage_rows(db, "sess-issued") if r["task"] == PLUGIN_TASK] == [1]
    assert _usage_rows(db, "sess-rotated") == []


def test_side_call_leaves_the_parent_sessions_terminal_environment_alive(provider, issued):
    """The fork shares the parent's session_id, and CLI/gateway key a session's terminal
    sandbox by it: tearing the fork down must not tear down the session's live tool state."""
    from tools import terminal_tool

    class _Env:
        cleaned = False

        def cleanup(self, **_kwargs):
            _Env.cleaned = True

    parent = _make_parent(session_id="sess-env")
    _run_parent_turn(parent, provider, tool_round=False)
    env = _Env()
    terminal_tool._active_environments["sess-env"] = env
    try:
        provider.replies.append(_response("Ask: why 42?", usage=_usage()))
        assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") == "Ask: why 42?"
        assert terminal_tool._active_environments.get("sess-env") is env
        assert not _Env.cleaned
    finally:
        terminal_tool._active_environments.pop("sess-env", None)


def test_side_call_runs_under_the_turns_profile_and_secret_scope(provider, issued, tmp_path):
    """The plugin (of the turn's profile) redeems from its own thread, where no profile is bound;
    the request still resolves the parent turn's HERMES_HOME and secret scope."""
    from agent.secret_scope import current_secret_scope_home, reset_secret_scope, set_secret_scope
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    parent = _make_parent()
    home_token = set_hermes_home_override(profile_home)
    scope_token = set_secret_scope({"OPENROUTER_API_KEY": "sk-work"}, profile_home=str(profile_home))
    try:
        _run_parent_turn(parent, provider, tool_round=False)
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)

    seen: dict = {}

    def _reply():
        seen.update(home=str(get_hermes_home()), secret_home=current_secret_scope_home())
        return _response("Ask: why 42?", usage=_usage())

    provider.replies.append(_reply)
    answers: list = []
    worker = threading.Thread(
        target=lambda: answers.append(
            redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt", home=profile_home)))
    worker.start()
    worker.join(10)
    assert answers == ["Ask: why 42?"]
    assert seen == {"home": str(profile_home), "secret_home": str(profile_home)}


# ── 4. no recursive post_llm_call ────────────────────────────────────────────────────────


def test_side_call_fires_no_post_llm_call(provider, issued):
    parent = _make_parent()
    _run_parent_turn(parent, provider, tool_round=False)
    fired: list = []
    real_invoke = __import__("hermes_cli.plugins", fromlist=["invoke_hook"]).invoke_hook

    def _spy(name, **kwargs):
        fired.append(name)
        return real_invoke(name, **kwargs)

    provider.replies.append(_response("Ask: why 42?", usage=_usage()))
    with patch("hermes_cli.plugins.invoke_hook", side_effect=_spy), \
            patch("hermes_cli.lifecycle.invoke_hook", side_effect=_spy, create=True):
        assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") == "Ask: why 42?"
    assert "post_llm_call" not in fired and "transform_llm_output" not in fired


# ── 5. refusals send no request ──────────────────────────────────────────────────────────


def test_moa_runtime_is_refused(provider, issued):
    parent = _make_parent()
    _run_parent_turn(parent, provider, tool_round=False)
    parent.provider = "moa"
    sent = len(provider.requests)
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") is None
    assert len(provider.requests) == sent


def test_agent_as_provider_is_refused(provider, issued):
    parent = _make_parent()
    _run_parent_turn(parent, provider, tool_round=False)
    provider.client.SUPPORTS_HERMES_TOOL_CALLS = False  # an agent CLI shim runs its own tools
    sent = len(provider.requests)
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") is None
    assert len(provider.requests) == sent


def test_runtime_changed_since_the_turn_is_refused(provider, issued):
    """A turn routed to another model (voice chat) restores the main runtime after
    ``post_llm_call``: the captured prompt must never go out on a different model."""
    parent = _make_parent()
    _run_parent_turn(parent, provider, tool_round=False)
    parent.model = "other/model"
    sent = len(provider.requests)
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") is None
    assert len(provider.requests) == sent


def test_server_side_tool_in_final_request_is_refused(provider, issued, monkeypatch):
    parent = _make_parent()
    _run_parent_turn(parent, provider, tool_round=False)
    real_build = __import__("agent.turn_api_request", fromlist=["build_request_payload"]).build_request_payload

    def _with_native_search(*args, **kwargs):
        built = real_build(*args, **kwargs)
        built.api_kwargs["tools"] = [*built.api_kwargs["tools"], {"type": "web_search"}]
        return built

    monkeypatch.setattr("agent.turn_api_request.build_request_payload", _with_native_search)
    sent = len(provider.requests)
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") is None
    assert len(provider.requests) == sent


@pytest.mark.parametrize("api_kwargs, server_side", [
    ({"tools": [{"type": "function", "function": {"name": "x"}}]}, False),
    ({"tools": [{"name": "x", "input_schema": {}}]}, False),  # Anthropic client tool: no type
    ({"tools": [{"type": "custom", "name": "x"}]}, False),
    ({"tools": [{"type": "web_search"}]}, True),
    ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}, True),
    ({"extra_body": {"plugins": [{"id": "pareto-router", "min_coding_score": 0.5}]}}, False),
    ({"extra_body": {"plugins": [{"id": "web"}]}}, True),
    ({"web_search_options": {}}, True),
    # request_overrides land in extra_body, whose ``tools`` the SDK merge sends over the top level.
    ({"extra_body": {"tools": [{"type": "web_search"}]}}, True),
    ({"extra_body": {"tools": [{"type": "function", "function": {"name": "x"}}]}}, False),
])
def test_server_side_tool_predicate(api_kwargs, server_side):
    from agent.side_question_request import has_server_side_tools

    assert has_server_side_tools(api_kwargs) is server_side


# ── 6. timeout and provider errors ───────────────────────────────────────────────────────


def test_timeout_interrupts_the_inflight_request(provider, issued):
    parent = _make_parent()
    _run_parent_turn(parent, provider, tool_round=False)
    released = threading.Event()
    interrupted: list = []
    real_interrupt = AIAgent.interrupt

    def _spy_interrupt(self, *args, **kwargs):
        interrupted.append(kwargs.get("tool_reason"))
        released.set()
        return real_interrupt(self, *args, **kwargs)

    def _hang():
        # Held until the timer interrupts the fork, then fail like an aborted connection.
        assert released.wait(10)
        raise InterruptedError("aborted")

    provider.replies.append(_hang)
    with patch.object(AIAgent, "interrupt", _spy_interrupt):
        assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt", timeout=0.5) is None
    assert interrupted == ["side question timed out"]


def test_timeout_counts_the_fork_build(provider, issued, monkeypatch):
    """Runtime resolution in the fork build is part of the caller's budget: a build that
    outlasts the timeout sends nothing."""
    from agent import side_question_request

    parent = _make_parent()
    _run_parent_turn(parent, provider, tool_round=False)
    real_build = side_question_request._build_fork

    def _slow_build(*args, **kwargs):
        # Outlast the budget: hold the build until the side question's timer has fired (a Timer
        # thread that is already gone has fired too).
        for timer in [t for t in threading.enumerate() if isinstance(t, threading.Timer)
                      and getattr(t.function, "__name__", "") == "_on_timeout"]:
            assert timer.finished.wait(10)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(side_question_request, "_build_fork", _slow_build)
    provider.replies.append(_response("too late", usage=_usage()))
    sent = len(provider.requests)
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt", timeout=0.1) is None
    assert len(provider.requests) == sent


def test_provider_error_returns_none_without_fallback(provider, issued):
    parent = _make_parent(fallback_model={"provider": "openrouter", "model": "fallback/model"})
    _run_parent_turn(parent, provider, tool_round=False)
    provider.replies += [RuntimeError("upstream 500"), _response("from the fallback", usage=_usage())]
    sent = len(provider.requests)
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") is None
    assert len(provider.requests) == sent + 1
    assert all(r.get("model") != "fallback/model" for r in provider.requests)
