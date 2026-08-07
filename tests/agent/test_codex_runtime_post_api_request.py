"""Codex app-server turns must carry the ordinary loop's API lifecycle.

The chat-completions loop wraps every provider request in
``pre_api_request`` → ``post_api_request`` / ``api_request_error``, dispatched
through ``hermes_cli.lifecycle`` so first-party observability is notified
ahead of the compatibility plugin hooks.  The codex app-server runtime is an
early-return path that bypasses that loop entirely, so subscription-billed
codex turns reached no API-hook consumer at all: usage ledgers, spend guards
and the built-in Relay shared-metrics observer were blind to them.

Consumers key a model call on ``api_request_id`` and drop a terminal event
whose start they never observed, so the pre/post pair — and one terminal
event on every exit path — is the actual contract, not just the post hook.
"""
import time

import pytest

from agent.codex_runtime import run_codex_app_server_turn
from agent.transports.codex_app_server_session import TurnResult
from run_agent import AIAgent

_API_HOOKS = ("pre_api_request", "post_api_request", "api_request_error")

# docs/observability/README.md — "Request-Scoped API Hooks".  post_api_request
# delivers the identity/runtime fields plus these.  A field with no honest
# app-server equivalent is reported as None, never fabricated or omitted.
_POST_CONTRACT_FIELDS = (
    "session_id", "task_id", "turn_id", "api_request_id",
    "platform", "model", "provider", "base_url", "api_mode",
    "api_duration", "started_at", "ended_at",
    "finish_reason", "message_count", "response_model",
    "usage", "assistant_content_chars", "assistant_tool_call_count",
    "response", "assistant_message",
)


class _FakeCodexSession:
    """Stands in for the codex subprocess client, without spawning one."""

    def __init__(self, turn=None, raises=None):
        self._turn = turn
        self._raises = raises
        self.closed = False

    def run_turn(self, *, user_input):
        self.user_input = user_input
        if self._raises is not None:
            raise self._raises
        return self._turn

    def close(self):
        self.closed = True


def _turn(**overrides):
    turn = TurnResult(
        final_text="codex answer",
        projected_messages=[
            {
                "role": "assistant",
                "content": "codex answer",
                "tool_calls": [{"id": "call_1", "function": {"name": "shell"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
        tool_iterations=1,
        thread_id="thread-1",
        turn_id="codex-turn-1",
        token_usage_last={
            "inputTokens": 12000,
            "cachedInputTokens": 3000,
            "outputTokens": 800,
            "reasoningOutputTokens": 200,
            "totalTokens": 16000,
        },
    )
    for name, value in overrides.items():
        setattr(turn, name, value)
    return turn


def _agent(session):
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.model = "gpt-5.1-codex"
    agent.provider = "openai-codex"
    agent.api_mode = "codex_app_server"
    agent.platform = "cli"
    agent.session_id = "sess-codex"
    # Stamped by turn_context before the runtime is handed the turn.
    agent._current_turn_id = "turn-abc"
    agent._codex_session = session
    agent.tool_progress_callback = None
    return agent


def _run(agent, messages=None):
    return run_codex_app_server_turn(
        agent,
        user_message="hello codex",
        original_user_message="hello codex",
        messages=messages if messages is not None else [
            {"role": "user", "content": "hello codex"},
        ],
        effective_task_id="task-1",
    )


@pytest.fixture()
def lifecycle_events(monkeypatch):
    """Record what the real lifecycle dispatcher hands each kind of consumer.

    Nothing here stubs ``hermes_cli.lifecycle`` itself: the runtime has to go
    through it for either consumer to be reached at all.
    """
    from hermes_cli import observability, plugins

    events = []

    monkeypatch.setattr(
        observability, "handles_hook", lambda hook_name: hook_name in _API_HOOKS
    )
    monkeypatch.setattr(
        observability,
        "observe_lifecycle",
        lambda hook_name, **kwargs: events.append(("builtin", hook_name, kwargs)),
    )

    manager = plugins.get_plugin_manager()
    saved = {hook: list(manager._hooks.get(hook, [])) for hook in _API_HOOKS}
    for hook in _API_HOOKS:
        manager._hooks[hook] = [
            lambda _hook=hook, **kwargs: events.append(("plugin", _hook, kwargs))
        ]
    try:
        yield events
    finally:
        for hook, callbacks in saved.items():
            manager._hooks[hook] = callbacks


def _of(events, kind, hook_name):
    return [
        kwargs
        for got_kind, got_hook, kwargs in events
        if got_kind == kind and got_hook == hook_name
    ]


def test_turn_reaches_builtin_observability_and_plugins(lifecycle_events):
    """Both lifecycle consumers see the turn, built-in observer first."""
    _run(_agent(_FakeCodexSession(_turn())))

    assert [(kind, hook) for kind, hook, _ in lifecycle_events] == [
        ("builtin", "pre_api_request"),
        ("plugin", "pre_api_request"),
        ("builtin", "post_api_request"),
        ("plugin", "post_api_request"),
    ]


def test_pre_and_post_correlate_on_one_api_request_id(lifecycle_events):
    """A terminal event is dropped by consumers unless it matches its start."""
    _run(_agent(_FakeCodexSession(_turn())))

    pre = _of(lifecycle_events, "builtin", "pre_api_request")[0]
    post = _of(lifecycle_events, "builtin", "post_api_request")[0]

    assert pre["turn_id"] == "turn-abc"
    assert pre["api_request_id"].startswith("turn-abc")
    for field in ("task_id", "turn_id", "api_request_id", "session_id",
                  "platform", "model", "provider", "base_url", "api_mode",
                  "api_call_count"):
        assert post[field] == pre[field], field


def test_tool_hooks_inside_the_turn_share_the_request_id(lifecycle_events):
    """Tool-level hooks correlate to the request that caused them."""
    agent = _agent(_FakeCodexSession(_turn()))
    _run(agent)

    pre = _of(lifecycle_events, "builtin", "pre_api_request")[0]
    assert agent._current_api_request_id == pre["api_request_id"]


def test_post_reports_the_documented_contract(lifecycle_events):
    """Payload parity with the ordinary loop, honest values only."""
    started = time.time()
    _run(_agent(_FakeCodexSession(_turn())))

    post = _of(lifecycle_events, "builtin", "post_api_request")[0]
    for field in _POST_CONTRACT_FIELDS:
        assert field in post, field

    assert post["finish_reason"] == "stop"
    assert post["api_call_count"] == 1
    assert post["message_count"] == 1
    assert post["started_at"] >= started
    assert post["ended_at"] >= post["started_at"]
    assert post["api_duration"] == pytest.approx(
        post["ended_at"] - post["started_at"]
    )
    assert post["assistant_content_chars"] == len("codex answer")
    assert post["assistant_tool_call_count"] == 1
    # The app server never echoes the model it served the turn with, and no
    # normalized provider message object exists on this path.
    assert post["response_model"] is None
    assert post["assistant_message"] is None
    assert post["response"]["assistant_message"]["content"] == "codex answer"
    assert post["response"]["finish_reason"] == "stop"

    usage = post["usage"]
    assert usage["input_tokens"] == 12000
    assert usage["cache_read_tokens"] == 3000
    assert usage["output_tokens"] == 800
    assert usage["total_tokens"] == 16000
    # ``usage`` is token buckets; cost belongs to the turn result, not here.
    assert "estimated_cost_usd" not in usage


def test_turn_without_token_usage_still_ends_the_model_call(lifecycle_events):
    """Codex omits usage on some turns; they still cost an API call."""
    agent = _agent(_FakeCodexSession(_turn(token_usage_last=None)))
    _run(agent)

    post = _of(lifecycle_events, "builtin", "post_api_request")
    assert len(post) == 1
    assert post[0]["usage"] is None
    assert post[0]["api_call_count"] == 1
    assert agent.session_api_calls == 1


def test_interrupted_turn_reports_its_finish_reason(lifecycle_events):
    _run(_agent(_FakeCodexSession(_turn(interrupted=True, final_text=""))))

    post = _of(lifecycle_events, "builtin", "post_api_request")[0]
    assert post["finish_reason"] == "interrupt"


def test_failed_turn_terminates_the_model_call(lifecycle_events):
    """A turn codex never returned must not leave a start hanging open."""
    session = _FakeCodexSession(raises=RuntimeError("app-server died"))
    result = _run(_agent(session))

    assert result["completed"] is False
    hooks = [hook for kind, hook, _ in lifecycle_events if kind == "builtin"]
    assert hooks == ["pre_api_request", "api_request_error"]

    pre = _of(lifecycle_events, "builtin", "pre_api_request")[0]
    err = _of(lifecycle_events, "builtin", "api_request_error")[0]
    assert err["api_request_id"] == pre["api_request_id"]
    assert err["retryable"] is False
    assert err["error"] == {
        "type": "RuntimeError",
        "message": "app-server died",
    }


def test_pre_describes_the_request_hermes_actually_sent(lifecycle_events):
    _run(_agent(_FakeCodexSession(_turn())))

    pre = _of(lifecycle_events, "builtin", "pre_api_request")[0]
    assert pre["request"]["body"]["input"] == "hello codex"
    assert pre["request_char_count"] == len("hello codex")
    assert pre["request_messages"] == [
        {"role": "user", "content": "hello codex"},
    ]


def test_broken_lifecycle_dispatch_never_breaks_the_turn(monkeypatch):
    """Observability is best-effort; the turn and its accounting survive it."""
    from hermes_cli import lifecycle

    def _boom(*args, **kwargs):
        raise RuntimeError("dispatch exploded")

    monkeypatch.setattr(lifecycle, "has_hook", lambda hook_name: True)
    monkeypatch.setattr(lifecycle, "invoke_hook", _boom)

    agent = _agent(_FakeCodexSession(_turn()))
    result = _run(agent)

    assert result["completed"] is True
    assert result["final_response"] == "codex answer"
    assert agent.session_api_calls == 1
    assert agent.session_total_tokens == 16000
