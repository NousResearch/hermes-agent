"""Attempt-scoped provider lifecycle: stale-worker isolation + detached aux.

Each real provider attempt owns a unique lifecycle token captured ONCE at the
attempt's main entry. Transports, terminal callbacks and error classifiers
operate only on the token they captured — a stale late worker from an older
attempt can never flip the current attempt's phase, and auxiliary calls
(memory/iteration-limit summaries) run on fresh detached tokens instead of
inheriting the main loop's (possibly already terminal) token.

Both races here are deterministic: attempt A is abandoned by the stale
detector while its worker is still alive; attempt B must classify its own
pre-terminal network error as retryable regardless of A's late terminal.
"""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent.chat_completion_helpers as cch
from run_agent import AIAgent

LOCAL_REASON = "local_post_response_error"


def _chunk(text=None, finish=None, usage=None):
    delta = SimpleNamespace(content=text, reasoning_content=None, reasoning=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], model="m", usage=usage)


class _FakeStreamClient:
    def __init__(self, factory):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._factory = factory

    def _create(self, **kw):
        self.calls += 1
        return self._factory(self.calls)

    def close(self):
        pass


def _agent(fake):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(model="m", api_key="k", base_url="https://x.invalid/v1",
                        provider="custom", quiet_mode=False, skip_context_files=True,
                        skip_memory=True)
    agent._cached_system_prompt = "s"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_iterations = 4
    agent._disable_streaming = False
    agent.stream_delta_callback = lambda t: None
    agent.client = fake
    return agent


def _response(text="ok"):
    msg = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_stale_late_worker_cannot_flip_current_attempt():
    """A stale attempt's LATE terminal report must not flip the CURRENT
    attempt's phase. Non-streaming path, fully deterministic:

    1. Attempt A's create blocks on a gate — its worker stays alive when the
       main-side stale-call detector (0.2s) abandons the attempt.
    2. Attempt B starts in-flight (new token), releases A's gate, waits until
       A's late terminal mark has REALLY landed on A's token via the actual
       ``_dispatch_nonstreaming_api_request`` code path, then hits a
       pre-terminal network error.
    3. B MUST retry normally (no local_post_response_error) and C succeeds.
    """
    gate_a = threading.Event()
    tokens = []

    real_new = cch._new_provider_attempt
    def spy_new(agent):
        tok = real_new(agent)
        tokens.append(tok)
        return tok

    fake = MagicMock()

    def create_side_effect(**_kw):
        n = fake.client.chat.completions.create.call_count
        if n == 1:
            # Attempt A: block until B releases us — the worker outlives the
            # stale kill, then returns normally so the dispatch helper marks
            # terminal on the token IT captured (tokens[0]).
            gate_a.wait(timeout=30)
            return _response("a-late")
        if n == 2:
            # Attempt B is now in-flight. Let A's late terminal report land
            # for real, then fail pre-terminal.
            gate_a.set()
            deadline = time.time() + 15
            while not tokens[0].terminal_received:
                if time.time() > deadline:
                    raise AssertionError("A's late terminal mark never landed")
                time.sleep(0.01)
            raise ConnectionError("attempt B wire down before terminal")
        # Attempt C: success.
        return _response("final")

    fake.client.chat.completions.create.side_effect = create_side_effect

    agent = _agent(fake.client)
    agent._disable_streaming = True
    # Shrink the non-streaming stale-call timeout so attempt A is abandoned
    # quickly while its worker stays blocked on the gate.
    agent._compute_non_stream_stale_timeout = lambda _payload: 0.2

    with ExitStack() as st:
        st.enter_context(patch("run_agent.jittered_backoff", return_value=0))
        st.enter_context(patch.object(agent, "_save_trajectory"))
        st.enter_context(patch.object(agent, "_persist_session"))
        st.enter_context(patch.object(agent, "_cleanup_task_resources"))
        st.enter_context(patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake.client))
        st.enter_context(patch.object(cch, "_new_provider_attempt", spy_new))
        try:
            result = agent.run_conversation("race probe")
        finally:
            gate_a.set()

    # Three distinct attempts: A stale, B network error, C success.
    assert len(tokens) == 3
    assert tokens[0] is not tokens[1] and tokens[1] is not tokens[2]
    # The late report landed on A's own token (allowed), never on B's.
    assert tokens[0].terminal_received
    assert not tokens[1].terminal_received, (
        f"stale terminal write leaked into attempt B: {tokens[1].phase}"
    )
    # B retried normally: three wire calls, no local post-response error.
    assert fake.client.chat.completions.create.call_count == 3
    assert result["completed"] is True
    assert result["final_response"] == "final"
    assert result.get("turn_exit_reason") != LOCAL_REASON
    # The loop's current attempt is C and reached terminal legitimately.
    assert agent._provider_attempt is tokens[2]
    assert tokens[2].terminal_received



def test_worker_delayed_before_dispatch_captures_own_attempt():
    """Delayed-worker regression. A non-streaming worker that is delayed
    BEFORE entering ``_dispatch_nonstreaming_api_request`` must still mark
    ITS OWN attempt token — never the newer attempt's.

    Deterministic sequence:

    1. Attempt A's worker starts but blocks at the dispatch door (gate) —
       it is still parked there when the main-side stale-call detector
       (0.2s) abandons the attempt and the loop creates attempt B.
    2. B runs in-flight, releases A's gate, and waits until A's late
       terminal mark has REALLY landed (via the real dispatch path).
    3. Assert A's token is terminal and B's is NOT, then B hits a
       pre-terminal ConnectionError, retries normally, and C succeeds.

    With the old capture-at-dispatch code, A's delayed worker would re-read
    agent._provider_attempt inside the dispatch helper, capture B's token,
    and flip B terminal — B's network error would then be misclassified as
    local_post_response_error and never retried.
    """
    gate_a = threading.Event()
    tokens = []

    real_new = cch._new_provider_attempt
    def spy_new(agent):
        tok = real_new(agent)
        tokens.append(tok)
        return tok

    real_dispatch = cch._dispatch_nonstreaming_api_request
    dispatch_calls = {"n": 0}
    _tls = threading.local()
    def gated_dispatch(agent, api_kwargs, *, make_client, attempt):
        dispatch_calls["n"] += 1
        ordinal = dispatch_calls["n"]
        # Tag this thread so create_side_effect knows which ATTEMPT it is
        # serving — A's worker is parked pre-dispatch, so the create call
        # ORDER does not match the attempt order.
        _tls.ordinal = ordinal
        if ordinal == 1:
            # Attempt A's worker parks HERE — before any provider I/O and
            # before any attempt capture inside the dispatch helper.
            gate_a.wait(timeout=30)
        return real_dispatch(agent, api_kwargs, make_client=make_client, attempt=attempt)

    fake = MagicMock()

    def create_side_effect(**_kw):
        ordinal = getattr(_tls, "ordinal", None)
        if ordinal == 1:
            # Attempt A (released by B below): returns normally so the real
            # dispatch path marks terminal on the token it was HANDED at the
            # attempt entry (tokens[0]).
            return _response("a-late")
        if ordinal == 2:
            # Attempt B is now in-flight with its own token. Release A's
            # parked worker and wait for A's late terminal mark to land.
            gate_a.set()
            deadline = time.time() + 15
            while not tokens[0].terminal_received:
                if time.time() > deadline:
                    raise AssertionError("A's late terminal mark never landed")
                time.sleep(0.01)
            # A marked its own token; B's must still be untouched.
            assert not tokens[1].terminal_received, (
                f"stale pre-dispatch worker flipped attempt B: {tokens[1].phase}"
            )
            raise ConnectionError("attempt B wire down before terminal")
        # Attempt C: success.
        return _response("final")

    fake.client.chat.completions.create.side_effect = create_side_effect

    agent = _agent(fake.client)
    agent._disable_streaming = True
    agent._compute_non_stream_stale_timeout = lambda _payload: 0.2

    with ExitStack() as st:
        st.enter_context(patch("run_agent.jittered_backoff", return_value=0))
        st.enter_context(patch.object(agent, "_save_trajectory"))
        st.enter_context(patch.object(agent, "_persist_session"))
        st.enter_context(patch.object(agent, "_cleanup_task_resources"))
        st.enter_context(patch.object(agent, "_create_request_openai_client", lambda *a, **k: fake.client))
        st.enter_context(patch.object(cch, "_new_provider_attempt", spy_new))
        st.enter_context(patch.object(cch, "_dispatch_nonstreaming_api_request", gated_dispatch))
        try:
            result = agent.run_conversation("race probe")
        finally:
            gate_a.set()

    # Three distinct attempts: A stale (worker parked pre-dispatch), B
    # pre-terminal network error, C success.
    assert len(tokens) == 3
    assert tokens[0] is not tokens[1] and tokens[1] is not tokens[2]
    assert tokens[0].terminal_received
    assert not tokens[1].terminal_received, (
        f"stale terminal write leaked into attempt B: {tokens[1].phase}"
    )
    assert fake.client.chat.completions.create.call_count == 3
    assert result["completed"] is True
    assert result["final_response"] == "final"
    assert result.get("turn_exit_reason") != LOCAL_REASON
    assert agent._provider_attempt is tokens[2]
    assert tokens[2].terminal_received



def test_aux_codex_stream_keeps_internal_reconnect_and_does_not_touch_loop_token():
    """An auxiliary Codex call (e.g. iteration-limit summary) invoked WITHOUT
    an explicit attempt must run on a detached token:

    - ``agent._provider_attempt`` holds an OLD, already-terminal main-loop
      token;
    - the auxiliary stream's FIRST connection dies pre-terminal
      (ConnectionError) — the codex runtime's one internal reconnect must
      still fire (wire create_calls == 2);
    - the old loop token must be left completely untouched.
    """
    agent = _agent(MagicMock())

    old_token = cch.ProviderAttemptLifecycle()
    old_token.mark(cch.PROVIDER_PHASE_TERMINAL_RECEIVED)
    agent._provider_attempt = old_token

    terminal_event = SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            status="completed", id="r1", model="m",
            usage=None, incomplete_details=None, error=None,
        ),
    )

    class _Stream:
        def __init__(self, events):
            self._events = events
        def __iter__(self):
            return iter(self._events)
        def close(self):
            pass

    create = MagicMock(
        side_effect=[
            ConnectionError("aux stream wire down before terminal"),
            _Stream([terminal_event]),
        ]
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=create), close=lambda: None)

    result = agent._run_codex_stream({"model": "m"}, client=client, attempt=None)

    # Internal reconnect preserved: first pre-terminal failure retried once.
    assert create.call_count == 2
    assert getattr(result, "status", "completed") == "completed"
    # The old main-loop token was never adopted or mutated by the aux call.
    assert agent._provider_attempt is old_token
    assert old_token.phase == cch.PROVIDER_PHASE_TERMINAL_RECEIVED



def test_aux_anthropic_create_marks_only_detached_token():
    """``_anthropic_messages_create(..., attempt=None)`` (auxiliary path) must
    fire ``on_response_received`` on a detached token, never on
    ``agent._provider_attempt``."""
    captured = {}

    def fake_create_anthropic_message(client, api_kwargs, *, log_prefix="", prefer_stream=False, on_response_received=None):
        captured["on_response_received"] = on_response_received
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], stop_reason="end_turn")

    agent = _agent(MagicMock())
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._is_anthropic_oauth = False
    old_token = cch.ProviderAttemptLifecycle()
    agent._provider_attempt = old_token

    with patch("agent.anthropic_adapter.create_anthropic_message", fake_create_anthropic_message):
        agent._anthropic_messages_create({"model": "claude-test", "messages": []})

    cb = captured.get("on_response_received")
    assert callable(cb)
    cb()  # simulate the provider terminal callback
    # The callback marked a DETACHED token; the loop's token is untouched.
    assert old_token.phase == cch.PROVIDER_PHASE_IN_FLIGHT
    assert agent._provider_attempt is old_token
