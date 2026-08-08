from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from run_agent import AIAgent


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def test_moa_virtual_provider_aggregator_is_actor(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if kwargs["task"] == "moa_reference":
            return _response("reference advice")
        return _response("aggregator acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    agent = AIAgent(
        api_key="moa-virtual-provider",
        base_url="http://127.0.0.1/v1",
        model="review",
        provider="moa",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["file"],
        max_iterations=1,
    )
    monkeypatch.setattr(
        agent,
        "_create_request_openai_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("MoA calls must use MoAClient, not a request OpenAI client")
        ),
    )

    result = agent.run_conversation("solve this")

    assert result["final_response"] == "aggregator acted"
    assert agent.base_url == "moa://local"
    assert [(c["task"], c["provider"], c["model"]) for c in calls] == [
        ("moa_reference", "openai-codex", "gpt-5.5"),
        ("moa_aggregator", "openrouter", "anthropic/claude-opus-4.8"),
    ]
    assert calls[1]["tools"] is not None


def test_moa_runtime_provider_uses_virtual_endpoint():
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="moa", target_model="review")

    assert runtime["provider"] == "moa"
    assert runtime["base_url"] == "moa://local"
    assert runtime["api_key"] == "moa-virtual-provider"


def test_moa_primary_restore_rebuilds_virtual_facade(monkeypatch, tmp_path):
    """MoA sessions must restore from fallback without constructing OpenAI().

    Regression for a long-lived MoA session that failed over to a real provider:
    the next turn restored provider/model to MoA but tried to rebuild the shared
    client from MoA's empty client_kwargs, raising "api_key client option must be
    set" and then "Failed to recreate closed OpenAI client".
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    agent = AIAgent(
        api_key="moa-virtual-provider",
        base_url="moa://local",
        model="review",
        provider="moa",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["file"],
        max_iterations=1,
    )
    primary_client = agent.client

    def fail_openai_rebuild(*_args, **_kwargs):
        raise AssertionError("MoA restore must not build a real OpenAI client")

    monkeypatch.setattr(agent, "_create_openai_client", fail_openai_rebuild)
    setattr(agent, "_fallback_activated", True)
    setattr(agent, "provider", "zai")
    setattr(agent, "model", "glm-5.2")
    agent.base_url = "https://api.z.ai/api/coding/paas/v4"
    agent.api_key = "fallback-key"
    setattr(agent, "_client_kwargs", {"api_key": "fallback-key", "base_url": agent.base_url})
    agent.client = SimpleNamespace(close=lambda: None, _client=SimpleNamespace(is_closed=True))

    assert agent._restore_primary_runtime() is True
    assert getattr(agent, "provider") == "moa"
    assert getattr(agent, "model") == "review"
    assert agent.client is not primary_client
    assert hasattr(agent.client.chat, "completions")
    assert getattr(agent, "_fallback_activated") is False


def test_moa_restored_facade_still_emits_reference_events(monkeypatch, tmp_path):
    """A restored MoA facade must keep the reference_callback relay wired.

    Regression for the naive-rebuild flaw in the original #53802 approach:
    ``MoAClient(preset)`` without ``reference_callback`` restores a *working*
    facade that silently stops emitting ``moa.reference``/``moa.aggregating``
    display events for the rest of the session. The shared ``build_moa_facade``
    factory rewires the relay to ``agent.tool_progress_callback`` on restore.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    agent = AIAgent(
        api_key="moa-virtual-provider",
        base_url="moa://local",
        model="review",
        provider="moa",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["file"],
        max_iterations=1,
    )

    # Simulate a fallback to a real provider, then restore.
    setattr(agent, "_fallback_activated", True)
    setattr(agent, "provider", "zai")
    setattr(agent, "model", "glm-5.2")
    agent.base_url = "https://api.z.ai/api/coding/paas/v4"
    agent.api_key = "fallback-key"
    setattr(agent, "_client_kwargs", {"api_key": "fallback-key", "base_url": agent.base_url})
    agent.client = SimpleNamespace(close=lambda: None, _client=SimpleNamespace(is_closed=True))
    assert agent._restore_primary_runtime() is True

    # The relay reads tool_progress_callback at emit time — attach a recorder
    # and fire the facade's internal _emit exactly as the fan-out does.
    events = []

    def record_progress(event, *args, **kwargs):
        events.append((event, args, kwargs))

    agent.tool_progress_callback = record_progress
    completions = agent.client.chat.completions
    assert completions.reference_callback is not None, (
        "restored MoA facade lost its reference_callback relay"
    )
    completions._emit(
        "moa.reference", index=0, count=1, label="openai-codex/gpt-5.5", text="advice"
    )
    completions._emit("moa.aggregating", aggregator="openrouter", ref_count=1)

    assert [e[0] for e in events] == ["moa.reference", "moa.aggregating"]
    ref_event = events[0]
    assert ref_event[1][0] == "openai-codex/gpt-5.5"
    assert ref_event[1][1] == "advice"
    assert ref_event[2] == {"moa_index": 0, "moa_count": 1}
















def test_call_llm_extra_headers_reach_transport_create(monkeypatch):
    """extra_headers must reach the SDK client's create() kwargs.

    Transport-boundary regression for #60293: mocking call_llm proves nothing
    about delivery — this asserts the header survives call_llm's request
    building and lands in the kwargs handed to chat.completions.create().
    """
    from types import SimpleNamespace

    from agent import auxiliary_client as ac

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _response("ok")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
        base_url="https://api.githubcopilot.com",
    )
    monkeypatch.setattr(
        ac,
        "_resolve_task_provider_model",
        lambda *a, **k: (
            "copilot",
            "claude-sonnet-4.6",
            "https://api.githubcopilot.com",
            "copilot-token",
            "chat_completions",
        ),
    )
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (fake_client, "claude-sonnet-4.6"))
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task, **_kw: resp)

    ac.call_llm(
        provider="copilot",
        model="claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        extra_headers={"x-initiator": "user"},
    )

    assert captured.get("extra_headers") == {"x-initiator": "user"}
    # And it must not leak into unrelated request fields.
    assert "x-initiator" not in captured.get("extra_body", {}) if captured.get("extra_body") else True


def test_retry_same_provider_sync_preserves_extra_headers(monkeypatch):
    """The same-provider retry rebuild must carry extra_headers through.

    Regression for #60293's follow-up: a credential-refresh/pool-rotation
    retry rebuilds the request kwargs from scratch — without forwarding
    extra_headers, the retried Copilot advisor call silently loses its
    ``x-initiator: user`` attribution and can be rejected.
    """
    from types import SimpleNamespace

    from agent import auxiliary_client as ac

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _response("retried ok")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
        base_url="https://api.githubcopilot.com",
    )
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (fake_client, "claude-sonnet-4.6"))
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, task, **_kw: resp)

    ac._retry_same_provider_sync(
        task=None,
        resolved_provider="copilot",
        resolved_model="claude-sonnet-4.6",
        resolved_base_url="https://api.githubcopilot.com",
        resolved_api_key="copilot-token",
        resolved_api_mode="chat_completions",
        main_runtime=None,
        final_model="claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        temperature=None,
        max_tokens=None,
        tools=None,
        effective_timeout=30.0,
        effective_extra_body={},
        reasoning_config=None,
        extra_headers={"x-initiator": "user"},
    )

    assert captured.get("extra_headers") == {"x-initiator": "user"}






def test_reference_messages_drops_system_but_renders_tools_as_text():
    """System prompt is dropped, but tool calls + results are RENDERED as text.

    A reference must see what the agent did (tool calls) and what came back
    (tool results) to give an informed judgement — so neither is stripped. They
    are flattened to text so the view carries zero tool-role messages / no
    tool_calls arrays (strict providers reject those), while the reference
    still has the full picture. The view ends on a user turn.
    """
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "system", "content": "huge hermes system prompt"},
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "tool result"},
        {"role": "assistant", "content": "here is my answer"},
    ]

    view = _reference_messages(messages)

    # Wire-format safety: only user/assistant text, no tool roles / tool_calls.
    assert all(m["role"] in ("user", "assistant") for m in view)
    assert all("tool_calls" not in m for m in view)
    # System prompt is gone.
    assert all("huge hermes system prompt" not in m["content"] for m in view)
    # The agent's action and the tool result are PRESERVED as text.
    joined = "\n".join(m["content"] for m in view)
    assert "[called tool: f(" in joined
    assert "[tool result: tool result]" in joined
    assert "here is my answer" in joined
    # Ends on a user turn (advisory request appended after the final assistant).
    assert view[-1]["role"] == "user"


def test_reference_messages_ends_with_user_not_assistant_prefill():
    """Advisory reference views must never end on an assistant turn.

    Mid-tool-loop the conversation ends on an assistant/tool exchange. Anthropic
    (and OpenRouter→Anthropic) treat a trailing assistant turn as an assistant
    prefill to continue, and no-prefill models (e.g. Claude Opus 4.8) reject it
    with ``400 ... must end with a user message``. We append a synthetic user
    turn asking for judgement rather than DELETING the agent's latest context —
    the reference must still see the current state to advise on it.
    """
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2 current"},
        {
            "role": "assistant",
            "content": "let me reason then call a tool",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "the tool output"},
    ]

    view = _reference_messages(messages)

    assert view, "advisory view should not be empty"
    assert view[-1]["role"] == "user"
    joined = "\n".join(m["content"] for m in view)
    # The agent's latest action and its result are preserved, not dropped.
    assert "let me reason then call a tool" in joined
    assert "[called tool: f(" in joined
    assert "[tool result: the tool output]" in joined
    # Earlier context preserved too.
    assert "q1" in joined and "a1" in joined and "q2 current" in joined








def test_run_reference_prepends_advisory_system_prompt(monkeypatch):
    """Each reference call gets the advisory-role system prompt first.

    Without it the reference assumes it is the acting agent and refuses ("I
    can't access repositories/URLs from here") or tries to call tools it
    doesn't have. The system prompt reframes it as an analyst advising the
    aggregator, and the advisory transcript still ends on a user turn.
    """
    from agent.moa_loop import _REFERENCE_SYSTEM_PROMPT, _run_reference

    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response("advice")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    label, text, _acct = _run_reference(
        {"provider": "openai-codex", "model": "gpt-5.5"},
        [{"role": "user", "content": "review this PR"}],
    )

    assert text == "advice"
    msgs = captured["messages"]
    assert msgs[0] == {"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}
    assert msgs[-1]["role"] == "user"








def test_references_run_in_parallel(monkeypatch):
    """References fan out concurrently (delegate-batch semantics), not serially.

    Each reference sleeps; wall-time must approximate the slowest single call,
    not the sum. Order is preserved and a failing reference is isolated.
    """
    import time

    from agent import moa_loop

    # Force _extract_text down its fallback path (no transport normalize).
    monkeypatch.setattr(moa_loop, "get_transport", lambda *_a, **_k: None)

    barrier_hits = []

    def slow_call_llm(**kwargs):
        barrier_hits.append(time.monotonic())
        model = kwargs["model"]
        if model == "boom":
            raise RuntimeError("kaboom")
        time.sleep(0.5)
        return _response(f"resp-{kwargs['provider']}")

    monkeypatch.setattr(moa_loop, "call_llm", slow_call_llm)

    refs = [
        {"provider": "p1", "model": "ok"},
        {"provider": "moa", "model": "preset"},  # recursion guard, not dispatched
        {"provider": "p2", "model": "boom"},  # failure isolated
        {"provider": "p3", "model": "ok"},
    ]

    start = time.monotonic()
    out = moa_loop._run_references_parallel(
        refs, [{"role": "user", "content": "hi"}], temperature=0.6, max_tokens=64
    )
    elapsed = time.monotonic() - start

    # Two 0.5s sleeps run concurrently → well under the 1.0s serial floor.
    # Threshold sits at 0.95s (not tight against 0.5s) to tolerate CI
    # thread-pool startup jitter while still failing hard if the two calls
    # ran serially (which would be ≥1.0s).
    assert elapsed < 0.95, f"references did not run in parallel (took {elapsed:.2f}s)"
    # Output order matches input order (stable Reference N labelling).
    assert [label for label, _, _ in out] == ["p1:ok", "moa:preset", "p2:boom", "p3:ok"]
    assert "recursively reference MoA" in out[1][1]
    assert out[2][1].startswith("[failed:")
    assert out[0][1] == "resp-p1"


def test_references_parallel_without_agent_is_unaffected(monkeypatch):
    """No agent passed (the pre-fix call shape) must behave exactly as
    before: block until every reference completes, no interrupt check."""
    import time

    from agent import moa_loop

    monkeypatch.setattr(moa_loop, "get_transport", lambda *_a, **_k: None)
    # Poll interval shorter than the reference's own sleep so the assertion
    # below would catch a regression that waits a whole poll cycle extra.
    monkeypatch.setattr(moa_loop, "_REFERENCE_POLL_INTERVAL_S", 0.05)

    def slow_call_llm(**kwargs):
        time.sleep(0.2)
        return _response(f"resp-{kwargs['provider']}")

    monkeypatch.setattr(moa_loop, "call_llm", slow_call_llm)

    refs = [{"provider": "p1", "model": "ok"}]
    out = moa_loop._run_references_parallel(
        refs, [{"role": "user", "content": "hi"}],
    )

    assert out[0][1] == "resp-p1"


def test_references_parallel_interrupt_aborts_wait(monkeypatch):
    """A user interrupt mid-fanout must stop the wait instead of blocking
    until every reference (including a wedged one) finishes or times out on
    its own — mirroring the interrupt check agent.tool_executor already
    applies to its own concurrent tool batch."""
    import threading
    import time

    from agent import moa_loop

    monkeypatch.setattr(moa_loop, "get_transport", lambda *_a, **_k: None)
    monkeypatch.setattr(moa_loop, "_REFERENCE_POLL_INTERVAL_S", 0.05)

    fake_agent = SimpleNamespace(_interrupt_requested=False)
    release_wedged = threading.Event()

    def fake_call_llm(**kwargs):
        if kwargs["provider"] == "fast":
            # Simulate the interrupt arriving right after the fast reference
            # finishes, while the wedged one is still in flight.
            fake_agent._interrupt_requested = True
            return _response("fast output")
        # "wedged" — never returns within the test unless released, standing
        # in for a reference whose own (possibly very long) timeout hasn't
        # elapsed yet.
        release_wedged.wait(timeout=5)
        return _response("should not be observed")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)

    refs = [
        {"provider": "fast", "model": "m1"},
        {"provider": "wedged", "model": "m2"},
    ]
    try:
        start = time.monotonic()
        out = moa_loop._run_references_parallel(
            refs, [{"role": "user", "content": "hi"}], agent=fake_agent,
        )
        elapsed = time.monotonic() - start

        # Must return promptly once interrupted, not block for the wedged
        # reference's full (5s test-simulated) duration.
        assert elapsed < 2.0, f"interrupt did not abort the wait (took {elapsed:.2f}s)"
        assert out[0][1] == "fast output"
        assert "interrupted" in out[1][1]
    finally:
        release_wedged.set()  # don't leak a blocked thread past the test


def _ref_config(home, fanout: str | None = None):
    home.mkdir()
    fanout_line = f"\n      fanout: {fanout}" if fanout else ""
    (home / "config.yaml").write_text(
        f"""
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
        - provider: openrouter
          model: anthropic/claude-opus-4.8
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8{fanout_line}
""".strip(),
        encoding="utf-8",
    )








def test_slot_runtime_anthropic_oauth_routes_through_provider_branch(monkeypatch):
    """Native anthropic slots must keep their provider identity, not collapse to custom.

    anthropic OAuth setup-tokens (sk-ant-oat*) require Bearer auth + the
    ``anthropic-beta: oauth-*`` header, which only the anthropic provider branch
    of call_llm adds. _slot_runtime forwards the resolved base_url/api_key for
    every provider now; the single chokepoint that must NOT collapse anthropic
    to provider=custom (which would send the token as x-api-key → bare 429) is
    _resolve_task_provider_model via _preserve_provider_with_base_url.
    """
    from agent import moa_loop
    from agent.auxiliary_client import _resolve_task_provider_model

    def fake_resolve(*, requested, target_model=None):
        return {
            "provider": requested,
            "base_url": "https://resolved.example/v1",
            "api_key": "resolved-key",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )

    # _slot_runtime forwards the resolved endpoint for anthropic like any slot.
    anthropic_rt = moa_loop._slot_runtime(
        {"provider": "anthropic", "model": "claude-opus-4-8"}
    )
    assert anthropic_rt["provider"] == "anthropic"
    assert anthropic_rt["base_url"] == "https://resolved.example/v1"

    # The chokepoint preserves anthropic identity despite the explicit base_url,
    # so call_llm routes through the anthropic provider branch (not custom).
    resolved_provider, _model, base_url, _api_key, _mode = _resolve_task_provider_model(
        task="moa_reference",
        provider="anthropic",
        model="claude-opus-4-8",
        base_url="https://resolved.example/v1",
        api_key="resolved-key",
    )
    assert resolved_provider == "anthropic"

    # A generic provider (openrouter) is likewise forwarded and preserved.
    other_rt = moa_loop._slot_runtime(
        {"provider": "openrouter", "model": "some-model"}
    )
    assert other_rt["provider"] == "openrouter"
    assert other_rt["model"] == "some-model"
    assert other_rt["base_url"] == "https://resolved.example/v1"
    assert other_rt["api_key"] == "resolved-key"


def _response_with_usage(content="advice", *, prompt=100, completion=50, cached=0):
    """A fake response carrying OpenAI-style usage so normalize_usage works."""
    details = SimpleNamespace(cached_tokens=cached, cache_write_tokens=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=details,
        output_tokens_details=None,
    )
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage, model="fake-model")


def test_run_reference_captures_usage_and_cost(monkeypatch):
    """A reference call returns per-advisor CanonicalUsage + priced cost.

    Before this, _run_reference discarded response.usage entirely, so the
    advisor fan-out was invisible to cost tracking.
    """
    from agent.moa_loop import _RefAccounting, _run_reference
    from agent.usage_pricing import CanonicalUsage

    monkeypatch.setattr(
        "agent.moa_loop.call_llm",
        lambda **kw: _response_with_usage(prompt=1000, completion=200, cached=400),
    )
    # Keep runtime resolution + pricing deterministic.
    monkeypatch.setattr(
        "agent.moa_loop._slot_runtime",
        lambda slot: {"provider": "openrouter", "model": slot.get("model")},
    )
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *a, **k: SimpleNamespace(amount_usd=0.0123, status="estimated", source="table"),
    )

    label, text, acct = _run_reference(
        {"provider": "openrouter", "model": "vendor/adv-model"},
        [{"role": "user", "content": "state?"}],
    )

    assert text == "advice"
    assert isinstance(acct, _RefAccounting)
    assert isinstance(acct.usage, CanonicalUsage)
    # prompt_tokens=1000 with 400 cached → 600 fresh input + 400 cache_read.
    assert acct.usage.input_tokens == 600
    assert acct.usage.cache_read_tokens == 400
    assert acct.usage.output_tokens == 200
    assert acct.cost_usd == 0.0123




def test_canonical_usage_add():
    """CanonicalUsage sums per bucket (used to fold advisor tokens in)."""
    from agent.usage_pricing import CanonicalUsage

    a = CanonicalUsage(input_tokens=100, output_tokens=20, cache_read_tokens=5)
    b = CanonicalUsage(input_tokens=50, output_tokens=10, cache_write_tokens=3)
    total = a + b
    assert total.input_tokens == 150
    assert total.output_tokens == 30
    assert total.cache_read_tokens == 5
    assert total.cache_write_tokens == 3
    assert total.request_count == 2






def test_reference_guidance_appended_at_end_in_tool_loop():
    """In an agentic loop the reference block must land at the END of the prompt.

    The most recent user turn is the original task near the top of the context;
    merging the per-turn (volatile) reference block into it would diverge the
    prompt prefix early and defeat the server's KV-cache reuse, forcing a full
    re-prefill of the whole conversation on every tool-loop step.
    """
    from agent.moa_loop import _attach_reference_guidance

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "ORIGINAL TASK"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "tool result", "tool_call_id": "1"},
    ]
    _attach_reference_guidance(messages, "REFERENCE BLOCK")

    # The original (top-of-context) user turn is untouched, so the prefix stays
    # cache-reusable across steps.
    assert messages[1]["content"] == "ORIGINAL TASK"
    # The reference block is appended as a new trailing turn, not merged upstream.
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "REFERENCE BLOCK"
    assert len(messages) == 5




def test_reference_messages_flattens_cache_decorated_content():
    """Cache-decorated turns (content-part lists) must not blind the references.

    conversation_loop runs apply_anthropic_cache_control BEFORE the MoA facade
    when the preset's aggregator is a cache-honoring Claude route (post-#57675).
    That converts string content into [{"type": "text", "text": ...,
    "cache_control": ...}] lists. The advisory view previously read only string
    content, so the user's ENTIRE prompt flattened to "" — Claude references
    then 400'd ("messages: at least one message is required") while tolerant
    models answered "no user request is present" (live incident, Jul 14 2026,
    preset "closed", session 20260714_001520_28157b).
    """
    from agent.moa_loop import _reference_messages
    from agent.prompt_caching import apply_anthropic_cache_control

    plain = [
        {"role": "system", "content": "hermes system prompt"},
        {"role": "user", "content": "Can we get codex usage resets into hermes?"},
    ]
    decorated = apply_anthropic_cache_control(plain, native_anthropic=False)
    # Premise: decoration really converts the user turn to a content-part list.
    assert isinstance(decorated[1]["content"], list)

    view = _reference_messages(decorated)

    assert view == [
        {"role": "user", "content": "Can we get codex usage resets into hermes?"}
    ]
    # Invariant: decorated and undecorated transcripts produce the SAME
    # advisory view — so decoration can never change what references see,
    # and the advisory prefix stays byte-stable for advisor prompt caching.
    assert view == _reference_messages(plain)













def test_prepared_aggregator_preserves_reasoning_config(monkeypatch):
    """Prepared MoA requests retain the acting aggregator reasoning policy."""
    from agent import moa_loop

    captured = {}
    expected_reasoning = {"enabled": True, "effort": "high"}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response("aggregator acted")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(moa_loop, "_aggregator_reasoning_config", lambda _slot: expected_reasoning)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )

    facade = moa_loop.MoAChatCompletions("review")
    facade._call_prepared_aggregator(
        {
            "messages": [{"role": "user", "content": "question"}],
            "aggregator": {"provider": "openrouter", "model": "aggregator"},
            "aggregator_temperature": None,
        },
        {},
    )

    assert captured["reasoning_config"] == expected_reasoning


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("copilot", "https://api.githubcopilot.com"),
        ("github-copilot", "https://models.github.ai/inference"),
    ],
)
def test_prepared_copilot_aggregator_marks_only_first_call_user_initiated(
    monkeypatch, provider, base_url
):
    from agent import moa_loop

    calls = []
    agent = SimpleNamespace(_is_user_initiated_turn=True)

    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": slot["provider"],
            "model": slot["model"],
            "base_url": base_url,
        },
    )
    monkeypatch.setattr(
        moa_loop,
        "call_llm",
        lambda **kwargs: calls.append(kwargs) or _response("aggregator acted"),
    )

    facade = moa_loop.MoAChatCompletions("review", agent=agent)
    prepared = {
        "messages": [{"role": "user", "content": "question"}],
        "aggregator": {"provider": provider, "model": "claude-sonnet-4.6"},
        "aggregator_temperature": None,
    }
    request_headers = {"x-trace-id": "keep", "x-initiator": "agent"}

    facade._call_prepared_aggregator(
        prepared,
        {"extra_headers": request_headers},
    )
    facade._call_prepared_aggregator(
        prepared,
        {"extra_headers": request_headers},
    )

    assert calls[0]["extra_headers"] == {
        "x-trace-id": "keep",
        "x-initiator": "user",
    }
    assert calls[1]["extra_headers"] == request_headers
    assert request_headers == {"x-trace-id": "keep", "x-initiator": "agent"}
    assert agent._is_user_initiated_turn is False


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("custom:copilot", "https://custom.example/v1"),
        ("custom:github-copilot", "https://custom.example/v1"),
        ("copilot", "https://custom.example/v1"),
        ("github-copilot", "https://custom.example/v1"),
        ("copilot-acp", ""),
        ("github-copilot-acp", ""),
        ("copilot-acp-agent", ""),
    ],
)
def test_prepared_non_http_copilot_aggregator_preserves_request_headers(
    monkeypatch, provider, base_url
):
    from agent import moa_loop

    captured = {}
    agent = SimpleNamespace(_is_user_initiated_turn=True)
    request_headers = {"x-trace-id": "keep"}

    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": slot["provider"],
            "model": slot["model"],
            "base_url": base_url,
        },
    )
    monkeypatch.setattr(
        moa_loop,
        "call_llm",
        lambda **kwargs: captured.update(kwargs) or _response("aggregator acted"),
    )

    facade = moa_loop.MoAChatCompletions("review", agent=agent)
    facade._call_prepared_aggregator(
        {
            "messages": [{"role": "user", "content": "question"}],
            "aggregator": {"provider": provider, "model": "aggregator"},
            "aggregator_temperature": None,
        },
        {"extra_headers": request_headers},
    )

    assert captured["extra_headers"] == request_headers
    assert agent._is_user_initiated_turn is True







def test_aggregate_moa_context_sanitizes_failed_reference_and_forwards_timeout(monkeypatch):
    from agent import moa_loop
    from agent.usage_pricing import CanonicalUsage

    outputs = [
        ("good-model", "useful advice", moa_loop._RefAccounting(CanonicalUsage())),
        (
            "bad-model",
            "[failed: HTTP 401 key=super-secret]",
            moa_loop._RefAccounting(CanonicalUsage()),
        ),
    ]
    fanout_kwargs = {}
    aggregator_calls = []

    def fake_fanout(*args, **kwargs):
        fanout_kwargs.update(kwargs)
        return outputs

    def fake_call_llm(**kwargs):
        aggregator_calls.append(kwargs)
        return _response("synthesized guidance")

    monkeypatch.setattr(moa_loop, "_run_references_parallel", fake_fanout)
    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )

    result = moa_loop.aggregate_moa_context(
        user_prompt="review this",
        api_messages=[{"role": "user", "content": "review this"}],
        reference_models=[
            {"provider": "openrouter", "model": "good-model"},
            {"provider": "openrouter", "model": "bad-model"},
        ],
        aggregator={"provider": "openrouter", "model": "aggregator"},
        reference_timeout=17.5,
        degraded_reference_policy="loud",
    )

    assert fanout_kwargs["reference_timeout"] == 17.5
    private_prompt = aggregator_calls[0]["messages"][0]["content"]
    assert "useful advice" in private_prompt
    assert "super-secret" not in private_prompt
    assert "Reference models unavailable: bad-model" in private_prompt
    assert "super-secret" not in result






def test_aggregate_skips_aggregator_when_all_references_failed(monkeypatch):
    """When every reference returns [failed: …], the aggregator is skipped entirely."""
    from agent.moa_loop import aggregate_moa_context

    call_count = {"n": 0}

    def fake_call_llm(**kwargs):
        call_count["n"] += 1
        if kwargs["task"] == "moa_reference":
            raise RuntimeError("provider down key=super-secret")
        raise AssertionError("aggregator should not be called when all references fail")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    monkeypatch.setattr(
        "agent.moa_loop._slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )

    result = aggregate_moa_context(
        user_prompt="do something",
        api_messages=[{"role": "user", "content": "do something"}],
        reference_models=[
            {"provider": "openai", "model": "gpt-4"},
            {"provider": "anthropic", "model": "claude-opus"},
        ],
        aggregator={"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    )

    # The aggregator LLM call was never made.
    assert call_count["n"] == 2  # only the two reference calls
    # The result carries a sanitized unavailability notice (never raw
    # provider error text) so the main agent can still act.
    assert "all reference models failed" in result
    assert "Reference models unavailable" in result
    assert "super-secret" not in result




def _facade_all_failed_fixture(monkeypatch, tmp_path, policy):
    """Common scaffolding: a 'review' preset whose references ALL fail."""
    from agent import moa_loop
    from agent.usage_pricing import CanonicalUsage

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"""
moa:
  default_preset: review
  presets:
    review:
      degraded_reference_policy: {policy}
      reference_models:
        - provider: openrouter
          model: bad-model-a
        - provider: openrouter
          model: bad-model-b
      aggregator:
        provider: openrouter
        model: aggregator
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    outputs = [
        (
            "bad-model-a",
            "[failed: HTTP 401 key=super-secret]",
            moa_loop._RefAccounting(CanonicalUsage(input_tokens=5), 0.05),
        ),
        (
            "bad-model-b",
            "[failed: timeout after 900s]",
            moa_loop._RefAccounting(CanonicalUsage(input_tokens=3), 0.03),
        ),
    ]
    aggregator_calls = []

    def fake_call_llm(**kwargs):
        aggregator_calls.append(kwargs)
        return _response("aggregator acted alone")

    monkeypatch.setattr(moa_loop, "_run_references_parallel", lambda *a, **k: outputs)
    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )
    return moa_loop, outputs, aggregator_calls






def test_interrupted_but_completed_reference_keeps_real_accounting(monkeypatch):
    """A reference that finishes between the interrupt check and the reap
    must keep its REAL output and accounting — the call billed."""
    from concurrent.futures import wait as real_wait

    from agent import moa_loop

    monkeypatch.setattr(moa_loop, "get_transport", lambda *_a, **_k: None)
    monkeypatch.setattr(moa_loop, "_REFERENCE_POLL_INTERVAL_S", 0.05)

    fake_agent = SimpleNamespace(_interrupt_requested=True)

    def fake_call_llm(**kwargs):
        return _response_with_usage("slowish output", prompt=11, completion=4)

    # Force the exact race: the wait loop reports the future as still
    # pending (so the interrupt path is taken) even though the underlying
    # call has already completed — the reap must then hit the done() branch
    # and keep the real result instead of writing a placeholder.
    def fake_wait(pending, timeout=None):
        real_wait(pending)  # let the call actually finish (it billed)
        return set(), set(pending)  # report it as still pending

    monkeypatch.setattr(moa_loop, "_futures_wait", fake_wait)
    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )

    out = moa_loop._run_references_parallel(
        [{"provider": "slowish", "model": "m1"}],
        [{"role": "user", "content": "hi"}],
        agent=fake_agent,
    )

    # The completed call's real output + usage must survive the reap.
    assert out[0][1] == "slowish output"
    acct = out[0][2]
    assert isinstance(acct, moa_loop._RefAccounting)
    assert acct.usage.input_tokens == 11


def test_late_completing_interrupted_reference_feeds_accounting_sink(monkeypatch):
    """A reference still in flight at interrupt time gets a placeholder in
    the results, but its eventual REAL accounting must reach the sink."""
    import threading
    import time

    from agent import moa_loop

    monkeypatch.setattr(moa_loop, "get_transport", lambda *_a, **_k: None)
    monkeypatch.setattr(moa_loop, "_REFERENCE_POLL_INTERVAL_S", 0.05)

    fake_agent = SimpleNamespace(_interrupt_requested=False)
    release = threading.Event()
    sink_calls = []
    sink_seen = threading.Event()

    def sink(label, accounting):
        sink_calls.append((label, accounting))
        sink_seen.set()

    def fake_call_llm(**kwargs):
        if kwargs["provider"] == "fast":
            fake_agent._interrupt_requested = True
            return _response("fast output")
        # wedged: blocks past the interrupt, completes later.
        release.wait(timeout=5)
        return _response_with_usage("late output", prompt=21, completion=2)

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )

    out = moa_loop._run_references_parallel(
        [
            {"provider": "fast", "model": "m1"},
            {"provider": "wedged", "model": "m2"},
        ],
        [{"role": "user", "content": "hi"}],
        agent=fake_agent,
        late_accounting_sink=sink,
    )

    # The wedged slot returned a placeholder with zeroed accounting…
    assert out[1][1] == moa_loop._INTERRUPTED_REFERENCE_NOTE
    assert out[1][2].usage.input_tokens == 0

    # …then completes late; its real billed usage must reach the sink.
    release.set()
    assert sink_seen.wait(timeout=5), "late accounting sink never called"
    label, acct = sink_calls[0]
    assert "wedged" in label
    assert acct.usage.input_tokens == 21


def test_facade_does_not_cache_interrupted_reference_results(monkeypatch, tmp_path):
    """An interrupted fan-out is a partial snapshot — caching it would replay
    placeholder notes on every later iteration of the turn. The facade must
    leave the cache empty so the next create() re-runs the references, and
    a late-completing reference's real spend must land in pending usage."""
    from agent import moa_loop
    from agent.usage_pricing import CanonicalUsage

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openrouter
          model: advisor
      aggregator:
        provider: openrouter
        model: aggregator
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    interrupted_outputs = [
        (
            "openrouter:advisor",
            moa_loop._INTERRUPTED_REFERENCE_NOTE,
            moa_loop._RefAccounting(CanonicalUsage()),
        )
    ]

    def fake_fanout(*args, **kwargs):
        return list(interrupted_outputs)

    monkeypatch.setattr(moa_loop, "_run_references_parallel", fake_fanout)
    monkeypatch.setattr(moa_loop, "call_llm", lambda **k: _response("acted"))
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )

    facade = moa_loop.MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "go"}], tools=[])

    # Interrupted results must not be cached as this state's advice.
    assert facade._ref_cache_key is None
    assert facade._ref_cache_outputs == []

    # A late completion depositing real spend is picked up by consume().
    facade._record_late_reference_accounting(
        "openrouter:advisor",
        moa_loop._RefAccounting(CanonicalUsage(input_tokens=33), 0.42),
    )
    usage, cost = facade.consume_reference_usage()
    assert usage.input_tokens == 33
    assert cost == pytest.approx(0.42)
    # And consume() drained it — no double count.
    usage2, cost2 = facade.consume_reference_usage()
    assert usage2.input_tokens == 0
    assert cost2 is None


class _CountingCtxLen:
    """Stub for get_model_context_length that counts resolutions."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def _trim(messages, *, window=1000, reserve=None, cache=None, counting=None,
          monkeypatch=None):
    from agent import model_metadata, moa_loop

    stub = counting or _CountingCtxLen(window)
    monkeypatch.setattr(model_metadata, "get_model_context_length", stub)
    return moa_loop._trim_messages_for_reference(
        messages,
        {"provider": "openrouter", "model": "small-window"},
        {"provider": "openrouter", "model": "small-window"},
        reserve_output_tokens=reserve,
        context_length_cache=cache,
    )


def _advisory_view(n_pairs, chunk="x" * 400):
    """A text-only advisory view: system + n user/assistant pairs + trailing user."""
    msgs = [{"role": "system", "content": "advisory system prompt"}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"u{i} {chunk}"})
        msgs.append({"role": "assistant", "content": f"a{i} {chunk}"})
    msgs.append({"role": "user", "content": "judge the state above"})
    return msgs












def test_reference_trim_context_length_cache_hits_once(monkeypatch):
    """A shared per-turn cache resolves each (provider, model) window once."""
    cache = {}
    stub = _CountingCtxLen(10_000_000)
    msgs = _advisory_view(2)
    for _ in range(4):
        _trim(list(msgs), cache=cache, counting=stub, monkeypatch=monkeypatch)
    assert stub.calls == 1
    assert cache == {("openrouter", "small-window"): 10_000_000}


def test_reference_trim_caches_resolution_failures(monkeypatch):
    """A failing metadata source is probed once, not per reference call."""
    cache = {}
    stub = _CountingCtxLen(RuntimeError("metadata down"))
    msgs = _advisory_view(2)
    for _ in range(3):
        out = _trim(list(msgs), cache=cache, counting=stub, monkeypatch=monkeypatch)
        assert out == msgs
    assert stub.calls == 1
    assert cache == {("openrouter", "small-window"): None}




def test_render_tool_calls_tolerates_namespace_shapes():
    """SDK-shaped (SimpleNamespace) tool_call entries must render their real
    function name+args, not degrade to '[called tool: tool]'."""
    from agent.moa_loop import _render_tool_calls

    ns_call = SimpleNamespace(
        function=SimpleNamespace(name="web_search", arguments='{"query": "x"}')
    )
    dict_call = {"function": {"name": "read_file", "arguments": '{"path": "y"}'}}
    mixed = _render_tool_calls([ns_call, dict_call])

    assert '[called tool: web_search({"query": "x"})]' in mixed
    assert '[called tool: read_file({"path": "y"})]' in mixed

    # Dict entry with a namespace-shaped nested function also renders.
    hybrid = {"function": SimpleNamespace(name="terminal", arguments=None)}
    assert _render_tool_calls([hybrid]) == "[called tool: terminal]"

    # Degenerate shapes still fall back safely.
    assert _render_tool_calls([SimpleNamespace()]) == "[called tool: tool]"
