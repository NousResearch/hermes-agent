from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_moa_does_not_cap_output_tokens(monkeypatch, tmp_path):
    """MoA must not inject an output cap on reference or aggregator calls.

    The preset's old hardcoded max_tokens=4096 truncated long aggregator
    syntheses. MoA now passes max_tokens=None (no caller cap), so call_llm
    omits the parameter and each model uses its real maximum. Regression for
    the "no limit on MoA models" fix.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      max_tokens: 4096
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
        base_url="moa://local",
        model="review",
        provider="moa",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["file"],
        max_iterations=1,
    )
    agent.run_conversation("solve this")

    # Even with a preset max_tokens: 4096 present in config, neither the
    # reference nor the aggregator call carries a cap — MoA passes None and
    # call_llm omits the parameter so the model uses its full output budget.
    ref_call = next(c for c in calls if c["task"] == "moa_reference")
    agg_call = next(c for c in calls if c["task"] == "moa_aggregator")
    assert ref_call.get("max_tokens") is None
    assert agg_call.get("max_tokens") is None


def test_moa_slots_routed_through_resolve_runtime_provider(monkeypatch):
    """Reference + aggregator slots must be called via their provider's real
    runtime (resolve_runtime_provider), not a bare provider/model call.

    This is the "call any model the way it's called elsewhere" contract: each
    slot's resolved base_url/api_key is passed through to call_llm so the
    provider's actual API surface (anthropic_messages, max_completion_tokens,
    custom endpoints) applies — same as if the model were the acting model.
    """
    from agent import moa_loop

    resolved = []

    def fake_resolve(*, requested, target_model=None):
        resolved.append((requested, target_model))
        return {
            "provider": requested,
            "api_mode": "chat_completions",
            "base_url": f"https://{requested}.example/v1",
            "api_key": f"key-for-{requested}",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )

    rt = moa_loop._slot_runtime({"provider": "minimax", "model": "MiniMax-M2"})
    assert ("minimax", "MiniMax-M2") in resolved
    assert rt["provider"] == "minimax"
    assert rt["model"] == "MiniMax-M2"
    assert rt["base_url"] == "https://minimax.example/v1"
    assert rt["api_key"] == "key-for-minimax"


def test_moa_codex_slot_preserves_provider_identity(monkeypatch):
    """Codex slots must not become custom chat-completions endpoints.

    _resolve_task_provider_model treats any explicit base_url as provider=custom.
    For openai-codex that bypasses the Codex auxiliary branch, losing the
    Cloudflare headers and Responses adapter required for chatgpt.com/backend-api/codex.
    """
    from agent import moa_loop

    def fake_resolve(*, requested, target_model=None):
        return {
            "provider": requested,
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-oauth-token",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )

    rt = moa_loop._slot_runtime({"provider": "openai-codex", "model": "gpt-5.5"})

    assert rt == {"provider": "openai-codex", "model": "gpt-5.5"}


def test_moa_slot_runtime_falls_back_on_resolution_error(monkeypatch):
    """A slot whose provider can't be resolved still attempts the call with the
    bare provider/model rather than aborting the whole MoA turn."""
    from agent import moa_loop

    def boom(*, requested, target_model=None):
        raise RuntimeError("unknown provider")

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", boom
    )

    rt = moa_loop._slot_runtime({"provider": "mystery", "model": "x"})
    assert rt == {"provider": "mystery", "model": "x"}
    assert "base_url" not in rt
    assert "api_key" not in rt


def test_reference_messages_strips_system_and_tool_history():
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

    trimmed = _reference_messages(messages)

    # System prompt, tool-call-only assistant turn, and tool result are gone.
    assert all(m["role"] in ("user", "assistant") for m in trimmed)
    assert all("tool_calls" not in m for m in trimmed)
    # The advisory view must end on a user turn — a trailing assistant turn is
    # treated by Anthropic as an assistant prefill (400 on no-prefill models).
    # The only kept user turn here is the prompt, so the trailing assistant
    # answer is stripped.
    assert trimmed == [
        {"role": "user", "content": "do the thing"},
    ]


def test_reference_messages_ends_with_user_not_assistant_prefill():
    """Advisory reference views must never end on an assistant turn.

    Mid-tool-loop, the last assistant turn carries interleaved reasoning text
    plus tool calls; its text survives the trim while the following tool result
    is dropped, leaving a trailing assistant turn. Anthropic (and
    OpenRouter→Anthropic) treat that as an assistant prefill the model should
    continue, and no-prefill models (e.g. Claude Opus 4.8) reject it with
    ``400 ... must end with a user message``. The trim must drop trailing
    assistant turns while preserving intervening ones.
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
        {"role": "tool", "tool_call_id": "c1", "content": "tool result"},
    ]

    trimmed = _reference_messages(messages)

    assert trimmed, "advisory view should not be empty"
    assert trimmed[-1]["role"] == "user"
    # Intervening assistant context is preserved; only the trailing one drops.
    assert trimmed == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2 current"},
    ]


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

    label, text = _run_reference(
        {"provider": "openai-codex", "model": "gpt-5.5"},
        [{"role": "user", "content": "review this PR"}],
    )

    assert text == "advice"
    msgs = captured["messages"]
    assert msgs[0] == {"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}
    assert msgs[-1]["role"] == "user"


def test_moa_facade_references_get_trimmed_messages(monkeypatch, tmp_path):
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
        return _response("ok")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    facade.create(
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "question"},
            {"role": "tool", "tool_call_id": "x", "content": "leftover"},
        ],
        tools=[{"type": "function"}],
    )

    ref_call = next(c for c in calls if c["task"] == "moa_reference")
    # Reference gets the advisory-role system prompt first, then user turns
    # only — never the agent's own system prompt or tool-role messages.
    ref_msgs = ref_call["messages"]
    assert ref_msgs[0]["role"] == "system"
    assert "reference advisor" in ref_msgs[0]["content"].lower()
    assert "huge hermes system prompt" not in ref_msgs[0]["content"]
    assert all(m["role"] == "user" for m in ref_msgs[1:])
    assert ref_call.get("tools") in (None, [])
    # Aggregator still receives the original messages + tool schema.
    agg_call = next(c for c in calls if c["task"] == "moa_aggregator")
    assert agg_call["tools"] is not None


def test_moa_disabled_preset_skips_references(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      enabled: false
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
        return _response("aggregator only")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "question"}], tools=[{"type": "function"}])

    tasks = [c["task"] for c in calls]
    # No reference fan-out — only the aggregator runs.
    assert tasks == ["moa_aggregator"]
    # Aggregator gets the unmodified user message (no MoA guidance appended).
    agg_call = calls[0]
    assert agg_call["messages"][-1]["content"] == "question"


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
    assert elapsed < 0.9, f"references did not run in parallel (took {elapsed:.2f}s)"
    # Output order matches input order (stable Reference N labelling).
    assert [label for label, _ in out] == ["p1:ok", "moa:preset", "p2:boom", "p3:ok"]
    assert "recursively reference MoA" in out[1][1]
    assert out[2][1].startswith("[failed:")
    assert out[0][1] == "resp-p1"


def _ref_config(home):
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
        - provider: openrouter
          model: anthropic/claude-opus-4.8
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )


def test_moa_facade_emits_reference_then_aggregating(monkeypatch, tmp_path):
    """The facade reports each reference's output, then an aggregating signal,
    so frontends can render reference blocks before the aggregator acts."""
    home = tmp_path / ".hermes"
    _ref_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            return _response(f"advice from {kwargs['model']}")
        return _response("aggregator acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from agent.moa_loop import MoAChatCompletions

    events = []
    facade = MoAChatCompletions("review", reference_callback=lambda ev, **kw: events.append((ev, kw)))
    facade.create(messages=[{"role": "user", "content": "q"}], tools=[{"type": "function"}])

    ref_events = [e for e in events if e[0] == "moa.reference"]
    agg_events = [e for e in events if e[0] == "moa.aggregating"]
    # One block per reference model, labelled by source, with index/count.
    assert len(ref_events) == 2
    assert ref_events[0][1]["label"] == "openai-codex:gpt-5.5"
    assert ref_events[0][1]["index"] == 1 and ref_events[0][1]["count"] == 2
    assert "advice from" in ref_events[0][1]["text"]
    # Exactly one aggregating signal, after the references, naming the aggregator.
    assert len(agg_events) == 1
    assert agg_events[0][1]["aggregator"] == "openrouter:anthropic/claude-opus-4.8"
    assert agg_events[0][1]["ref_count"] == 2


def test_moa_facade_caches_references_within_a_turn(monkeypatch, tmp_path):
    """References run + emit ONCE per user turn, not per tool-loop iteration.

    The agent loop calls create() once per iteration; the advisory message
    view is identical across iterations (tool/tool_call turns are stripped),
    so re-running references would multiply their cost and re-spam the display.
    """
    home = tmp_path / ".hermes"
    _ref_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            ref_runs.append(kwargs["model"])
            return _response("advice")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from agent.moa_loop import MoAChatCompletions

    events = []
    facade = MoAChatCompletions("review", reference_callback=lambda ev, **kw: events.append(ev))

    base_msgs = [{"role": "user", "content": "do the thing"}]
    # Iteration 1: model emits a tool call.
    facade.create(messages=base_msgs, tools=[{"type": "function"}])
    # Iteration 2: same turn — a tool result was appended, but the advisory
    # view (which strips tool turns) is unchanged, so references must be reused.
    facade.create(
        messages=base_msgs
        + [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ],
        tools=[{"type": "function"}],
    )

    # 2 reference models, run once total (not once per iteration).
    assert len(ref_runs) == 2
    # Reference blocks emitted once (2 reference events + 1 aggregating).
    assert events.count("moa.reference") == 2
    assert events.count("moa.aggregating") == 1


def test_moa_facade_reruns_references_on_new_turn(monkeypatch, tmp_path):
    """A genuinely new user message invalidates the cache and re-runs refs."""
    home = tmp_path / ".hermes"
    _ref_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            ref_runs.append(kwargs["model"])
            return _response("advice")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "turn one"}], tools=[])
    facade.create(messages=[{"role": "user", "content": "turn two"}], tools=[])

    # 2 references × 2 distinct turns = 4 reference runs.
    assert len(ref_runs) == 4
