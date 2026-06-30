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


def test_moa_copilot_slot_resolves_api_mode_per_model(monkeypatch):
    """Copilot slots must resolve api_mode from the SLOT's model, not the
    session default.

    Copilot serves GPT-5.x via the Responses API (codex_responses) and Claude
    via chat_completions. ``resolve_runtime_provider`` derives api_mode from the
    session default model, so for a Copilot slot it returns the same mode for
    every model in a preset. ``_slot_runtime`` must override that by resolving
    api_mode from each slot's own model via ``copilot_model_api_mode`` — and
    must keep the provider identified by name (no base_url) so call_llm builds
    the real Copilot request instead of a generic custom endpoint.
    """
    from agent import moa_loop

    # resolve_runtime_provider returns the (wrong, session-default) mode for
    # every Copilot model — the bug this fix works around.
    def fake_resolve(*, requested, target_model=None):
        return {
            "provider": "copilot",
            "api_mode": "chat_completions",  # session default leaks to all slots
            "base_url": "https://api.githubcopilot.com",
            "api_key": "copilot-token",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )

    # Per-model resolver: GPT-5.x → responses, everything else → chat.
    def fake_copilot_mode(model_id, **_kwargs):
        return "codex_responses" if str(model_id).startswith("gpt-5") else "chat_completions"

    monkeypatch.setattr(
        "hermes_cli.models.copilot_model_api_mode", fake_copilot_mode
    )

    gpt = moa_loop._slot_runtime({"provider": "copilot", "model": "gpt-5.5"})
    claude = moa_loop._slot_runtime({"provider": "copilot", "model": "claude-opus-4.8"})

    # api_mode comes from the slot's OWN model, not the shared session default.
    assert gpt["api_mode"] == "codex_responses"
    assert claude["api_mode"] == "chat_completions"
    # Provider stays identified by name; no base_url flattening to custom.
    assert gpt["provider"] == "copilot" and "base_url" not in gpt
    assert claude["provider"] == "copilot" and "base_url" not in claude


def test_moa_mixed_copilot_preset_does_not_cross_contaminate_api_mode(monkeypatch):
    """A mixed Copilot preset (GPT-5.x reference + Claude aggregator) must give
    each model its own api_mode.

    Regression for the client-cache-poisoning failure: because api_mode is part
    of the auxiliary client cache key, a single shared api_mode across Copilot
    slots let the first-built client (e.g. gpt-5.5 → Responses) poison the cache
    for later Claude slots, which then 400'd with "does not support Responses
    API". Resolving api_mode per model keys the cache per mode, so the modes the
    slots request are mode-correct and never cross-contaminate.
    """
    from agent import moa_loop

    def fake_resolve(*, requested, target_model=None):
        return {
            "provider": "copilot",
            "api_mode": "chat_completions",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "copilot-token",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )

    def fake_copilot_mode(model_id, **_kwargs):
        return "codex_responses" if str(model_id).startswith("gpt-5") else "chat_completions"

    monkeypatch.setattr(
        "hermes_cli.models.copilot_model_api_mode", fake_copilot_mode
    )

    preset_models = [
        "gpt-5.5",            # Responses
        "claude-sonnet-4.6",  # chat
        "claude-opus-4.6",    # chat
        "claude-opus-4.8",    # chat (aggregator)
    ]
    modes = {
        m: moa_loop._slot_runtime({"provider": "copilot", "model": m})["api_mode"]
        for m in preset_models
    }

    assert modes["gpt-5.5"] == "codex_responses"
    assert modes["claude-sonnet-4.6"] == "chat_completions"
    assert modes["claude-opus-4.6"] == "chat_completions"
    assert modes["claude-opus-4.8"] == "chat_completions"
    # The GPT-5.x Responses mode must NOT leak onto any Claude slot.
    assert {modes[m] for m in preset_models if m.startswith("claude")} == {"chat_completions"}


def test_moa_copilot_slot_falls_back_when_per_model_resolver_unavailable(monkeypatch):
    """If the per-model Copilot resolver raises, the slot still attempts the
    call using the runtime-resolved api_mode rather than aborting."""
    from agent import moa_loop

    def fake_resolve(*, requested, target_model=None):
        return {
            "provider": "copilot",
            "api_mode": "chat_completions",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "copilot-token",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )

    def boom(model_id, **_kwargs):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(
        "hermes_cli.models.copilot_model_api_mode", boom
    )

    rt = moa_loop._slot_runtime({"provider": "copilot", "model": "gpt-5.5"})
    # Provider identity preserved; falls back to the runtime-resolved api_mode.
    assert rt["provider"] == "copilot"
    assert rt["model"] == "gpt-5.5"
    assert rt["api_mode"] == "chat_completions"
    assert "base_url" not in rt


@pytest.mark.parametrize("provider", ["anthropic", "minimax-oauth", "qwen-oauth"])
def test_moa_provider_backed_slot_survives_aux_resolution(monkeypatch, provider):
    """MoA can pass resolved endpoints for provider-backed slots without
    call_llm flattening them to generic custom endpoints.

    ``_slot_runtime`` resolves a provider-backed slot to ``provider`` plus a
    concrete ``base_url``/``api_key``/``api_mode``; ``_run_reference`` then
    forwards that dict to ``call_llm``. ``call_llm`` resolves the routing tuple
    via ``_resolve_task_provider_model`` (which takes everything except
    ``api_mode``, handled separately). The provider identity must survive that
    resolution rather than being flattened to ``custom``.
    """
    from agent import moa_loop
    from agent.auxiliary_client import _resolve_task_provider_model

    def fake_resolve(*, requested, target_model=None):
        return {
            "provider": requested,
            "api_mode": "anthropic_messages",
            "base_url": f"https://{requested}.example/v1",
            "api_key": f"token-for-{requested}",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )

    rt = moa_loop._slot_runtime({"provider": provider, "model": "test-model"})
    # api_mode is forwarded to call_llm directly, not to _resolve_task_provider_model.
    resolver_kwargs = {k: v for k, v in rt.items() if k != "api_mode"}
    resolved_provider, model, base_url, api_key, _mode = _resolve_task_provider_model(
        task="moa_reference",
        **resolver_kwargs,
    )

    assert resolved_provider == provider
    assert model == "test-model"
    assert base_url == f"https://{provider}.example/v1"
    assert api_key == f"token-for-{provider}"


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


def test_reference_messages_truncates_large_tool_results():
    """Large tool results are previewed head+tail, not replayed verbatim."""
    from agent.moa_loop import _REFERENCE_TOOL_RESULT_BUDGET, _reference_messages

    huge = "A" * (_REFERENCE_TOOL_RESULT_BUDGET * 3)
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": huge},
    ]

    view = _reference_messages(messages)
    joined = "\n".join(m["content"] for m in view)
    assert "chars omitted" in joined
    # The folded result is far smaller than the raw payload.
    assert len(joined) < len(huge)


def test_reference_messages_fresh_user_turn_ends_on_that_user():
    """A fresh user prompt with no agent action yet ends on that user turn."""
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2 current"},
    ]

    view = _reference_messages(messages)
    assert view[-1] == {"role": "user", "content": "q2 current"}


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
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [{"id": "x", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "x", "content": "tool output"},
        ],
        tools=[{"type": "function"}],
    )

    ref_call = next(c for c in calls if c["task"] == "moa_reference")
    ref_msgs = ref_call["messages"]
    # Advisory-role system prompt first; the agent's own system prompt is gone.
    assert ref_msgs[0]["role"] == "system"
    assert "reference advisor" in ref_msgs[0]["content"].lower()
    assert "system prompt" not in ref_msgs[0]["content"]
    # No tool-role messages and no tool_calls arrays leak to the reference.
    assert all(m["role"] in ("system", "user", "assistant") for m in ref_msgs)
    assert all("tool_calls" not in m for m in ref_msgs)
    # The agent's action + tool result ARE preserved, rendered as text.
    joined = "\n".join(m["content"] for m in ref_msgs[1:])
    assert "[called tool: lookup(" in joined
    assert "[tool result: tool output]" in joined
    # Ends on a user turn (advisory request after the final assistant block).
    assert ref_msgs[-1]["role"] == "user"
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


def test_moa_facade_reruns_references_on_new_tool_result(monkeypatch, tmp_path):
    """References re-run when a new tool result advances the task state.

    The agent loop calls create() once per tool-loop iteration. References must
    judge the LATEST state, so a new tool result is a cache MISS and re-runs the
    references — but a redundant create() call with the SAME state is a cache
    HIT (no re-run, no re-emit), so we don't fire on a pure no-op re-call.
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
    # Iteration 1: fresh user turn — references run (2 models).
    facade.create(messages=base_msgs, tools=[{"type": "function"}])
    after_tool = base_msgs + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    # Iteration 2: a NEW tool result advanced the state → references re-run.
    facade.create(messages=after_tool, tools=[{"type": "function"}])
    # Iteration 3: identical state (no new tool/user input) → cache hit, no re-run.
    facade.create(messages=after_tool, tools=[{"type": "function"}])

    # 2 models × 2 distinct states (fresh turn + new tool result) = 4 runs.
    # The redundant 3rd call adds none.
    assert len(ref_runs) == 4
    assert events.count("moa.reference") == 4
    assert events.count("moa.aggregating") == 2


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
