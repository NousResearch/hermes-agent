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

    _slot_runtime forwards the resolved base_url/api_key/api_mode; the single
    chokepoint that must NOT collapse openai-codex to provider=custom is
    _resolve_task_provider_model (via _preserve_provider_with_base_url). If it
    collapsed, the Codex auxiliary branch — Cloudflare headers + Responses
    adapter for chatgpt.com/backend-api/codex — would be bypassed.
    """
    from agent import moa_loop
    from agent.auxiliary_client import _resolve_task_provider_model

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
    # _slot_runtime forwards the resolved endpoint unconditionally now.
    assert rt["provider"] == "openai-codex"
    assert rt["model"] == "gpt-5.5"
    assert rt["base_url"] == "https://chatgpt.com/backend-api/codex"

    # The chokepoint preserves openai-codex identity despite the explicit
    # base_url (api_mode is forwarded to call_llm directly, not the resolver).
    resolver_kwargs = {k: v for k, v in rt.items() if k != "api_mode"}
    resolved_provider, _model, base_url, _api_key, _mode = _resolve_task_provider_model(
        task="moa_reference",
        **resolver_kwargs,
    )
    assert resolved_provider == "openai-codex"
    assert base_url == "https://chatgpt.com/backend-api/codex"


@pytest.mark.parametrize("provider", ["minimax-oauth", "qwen-oauth"])
def test_moa_provider_backed_slot_survives_aux_resolution(monkeypatch, provider):
    """MoA can pass resolved endpoints for provider-backed slots without
    call_llm flattening them to generic custom endpoints.

    ``_slot_runtime`` resolves a provider-backed slot to ``provider`` plus a
    concrete ``base_url``/``api_key``/``api_mode``; ``_run_reference`` then
    forwards that dict to ``call_llm``. ``call_llm`` resolves the routing tuple
    via ``_resolve_task_provider_model`` (which takes everything except
    ``api_mode``, handled separately). The provider identity must survive that
    resolution rather than being flattened to ``custom``.

    NOTE: providers in the ``_slot_runtime`` name-preservation set (anthropic,
    bedrock, nous, openai-codex, xai-oauth) are intentionally NOT forwarded —
    they're covered by their own dedicated tests. This case covers the
    forward-the-resolved-endpoint path for providers that are NOT in the set.
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


def test_moa_copilot_reference_forwards_user_initiator_header(monkeypatch):
    """Copilot MoA advisors must carry the same user-turn attribution as main calls.

    Copilot Pro/Pro+ gates some premium chat models on the ``x-initiator``
    request header. MoA references are direct fan-out for the user's current
    turn, so Copilot advisors need ``x-initiator: user`` rather than inheriting
    the Copilot language-server default attribution.
    """
    from agent import moa_loop

    calls = []

    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda _slot: {
            "provider": "copilot",
            "model": "claude-sonnet-4.6",
            "api_mode": "chat_completions",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "copilot-token",
        },
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("copilot advice")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)

    _label, text, _acct = moa_loop._run_reference(
        {"provider": "copilot", "model": "claude-sonnet-4.6"},
        [{"role": "user", "content": "solve this"}],
    )

    assert text == "copilot advice"
    assert calls[0]["task"] == "moa_reference"
    assert calls[0]["extra_headers"] == {"x-initiator": "user"}


def test_moa_non_copilot_reference_does_not_forward_initiator_header(monkeypatch):
    """The Copilot attribution header must stay scoped to Copilot advisors."""
    from agent import moa_loop

    calls = []

    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda _slot: {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4.6",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "openrouter-token",
        },
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("openrouter advice")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)

    _label, text, _acct = moa_loop._run_reference(
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
        [{"role": "user", "content": "solve this"}],
    )

    assert text == "openrouter advice"
    assert calls[0]["task"] == "moa_reference"
    assert calls[0]["extra_headers"] is None


@pytest.mark.parametrize(
    "provider_spelling",
    ["copilot", "github-copilot", "github", "github-models", "Copilot", "copilot-acp"],
)
def test_moa_copilot_alias_spellings_forward_initiator_header(
    monkeypatch, provider_spelling
):
    """Every Copilot alias spelling must trigger the x-initiator header.

    Slot configs spell the provider inconsistently (github, github-copilot,
    github-models, copilot-acp, mixed case); the header gate goes through the
    auxiliary client's canonical alias normalization so all of them get the
    user-turn attribution, not just the literal string "copilot".
    """
    from agent import moa_loop

    calls = []

    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda _slot: {
            "provider": provider_spelling,
            "model": "claude-sonnet-4.6",
            "api_mode": "chat_completions",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "copilot-token",
        },
    )

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("copilot advice")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)

    _label, text, _acct = moa_loop._run_reference(
        {"provider": provider_spelling, "model": "claude-sonnet-4.6"},
        [{"role": "user", "content": "solve this"}],
    )

    assert text == "copilot advice"
    assert calls[0]["extra_headers"] == {"x-initiator": "user"}


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


def test_reference_messages_drops_empty_user_turns():
    """Empty user turns must not leak into the advisory view.

    A user message whose content is "" or a non-string/multimodal payload
    (flattened to "" by the text-extraction step) carries nothing advisory.
    Strict providers (Kimi/Moonshot and others that enforce non-empty user
    content) reject such a message with
    400 "message ... with role 'user' must not be empty", while lenient
    providers (DeepSeek) accept it — so a fan-out over the identical rendered
    view fails on one reference and passes on another. The renderer must emit
    NO empty user turn, mirroring how empty assistant turns are dropped.
    """
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": '{"path":"c.yaml"}'}}
        ]},
        {"role": "tool", "content": "some result"},
        {"role": "user", "content": ""},  # empty string user turn
        {"role": "user", "content": [{"type": "text", "text": "multimodal"}]},  # non-string -> ""
    ]

    view = _reference_messages(messages)

    # No user turn in the view may be empty/whitespace-only.
    empty_users = [
        m for m in view
        if m.get("role") == "user" and not str(m.get("content", "")).strip()
    ]
    assert empty_users == [], f"empty user turn leaked into advisory view: {empty_users}"
    # The real user prompt survives and the view still ends on a user turn.
    assert view[0] == {"role": "user", "content": "real question"}
    assert view[-1]["role"] == "user"


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
    # Threshold sits at 0.95s (not tight against 0.5s) to tolerate CI
    # thread-pool startup jitter while still failing hard if the two calls
    # ran serially (which would be ≥1.0s).
    assert elapsed < 0.95, f"references did not run in parallel (took {elapsed:.2f}s)"
    # Output order matches input order (stable Reference N labelling).
    assert [label for label, _, _ in out] == ["p1:ok", "moa:preset", "p2:boom", "p3:ok"]
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


def test_references_parallel_sum_and_consume(monkeypatch, tmp_path):
    """create() sums advisor usage + cost once per turn; consume clears it.

    Repeat tool-iterations within a turn reuse the cache and contribute ZERO
    additional advisor spend (otherwise advisor cost multiplies by iteration
    count).
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
        - provider: openrouter
          model: adv-a
        - provider: openrouter
          model: adv-b
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            return _response_with_usage(prompt=1000, completion=100, cached=0)
        return _response("aggregator acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    monkeypatch.setattr(
        "agent.moa_loop._slot_runtime",
        lambda slot: {"provider": "openrouter", "model": slot.get("model")},
    )
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *a, **k: SimpleNamespace(amount_usd=0.01, status="estimated", source="table"),
    )

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "turn one"}], tools=[])

    usage, cost = facade.consume_reference_usage()
    # Two advisors × (1000 input, 100 output) = 2000 input, 200 output.
    assert usage.input_tokens == 2000
    assert usage.output_tokens == 200
    # Two advisors × $0.01 each = $0.02.
    assert cost == pytest.approx(0.02)

    # consume clears — a second consume with no new create() is zeroed.
    usage2, cost2 = facade.consume_reference_usage()
    assert usage2.input_tokens == 0
    assert cost2 is None

    # A repeat create() with the SAME advisory view is a cache HIT: advisors
    # do not re-run, so pending advisor spend is zero (no double-charge).
    facade.create(messages=[{"role": "user", "content": "turn one"}], tools=[])
    usage3, cost3 = facade.consume_reference_usage()
    assert usage3.input_tokens == 0
    assert cost3 is None


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


def test_moa_full_trace_written_when_enabled(monkeypatch, tmp_path):
    """With moa.save_traces on, a full MoA turn is written to JSONL.

    Asserts the record captures each reference's FULL input messages + output
    and the aggregator's FULL input (incl. injected reference guidance) +
    output — the true full turn, auditable offline.
    """
    import json

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  save_traces: true
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openrouter
          model: adv-a
        - provider: openrouter
          model: adv-b
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            # Echo the model so we can prove per-reference output is captured.
            model = kwargs.get("model", "?")
            return _response_with_usage(content=f"advice from {model}", prompt=500, completion=80)
        return _response("AGGREGATOR FINAL ANSWER")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    monkeypatch.setattr(
        "agent.moa_loop._slot_runtime",
        lambda slot: {"provider": "openrouter", "model": slot.get("model")},
    )
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *a, **k: SimpleNamespace(amount_usd=0.001, status="estimated", source="table"),
    )

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    # Non-streaming create() → aggregator output captured inline.
    facade.create(messages=[{"role": "user", "content": "please review the plan"}], tools=[])
    facade.consume_and_save_trace(session_id="sess-xyz")

    trace_file = home / "moa-traces" / "sess-xyz.jsonl"
    assert trace_file.exists(), "trace file not written"
    lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])

    # Turn framing.
    assert rec["session_id"] == "sess-xyz"
    assert rec["preset"] == "review"

    # Both references captured, each with FULL input messages + output.
    assert len(rec["references"]) == 2
    for ref in rec["references"]:
        assert ref["model"] in ("adv-a", "adv-b")
        assert ref["provider"] == "openrouter"
        # Full input messages present (system advisory prompt + advisory view).
        assert isinstance(ref["input_messages"], list) and len(ref["input_messages"]) >= 2
        assert ref["input_messages"][0]["role"] == "system"
        # Full output present and model-specific.
        assert ref["output"] == f"advice from {ref['model']}"
        assert ref["usage"]["input_tokens"] == 500
        assert ref["cost_usd"] == 0.001

    # Aggregator: full input (with injected reference guidance) + inline output.
    agg = rec["aggregator"]
    assert agg["model"] == "anthropic/claude-opus-4.8"
    assert agg["streamed"] is False
    assert agg["output"] == "AGGREGATOR FINAL ANSWER"
    agg_text = json.dumps(agg["input_messages"])
    assert "Mixture of Agents reference context" in agg_text
    assert "advice from adv-a" in agg_text and "advice from adv-b" in agg_text


def test_moa_trace_not_written_when_disabled(monkeypatch, tmp_path):
    """Default (save_traces off) writes nothing."""
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
          model: adv-a
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            return _response_with_usage(content="advice")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    monkeypatch.setattr(
        "agent.moa_loop._slot_runtime",
        lambda slot: {"provider": "openrouter", "model": slot.get("model")},
    )

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "hi"}], tools=[])
    facade.consume_and_save_trace(session_id="sess-off")

    assert not (home / "moa-traces").exists()


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


def test_reference_guidance_merges_into_trailing_user_in_plain_chat():
    """Plain chat ends on the user turn, so the block merges there (still at end)."""
    from agent.moa_loop import _attach_reference_guidance

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ]
    _attach_reference_guidance(messages, "REFERENCE BLOCK")

    # No extra message; the block joins the trailing user turn (which is the end).
    assert len(messages) == 2
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "hello\n\nREFERENCE BLOCK"


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


def test_reference_messages_flattens_multimodal_user_turn():
    """Multimodal user turns (text + image parts) keep their text in the view.

    Image parts carry no advisory text and are skipped; the text part must
    survive. Previously the whole turn flattened to "".
    """
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "what is in this screenshot?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ]

    view = _reference_messages(messages)

    assert view == [{"role": "user", "content": "what is in this screenshot?"}]
    # No base64 payload leaks into the advisory view.
    assert all("base64" not in m["content"] for m in view)


def test_reference_messages_image_only_user_turn_gets_placeholder():
    """An image-only user turn must not become an empty user message.

    Anthropic rejects empty text blocks (the original 400 class) and silently
    skipping the turn would misalign user/assistant alternation in the view —
    so a placeholder stands in for the non-text content.
    """
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
        {"role": "assistant", "content": "I see a diagram."},
        {"role": "user", "content": "now explain it"},
    ]

    view = _reference_messages(messages)

    assert view[0]["role"] == "user"
    assert view[0]["content"].strip(), "image-only turn must not be empty"
    assert "non-text" in view[0]["content"]
    assert view[-1] == {"role": "user", "content": "now explain it"}


def test_reference_messages_flattens_structured_assistant_and_tool_content():
    """Assistant and tool turns with content-part lists are flattened too.

    Multimodal tool results (e.g. computer_use screenshots) and adapter-shaped
    assistant turns arrive as lists; their text must reach the references and
    their image parts must not leak.
    """
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "user", "content": "check the screen"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "taking a screenshot"}],
            "tool_calls": [{"id": "c1", "function": {"name": "capture", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": [
            {"type": "text", "text": "screenshot captured: login page visible"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
        ]},
    ]

    view = _reference_messages(messages)

    joined = "\n".join(m["content"] for m in view)
    assert "taking a screenshot" in joined
    assert "[called tool: capture(" in joined
    assert "[tool result: screenshot captured: login page visible]" in joined
    assert "BBBB" not in joined
    assert view[-1]["role"] == "user"


def test_reference_guidance_appends_text_part_to_decorated_trailing_user():
    """A cache-decorated trailing user turn still receives the guidance block.

    Decoration converts the trailing user turn to a content-part list; the
    guidance must be appended as a NEW text part AFTER the cache_control-marked
    part (cached prefix stays byte-stable, no consecutive-user-turn 400s), not
    silently dropped and not added as a second user message.
    """
    from agent.moa_loop import _attach_reference_guidance

    marked_part = {
        "type": "text",
        "text": "hello",
        "cache_control": {"type": "ephemeral"},
    }
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [dict(marked_part)]},
    ]
    _attach_reference_guidance(messages, "REFERENCE BLOCK")

    # No extra message (would break user/user alternation).
    assert len(messages) == 2
    content = messages[-1]["content"]
    assert isinstance(content, list) and len(content) == 2
    # The cache-marked part is byte-identical (prefix stability).
    assert content[0] == marked_part
    # The guidance rides as a trailing text part outside the cached span.
    assert content[1] == {"type": "text", "text": "\n\nREFERENCE BLOCK"}


def test_reference_messages_drops_whitespace_only_string_user_turn():
    """A whitespace-only STRING user turn is dropped, not placeholdered.

    The non-text placeholder exists for structured content (image-only turns)
    where a real turn happened that the reference should know about. A bare
    whitespace string carries nothing — emitting it would 400 strict
    providers (Kimi/Moonshot 'role user must not be empty'), and
    placeholdering it would fabricate an attachment that never existed.
    """
    from agent.moa_loop import _reference_messages

    messages = [
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "real"},
    ]

    view = _reference_messages(messages)

    assert view[0] == {"role": "assistant", "content": "a"}
    assert view[-1] == {"role": "user", "content": "real"}
    assert all(str(m["content"]).strip() for m in view)

def test_moa_pre_api_compression_includes_reference_guidance(monkeypatch, tmp_path):
    """The aggregator must not receive guidance that pushes it past compression.

    The normal pre-API check sees only the persisted conversation.  MoA adds
    reference guidance later, inside ``MoAChatCompletions.create()``, so this
    regression drives a raw request just below the threshold and makes the
    injected guidance cross it.  Compression must occur before the aggregator
    request and leave the rebuilt request below the threshold.
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
        - provider: openrouter
          model: advisor
      aggregator:
        provider: openrouter
        model: aggregator
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    events = []
    compression_inputs = []
    aggregator_request_tokens = []

    def fake_estimate(messages, *args, **kwargs):
        rendered = str(messages)
        raw_tokens = 80 if "PRE_COMPACTION_HISTORY" in rendered else 20
        guidance_tokens = 40 if "Mixture of Agents reference context" in rendered else 0
        return raw_tokens + guidance_tokens

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            events.append("reference")
            return _response("advisor guidance")
        events.append("aggregator")
        aggregator_request_tokens.append(fake_estimate(kwargs["messages"]))
        return _response("aggregator acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    monkeypatch.setattr("agent.turn_context.estimate_request_tokens_rough", fake_estimate)
    monkeypatch.setattr("agent.conversation_loop.estimate_request_tokens_rough", fake_estimate)

    agent = AIAgent(
        api_key="moa-virtual-provider",
        base_url="moa://local",
        model="review",
        provider="moa",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["file"],
        max_iterations=3,
    )
    compressor = getattr(agent, "context_compressor")
    compressor.threshold_tokens = 100

    def fake_compress(messages, *_args, **_kwargs):
        events.append("compress")
        compression_inputs.append(messages)
        return ([{"role": "user", "content": "SUMMARY"}], "system")

    monkeypatch.setattr(agent, "_compress_context", fake_compress)

    result = agent.run_conversation(
        "PRE_COMPACTION_HISTORY",
        conversation_history=[{"role": "assistant", "content": "prior response"}],
    )

    assert result["final_response"] == "aggregator acted"
    assert events.index("compress") < events.index("aggregator")
    assert events.count("reference") == 1
    assert all("Mixture of Agents reference context" not in str(item) for item in compression_inputs)
    assert aggregator_request_tokens == [60]


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
