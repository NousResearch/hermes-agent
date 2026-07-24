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


def test_moa_gemini_aggregator_sanitize_uses_real_model(monkeypatch, tmp_path):
    """MoA turns must sanitize tool_calls against the AGGREGATOR model, not the preset.

    Regression for #66212 / #65092: under MoA, ``agent.model`` holds the
    virtual preset name (e.g. "review"), so passing it to
    _sanitize_tool_calls_for_strict_api makes
    _model_consumes_thought_signature() return False and strips
    ``extra_content`` (Gemini thought_signature) from replayed tool_calls —
    the Gemini aggregator then 400s with "Function call is missing a
    thought_signature in functionCall parts."
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
        provider: gemini
        model: gemini-3-pro-preview
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    sanitize_models = []

    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path": "x"}'),
    )

    responses = iter(
        [
            _response(None, tool_calls=[tool_call]),
            _response("aggregator done"),
        ]
    )

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            return _response("reference advice")
        return next(responses)

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
        max_iterations=3,
    )

    real_sanitize = type(agent)._sanitize_tool_calls_for_strict_api

    def spy_sanitize(api_msg, model=None):
        sanitize_models.append(model)
        return real_sanitize(api_msg, model=model)

    monkeypatch.setattr(
        type(agent), "_sanitize_tool_calls_for_strict_api", staticmethod(spy_sanitize)
    )
    monkeypatch.setattr(
        agent, "execute_tool", lambda *_a, **_k: "file contents", raising=False
    )

    result = agent.run_conversation("read the file")

    assert result["final_response"] == "aggregator done"
    # Once the history contains an assistant tool_call turn, the sanitize
    # pass must be asked about the REAL aggregator model — never the virtual
    # preset name (which would strip Gemini's thought_signature). The very
    # first API call may still see the preset (the facade hasn't resolved a
    # slot yet), but no tool_calls exist in history at that point.
    assert any(m == "gemini-3-pro-preview" for m in sanitize_models), sanitize_models
    first_resolved = sanitize_models.index("gemini-3-pro-preview")
    assert all(
        m == "gemini-3-pro-preview" for m in sanitize_models[first_resolved:]
    ), sanitize_models


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


def test_moa_disabled_reference_is_not_called(monkeypatch, tmp_path):
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
          enabled: false
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
          enabled: true
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
            return _response(f"reference from {kwargs['provider']}:{kwargs['model']}")
        return _response("aggregator acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "question"}], tools=[{"type": "function"}])

    reference_calls = [c for c in calls if c["task"] == "moa_reference"]
    assert [(c["provider"], c["model"]) for c in reference_calls] == [
        ("openrouter", "deepseek/deepseek-v4-pro")
    ]
    assert calls[-1]["task"] == "moa_aggregator"


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

    Pins fanout: per_iteration explicitly (the default became user_turn,
    #67199). In this mode the agent loop calls create() once per tool-loop
    iteration and references must judge the LATEST state, so a new tool
    result is a cache MISS and re-runs the references — but a redundant
    create() call with the SAME state is a cache HIT (no re-run, no
    re-emit), so we don't fire on a pure no-op re-call.
    """
    home = tmp_path / ".hermes"
    _ref_config(home, fanout="per_iteration")
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



def test_reference_filtering_preserves_accounting_triples():
    from agent.moa_loop import (
        _RefAccounting,
        _failed_reference_labels,
        _successful_references,
    )
    from agent.usage_pricing import CanonicalUsage

    good_accounting = _RefAccounting(CanonicalUsage(input_tokens=7), 0.07)
    failed_accounting = _RefAccounting(CanonicalUsage(input_tokens=5), 0.05)
    outputs = [
        ("good-model", "useful advice", good_accounting),
        ("bad-model", "[failed: raw provider secret]", failed_accounting),
    ]

    successful = _successful_references(outputs)
    assert successful == [outputs[0]]
    assert successful[0][2] is good_accounting
    assert _failed_reference_labels(outputs) == ["bad-model"]


def test_reference_filtering_excludes_recursion_guard_skips():
    """[skipped: …] recursion-guard notes are internal sentinels, not advice —
    they must be filtered out of the aggregator prompt like [failed: …]."""
    from agent.moa_loop import (
        _RefAccounting,
        _failed_reference_labels,
        _is_failed_reference,
        _successful_references,
    )
    from agent.usage_pricing import CanonicalUsage

    outputs = [
        ("good-model", "useful advice", _RefAccounting(CanonicalUsage())),
        (
            "moa:nested",
            "[skipped: MoA presets cannot recursively reference MoA]",
            _RefAccounting(CanonicalUsage()),
        ),
    ]

    assert _is_failed_reference(outputs[1][1])
    assert _successful_references(outputs) == [outputs[0]]
    assert _failed_reference_labels(outputs) == ["moa:nested"]


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


def test_moa_facade_sanitizes_failures_without_breaking_accounting(monkeypatch, tmp_path):
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
      reference_timeout: 19
      degraded_reference_policy: silent
      reference_models:
        - provider: openrouter
          model: good-model
        - provider: openrouter
          model: bad-model
      aggregator:
        provider: openrouter
        model: aggregator
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    outputs = [
        (
            "good-model",
            "useful advice",
            moa_loop._RefAccounting(CanonicalUsage(input_tokens=7), 0.07),
        ),
        (
            "bad-model",
            "[failed: HTTP 401 key=super-secret]",
            moa_loop._RefAccounting(CanonicalUsage(input_tokens=5), 0.05),
        ),
    ]
    fanout_kwargs = {}
    aggregator_calls = []

    def fake_fanout(*args, **kwargs):
        fanout_kwargs.update(kwargs)
        return outputs

    def fake_call_llm(**kwargs):
        aggregator_calls.append(kwargs)
        return _response("aggregator acted")

    monkeypatch.setattr(moa_loop, "_run_references_parallel", fake_fanout)
    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {"provider": slot["provider"], "model": slot["model"]},
    )

    facade = moa_loop.MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "review this"}], tools=[])

    assert fanout_kwargs["reference_timeout"] == 19.0
    private_prompt = str(aggregator_calls[0]["messages"])
    assert "useful advice" in private_prompt
    assert "super-secret" not in private_prompt
    assert "Reference models unavailable" not in private_prompt
    usage, cost = facade.consume_reference_usage()
    assert usage.input_tokens == 12
    assert cost == pytest.approx(0.12)
    assert facade._pending_trace["reference_outputs"] == outputs


def test_run_reference_forwards_configured_timeout(monkeypatch):
    from agent import moa_loop

    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("advice")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {"provider": "openrouter", "model": slot["model"]},
    )
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *args, **kwargs: SimpleNamespace(
            amount_usd=None, status="unavailable", source=None
        ),
    )

    label, text, accounting = moa_loop._run_reference(
        {"provider": "openrouter", "model": "advisor"},
        [{"role": "user", "content": "review"}],
        reference_timeout=23.5,
    )

    assert (label, text) == ("openrouter:advisor", "advice")
    assert calls[0]["timeout"] == 23.5
    assert isinstance(accounting, moa_loop._RefAccounting)


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


def test_aggregate_skips_aggregator_when_all_references_skipped(monkeypatch):
    """References that are skipped (MoA recursion guard) also trigger the early return."""
    from agent.moa_loop import aggregate_moa_context

    def fake_call_llm(**kwargs):
        raise AssertionError("aggregator should not be called when all references are skipped")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    # Both reference models are "moa" — they hit the recursion guard and are
    # returned as "[skipped: …]" without calling call_llm at all.
    result = aggregate_moa_context(
        user_prompt="do something",
        api_messages=[{"role": "user", "content": "do something"}],
        reference_models=[
            {"provider": "moa", "model": "preset-a"},
            {"provider": "moa", "model": "preset-b"},
        ],
        aggregator={"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    )

    assert "all reference models failed" in result
    assert "Reference models unavailable" in result


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


def test_moa_facade_acts_aggregator_alone_when_all_references_fail_loud(
    monkeypatch, tmp_path
):
    """Facade path (MoAChatCompletions.create): when every reference fails,
    the aggregator acts alone — no 'use the reference responses below'
    guidance wrapping a wall of failure sentinels. Under the loud policy the
    sanitized unavailability notice is still disclosed."""
    moa_loop, outputs, aggregator_calls = _facade_all_failed_fixture(
        monkeypatch, tmp_path, "loud"
    )

    facade = moa_loop.MoAChatCompletions("review")
    response = facade.create(
        messages=[{"role": "user", "content": "review this"}], tools=[]
    )

    # The aggregator still acted (it IS the acting model)…
    assert len(aggregator_calls) == 1
    prompt = str(aggregator_calls[0]["messages"])
    # …but got no failure sentinels or raw provider error text…
    assert "[failed:" not in prompt
    assert "super-secret" not in prompt
    assert "Use the reference responses below" not in prompt
    # …only the sanitized loud-policy notice.
    assert "Reference models unavailable" in prompt
    assert "bad-model-a" in prompt
    # Accounting for the failed fan-out is still folded into the turn.
    usage, cost = facade.consume_reference_usage()
    assert usage.input_tokens == 8
    assert cost == pytest.approx(0.08)
    assert response.choices[0].message.content == "aggregator acted alone"


def test_moa_facade_acts_aggregator_alone_when_all_references_fail_silent(
    monkeypatch, tmp_path
):
    """Silent policy: all-failed turns attach no reference guidance at all."""
    moa_loop, _outputs, aggregator_calls = _facade_all_failed_fixture(
        monkeypatch, tmp_path, "silent"
    )

    facade = moa_loop.MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "review this"}], tools=[])

    assert len(aggregator_calls) == 1
    prompt = str(aggregator_calls[0]["messages"])
    assert "[failed:" not in prompt
    assert "Reference models unavailable" not in prompt
    assert "Mixture of Agents reference context" not in prompt


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


def test_reference_trim_untouched_when_within_window(monkeypatch):
    msgs = _advisory_view(2)
    out = _trim(list(msgs), window=10_000_000, monkeypatch=monkeypatch)
    assert out == msgs


def test_reference_trim_drops_oldest_and_keeps_invariants(monkeypatch):
    from agent.model_metadata import estimate_messages_tokens_rough

    msgs = _advisory_view(30)
    out = _trim(list(msgs), window=4000, reserve=100, monkeypatch=monkeypatch)

    # Something was dropped and it fits the (window*0.9 - reserve) budget…
    assert len(out) < len(msgs)
    # System prompt survives at index 0.
    assert out[0]["role"] == "system"
    # User-first after the system prompt — never assistant-first.
    assert out[1]["role"] == "user"
    # Trailing synthetic user turn survives.
    assert out[-1] == msgs[-1]
    # Oldest frames were the ones dropped: the kept body is a contiguous
    # suffix of the original body.
    assert out[1:] == msgs[len(msgs) - len(out) + 1:]


def test_reference_trim_estimates_after_system_prompt(monkeypatch):
    """The advisory system prompt counts against the budget: a view that fits
    without it but not with it must still be trimmed."""
    from agent import model_metadata, moa_loop

    msgs = _advisory_view(6)
    body_tokens = model_metadata.estimate_messages_tokens_rough(msgs[1:])
    total_tokens = model_metadata.estimate_messages_tokens_rough(msgs)
    assert total_tokens > body_tokens

    # Pick a window where (body fits) but (system+body does not):
    # budget = window*0.9 - reserve(100) must sit between the two.
    window = int((body_tokens + (total_tokens - body_tokens) / 2 + 100) / 0.9)
    out = _trim(list(msgs), window=window, reserve=100, monkeypatch=monkeypatch)
    assert len(out) < len(msgs)
    assert out[0]["role"] == "system"


def test_reference_trim_reserves_output_tokens(monkeypatch):
    """With a huge output reserve the budget shrinks and forces a trim that
    a reserve-less estimate would not need."""
    msgs = _advisory_view(10)
    from agent.model_metadata import estimate_messages_tokens_rough

    total = estimate_messages_tokens_rough(msgs)
    window = int((total + 500) / 0.9)  # fits easily with a small reserve
    out_small = _trim(list(msgs), window=window, reserve=100, monkeypatch=monkeypatch)
    assert out_small == msgs
    out_big = _trim(list(msgs), window=window, reserve=total, monkeypatch=monkeypatch)
    assert len(out_big) < len(msgs)


def test_reference_trim_unresolvable_window_is_a_noop(monkeypatch):
    msgs = _advisory_view(3)
    stub = _CountingCtxLen(RuntimeError("metadata down"))
    out = _trim(list(msgs), counting=stub, monkeypatch=monkeypatch)
    assert out == msgs


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


def test_run_reference_trims_oversized_view_before_calling(monkeypatch):
    """End-to-end: _run_reference sends a trimmed request for a small-window
    model instead of letting the provider 400."""
    from agent import model_metadata, moa_loop

    sent = {}

    def fake_call_llm(**kwargs):
        sent.update(kwargs)
        return _response_with_usage("advice")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {"provider": "openrouter", "model": slot["model"]},
    )
    monkeypatch.setattr(
        model_metadata, "get_model_context_length", lambda **k: 3000
    )

    view = _advisory_view(40)[1:]  # advisory view has no system prompt
    label, text, _acct = moa_loop._run_reference(
        {"provider": "openrouter", "model": "small-window"},
        view,
        max_tokens=200,
    )

    assert text == "advice"
    sent_messages = sent["messages"]
    # System prompt was prepended and survived the trim.
    assert sent_messages[0]["role"] == "system"
    # The request actually shrank.
    assert len(sent_messages) < len(view) + 1
    # And ends with the synthetic trailing user turn.
    assert sent_messages[-1]["role"] == "user"


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
