"""A one-turn model override must not become the route other clients read, and a same-provider
override must keep the session endpoint and key."""
from types import SimpleNamespace

from gateway.session_cli_turn import cli_turn_scope, one_turn_runtime
from hermes_cli.model_switch import ModelSwitchResult


def test_settled_info_publishes_the_policy_route_and_drops_the_cached_agent(monkeypatch):
    from gateway.session_local_inventory import publish_settled_info
    published, evicted = [], []
    policy = SimpleNamespace(current_model=lambda _authority: "session-model", provider="openrouter", source="cli")
    live = SimpleNamespace(route="route-1", source=SimpleNamespace())
    authority = SimpleNamespace(profile_id="p", sessions={"s": live})
    turn = SimpleNamespace(
        _ctx=SimpleNamespace(inbound_message_id="adm", session_key="route-1"),
        _approval_owner=(authority, "s", 1),
        _runner=SimpleNamespace(_evict_cached_agent=lambda key: evicted.append(key)),
        _publish_execution=lambda kind, payload: published.append((kind, payload)),
    )
    agent = SimpleNamespace(model="once-model", provider="anthropic", tools=None, _cached_system_prompt="",
                            context_compressor=None)

    from gateway import session_local, session_policy
    monkeypatch.setattr(session_local, "local_session_info", lambda *_a, **_k: {
        "model": "once-model", "provider": "anthropic"})
    monkeypatch.setattr(session_policy, "policy_for_source", lambda *_a, **_k: policy)

    with cli_turn_scope("adm", {"model_override": {"model": "once-model", "provider": "anthropic"}}):
        publish_settled_info(turn, agent)

    assert evicted == ["route-1"]
    assert published and published[0][0] == "session.info"
    assert published[0][1]["model"] == "session-model"
    assert published[0][1]["provider"] == "openrouter"


def test_same_provider_override_keeps_the_session_endpoint_and_key(monkeypatch):
    def _resolve(**_k):
        return ModelSwitchResult(
            success=True, new_model="openai/gpt-4.1", target_provider="openrouter",
            api_key="", base_url="https://openrouter.ai/api/v1", api_mode="chat_completions",
            request_overrides={})

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _resolve)
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda _config: [])
    pool = object()
    model, runtime = one_turn_runtime(
        {"model": "openai/gpt-4.1", "provider": "openrouter"},
        "session-model",
        {"provider": "openrouter", "api_key": "sk-session", "base_url": "https://mirror.example/api/v1",
         "credential_pool": pool, "request_overrides": {"provider": {"only": ["anthropic"]}},
         "api_mode": "chat_completions"},
        {})
    assert model == "openai/gpt-4.1"
    assert runtime["base_url"] == "https://mirror.example/api/v1"
    assert runtime["api_key"] == "sk-session"
    assert runtime["credential_pool"] is pool
    assert runtime["request_overrides"] == {"provider": {"only": ["anthropic"]}}


def _same_provider_override(monkeypatch, provider, model, api_mode, base_url, session):
    def _resolve(**_k):
        return ModelSwitchResult(success=True, new_model=model, target_provider=provider, api_key="",
                                 base_url=base_url, api_mode=api_mode, request_overrides={})

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _resolve)
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda _config: [])
    return one_turn_runtime({"model": model, "provider": provider}, "session-model",
                            {"provider": provider, "api_key": "tok-session", **session}, {})[1]


def test_same_provider_override_runs_on_the_targets_wire(monkeypatch):
    """Copilot serves GPT-5 on the Responses API and Claude on chat completions."""
    runtime = _same_provider_override(
        monkeypatch, "copilot", "gpt-5.4", "codex_responses", "https://api.githubcopilot.com",
        {"base_url": "https://api.githubcopilot.com", "api_mode": "chat_completions"})
    assert runtime["api_mode"] == "codex_responses"
    assert (runtime["base_url"], runtime["api_key"]) == ("https://api.githubcopilot.com", "tok-session")


def test_same_provider_override_reshapes_the_opencode_endpoint_for_its_wire(monkeypatch):
    runtime = _same_provider_override(
        monkeypatch, "opencode-zen", "minimax-m2.7", "anthropic_messages", "https://opencode.ai/zen",
        {"base_url": "https://opencode.ai/zen/v1", "api_mode": "chat_completions"})
    assert (runtime["api_mode"], runtime["base_url"]) == ("anthropic_messages", "https://opencode.ai/zen")
