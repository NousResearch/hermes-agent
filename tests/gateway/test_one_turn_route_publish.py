"""A one-turn model override must not become the route other clients read, and a same-provider
override must keep the session endpoint and key."""
from types import SimpleNamespace

import pytest

from gateway.session_cli_turn import cli_turn_scope, one_turn_runtime
from hermes_cli.model_switch import ModelSwitchResult


def test_settled_info_publishes_the_policy_route_and_drops_the_cached_agent(monkeypatch):
    from gateway.session_local_inventory import publish_settled_info
    published, evicted = [], []
    policy = SimpleNamespace(effective=lambda _authority, secrets=True: (
        {"model": {"provider": "openrouter"}}, "session-model"), source="cli")
    live = SimpleNamespace(route="route-1", source=SimpleNamespace())
    authority = SimpleNamespace(profile_id="p", sessions={"s": live})
    turn = SimpleNamespace(
        _ctx=SimpleNamespace(inbound_message_id="adm", session_key="route-1"),
        _approval_owner=(authority, "s", 1),
        _runner=SimpleNamespace(_evict_cached_agent=lambda key, **_k: evicted.append(key)),
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
    """A provider serving GPT-5 on the Responses API and Claude on chat completions (``api_mode_for_model``)."""
    from providers.base import ProviderProfile

    class _PerModelWire(ProviderProfile):
        def api_mode_for_model(self, model, *, api_key=None):
            return "codex_responses" if model.startswith("gpt-5") else "chat_completions"

    acme = _PerModelWire(name="acme-wire", base_url="https://api.acme-wire.test")
    monkeypatch.setattr("providers.get_provider_profile", lambda name: acme if name == "acme-wire" else None)
    runtime = _same_provider_override(
        monkeypatch, "acme-wire", "gpt-5.4", "codex_responses", "https://api.acme-wire.test",
        {"base_url": "https://api.acme-wire.test", "api_mode": "chat_completions"})
    assert runtime["api_mode"] == "codex_responses"
    assert (runtime["base_url"], runtime["api_key"]) == ("https://api.acme-wire.test", "tok-session")


def test_same_provider_override_reshapes_the_opencode_endpoint_for_its_wire(monkeypatch):
    runtime = _same_provider_override(
        monkeypatch, "opencode-zen", "minimax-m2.7", "anthropic_messages", "https://opencode.ai/zen",
        {"base_url": "https://opencode.ai/zen/v1", "api_mode": "chat_completions"})
    assert (runtime["api_mode"], runtime["base_url"]) == ("anthropic_messages", "https://opencode.ai/zen")


def _running_runner(route, agent):
    """A real ``GatewayRunner`` whose cache holds *agent* while its turn still owns the running slot."""
    import threading
    from collections import OrderedDict
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache[route] = (agent, "sig-once", 0, "s")
    runner._running_agents[route] = agent
    runner._persist_active_agents = lambda: None
    return runner


def test_a_one_turn_agent_is_released_once_its_running_slot_clears(monkeypatch):
    """Review of #140969: evicting the one-turn agent while its turn still ran skipped the release
    (``_evict_cached_agent`` never tears down a running agent) and nothing retried it, so its clients
    (and a Codex app-server child) outlived every sweep."""
    import threading
    from unittest.mock import MagicMock
    from gateway import session_local, session_policy
    from gateway.session_local_inventory import publish_settled_info
    from tests.waiting import wait_until
    route = "agent:main:local:dm:s"
    agent = MagicMock()
    agent.model, agent.provider, agent.tools, agent._cached_system_prompt = "once-model", "anthropic", None, ""
    agent.context_compressor = None
    released = threading.Event()
    agent.release_clients.side_effect = lambda: released.set()
    runner = _running_runner(route, agent)
    live = SimpleNamespace(route=route, source=SimpleNamespace())
    authority = SimpleNamespace(profile_id="p", sessions={"s": live}, runner=runner)
    published = []
    turn = SimpleNamespace(_ctx=SimpleNamespace(inbound_message_id="adm", session_key=route),
                           _approval_owner=(authority, "s", 1), _runner=runner,
                           _publish_execution=lambda kind, payload: published.append(payload))
    policy = SimpleNamespace(effective=lambda _a, secrets=True: ({"model": {"provider": "openrouter"}}, "session-model"))
    monkeypatch.setattr(session_local, "local_session_info", lambda *_a, **_k: {"model": "once-model"})
    monkeypatch.setattr(session_policy, "policy_for_source", lambda *_a, **_k: policy)

    with cli_turn_scope("adm", {"model_override": {"model": "once-model", "provider": "anthropic"}}):
        publish_settled_info(turn, agent)

    assert published, "session.info was not published"
    # Out of the cache at once, so nothing (resume, Desktop) reads the one-turn route off it ...
    assert route not in runner._agent_cache
    # ... and not torn down while its turn still runs.
    assert not agent.release_clients.called
    runner._release_running_agent_state(route)  # the turn ends
    assert wait_until(released.is_set, timeout=5, what="the one-turn agent was released")
    assert agent.release_clients.call_count == 1


def test_settled_info_pairs_model_and_provider_from_one_profile_read(monkeypatch):
    """Review of #140969: ``policy.provider`` read the frozen config while ``current_model`` followed
    the profile, so a following Bot Chat whose profile moved published the new model with the old
    provider."""
    from gateway import session_local, session_policy
    from gateway.session_local_inventory import publish_settled_info
    published = []

    class _Policy:
        provider = "anthropic"  # the frozen config's

        def effective(self, _authority, *, secrets=True):
            assert secrets is False
            return {"model": {"provider": "openrouter", "default": "openai/gpt-b"}}, "openai/gpt-b"

        def current_model(self, _authority):
            return "openai/gpt-b"

    live = SimpleNamespace(route="route-1", source=SimpleNamespace())
    authority = SimpleNamespace(profile_id="p", sessions={"s": live})
    turn = SimpleNamespace(_ctx=SimpleNamespace(inbound_message_id="adm"), _approval_owner=(authority, "s", 1),
                           _runner=SimpleNamespace(), _publish_execution=lambda kind, payload: published.append(payload))
    agent = SimpleNamespace(model="once-model", provider="nous", tools=None, _cached_system_prompt="",
                            context_compressor=None)
    monkeypatch.setattr(session_local, "local_session_info", lambda *_a, **_k: {"model": "once-model", "provider": "nous"})
    monkeypatch.setattr(session_policy, "policy_for_source", lambda *_a, **_k: _Policy())
    with cli_turn_scope("adm", {"model_override": {"model": "once-model", "provider": "nous"}}):
        publish_settled_info(turn, agent)
    assert (published[-1]["model"], published[-1]["provider"]) == ("openai/gpt-b", "openrouter")


def test_same_provider_override_keeps_the_sessions_wire_when_the_model_does_not_pick_one(monkeypatch):
    """Review of #140969: ``result.api_mode`` is derived from the provider's DEFAULT host. A session on
    openai-api behind a chat-completions proxy kept the proxy but took ``codex_responses``, a wire the
    proxy does not speak. Only a provider that picks its wire per model overrides the session's."""
    runtime = _same_provider_override(
        monkeypatch, "openai-api", "gpt-5.4", "codex_responses", "https://api.openai.com/v1",
        {"base_url": "https://proxy.example/v1", "api_mode": "chat_completions"})
    assert (runtime["base_url"], runtime["api_mode"]) == ("https://proxy.example/v1", "chat_completions")


SESSION_URL, ALIAS_URL = "http://127.0.0.1:7001/v1", "http://127.0.0.1:7002/v1"


@pytest.fixture
def alias_config(monkeypatch):
    """A ``model_aliases`` entry on the session's own provider with its own endpoint and key, resolved
    by the real ``switch_model`` (only the catalog probe is stubbed)."""
    cfg = {"model": {"default": "session-model", "provider": "custom", "base_url": SESSION_URL},
           "model_aliases": {"fast": {"model": "fast-model", "provider": "custom", "base_url": ALIAS_URL,
                                      "api_key": "sk-alias"}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **k: {
        "accepted": True, "persist": True, "recognized": True, "message": ""})
    monkeypatch.setattr("hermes_cli.model_switch.DIRECT_ALIASES", {})
    return cfg


@pytest.mark.parametrize("override", [{"model": "fast"}, {"model": "fast-model", "provider": "custom"}],
                         ids=["alias-name", "cli-resolved-form"])
def test_a_same_provider_alias_runs_on_its_own_endpoint_and_key(alias_config, override):
    """Review of #140969: the same-provider branch kept the session's endpoint and key for an alias that
    names its own, so the turn asked the session's server for the alias's model with the session's key."""
    model, runtime = one_turn_runtime(
        override, "session-model",
        {"provider": "custom", "api_key": "sk-session", "base_url": SESSION_URL, "api_mode": "chat_completions",
         "credential_pool": object()},
        alias_config)
    assert model == "fast-model"
    assert (runtime["base_url"], runtime["api_key"]) == (ALIAS_URL, "sk-alias")
    assert runtime["credential_pool"] is None  # the session's pool holds the session's keys


def test_a_plain_same_provider_pick_still_keeps_the_session_endpoint(alias_config):
    _model, runtime = one_turn_runtime(
        {"model": "other-model", "provider": "custom"}, "session-model",
        {"provider": "custom", "api_key": "sk-session", "base_url": SESSION_URL, "api_mode": "chat_completions"},
        alias_config)
    assert (runtime["base_url"], runtime["api_key"]) == (SESSION_URL, "sk-session")


PROXY = "https://llm-proxy.example/anthropic"


@pytest.mark.parametrize("typed,aliases", [
    ("claude-haiku-4-5", {}),
    ("haiku", {}),
    ("quick", {"quick": {"model": "claude-haiku-4-5", "provider": "anthropic"}}),
    ("claude-haiku-4-5", {"quick": {"model": "claude-haiku-4-5", "provider": "anthropic"}}),
    ("claude-haiku-4-5", {"claude-haiku-4-5": {"model": "claude-haiku-4-5", "provider": "anthropic"}}),
], ids=["model-id", "builtin-alias", "endpointless-alias", "model-id-reverse-matching-it", "alias-named-like-its-model"])
def test_a_same_provider_alias_without_an_endpoint_keeps_the_session_route(monkeypatch, typed, aliases):
    """Review of #140969: every alias kind took the "alias runs whole" branch, so ``/model haiku --once`` on
    a session behind a proxy, or rotating a pool, ran that turn on the provider's default host and key.
    Only an alias that names its own endpoint leaves the session's route."""
    cfg = {"model": {"default": "claude-sonnet-4-6", "provider": "anthropic"}, "model_aliases": aliases}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **k: {
        "accepted": True, "persist": True, "recognized": True, "message": ""})
    monkeypatch.setattr("hermes_cli.model_switch.DIRECT_ALIASES", {})
    monkeypatch.setattr("hermes_cli.model_switch.list_provider_models",
                        lambda provider: ["claude-haiku-4-5", "claude-sonnet-4-6"] if provider == "anthropic" else [])
    monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    # The provider's default route, which the explicit provider re-resolves.
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_k: {
        "provider": "anthropic", "api_key": "sk-default", "base_url": "https://api.anthropic.com",
        "api_mode": "anthropic_messages"})
    pool = object()
    model, runtime = one_turn_runtime(
        {"model": typed, "provider": "anthropic"}, "claude-sonnet-4-6",
        {"provider": "anthropic", "api_key": "sk-session", "base_url": PROXY, "api_mode": "anthropic_messages",
         "credential_pool": pool, "request_overrides": {"metadata": {"team": "a"}}},
        cfg)
    assert model == "claude-haiku-4-5"
    assert (runtime["base_url"], runtime["api_key"]) == (PROXY, "sk-session")
    assert runtime["credential_pool"] is pool
    assert runtime["request_overrides"] == {"metadata": {"team": "a"}}


@pytest.mark.parametrize("typed", ["opus", "claude-opus-5"], ids=["provider-alias", "declared-id"])
def test_a_process_providers_declared_model_keeps_the_session_command(monkeypatch, typed):
    """An ``external_process`` provider's own alias or declared id resolves like an alias, and the turn
    still runs on the session's command."""
    from providers.base import ProviderProfile
    profile = ProviderProfile(name="proc-provider", auth_type="external_process", base_url="process://proc-provider",
                              fallback_models=("claude-sonnet-5", "claude-opus-5"), model_aliases={"opus": "claude-opus-5"})
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile if name == profile.name else None)
    cfg = {"model": {"default": "claude-sonnet-5", "provider": profile.name}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("hermes_cli.model_switch.DIRECT_ALIASES", {})
    monkeypatch.setattr("hermes_cli.model_switch.list_provider_models", lambda provider: [])
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **k: {
        "accepted": True, "persist": True, "recognized": True, "message": ""})
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_k: {
        "provider": profile.name, "api_key": "external-process", "base_url": profile.base_url,
        "api_mode": "chat_completions"})
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda _config: [])
    model, runtime = one_turn_runtime(
        {"model": typed, "provider": profile.name}, "claude-sonnet-5",
        {"provider": profile.name, "api_key": "external-process", "base_url": profile.base_url,
         "api_mode": "chat_completions", "command": "proc-cli", "args": ["--serve"]},
        cfg)
    assert model == "claude-opus-5"
    assert (runtime["command"], runtime["args"]) == ("proc-cli", ["--serve"])


def test_an_alias_runs_on_the_endpoint_the_switch_resolved_when_the_config_changes_after(monkeypatch):
    """The owner re-read the alias table after ``switch_model`` had resolved and validated the alias's
    endpoint, so a ``config.yaml`` edit in between (its ``base_url`` removed) ran the alias's model on the
    session's endpoint and key. The route the switch resolved decides."""
    from hermes_cli import model_switch
    from hermes_constants import get_config_path
    path = get_config_path()
    config = ("model:\n  default: session-model\n  provider: custom\n  base_url: " + SESSION_URL + "\n"
              "model_aliases:\n  fast:\n    model: fast-model\n    provider: custom\n")
    path.write_text(config + "    base_url: " + ALIAS_URL + "\n    api_key: sk-alias\n", encoding="utf-8")
    edited = []

    def validate(*_a, **_k):
        path.write_text(config, encoding="utf-8")  # the edit lands after the endpoint was applied
        edited.append(True)
        return {"accepted": True, "persist": True, "recognized": True, "message": ""}

    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", validate)
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})
    monkeypatch.setattr(model_switch, "_DIRECT_ALIAS_IDENTITY", None)
    monkeypatch.setattr(model_switch, "_DIRECT_ALIAS_LOADED", None)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    model, runtime = one_turn_runtime(
        {"model": "fast", "provider": "custom"}, "session-model",
        {"provider": "custom", "api_key": "sk-session", "base_url": SESSION_URL, "api_mode": "chat_completions",
         "credential_pool": object()},
        {})
    assert edited, "the switch never validated the pick"
    assert model == "fast-model"
    assert (runtime["base_url"], runtime["api_key"]) == (ALIAS_URL, "sk-alias")
    assert runtime["credential_pool"] is None


def test_a_one_turn_override_carries_its_reasoning_level_for_that_admission_only():
    from gateway.session_cli_turn import admit_cli_turn, one_turn_reasoning
    from hermes_state_runtime import RuntimeStoreError
    committed = admit_cli_turn({"model_override": {"model": "m", "provider": "p", "reasoning": "high"}})
    assert committed["cli_turn_v1"]["model_override"] == {"model": "m", "provider": "p", "reasoning": "high"}
    with pytest.raises(RuntimeStoreError):
        admit_cli_turn({"model_override": {"model": "m", "reasoning": "hgih"}})
    session_level = {"enabled": True, "effort": "low"}
    turn = SimpleNamespace(_ctx=SimpleNamespace(inbound_message_id="adm"))
    with cli_turn_scope("adm", committed["cli_turn_v1"]):
        assert one_turn_reasoning(turn, session_level) == {"enabled": True, "effort": "high"}
    with cli_turn_scope("other", committed["cli_turn_v1"]):
        assert one_turn_reasoning(turn, session_level) is session_level
