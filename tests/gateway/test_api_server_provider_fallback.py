"""An explicitly requested api_server provider walks the fallback chain at resolution time.

A client that names ``provider`` (+ ``model``) in the request body got a controlled error text
(``_ProviderAuthResolutionError``) when that provider's credentials were missing or its pool
was benched for the model — although the very same agent, had the 429 arrived mid-turn, would
have switched to ``fallback_model`` (``_create_agent`` wires the chain for every unlocked
session) and the gateway default already walks the chain at resolution time (#81209). The
explicit-provider path now does the same; a confirmed model lock and a ``model_routes`` route
keep failing closed. The switch is visible to the client the way a mid-turn fallback is: the
turn's runtime metadata reports the actual provider/model beside the requested pair.

Real credential pool against a real temp auth store and config.yaml; the agent itself is a
capturing stand-in.
"""
import json
import time

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _ProviderAuthResolutionError

KEY_A = "sk-ant-oat01-seat-a"
KEY_B = "sk-ant-oat01-seat-b"
MODEL_X = "claude-opus-4-5"
FIXTURE_KEY = "fixture-key"


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "config.yaml").write_text(json.dumps({
        "model": {"default": "global/model", "provider": "local-fixture"},
        "providers": {"local-fixture": {"api": "http://127.0.0.1:1/v1", "api_key": FIXTURE_KEY}},
        "fallback_model": [{"provider": "local-fixture", "model": "fixture"}],
    }))
    benched = {MODEL_X: time.time() + 3600}
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [
        {"id": seat, "label": seat, "auth_type": "oauth", "priority": prio, "source": "manual:hermes_pkce",
         "access_token": key, "refresh_token": f"rt-{seat}", "expires_at_ms": int((time.time() + 86400) * 1000),
         "model_cooldowns": benched}
        for seat, key, prio in (("seat-a", KEY_A, 0), ("seat-b", KEY_B, 1))]}}))
    return root


@pytest.fixture
def adapter(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)
    # The global default runtime is not under test; keep it a plain, healthy route.
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {
        "provider": "local-fixture", "api_key": FIXTURE_KEY, "base_url": "http://127.0.0.1:1/v1",
        "api_mode": "chat_completions"})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": {}))
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    return adapter


def _capturing_agent(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model")
            self.provider = kwargs.get("provider")

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    return captured


def test_resolver_walks_the_chain_only_when_asked(home):
    from gateway.platforms.api_server_provider_runtime import _resolve_request_runtime_agent_kwargs

    with pytest.raises(RuntimeError, match="rate-limited|credentials"):
        _resolve_request_runtime_agent_kwargs("anthropic", MODEL_X)

    runtime = _resolve_request_runtime_agent_kwargs("anthropic", MODEL_X, allow_fallback=True)
    assert runtime["model"] == "fixture"
    assert runtime["provider"] == "local-fixture"
    assert runtime["api_key"] == FIXTURE_KEY
    assert getattr(runtime.get("credential_pool"), "provider", "") != "anthropic"


def test_explicit_provider_benched_for_the_model_starts_on_the_fallback(home, adapter, monkeypatch):
    captured = _capturing_agent(monkeypatch)

    agent = adapter._create_agent(session_id="s1", requested_provider="anthropic", requested_model=MODEL_X)

    assert captured["model"] == "fixture"
    assert captured["provider"] == "local-fixture"
    assert captured["api_key"] == FIXTURE_KEY
    assert getattr(captured.get("credential_pool"), "provider", "") != "anthropic"
    # What the client is told afterwards: actual runtime beside the requested pair.
    meta = adapter._turn_runtime_metadata(
        agent, route=None, requested_runtime={"provider": "anthropic", "model": MODEL_X},
        route_source="raw_request", confirmed_runtime_lock=False)
    assert meta["provider"] == "local-fixture" and meta["model"] == "fixture"
    assert meta["requested"] == {"provider": "anthropic", "model": MODEL_X}


def test_confirmed_lock_still_fails_closed(home, adapter, monkeypatch):
    _capturing_agent(monkeypatch)
    with pytest.raises(_ProviderAuthResolutionError):
        adapter._create_agent(
            session_id="s1", requested_provider="anthropic", requested_model=MODEL_X, confirmed_runtime_lock=True)


def test_route_provider_does_not_walk_the_chain(home, monkeypatch):
    """A ``model_routes`` alias may pin its own transport; it keeps today's behaviour (no fallback, not required)."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "model_routes": {"alias": {"model": MODEL_X, "provider": "anthropic"}}}))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {
        "provider": "local-fixture", "api_key": FIXTURE_KEY, "base_url": "http://127.0.0.1:1/v1",
        "api_mode": "chat_completions"})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": {}))
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    captured = _capturing_agent(monkeypatch)

    adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))

    assert captured["model"] == MODEL_X  # the route's model, not a fallback entry's
