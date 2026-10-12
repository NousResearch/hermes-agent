"""A persisted bare-``custom`` /model override must not be rehydrated with another endpoint's key.

``api_key`` is never persisted, so after a gateway restart ``_rehydrate_session_model_override``
re-resolves credentials by provider name. For bare ``custom`` the name carries no endpoint: it
resolves config's custom endpoint, whose key was then copied into an override that keeps the
session's own ``base_url``, and the next turn built an AIAgent sending config's key to the
session's host. The key of a bare-custom runtime belongs to its URL, so a mismatch is resolved
again for the override's own endpoint (host-gated keys only), on rehydrate and on the per-turn
resolving path alike. Same-endpoint overrides and named providers are unchanged.
"""
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore

SESSION_URL = "https://session.example.test/v1"
CONFIG_URL = "https://other.example.test/v1"


@pytest.fixture
def other_endpoint_home(tmp_path, monkeypatch):
    """Config's bare custom is other.example.test with its own literal key sk-OTHER."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: some-model\n"
        "  provider: custom\n"
        f"  base_url: {CONFIG_URL}\n"
        "  api_mode: chat_completions\n"
        "  api_key: sk-OTHER\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
                "CUSTOM_BASE_URL", "ANTHROPIC_API_KEY", "EXAMPLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """SessionStores over one sessions dir (a second instance is a restart), without SQLite."""
    import hermes_state

    def _raise(*_a, **_k):
        raise RuntimeError("SQLite disabled in test")

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return lambda: SessionStore(sessions_dir=sessions, config=GatewayConfig())


def _restarted_runner(store_factory, base_url, provider="custom", model="some-model-2"):
    """Persist an override through one store, return a fresh runner over a second store."""
    store = store_factory()
    source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", chat_type="dm")
    session_key = store.get_or_create_session(source).session_key
    store.set_model_override(session_key, {
        "model": model, "provider": provider, "base_url": base_url,
        "api_key": "sk-SESSION-never-persisted", "api_mode": "chat_completions",
    })
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store_factory()
    return runner, session_key


def _build_agent(model, runtime):
    from run_agent import AIAgent

    kwargs = {k: v for k, v in runtime.items() if k not in ("model", "request_overrides", "capabilities")}
    return AIAgent(model=model, quiet_mode=True, skip_context_files=True, skip_memory=True, **kwargs)


def test_rehydrated_override_on_another_endpoint_never_gets_config_key(other_endpoint_home, store_factory):
    """P4: config custom @ other.example.test (sk-OTHER), session override custom @
    session.example.test, gateway restarted. No agent may be built for the session host with sk-OTHER."""
    runner, session_key = _restarted_runner(store_factory, SESSION_URL)

    model, runtime = runner._resolve_session_agent_runtime(session_key=session_key)

    assert model == "some-model-2"
    assert runtime["base_url"].rstrip("/") == SESSION_URL
    assert runtime["api_key"] != "sk-OTHER"
    agent = _build_agent(model, runtime)
    assert "session.example.test" in str(agent.client.base_url)
    assert agent.client.api_key != "sk-OTHER"


def test_rehydrated_override_gets_its_own_endpoints_key(other_endpoint_home, store_factory, monkeypatch):
    """The session endpoint's own key (OPENAI_API_KEY issued for OPENAI_BASE_URL == the session URL)
    is what the rehydrated override carries, not config's key for the other host."""
    monkeypatch.setenv("OPENAI_BASE_URL", SESSION_URL)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-SESSION")
    runner, session_key = _restarted_runner(store_factory, SESSION_URL)

    model, runtime = runner._resolve_session_agent_runtime(session_key=session_key)

    assert runtime["base_url"].rstrip("/") == SESSION_URL
    assert runtime["api_key"] == "sk-SESSION"
    agent = _build_agent(model, runtime)
    assert agent.client.api_key == "sk-SESSION"


@pytest.mark.parametrize("persisted_url", [CONFIG_URL, CONFIG_URL + "/"])
def test_rehydrated_override_on_config_endpoint_keeps_its_key(other_endpoint_home, store_factory, persisted_url):
    """P4b: the override IS config's endpoint (trailing slash ignored): it still gets sk-OTHER."""
    runner, session_key = _restarted_runner(store_factory, persisted_url)

    model, runtime = runner._resolve_session_agent_runtime(session_key=session_key)

    assert runtime["base_url"].rstrip("/") == CONFIG_URL
    assert runtime["api_key"] == "sk-OTHER"
    assert _build_agent(model, runtime).client.api_key == "sk-OTHER"


def test_keyless_override_resolving_path_never_pairs_config_key(other_endpoint_home):
    """The shape a failed re-resolution leaves (no api_key) takes the per-turn resolving path; that
    path must not pair config's key with the override's own endpoint either."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "s": {"model": "some-model-2", "provider": "custom", "base_url": SESSION_URL,
              "api_key": None, "api_mode": "chat_completions"},
    }

    _, runtime = runner._resolve_session_agent_runtime(session_key="s")

    assert not (runtime["base_url"].rstrip("/") == SESSION_URL and runtime["api_key"] == "sk-OTHER")


def test_named_provider_override_rehydrates_as_before(store_factory):
    """A named provider's key belongs to the provider, not to one URL: a persisted base_url that
    differs from the resolved one is kept with the resolved key, from a single resolution."""
    runner, session_key = _restarted_runner(
        store_factory, "https://api.z.ai/api/coding/paas/v4", provider="zai", model="glm-5")
    calls = []

    def _resolve(provider, target_model=None, **kwargs):
        calls.append((provider, target_model, kwargs))
        return {"provider": "zai", "api_key": "sk-zai", "base_url": "https://api.z.ai/api/paas/v4",
                "api_mode": "chat_completions", "credential_pool": None}

    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", side_effect=_resolve):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert override["base_url"] == "https://api.z.ai/api/coding/paas/v4"
    assert override["api_key"] == "sk-zai"
    assert calls == [("zai", "glm-5", {})]


def test_bare_custom_override_without_base_url_keeps_the_config_key(other_endpoint_home):
    """No endpoint of its own to tell apart: resolve by name, as before, never raise."""
    from gateway.run_agent_cache import _resolve_override_runtime

    runtime = _resolve_override_runtime("custom", None, None)

    assert runtime["api_key"] == "sk-OTHER"
    assert runtime["base_url"].rstrip("/") == CONFIG_URL


def test_override_runtime_refuses_when_the_endpoint_never_resolves_to_itself(other_endpoint_home):
    """Fail closed: a resolver that lands on another host twice yields no runtime, not its key."""
    from gateway.run_agent_cache import _resolve_override_runtime

    elsewhere = {"api_key": "sk-ELSEWHERE", "base_url": "https://elsewhere.example.test/v1", "provider": "custom"}
    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=dict(elsewhere)):
        with pytest.raises(RuntimeError):
            _resolve_override_runtime("custom", None, SESSION_URL)
