"""Codex accounts must be eligible for the final default/session/channel model.

Synthetic pool-only setup: two accounts reject TARGET; the third rejects OTHER.
The real resolver and session store run against a disposable home, without inference.
"""

import json
from pathlib import Path

import pytest

TARGET = "gpt-5.3-codex"
OTHER = "gpt-5.3-codex-mini"
BASE = "https://chatgpt.com/backend-api/codex"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    _config(root, TARGET)
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": f"account-{i}", "label": f"account-{i}", "auth_type": "oauth",
         "source": "manual", "priority": i, "access_token": f"synthetic-{i}",
         "refresh_token": f"synthetic-refresh-{i}", "expires_at_ms": 4_000_000_000_000,
         "base_url": BASE, "model_cooldowns": {TARGET if i < 2 else OTHER: 4_000_000_000}}
        for i in range(3)
    ]}}))
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", root)
    # Any accidental refresh/probe is a test failure, never a provider request.
    import httpx
    def no_network(*args, **kwargs):
        pytest.fail("Credential routing must not make a provider request")
    monkeypatch.setattr(httpx.Client, "send", no_network)
    return root


def _config(home, model, **extra):
    (home / "config.yaml").write_text(json.dumps({
        "model": {"provider": "openai-codex", "default": model}, **extra,
    }))


def _runner(home):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionStore
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(sessions_dir=home / "sessions", config=runner.config)
    return runner


def _source():
    from gateway.config import Platform
    from gateway.session import SessionSource
    return SessionSource(platform=Platform.TELEGRAM, chat_id="routing-test", user_id="test")


def _assert_route(model, runtime, expected_model, account):
    assert model == expected_model
    assert runtime["provider"] == "openai-codex"
    assert runtime["base_url"] == BASE
    assert runtime["api_key"] == f"synthetic-{account}"
    pool = runtime["credential_pool"]
    assert pool is not None
    assert pool.select(model=model).id == f"account-{account}"


def test_default_uses_account_with_only_unrelated_model_bench(home):
    runner = _runner(home)
    _assert_route(*runner._resolve_session_agent_runtime(source=_source()), TARGET, 2)


@pytest.mark.parametrize("path", ["fresh", "cached", "rehydrated"])
@pytest.mark.parametrize("model,account", [(TARGET, 2), (OTHER, 0)])
def test_session_override_attaches_pool_for_its_model(home, path, model, account):
    _config(home, OTHER if model == TARGET else TARGET)
    runner = _runner(home)
    source = _source()
    key = runner.session_store.get_or_create_session(source).session_key
    override = {"model": model, "provider": "openai-codex", "base_url": BASE,
                "api_key": f"synthetic-{account}", "api_mode": "codex_responses"}
    if path == "rehydrated":
        runner.session_store.set_model_override(key, override)
        runner = _runner(home)
    else:
        runner._session_state(key).conversation.model_override = override
    if path == "cached":
        resolved = runner._apply_session_model_override(key, OTHER, {})
    else:
        resolved = runner._resolve_session_agent_runtime(source=source, session_key=key)
    _assert_route(*resolved, model, account)


@pytest.mark.parametrize("provider", [None, "openai-codex"])
def test_channel_model_is_chosen_before_credentials(home, provider):
    from gateway.config import ChannelOverride, PlatformConfig
    runner = _runner(home)
    source = _source()
    runner.config.platforms[source.platform] = PlatformConfig(
        enabled=True, channel_overrides={source.chat_id: ChannelOverride(model=OTHER, provider=provider)})
    _assert_route(*runner._resolve_session_agent_runtime(source=source), OTHER, 0)


def test_explicit_provider_model_ignores_opposite_config_default(home):
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    runtime = _resolve_runtime_agent_kwargs_for_provider("openai-codex", target_model=OTHER)
    _assert_route(OTHER, runtime, OTHER, 0)


def test_unknown_model_pool_read_stays_conservative(home):
    from gateway.run import _credential_pool_for_provider
    assert _credential_pool_for_provider("openai-codex") is None


@pytest.mark.asyncio
async def test_global_save_clears_override_and_still_selects_eligible_account(home):
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.model_switch import ModelSwitchResult
    _config(home, OTHER)
    runner = _runner(home)
    source = _source()
    key = runner.session_store.get_or_create_session(source).session_key
    result = ModelSwitchResult(success=True, new_model=TARGET, target_provider="openai-codex",
                               api_key="synthetic-2", base_url=BASE, api_mode="codex_responses")
    ctx = _ModelSwitchContext(session_key=key, source=source, config_path=home / "config.yaml",
                              persist_global=True, current_model=OTHER)
    warning = await runner._record_model_switch(result, ctx, source=source, one_turn=False, picker=False)
    assert warning is None
    assert runner._session_model_override(key) is None
    assert runner.session_store.get_model_override(key) is None
    from hermes_cli.config_effective import load_user_config_effective
    assert load_user_config_effective()["model"]["default"] == TARGET
    _assert_route(*runner._resolve_session_agent_runtime(source=source, session_key=key), TARGET, 2)
