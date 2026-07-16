"""Public Gateway credential rebinds are scoped, verified, and ephemeral."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, rebind_gateway_session_credentials
from gateway.session import AsyncSessionStore, SessionSource, SessionStore


class _Store:
    def __init__(self, entry):
        self.entry = entry
        self.transcript = [{"role": "user", "content": "keep me"}]

    def lookup_by_session_id(self, session_id):
        if self.entry is not None and self.entry.session_id == session_id:
            return self.entry
        return None

    def set_model_override(self, *_args, **_kwargs):
        raise AssertionError("credential rebind must not persist the override")


class _Pool:
    def __init__(self, provider="openrouter", credential_id="cred-b", token="token-b"):
        self.provider = provider
        self._selected = SimpleNamespace(
            id=credential_id,
            runtime_api_key=token,
        )

    def current(self):
        return self._selected


def _runner(*, override_provider="openrouter", with_override=True, with_cache=True):
    runner = object.__new__(GatewayRunner)
    entry = SimpleNamespace(
        session_id="session-1",
        session_key="route-1",
        model_override=None,
    )
    runner.session_store = _Store(entry)
    runner._async_session_store = None
    runner._running_agents = {}
    runner._session_model_overrides = {}
    if with_override:
        runner._session_model_overrides["route-1"] = {
            "model": "model-a",
            "provider": override_provider,
            "api_key": "token-a",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "marker": "preserve",
        }
    runner._session_model_overrides["route-2"] = {
        "model": "other-model",
        "provider": "anthropic",
        "api_key": "other-token",
    }
    runner._agent_cache = {}
    if with_cache:
        agent = SimpleNamespace(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model="model-a",
        )
        runner._agent_cache["route-1"] = (agent, "sig", 1, "session-1")
    runner._agent_cache["route-2"] = (object(), "other-sig")
    runner._agent_cache_lock = threading.Lock()
    runner._evict_cached_agent = MagicMock()
    return runner


def _runtime(
    *,
    pool=None,
    token="token-b",
    provider="openrouter",
    requested_provider="openrouter",
):
    return {
        "provider": provider,
        "requested_provider": requested_provider,
        "api_key": token,
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "credential_pool": pool if pool is not None else _Pool(),
    }


@pytest.mark.asyncio
async def test_rebind_crosses_real_persistence_and_next_runtime_resolution(
    tmp_path, monkeypatch
):
    import hermes_state

    hermes_home = tmp_path / "hermes-home"
    sessions_dir = hermes_home / "sessions"
    state_db = hermes_home / "state.db"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", state_db)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
        user_name="tester",
        chat_type="dm",
    )
    store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    entry = store.get_or_create_session(source)
    store.set_model_override(
        entry.session_key,
        {
            "model": "model-a",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
        },
    )
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "keep me"})
    store._db.close()

    restarted_store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    restarted_entry = restarted_store.lookup_by_session_id(entry.session_id)
    assert restarted_entry is not None
    assert restarted_entry.model_override == {
        "model": "model-a",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    }
    transcript_before = restarted_store.load_transcript(entry.session_id)

    runner = object.__new__(GatewayRunner)
    runner.session_store = restarted_store
    runner._async_session_store = AsyncSessionStore(restarted_store)
    runner._running_agents = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._evict_cached_agent = MagicMock()

    pool = CredentialPool(
        "openrouter",
        [
            PooledCredential(
                provider="openrouter",
                id="cred-a",
                label="A",
                auth_type="api_key",
                priority=0,
                source="manual",
                access_token="token-a",
            ),
            PooledCredential(
                provider="openrouter",
                id="cred-b",
                label="B",
                auth_type="api_key",
                priority=1,
                source="manual",
                access_token="token-b",
            ),
        ],
    )
    assert pool.acquire_lease("cred-b") == "cred-b"
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: _runtime(pool=pool, token="token-b"),
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda _config: "global")

    result = await runner.rebind_session_credentials(
        session_id=entry.session_id,
        provider="openrouter",
        credential_id="cred-b",
    )
    model, runtime = runner._resolve_session_agent_runtime(
        session_key=entry.session_key,
        user_config={},
    )

    assert result["ok"] is True
    assert model == "model-a"
    assert runtime["api_key"] == "token-b"
    assert runtime["credential_pool"] is pool
    assert restarted_store.load_transcript(entry.session_id) == transcript_before
    assert restarted_store.get_model_override(entry.session_key) == {
        "model": "model-a",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    }

    restarted_store._db.close()
    for persisted_file in hermes_home.rglob("*"):
        if persisted_file.is_file():
            assert b"token-b" not in persisted_file.read_bytes()


@pytest.mark.asyncio
async def test_rebind_refreshes_only_ephemeral_target_session_state(monkeypatch):
    runner = _runner()
    original_override = runner._session_model_overrides["route-1"]
    other_override = runner._session_model_overrides["route-2"]
    other_cache = runner._agent_cache["route-2"]
    transcript = runner.session_store.transcript
    pool = _Pool()
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: _runtime(pool=pool),
    )

    result = await runner.rebind_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result == {
        "ok": True,
        "code": "rebound",
        "session_id": "session-1",
        "provider": "openrouter",
        "credential_id": "cred-b",
        "session_override_refreshed": True,
    }
    refreshed = runner._session_model_overrides["route-1"]
    assert refreshed is not original_override
    assert refreshed == {
        "model": "model-a",
        "provider": "openrouter",
        "api_key": "token-b",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "credential_pool": pool,
        "marker": "preserve",
    }
    assert runner._session_model_overrides["route-2"] is other_override
    assert runner._agent_cache["route-2"] is other_cache
    assert runner.session_store.transcript is transcript
    runner._evict_cached_agent.assert_called_once_with("route-1")
    assert "token-a" not in str(result)
    assert "token-b" not in str(result)


@pytest.mark.asyncio
async def test_rebind_can_invalidate_cached_agent_without_model_override(monkeypatch):
    runner = _runner(with_override=False)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: _runtime(),
    )

    result = await runner.rebind_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result["ok"] is True
    assert result["session_override_refreshed"] is False
    assert "route-1" not in runner._session_model_overrides
    runner._evict_cached_agent.assert_called_once_with("route-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "expected_code"),
    [
        (_runtime(pool=_Pool(credential_id="cred-c")), "credential_mismatch"),
        (_runtime(pool=_Pool(token="different-token")), "credential_mismatch"),
        (_runtime(token=""), "runtime_token_missing"),
        ({**_runtime(), "credential_pool": None}, "credential_pool_missing"),
        (
            _runtime(pool=_Pool(provider="anthropic")),
            "provider_mismatch",
        ),
    ],
)
async def test_rebind_fails_closed_on_resolver_disagreement(
    monkeypatch, runtime, expected_code
):
    runner = _runner()
    original_override = runner._session_model_overrides["route-1"]
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: runtime,
    )

    result = await runner.rebind_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result["ok"] is False
    assert result["code"] == expected_code
    assert runner._session_model_overrides["route-1"] is original_override
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_rebind_rejects_credential_from_another_session_provider(monkeypatch):
    runner = _runner(override_provider="anthropic")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: _runtime(),
    )

    result = await runner.rebind_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result["ok"] is False
    assert result["code"] == "session_provider_mismatch"
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_rebind_refuses_busy_session_before_resolving(monkeypatch):
    runner = _runner()
    runner._running_agents["route-1"] = object()
    resolver = MagicMock(return_value=_runtime())
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        resolver,
    )

    result = await runner.rebind_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result["code"] == "session_busy"
    resolver.assert_not_called()
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_rebind_restores_persisted_override_without_writing_it(monkeypatch):
    runner = _runner(with_override=False, with_cache=False)
    runner.session_store.entry.model_override = {
        "model": "model-a",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: _runtime(),
    )

    result = await runner.rebind_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result["ok"] is True
    assert runner._session_model_overrides["route-1"]["api_key"] == "token-b"
    assert runner.session_store.entry.model_override == {
        "model": "model-a",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    }
    runner._evict_cached_agent.assert_called_once_with("route-1")


@pytest.mark.asyncio
async def test_rebind_refuses_session_without_provider_bound_runtime(monkeypatch):
    runner = _runner(with_override=False, with_cache=False)
    resolver = MagicMock(return_value=_runtime())
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        resolver,
    )

    result = await runner.rebind_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result["ok"] is False
    assert result["code"] == "session_runtime_missing"
    resolver.assert_not_called()
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_rebind_refuses_session_change_during_resolution(monkeypatch):
    runner = _runner()

    async def change_session_then_resolve(func, *args, **kwargs):
        result = func(*args, **kwargs)
        runner.session_store.entry = SimpleNamespace(
            session_id="session-1",
            session_key="route-replaced",
            model_override=None,
        )
        return result

    monkeypatch.setattr("gateway.run.asyncio.to_thread", change_session_then_resolve)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: _runtime(),
    )

    result = await runner.rebind_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result["code"] == "session_changed"
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_public_rebind_reports_missing_gateway_without_secrets(monkeypatch):
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)

    result = await rebind_gateway_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result["ok"] is False
    assert result["code"] == "gateway_unavailable"


@pytest.mark.asyncio
async def test_public_rebind_rejects_stopped_gateway(monkeypatch):
    stopped_runner = SimpleNamespace(
        _running=False,
        _shutdown_event=SimpleNamespace(is_set=lambda: False),
        rebind_session_credentials=AsyncMock(),
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: stopped_runner)

    result = await rebind_gateway_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result["ok"] is False
    assert result["code"] == "gateway_unavailable"
    stopped_runner.rebind_session_credentials.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_context_delegates_to_public_gateway_rebind(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    facade = AsyncMock(return_value={"ok": True, "code": "rebound"})
    monkeypatch.setattr("gateway.run.rebind_gateway_session_credentials", facade)
    context = PluginContext(
        PluginManifest(name="credential-switcher", source="user"),
        PluginManager(),
    )

    result = await context.rebind_gateway_session_credentials(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )

    assert result == {"ok": True, "code": "rebound"}
    facade.assert_awaited_once_with(
        session_id="session-1",
        provider="openrouter",
        credential_id="cred-b",
    )
