import threading
import time
from unittest.mock import MagicMock

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    STATUS_EXHAUSTED,
    CredentialPool,
    PooledCredential,
    _CodexUsageStatus,
)
from gateway.run import GatewayRunner


def _entry(entry_id: str, value: str) -> PooledCredential:
    return PooledCredential(
        provider="openai-codex",
        id=entry_id,
        label="stale exhaustion",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual",
        access_token=value,
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_code=429,
        last_error_reason="usage_limit_reached",
        last_error_reset_at=time.time() + 3600,
    )


@pytest.mark.asyncio
async def test_gateway_runtime_resolution_offloads_codex_live_usage_probe(monkeypatch):
    pool = CredentialPool("openai-codex", [_entry("cred-one", "opaque-one")])
    monkeypatch.setattr(pool, "_persist", MagicMock())
    main_thread = threading.get_ident()
    probe_threads = []

    def fake_usage_status(_candidate):
        probe_threads.append(threading.get_ident())
        return _CodexUsageStatus(available=True, allowed=True)

    monkeypatch.setattr(
        "agent.credential_pool._fetch_codex_entry_usage_status", fake_usage_status
    )
    runner = object.__new__(GatewayRunner)

    def resolve_runtime(**_kwargs):
        selected = pool.select()
        assert selected is not None
        return "gpt-test", {
            "provider": "openai-codex",
            "credential_pool_entry_id": selected.id,
        }

    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", resolve_runtime)
    model, runtime = await runner._resolve_session_agent_runtime_off_loop()

    assert model == "gpt-test"
    assert runtime["credential_pool_entry_id"] == "cred-one"
    assert probe_threads and all(thread_id != main_thread for thread_id in probe_threads)
    pool._persist.assert_called_once()


def test_entry_id_for_api_key_is_read_only(monkeypatch):
    entry = _entry("cred-match", "opaque-match")
    entry.last_status = "ok"
    pool = CredentialPool("openai-codex", [entry])
    select_spy = MagicMock(wraps=pool.select)
    monkeypatch.setattr(pool, "select", select_spy)

    assert pool.entry_id_for_api_key("opaque-match") == "cred-match"
    assert pool.entry_id_for_api_key("other") is None
    select_spy.assert_not_called()


def test_sync_codex_selection_reconciles_before_return(monkeypatch):
    pool = CredentialPool("openai-codex", [_entry("cred-two", "opaque-two")])
    monkeypatch.setattr(pool, "_persist", MagicMock())
    monkeypatch.setattr(
        "agent.credential_pool._fetch_codex_entry_usage_status",
        lambda _candidate: _CodexUsageStatus(available=True, allowed=True),
    )

    assert pool.select().id == "cred-two"
    pool._persist.assert_called_once()
