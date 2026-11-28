"""The run_agent singleton refresh re-checks Codex usage admission."""

from agent.usage_admission import policy
from tests.agent.test_run_agent_codex_responses import _build_agent


def test_try_refresh_codex_singleton_rechecks_usage_admission(monkeypatch):
    from datetime import UTC, datetime, timedelta

    from agent import credential_pool
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
    from agent.credential_pool import (
        AUTH_TYPE_OAUTH,
        CredentialPool,
        PooledCredential,
    )

    agent = _build_agent(monkeypatch)
    agent.provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent._credential_pool = None
    old_key = agent.api_key

    def _fake_resolve(force_refresh=False, refresh_if_expiring=True, **_):
        return {
            "api_key": "fresh-singleton-token" if force_refresh else old_key,
            "base_url": "https://chatgpt.com/backend-api/codex",
        }

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        _fake_resolve,
    )
    pool = CredentialPool(
        "openai-codex",
        [
            PooledCredential(
                provider="openai-codex",
                id="singleton-entry",
                label="singleton",
                auth_type=AUTH_TYPE_OAUTH,
                priority=0,
                source="device_code",
                access_token="fresh-singleton-token",
                refresh_token="refresh-token",
                base_url="https://chatgpt.com/backend-api/codex",
            )
        ],
    )
    monkeypatch.setattr(
        policy,
        "get_pool_usage_limits",
        lambda _provider: {"singleton-entry": 80.0},
    )
    monkeypatch.setattr(credential_pool, "load_pool", lambda _provider: pool)
    probed_tokens = []

    def _usage(_provider, *, api_key, **_kwargs):
        probed_tokens.append(api_key)
        return AccountUsageSnapshot(
            provider="openai-codex",
            source="usage_api",
            fetched_at=datetime.now(UTC),
            windows=(
                AccountUsageWindow(
                    label="Weekly",
                    used_percent=80,
                    limit_window_seconds=604_800,
                    reset_at=datetime.now(UTC) + timedelta(days=7),
                ),
            ),
        )

    monkeypatch.setattr("agent.account_usage.fetch_account_usage", _usage)

    assert agent._try_refresh_codex_client_credentials(force=True) is False
    assert probed_tokens == ["fresh-singleton-token"]
    assert agent.api_key == old_key


def test_try_refresh_codex_singleton_admission_failure_fails_open(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent._credential_pool = None
    old_key = agent.api_key

    def _fake_resolve(force_refresh=False, refresh_if_expiring=True, **_):
        return {
            "api_key": "fresh-singleton-token" if force_refresh else old_key,
            "base_url": "https://chatgpt.com/backend-api/codex",
        }

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "agent.usage_admission.policy.usage_admission_credential_denied",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    monkeypatch.setattr(
        agent,
        "_replace_primary_openai_client",
        lambda *, reason: True,
    )

    assert agent._try_refresh_codex_client_credentials(force=True) is True
    assert agent.api_key == "fresh-singleton-token"
