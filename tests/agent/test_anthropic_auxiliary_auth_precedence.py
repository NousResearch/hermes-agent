"""Credential precedence tests for the native Anthropic auxiliary route."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_canonical_resolver_can_exclude_orphaned_env_pool_snapshot(
    tmp_path, monkeypatch
):
    """Re-resolution must not loop back to an orphaned env-seeded pool row."""
    hermes_home = tmp_path / "hermes"
    fake_home = tmp_path / "home"
    hermes_home.mkdir()
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(fake_home))
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    stale_token = "cc-orphaned-env-pool-token"
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [
                        {
                            "id": "orphaned-env-row",
                            "label": "orphaned env snapshot",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "env:ANTHROPIC_TOKEN",
                            "access_token": stale_token,
                            "base_url": "https://api.anthropic.com",
                        }
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: None,
    )

    from agent.anthropic_adapter import resolve_anthropic_token

    assert resolve_anthropic_token() == stale_token
    assert resolve_anthropic_token(allow_pool_fallback=False) is None


def test_anthropic_refresh_propagates_secret_scope_errors():
    """Missing secret scope is an auth failure, not a refresh miss."""
    from agent.auxiliary_client import _refresh_provider_credentials
    from agent.secret_scope import UnscopedSecretError

    with (
        patch(
            "agent.anthropic_credentials.read_claude_code_credentials",
            return_value=None,
        ),
        patch(
            "agent.anthropic_credentials.resolve_anthropic_token",
            side_effect=UnscopedSecretError("scope required"),
        ),
        pytest.raises(UnscopedSecretError, match="scope required"),
    ):
        _refresh_provider_credentials("anthropic")


@pytest.mark.parametrize(
    "env_source",
    ["env:ANTHROPIC_TOKEN", "env:CLAUDE_CODE_OAUTH_TOKEN"],
)
def test_env_backed_pool_entry_defers_to_canonical_anthropic_resolver(
    tmp_path, monkeypatch, env_source
):
    """A stale env pool token must not shadow current refreshable credentials."""
    from agent.auxiliary_client import _try_anthropic

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = SimpleNamespace(
        access_token="stale-env-token",
        runtime_api_key="",
        source=env_source,
        base_url="https://api.anthropic.com",
        runtime_base_url="https://api.anthropic.com",
        auth_type="oauth",
    )

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value="current-resolved-token",
        ) as mock_resolve,
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as mock_build,
    ):
        client, model = _try_anthropic()

    assert client is not None
    assert model == "claude-haiku-4-5-20251001"
    assert mock_build.call_args.args[0] == "current-resolved-token"
    mock_resolve.assert_called_once_with(allow_pool_fallback=False)


def test_auto_route_reresolves_inherited_anthropic_oauth_snapshot(
    tmp_path, monkeypatch
):
    """An auto auxiliary must not pin the main chat's rotating OAuth snapshot."""
    from agent.auxiliary_client import _resolve_auto_route

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = SimpleNamespace(
        access_token="cc-stale-pool-token",
        runtime_api_key="",
        source="env:CLAUDE_CODE_OAUTH_TOKEN",
        base_url="https://api.anthropic.com",
        runtime_base_url="https://api.anthropic.com",
        auth_type="oauth",
    )
    main_runtime = {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "base_url": "https://api.anthropic.com",
        "api_key": "cc-stale-main-runtime-token",
        "api_mode": "anthropic_messages",
    }

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value="cc-current-canonical-token",
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as mock_build,
    ):
        client, model, provider = _resolve_auto_route(main_runtime=main_runtime)

    assert client is not None
    assert provider == "anthropic"
    assert model == "claude-haiku-4-5-20251001"
    assert mock_build.call_args.args[0] == "cc-current-canonical-token"


def test_auto_route_preserves_inherited_anthropic_api_key(tmp_path, monkeypatch):
    """A durable main-runtime API key remains authoritative for auxiliaries."""
    from agent.auxiliary_client import _resolve_auto_route

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = SimpleNamespace(
        access_token="cc-unused-pool-token",
        runtime_api_key="",
        source="env:CLAUDE_CODE_OAUTH_TOKEN",
        base_url="https://api.anthropic.com",
        runtime_base_url="https://api.anthropic.com",
        auth_type="oauth",
    )
    main_runtime = {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "base_url": "https://api.anthropic.com",
        "api_key": "fixture-static-main-api-key",
        "api_mode": "anthropic_messages",
    }

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            side_effect=AssertionError("API key must skip OAuth resolution"),
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as mock_build,
    ):
        client, model, provider = _resolve_auto_route(main_runtime=main_runtime)

    assert client is not None
    assert provider == "anthropic"
    assert model == "claude-haiku-4-5-20251001"
    assert mock_build.call_args.args[0] == "fixture-static-main-api-key"


def test_auto_route_preserves_oauth_snapshot_for_custom_endpoint(
    tmp_path, monkeypatch
):
    """Native OAuth discovery must never overwrite endpoint-bound auth."""
    from agent.auxiliary_client import _resolve_auto_route

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = "https://proxy.api.anthropic.com/anthropic"
    entry = SimpleNamespace(
        access_token="cc-endpoint-pool-token",
        runtime_api_key="",
        source="env:CLAUDE_CODE_OAUTH_TOKEN",
        base_url=endpoint,
        runtime_base_url=endpoint,
        auth_type="oauth",
    )
    main_runtime = {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "base_url": endpoint,
        "api_key": "cc-endpoint-runtime-token",
        "api_mode": "anthropic_messages",
    }

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            side_effect=AssertionError(
                "custom endpoint must keep its runtime credential"
            ),
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as mock_build,
    ):
        client, model, provider = _resolve_auto_route(main_runtime=main_runtime)

    assert client is not None
    assert provider == "anthropic"
    assert model == "claude-haiku-4-5-20251001"
    assert mock_build.call_args.args[0] == "cc-endpoint-runtime-token"
    assert mock_build.call_args.args[1] == endpoint


def test_auto_route_rebuilds_anthropic_once_after_401_with_current_token(
    tmp_path, monkeypatch
):
    """A 401 refreshes and rebuilds native Anthropic without changing the route."""
    from agent.auxiliary_client import call_llm

    class _Auth401(Exception):
        status_code = 401

    hermes_home = tmp_path / "hermes"
    fake_home = tmp_path / "home"
    hermes_home.mkdir()
    (fake_home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("ANTHROPIC_TOKEN", "cc-fixture-token-a")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [
                        {
                            "id": "persisted-env-row",
                            "label": "persisted env snapshot",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "env:ANTHROPIC_TOKEN",
                            "access_token": "cc-fixture-token-a",
                            "base_url": "https://api.anthropic.com",
                        }
                    ]
                },
            }
        )
    )
    main_runtime = {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "base_url": "https://api.anthropic.com",
        "api_key": "cc-stale-main-runtime-token",
        "api_mode": "anthropic_messages",
    }
    route_before = dict(main_runtime)

    def _build_client(token, _base_url):
        return SimpleNamespace(token=token, close=MagicMock())

    def _create_message(client, _kwargs, **_unused):
        if client.token == "cc-fixture-token-a":
            (fake_home / ".claude" / ".credentials.json").write_text(
                json.dumps(
                    {
                        "claudeAiOauth": {
                            "accessToken": "cc-fixture-token-b",
                            "refreshToken": "fixture-refresh-token-b",
                            "expiresAt": 4102444800000,
                        }
                    }
                )
            )
            from agent.auxiliary_client import _peek_pool_entry, _pool_runtime_api_key

            persisted = _peek_pool_entry("anthropic")
            assert persisted is not None
            assert _pool_runtime_api_key(persisted) == "cc-fixture-token-a"
            raise _Auth401("expired OAuth fixture")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="fresh auxiliary response")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=2),
        )

    with (
        patch("agent.auxiliary_client._client_cache", {}),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            side_effect=_build_client,
        ) as mock_build,
        patch(
            "agent.anthropic_adapter.create_anthropic_message",
            side_effect=_create_message,
        ),
    ):
        response = call_llm(
            provider="auto",
            main_runtime=main_runtime,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
        )

    assert response.choices[0].message.content == "fresh auxiliary response"
    assert [call.args[0] for call in mock_build.call_args_list] == [
        "cc-fixture-token-a",
        "cc-fixture-token-b",
    ]
    assert main_runtime == route_before


def test_explicit_anthropic_key_wins_over_env_backed_pool(tmp_path, monkeypatch):
    """A caller-supplied credential is authoritative for that auxiliary call."""
    from agent.auxiliary_client import _try_anthropic

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = SimpleNamespace(
        access_token="stale-env-token",
        runtime_api_key="",
        source="env:ANTHROPIC_TOKEN",
        base_url="https://api.anthropic.com",
        runtime_base_url="https://api.anthropic.com",
        auth_type="oauth",
    )

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            side_effect=AssertionError("explicit key must skip automatic resolution"),
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as mock_build,
    ):
        client, _model = _try_anthropic(explicit_api_key="explicit-call-token")

    assert client is not None
    assert mock_build.call_args.args[0] == "explicit-call-token"


def test_manual_anthropic_pool_key_is_preserved(tmp_path, monkeypatch):
    """A manually managed pool credential must not be replaced implicitly."""
    from agent.auxiliary_client import _try_anthropic

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = SimpleNamespace(
        access_token="manual-pool-token",
        runtime_api_key="",
        source="manual",
        base_url="https://api.anthropic.com",
        runtime_base_url="https://api.anthropic.com",
        auth_type="api_key",
    )

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            side_effect=AssertionError("manual pool entry must skip automatic resolution"),
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as mock_build,
    ):
        client, _model = _try_anthropic()

    assert client is not None
    assert mock_build.call_args.args[0] == "manual-pool-token"


def test_auxiliary_anthropic_api_key_pool_entry_is_preserved(
    tmp_path, monkeypatch
):
    """Explicit Anthropic API keys must not be replaced by OAuth discovery."""
    from agent.auxiliary_client import _try_anthropic

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = SimpleNamespace(
        access_token="explicit-api-key",
        runtime_api_key="",
        source="env:ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        runtime_base_url="https://api.anthropic.com",
        auth_type="api_key",
    )

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            side_effect=AssertionError("API key must skip OAuth resolution"),
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as mock_build,
    ):
        client, _model = _try_anthropic()

    assert client is not None
    assert mock_build.call_args.args[0] == "explicit-api-key"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.services.ai.azure.com/anthropic",
        "https://proxy.api.anthropic.com/anthropic",
    ],
)
def test_auxiliary_custom_endpoint_keeps_endpoint_bound_pool_token(
    tmp_path, monkeypatch, endpoint
):
    """OAuth discovery must not replace credentials bound to a custom endpoint."""
    from agent.auxiliary_client import _try_anthropic

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = SimpleNamespace(
        access_token="endpoint-bound-token",
        runtime_api_key="",
        source="env:ANTHROPIC_TOKEN",
        base_url=endpoint,
        runtime_base_url=endpoint,
        auth_type="oauth",
    )

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            side_effect=AssertionError(
                "custom endpoint must keep its bound credential"
            ),
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ) as mock_build,
    ):
        client, _model = _try_anthropic()

    assert client is not None
    assert mock_build.call_args.args[0] == "endpoint-bound-token"
    assert mock_build.call_args.args[1] == endpoint


def test_auxiliary_scope_error_remains_fail_closed(tmp_path, monkeypatch):
    """Auxiliary resolution must not swallow a missing multiplex secret scope."""
    from agent.auxiliary_client import _try_anthropic
    from agent.secret_scope import UnscopedSecretError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = SimpleNamespace(
        access_token="stale-env-token",
        runtime_api_key="",
        source="env:ANTHROPIC_TOKEN",
        base_url="https://api.anthropic.com",
        runtime_base_url="https://api.anthropic.com",
        auth_type="oauth",
    )

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            side_effect=UnscopedSecretError("scope required"),
        ),
    ):
        with pytest.raises(UnscopedSecretError, match="scope required"):
            _try_anthropic()


def test_auxiliary_without_canonical_token_rejects_env_pool_entry(
    tmp_path, monkeypatch
):
    """Missing canonical auth makes the env-backed auxiliary route unavailable."""
    from agent.auxiliary_client import _try_anthropic

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = SimpleNamespace(
        access_token="stale-env-token",
        runtime_api_key="",
        source="env:ANTHROPIC_TOKEN",
        base_url="https://api.anthropic.com",
        runtime_base_url="https://api.anthropic.com",
        auth_type="oauth",
    )

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry",
            return_value=(True, entry),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value=None,
        ),
        patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
    ):
        client, model = _try_anthropic()

    assert client is None
    assert model is None
    mock_build.assert_not_called()
