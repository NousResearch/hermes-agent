"""Regression tests for issue #56810.

When an auxiliary task is configured to use a non-API-key provider (e.g.
``vertex``, which uses GCP Application Default Credentials) and the
client fails to build, the failure path must surface an actionable error
message that points the user at the *actual* auth mechanism — not a
made-up ``VERTEX_API_KEY`` env var.

These tests cover the sync ``call_llm`` path and the async
``async_call_llm`` path, plus a control case for genuine API-key
providers (must keep the existing helpful message).
"""

import pytest
from types import SimpleNamespace
from unittest.mock import patch


# A representative cross-section of PROVIDER_REGISTRY auth_types —
# the existing auxiliary_client.py fall-through covers all of them with
# the same misleading RuntimeError, so the test matrix is auth_type
# × call_llm / async_call_llm.
_NON_API_KEY_AUTHTYPES = [
    # (provider_id, auth_type, must_mention_substr)
    ("vertex", "vertex", "Application Default Credentials"),
    # AWS SDK auth — must mention AWS credentials, not "AWS_API_KEY"
    ("bedrock", "aws_sdk", "AWS credentials"),
]


_API_KEY_AUTHTYPES = [
    # (provider_id, expected_env_var_substr)
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
        "VERTEX_API_KEY", "BEDROCK_API_KEY", "AWS_ACCESS_KEY_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    import agent.auxiliary_client as _aux_mod
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()
    yield
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()


def _pconfig(provider_id: str, auth_type: str):
    """Build a stand-in PROVIDER_REGISTRY entry with the given auth_type."""
    return SimpleNamespace(provider_id=provider_id, auth_type=auth_type)


def _fake_registry(mapping):
    """Return a fake PROVIDER_REGISTRY dict usable with monkeypatch."""
    return {pid: _pconfig(pid, atype) for pid, atype in mapping.items()}


def _patch_registry(monkeypatch, registry):
    """Install a fake PROVIDER_REGISTRY both in hermes_cli.auth and the
    auxiliary_client module-level reference (whichever it uses)."""
    import sys
    fake_module = SimpleNamespace(PROVIDER_REGISTRY=registry)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_module)
    # Also patch the module's name in case it imported the symbol directly.
    from agent import auxiliary_client as aux
    if hasattr(aux, "PROVIDER_REGISTRY"):
        monkeypatch.setattr(aux, "PROVIDER_REGISTRY", registry, raising=False)
    return aux


class TestVertexAuxClientErrorMessage:
    """Bug #56810: vertex provider raising VERTEX_API_KEY is misleading."""

    @pytest.mark.parametrize(
        "provider_id,auth_type,expected_substr",
        _NON_API_KEY_AUTHTYPES,
    )
    def test_sync_call_llm_vertex_raises_actionable_error(
        self, monkeypatch, provider_id, auth_type, expected_substr,
    ):
        """Sync path: when client is None and no fallback exists, error
        must NOT mention a fabricated env var. Must mention the real auth."""
        registry = _fake_registry({provider_id: auth_type})
        aux = _patch_registry(monkeypatch, registry)

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(None, None),
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
            return_value=(None, None, ""),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=(provider_id, f"{provider_id}/some-model",
                          None, None, None),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                aux.call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                )

        msg = str(exc_info.value)
        # Must NOT claim a fake env var
        assert f"{provider_id.upper()}_API_KEY" not in msg, (
            f"Misleading message still references "
            f"{provider_id.upper()}_API_KEY: {msg}"
        )
        # Must mention the actual auth mechanism
        assert expected_substr in msg, (
            f"Expected actionable substring {expected_substr!r} in error: {msg}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider_id,auth_type,expected_substr",
        _NON_API_KEY_AUTHTYPES,
    )
    async def test_async_call_llm_vertex_raises_actionable_error(
        self, monkeypatch, provider_id, auth_type, expected_substr,
    ):
        """Async path: same bug must be fixed."""
        registry = _fake_registry({provider_id: auth_type})
        aux = _patch_registry(monkeypatch, registry)

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(None, None),
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
            return_value=(None, None, ""),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=(provider_id, f"{provider_id}/some-model",
                          None, None, None),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await aux.async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                )

        msg = str(exc_info.value)
        assert f"{provider_id.upper()}_API_KEY" not in msg, (
            f"Misleading message still references "
            f"{provider_id.upper()}_API_KEY: {msg}"
        )
        assert expected_substr in msg, (
            f"Expected actionable substring {expected_substr!r} in error: {msg}"
        )


class TestApiKeyProviderStillHelpful:
    """Regression guard: genuine API-key providers must keep their
    existing actionable 'Set X_API_KEY' message."""

    @pytest.mark.parametrize(
        "provider_id,env_substr", _API_KEY_AUTHTYPES,
    )
    def test_sync_call_llm_api_key_provider_message_preserved(
        self, monkeypatch, provider_id, env_substr,
    ):
        registry = _fake_registry({provider_id: "api_key"})
        aux = _patch_registry(monkeypatch, registry)

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(None, None),
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
            return_value=(None, None, ""),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=(provider_id, f"{provider_id}/some-model",
                          None, None, None),
        ):
            with pytest.raises(RuntimeError, match=env_substr):
                aux.call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                )


class TestUnknownAuthTypeStillActionable:
    """Unknown auth_types must NOT lie about API keys."""

    def test_sync_call_llm_unknown_auth_type_message(self, monkeypatch):
        registry = _fake_registry(
            {"custom-future-provider": "exotic_quantum_auth"}
        )
        aux = _patch_registry(monkeypatch, registry)

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(None, None),
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
            return_value=(None, None, ""),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom-future-provider", "model/x",
                          None, None, None),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                aux.call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                )

        msg = str(exc_info.value)
        # Must NOT mention a fake API_KEY env var
        assert "CUSTOM-FUTURE-PROVIDER_API_KEY" not in msg
        # Must mention the actual provider and ask user to switch or check auth
        assert "custom-future-provider" in msg
        assert "hermes model" in msg


class TestFallbackChainStillWorks:
    """Regression guard: when the configured fallback chain returns a
    valid client, the misleading error must NOT fire — we should use
    the fallback. (Existing behaviour preserved.)"""

    def test_sync_call_llm_fallback_used_when_available(self, monkeypatch):
        registry = _fake_registry({"vertex": "vertex"})
        aux = _patch_registry(monkeypatch, registry)

        fallback_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(
                            message=SimpleNamespace(content="from fallback"),
                        )],
                        id="fb-1",
                    )
                )
            )
        )

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(None, None),
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
            return_value=(fallback_client, "vertex-model", "vertex"),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("vertex", "vertex-model", None, None, None),
        ), patch(
            "agent.auxiliary_client._validate_llm_response",
            side_effect=lambda r, _task: r,
        ):
            result = aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result.choices[0].message.content == "from fallback"