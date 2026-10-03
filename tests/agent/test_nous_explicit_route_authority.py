"""Explicit Nous auxiliary/fallback routes keep their configured account and origin.

These paths already pass ``explicit_api_key`` and ``explicit_base_url`` into the
central provider router.  The Nous-specific branch must not replace that pair
with whichever Portal account happens to be active in the ambient auth store.
All credentials and endpoints in this module are inert test values.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


_SELECTED_KEY = "test-selected-nous-key"
_SELECTED_URL = "https://selected-nous.invalid/v1"
_SELECTED_MODEL = "google/gemini-3.6-flash"


def test_auxiliary_nous_fallback_keeps_explicit_credential_and_origin():
    """A task fallback pin is authoritative; resolution never consults ambient Nous auth."""
    from agent import auxiliary_client as aux

    fallback = {
        "provider": "nous",
        "model": _SELECTED_MODEL,
        "base_url": _SELECTED_URL,
        "api_key": _SELECTED_KEY,
    }
    with (
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": [fallback]}),
        patch.object(
            aux,
            "_read_nous_auth",
            side_effect=AssertionError("explicit route must not read the ambient Nous account"),
        ),
        patch.object(
            aux,
            "_resolve_nous_runtime_api",
            side_effect=AssertionError("explicit route must not resolve ambient Nous credentials"),
        ),
        aux.aux_probe_mode(),
    ):
        client, model, label = aux._try_configured_fallback_chain(
            "compression", "openrouter", reason="test primary unavailable"
        )

    assert client is not None
    assert client.api_key == _SELECTED_KEY
    assert client.base_url == _SELECTED_URL
    assert model == _SELECTED_MODEL
    assert label == "fallback_chain[0](nous)"


def test_main_fallback_accepts_explicit_nous_route_without_ambient_auth():
    """An inline fallback key makes the route reachable and remains bound after activation."""
    from agent import auxiliary_client as aux

    fallback = {
        "provider": "nous",
        "model": _SELECTED_MODEL,
        "base_url": _SELECTED_URL,
        "api_key": _SELECTED_KEY,
    }
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-primary-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test/primary",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[fallback],
        )
    agent.client = MagicMock()
    agent.context_compressor = None
    agent._entitlement_rejected_models = {("nous", _SELECTED_MODEL)}

    ambient_pool = MagicMock(provider="nous")
    ambient_pool.has_credentials.return_value = True
    ambient_pool.has_available.return_value = False
    ambient_pool.next_available_at.return_value = None

    def _fake_client(*, api_key, base_url, **_kwargs):
        return SimpleNamespace(api_key=api_key, base_url=base_url, _custom_headers={})

    with (
        patch(
            "hermes_cli.auth.get_provider_auth_state",
            side_effect=AssertionError("explicit fallback must not require ambient Nous auth"),
        ) as ambient_auth,
        patch.object(
            aux,
            "_read_nous_auth",
            side_effect=AssertionError("explicit fallback must not read the ambient Nous account"),
        ),
        patch.object(
            aux,
            "_resolve_nous_runtime_api",
            side_effect=AssertionError("explicit fallback must not resolve ambient Nous credentials"),
        ),
        patch.object(aux, "_create_openai_client", side_effect=_fake_client),
        patch("hermes_cli.models.get_nous_recommended_aux_model", return_value=_SELECTED_MODEL),
        patch("agent.credential_pool.load_pool", return_value=ambient_pool),
    ):
        assert agent._try_activate_fallback() is True

    ambient_auth.assert_not_called()
    assert agent.provider == "nous"
    assert agent.api_key == _SELECTED_KEY
    assert agent.base_url == _SELECTED_URL
    assert agent._client_kwargs == {"api_key": _SELECTED_KEY, "base_url": _SELECTED_URL}
    assert agent._credential_pool is None
