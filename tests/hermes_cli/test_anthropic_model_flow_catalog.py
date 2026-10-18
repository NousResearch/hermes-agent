"""Anthropic setup must offer live-only models and retain its outage fallback."""
from unittest.mock import patch

import pytest

LIVE_IDS = ["claude-sonnet-5-5", "claude-haiku-5-5"]


@pytest.fixture()
def hermetic_home(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "anthropic-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def run_flow(live_ids):
    from hermes_cli import model_setup_flows as flows

    captured = {}

    def pick(rows, prompt, **kwargs):
        captured["rows"] = rows
        return None

    with patch("hermes_cli.auth.get_anthropic_key", return_value="test-api-key"), \
         patch("agent.anthropic_credentials.read_claude_code_credentials", return_value=None), \
         patch.object(flows, "_prompt_auth_credentials_choice", return_value="use"), \
         patch("hermes_cli.models._fetch_anthropic_models", return_value=live_ids), \
         patch.object(flows, "_pick_model_or_prompt", side_effect=pick), \
         patch.object(flows, "_finish_model") as finish:
        flows._model_flow_anthropic({})
    return captured["rows"], finish


def test_live_models_reach_wizard(hermetic_home):
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS

    rows, _ = run_flow(LIVE_IDS)
    assert set(LIVE_IDS).issubset(rows)
    assert set(_PROVIDER_MODELS["anthropic"]).issubset(rows)
    assert len(rows) == len(set(rows))


def test_wizard_matches_shared_catalog(hermetic_home):
    from hermes_cli.models import provider_model_ids

    rows, _ = run_flow(LIVE_IDS)
    with patch("hermes_cli.models._fetch_anthropic_models", return_value=LIVE_IDS):
        assert rows == provider_model_ids("anthropic")


@pytest.mark.parametrize("live_ids", [None, []])
def test_discovery_failure_keeps_fallback(hermetic_home, live_ids):
    from hermes_cli.models_catalog_static import CuratedFallbackModels, _PROVIDER_MODELS

    rows, _ = run_flow(live_ids)
    assert rows == _PROVIDER_MODELS["anthropic"]
    assert isinstance(rows, CuratedFallbackModels)


def test_finish_contract_unchanged(hermetic_home):
    _, finish = run_flow(LIVE_IDS)
    finish.assert_called_once()
    assert finish.call_args.args[1] == "anthropic"
    assert finish.call_args.kwargs == {"drop_base_url": True, "drop_api_mode": True}
