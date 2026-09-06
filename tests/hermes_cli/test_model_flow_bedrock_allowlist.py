"""The interactive ``hermes model`` Bedrock flow keeps the allowlist's visibility postcondition.

``discover_bedrock_models`` honors ``bedrock.discovery.model_allowlist``, so an empty live
result is usually the allowlist matching nothing (or an allowlist in force while credentials are
down). Neither wizard branch may then offer the whole curated table: it names the very ids the
allowlist exists to hide. With the allowlist unset, the historical curated fallback is unchanged
(positive control). The curated table is replaced by a synthetic one so no test pins a real id.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.bedrock_adapter import BEDROCK_OPENAI_RESPONSES_MODEL_IDS
from hermes_cli import model_setup_flows_bedrock as flow_mod

_CURATED = ["us.keep-me", "openai.hide-me"]  # deliberately carries no Mantle id
_MANTLE_ID = BEDROCK_OPENAI_RESPONSES_MODEL_IDS[0]


def _config(allowlist):
    return {"bedrock": {"discovery": {"model_allowlist": allowlist}}}


@pytest.fixture
def flow_patches():
    """Drive the wizard headless: no prints, AWS present, a synthetic curated table."""
    with patch.object(flow_mod, "_say"), \
         patch("hermes_cli.models._PROVIDER_MODELS", {"bedrock": list(_CURATED)}), \
         patch("agent.bedrock_adapter.has_aws_credentials", return_value=True), \
         patch("agent.bedrock_adapter.resolve_aws_auth_env_var",
               return_value="AWS_ACCESS_KEY_ID"), \
         patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="us-east-1"):
        yield


def _offered(auth_choice: str, config: dict) -> list:
    """Run ``_model_flow_bedrock`` to the picker and return the model list it was offered.

    ``_ask`` answers: region ("" keeps the resolved one), then *auth_choice* ("1" = IAM chain,
    whose live discovery is stubbed empty; "2" = Bedrock API key, which never runs discovery).
    """
    calls: dict = {}

    def _pick(model_list, prompt, **kwargs):
        calls["model_list"] = list(model_list)
        return model_list[0]

    with patch.object(flow_mod, "_ask", side_effect=["", auth_choice]), \
         patch("agent.bedrock_adapter.discover_bedrock_models", return_value=[]), \
         patch("hermes_cli.auth._resolve_api_key_provider_secret", return_value=("key", "env")), \
         patch.object(flow_mod, "_pick_model_or_prompt", side_effect=_pick), \
         patch.object(flow_mod, "_finish_model"), \
         patch("hermes_cli.config.load_config_readonly", return_value=config):
        flow_mod._model_flow_bedrock({})
    return calls.get("model_list", [])


def test_curated_fallback_is_projected_through_the_allowlist(flow_patches):
    """Both branches offer the curated table only as its allowlisted projection (case-insensitive),
    over the same pool ``models_bedrock._bedrock_catalog`` uses: the curated ids plus the Mantle
    ids the control plane never lists, so a Mantle-only allowlist offers the Mantle id even when
    the curated table happens not to carry it. Unset allowlist: the table is offered intact."""
    assert _offered("1", _config(["US.Keep-Me"])) == ["us.keep-me"]
    assert _offered("2", _config(["US.Keep-Me"])) == ["us.keep-me"]

    assert _offered("1", _config([_MANTLE_ID.upper()])) == [_MANTLE_ID]
    assert _offered("2", _config([_MANTLE_ID.upper()])) == [_MANTLE_ID]

    assert _offered("1", _config([])) == _CURATED  # positive control
    assert _offered("2", _config([])) == _CURATED


def test_zero_match_stops_before_the_picker(flow_patches, capsys):
    """An allowlist matching nothing stops both branches with a message instead of falling back
    to the unfiltered curated list (IAM branch: live discovery is empty because the allowlist
    matched nothing; API-key branch: the curated table has no allowlisted id)."""
    config = _config(["nothing.matches-this"])
    for auth_choice in ("1", "2"):
        with patch.object(flow_mod, "_ask", side_effect=["", auth_choice]), \
             patch("agent.bedrock_adapter.discover_bedrock_models", return_value=[]) as discover, \
             patch("hermes_cli.auth._resolve_api_key_provider_secret", return_value=("key", "env")), \
             patch.object(flow_mod, "_pick_model_or_prompt") as pick, \
             patch("hermes_cli.config.load_config_readonly", return_value=config):
            assert flow_mod._model_flow_bedrock({}) is None, auth_choice
        pick.assert_not_called()
        assert "No models match bedrock.discovery.model_allowlist" in capsys.readouterr().out
        if auth_choice == "1":
            discover.assert_called_once_with("us-east-1")
        else:
            discover.assert_not_called()
