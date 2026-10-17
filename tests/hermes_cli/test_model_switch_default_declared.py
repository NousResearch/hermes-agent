"""Regression tests for #138261: a model named by ``model.default`` must pass the switch
validator's config-declaration override even when no live/curated listing carries the id.

The reporter's Fireworks account-level router id serves fine on the inference endpoint and is
rendered as the current selection, yet ``/api/model/set`` (and ``/model``) rejected it with
``not found in this provider's model listing`` — ``_config_declares_model()`` consulted only
``providers:``/``custom_providers:``, never the most explicit declaration there is.
"""

from unittest.mock import patch

import hermes_yaml as yaml

from hermes_cli.model_switch import switch_model

_ROUTER_ID = "accounts/fireworks/routers/deepseek-v4p1-flash-us"
_BASE_URL = "https://api.fireworks.ai/inference/v1"

_REJECTED_VALIDATION = {
    "accepted": False, "persist": False, "recognized": False,
    "message": f"Model `{_ROUTER_ID}` was not found in this provider's model listing.",
}


def _write_config(tmp_path, model_cfg):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": model_cfg}))


def _switch_to(model_id, provider):
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, p: model), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=_REJECTED_VALIDATION), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"api_key": "***", "base_url": _BASE_URL, "api_mode": "chat_completions"}):
        return switch_model(
            raw_input=model_id, current_provider=provider, current_model="old-model",
            current_base_url=_BASE_URL, user_providers={}, custom_providers=[],
        )


def test_model_default_named_model_passes_listing_rejection(monkeypatch, tmp_path):
    # The router id sits in model.default with its provider — the declaration must win over the
    # rejected live-listing validation (#138261).
    _write_config(tmp_path, {"default": _ROUTER_ID, "provider": "fireworks"})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    result = _switch_to(_ROUTER_ID, "fireworks")
    assert result.success is True
    assert result.error_message == ""


def test_model_default_is_scoped_to_its_provider(monkeypatch, tmp_path):
    # The declaration must not widen: the same id picked on an unrelated provider stays rejected.
    _write_config(tmp_path, {"default": _ROUTER_ID, "provider": "fireworks"})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    result = _switch_to(_ROUTER_ID, "openai")
    assert result.success is False
    assert "was not found in this provider's model listing" in result.error_message


def test_model_default_without_provider_pair_stays_rejected(monkeypatch, tmp_path):
    # A bare model.default routes via the resolver rather than declaring the id for one provider;
    # the validator stays strict there so the override is not simply weakened.
    _write_config(tmp_path, {"default": _ROUTER_ID})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    result = _switch_to(_ROUTER_ID, "fireworks")
    assert result.success is False
