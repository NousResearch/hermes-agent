"""Regression tests for #138261: a model named by ``model.default`` is a config declaration.

``_config_declares_model`` is the model-switch escape hatch that accepts a model the
remote /v1/models listing does not know. It consulted ``providers:`` and
``custom_providers:`` but never ``model.default`` - the most explicit declaration a
user can make - so the configured default model was rejected by the switch validator
with "not found in this provider's model listing".
"""

from unittest.mock import patch

from hermes_cli.model_switch import _config_declares_model

ROUTER_ID = "accounts/fireworks/routers/deepseek-v4p1-flash-us"

_CONFIG = {"model": {"provider": "fireworks", "default": ROUTER_ID}}


def _declares(new_model: str, target_provider: str, config: dict) -> bool:
    with patch("hermes_cli.config.load_config", return_value=config):
        return _config_declares_model(new_model, target_provider, "", None, None)


class TestModelDefaultDeclaration:
    def test_default_model_accepted_for_its_provider(self):
        assert _declares(ROUTER_ID, "fireworks", _CONFIG) is True

    def test_default_model_rejected_for_a_different_provider(self):
        assert _declares(ROUTER_ID, "openrouter", _CONFIG) is False

    def test_no_default_configured_keeps_old_behavior(self):
        assert _declares(ROUTER_ID, "fireworks", {}) is False

    def test_dict_form_default_accepted(self):
        config = {"model": {"default": {"model": ROUTER_ID, "provider": "fireworks"}}}
        assert _declares(ROUTER_ID, "fireworks", config) is True

    def test_other_model_still_rejected(self):
        assert (
            _declares("accounts/fireworks/models/other-model", "fireworks", _CONFIG)
            is False
        )


    def test_string_default_without_provider_is_no_wildcard(self):
        # A bare-string default with no model.provider sibling has no provider
        # scope; it must not act as a wildcard accept for an unscoped target.
        config = {"model": {"default": ROUTER_ID}}
        assert _declares(ROUTER_ID, "", config) is False
