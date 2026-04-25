"""Unit tests for _resolve_gateway_model() with model.gateway override."""

from gateway.run import _resolve_gateway_model


class TestResolveGatewayModel:
    """_resolve_gateway_model reads model from config.yaml dict."""

    def test_gateway_override_takes_priority(self):
        """model.gateway should be returned when set."""
        result = _resolve_gateway_model({
            "model": {
                "default": "cli-model",
                "gateway": "gateway-model",
            }
        })
        assert result == "gateway-model"

    def test_falls_back_to_default_when_no_gateway(self):
        """Without model.gateway, falls back to model.default."""
        result = _resolve_gateway_model({
            "model": {
                "default": "shared-model",
            }
        })
        assert result == "shared-model"

    def test_falls_back_to_legacy_model_key(self):
        """Legacy 'model' key inside model dict still works."""
        result = _resolve_gateway_model({
            "model": {
                "model": "legacy-model",
            }
        })
        assert result == "legacy-model"

    def test_empty_gateway_falls_through(self):
        """Empty string gateway should fall back to default."""
        result = _resolve_gateway_model({
            "model": {
                "default": "cli-model",
                "gateway": "",
            }
        })
        # Empty string is falsy, so falls through to default
        assert result == "cli-model"

    def test_string_model_passthrough(self):
        """Old format: model is a plain string."""
        result = _resolve_gateway_model({"model": "plain-string-model"})
        assert result == "plain-string-model"

    def test_missing_model_returns_empty(self):
        """No model key at all returns empty string."""
        result = _resolve_gateway_model({})
        assert result == ""

    def test_none_config_returns_empty(self):
        """None config (load failure) returns empty string."""
        result = _resolve_gateway_model(None)
        assert result == ""
