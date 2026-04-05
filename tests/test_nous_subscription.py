"""Tests for Nous subscription feature state management.

Tests NousFeatureState dataclass properties, subscription feature resolution,
and integration with auth/config backends.
"""

from unittest.mock import MagicMock, patch
import pytest

from hermes_cli.nous_subscription import (
    NousFeatureState,
    NousSubscriptionFeatures,
    _model_config_dict,
)


class TestNousFeatureState:
    """Test NousFeatureState dataclass."""

    def test_feature_state_creation(self):
        """Feature state should be immutable and serialize."""
        state = NousFeatureState(
            key="web",
            label="Web Search",
            included_by_default=True,
            available=True,
            active=True,
            managed_by_nous=False,
            direct_override=False,
            toolset_enabled=True,
        )
        assert state.key == "web"
        assert state.active is True

    def test_feature_state_frozen(self):
        """Feature state should be immutable."""
        state = NousFeatureState(
            key="web",
            label="Web Search",
            included_by_default=True,
            available=True,
            active=True,
            managed_by_nous=False,
            direct_override=False,
            toolset_enabled=True,
        )
        with pytest.raises(AttributeError):
            state.active = False


class TestNousSubscriptionFeatures:
    """Test NousSubscriptionFeatures aggregation."""

    def _make_features(self) -> dict:
        """Factory for feature state dict."""
        return {
            "web": NousFeatureState(
                key="web", label="Web", included_by_default=True,
                available=True, active=True, managed_by_nous=False,
                direct_override=False, toolset_enabled=True,
            ),
            "image_gen": NousFeatureState(
                key="image_gen", label="Image Generation", included_by_default=True,
                available=False, active=False, managed_by_nous=False,
                direct_override=False, toolset_enabled=False,
            ),
            "tts": NousFeatureState(
                key="tts", label="Text-to-Speech", included_by_default=True,
                available=True, active=False, managed_by_nous=True,
                direct_override=False, toolset_enabled=True,
            ),
            "browser": NousFeatureState(
                key="browser", label="Browser", included_by_default=False,
                available=True, active=True, managed_by_nous=False,
                direct_override=True, toolset_enabled=True,
            ),
            "modal": NousFeatureState(
                key="modal", label="Modal", included_by_default=False,
                available=True, active=False, managed_by_nous=False,
                direct_override=False, toolset_enabled=False,
            ),
        }

    def test_subscription_creation(self):
        """Subscription should aggregate features."""
        features_dict = self._make_features()
        sub = NousSubscriptionFeatures(
            subscribed=True,
            nous_auth_present=True,
            provider_is_nous=True,
            features=features_dict,
        )
        assert sub.subscribed is True
        assert sub.web.active is True

    def test_subscription_property_access(self):
        """Should expose features via properties."""
        features_dict = self._make_features()
        sub = NousSubscriptionFeatures(
            subscribed=True,
            nous_auth_present=True,
            provider_is_nous=True,
            features=features_dict,
        )
        assert sub.web.key == "web"
        assert sub.image_gen.available is False
        assert sub.tts.managed_by_nous is True
        assert sub.browser.direct_override is True

    def test_subscription_items_ordering(self):
        """items() should return features in canonical order."""
        features_dict = self._make_features()
        sub = NousSubscriptionFeatures(
            subscribed=False,
            nous_auth_present=False,
            provider_is_nous=False,
            features=features_dict,
        )
        ordered_keys = [item.key for item in sub.items()]
        assert ordered_keys == ["web", "image_gen", "tts", "browser", "modal"]

    def test_unsubscribed_state(self):
        """Unsubscribed should be representable."""
        empty_features = {
            "web": NousFeatureState(
                key="web", label="Web", included_by_default=False,
                available=False, active=False, managed_by_nous=False,
                direct_override=False, toolset_enabled=False,
            ),
            "image_gen": NousFeatureState(
                key="image_gen", label="Image Gen", included_by_default=False,
                available=False, active=False, managed_by_nous=False,
                direct_override=False, toolset_enabled=False,
            ),
            "tts": NousFeatureState(
                key="tts", label="TTS", included_by_default=False,
                available=False, active=False, managed_by_nous=False,
                direct_override=False, toolset_enabled=False,
            ),
            "browser": NousFeatureState(
                key="browser", label="Browser", included_by_default=False,
                available=False, active=False, managed_by_nous=False,
                direct_override=False, toolset_enabled=False,
            ),
            "modal": NousFeatureState(
                key="modal", label="Modal", included_by_default=False,
                available=False, active=False, managed_by_nous=False,
                direct_override=False, toolset_enabled=False,
            ),
        }
        sub = NousSubscriptionFeatures(
            subscribed=False,
            nous_auth_present=False,
            provider_is_nous=False,
            features=empty_features,
        )
        assert sub.subscribed is False
        assert all(not item.available for item in sub.items())


class TestModelConfigDict:
    """Test model config parsing helper."""

    def test_model_config_dict_empty(self):
        """Empty config should return empty dict."""
        result = _model_config_dict({})
        assert result == {}

    def test_model_config_dict_missing_model_key(self):
        """Missing 'model' key should return empty dict."""
        result = _model_config_dict({"other_key": "value"})
        assert result == {}

    def test_model_config_dict_from_dict(self):
        """Dict model config should pass through."""
        config = {"model": {"default": "claude-3", "gpt": "gpt-4"}}
        result = _model_config_dict(config)
        assert result == {"default": "claude-3", "gpt": "gpt-4"}

    def test_model_config_dict_from_string(self):
        """String model config should wrap in default."""
        config = {"model": "claude-3"}
        result = _model_config_dict(config)
        assert result == {"default": "claude-3"}

    def test_model_config_dict_from_empty_string(self):
        """Empty string should return empty dict."""
        config = {"model": ""}
        result = _model_config_dict(config)
        assert result == {}

    def test_model_config_dict_from_whitespace_string(self):
        """Whitespace-only string should return empty dict."""
        config = {"model": "   "}
        result = _model_config_dict(config)
        assert result == {}

    def test_model_config_dict_from_invalid_type(self):
        """Non-dict, non-string types should return empty dict."""
        result = _model_config_dict({"model": 123})
        assert result == {}

        result = _model_config_dict({"model": ["a", "b"]})
        assert result == {}

    def test_model_config_dict_preserves_dict_mutations(self):
        """Should return a copy, not mutate original."""
        original = {"model": {"default": "claude-3"}}
        result = _model_config_dict(original)
        result["new_key"] = "new_value"
        assert "new_key" not in original.get("model", {})
