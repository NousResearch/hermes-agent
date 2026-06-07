"""Tests for channel_models per-route model overlay resolution."""

from gateway.platforms.base import resolve_channel_model


class TestResolveChannelModel:
    def test_no_models_returns_none(self):
        assert resolve_channel_model({}, "123") is None

    def test_match_by_channel_id(self):
        extra = {"channel_models": {"100": "anthropic/claude-opus-4-8"}}
        assert resolve_channel_model(extra, "100") == "anthropic/claude-opus-4-8"

    def test_match_by_parent_id(self):
        extra = {"channel_models": {"200": "openai/gpt-5"}}
        assert resolve_channel_model(extra, "999", parent_id="200") == "openai/gpt-5"

    def test_channel_id_wins_over_parent(self):
        extra = {"channel_models": {"100": "for-channel", "200": "for-parent"}}
        assert resolve_channel_model(extra, "100", parent_id="200") == "for-channel"

    def test_blank_is_absent(self):
        assert resolve_channel_model({"channel_models": {"100": "  "}}, "100") is None

    def test_non_dict_returns_none(self):
        assert resolve_channel_model({"channel_models": ["x"]}, "100") is None
