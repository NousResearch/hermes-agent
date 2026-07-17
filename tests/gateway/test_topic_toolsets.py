"""Tests for per-topic toolset overrides (platforms.<name>.extra.topic_toolsets)."""

from gateway.run import _get_topic_toolsets_override


def _config(topic_toolsets=None, extra=None, platform="telegram"):
    if extra is None:
        extra = {}
        if topic_toolsets is not None:
            extra["topic_toolsets"] = topic_toolsets
    return {"platforms": {platform: {"enabled": True, "extra": extra}}}


class TestGetTopicToolsetsOverride:
    def test_none_when_no_thread_id(self):
        cfg = _config({"137": ["telegram", "skills"]})
        assert _get_topic_toolsets_override(cfg, "telegram", None) is None
        assert _get_topic_toolsets_override(cfg, "telegram", "") is None

    def test_none_when_platform_missing(self):
        cfg = _config({"137": ["telegram"]})
        assert _get_topic_toolsets_override(cfg, "discord", "137") is None

    def test_none_when_no_platforms_section(self):
        assert _get_topic_toolsets_override({}, "telegram", "137") is None

    def test_none_when_extra_missing_or_wrong_type(self):
        cfg = {"platforms": {"telegram": {"enabled": True}}}
        assert _get_topic_toolsets_override(cfg, "telegram", "137") is None
        cfg = {"platforms": {"telegram": {"enabled": True, "extra": "nope"}}}
        assert _get_topic_toolsets_override(cfg, "telegram", "137") is None

    def test_none_when_topic_not_listed(self):
        cfg = _config({"137": ["telegram"]})
        assert _get_topic_toolsets_override(cfg, "telegram", "999") is None

    def test_returns_sorted_deduplicated_list(self):
        cfg = _config({"137": ["web_search", "telegram", "web_search"]})
        assert _get_topic_toolsets_override(cfg, "telegram", "137") == [
            "telegram",
            "web_search",
        ]

    def test_matches_integer_keys_from_yaml(self):
        """YAML users often write bare integer topic ids."""
        cfg = _config({137: ["files"]})
        assert _get_topic_toolsets_override(cfg, "telegram", "137") == ["files"]

    def test_string_key_takes_precedence_over_int_key(self):
        cfg = _config({"137": ["files"], 137: ["web_search"]})
        assert _get_topic_toolsets_override(cfg, "telegram", "137") == ["files"]

    def test_empty_list_means_no_override(self):
        """An empty per-topic list is "not configured", not "disable all"."""
        cfg = _config({"137": []})
        assert _get_topic_toolsets_override(cfg, "telegram", "137") is None

    def test_non_list_value_is_ignored(self):
        cfg = _config({"137": "telegram"})
        assert _get_topic_toolsets_override(cfg, "telegram", "137") is None

    def test_non_dict_topic_toolsets_is_ignored(self):
        cfg = _config(extra={"topic_toolsets": ["telegram"]})
        assert _get_topic_toolsets_override(cfg, "telegram", "137") is None

    def test_values_coerced_to_str(self):
        cfg = _config({"137": ["telegram", 42]})
        assert _get_topic_toolsets_override(cfg, "telegram", "137") == [
            "42",
            "telegram",
        ]
