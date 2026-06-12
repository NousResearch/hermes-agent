"""Unit tests for the declarative system-prompt fragment override engine
(agent/prompt_overrides.py)."""

from agent.prompt_overrides import (
    FRAGMENT_KEYS,
    VALID_MODES,
    apply_fragment_override,
    normalize_overrides,
)


class TestNormalizeOverrides:
    def test_empty_and_none(self):
        assert normalize_overrides(None) == {}
        assert normalize_overrides({}) == {}

    def test_non_mapping_ignored(self):
        assert normalize_overrides("nope") == {}
        assert normalize_overrides(["a", "b"]) == {}

    def test_bare_string_is_replace_shorthand(self):
        out = normalize_overrides({"steer_channel": "short note"})
        assert out == {"steer_channel": {"mode": "replace", "text": "short note"}}

    def test_full_spec(self):
        out = normalize_overrides(
            {"task_completion": {"mode": "append", "text": "more"}}
        )
        assert out == {"task_completion": {"mode": "append", "text": "more"}}

    def test_remove_drops_text(self):
        out = normalize_overrides({"google_operational": {"mode": "remove"}})
        assert out == {"google_operational": {"mode": "remove", "text": ""}}

    def test_unknown_key_dropped(self):
        assert normalize_overrides({"bogus": "x"}) == {}

    def test_invalid_mode_dropped(self):
        assert normalize_overrides({"identity": {"mode": "sideways", "text": "x"}}) == {}

    def test_mode_case_insensitive(self):
        out = normalize_overrides({"identity": {"mode": "REPLACE", "text": "x"}})
        assert out["identity"]["mode"] == "replace"

    def test_missing_mode_defaults_replace(self):
        out = normalize_overrides({"identity": {"text": "x"}})
        assert out["identity"]["mode"] == "replace"

    def test_append_empty_text_is_noop_dropped(self):
        assert normalize_overrides({"identity": {"mode": "append", "text": "  "}}) == {}

    def test_replace_empty_text_allowed(self):
        # replace with "" is a legitimate way to blank a fragment via text
        out = normalize_overrides({"identity": {"mode": "replace", "text": ""}})
        assert out == {"identity": {"mode": "replace", "text": ""}}

    def test_non_string_text_dropped(self):
        assert normalize_overrides({"identity": {"mode": "replace", "text": 5}}) == {}

    def test_non_dict_non_str_spec_dropped(self):
        assert normalize_overrides({"identity": ["a"]}) == {}

    def test_mixed_valid_and_invalid(self):
        out = normalize_overrides(
            {
                "task_completion": "keep this",
                "bogus": "drop this",
                "google_operational": {"mode": "remove"},
            }
        )
        assert set(out) == {"task_completion", "google_operational"}


class TestApplyFragmentOverride:
    def test_no_overrides_passthrough(self):
        assert apply_fragment_override(None, "identity", "ID") == "ID"
        assert apply_fragment_override({}, "identity", "ID") == "ID"

    def test_unmatched_key_passthrough(self):
        ov = normalize_overrides({"identity": "X"})
        assert apply_fragment_override(ov, "task_completion", "ORIG") == "ORIG"

    def test_replace(self):
        ov = normalize_overrides({"identity": {"mode": "replace", "text": "NEW"}})
        assert apply_fragment_override(ov, "identity", "OLD") == "NEW"

    def test_remove_returns_none(self):
        ov = normalize_overrides({"identity": {"mode": "remove"}})
        assert apply_fragment_override(ov, "identity", "OLD") is None

    def test_append(self):
        ov = normalize_overrides({"identity": {"mode": "append", "text": "B"}})
        assert apply_fragment_override(ov, "identity", "A") == "A\n\nB"

    def test_prepend(self):
        ov = normalize_overrides({"identity": {"mode": "prepend", "text": "B"}})
        assert apply_fragment_override(ov, "identity", "A") == "B\n\nA"

    def test_append_to_empty_default_yields_new(self):
        ov = normalize_overrides({"identity": {"mode": "append", "text": "B"}})
        assert apply_fragment_override(ov, "identity", "") == "B"
        assert apply_fragment_override(ov, "identity", None) == "B"

    def test_prepend_to_empty_default_yields_new(self):
        ov = normalize_overrides({"identity": {"mode": "prepend", "text": "B"}})
        assert apply_fragment_override(ov, "identity", "   ") == "B"


class TestFragmentKeyRegistry:
    def test_keys_present_and_documented(self):
        assert len(FRAGMENT_KEYS) == 16
        for key, desc in FRAGMENT_KEYS.items():
            assert isinstance(key, str) and key
            assert isinstance(desc, str) and desc.strip()

    def test_core_pushy_blocks_addressable(self):
        # The fragments that motivated this feature must be overridable.
        for key in (
            "task_completion",
            "tool_use_enforcement",
            "execution_discipline",
            "google_operational",
        ):
            assert key in FRAGMENT_KEYS

    def test_valid_modes(self):
        assert VALID_MODES == ("replace", "append", "prepend", "remove")
