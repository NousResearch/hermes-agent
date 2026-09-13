"""Tests for agent.turn_voice_context: the trusted per-turn voice/input-modality signal (#109455)."""

import pytest

from agent.turn_voice_context import parse_voice_context


class TestParseVoiceContext:
    @pytest.mark.parametrize("raw", [None, "", "voice", 1, True, [], ["voice"]])
    def test_non_mapping_input_returns_empty(self, raw):
        assert parse_voice_context(raw) == {}

    def test_empty_dict_returns_empty(self):
        assert parse_voice_context({}) == {}

    def test_dict_asserting_nothing_returns_empty(self):
        # Default modality, inactive session, no surface — indistinguishable from "not provided".
        assert parse_voice_context({"input_modality": "text", "voice_session_active": False}) == {}

    def test_voice_modality_is_preserved(self):
        result = parse_voice_context({"input_modality": "voice"})
        assert result == {"input_modality": "voice", "voice_session_active": False, "client_surface": ""}

    def test_active_session_alone_is_preserved(self):
        # Typed message mid voice-session: modality is text, but the session flag still matters.
        result = parse_voice_context({"voice_session_active": True})
        assert result == {"input_modality": "text", "voice_session_active": True, "client_surface": ""}

    def test_surface_alone_is_preserved(self):
        result = parse_voice_context({"client_surface": "desktop"})
        assert result == {"input_modality": "text", "voice_session_active": False, "client_surface": "desktop"}

    def test_full_shape_round_trips(self):
        raw = {"input_modality": "voice", "voice_session_active": True, "client_surface": "desktop"}
        assert parse_voice_context(raw) == raw

    def test_unknown_modality_falls_back_to_text(self):
        result = parse_voice_context({"input_modality": "telepathy", "voice_session_active": True})
        assert result["input_modality"] == "text"

    def test_non_string_surface_is_dropped(self):
        result = parse_voice_context({"input_modality": "voice", "client_surface": 12345})
        assert result["client_surface"] == ""

    def test_surface_is_trimmed_and_capped(self):
        result = parse_voice_context({"input_modality": "voice", "client_surface": "  desktop  "})
        assert result["client_surface"] == "desktop"
        long_surface = "x" * 500
        result = parse_voice_context({"input_modality": "voice", "client_surface": long_surface})
        assert len(result["client_surface"]) == 40

    def test_extra_keys_are_ignored(self):
        result = parse_voice_context({"input_modality": "voice", "tts_provider": "elevenlabs"})
        assert "tts_provider" not in result

    @pytest.mark.parametrize("unhashable", [[], {}, ["voice"], {"nested": True}])
    def test_unhashable_modality_normalizes_to_text_instead_of_raising(self, unhashable):
        # Regression: `modality in _VALID_MODALITIES` on a list/dict raised TypeError
        # (unhashable type), defeating this function's whole "never raise on garbage" contract.
        result = parse_voice_context({"input_modality": unhashable, "voice_session_active": True})
        assert result["input_modality"] == "text"

    def test_string_false_does_not_mean_active(self):
        # Regression: bool("false") is True in Python — a naive bool() cast would silently
        # invert an explicit voice_session_active: "false" sent by a JSON-ish caller.
        assert parse_voice_context({"voice_session_active": "false"}) == {}
        result = parse_voice_context({"input_modality": "voice", "voice_session_active": "false"})
        assert result["voice_session_active"] is False

    @pytest.mark.parametrize("truthy_string", ["true", "True", "1", "yes"])
    def test_recognized_truthy_strings_activate_the_session(self, truthy_string):
        result = parse_voice_context({"voice_session_active": truthy_string})
        assert result["voice_session_active"] is True
