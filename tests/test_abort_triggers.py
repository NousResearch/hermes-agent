"""Tests for natural-language abort trigger detection.

Validates the natural-language abort trigger system
(see tools/abort_triggers.py for background).
"""
import importlib.util
import os
import pytest

# Load the module directly from its file path to avoid pulling in the
# full tools package (which has heavy dependencies like httpx).
_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "tools", "abort_triggers.py")
_spec = importlib.util.spec_from_file_location("abort_triggers", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

is_abort_request = _mod.is_abort_request
ABORT_REPLY = _mod.ABORT_REPLY
_normalize = _mod._normalize


# ── Normalization ─────────────────────────────────────────────────────

class TestNormalize:
    def test_strip_and_lowercase(self):
        assert _normalize("  STOP  ") == "stop"

    def test_trailing_punctuation_removed(self):
        assert _normalize("STOP!!!") == "stop"
        assert _normalize("stop...") == "stop"
        assert _normalize("cancel?!") == "cancel"

    def test_multiple_spaces_collapsed(self):
        assert _normalize("stop   hermes") == "stop hermes"

    def test_empty_string(self):
        assert _normalize("") == ""
        assert _normalize("   ") == ""


# ── Single-word triggers ──────────────────────────────────────────────

class TestSingleWordTriggers:
    @pytest.mark.parametrize("text", [
        "stop", "STOP", "Stop", "abort", "ABORT", "cancel",
        "halt", "quit",
    ])
    def test_english_triggers(self, text):
        assert is_abort_request(text) is True

    def test_with_trailing_punctuation(self):
        assert is_abort_request("STOP!!!") is True
        assert is_abort_request("stop...") is True
        assert is_abort_request("cancel?!") is True

    def test_with_whitespace(self):
        assert is_abort_request("  stop  ") is True


# ── Multi-word trigger phrases ────────────────────────────────────────

class TestMultiWordTriggers:
    @pytest.mark.parametrize("text", [
        "stop hermes", "STOP HERMES", "Stop Hermes",
        "stop agent", "stop please", "stop now",
        "stop it", "stop that", "stop everything",
        "please stop", "don't do that", "do not do that",
        "stop don't do anything",
    ])
    def test_english_phrases(self, text):
        assert is_abort_request(text) is True

    def test_legacy_agent_phrases(self):
        """Legacy phrases from other agent ecosystems."""
        assert is_abort_request("STOP OPENCLAW") is True
        assert is_abort_request("stop openclaw") is True

    def test_with_punctuation(self):
        assert is_abort_request("STOP HERMES!!!") is True
        assert is_abort_request("stop please...") is True


# ── Negative cases (should NOT trigger abort) ─────────────────────────

class TestNonAbortMessages:
    @pytest.mark.parametrize("text", [
        "hello",
        "can you stop and explain?",
        "stopping the server",
        "please cancel the subscription for user 123",
        "how do I abort a git merge?",
        "the process should stop when memory exceeds 1GB",
        "stop by the store on your way home",
        "/stop",  # slash commands handled by command dispatch
        "",
        "   ",
    ])
    def test_normal_messages_not_matched(self, text):
        assert is_abort_request(text) is False

    def test_slash_commands_excluded(self):
        assert is_abort_request("/stop") is False
        assert is_abort_request("/abort") is False


# ── Reply message ─────────────────────────────────────────────────────

class TestAbortReply:
    def test_reply_format(self):
        assert "Agent was aborted" in ABORT_REPLY
        assert "⚙️" in ABORT_REPLY
