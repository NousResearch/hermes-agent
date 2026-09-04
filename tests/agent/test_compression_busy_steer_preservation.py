"""Regression coverage for busy-steer preservation across compression.

The busy-input ``steer`` mode rides user intent inside a ``role=tool`` result
as an out-of-band marker, which the ``_is_real_user_message``-only checks
would miss. f40333a80e taught ``_ensure_compressed_has_user_turn`` to treat
those markers as human intent (and to anchor the LAST intent-bearing row, so
a newer real user request outranks an older consumed steer — #100053); these
tests pin the pure detection/extraction helpers and that ordering contract.
"""

from agent.conversation_compression import (
    _compressed_has_busy_steer,
    _ensure_compressed_has_user_turn,
    _extract_steer_text_from_message,
    _message_contains_busy_steer,
)
from agent.prompt_builder import format_steer_marker


def _tool_message(text: str) -> dict:
    return {"role": "tool", "content": text}


def _tool_steer_message(steer_text: str) -> dict:
    return _tool_message(format_steer_marker(steer_text))


# ---------------------------------------------------------------------------
# marker detection / extraction helpers
# ---------------------------------------------------------------------------


class TestBusySteerMarkerDetection:
    def test_tool_result_with_steer_marker_detected(self):
        assert _message_contains_busy_steer(
            _tool_steer_message("run the deploy")
        ) is True

    def test_plain_tool_result_not_detected(self):
        assert _message_contains_busy_steer(
            _tool_message("just some tool output")
        ) is False

    def test_plain_user_message_not_detected(self):
        assert _message_contains_busy_steer({"role": "user", "content": "hi"}) is False

    def test_content_list_handled(self):
        msg = {"role": "tool", "content": [{"type": "text", "text": format_steer_marker("x")}]}
        assert _message_contains_busy_steer(msg) is True

    def test_missing_content_not_detected(self):
        assert _message_contains_busy_steer({"role": "tool"}) is False


class TestBusySteerExtraction:
    def test_extracts_inner_text(self):
        assert _extract_steer_text_from_message(
            _tool_steer_message("please summarize the thread")
        ) == "please summarize the thread"

    def test_no_marker_returns_none(self):
        assert _extract_steer_text_from_message(
            _tool_message("plain output")
        ) is None

    def test_unclosed_marker_returns_none(self):
        open_marker = "[OUT-OF-BAND USER MESSAGE"
        msg = _tool_message(f"{open_marker}\nnever closed")
        assert _extract_steer_text_from_message(msg) is None

    def test_empty_steer_text_returns_none(self):
        assert _extract_steer_text_from_message(
            _tool_message(format_steer_marker("   "))
        ) is None

    def test_missing_content_returns_none(self):
        assert _extract_steer_text_from_message({"role": "tool"}) is None


class TestCompressedHasBusySteer:
    def test_true_when_any_row_carries_marker(self):
        messages = [
            {"role": "assistant", "content": "ok"},
            _tool_steer_message("stop and check"),
        ]
        assert _compressed_has_busy_steer(messages) is True

    def test_false_when_no_marker(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert _compressed_has_busy_steer(messages) is False

    def test_false_on_empty(self):
        assert _compressed_has_busy_steer([]) is False


# ---------------------------------------------------------------------------
# _ensure_compressed_has_user_turn behaviour
# ---------------------------------------------------------------------------


class TestEnsureCompressedHasUserTurn:
    def test_real_user_already_in_compressed(self):
        compressed = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        assert _ensure_compressed_has_user_turn([], compressed) == "already_present"

    def test_steer_marker_in_compressed_counts_as_present(self):
        compressed = [
            {"role": "assistant", "content": "working"},
            _tool_steer_message("keep going"),
        ]
        assert _ensure_compressed_has_user_turn([], compressed) == "already_present"

    def test_anchors_latest_steer_tool(self):
        original = [_tool_steer_message("wrap it up now")]
        compressed = [{"role": "assistant", "content": "working"}]
        outcome = _ensure_compressed_has_user_turn(original, compressed)
        assert outcome == "inserted"
        assert compressed[0]["role"] == "user"
        assert "wrap it up now" in str(compressed[0]["content"])

    def test_real_user_outranks_older_steer(self):
        """#100053: [user A, tool(steer B), user C] must anchor C, not B."""
        original = [
            {"role": "user", "content": "first request A"},
            _tool_steer_message("mid-turn steer B"),
            {"role": "user", "content": "latest request C"},
        ]
        compressed = [{"role": "assistant", "content": "compressed tail"}]
        _ensure_compressed_has_user_turn(original, compressed)
        anchor = next(m for m in compressed if m.get("role") == "user")
        assert "latest request C" in str(anchor["content"])
        assert "steer B" not in str(anchor["content"])

    def test_newer_steer_outranks_older_user(self):
        """A steer delivered AFTER the last real user row is the newest intent."""
        original = [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "done"},
            _tool_steer_message("now do the follow-up"),
        ]
        compressed = [{"role": "assistant", "content": "compressed tail"}]
        _ensure_compressed_has_user_turn(original, compressed)
        anchor = next(m for m in compressed if m.get("role") == "user")
        assert "now do the follow-up" in str(anchor["content"])

    def test_placeholder_appended_without_any_intent(self):
        original = [
            {"role": "assistant", "content": "system-ish content only"},
        ]
        compressed = [{"role": "assistant", "content": "compressed tail"}]
        assert _ensure_compressed_has_user_turn(original, compressed) == "placeholder_appended"
