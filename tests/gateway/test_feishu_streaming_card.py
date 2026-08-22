"""Tests for the Feishu streaming card skeleton (Phase 4 / L4).

Covers:
  - Session state machine: THINKING -> STREAMING -> FINAL/ERROR transitions.
  - Terminal-state semantics: refusing to unstick from FINAL/ERROR.
  - CardKit v2 payload shape: required ``config`` / ``header`` / ``elements``.
  - Content accumulation via append_token.
  - Header label and template color per state.
  - Body element content per state (thinking / streaming / final / error).
  - Long-content truncation (>25KB) for the 30KB payload cap.
  - Footer support.
  - Idempotent transitions (re-applying the same state is a no-op).

The full L4 (sidecar process, token-stream callback, live card_id
patching) is multi-week scope and not covered here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from plugins.platforms.feishu import feishu_streaming_card as sc  # noqa: E402


# --- State machine tests ----------------------------------------------------

class TestStateMachine:
    def test_initial_state_is_thinking(self):
        s = sc.StreamingCardSession()
        assert s.state == sc.CardState.THINKING

    def test_append_token_transitions_thinking_to_streaming(self):
        s = sc.StreamingCardSession()
        s.append_token("hello")
        assert s.state == sc.CardState.STREAMING

    def test_append_token_accumulates(self):
        s = sc.StreamingCardSession()
        s.append_token("foo ")
        s.append_token("bar")
        assert s.content_buffer == ["foo ", "bar"]

    def test_finalize_sets_final(self):
        s = sc.StreamingCardSession()
        s.append_token("data")
        s.finalize()
        assert s.state == sc.CardState.FINAL

    def test_fail_sets_error_with_message(self):
        s = sc.StreamingCardSession()
        s.fail("timeout")
        assert s.state == sc.CardState.ERROR
        assert s.error_message == "timeout"

    def test_terminal_state_refuses_revert(self):
        """FINAL is terminal — re-entering STREAMING must be refused."""
        s = sc.StreamingCardSession()
        s.finalize()
        sc.transition(s, sc.CardState.STREAMING)
        assert s.state == sc.CardState.FINAL

    def test_error_terminal_state_refuses_revert(self):
        s = sc.StreamingCardSession()
        s.fail("boom")
        sc.transition(s, sc.CardState.FINAL)
        assert s.state == sc.CardState.ERROR

    def test_idempotent_transition(self):
        s = sc.StreamingCardSession()
        sc.transition(s, sc.CardState.STREAMING)
        sc.transition(s, sc.CardState.STREAMING)  # no-op
        assert s.state == sc.CardState.STREAMING


# --- Card payload shape -----------------------------------------------------

class TestCardPayload:
    def test_payload_has_required_keys(self):
        s = sc.StreamingCardSession()
        card = sc.render_streaming_card(s)
        assert "config" in card
        assert "header" in card
        assert "elements" in card

    def test_wide_screen_mode_set(self):
        s = sc.StreamingCardSession()
        card = sc.render_streaming_card(s)
        assert card["config"].get("wide_screen_mode") is True

    def test_header_has_title_and_template(self):
        s = sc.StreamingCardSession()
        card = sc.render_streaming_card(s)
        assert "title" in card["header"]
        assert "template" in card["header"]
        assert card["header"]["title"]["tag"] == "plain_text"

    def test_header_template_per_state(self):
        for state, expected in [
            (sc.CardState.THINKING, "blue"),
            (sc.CardState.STREAMING, "blue"),
            (sc.CardState.FINAL, "green"),
            (sc.CardState.ERROR, "red"),
        ]:
            s = sc.StreamingCardSession()
            s.state = state
            card = sc.render_streaming_card(s)
            assert card["header"]["template"] == expected, (
                f"state={state} expected template {expected}, "
                f"got {card['header']['template']}"
            )


# --- Body content per state -------------------------------------------------

class TestBodyContent:
    def test_thinking_state_shows_placeholder(self):
        s = sc.StreamingCardSession()
        card = sc.render_streaming_card(s)
        body = card["elements"][0]
        assert "思考" in body["content"]

    def test_streaming_with_content_shows_content(self):
        s = sc.StreamingCardSession()
        s.append_token("partial response")
        card = sc.render_streaming_card(s)
        body = card["elements"][0]
        assert "partial response" in body["content"]

    def test_streaming_without_content_shows_placeholder(self):
        s = sc.StreamingCardSession()
        s.state = sc.CardState.STREAMING  # no tokens
        card = sc.render_streaming_card(s)
        body = card["elements"][0]
        assert "生成第一条" in body["content"]

    def test_error_state_shows_error_message(self):
        s = sc.StreamingCardSession()
        s.fail("rate limit exceeded")
        card = sc.render_streaming_card(s)
        body = card["elements"][0]
        assert "rate limit exceeded" in body["content"]

    def test_final_with_empty_content_shows_placeholder(self):
        s = sc.StreamingCardSession()
        s.finalize()
        card = sc.render_streaming_card(s)
        body = card["elements"][0]
        assert "响应为空" in body["content"]


# --- Truncation and footer --------------------------------------------------

class TestTruncationAndFooter:
    def test_long_content_truncated(self):
        s = sc.StreamingCardSession()
        s.state = sc.CardState.STREAMING
        s.content_buffer = ["x" * 30000]
        card = sc.render_streaming_card(s)
        body = card["elements"][0]
        assert len(body["content"]) < 26000
        assert "截断" in body["content"]

    def test_short_content_not_truncated(self):
        s = sc.StreamingCardSession()
        s.state = sc.CardState.STREAMING
        s.content_buffer = ["x" * 1000]
        card = sc.render_streaming_card(s)
        body = card["elements"][0]
        assert "截断" not in body["content"]
        assert body["content"].count("x") == 1000

    def test_footer_appended(self):
        s = sc.StreamingCardSession()
        s.state = sc.CardState.STREAMING
        s.append_token("body")
        card = sc.render_streaming_card(s, footer="see also: doc.html")
        # Body + footer element
        assert len(card["elements"]) == 2
        assert "see also: doc.html" in card["elements"][1]["content"]
        assert "---" in card["elements"][1]["content"]  # separator

    def test_no_footer_keeps_single_element(self):
        s = sc.StreamingCardSession()
        card = sc.render_streaming_card(s)
        assert len(card["elements"]) == 1