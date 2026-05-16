from gateway.run import _apply_final_turn_trailing_lines
from gateway.session_hygiene import (
    build_session_quota_recommendation,
    session_quota_threshold_for_tokens,
)


def test_session_quota_thresholds_and_messages():
    below = build_session_quota_recommendation(
        session_id="s1",
        prompt_tokens=39_999,
        context_length=200_000,
    )
    assert below is None

    heads_up = build_session_quota_recommendation(
        session_id="s1",
        prompt_tokens=40_000,
        context_length=200_000,
    )
    assert heads_up is not None
    assert heads_up.threshold == 40_000
    assert heads_up.level == "heads_up"
    assert heads_up.dedupe_key == "session-quota:s1:40000"
    assert "40,000" in heads_up.text
    assert "keep going" in heads_up.text.lower()
    assert "auto" not in heads_up.text.lower()

    strong = build_session_quota_recommendation(
        session_id="s1",
        prompt_tokens=100_001,
        context_length=150_000,
    )
    assert strong is not None
    assert strong.threshold == 100_000
    assert strong.level == "strong_economy"
    assert "economy" in strong.text.lower()
    assert "reset" in strong.text.lower()


def test_session_quota_returns_highest_crossed_threshold():
    assert session_quota_threshold_for_tokens(70_000) == 70_000
    assert session_quota_threshold_for_tokens(99_999) == 70_000
    assert session_quota_threshold_for_tokens(130_000) == 130_000


def test_session_quota_warned_threshold_dedupe():
    rec = build_session_quota_recommendation(
        session_id="s1",
        prompt_tokens=100_000,
        warned_thresholds=[40_000, 70_000, 100_000],
    )
    assert rec is None

    rec = build_session_quota_recommendation(
        session_id="s1",
        prompt_tokens=130_000,
        warned_thresholds=[40_000, 70_000, 100_000],
    )
    assert rec is not None
    assert rec.threshold == 130_000
    assert rec.dedupe_key == "session-quota:s1:130000"


def test_quota_note_appends_only_to_non_streaming_final_response():
    response, trailing_lines, quota_delivered = _apply_final_turn_trailing_lines(
        "final answer",
        quota_line="⚠️ quota note",
        footer_line="model · 50%",
        already_sent=False,
    )

    assert response == "final answer\n\n⚠️ quota note\nmodel · 50%"
    assert trailing_lines == []
    assert quota_delivered is True


def test_streaming_quota_only_turn_has_no_trailing_send_or_persistence_marker():
    response, trailing_lines, quota_delivered = _apply_final_turn_trailing_lines(
        "streamed body already delivered",
        quota_line="⚠️ quota note",
        footer_line="",
        already_sent=True,
    )

    assert response == "streamed body already delivered"
    # Gateway sends only returned trailing_lines via adapter.send; quota-only
    # streaming turns must not produce a separate platform message.
    assert trailing_lines == []
    # The threshold must not be persisted as warned because the note was not
    # delivered; a later normal final-response turn can still warn.
    assert quota_delivered is False


def test_streaming_quota_not_marked_warned_allows_future_normal_warning():
    rec = build_session_quota_recommendation(
        session_id="s1",
        prompt_tokens=40_000,
        warned_thresholds=[],
    )
    assert rec is not None
    response, _, delivered = _apply_final_turn_trailing_lines(
        "streamed",
        quota_line=f"⚠️ {rec.text}",
        already_sent=True,
    )
    assert response == "streamed"
    assert delivered is False

    future = build_session_quota_recommendation(
        session_id="s1",
        prompt_tokens=41_000,
        warned_thresholds=[],
    )
    assert future is not None
    assert future.threshold == 40_000
