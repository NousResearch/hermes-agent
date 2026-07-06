from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from torben_gtm_feedback import (
    apply_feedback_to_radar,
    build_gtm_post_packet,
    ensure_engagement_dedupe_ttl,
    evaluate_liveness,
    record_feedback_event,
)


NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def test_feedback_events_grow_file_and_update_ranking_inputs(tmp_path: Path) -> None:
    feedback_path = tmp_path / "torben-gtm-radar-feedback.json"
    record_feedback_event(path=feedback_path, candidate_id="gtm-1", event="selected", now=NOW)
    feedback = record_feedback_event(path=feedback_path, candidate_id="gtm-1", event="edited", edit_summary="tightened hook", now=NOW)
    radar = {"findings": [{"id": "gtm-1", "content_score": 70}, {"id": "gtm-2", "content_score": 70}]}

    adjusted = apply_feedback_to_radar(radar, feedback)

    assert len(feedback["events"]) == 2
    assert feedback["current_focus"] == "gtm-1"
    assert adjusted["findings"][0]["feedback_applied"] is True
    assert adjusted["findings"][0]["feedback_score_bonus"] == 13
    assert adjusted["findings"][0]["content_score"] == 83
    assert adjusted["findings"][1]["content_score"] == 70


def test_liveness_escalates_once_after_four_zero_conversion_days(tmp_path: Path) -> None:
    state = tmp_path / "gtm-liveness.json"
    conversions = {
        "2026-07-03": 0,
        "2026-07-04": 0,
        "2026-07-05": 0,
        "2026-07-06": 0,
    }

    first = evaluate_liveness(conversions_by_day=conversions, state_path=state, today=date(2026, 7, 6))
    second = evaluate_liveness(conversions_by_day=conversions, state_path=state, today=date(2026, 7, 6))

    assert first["status"] == "escalate"
    assert first["wakeAgent"] is True
    assert first["public_actions_taken"] == 0
    assert first["external_mutations"] == 0
    assert second["status"] == "deduped"
    assert second["wakeAgent"] is False


def test_engagement_dedupe_entries_get_ttl_and_expire(tmp_path: Path) -> None:
    state = tmp_path / "engagement-state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "delivered_opportunities": {
                    "fresh": {"last_seen_at": "2026-07-05T12:00:00Z", "post_url": "https://x.com/fresh"},
                    "old": {"last_seen_at": "2026-06-01T12:00:00Z", "post_url": "https://x.com/old"},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = ensure_engagement_dedupe_ttl(state_path=state, now=NOW, ttl_days=14)
    updated = json.loads(state.read_text(encoding="utf-8"))

    assert payload["expired"] == 1
    assert payload["kept"] == 1
    assert "old" not in updated["delivered_opportunities"]
    assert updated["delivered_opportunities"]["fresh"]["ttl_days"] == 14
    assert updated["delivered_opportunities"]["fresh"]["expires_at"] == "2026-07-19T12:00:00Z"


def test_gtm_post_packet_is_packet_only_and_takes_no_public_action() -> None:
    packet = build_gtm_post_packet({"id": "gtm-1", "summary": "Runtime controls post", "recommended_action": "draft X post"})

    assert packet["category"] == "gtm_post"
    assert packet["status"] == "packet_only"
    assert packet["public_actions_taken"] == 0
    assert packet["external_actions_taken"] == []
    assert "no autonomous public post" in packet["blocked_actions"]
