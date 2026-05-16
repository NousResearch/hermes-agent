from datetime import datetime

from gateway.config import Platform
from gateway.session import SessionEntry
from gateway.session_hygiene import build_session_quota_recommendation
from hermes_cli.web_server import _cockpit_session_quota_recommendation_from_entries


def test_session_entry_persists_quota_warning_thresholds():
    entry = SessionEntry(
        session_key="discord:chan:user",
        session_id="s1",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        platform=Platform.DISCORD,
        last_prompt_tokens=70_000,
        quota_warning_thresholds=[40_000, 70_000],
    )

    data = entry.to_dict()
    assert data["quota_warning_thresholds"] == [40_000, 70_000]
    round_tripped = SessionEntry.from_dict(data)
    assert round_tripped.quota_warning_thresholds == [40_000, 70_000]


def test_cockpit_session_quota_recommendation_uses_highest_active_entry():
    old = SessionEntry(
        session_key="discord:older",
        session_id="old",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        platform=Platform.DISCORD,
        last_prompt_tokens=70_000,
    )
    active = SessionEntry(
        session_key="discord:active",
        session_id="active",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 2),
        platform=Platform.DISCORD,
        last_prompt_tokens=130_000,
    )

    rec = _cockpit_session_quota_recommendation_from_entries([old, active])
    assert rec is not None
    assert rec["threshold"] == 130_000
    assert rec["level"] == "urgent"
    assert rec["dedupe_key"].startswith("session-quota-public:130000:")
    assert "active" not in rec["dedupe_key"]
    assert "start a fresh session soon" in rec["text"].lower()
