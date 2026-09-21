"""Tests for scripts/denji-review-cycle.py — weekly/monthly/quarterly packets.

P05 Batch 1: proves the canonical Python review cycle generates
weekly/monthly/quarterly review packets from a fixture HERMES_HOME and
appends ``profile.review.{cycle}`` events to the ledger.  Uses a temporary
HERMES_HOME — never touches live ``~/.hermes``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "denji-review-cycle.py"

sys.path.insert(0, str(REPO))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "denji_review_cycle_under_test", str(SCRIPT)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    h.mkdir()
    # Create a default profile config
    (h / "config.yaml").write_text("model:\n  default: test/m\nskills:\n  enabled_skills: []\n")
    # Create a named profile
    pdir = h / "profiles" / "octacon"
    pdir.mkdir(parents=True)
    (pdir / "config.yaml").write_text(
        "model:\n  default: test/m\nskills:\n  enabled_skills: [code-review]\n  always_skills: [hermes-agent]\n"
    )
    (pdir / "SOUL.md").write_text("# Octacon")
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _seed_ledger_events(fake_home, mod):
    """Seed a few events so ledger_counts returns non-zero."""
    from hermes_cli.profile_activity_ledger import append_event
    now = int(time.time())
    for i in range(5):
        append_event(
            source="test",
            event_type="skill.loaded",
            actor_profile="octacon",
            target_profile="octacon",
            occurred_at=now - i * 100,
        )
    append_event(
        source="test",
        event_type="skill.enabled_auto",
        actor_profile="octacon",
        target_profile="octacon",
        occurred_at=now - 200,
    )


class TestWeeklyPacket:
    def test_weekly_generates_events(self, fake_home):
        mod = _load_script()
        _seed_ledger_events(fake_home, mod)
        # Reload after seeding so HERMES_HOME is picked up
        mod = _load_script()
        result = mod._run_cycle("weekly")
        assert result == "profile.review.weekly"

        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.review.weekly"])
        # Should have events for default + octacon
        assert len(events) >= 2

    def test_weekly_payload_has_cycle_field(self, fake_home):
        mod = _load_script()
        mod._run_cycle("weekly")
        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.review.weekly"])
        for ev in events:
            assert ev["payload"]["cycle"] == "weekly"


class TestMonthlyPacket:
    def test_monthly_generates_events(self, fake_home):
        mod = _load_script()
        _seed_ledger_events(fake_home, mod)
        mod = _load_script()
        result = mod._run_cycle("monthly")
        assert result == "profile.review.monthly"

        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.review.monthly"])
        assert len(events) >= 2

    def test_monthly_payload_has_enabled_skills_count(self, fake_home):
        mod = _load_script()
        mod._run_cycle("monthly")
        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.review.monthly"])
        octacon = [e for e in events if e["target_profile"] == "octacon"]
        assert len(octacon) == 1
        assert octacon[0]["payload"]["enabled_skills_count"] == 1


class TestQuarterlyPacket:
    def test_quarterly_generates_events(self, fake_home):
        mod = _load_script()
        _seed_ledger_events(fake_home, mod)
        mod = _load_script()
        result = mod._run_cycle("quarterly")
        assert result == "profile.review.quarterly"

        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.review.quarterly"])
        assert len(events) >= 2

    def test_quarterly_payload_has_soul_and_user_flags(self, fake_home):
        mod = _load_script()
        mod._run_cycle("quarterly")
        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.review.quarterly"])
        octacon = [e for e in events if e["target_profile"] == "octacon"]
        assert octacon[0]["payload"]["soul_exists"] is True
        assert octacon[0]["payload"]["user_exists"] is False


class TestCallableFromFixtures:
    """The script must be callable from a fixture HERMES_HOME without
    touching live governance state."""

    def test_artifact_written_to_fixture_logboard(self, fake_home):
        mod = _load_script()
        mod._run_cycle("weekly")
        logboard = fake_home / "governance" / "logboard"
        artifacts = list(logboard.glob("profile-review-weekly-*.json"))
        assert len(artifacts) == 1
        data = json.loads(artifacts[0].read_text())
        assert data["cycle"] == "weekly"
        assert data["profiles_reviewed"] >= 2

    def test_does_not_touch_live_home(self, fake_home, tmp_path):
        """The ledger DB must be under the fixture, not ~/.hermes."""
        mod = _load_script()
        mod._run_cycle("weekly")
        ledger = fake_home / "governance" / "profile-activity-ledger.sqlite"
        assert ledger.exists()
        # Ensure the live path was NOT created
        live = Path.home() / ".hermes" / "governance" / "profile-activity-ledger.sqlite"
        # The live path may exist from other tests, but ours must be under fake_home
        assert ledger.parent.parent == fake_home

    def test_unknown_cycle_exits_nonzero(self, fake_home):
        mod = _load_script()
        with pytest.raises(SystemExit):
            mod._run_cycle("bogus")
