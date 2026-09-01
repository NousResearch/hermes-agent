"""P3.3 tests — deterministic targeted self-evaluation trigger."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "denji-self-eval-trigger.py"

sys.path.insert(0, str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location("selfeval_trigger", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "profiles" / "octacon").mkdir(parents=True)
    (h / "governance").mkdir()
    (h / "profiles" / "octacon" / "config.yaml").write_text("model:\n  default: t/m\n")
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _seed(home, event_type, actor, target, when, count=1):
    from hermes_cli.profile_activity_ledger import append_event
    for _ in range(count):
        append_event(
            source="t", event_type=event_type,
            event_id=f"s-{event_type}-{target}-{when}-{time.time_ns()}",
            actor_profile=actor, target_profile=target, occurred_at=when,
        )


class TestTriggerDecision:
    def test_repeated_failure_triggers(self, fake_home):
        mod = _load()
        now = int(time.time())
        for i in range(3):
            _seed(fake_home, "kanban.crashed", "octacon", "t", now - 100 - i)
        d = mod.decide_trigger("octacon", since=now - 86400, hermes_home=fake_home, now=now)
        reasons = [r["reason"] for r in d["reasons"]]
        assert "repeated_failure" in reasons
        assert d["trigger"] is True

    def test_material_config_change_triggers(self, fake_home):
        mod = _load()
        now = int(time.time())
        d = mod.decide_trigger("octacon", since=now - 86400, hermes_home=fake_home, now=now)
        # config.yaml was just created inside the window → material change
        reasons = [r["reason"] for r in d["reasons"]]
        assert "material_config_change" in reasons

    def test_quarterly_lead_review(self, fake_home):
        mod = _load()
        now = int(time.time())
        d = mod.decide_trigger(
            "octacon", since=now - 86400, hermes_home=fake_home, now=now,
            is_lead=True, force_quarter_boundary=True,
        )
        assert any(r["reason"] == "quarterly_lead_review" for r in d["reasons"])

    def test_quarterly_not_for_workers(self, fake_home):
        mod = _load()
        now = int(time.time())
        d = mod.decide_trigger(
            "octacon", since=now - 86400, hermes_home=fake_home, now=now,
            is_lead=False, force_quarter_boundary=True,
        )
        assert not any(r["reason"] == "quarterly_lead_review" for r in d["reasons"])

    def test_deterministic_same_input_same_decision(self, fake_home):
        mod = _load()
        now = int(time.time())
        d1 = mod.decide_trigger("octacon", since=now - 86400, hermes_home=fake_home, now=now)
        d2 = mod.decide_trigger("octacon", since=now - 86400, hermes_home=fake_home, now=now)
        assert d1 == d2

    def test_no_blanket_fleet_trigger(self, fake_home):
        """A quiet profile with fresh evidence produces no trigger."""
        mod = _load()
        now = int(time.time())
        # Config older than the window → no change; no failures seeded.
        import os
        old = now - 90 * 86400
        os.utime(fake_home / "profiles" / "octacon" / "config.yaml", (old, old))
        d = mod.decide_trigger("octacon", since=now - 86400, hermes_home=fake_home, now=now)
        assert d["trigger"] is False
        assert d["reasons"] == []


class TestEmitIdempotence:
    def test_emit_creates_event_and_idempotent(self, fake_home):
        mod = _load()
        now = int(time.time())
        for i in range(4):
            _seed(fake_home, "kanban.gave_up", "octacon", "x", now - 200 - i)
        d = mod.decide_trigger("octacon", since=now - 86400, hermes_home=fake_home, now=now)
        eid1 = mod.emit_trigger(d, hermes_home=fake_home)
        eid2 = mod.emit_trigger(d, hermes_home=fake_home)
        assert eid1 == eid2
        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.self_eval.trigger"])
        matching = [e for e in events if e.get("event_id") == eid1]
        assert len(matching) == 1  # idempotent by event id

    def test_no_trigger_emits_nothing(self, fake_home):
        mod = _load()
        now = int(time.time())
        import os
        old = now - 90 * 86400
        os.utime(fake_home / "profiles" / "octacon" / "config.yaml", (old, old))
        d = mod.decide_trigger("octacon", since=now - 86400, hermes_home=fake_home, now=now)
        assert mod.emit_trigger(d, hermes_home=fake_home) is None

    def test_trigger_event_cites_reason_and_evidence(self, fake_home):
        mod = _load()
        now = int(time.time())
        for i in range(3):
            _seed(fake_home, "kanban.crashed", "octacon", "t", now - 100 - i)
        d = mod.decide_trigger("octacon", since=now - 86400, hermes_home=fake_home, now=now)
        eid = mod.emit_trigger(d, hermes_home=fake_home)
        from hermes_cli.profile_activity_ledger import query_events
        ev = [e for e in query_events(event_types=["profile.self_eval.trigger"]) if e.get("event_id") == eid][0]
        payload = ev["payload"]
        assert payload["reasons"] and payload["reasons"][0]["evidence_ref"]

    def test_does_not_touch_reminder_or_cron(self, fake_home):
        """The trigger path must not modify any cron record or reminder."""
        mod = _load()
        now = int(time.time())
        d = mod.decide_trigger("octacon", since=now - 86400, hermes_home=fake_home, now=now)
        mod.emit_trigger(d, hermes_home=fake_home)
        cron_dir = fake_home / "cron"
        assert not cron_dir.exists()  # nothing created
        # Ledger events only.
        ledger = fake_home / "governance" / "profile-activity-ledger.sqlite"
        assert ledger.exists()