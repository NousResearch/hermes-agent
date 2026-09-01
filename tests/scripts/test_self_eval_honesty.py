"""C6 adversarial tests — self-eval emission failure honesty."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "denji-self-eval-trigger.py"
sys.path.insert(0, str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location("selfeval_trigger_c6", str(SCRIPT))
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


def _decision(mod, home):
    import time
    now = int(time.time())
    for i in range(4):
        from hermes_cli.profile_activity_ledger import append_event
        append_event(
            source="t", event_type="kanban.crashed",
            event_id=f"c6-{i}-{time.time_ns()}",
            actor_profile="octacon", target_profile="x",
            occurred_at=now - 100 - i,
        )
    return mod.decide_trigger("octacon", since=now - 86400,
                              hermes_home=home, now=now)


class TestC6Honesty:
    def test_append_failure_returns_none_or_raises(self, fake_home, monkeypatch):
        """RED: ledger append exception must NOT return a success-shaped id."""
        mod = _load()
        d = _decision(mod, fake_home)
        assert d["trigger"] is True

        import hermes_cli.profile_activity_ledger as pal
        def broken_append(**kwargs):
            raise RuntimeError("ledger unavailable")
        monkeypatch.setattr(pal, "append_event", broken_append)
        result = mod.emit_trigger(d, hermes_home=fake_home)
        assert result is None  # no false success

    def test_success_only_after_confirmed_append(self, fake_home):
        mod = _load()
        d = _decision(mod, fake_home)
        eid = mod.emit_trigger(d, hermes_home=fake_home)
        assert eid is not None
        # And the event genuinely exists
        from hermes_cli.profile_activity_ledger import query_events
        events = [e for e in query_events(event_types=["profile.self_eval.trigger"])
                  if e.get("event_id") == eid]
        assert len(events) == 1

    def test_wrong_hermes_home_isolated(self, fake_home, tmp_path):
        """Writing to the wrong HERMES_HOME must not land the event in the
        intended ledger, and must be honestly reported against the path used."""
        mod = _load()
        d = _decision(mod, fake_home)
        wrong_home = tmp_path / "elsewhere"
        wrong_home.mkdir()
        result = mod.emit_trigger(d, hermes_home=wrong_home)
        # The intended ledger must NOT have received the trigger event.
        import sqlite3
        intended = fake_home / "governance" / "profile-activity-ledger.sqlite"
        con = sqlite3.connect(f"file:{intended}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT COUNT(*) FROM activity_events WHERE event_type = 'profile.self_eval.trigger'"
        ).fetchone()[0]
        con.close()
        # Whatever emit reported, the intended ledger gained nothing from
        # a write directed at the wrong home (emit was never called with it).
        assert rows == 0
        if result is not None:
            wrong = wrong_home / "governance" / "profile-activity-ledger.sqlite"
            con2 = sqlite3.connect(f"file:{wrong}?mode=ro", uri=True)
            wrong_rows = con2.execute(
                "SELECT COUNT(*) FROM activity_events WHERE event_type = 'profile.self_eval.trigger'"
            ).fetchone()[0]
            con2.close()
            assert wrong_rows == 1  # event went to the home it was given