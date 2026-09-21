"""Tests for scripts/denji-self-eval-submit.py — self-assessment submission.

P05 Batch 1: proves the self-assessment submission contract validates a
payload (profile, cycle, answers, ts) and appends a validated event to the
ledger.  Uses a temporary HERMES_HOME — never touches live ``~/.hermes``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "denji_self_eval_submit_under_test",
        str(SCRIPTS / "denji-self-eval-submit.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


class TestValidation:
    def test_valid_payload_passes(self):
        mod = _load_script()
        result = mod.validate_payload({
            "profile": "octacon",
            "cycle": "weekly",
            "answers": {"q1": 4},
            "ts": 1700000000,
        })
        assert result["profile"] == "octacon"
        assert result["cycle"] == "weekly"
        assert result["answers"] == {"q1": 4}
        assert result["ts"] == 1700000000

    def test_empty_profile_rejected(self):
        mod = _load_script()
        with pytest.raises(mod.SubmissionError, match="profile"):
            mod.validate_payload({"profile": "", "cycle": "weekly", "answers": {}})

    def test_missing_profile_rejected(self):
        mod = _load_script()
        with pytest.raises(mod.SubmissionError, match="profile"):
            mod.validate_payload({"cycle": "weekly", "answers": {}})

    def test_invalid_cycle_rejected(self):
        mod = _load_script()
        with pytest.raises(mod.SubmissionError, match="cycle"):
            mod.validate_payload({"profile": "octacon", "cycle": "bogus", "answers": {}})

    def test_answers_not_dict_rejected(self):
        mod = _load_script()
        with pytest.raises(mod.SubmissionError, match="answers"):
            mod.validate_payload({
                "profile": "octacon", "cycle": "weekly", "answers": "not a dict"
            })

    def test_ts_defaults_to_now(self):
        mod = _load_script()
        before = int(time.time())
        result = mod.validate_payload({
            "profile": "octacon", "cycle": "weekly", "answers": {}
        })
        after = int(time.time())
        assert before <= result["ts"] <= after

    def test_negative_ts_rejected(self):
        mod = _load_script()
        with pytest.raises(mod.SubmissionError, match="ts"):
            mod.validate_payload({
                "profile": "octacon", "cycle": "weekly", "answers": {}, "ts": -1
            })

    def test_actor_defaults_to_profile(self):
        mod = _load_script()
        result = mod.validate_payload({
            "profile": "octacon", "cycle": "monthly", "answers": {}
        })
        assert result["actor_profile"] == "octacon"

    def test_explicit_actor_preserved(self):
        mod = _load_script()
        result = mod.validate_payload({
            "profile": "octacon", "cycle": "monthly", "answers": {},
            "actor_profile": "kensei",
        })
        assert result["actor_profile"] == "kensei"


class TestSubmitAndRecord:
    def test_submit_appends_event(self, fake_home):
        mod = _load_script()
        result = mod.submit_self_eval(
            {"profile": "octacon", "cycle": "weekly", "answers": {"q1": 4}},
            hermes_home=fake_home,
        )
        assert result["verified"] is True
        assert result["event_id"].startswith("pal_")

    def test_event_queryable_after_submit(self, fake_home):
        mod = _load_script()
        mod.submit_self_eval(
            {"profile": "octacon", "cycle": "weekly", "answers": {"q1": 4}},
            hermes_home=fake_home,
        )
        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(
            event_types=["profile.self_eval.submit"],
            target_profile="octacon",
        )
        assert len(events) == 1
        assert events[0]["payload"]["cycle"] == "weekly"
        assert events[0]["payload"]["answers"] == {"q1": 4}

    def test_multiple_submissions_recorded(self, fake_home):
        mod = _load_script()
        for cycle in ("weekly", "monthly", "quarterly"):
            mod.submit_self_eval(
                {"profile": "octacon", "cycle": cycle, "answers": {}},
                hermes_home=fake_home,
            )
        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.self_eval.submit"])
        assert len(events) == 3

    def test_submit_with_explicit_ts(self, fake_home):
        mod = _load_script()
        ts = 1700000000
        mod.submit_self_eval(
            {"profile": "octacon", "cycle": "weekly", "answers": {}, "ts": ts},
            hermes_home=fake_home,
        )
        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.self_eval.submit"])
        assert events[0]["occurred_at"] == ts

    def test_invalid_payload_does_not_record(self, fake_home):
        mod = _load_script()
        with pytest.raises(mod.SubmissionError):
            mod.submit_self_eval(
                {"profile": "", "cycle": "weekly", "answers": {}},
                hermes_home=fake_home,
            )
        from hermes_cli.profile_activity_ledger import query_events
        events = query_events(event_types=["profile.self_eval.submit"])
        assert len(events) == 0
