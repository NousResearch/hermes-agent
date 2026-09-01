"""P3.2 tests — four-dimensional profile review in denji-review-cycle.py.

Covers the mandated fixture matrix:
  - healthy active gateway;
  - cold/standby worker;
  - failing active profile;
  - missing dependency / config invalidity;
  - direct delegation workload;
  - recurring finding / rework quality evidence;
  - absent dimension evidence;
  - active gateway with zero ledger volume is NOT dormant.
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
        "denji_review_cycle_4dim", str(SCRIPT)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "profiles").mkdir(parents=True)
    (h / "governance").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _add_profile(home, name, config=True, soul=True):
    d = home / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    if config:
        (d / "config.yaml").write_text("model:\n  default: test/m\nskills:\n  enabled_skills: []\n")
    elif config is False:
        (d / "config.yaml").write_text("")  # explicitly invalid/empty
    if soul:
        (d / "SOUL.md").write_text(f"# {name}")
    return d


def _registry(home, profiles):
    import yaml
    doc = {
        "schema_version": 1,
        "root": {"name": "KENSEI", "description": "root"},
        "profiles": profiles,
    }
    (home / "governance" / "profile-registry.yaml").write_text(
        yaml.safe_dump(doc, sort_keys=True), encoding="utf-8"
    )


def _seed(home, event_type, actor, target, occurred, count=1):
    from hermes_cli.profile_activity_ledger import append_event
    for _ in range(count):
        append_event(
            source="test", event_type=event_type, event_id=f"seed-{event_type}-{actor}-{target}-{occurred}-{time.time_ns()}",
            actor_profile=actor, target_profile=target, occurred_at=occurred,
        )


class TestVerdictAssembly:
    def test_healthy_active_gateway(self, fake_home, monkeypatch):
        mod = _load_script()
        _add_profile(fake_home, "octacon")
        monkeypatch.setattr(mod, "_runtime_dimension", lambda p, s: {
            "dimension": "runtime", "verdict": "ACTIVE",
            "evidence": {"unit_active": True}, "method_version": mod.REVIEW_VERSION,
        })
        now = int(time.time())
        review = mod._build_review("octacon", now - 7 * 86400, "weekly")
        assert review["verdict"] == "HEALTHY"
        assert set(review["dimensions"]) == {"runtime", "workload", "quality", "capability"}

    def test_active_gateway_with_zero_ledger_volume_not_dormant(self, fake_home, monkeypatch):
        mod = _load_script()
        _add_profile(fake_home, "remii")
        # Active runtime, ZERO ledger events of any kind.
        monkeypatch.setattr(mod, "_runtime_dimension", lambda p, s: {
            "dimension": "runtime", "verdict": "ACTIVE",
            "evidence": {"unit_active": True, "heartbeat_events": 0},
            "method_version": mod.REVIEW_VERSION,
        })
        review = mod._build_review("remii", int(time.time()) - 7 * 86400, "weekly")
        assert review["verdict"] in ("HEALTHY", "OBSERVE")
        assert review["verdict"] != "COLD/STANDBY"
        reasons = " ".join(review["reasons"]).lower()
        assert "dormant" not in reasons

    def test_cold_standby_worker(self, fake_home, monkeypatch):
        mod = _load_script()
        _add_profile(fake_home, "wesker-backup")
        _registry(fake_home, [
            {"name": "wesker-backup", "kind": "worker", "parent": "wesker",
             "lifecycle": "standby", "domains": ["backup"], "gateway_unit": None},
        ])
        monkeypatch.setattr(mod, "_runtime_dimension", lambda p, s: {
            "dimension": "runtime", "verdict": "INACTIVE",
            "evidence": {"unit_active": False, "heartbeat_events": 0},
            "method_version": mod.REVIEW_VERSION,
        })
        review = mod._build_review("wesker-backup", int(time.time()) - 7 * 86400, "weekly")
        assert review["verdict"] == "COLD/STANDBY"

    def test_failing_active_profile_action_required(self, fake_home, monkeypatch):
        mod = _load_script()
        _add_profile(fake_home, "octacon")
        monkeypatch.setattr(mod, "_runtime_dimension", lambda p, s: {
            "dimension": "runtime", "verdict": "ACTIVE", "evidence": {},
            "method_version": mod.REVIEW_VERSION,
        })
        monkeypatch.setattr(mod, "_quality_dimension", lambda p, s: {
            "dimension": "quality", "verdict": "ATTENTION",
            "evidence": {"failures": 4, "rework": 0, "governance_findings": 0},
        })
        review = mod._build_review("octacon", int(time.time()) - 7 * 86400, "weekly")
        assert review["verdict"] == "ACTION REQUIRED"

    def test_missing_config_capability_degraded(self, fake_home, monkeypatch):
        mod = _load_script()
        _add_profile(fake_home, "gojo-mailbox", config=False)  # empty/invalid config
        monkeypatch.setattr(mod, "_runtime_dimension", lambda p, s: {
            "dimension": "runtime", "verdict": "ACTIVE", "evidence": {},
            "method_version": mod.REVIEW_VERSION,
        })
        review = mod._build_review("gojo-mailbox", int(time.time()) - 7 * 86400, "weekly")
        assert review["verdict"] == "ACTION REQUIRED"
        assert any("capability" in r.lower() for r in review["reasons"])

    def test_direct_delegation_counts_as_workload(self, fake_home):
        mod = _load_script()
        _add_profile(fake_home, "quan-security")
        now = int(time.time())
        # Direct delegation events (Phase 2 identity-preserving telemetry)
        _seed(fake_home, "delegation.started", "kensei", "quan-security", now - 100, count=3)
        review = mod._build_review("quan-security", now - 7 * 86400, "weekly")
        wl = review["dimensions"]["workload"]
        assert wl["evidence"]["delegations_received"] == 3
        assert wl["verdict"] == "EVIDENT"
        assert review["verdict"] != "COLD/STANDBY"

    def test_recurring_findings_produce_observe_or_action(self, fake_home, monkeypatch):
        mod = _load_script()
        _add_profile(fake_home, "light")
        monkeypatch.setattr(mod, "_quality_dimension", lambda p, s: {
            "dimension": "quality", "verdict": "WATCH",
            "evidence": {"failures": 1, "rework": 2, "governance_findings": 0},
        })
        review = mod._build_review("light", int(time.time()) - 7 * 86400, "weekly")
        assert review["verdict"] == "OBSERVE"

    def test_absent_dimension_evidence_gives_observe_or_cold(self, fake_home):
        mod = _load_script()
        _add_profile(fake_home, "skill-broker")
        # No runtime (no unit), no ledger events at all.
        review = mod._build_review("skill-broker", int(time.time()) - 7 * 86400, "weekly")
        assert review["verdict"] == "COLD/STANDBY"

    def test_verdicts_within_allowed_set(self, fake_home, monkeypatch):
        mod = _load_script()
        _add_profile(fake_home, "octacon-testrunner")
        review = mod._build_review("octacon-testrunner", int(time.time()) - 7 * 86400, "monthly")
        assert review["verdict"] in mod.ALLOWED_VERDICTS

    def test_regression_no_synthetic_score(self, fake_home, monkeypatch):
        """Dimensions must not be collapsed into a numeric score."""
        mod = _load_script()
        _add_profile(fake_home, "octacon")
        review = mod._build_review("octacon", int(time.time()) - 7 * 86400, "weekly")

        def _walk(obj):
            if isinstance(obj, dict):
                assert "score" not in obj
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    _walk(v)
        _walk(review)

    def test_monthly_scope_excludes_retired(self, fake_home):
        mod = _load_script()
        _add_profile(fake_home, "moss")
        _add_profile(fake_home, "wesker")
        _registry(fake_home, [
            {"name": "moss", "kind": "worker", "parent": "octacon",
             "lifecycle": "retired", "domains": [], "gateway_unit": None},
            {"name": "wesker", "kind": "lead", "parent": "KENSEI",
             "lifecycle": "active", "domains": [], "gateway_unit": "u"},
        ])
        in_scope, note = mod._profiles_for_cycle(
            "monthly", ["moss", "wesker"]
        )
        # _profiles_for_cycle returns candidates; _run_cycle applies the
        # registry filter — assert the lifecycle map drives exclusion.
        lifecycle = mod._registry_lifecycle()
        monthly_scope = [
            p for p in ("moss", "wesker")
            if lifecycle.get(p, "active") in ("active", "standby")
        ]
        assert "moss" not in monthly_scope
        assert "wesker" in monthly_scope