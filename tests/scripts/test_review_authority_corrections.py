"""C8 adversarial tests — P3.2 review uses real authority/taxonomy/scope."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "denji-review-cycle.py"
sys.path.insert(0, str(REPO))


def _load():
    spec = importlib.util.spec_from_file_location("drc_c8", str(SCRIPT))
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


class TestC8RegistryGatewayUnit:
    def test_runtime_uses_registry_gateway_unit(self, fake_home, monkeypatch):
        """RED: registry declares a nonstandard unit name — runtime must
        check THAT unit, not hermes-gateway-<profile>."""
        mod = _load()
        (fake_home / "profiles" / "odd").mkdir()
        _registry(fake_home, [
            {"name": "odd", "kind": "lead", "parent": "KENSEI",
             "lifecycle": "active", "domains": [],
             "gateway_unit": "hermes-gateway-oddname.service"},
        ])
        checked = {}
        def fake_run(cmd, **kwargs):
            checked["unit"] = cmd[2] if len(cmd) > 2 else None
            class R:
                returncode = 0
                stdout = "active\n"
            return R()
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        mod._runtime_dimension("odd", int(time.time()) - 86400)
        assert checked["unit"] == "hermes-gateway-oddname.service"

    def test_default_root_profile_maps_to_base_gateway(self, fake_home, monkeypatch):
        """RED: the default/root profile must map to hermes-gateway.service,
        not hermes-gateway-default.service."""
        mod = _load()
        (fake_home / "config.yaml").write_text("model:\n  default: t\n")
        _registry(fake_home, [])
        checked = {}
        def fake_run(cmd, **kwargs):
            checked["unit"] = cmd[2] if len(cmd) > 2 else None
            class R:
                returncode = 1
                stdout = "inactive\n"
            return R()
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        mod._runtime_dimension("default", int(time.time()) - 86400)
        assert checked["unit"] == "hermes-gateway.service"


class TestC8FindingTaxonomy:
    def test_phase2_finding_taxonomy_counts(self, fake_home, monkeypatch):
        """RED: three real governance.finding.opened events must NOT yield CLEAN."""
        mod = _load()
        (fake_home / "profiles" / "octacon").mkdir()
        from hermes_cli.profile_activity_ledger import append_event
        now = int(time.time())
        for i in range(3):
            append_event(
                source="t", event_type="governance.finding.opened",
                event_id=f"c8-{i}-{time.time_ns()}",
                actor_profile="octacon", target_profile="octacon",
                occurred_at=now - 100 - i,
            )
        dim = mod._quality_dimension("octacon", now - 86400)
        assert dim["evidence"]["governance_findings_open"] >= 3
        assert dim["verdict"] in ("WATCH", "ATTENTION")

    def test_resolved_findings_do_not_escalate(self, fake_home):
        mod = _load()
        (fake_home / "profiles" / "octacon").mkdir()
        from hermes_cli.profile_activity_ledger import append_event
        now = int(time.time())
        # one open + three resolved: recurrence semantics = OPEN findings count
        append_event(source="t", event_type="governance.finding.opened",
                     event_id=f"c8o-{time.time_ns()}", actor_profile="octacon",
                     target_profile="octacon", occurred_at=now - 100)
        for i in range(3):
            append_event(source="t", event_type="governance.finding.resolved",
                         event_id=f"c8r-{i}-{time.time_ns()}",
                         actor_profile="octacon", target_profile="octacon",
                         occurred_at=now - 90 + i)
        dim = mod._quality_dimension("octacon", now - 86400)
        # opened(1) and resolved(3) in-window: nothing remains open
        assert dim["evidence"]["governance_findings_open"] == 0
        assert dim["evidence"]["governance_findings_resolved"] == 3
        assert dim["verdict"] in ("CLEAN", "WATCH")  # never ATTENTION from resolved work


class TestC8Scope:
    def test_monthly_scope_includes_changed_frozen_profile(self, fake_home):
        """RED: a changed frozen/retired profile must be in monthly scope."""
        mod = _load()
        (fake_home / "profiles" / "moss").mkdir()
        (fake_home / "profiles" / "moss" / "config.yaml").write_text(
            "model:\n  default: t\n")
        # config mtime now → changed inside window
        _registry(fake_home, [
            {"name": "moss", "kind": "worker", "parent": "octacon",
             "lifecycle": "frozen", "domains": [], "gateway_unit": None},
        ])
        in_scope = mod._monthly_scope(
            ["moss"], int(time.time()) - 7 * 86400, lifecycle_map=mod._registry_lifecycle()
        )
        assert "moss" in in_scope  # changed → included despite frozen

    def test_monthly_scope_excludes_unchanged_frozen(self, fake_home):
        mod = _load()
        (fake_home / "profiles" / "moss").mkdir()
        (fake_home / "profiles" / "moss" / "config.yaml").write_text(
            "model:\n  default: t\n")
        import os
        old = int(time.time()) - 90 * 86400
        os.utime(fake_home / "profiles" / "moss" / "config.yaml", (old, old))
        _registry(fake_home, [
            {"name": "moss", "kind": "worker", "parent": "octacon",
             "lifecycle": "frozen", "domains": [], "gateway_unit": None},
        ])
        in_scope = mod._monthly_scope(
            ["moss"], int(time.time()) - 7 * 86400, lifecycle_map=mod._registry_lifecycle()
        )
        assert "moss" not in in_scope


class TestC8UnknownRuntime:
    def test_uncheckable_runtime_is_unknown_not_inactive(self, fake_home, monkeypatch):
        mod = _load()
        (fake_home / "profiles" / "octacon").mkdir()
        def broken_run(cmd, **kwargs):
            raise RuntimeError("systemctl unavailable")
        monkeypatch.setattr(mod.subprocess, "run", broken_run)
        dim = mod._runtime_dimension("octacon", int(time.time()) - 86400)
        assert dim["verdict"] == "UNKNOWN"