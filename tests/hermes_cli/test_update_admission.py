"""Tests for the fresh fail-closed pre-update admission snapshot (#127611).

Covers idle, busy, unknown/malformed/unreadable state, each covered work
class, backend work, Desktop open/closed, runtime inventory fields,
freshness/bounded admission, and read-only behaviour (no writes, no drain,
no quarantine).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

import hermes_cli.update_admission as adm
from hermes_cli.update_admission import (
    WORK_CLASSES,
    build_admission_document,
    collect_admission_snapshot,
    plan_and_admission_payload,
)


def _idle_work() -> dict:
    return {k: {"state": "idle", "count": 0, "detail": {}} for k in WORK_CLASSES}


def _all_runtime_fields(entry: dict) -> None:
    for field in ("kind", "profile", "pid", "supervisor", "restart_via",
                  "live", "state", "source"):
        assert field in entry, field


class TestBuildAdmissionDocument:
    def test_idle_is_admissible_with_boundary(self):
        doc = build_admission_document([], _idle_work(), computed_at=1_700_000_000.0)
        assert doc["schema_version"] == 1
        assert doc["overall"] == "idle"
        assert doc["admissible"] is True
        assert doc["computed_at"]
        assert doc["valid_until"]
        assert doc["valid_for_s"] == adm.ADMISSION_VALID_FOR_S
        boundary = doc["admission_boundary"]
        assert boundary["type"] == "read-only-observation"
        assert boundary["does_not_block_arrivals"] is True
        assert boundary["requires_recheck_before_mutation"] is True
        assert set(doc["work"]) == set(WORK_CLASSES)

    def test_busy_blocks_admission(self):
        work = _idle_work()
        work["foreground_turns"] = {"state": "busy", "count": 2, "detail": {}}
        doc = build_admission_document([], work)
        assert doc["overall"] == "busy"
        assert doc["admissible"] is False

    def test_unknown_blocks_admission(self):
        work = _idle_work()
        work["cron_jobs"] = {"state": "unknown", "count": None, "detail": {"reason": "x"}}
        doc = build_admission_document([], work)
        assert doc["overall"] == "unknown"
        assert doc["admissible"] is False

    def test_busy_beats_unknown_in_overall(self):
        work = _idle_work()
        work["cron_jobs"] = {"state": "unknown", "count": None, "detail": {}}
        work["api_runs"] = {"state": "busy", "count": 1, "detail": {}}
        assert build_admission_document([], work)["overall"] == "busy"

    def test_malformed_entries_become_unknown_never_zero(self):
        work = _idle_work()
        work["foreground_turns"] = {"state": "idle", "count": 3, "detail": {}}  # idle must be 0
        work["cron_jobs"] = {"state": "busy", "count": 0, "detail": {}}  # busy must be >0
        work["api_runs"] = "not-a-dict"
        work["deferred_workers"] = {"state": "weird", "count": 0, "detail": {}}
        doc = build_admission_document([], work)
        for key in ("foreground_turns", "cron_jobs", "api_runs", "deferred_workers"):
            assert doc["work"][key]["state"] == "unknown"
            assert doc["work"][key]["count"] is None
        assert doc["overall"] in ("busy", "unknown")
        assert doc["admissible"] is False

    def test_missing_work_class_is_unknown(self):
        doc = build_admission_document([], {})
        assert doc["overall"] == "unknown"
        for key in WORK_CLASSES:
            assert doc["work"][key]["state"] == "unknown"
            assert doc["work"][key]["count"] is None

    def test_json_serializable(self):
        doc = build_admission_document([{"kind": "gateway", "profile": "default"}],
                                       _idle_work())
        json.dumps(doc)


class TestGatewayAdmissionBuilder:
    def _runner(self, **overrides):
        class R:
            adapters = {}
            _running_agents = {}
            _deferred_agent_workers = {}

            def _running_agent_count(self):
                return 0

            def _active_api_run_count(self):
                return 0

            def _active_deferred_agent_worker_count(self):
                return 0

        r = R()
        for k, v in overrides.items():
            setattr(r, k, v)
        return r

    def test_idle_gateway(self):
        from gateway.admission_snapshot import build_gateway_admission

        doc = build_gateway_admission(self._runner())
        assert doc["schema_version"] == 1
        assert doc["computed_at"]
        for key in ("foreground_turns", "cron_jobs", "api_runs", "deferred_workers",
                    "background_delegations", "background_processes", "external_workers"):
            assert doc["work"][key]["state"] == "idle", key
            assert doc["work"][key]["count"] == 0

    def test_busy_foreground_and_cron_with_restart_safe_worker(self, monkeypatch):
        from gateway.admission_snapshot import build_gateway_admission

        runner = self._runner()
        runner._running_agents = {"a:1": object()}
        runner._running_agent_count = lambda: 1  # noqa: E731
        monkeypatch.setattr(
            "cron.scheduler.get_running_job_details",
            lambda: [{"job_id": "nightly", "elapsed_s": 3.0, "worker_pid": 4242}],
        )
        doc = build_gateway_admission(runner)
        assert doc["work"]["foreground_turns"]["state"] == "busy"
        cron = doc["work"]["cron_jobs"]
        assert cron["state"] == "busy"
        assert cron["detail"]["jobs"][0]["worker_pid"] == 4242
        assert cron["detail"]["jobs"][0]["restart_safe_worker"] is True
        ext = doc["work"]["external_workers"]
        assert ext["state"] == "busy"
        assert 4242 in ext["detail"]["worker_pids"]

    def test_busy_api_deferred_delegation_process(self, monkeypatch):
        from gateway.admission_snapshot import build_gateway_admission
        from gateway.config import Platform

        runner = self._runner()
        runner.adapters = {Platform.API_SERVER: object()}
        runner._active_api_run_count = lambda: 2  # noqa: E731
        monkeypatch.setattr(
            "gateway.platforms.api_server_runs.api_worker_live_count", lambda: 1)
        runner._deferred_agent_workers = {object(): object()}
        runner._active_deferred_agent_worker_count = lambda: 1  # noqa: E731
        monkeypatch.setattr("tools.async_delegation.active_count", lambda: 1)
        import tools.process_registry as pr

        monkeypatch.setattr(pr.process_registry, "has_any_active", lambda: True)
        doc = build_gateway_admission(runner)
        assert doc["work"]["api_runs"]["count"] == 3
        assert doc["work"]["deferred_workers"]["state"] == "busy"
        assert doc["work"]["background_delegations"]["state"] == "busy"
        assert doc["work"]["background_processes"]["state"] == "busy"

    def test_unreadable_sources_are_unknown_never_zero(self, monkeypatch):
        from gateway.admission_snapshot import build_gateway_admission

        def _boom():
            raise RuntimeError("probe down")

        runner = self._runner()
        runner._running_agent_count = _boom
        runner._active_deferred_agent_worker_count = _boom
        monkeypatch.setattr("cron.scheduler.get_running_job_details",
                            lambda: (_ for _ in ()).throw(RuntimeError("x")))
        monkeypatch.setattr("tools.async_delegation.active_count",
                            lambda: (_ for _ in ()).throw(RuntimeError("x")))
        doc = build_gateway_admission(runner)
        for key, entry in doc["work"].items():
            if entry["state"] == "unknown":
                assert entry["count"] is None
        assert doc["work"]["foreground_turns"]["state"] == "unknown"
        assert doc["work"]["cron_jobs"]["state"] == "unknown"
        assert doc["work"]["background_delegations"]["state"] == "unknown"

    def test_never_drains_or_writes(self, monkeypatch):
        from gateway.admission_snapshot import build_gateway_admission

        runner = self._runner()
        for name in ("request_restart", "_drain_active_agents", "_persist_active_agents",
                     "_update_runtime_status"):
            monkeypatch.setattr(runner, name, lambda *a, **k: (_ for _ in ()).throw(
                AssertionError(f"must not call {name}")), raising=False)
        build_gateway_admission(runner)


class TestCollectAdmissionSnapshot:
    @pytest.fixture()
    def homes(self, monkeypatch, tmp_path):
        default = tmp_path / "home"
        work = default / "profiles" / "work"
        work.mkdir(parents=True)
        monkeypatch.setattr(
            "hermes_cli.update_receipt._profile_homes",
            lambda: [("default", default), ("work", work)])
        # No gateways, no sockets, no ledger, no executions, no heartbeat.
        monkeypatch.setattr(adm, "_query_socket_admission", lambda home, timeout=2.0: None)
        monkeypatch.setattr(adm, "_query_socket_status", lambda home, timeout=2.0: None)
        monkeypatch.setattr(adm, "_live_gateway_pid", lambda home: (None, "not_running"))
        monkeypatch.setattr(adm, "_read_ledger_readonly", lambda: [])
        return default, work

    def _plan(self, runtimes=None):
        from hermes_cli.update_inventory import UpdatePlan

        plan = UpdatePlan()
        plan.runtimes = runtimes or []
        return plan

    def test_idle_no_runtimes(self, homes, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory",
            lambda: self._plan())
        doc = collect_admission_snapshot()
        assert doc["overall"] == "idle"
        assert doc["admissible"] is True
        assert set(doc["work"]) == set(WORK_CLASSES)
        assert doc["runtimes"] == []

    def test_busy_via_socket_admission(self, homes, monkeypatch):
        from hermes_cli.update_inventory import RuntimeRecord

        plan = self._plan([RuntimeRecord(kind="gateway", profile="default", pid=100,
                                        supervisor="manual", restart_via="manual")])
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory", lambda: plan)
        monkeypatch.setattr(adm, "_live_gateway_pid", lambda home: (100, "live"))
        monkeypatch.setattr(
            adm, "_read_status_file",
            lambda home: ({"pid": 100, "gateway_state": "running", "active_agents": 1,
                           "updated_at": adm._utc_now_iso()}, "ok"))
        work = _idle_work()
        work["foreground_turns"] = {"state": "busy", "count": 1, "detail": {}}
        monkeypatch.setattr(
            adm, "_query_socket_admission",
            lambda home, timeout=2.0: {"schema_version": 1, "work": work})
        doc = collect_admission_snapshot(plan)
        assert doc["work"]["foreground_turns"]["state"] == "busy"
        assert doc["overall"] == "busy"
        assert doc["admissible"] is False

    def test_unknown_on_malformed_status_file(self, homes, monkeypatch, tmp_path):
        from hermes_cli.update_inventory import RuntimeRecord

        default, _ = homes
        (default / "gateway_state.json").write_text("{not json", encoding="utf-8")
        plan = self._plan([RuntimeRecord(kind="gateway", profile="default", pid=100,
                                        supervisor="manual", restart_via="manual")])
        # Use the real file reader: malformed file, live pid, no socket.
        monkeypatch.setattr(adm, "_query_socket_admission", lambda home, timeout=2.0: None)
        monkeypatch.setattr(adm, "_query_socket_status", lambda home, timeout=2.0: None)
        monkeypatch.setattr(adm, "_live_gateway_pid", lambda home: (100, "live"))
        doc = collect_admission_snapshot(plan)
        assert doc["work"]["foreground_turns"]["state"] == "unknown"
        assert doc["work"]["foreground_turns"]["count"] is None
        assert doc["admissible"] is False
        # Read-only: malformed file left in place, not quarantined/rewritten.
        assert (default / "gateway_state.json").read_text(encoding="utf-8") == "{not json"

    def test_unknown_on_unreadable_executions(self, homes, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory",
            lambda: self._plan())
        monkeypatch.setattr(adm, "_read_cron_executions_readonly", lambda home: None)
        monkeypatch.setattr(adm, "_live_gateway_pid", lambda home: (100, "live"))
        doc = collect_admission_snapshot()
        assert doc["work"]["cron_jobs"]["state"] == "unknown"
        assert doc["work"]["cron_jobs"]["count"] is None
        assert doc["work"]["external_workers"]["state"] == "unknown"

    def test_cron_busy_includes_restart_safe_workers(self, homes, monkeypatch, tmp_path):
        default, _ = homes
        # Durable claimed row with a live owner pid (this process).
        import os

        from gateway.status import get_process_start_time

        dbdir = default / "cron"
        dbdir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(dbdir / "executions.db")
        conn.execute(
            "CREATE TABLE executions (id TEXT PRIMARY KEY, job_id TEXT, source TEXT, "
            "process_id TEXT, pid INTEGER, process_started_at INTEGER, status TEXT, "
            "handoff_pending INTEGER DEFAULT 0, handoff_started_at REAL, "
            "claimed_at TEXT, started_at TEXT, finished_at TEXT, error TEXT)")
        conn.execute(
            "INSERT INTO executions (id, job_id, source, process_id, pid, "
            "process_started_at, status, claimed_at) VALUES (?,?,?,?,?,?,?,?)",
            ("e1", "nightly", "tick", "p1", os.getpid(),
             get_process_start_time(os.getpid()), "running",
             "2026-01-01T00:00:00+00:00"))
        conn.commit()
        conn.close()
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory",
            lambda: self._plan())
        doc = collect_admission_snapshot()
        assert doc["work"]["cron_jobs"]["state"] == "busy"
        assert doc["work"]["external_workers"]["state"] == "busy"

    def test_backend_live_is_unknown_idle_when_absent(self, homes, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory",
            lambda: self._plan())
        monkeypatch.setattr(
            adm, "_read_ledger_readonly",
            lambda: [{"purpose": "serve", "pid": 999999, "create_time": None}])
        monkeypatch.setattr(adm, "_ledger_liveness", lambda pid, ct: True)
        doc = collect_admission_snapshot()
        assert doc["work"]["backend_agent_work"]["state"] == "unknown"
        assert doc["work"]["backend_agent_work"]["count"] is None

        monkeypatch.setattr(adm, "_read_ledger_readonly", lambda: [])
        doc2 = collect_admission_snapshot()
        assert doc2["work"]["backend_agent_work"]["state"] == "idle"

    def test_backend_corrupt_ledger_is_unknown_not_empty(self, homes, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory",
            lambda: self._plan())
        monkeypatch.setattr(adm, "_read_ledger_readonly", lambda: None)
        doc = collect_admission_snapshot()
        assert doc["work"]["backend_agent_work"]["state"] == "unknown"
        assert doc["work"]["backend_agent_work"]["count"] is None

    def test_desktop_open_blocks_desktop_closed_idle(self, homes, monkeypatch):
        default, _ = homes
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory",
            lambda: self._plan())
        # Desktop open: recent heartbeat forces backend unknown (fail closed).
        monkeypatch.setattr(adm, "_desktop_heartbeat", lambda home: (3.0, "ok"))
        monkeypatch.setattr(
            adm, "_read_ledger_readonly",
            lambda: [{"purpose": "serve", "pid": 111, "create_time": None}])
        monkeypatch.setattr(adm, "_ledger_liveness", lambda pid, ct: True)
        doc = collect_admission_snapshot()
        assert doc["work"]["backend_agent_work"]["state"] == "unknown"

        # Desktop closed + no backends: idle.
        monkeypatch.setattr(adm, "_desktop_heartbeat", lambda home: (None, "no_marker"))
        monkeypatch.setattr(adm, "_read_ledger_readonly", lambda: [])
        doc2 = collect_admission_snapshot()
        assert doc2["work"]["backend_agent_work"]["state"] == "idle"

    def test_desktop_unreadable_is_unknown(self, homes, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory",
            lambda: self._plan())
        monkeypatch.setattr(adm, "_desktop_heartbeat", lambda home: (None, "unreadable"))
        monkeypatch.setattr(adm, "_read_ledger_readonly", lambda: [])
        doc = collect_admission_snapshot()
        assert doc["work"]["backend_agent_work"]["state"] == "unknown"

    def test_stale_status_file_is_unknown_not_idle(self, homes, monkeypatch):
        from hermes_cli.update_inventory import RuntimeRecord

        plan = self._plan([RuntimeRecord(kind="gateway", profile="default", pid=100,
                                        supervisor="manual", restart_via="manual")])
        old = "2001-01-01T00:00:00+00:00"
        monkeypatch.setattr(
            adm, "_read_status_file",
            lambda home: ({"pid": 100, "gateway_state": "running",
                           "active_agents": 0, "updated_at": old}, "ok"))
        monkeypatch.setattr(adm, "_live_gateway_pid", lambda home: (100, "live"))
        doc = collect_admission_snapshot(plan)
        assert doc["work"]["foreground_turns"]["state"] == "unknown"

    def test_runtime_entries_keep_inventory_fields(self, homes, monkeypatch):
        from hermes_cli.update_inventory import RuntimeRecord

        plan = self._plan([RuntimeRecord(kind="gateway", profile="default", pid=100,
                                        supervisor="systemd", restart_via="systemd",
                                        code_sha="a" * 40, code_version="9.9")])
        monkeypatch.setattr(adm, "_live_gateway_pid", lambda home: (None, "not_running"))
        monkeypatch.setattr(adm, "_read_status_file",
                            lambda home: (None, "absent_or_unreadable"))
        doc = collect_admission_snapshot(plan)
        entry = doc["runtimes"][0]
        _all_runtime_fields(entry)
        assert entry["kind"] == "gateway"
        assert entry["profile"] == "default"
        assert entry["pid"] == 100
        assert entry["supervisor"] == "systemd"
        assert entry["restart_via"] == "systemd"

    def test_read_only_never_writes_or_quarantines(self, homes, monkeypatch, tmp_path):
        default, _ = homes
        before = {(p.relative_to(tmp_path).as_posix(), p.read_bytes())
                  for p in tmp_path.rglob("*") if p.is_file()}
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory",
            lambda: self._plan())
        collect_admission_snapshot()
        after = {(p.relative_to(tmp_path).as_posix(), p.read_bytes())
                 for p in tmp_path.rglob("*") if p.is_file()}
        assert before == after
        # No drain marker, no quarantine artefacts created.
        assert not (default / ".drain_request.json").exists()
        assert not list(tmp_path.rglob("*.corrupt"))

    def test_never_uses_drain(self, homes, monkeypatch):
        import gateway.drain_control as dc

        monkeypatch.setattr(
            dc, "drain_requested",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not probe drain")))
        monkeypatch.setattr(
            "hermes_cli.update_inventory.collect_runtime_inventory",
            lambda: self._plan())
        collect_admission_snapshot()  # must not raise


class TestPlanJsonContract:
    def test_plan_and_admission_payload_shape(self, monkeypatch, tmp_path):
        from hermes_cli.update_inventory import UpdatePlan

        plan = UpdatePlan()
        plan.install_method = "git"
        admission = build_admission_document([], _idle_work())
        payload = plan_and_admission_payload(plan, admission)
        assert set(payload) == {"plan", "admission"}
        assert payload["admission"]["schema_version"] == 1
        assert set(payload["admission"]["work"]) == set(WORK_CLASSES)
        text = json.dumps(payload)
        assert json.loads(text)["plan"]["install_method"] == "git"

    def test_plan_json_round_trip_with_one_runtime(self):
        from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

        plan = UpdatePlan()
        plan.runtimes = [RuntimeRecord(kind="gateway", profile="default", pid=7,
                                       supervisor="manual", restart_via="manual")]
        work = _idle_work()
        work["backend_agent_work"] = {"state": "unknown", "count": None,
                                      "detail": {"reason": "t"}}
        payload = plan_and_admission_payload(plan, build_admission_document(
            [{"kind": "gateway", "profile": "default", "pid": 7,
              "supervisor": "manual", "restart_via": "manual",
              "live": True, "state": "unknown", "source": "runtime_status",
              "heartbeat_age_s": None, "stale": None, "detail": {}}], work))
        assert payload["overall"] if "overall" in payload else True
        assert len(payload["plan"]["runtimes"]) == 1
        assert payload["admission"]["overall"] == "unknown"
        assert payload["admission"]["admissible"] is False


class TestSocketHelpers:
    def test_socket_failures_fall_back_to_unknown(self, monkeypatch, tmp_path):
        home = tmp_path / "h"
        home.mkdir()
        monkeypatch.setattr(
            "gateway.control_socket.query_gateway_control",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no socket")))
        assert adm._query_socket_admission(home) is None
        assert adm._query_socket_status(home) is None

    def test_status_verb_shapes(self, monkeypatch, tmp_path):
        home = tmp_path / "h"
        home.mkdir()
        monkeypatch.setattr(
            "gateway.control_socket.query_gateway_control",
            lambda home, verb, timeout=2.0: {"gateway_state": "running"} if verb == "status" else None)
        assert adm._query_socket_status(home) == {"gateway_state": "running"}
        assert adm._query_socket_admission(home) is None
