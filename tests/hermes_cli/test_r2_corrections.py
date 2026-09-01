"""R2 adversarial RED witnesses — Round 2 corrections.

R2-1  shadow exactly-once atomic under concurrent writers
R2-2  schema install must not commit the caller's transaction
R2-3  fail-closed legacy duplicate migration
R2-4  gate proves the full review→requirement→evidence chain
R2-5  pilot provenance not self-mintable
R2-6  emit_trigger honours the supplied home
R2-7  human provenance via validated registry + genuinely human seam
R2-8  dashboard raw-list validation + safe invalid root
R2-9  strict force_mode semantics
R2-10 identity/state finding recurrence
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import hermaguard_events as he
from hermes_cli import shadow_classifier as sc
from hermes_cli import pilot_evidence as pe


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: True)
    conn = kb.connect()
    yield conn, home
    conn.close()


def _task(conn, tier="full", kind="task", created_by="dashboard"):
    return kb.create_task(
        conn, title="t", assignee="octacon",
        body="## Problem\nx\n## Success Criteria\ny", triage=True,
        tier=tier, task_kind=kind, created_by=created_by,
    )


def _review(conn, tid):
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    conn.commit()
    kb.request_review(conn, tid, force=True)
    conn.commit()
    row = conn.execute(
        "SELECT id FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _report(home, tid, content=b"hermaguard report"):
    artifact = home / "feature-artifacts" / tid
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "hermaguard-report.md").write_bytes(content)
    return artifact


# ── R2-1 ─────────────────────────────────────────────────────────────────────

class TestR21ShadowAtomicity:
    def test_two_connections_no_duplicate_suggestion(self, env):
        """RED: two racing writers produce duplicate same-version suggestions."""
        conn, home = env
        tid = _task(conn)
        conn.commit()
        db_path = str(conn.execute("PRAGMA database_list").fetchone()[2])
        barrier = threading.Barrier(2)

        def worker():
            con2 = sqlite3.connect(db_path, timeout=10)
            try:
                barrier.wait()
                s = sc.suggest(con2, tid)
                if s is not None:
                    sc.insert_shadow_event(con2, tid, s)
                    con2.commit()
            finally:
                con2.close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rows = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, sc.EVENT_KIND),
        ).fetchone()[0]
        assert rows == 1


# ── R2-2 ─────────────────────────────────────────────────────────────────────

class TestR22CallerTransaction:
    def test_ensure_does_not_commit_caller_txn(self, env):
        """RED: executescript() commits the caller's pending write, so a
        later rollback cannot undo it."""
        conn, home = env
        tid = _task(conn)
        # kanban_db uses isolation_level=None (autocommit); emulate a caller
        # with an explicit open transaction the way write_txn callers do.
        conn.execute("BEGIN")
        conn.execute("UPDATE tasks SET title = 'canary-title' WHERE id = ?", (tid,))
        # caller has an UNCOMMITTED write inside an explicit transaction
        he.ensure_uniqueness_invariants(conn)
        conn.rollback()
        title = conn.execute("SELECT title FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        assert title != "canary-title"  # rollback must undo the canary


# ── R2-3 ─────────────────────────────────────────────────────────────────────

class TestR23LegacyDuplicates:
    def test_index_install_fails_closed_on_duplicates(self, env):
        """RED: legacy DB with duplicate requirement rows must fail with a
        bounded diagnostic, not crash every later call."""
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        # Inject duplicate requirement rows directly (legacy residue)
        payload = json.dumps({"review_event_id": rid, "policy_version": "old"})
        for _ in range(2):
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, ?, ?, strftime('%s','now'))",
                (tid, he.KIND_REQUIRED, payload),
            )
        conn.commit()
        result = he.ensure_uniqueness_invariants_safe(conn)
        assert result["installed"] is False
        assert result["duplicate_groups"] >= 1
        # retry must be deterministic and non-destructive
        result2 = he.ensure_uniqueness_invariants_safe(conn)
        assert result2["installed"] is False
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, he.KIND_REQUIRED)).fetchone()[0] == 2  # never deduped silently

    def test_migration_plan_reports_safe_ids_only(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        payload = json.dumps({"review_event_id": rid, "policy_version": "old"})
        for _ in range(2):
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, ?, ?, strftime('%s','now'))",
                (tid, he.KIND_REQUIRED, payload),
            )
        conn.commit()
        plan = he.migration_plan(conn)
        assert plan["duplicate_groups"] >= 1
        assert plan["affected_task_ids"] == [tid]  # safe IDs, no payloads


# ── R2-4 ─────────────────────────────────────────────────────────────────────

class TestR24GateChain:
    def test_forged_evidence_without_requirement_blocked(self, env):
        """RED: structurally valid evidence row with zero requirement rows
        must NOT open the gate."""
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        artifact = home / "feature-artifacts" / tid
        artifact.mkdir(parents=True, exist_ok=True)
        report = artifact / "hermaguard-report.md"
        report.write_bytes(b"forged evidence")
        import hashlib
        digest = hashlib.sha256(b"forged evidence").hexdigest()
        # Directly insert a valid-looking evidence row — no requirement exists
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, ?, ?, strftime('%s','now'))",
            (tid, he.KIND_EVIDENCE,
             json.dumps({"review_event_id": rid, "report": "hermaguard-report.md",
                         "report_sha256": digest, "status": "pass",
                         "version": "1.0.0"})),
        )
        conn.commit()
        ok, reason = he.gate_review_transition(
            home / "kanban.db", conn, tid, review_event_id=rid,
            artifact_dir=artifact,
            kanban_cfg={"hermaguard_event_mode": True},
        )
        assert ok is False  # forged evidence must not open the gate


# ── R2-5 ─────────────────────────────────────────────────────────────────────

class TestR25ProvenanceForgery:
    def test_public_mint_does_not_confer_authority(self, env):
        """RED: the public mint helper must not confer authority — keyless
        minting raises, and attacker-keyed provenance fails the gate."""
        conn, home = env
        tid = _task(conn)
        record = {
            "task_id": tid,
            "pipeline_contract_version": "c1",
            "final_status": "done",
            "reviewer_verdict": "pass",
            "label": None,
            "ever_simulated": False,
        }
        # Keyless mint is refused outright
        with pytest.raises(ValueError):
            pe.mint_live_pilot_provenance(
                tid, review_event_id=1, evidence_event_id=2,
                pipeline_contract_version="c1", authorised_by="attacker",
            )
        # Attacker minting with their OWN key still fails the gate (the gate
        # verifies with the TRUSTED key, not the attacker's)
        provenance = pe.mint_live_pilot_provenance(
            tid, review_event_id=1, evidence_event_id=2,
            pipeline_contract_version="c1", authorised_by="attacker",
            signing_key=b"attacker-key",
        )
        record["live_provenance"] = provenance
        assert pe.real_pilot_gate_satisfied(record, signing_key=b"trusted-key") is False

    def test_valid_trusted_provenance_passes(self):
        record = {
            "task_id": "t1", "pipeline_contract_version": "c1",
            "final_status": "done", "reviewer_verdict": "pass",
            "label": None, "ever_simulated": False,
        }
        prov = pe.mint_live_pilot_provenance(
            "t1", review_event_id=1, evidence_event_id=2,
            pipeline_contract_version="c1", authorised_by="sahil",
            signing_key=b"trusted-key",
        )
        record["live_provenance"] = prov
        assert pe.real_pilot_gate_satisfied(record, signing_key=b"trusted-key") is True

    def test_wrong_key_rejected(self, env):
        record = {
            "task_id": "t1", "pipeline_contract_version": "c1",
            "final_status": "done", "reviewer_verdict": "pass",
            "label": None, "ever_simulated": False,
        }
        # signed with key A, verified with key B
        from hermes_cli import pilot_evidence as pe
        prov = pe.mint_live_pilot_provenance(
            "t1", review_event_id=1, evidence_event_id=2,
            pipeline_contract_version="c1", authorised_by="sahil",
            signing_key=b"key-a",
        )
        record["live_provenance"] = prov
        assert pe.real_pilot_gate_satisfied(record, signing_key=b"key-b") is False


# ── R2-6 ─────────────────────────────────────────────────────────────────────

class TestR26ExplicitHome:
    def test_emit_trigger_writes_to_supplied_home(self, env, tmp_path, monkeypatch):
        """RED: emit_trigger(..., hermes_home=B) must write/verify against B,
        not the process HERMES_HOME."""
        import importlib.util
        import sys
        home_a = tmp_path / "home-a"
        (home_a / "profiles" / "octacon").mkdir(parents=True)
        (home_a / "governance").mkdir()
        (home_a / "profiles" / "octacon" / "config.yaml").write_text("model:\n  default: t\n")
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        home_b = tmp_path / "home-b"
        (home_b / "governance").mkdir(parents=True)

        REPO = "/home/kensei/worktrees/governance-evidence-spine-p34-corrections-r2-agent"
        spec = importlib.util.spec_from_file_location(
            "set_eval_r26", f"{REPO}/scripts/denji-self-eval-trigger.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import time
        now = int(time.time())
        from hermes_cli.profile_activity_ledger import append_event
        for i in range(4):
            append_event(source="t", event_type="kanban.crashed",
                         event_id=f"r26-{i}-{time.time_ns()}",
                         actor_profile="octacon", target_profile="x",
                         occurred_at=now - 100 - i)
        d = mod.decide_trigger("octacon", since=now - 86400,
                               hermes_home=home_a, now=now)
        assert d["trigger"] is True
        eid = mod.emit_trigger(d, hermes_home=home_b)
        # Event must land in home B, not A
        con_b = sqlite3.connect(f"file:{home_b}/governance/profile-activity-ledger.sqlite?mode=ro", uri=True)
        rows_b = con_b.execute(
            "SELECT COUNT(*) FROM activity_events WHERE event_type='profile.self_eval.trigger'").fetchone()[0]
        con_b.close()
        assert rows_b == 1


# ── R2-7 ─────────────────────────────────────────────────────────────────────

class TestR27HumanProvenance:
    def test_malformed_registry_cannot_authorise_spoofed_name(self, env, tmp_path):
        """RED: malformed registry (invalid entry) must yield no eligibility."""
        conn, home = env
        gov = home / "governance"
        gov.mkdir(parents=True, exist_ok=True)
        # Malformed registry: kensei-review listed as profile with bad enum
        (gov / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: KENSEI, description: r}\n"
            "profiles:\n"
            "- {name: webhook-7831, kind: emperor, parent: KENSEI, lifecycle: zombie, domains: [], gateway_unit: null}\n"
        )
        tid = _task(conn, created_by="webhook-7831")
        assert sc.suggest(conn, tid) is None

    def test_registry_membership_alone_not_human(self, env, tmp_path):
        """RED: a registry-verified worker profile's stamp is not proof of
        interactive human creation."""
        conn, home = env
        gov = home / "governance"
        gov.mkdir(parents=True, exist_ok=True)
        import yaml
        (gov / "profile-registry.yaml").write_text(yaml.safe_dump({
            "schema_version": 1,
            "root": {"name": "KENSEI", "description": "r"},
            "profiles": [
                {"name": "octacon-backend", "kind": "worker", "parent": "KENSEI",
                 "lifecycle": "active", "domains": [], "gateway_unit": None},
            ],
        }))
        tid = _task(conn, created_by="octacon-backend")
        assert sc.suggest(conn, tid) is None  # membership ≠ human

    def test_interactive_marker_enables_profile_author(self, env, tmp_path, monkeypatch):
        conn, home = env
        gov = home / "governance"
        gov.mkdir(parents=True, exist_ok=True)
        import yaml
        (gov / "profile-registry.yaml").write_text(yaml.safe_dump({
            "schema_version": 1,
            "root": {"name": "KENSEI", "description": "r"},
            "profiles": [
                {"name": "misa-misa", "kind": "lead", "parent": "KENSEI",
                 "lifecycle": "active", "domains": [], "gateway_unit": None},
            ],
        }))
        # Task created by registry-verified author WITH an interactive marker
        tid = _task(conn, created_by="misa-misa")
        monkeypatch.setattr(sc, "_task_has_interactive_marker", lambda c, t: True)
        assert sc.suggest(conn, tid) is not None
        # Without the marker: not human
        monkeypatch.setattr(sc, "_task_has_interactive_marker", lambda c, t: False)
        assert sc.suggest(conn, tid) is None


# ── R2-8 ─────────────────────────────────────────────────────────────────────

class TestR28DashboardValidation:
    @pytest.fixture
    def dash(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes-home"
        (home / "profiles").mkdir(parents=True)
        (home / "governance").mkdir()
        import sys
        sys.path.insert(0, "/home/kensei/worktrees/governance-evidence-spine-p34-corrections-r2-dashboard")
        # R2-8 isolation: 'backend' may already be cached from another
        # selector file (module-name collision) — force a fresh import of
        # the r2 dashboard's backend for this fixture.
        import importlib
        saved_backend = sys.modules.pop("backend", None)
        from backend import profile_docs
        monkeypatch.setattr(profile_docs, "HERMES_HOME", home)
        monkeypatch.setattr(profile_docs, "PROFILES_DIR", home / "profiles")
        monkeypatch.setattr(profile_docs, "GATEWAY_PROFILES", ["octacon"])
        yield home, profile_docs
        sys.modules.pop("backend", None)
        if saved_backend is not None:
            sys.modules["backend"] = saved_backend
        sys.path.remove("/home/kensei/worktrees/governance-evidence-spine-p34-corrections-r2-dashboard")

    def test_duplicate_entries_rejected(self, dash):
        home, profile_docs = dash
        (home / "profiles" / "octacon").mkdir()
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: KENSEI, description: r}\n"
            "profiles:\n"
            "- {name: octacon, kind: lead, parent: KENSEI, lifecycle: active, domains: [], gateway_unit: null}\n"
            "- {name: octacon, kind: lead, parent: KENSEI, lifecycle: active, domains: [], gateway_unit: null}\n"
        )
        h = profile_docs.profile_hierarchy()
        node = next(n for n in h["nodes"] if n["name"] == "octacon")
        assert node.get("registry_state") == "invalid_registry"

    def test_missing_required_fields_rejected(self, dash):
        home, profile_docs = dash
        (home / "profiles" / "octacon").mkdir()
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: KENSEI, description: r}\n"
            "profiles:\n"
            "- {name: octacon, kind: lead}\n"  # missing parent/lifecycle/domains/gateway_unit
        )
        h = profile_docs.profile_hierarchy()
        node = next(n for n in h["nodes"] if n["name"] == "octacon")
        assert node.get("registry_state") == "invalid_registry"

    def test_invalid_registry_safe_root(self, dash):
        """RED: invalid registry must NOT control the returned root."""
        home, profile_docs = dash
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\n"
            "root: {name: EVIL_ROOT, description: r}\n"
            "profiles:\n"
            "- {name: octacon, kind: emperor, parent: KENSEI, lifecycle: active, domains: [], gateway_unit: null}\n"
        )
        h = profile_docs.profile_hierarchy()
        assert h["root"] == "KENSEI"  # safe constant

    def test_empty_profiles_rejected(self, dash):
        home, profile_docs = dash
        (home / "governance" / "profile-registry.yaml").write_text(
            "schema_version: 1\nroot: {name: KENSEI, description: r}\nprofiles: []\n"
        )
        h = profile_docs.profile_hierarchy()
        assert h["root"] == "KENSEI"


# ── R2-9 ─────────────────────────────────────────────────────────────────────

class TestR29ForceMode:
    def test_non_boolean_force_mode_rejected(self, env):
        conn, home = env
        tid = _task(conn)
        rid = _review(conn, tid)
        assert he.emit_requirement_on_review(conn, tid, rid, force_mode="yes") is None
        assert he.emit_requirement_on_review(conn, tid, rid, force_mode=1) is None

    def test_hook_honours_explicit_true_with_config_off(self, env, monkeypatch):
        conn, home = env
        monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: False)
        tid = _task(conn)
        rid = _review(conn, tid)
        he.hook_request_review(conn, tid, rid, force_mode=True)
        events = [e for e in kb.list_events(conn, tid) if e.kind == he.KIND_REQUIRED]
        assert len(events) == 1  # explicit True honoured despite config off

    def test_hook_explicit_false_overrides_config_on(self, env, monkeypatch):
        conn, home = env
        monkeypatch.setattr(he, "event_mode_enabled", lambda cfg=None: True)
        tid = _task(conn)
        rid = _review(conn, tid)
        he.hook_request_review(conn, tid, rid, force_mode=False)
        events = [e for e in kb.list_events(conn, tid) if e.kind == he.KIND_REQUIRED]
        assert events == []


# ── R2-10 ────────────────────────────────────────────────────────────────────

class TestR210FindingRecurrence:
    def test_many_updates_one_finding_is_one_open(self, fake_home_r2):
        """RED: one finding updated 3 times must count as ONE open finding."""
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        for i in range(3):
            append_event(source="t", event_type="governance.finding.opened" if i == 0
                         else "governance.finding.updated",
                         event_id=f"r210a-{i}-{time.time_ns()}",
                         actor_profile="octacon", target_profile="octacon",
                         object_id="finding-1", occurred_at=now - 100 + i)
        dim = mod._quality_dimension("octacon", now - 86400)
        assert dim["evidence"]["governance_findings_open"] == 1

    def test_opened_before_window_resolved_inside(self, fake_home_r2):
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        # opened 10 days ago, resolved today
        append_event(source="t", event_type="governance.finding.opened",
                     event_id=f"r210b-open-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 10 * 86400)
        append_event(source="t", event_type="governance.finding.resolved",
                     event_id=f"r210b-res-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 100)
        dim = mod._quality_dimension("octacon", now - 7 * 86400)
        # finding's current state is resolved → not open
        assert dim["evidence"]["governance_findings_open"] == 0

    def test_resolved_then_reopened_is_recurring(self, fake_home_r2):
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        append_event(source="t", event_type="governance.finding.opened",
                     event_id=f"r210c-o1-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 300)
        append_event(source="t", event_type="governance.finding.resolved",
                     event_id=f"r210c-r1-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 200)
        append_event(source="t", event_type="governance.finding.opened",
                     event_id=f"r210c-o2-{time.time_ns()}",
                     actor_profile="octacon", target_profile="octacon",
                     object_id="finding-1", occurred_at=now - 100)
        dim = mod._quality_dimension("octacon", now - 86400)
        # one identity currently open, but resolved+reopened → recurrence
        assert dim["evidence"]["governance_findings_open"] == 1
        assert dim["evidence"]["recurring_findings"] is True
        assert dim["verdict"] == "ATTENTION"

    def test_two_distinct_open_findings(self, fake_home_r2):
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        for fid in ("finding-1", "finding-2"):
            append_event(source="t", event_type="governance.finding.opened",
                         event_id=f"r210d-{fid}-{time.time_ns()}",
                         actor_profile="octacon", target_profile="octacon",
                         object_id=fid, occurred_at=now - 100)
        dim = mod._quality_dimension("octacon", now - 86400)
        assert dim["evidence"]["governance_findings_open"] == 2
        assert sorted(dim["evidence"]["governance_findings_open_ids"]) == ["finding-1", "finding-2"]

    def test_duplicate_replay_idempotent(self, fake_home_r2):
        mod = self._load()
        from hermes_cli.profile_activity_ledger import append_event
        import time
        now = int(time.time())
        # same event_id replayed (idempotent ledger) — must not inflate counts
        for _ in range(2):
            append_event(source="t", event_type="governance.finding.opened",
                         event_id="r210e-dup", actor_profile="octacon",
                         target_profile="octacon", object_id="finding-1",
                         occurred_at=now - 100)
        dim = mod._quality_dimension("octacon", now - 86400)
        assert dim["evidence"]["governance_findings_open"] == 1

    @staticmethod
    def _load():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "drc_r210",
            "/home/kensei/worktrees/governance-evidence-spine-p34-corrections-r2-agent/scripts/denji-review-cycle.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


@pytest.fixture
def fake_home_r2(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "profiles" / "octacon").mkdir(parents=True)
    (h / "governance").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h