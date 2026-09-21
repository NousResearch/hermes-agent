"""Tests for the /api/pipeline/dashboard endpoint.

Covers:
- Unauthenticated request is rejected with 401.
- Empty pipeline (no tasks in any stage) returns correct shape.
- Tasks in pipeline stages are grouped correctly.
- Council verdict is parsed from artifact files.
- Human-gate queue is populated for sign_off / final_sign_off tasks.
- Wait time is computed from pipeline_advanced event (epoch-seconds INTEGER).
- Denji panel events are surfaced.
- Tasks without a pipeline_stage are excluded.
"""

import json
import os
import time

import pytest


def _make_client(hermes_home=None):
    """Return (TestClient, session_header_name, session_token).

    Skips the test if fastapi/starlette is not installed.  Points the
    kanban DB at a fresh in-HERMES_HOME path so no real data leaks in.
    """
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client, _SESSION_HEADER_NAME, _SESSION_TOKEN


def _seed_db(db_path, tasks=None, events=None):
    """Insert rows into the kanban DB at db_path.

    Uses hermes_cli.kanban_db.connect() so that init_db runs first and
    the full schema (including all columns like claim_lock) is created.
    Then inserts the test rows directly via sqlite3 on the same file.

    ``tasks`` is a list of dicts with keys:
        id, title, assignee, pipeline_stage, pipeline_mode
    ``events`` is a list of dicts with keys:
        task_id, kind, payload (dict/str), created_at (int epoch)
    """
    from pathlib import Path
    from hermes_cli.kanban_db import connect

    # Run init_db via the canonical connect() so all columns are present.
    now_epoch = int(time.time())
    conn = connect(db_path=Path(db_path))
    for t in (tasks or []):
        conn.execute(
            "INSERT OR REPLACE INTO tasks "
            "(id, title, assignee, pipeline_stage, pipeline_mode, status, created_at) "
            "VALUES (?,?,?,?,?,'todo',?)",
            (
                t["id"],
                t.get("title", "") or "",
                t.get("assignee", ""),
                t.get("pipeline_stage"),
                t.get("pipeline_mode", "full"),
                now_epoch,
            ),
        )
    for e in (events or []):
        payload = e.get("payload", {})
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?,?,?,?)",
            (e["task_id"], e["kind"], payload, e.get("created_at", int(time.time()))),
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def pipeline_env(tmp_path, monkeypatch, _isolate_hermes_home):
    """Isolate the kanban DB and HERMES_HOME for pipeline tests."""
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Patch get_hermes_home so artifact path resolution uses tmp_path.
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    # Also patch it in web_server so the endpoint uses the same dir.
    import hermes_cli.web_server as ws
    monkeypatch.setattr(ws, "get_hermes_home", lambda: tmp_path)

    return tmp_path, db_path


class TestPipelineDashboardEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, pipeline_env):
        self.hermes_home, self.db_path = pipeline_env
        self.client, self.header_name, self.token = _make_client()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def test_unauthenticated_returns_401(self):
        client, header_name, token = _make_client()
        # Remove the auth header for this specific call.
        response = client.get(
            "/api/pipeline/dashboard",
            headers={header_name: "wrong-token"},
        )
        assert response.status_code == 401

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def test_empty_pipeline_returns_correct_shape(self):
        _seed_db(self.db_path)
        r = self.client.get("/api/pipeline/dashboard")
        assert r.status_code == 200
        body = r.json()
        assert "stages" in body
        assert "tasks_by_stage" in body
        assert "council_verdicts" in body
        assert "human_gate_queue" in body
        assert "denji_events" in body

    def test_stages_match_pipeline_stages_constant(self):
        from hermes_cli.feature_pipeline import PIPELINE_STAGES

        _seed_db(self.db_path)
        r = self.client.get("/api/pipeline/dashboard")
        assert r.json()["stages"] == PIPELINE_STAGES

    # ------------------------------------------------------------------
    # tasks_by_stage
    # ------------------------------------------------------------------

    def test_tasks_grouped_by_stage(self):
        _seed_db(
            self.db_path,
            tasks=[
                {"id": "t1", "title": "Alpha", "pipeline_stage": "research", "pipeline_mode": "full"},
                {"id": "t2", "title": "Beta", "pipeline_stage": "spec", "pipeline_mode": "express"},
                {"id": "t3", "title": "Gamma", "pipeline_stage": "research", "pipeline_mode": "full"},
            ],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        research_ids = [t["id"] for t in body["tasks_by_stage"].get("research", [])]
        assert "t1" in research_ids
        assert "t3" in research_ids
        spec_ids = [t["id"] for t in body["tasks_by_stage"].get("spec", [])]
        assert "t2" in spec_ids

    def test_tasks_without_stage_excluded(self):
        _seed_db(
            self.db_path,
            tasks=[
                {"id": "t_no_stage", "title": "Not in pipeline", "pipeline_stage": None},
                {"id": "t_with_stage", "title": "In pipeline", "pipeline_stage": "prd"},
            ],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        all_ids = [t["id"] for tasks in body["tasks_by_stage"].values() for t in tasks]
        assert "t_no_stage" not in all_ids
        assert "t_with_stage" in all_ids

    def test_task_fields_present(self):
        _seed_db(
            self.db_path,
            tasks=[{"id": "t1", "title": "My Task", "assignee": "denji", "pipeline_stage": "council"}],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        task = body["tasks_by_stage"]["council"][0]
        assert task["id"] == "t1"
        assert task["title"] == "My Task"
        assert task["assignee"] == "denji"
        assert task["pipeline_stage"] == "council"
        assert task["pipeline_mode"] == "full"

    # ------------------------------------------------------------------
    # council_verdicts
    # ------------------------------------------------------------------

    def test_council_verdict_parsed_from_artifact(self):
        artifacts_dir = self.hermes_home / "feature-artifacts" / "t1"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "council-verdict.md").write_text(
            "# Council Verdict\n\n**Verdict: APPROVED**\n\nAll good.\n"
        )
        _seed_db(
            self.db_path,
            tasks=[{"id": "t1", "pipeline_stage": "council"}],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        assert body["council_verdicts"].get("t1") == "APPROVED"

    def test_council_verdict_revise(self):
        artifacts_dir = self.hermes_home / "feature-artifacts" / "t2"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "council-verdict.md").write_text(
            "## Decision\n**Verdict: REVISE**\n\nNeeds work.\n"
        )
        _seed_db(
            self.db_path,
            tasks=[{"id": "t2", "pipeline_stage": "spec"}],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        assert body["council_verdicts"].get("t2") == "REVISE"

    def test_council_verdict_missing_artifact_excluded(self):
        _seed_db(
            self.db_path,
            tasks=[{"id": "t3", "pipeline_stage": "research"}],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        assert "t3" not in body["council_verdicts"]

    # ------------------------------------------------------------------
    # human_gate_queue
    # ------------------------------------------------------------------

    def test_human_gate_queue_contains_sign_off_tasks(self):
        _seed_db(
            self.db_path,
            tasks=[
                {"id": "h1", "title": "Sign Off Me", "pipeline_stage": "sign_off"},
                {"id": "h2", "title": "Final", "pipeline_stage": "final_sign_off"},
                {"id": "h3", "title": "Not a gate", "pipeline_stage": "research"},
            ],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        gate_ids = {t["id"] for t in body["human_gate_queue"]}
        assert "h1" in gate_ids
        assert "h2" in gate_ids
        assert "h3" not in gate_ids

    def test_human_gate_wait_time_computed(self):
        # Insert a pipeline_advanced event 2 hours ago (epoch INTEGER).
        entered_at = int(time.time()) - 7200
        _seed_db(
            self.db_path,
            tasks=[{"id": "h1", "pipeline_stage": "sign_off"}],
            events=[{
                "task_id": "h1",
                "kind": "pipeline_advanced",
                "payload": {"to_stage": "sign_off"},
                "created_at": entered_at,
            }],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        task = next(t for t in body["human_gate_queue"] if t["id"] == "h1")
        # Should be approximately 2 hours; allow 0.1h tolerance for test speed.
        assert task["waiting_hours"] is not None
        assert abs(task["waiting_hours"] - 2.0) < 0.1

    def test_human_gate_no_event_waiting_hours_is_none(self):
        _seed_db(
            self.db_path,
            tasks=[{"id": "h1", "pipeline_stage": "sign_off"}],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        task = next(t for t in body["human_gate_queue"] if t["id"] == "h1")
        assert task["waiting_hours"] is None

    # ------------------------------------------------------------------
    # denji_events
    # ------------------------------------------------------------------

    def test_denji_events_returned(self):
        now = int(time.time())
        _seed_db(
            self.db_path,
            tasks=[{"id": "d1", "pipeline_stage": "execute"}],
            events=[
                {"task_id": "d1", "kind": "denji_review_signal", "payload": {"signal": "ok"}, "created_at": now - 10},
                {"task_id": "d1", "kind": "pipeline_spawn", "payload": {"stage": "spec"}, "created_at": now - 20},
                {"task_id": "d1", "kind": "bypass_record", "payload": {"stage": "prd"}, "created_at": now - 30},
                # Non-Denji event — must NOT appear.
                {"task_id": "d1", "kind": "comment", "payload": {}, "created_at": now - 40},
            ],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        kinds = {e["kind"] for e in body["denji_events"]}
        assert "denji_review_signal" in kinds
        assert "pipeline_spawn" in kinds
        assert "bypass_record" in kinds
        assert "comment" not in kinds

    def test_denji_events_epoch_created_at(self):
        now = int(time.time())
        _seed_db(
            self.db_path,
            tasks=[{"id": "d1", "pipeline_stage": "execute"}],
            events=[
                {"task_id": "d1", "kind": "pipeline_spawn", "payload": {}, "created_at": now},
            ],
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        ev = next(e for e in body["denji_events"] if e["kind"] == "pipeline_spawn")
        # created_at must survive as a numeric value (not a string).
        assert isinstance(ev["created_at"], (int, float))
        assert abs(ev["created_at"] - now) < 5

    def test_denji_events_capped_at_20(self):
        now = int(time.time())
        events = [
            {"task_id": "d1", "kind": "pipeline_spawn", "payload": {}, "created_at": now - i}
            for i in range(30)
        ]
        _seed_db(
            self.db_path,
            tasks=[{"id": "d1", "pipeline_stage": "execute"}],
            events=events,
        )
        body = self.client.get("/api/pipeline/dashboard").json()
        assert len(body["denji_events"]) <= 20
