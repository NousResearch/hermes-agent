from __future__ import annotations

import time

from hermes_state import SessionDB
from agent.runtime_types import (
    ArtifactRecord,
    DelegationRecord,
    InterruptionRecord,
    RunEventRecord,
    RunRecord,
    RunStepRecord,
)


class RunStateStore:
    def __init__(self, session_db: SessionDB):
        self.db = session_db

    def create_run(self, record: RunRecord) -> None:
        payload = record.to_db_dict()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO runs (
                    id, session_id, parent_run_id, source, user_intent, state,
                    next_step, started_at, ended_at, final_status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"], payload["session_id"], payload["parent_run_id"], payload["source"],
                    payload["user_intent"], payload["state"], payload["next_step"], payload["started_at"],
                    payload["ended_at"], payload["final_status"], payload["metadata_json"],
                ),
            )

        self.db._execute_write(_do)

    def update_run_state(self, run_id: str, *, state: str, next_step: str | None = None) -> None:
        def _do(conn):
            conn.execute(
                "UPDATE runs SET state = ?, next_step = ? WHERE id = ?",
                (state, next_step, run_id),
            )

        self.db._execute_write(_do)

    def finish_run(self, run_id: str, *, final_status: str, state: str) -> None:
        ended_at = time.time()

        def _do(conn):
            conn.execute(
                "UPDATE runs SET state = ?, final_status = ?, ended_at = ? WHERE id = ?",
                (state, final_status, ended_at, run_id),
            )

        self.db._execute_write(_do)

    def create_step(self, record: RunStepRecord) -> None:
        payload = record.to_db_dict()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO run_steps (
                    id, run_id, step_index, step_type, status, started_at,
                    ended_at, input_summary, output_summary, tool_name, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"], payload["run_id"], payload["step_index"], payload["step_type"],
                    payload["status"], payload["started_at"], payload["ended_at"], payload["input_summary"],
                    payload["output_summary"], payload["tool_name"], payload["error"],
                ),
            )

        self.db._execute_write(_do)

    def finish_step(self, step_id: str, *, status: str, output_summary: str | None = None, error: str | None = None) -> None:
        ended_at = time.time()

        def _do(conn):
            conn.execute(
                "UPDATE run_steps SET status = ?, ended_at = ?, output_summary = COALESCE(?, output_summary), error = COALESCE(?, error) WHERE id = ?",
                (status, ended_at, output_summary, error, step_id),
            )

        self.db._execute_write(_do)

    def append_event(self, record: RunEventRecord) -> None:
        payload = record.to_db_dict()

        def _do(conn):
            conn.execute(
                "INSERT INTO run_events (id, run_id, step_id, event_type, payload_json, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    payload["id"], payload["run_id"], payload["step_id"], payload["event_type"],
                    payload["payload_json"], payload["timestamp"],
                ),
            )

        self.db._execute_write(_do)

    def create_interruption(self, record: InterruptionRecord) -> None:
        payload = record.to_db_dict()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO interruptions (
                    id, run_id, step_id, reason_type, waiting_on, snapshot_json,
                    resumable, created_at, resumed_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"], payload["run_id"], payload["step_id"], payload["reason_type"],
                    payload["waiting_on"], payload["snapshot_json"], payload["resumable"], payload["created_at"],
                    payload["resumed_at"], payload["status"],
                ),
            )

        self.db._execute_write(_do)

    def resume_interruption(self, interruption_id: str) -> None:
        resumed_at = time.time()

        def _do(conn):
            conn.execute(
                "UPDATE interruptions SET status = 'resumed', resumed_at = ? WHERE id = ?",
                (resumed_at, interruption_id),
            )

        self.db._execute_write(_do)

    def create_delegation(self, record: DelegationRecord) -> None:
        payload = record.to_db_dict()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO delegations (
                    id, parent_run_id, child_session_id, goal, context_summary,
                    allowed_toolsets_json, side_effect_policy, expected_output_type,
                    verification_status, status, created_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"], payload["parent_run_id"], payload["child_session_id"], payload["goal"],
                    payload["context_summary"], payload["allowed_toolsets_json"], payload["side_effect_policy"],
                    payload["expected_output_type"], payload["verification_status"], payload["status"],
                    payload["created_at"], payload["ended_at"],
                ),
            )

        self.db._execute_write(_do)

    def finish_delegation(self, delegation_id: str, *, status: str, verification_status: str) -> None:
        ended_at = time.time()

        def _do(conn):
            conn.execute(
                "UPDATE delegations SET status = ?, verification_status = ?, ended_at = ? WHERE id = ?",
                (status, verification_status, ended_at, delegation_id),
            )

        self.db._execute_write(_do)

    def create_artifact(self, record: ArtifactRecord) -> None:
        payload = record.to_db_dict()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO artifacts (
                    id, run_id, step_id, artifact_type, path_or_ref,
                    produced_by, purpose, is_final, delivered, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"], payload["run_id"], payload["step_id"], payload["artifact_type"],
                    payload["path_or_ref"], payload["produced_by"], payload["purpose"], payload["is_final"],
                    payload["delivered"], payload["created_at"],
                ),
            )

        self.db._execute_write(_do)
