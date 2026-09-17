"""Client projections must not expose model-only compaction scaffolding."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import tracemalloc

from aiohttp.test_utils import TestClient, TestServer
import pytest

from agent.compaction_display import project_compaction_message_for_display
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _is_compressed_summary_message,
)
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION


STANDALONE_SUMMARY = (
    f"{SUMMARY_PREFIX}\n\n"
    f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
    f"{_SUMMARY_END_MARKER}"
)
MERGED_CARRIER = (
    f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"
    "Refactor complete.\n\n"
    f"{_MERGED_SUMMARY_DELIMITER}\n\n"
    f"{STANDALONE_SUMMARY}"
)
REAL_USER = "test the browser controller again"
TODO_SNAPSHOT_METADATA = {
    "todo_snapshot": {
        "todos": [
            {"id": "plan", "content": "parent", "status": "completed"},
            {
                "id": "child",
                "content": "active child",
                "status": "in_progress",
                "parent": "plan",
            },
            {"id": "next", "content": "next", "status": "pending"},
            {"id": "skip", "content": "skip", "status": "cancelled"},
        ]
    }
}
TODO_ROW_277757_CONTENT = "\n".join(
    [
        "[Your active task list was preserved across context compression]",
        "- [ ] s12. Session 20260829_223026_c651fc: fold todo reconciliation into the single renderer-foundation PR (pending)",
        "- [ ] s09. Session 20260829_162810_cf60c3: finish terminal-path todo cleanup and renderer-foundation PR (pending)",
        "- [ ] s11. Session 20260829_215641_a9fba1: preserve RCA and route the optimistic user-bubble defect without an RCA PR (pending)",
        "- [ ] goal. Goal-control correction pushed to PR #98331 at 48450910e0eb; current CI has Python and Windows failures requiring diagnosis (pending)",
        "- [ ] integrate. Integrate all six currently CI-green Hermes PR heads into a current-upstream fork-integration line and verify exact remote state; no release/install — remote publication complete, canonical activation pending external restart authority (pending)",
        "  - [ ] int-canonical. Reconcile the clean canonical fork-integration checkout from e24e0372 to published 1540d425; requires externally controlled Hermes stop/reset/restart because live-source guard blocks in-process reset (pending)",
        "- [ ] verify. Read back every pushed branch and PR, verify CI/state, and report unresolved human or external blockers (pending)",
        "- [>] session-audit. Audit top-level campaign-related sessions missed by parent-session provenance (in_progress)",
        "  - [ ] session-audit-inventory. Inventory unarchived top-level sessions in the campaign time window (pending)",
        "  - [ ] session-audit-classify. Classify candidates using transcript, branch, worktree, and outcome evidence (pending)",
        "  - [ ] session-audit-report. Verify candidate scope and report exact cleanup recommendation without mutating sessions (pending)",
        "",
        "[Skills pruned during compression — reload before acting on these tasks]",
        "The task list above crossed the compression boundary verbatim, but the skill instructions that governed it were pruned. Before executing any preserved task that depends on these skills, reload them first: skill_view(name='software-development/systematic-debugging'); skill_view(name='software-development/desktop-ui-engineering'); skill_view(name='autonomous-ai-agents/coding-agent-handoff-recovery'); skill_view(name='diagnosing-bugs'); skill_view(name='devops/fleet-helmsman-integration'); skill_view(name='autonomous-ai-agents/hermes-agent'); skill_view(name='devops/hermes-windows-gateway-operations'); skill_view(name='github/hermes-fork-integration'); skill_view(name='github/github-pr-workflow'); skill_view(name='github/git-repository-reconciliation'); skill_view(name='github/upstream-feature-porting'); skill_view(name='devops/kanban-operations'); skill_view(name='critical-study'); skill_view(name='code-review'); skill_view(name='structured-adversarial-review'); skill_view(name='hermes-oneshot'); skill_view(name='devops/wsl-interop'); skill_view(name='autonomous-ai-agents/claude-code'); skill_view(name='design-taste-frontend'); skill_view(name='productivity/session-librarian'). After reloading, re-check that each pending task is still justified — findings recorded before the boundary may have invalidated it.",
    ]
)


def _row(role: str, content, **extra) -> dict:
    row = {"id": 1, "session_id": "s1", "role": role, "content": content}
    row.update(extra)
    return row


def _todo_call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "todo", "arguments": json.dumps({"todos": []})},
    }


def _todo_result(label: str) -> str:
    return json.dumps(
        {
            "todos": [
                {
                    "id": label,
                    "content": f"{label} state",
                    "status": "in_progress",
                }
            ]
        }
    )


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


def _messages_app(adapter: APIServerAdapter):
    from aiohttp import web

    app = web.Application()
    app.router.add_get(
        "/api/sessions/{session_id}/messages",
        adapter._handle_session_messages,
    )
    return app


class TestMessageProjection:
    @pytest.mark.parametrize(
        "display_metadata",
        [TODO_SNAPSHOT_METADATA, json.dumps(TODO_SNAPSHOT_METADATA)],
        ids=["decoded", "raw-json-text"],
    )
    def test_todo_snapshot_metadata_survives_message_projection(
        self, display_metadata
    ):
        projected = APIServerAdapter._message_response(
            _row(
                "user",
                "opaque todo carrier",
                display_kind="hidden",
                display_metadata=display_metadata,
            )
        )

        assert projected["display_metadata"] == display_metadata

    def test_standalone_summary_is_hidden_without_scaffolding(self):
        projected = APIServerAdapter._message_response(
            _row(
                "user",
                STANDALONE_SUMMARY,
                tool_calls=[{"id": "stale"}],
                reasoning="internal compression reasoning",
                reasoning_content="internal compression reasoning",
                reasoning_details=[{"type": "reasoning.summary", "summary": "internal"}],
                codex_reasoning_items=[{"type": "reasoning", "id": "internal"}],
                codex_message_items=[{"type": "message", "id": "internal"}],
            )
        )

        assert projected["content"] == ""
        assert projected["display_kind"] == "hidden"
        assert "tool_calls" not in projected
        assert "finish_reason" not in projected
        assert "reasoning" not in projected
        assert "reasoning_content" not in projected
        assert "reasoning_details" not in projected
        assert "codex_reasoning_items" not in projected
        assert "codex_message_items" not in projected

    def test_merged_carrier_preserves_only_real_prior_content(self):
        projected = APIServerAdapter._message_response(
            _row(
                "assistant",
                MERGED_CARRIER,
                tool_calls=[{"id": "prior-call"}],
                finish_reason="tool_calls",
            )
        )

        assert projected["content"] == "Refactor complete."
        assert "tool_calls" not in projected
        assert "finish_reason" not in projected
        assert "PRIOR CONTEXT" not in projected["content"]
        assert "CONTEXT COMPACTION" not in projected["content"]

    def test_merged_content_array_preserves_blocks_before_summary(self):
        projected = APIServerAdapter._message_response(
            _row(
                "user",
                [
                    {
                        "type": "text",
                        "text": f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{REAL_USER}",
                    },
                    {
                        "type": "text",
                        "text": f"{_MERGED_SUMMARY_DELIMITER}\n{STANDALONE_SUMMARY}",
                    },
                ],
            )
        )

        assert projected["content"] == [{"type": "text", "text": REAL_USER}]

    def test_real_message_that_mentions_marker_text_is_untouched(self):
        message = _row(
            "user",
            "please explain the string [CONTEXT COMPACTION] in this bug report",
            tool_calls=[{"id": "real"}],
            reasoning="real provider payload",
        )
        projected = project_compaction_message_for_display(message)

        assert projected == message
        assert projected is not message

    def test_unrelated_hidden_message_is_not_reclassified_as_compaction(self):
        message = _row("assistant", "ordinary hidden control row", display_kind="hidden")

        assert _is_compressed_summary_message(message) is False


class TestSummaryRecognizer:
    @pytest.mark.parametrize(
        "message",
        [
            _row("user", STANDALONE_SUMMARY),
            _row("assistant", MERGED_CARRIER),
            _row("assistant", "metadata-only", **{COMPRESSED_SUMMARY_METADATA_KEY: True}),
        ],
    )
    def test_recognizes_all_compaction_carrier_shapes(self, message):
        assert _is_compressed_summary_message(message) is True

    def test_ignores_real_message(self):
        assert _is_compressed_summary_message(_row("user", REAL_USER)) is False


class TestTurnTranscriptProjection:
    def test_run_completed_strips_scaffolding_but_keeps_real_carrier_content(self):
        result = {
            "messages": [
                {"role": "user", "content": REAL_USER},
                {"role": "assistant", "content": "checking the controller"},
                _row("user", STANDALONE_SUMMARY),
                _row("assistant", MERGED_CARRIER),
                {"role": "assistant", "content": "the controller is ready"},
            ],
            "final_response": "the controller is ready",
        }

        turn = APIServerAdapter._turn_transcript_messages(
            [{"role": "user", "content": REAL_USER}],
            REAL_USER,
            result,
        )

        assert [message.get("content") for message in turn] == [
            "checking the controller",
            "Refactor complete.",
            "the controller is ready",
        ]


class TestMessagesEndpointProjection:
    @staticmethod
    def _todo_state_read_metrics(session_db, session_id):
        statements: list[str] = []
        vm_steps = [0]
        decoder_calls = [0]
        original_decode_content = session_db._decode_content
        original_decode_metadata = session_db._decode_display_metadata
        original_wal_active = session_db._wal_active

        def count_step():
            vm_steps[0] += 1
            return 0

        def decode_content(value):
            decoder_calls[0] += 1
            return original_decode_content(value)

        def decode_metadata(value):
            decoder_calls[0] += 1
            return original_decode_metadata(value)

        # Pin the instrumented read to the writer connection; WAL readers use
        # the same query plan, but sqlite only exposes progress/trace hooks per
        # connection.
        session_db._wal_active = False
        session_db._decode_content = decode_content
        session_db._decode_display_metadata = decode_metadata
        session_db._conn.set_trace_callback(statements.append)
        session_db._conn.set_progress_handler(count_step, 1)
        tracemalloc.start()
        started = time.perf_counter()
        try:
            messages = session_db.get_todo_state_messages(session_id)
            elapsed = time.perf_counter() - started
            _current, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
            session_db._conn.set_progress_handler(None, 0)
            session_db._conn.set_trace_callback(None)
            session_db._decode_content = original_decode_content
            session_db._decode_display_metadata = original_decode_metadata
            session_db._wal_active = original_wal_active

        selects = [statement for statement in statements if statement.lstrip().upper().startswith("SELECT")]
        return {
            "decoder_calls": decoder_calls[0],
            "elapsed": elapsed,
            "messages": messages,
            "peak_bytes": peak_bytes,
            "selects": selects,
            "vm_steps": vm_steps[0],
        }

    @pytest.mark.parametrize("depth", [8, 128, 1024])
    def test_todo_state_materialized_lookup_has_fixed_work_at_history_depth(
        self,
        tmp_path,
        depth,
    ):
        db = SessionDB(tmp_path / f"todo-depth-{depth}.db")
        session_id = db.create_session(f"todo-depth-{depth}", "api_server")
        rejected = []
        for index in range(depth):
            rejected.extend(
                [
                    _row(
                        "assistant",
                        "",
                        tool_calls=[
                            {
                                "id": f"noise-{index}",
                                "type": "function",
                                "function": {"name": "not-todo", "arguments": "{}"},
                            }
                        ],
                    ),
                    _row(
                        "tool",
                        json.dumps({"todos": "not-a-list", "noise": index}),
                        tool_call_id=f"noise-{index}",
                    ),
                ]
            )
        if depth == 1024:
            rejected.append(
                _row(
                    "tool",
                    json.dumps({"todos": "x" * 2_000_000}),
                    tool_call_id="oversized-rejected",
                )
            )

        try:
            db.replace_messages(
                session_id,
                [_row("user", TODO_ROW_277757_CONTENT), *rejected],
            )
            metrics = self._todo_state_read_metrics(db, session_id)

            assert [message["content"] for message in metrics["messages"]] == [
                TODO_ROW_277757_CONTENT
            ]
            assert len(metrics["selects"]) == 1, metrics["selects"]
            assert metrics["decoder_calls"] <= 1
            assert metrics["vm_steps"] <= 200
            assert metrics["peak_bytes"] <= 512_000
            assert len(json.dumps(metrics["messages"]).encode("utf-8")) < 7_000
            assert metrics["elapsed"] < 0.1

            lookup_sql = getattr(db, "_TODO_STATE_LOOKUP_SQL", "")
            assert "LIMIT 1" in lookup_sql.upper()
            plan = " ".join(
                row["detail"]
                for row in db._conn.execute(
                    "EXPLAIN QUERY PLAN " + lookup_sql,
                    (session_id,),
                ).fetchall()
            )
            assert "idx_todo_authorities_latest" in plan, plan
            assert "SCAN messages" not in plan.upper(), plan
        finally:
            db.close()

    def test_todo_state_previous_schema_adds_columns_before_sparse_insert_triggers(self, tmp_path):
        db_path = tmp_path / "todo-previous-schema.db"
        seeded = SessionDB(db_path)
        try:
            session_id = seeded.create_session("todo-previous-schema", "api_server")
            seeded.append_message(session_id, "user", "ordinary legacy row")
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("DROP TRIGGER IF EXISTS trg_todo_authority_message_insert")
            raw.execute("DROP TRIGGER IF EXISTS trg_todo_pair_boundary_message_insert")
            raw.execute("DROP TRIGGER IF EXISTS trg_todo_authority_message_activity")
            raw.execute("DROP TRIGGER IF EXISTS trg_todo_authority_message_delete")
            raw.execute(
                "DROP TRIGGER IF EXISTS trg_todo_migration_reopen_on_authority_activity"
            )
            raw.execute(
                "DROP TRIGGER IF EXISTS trg_todo_migration_reopen_on_authority_delete"
            )
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")
            raw.execute("ALTER TABLE messages DROP COLUMN todo_pair_boundary_json")
            raw.execute("ALTER TABLE messages DROP COLUMN todo_authority_json")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))

        reopened = SessionDB(db_path)
        try:
            columns = {
                row["name"] for row in reopened._conn.execute("PRAGMA table_info(messages)")
            }
            assert {"todo_authority_json", "todo_pair_boundary_json"} <= columns
            triggers = {
                row["name"]
                for row in reopened._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                    "AND name LIKE 'trg_todo_%_message_insert'"
                )
            }
            assert triggers == {
                "trg_todo_authority_message_insert",
                "trg_todo_pair_boundary_message_insert",
            }
        finally:
            reopened.close()

    def test_current_schema_reconciles_new_migration_progress_columns(self, tmp_path):
        db_path = tmp_path / "todo-existing-current-progress-schema.db"
        seeded = SessionDB(db_path)
        try:
            session_id = seeded.create_session(
                "todo-existing-current-progress-schema", "api_server"
            )
            message_id = seeded.append_message(
                session_id, "assistant", "ordinary existing-current row"
            )
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            for trigger in (
                "trg_todo_migration_reopen_on_authority_activity",
                "trg_todo_migration_reopen_on_authority_delete",
            ):
                raw.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            raw.execute("DROP TABLE todo_authority_migrations")
            raw.execute(
                "CREATE TABLE todo_authority_migrations ("
                "session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE, "
                "before_message_id INTEGER NOT NULL, "
                "pending_results_json TEXT NOT NULL DEFAULT '[]', "
                "complete INTEGER NOT NULL DEFAULT 0)"
            )
            raw.execute(
                "INSERT INTO todo_authority_migrations "
                "(session_id, before_message_id, pending_results_json, complete) "
                "VALUES (?, ?, '[]', 1)",
                (session_id, message_id + 1),
            )
            assert raw.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION

        reopened = SessionDB(db_path)
        try:
            columns = {
                row["name"]
                for row in reopened._conn.execute(
                    "PRAGMA table_info(todo_authority_migrations)"
                )
            }
            assert {
                "deferred_pending_results_json",
                "phase",
                "authority_checked_message_id",
            } <= columns
            progress = reopened._conn.execute(
                "SELECT deferred_pending_results_json, phase, "
                "authority_checked_message_id, complete "
                "FROM todo_authority_migrations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert tuple(progress) == ("[]", "scan", None, 1)
            assert reopened._conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()[0] == SCHEMA_VERSION
        finally:
            reopened.close()

    def test_todo_state_upgrade_defers_legacy_work_and_eventually_recovers_old_carrier(
        self,
        tmp_path,
    ):
        db_path = tmp_path / "todo-bounded-lazy-upgrade.db"
        db = SessionDB(db_path)
        session_id = db.create_session("todo-bounded-lazy-upgrade", "api_server")
        try:
            db.replace_messages(
                session_id,
                [
                    _row("user", TODO_ROW_277757_CONTENT),
                    *[
                        _row("assistant", f"ordinary newer row {index}")
                        for index in range(2_048)
                    ],
                ],
            )
        finally:
            db.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP INDEX IF EXISTS idx_messages_todo_authority_latest")
            raw.execute("DROP INDEX IF EXISTS idx_messages_todo_pair_boundary")
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        reopened = SessionDB(db_path)
        try:
            # Opening performs schema reconciliation only. It must not traverse,
            # decode, or materialize legacy transcript history.
            assert reopened._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE todo_authority_json IS NOT NULL"
            ).fetchone()[0] == 0
            todo_message_indexes = reopened._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'messages' AND name LIKE 'idx_messages_todo_%'"
            ).fetchall()
            assert todo_message_indexes == []

            projected = []
            for _attempt in range(8):
                try:
                    projected = reopened.get_todo_state_messages(session_id)
                except RuntimeError as error:
                    assert error.__class__.__name__ == "TodoStateMigrationPendingError"
                    continue
                if projected:
                    break

            assert [message["content"] for message in projected] == [
                TODO_ROW_277757_CONTENT
            ]
        finally:
            reopened.close()

    def test_todo_state_migration_enforces_total_budgets_across_restart(self, tmp_path):
        db_path = tmp_path / "todo-bounded-resource-restart.db"
        db = SessionDB(db_path)
        session_id = db.create_session("todo-bounded-resource-restart", "api_server")
        try:
            db.replace_messages(
                session_id,
                [
                    _row("user", TODO_ROW_277757_CONTENT),
                    _row("user", "x" * 4_000_000, display_kind="hidden"),
                    _row("tool", "x" * 4_000_000, tool_call_id="oversized"),
                    *[
                        _row("assistant", f"ordinary newer row {index}")
                        for index in range(2_048)
                    ],
                ],
            )
        finally:
            db.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        previous_before = None
        projected = None
        observed = []
        for _attempt in range(8):
            reopened = SessionDB(db_path)
            tracemalloc.start()
            started = time.perf_counter()
            try:
                projected = reopened.get_todo_state_messages(session_id)
                elapsed = time.perf_counter() - started
                _current, peak_bytes = tracemalloc.get_traced_memory()
                stats = reopened._todo_migration_stats_for_tests()
                observed.append({**stats, "elapsed": elapsed, "peak_bytes": peak_bytes})
                progress = reopened._conn.execute(
                    "SELECT before_message_id, complete FROM todo_authority_migrations "
                    "WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                assert progress is not None
                before_message_id = int(progress["before_message_id"])
                if previous_before is not None:
                    assert before_message_id < previous_before
                previous_before = before_message_id
            finally:
                tracemalloc.stop()
                reopened.close()
            if projected:
                break

        assert [message["content"] for message in projected] == [
            TODO_ROW_277757_CONTENT
        ]
        assert observed
        assert max(stats["max_source_blob_bytes"] for stats in observed) >= 4_000_000
        for stats in observed:
            assert stats["rows"] <= SessionDB._TODO_MIGRATION_ROWS_PER_SLICE * SessionDB._TODO_MIGRATION_MAX_SLICES_PER_READ
            assert stats["selects"] <= SessionDB._TODO_MIGRATION_MAX_SLICES_PER_READ + 4
            assert stats["decoded_bytes"] <= SessionDB._TODO_MIGRATION_MAX_DECODED_BYTES_PER_READ
            assert stats["blob_reads"] <= stats["rows"] * 7 + 3
            assert stats["max_blob_read_bytes"] <= SessionDB._TODO_MIGRATION_PAYLOAD_BYTES
            assert stats["pending_results"] <= SessionDB._TODO_MIGRATION_MAX_PENDING_RESULTS
            assert stats["peak_bytes"] <= 3_000_000
            assert stats["elapsed"] < 0.25

    def test_todo_blob_resource_contract_covers_every_bounded_source_at_near_limit(
        self, session_db
    ):
        contract = session_db._todo_blob_resource_contract_for_tests()

        assert contract["field_read_caps_bytes"] == {
            "messages.content": 65_536,
            "messages.display_kind": 4_096,
            "messages.display_metadata": 65_536,
            "messages.role": 16,
            "messages.todo_authority_json": 1_000_000,
            "messages.tool_call_id": 4_096,
            "messages.tool_calls": 65_536,
            "messages.tool_name": 4_096,
            "state_meta.value": 32,
            "todo_authorities.authority_json": 1_000_000,
            "todo_authority_migrations.deferred_pending_results_json": 1_100_000,
            "todo_authority_migrations.pending_results_json": 1_100_000,
            "todo_pair_boundaries.boundary_json": 1_000_000,
        }
        assert contract["single_result_source_bytes"] == 65_536
        assert contract["materialized_authority_bytes"] == 1_000_000
        assert contract["pending_result_count"] == 16
        assert contract["pending_results_json_bytes"] == 1_100_000
        assert contract["attempt_decoded_bytes"] == 2_000_000
        assert contract["cooperative_deadline_seconds"] == 0.05
        assert contract["deadline_kind"] == "cooperative"
        assert contract["hard_deadline_seconds"] is None
        assert contract["deadline_checkpoints"] == [
            "before_each_slice_after_first",
            "before_each_row_after_first_progress",
        ]
        assert contract["non_preemptible_operations"] == [
            "sqlite_statement",
            "bounded_blob_read",
            "utf8_decode",
            "json_decode",
            "json_encode",
            "sqlite_write",
        ]

        session_id = session_db.create_session("todo-near-limit-authority", "api_server")
        message_id = session_db.append_message(
            session_id,
            "user",
            TODO_ROW_277757_CONTENT,
        )
        base_message = {
            "id": message_id,
            "session_id": session_id,
            "role": "user",
            "content": TODO_ROW_277757_CONTENT,
            "tool_call_id": None,
            "tool_name": None,
            "display_kind": None,
            "display_metadata": None,
            "timestamp": 1.0,
            "padding": "",
        }
        base_payload = json.dumps(
            [base_message], ensure_ascii=False, separators=(",", ":")
        )
        padding_bytes = (
            contract["materialized_authority_bytes"]
            - len(base_payload.encode("utf-8"))
        )
        assert padding_bytes > 0
        base_message["padding"] = "x" * padding_bytes
        authority_json = json.dumps(
            [base_message], ensure_ascii=False, separators=(",", ":")
        )
        assert len(authority_json.encode("utf-8")) == 1_000_000

        def _replace_authority(conn):
            conn.execute(
                "UPDATE messages SET todo_authority_json = ? WHERE id = ?",
                (authority_json, message_id),
            )
            conn.execute(
                "UPDATE todo_authorities SET authority_json = ? WHERE message_id = ?",
                (authority_json, message_id),
            )

        session_db._execute_write(_replace_authority)
        tracemalloc.start()
        try:
            projected = session_db.get_todo_state_messages(session_id)
            _current, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert [message["content"] for message in projected] == [
            TODO_ROW_277757_CONTENT
        ]
        assert peak_bytes <= contract["near_limit_tracemalloc_ceiling_bytes"]

        progress_session = session_db.create_session(
            "todo-near-limit-progress", "api_server"
        )
        progress_payload = "[\"" + "x" * (1_100_000 - 4) + "\"]"
        assert len(progress_payload.encode("utf-8")) == 1_100_000
        session_db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO todo_authority_migrations "
                "(session_id, before_message_id, pending_results_json, complete) "
                "VALUES (?, 1, ?, 0)",
                (progress_session, progress_payload),
            )
        )
        row = session_db._conn.execute(
            "SELECT rowid FROM todo_authority_migrations WHERE session_id = ?",
            (progress_session,),
        ).fetchone()
        status, decoded, source_bytes = session_db._read_bounded_blob_text(
            session_db._conn,
            table="todo_authority_migrations",
            column="pending_results_json",
            row_id=int(row["rowid"]),
            max_bytes=contract["field_read_caps_bytes"][
                "todo_authority_migrations.pending_results_json"
            ],
        )
        assert status == "ok"
        assert decoded == progress_payload
        assert source_bytes == 1_100_000

    def test_rewind_reopens_completed_legacy_migration_to_older_active_authority(
        self, tmp_path
    ):
        db_path = tmp_path / "todo-legacy-rewind-reopen.db"
        older_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "older. Session", 1
        )
        newest_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "newest. Session", 1
        )
        seeded = SessionDB(db_path)
        session_id = seeded.create_session("todo-legacy-rewind-reopen", "api_server")
        try:
            seeded.replace_messages(
                session_id,
                [
                    _row("user", older_content),
                    _row("assistant", "ordinary middle row"),
                    _row("user", newest_content),
                ],
            )
            rows = seeded._conn.execute(
                "SELECT id, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            older_id = int(rows[0]["id"])
            newest_id = int(rows[-1]["id"])
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE messages SET todo_pair_boundary_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        migrated = SessionDB(db_path)
        try:
            assert [
                message["content"]
                for message in migrated.get_todo_state_messages(session_id)
            ] == [newest_content]
            progress = migrated._conn.execute(
                "SELECT before_message_id, complete FROM todo_authority_migrations "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert progress is not None
            assert int(progress["before_message_id"]) == newest_id
            assert int(progress["complete"]) == 1

            result = migrated.rewind_to_message(session_id, newest_id)
            assert result["rewound_count"] == 1
            assert result["new_head_id"] != newest_id
        finally:
            migrated.close()

        reopened = SessionDB(db_path)
        try:
            projected = reopened.get_todo_state_messages(session_id)
            assert [message["content"] for message in projected] == [older_content]
            all_rows = reopened.get_messages(
                session_id,
                include_inactive=True,
                include_compacted=True,
            )
            assert [int(row["id"]) for row in all_rows] == [
                older_id,
                int(rows[1]["id"]),
                newest_id,
            ]
            active_rows = reopened.get_messages(session_id)
            assert [int(row["id"]) for row in active_rows] == [
                older_id,
                int(rows[1]["id"]),
            ]
        finally:
            reopened.close()

    def test_active_only_rewrite_after_compaction_deletes_newest_and_recovers_older_authority(
        self, tmp_path
    ):
        db_path = tmp_path / "todo-legacy-compaction-delete.db"
        older_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "older-compacted. Session", 1
        )
        newest_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "newest-deleted. Session", 1
        )
        seeded = SessionDB(db_path)
        session_id = seeded.create_session("todo-legacy-compaction-delete", "api_server")
        try:
            seeded.replace_messages(session_id, [_row("user", older_content)])
            seeded.archive_and_compact(
                session_id,
                [_row("assistant", "ordinary compacted live row")],
            )
            seeded.append_message(session_id, "user", newest_content)
            rows = seeded._conn.execute(
                "SELECT id, content, active, compacted FROM messages "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            older_id = int(rows[0]["id"])
            newest_id = int(rows[-1]["id"])
            assert int(rows[0]["active"]) == 0
            assert int(rows[0]["compacted"]) == 1
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE messages SET todo_pair_boundary_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        migrated = SessionDB(db_path)
        try:
            assert [
                message["content"]
                for message in migrated.get_todo_state_messages(session_id)
            ] == [newest_content]
            migrated.replace_messages(
                session_id,
                [_row("assistant", "ordinary active-only replacement")],
                active_only=True,
            )
            projected = migrated.get_todo_state_messages(session_id)
            assert [message["content"] for message in projected] == [older_content]
            remaining = migrated._conn.execute(
                "SELECT id, active, compacted FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            assert [int(row["id"]) for row in remaining[:-1]] == [older_id]
            assert all(int(row["id"]) != newest_id for row in remaining)
        finally:
            migrated.close()

        restarted = SessionDB(db_path)
        try:
            assert [
                message["content"]
                for message in restarted.get_todo_state_messages(session_id)
            ] == [older_content]
        finally:
            restarted.close()

    def test_full_destructive_rewrite_removes_all_legacy_authority_without_resurrection(
        self, tmp_path
    ):
        db_path = tmp_path / "todo-legacy-full-rewrite.db"
        older_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "older-full-rewrite. Session", 1
        )
        newest_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "newest-full-rewrite. Session", 1
        )
        seeded = SessionDB(db_path)
        session_id = seeded.create_session("todo-legacy-full-rewrite", "api_server")
        try:
            seeded.replace_messages(
                session_id,
                [
                    _row("user", older_content),
                    _row("assistant", "ordinary middle row"),
                    _row("user", newest_content),
                ],
            )
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE messages SET todo_pair_boundary_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        migrated = SessionDB(db_path)
        try:
            assert [
                message["content"]
                for message in migrated.get_todo_state_messages(session_id)
            ] == [newest_content]
            migrated.replace_messages(
                session_id,
                [_row("assistant", "ordinary destructive replacement")],
            )
            assert migrated.get_todo_state_messages(session_id) == []
            assert migrated._conn.execute(
                "SELECT COUNT(*) FROM todo_authorities WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0] == 0
        finally:
            migrated.close()

        restarted = SessionDB(db_path)
        try:
            assert restarted.get_todo_state_messages(session_id) == []
            progress = restarted._conn.execute(
                "SELECT complete FROM todo_authority_migrations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert progress is not None and int(progress["complete"]) == 1
        finally:
            restarted.close()

    def test_session_retention_cascades_migrated_and_progress_authority_state(
        self, tmp_path
    ):
        db_path = tmp_path / "todo-legacy-retention.db"
        older_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "older-retention. Session", 1
        )
        newest_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "newest-retention. Session", 1
        )
        seeded = SessionDB(db_path)
        session_id = seeded.create_session("todo-legacy-retention", "api_server")
        try:
            seeded.replace_messages(
                session_id,
                [_row("user", older_content), _row("user", newest_content)],
            )
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE messages SET todo_pair_boundary_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        migrated = SessionDB(db_path)
        try:
            assert [
                message["content"]
                for message in migrated.get_todo_state_messages(session_id)
            ] == [newest_content]
            migrated.end_session(session_id, "completed")
            migrated._execute_write(
                lambda conn: (
                    conn.execute(
                        "UPDATE messages SET timestamp = 0 WHERE session_id = ?",
                        (session_id,),
                    ),
                    conn.execute(
                        "UPDATE sessions SET started_at = 0 WHERE id = ?",
                        (session_id,),
                    ),
                )
            )
            assert migrated.prune_sessions(older_than_days=1, source="api_server") == 1
            for table in (
                "sessions",
                "messages",
                "todo_authorities",
                "todo_authority_migrations",
                "todo_pair_boundaries",
            ):
                assert migrated._conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE "
                    + ("id = ?" if table == "sessions" else "session_id = ?"),
                    (session_id,),
                ).fetchone()[0] == 0
        finally:
            migrated.close()

        restarted = SessionDB(db_path)
        try:
            assert restarted.get_session(session_id) is None
            assert restarted.get_todo_state_messages(session_id) == []
        finally:
            restarted.close()

    def test_near_limit_pending_state_makes_monotonic_progress_before_authority_read(
        self, tmp_path
    ):
        db_path = tmp_path / "todo-near-limit-progress-liveness.db"
        seeded = SessionDB(db_path)
        session_id = seeded.create_session(
            "todo-near-limit-progress-liveness", "api_server"
        )
        older_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "older-progress. Session", 1
        )
        authority_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "near-limit-authority. Session", 1
        )
        try:
            seeded.replace_messages(
                session_id,
                [
                    _row("user", older_content),
                    _row("assistant", "near-limit authority placeholder"),
                ],
            )
            rows = seeded._conn.execute(
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            authority_id = int(rows[-1]["id"])

            authority_message = {
                "id": authority_id,
                "session_id": session_id,
                "role": "user",
                "content": authority_content,
                "tool_call_id": None,
                "tool_name": None,
                "display_kind": None,
                "display_metadata": None,
                "timestamp": 2.0,
                "padding": "",
            }
            authority_json = json.dumps(
                [authority_message], ensure_ascii=False, separators=(",", ":")
            )
            authority_message["padding"] = "a" * (
                999_000 - len(authority_json.encode("utf-8"))
            )
            authority_json = json.dumps(
                [authority_message], ensure_ascii=False, separators=(",", ":")
            )
            assert len(authority_json.encode("utf-8")) == 999_000

            todo_result = json.dumps(
                {
                    "todos": [
                        {
                            "id": "pending",
                            "content": "pending near-limit result",
                            "status": "pending",
                        }
                    ]
                },
                separators=(",", ":"),
            )
            pending_results = [
                {
                    "id": authority_id + index + 1,
                    "session_id": session_id,
                    "role": "tool",
                    "content": todo_result,
                    "tool_call_id": f"pending-{index}",
                    "tool_name": "todo",
                    "display_kind": None,
                    "display_metadata": {"padding": ""},
                    "timestamp": 3.0 + index,
                }
                for index in range(16)
            ]
            pending_json = json.dumps(
                pending_results, ensure_ascii=False, separators=(",", ":")
            )
            pending_results[0]["display_metadata"]["padding"] = "p" * (
                1_099_000 - len(pending_json.encode("utf-8"))
            )
            pending_json = json.dumps(
                pending_results, ensure_ascii=False, separators=(",", ":")
            )
            assert len(pending_json.encode("utf-8")) == 1_099_000

            def _seed_stalled_progress(conn):
                conn.execute(
                    "DELETE FROM todo_authorities WHERE session_id = ?",
                    (session_id,),
                )
                conn.execute(
                    "UPDATE messages SET todo_authority_json = ? WHERE id = ?",
                    (authority_json, authority_id),
                )
                conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES "
                    "('todo_authority_legacy_high_water', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(authority_id),),
                )
                conn.execute(
                    "INSERT INTO todo_authority_migrations "
                    "(session_id, before_message_id, pending_results_json, complete) "
                    "VALUES (?, ?, ?, 0) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "before_message_id = excluded.before_message_id, "
                    "pending_results_json = excluded.pending_results_json, complete = 0",
                    (session_id, authority_id + 1, pending_json),
                )

            seeded._execute_write(_seed_stalled_progress)
            assert seeded.get_todo_state_messages(session_id) is None
            first_progress = seeded._conn.execute(
                "SELECT before_message_id, pending_results_json, complete "
                "FROM todo_authority_migrations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert first_progress is not None
            assert int(first_progress["before_message_id"]) == authority_id + 1
            assert len(first_progress["pending_results_json"].encode("utf-8")) < len(
                pending_json.encode("utf-8")
            )
            assert int(first_progress["complete"]) == 0
        finally:
            seeded.close()

        restarted = SessionDB(db_path)
        try:
            projected = restarted.get_todo_state_messages(session_id)
            assert [message["content"] for message in projected] == [
                authority_content
            ]
        finally:
            restarted.close()

    def test_invalid_near_limit_authority_cannot_repeat_identical_pending_state(
        self, tmp_path
    ):
        db_path = tmp_path / "todo-invalid-authority-progress.db"
        seeded = SessionDB(db_path)
        session_id = seeded.create_session(
            "todo-invalid-authority-progress", "api_server"
        )
        older_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "older-after-invalid-authority. Session", 1
        )
        try:
            seeded.replace_messages(
                session_id,
                [
                    _row("user", older_content),
                    _row("assistant", "invalid authority placeholder"),
                ],
            )
            rows = seeded._conn.execute(
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            authority_id = int(rows[-1]["id"])

            pending_result = {
                "id": authority_id + 1,
                "session_id": session_id,
                "role": "tool",
                "content": json.dumps(
                    {
                        "todos": [
                            {
                                "id": "pending",
                                "content": "near-limit pending result",
                                "status": "pending",
                            }
                        ]
                    },
                    separators=(",", ":"),
                ),
                "tool_call_id": "pending-invalid-authority",
                "tool_name": "todo",
                "display_kind": None,
                "display_metadata": {"padding": ""},
                "timestamp": 3.0,
            }
            pending_json = json.dumps(
                [pending_result], ensure_ascii=False, separators=(",", ":")
            )
            pending_result["display_metadata"]["padding"] = "p" * (
                1_099_000 - len(pending_json.encode("utf-8"))
            )
            pending_json = json.dumps(
                [pending_result], ensure_ascii=False, separators=(",", ":")
            )
            assert len(pending_json.encode("utf-8")) == 1_099_000

            invalid_authority = "{" + "a" * 889_998 + "}"
            assert len(invalid_authority.encode("utf-8")) == 890_000
            oversized_for_remaining_budget = "[" + "x" * 64_998 + "]"
            assert len(oversized_for_remaining_budget.encode("utf-8")) == 65_000

            def _seed_invalid_authority(conn):
                conn.execute(
                    "DELETE FROM todo_authorities WHERE session_id = ?",
                    (session_id,),
                )
                conn.execute(
                    "UPDATE messages SET todo_authority_json = ?, tool_calls = ? "
                    "WHERE id = ?",
                    (
                        invalid_authority,
                        oversized_for_remaining_budget,
                        authority_id,
                    ),
                )
                conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES "
                    "('todo_authority_legacy_high_water', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(authority_id),),
                )
                conn.execute(
                    "INSERT INTO todo_authority_migrations "
                    "(session_id, before_message_id, pending_results_json, "
                    " deferred_pending_results_json, phase, "
                    " authority_checked_message_id, complete) "
                    "VALUES (?, ?, ?, '[]', 'scan', NULL, 0) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "before_message_id = excluded.before_message_id, "
                    "pending_results_json = excluded.pending_results_json, "
                    "deferred_pending_results_json = '[]', phase = 'scan', "
                    "authority_checked_message_id = NULL, complete = 0",
                    (session_id, authority_id + 1, pending_json),
                )

            seeded._execute_write(_seed_invalid_authority)
        finally:
            seeded.close()

        seen_pending_states = set()
        observed_attempts = []
        projected = None
        for _attempt in range(6):
            restarted = SessionDB(db_path)
            try:
                tracemalloc.start()
                started = time.perf_counter()
                try:
                    result = restarted.get_todo_state_messages(session_id)
                finally:
                    elapsed = time.perf_counter() - started
                    _current, peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                observed_attempts.append(
                    {
                        **restarted._todo_migration_stats_for_tests(),
                        "wall_elapsed": elapsed,
                        "peak_bytes": peak_bytes,
                    }
                )
                if result is not None:
                    projected = result
                    break
                progress = restarted._conn.execute(
                    "SELECT before_message_id, length(pending_results_json) AS pending, "
                    "length(deferred_pending_results_json) AS deferred, phase, "
                    "authority_checked_message_id, complete "
                    "FROM todo_authority_migrations WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                assert progress is not None
                durable_state = tuple(progress)
                assert durable_state not in seen_pending_states
                seen_pending_states.add(durable_state)
            finally:
                restarted.close()

        assert projected is not None
        assert [message["content"] for message in projected] == [older_content]
        assert any(
            attempt["decoded_bytes"] >= 1_900_000 for attempt in observed_attempts
        )
        for attempt in observed_attempts:
            assert attempt["rows"] <= (
                SessionDB._TODO_MIGRATION_ROWS_PER_SLICE
                * SessionDB._TODO_MIGRATION_MAX_SLICES_PER_READ
            )
            assert (
                attempt["decoded_bytes"]
                <= SessionDB._TODO_MIGRATION_MAX_DECODED_BYTES_PER_READ
            )
            assert attempt["max_source_blob_bytes"] <= 1_100_000
            assert attempt["peak_bytes"] <= 6_000_000
            assert attempt["wall_elapsed"] < 1.0

    def test_corrupt_authority_probe_identity_recovers_after_restart(self, tmp_path):
        db_path = tmp_path / "todo-corrupt-probe-phase.db"
        seeded = SessionDB(db_path)
        session_id = seeded.create_session("todo-corrupt-probe-phase", "api_server")
        older_content = TODO_ROW_277757_CONTENT.replace(
            "s12. Session", "older-after-corrupt-probe. Session", 1
        )
        try:
            seeded.replace_messages(
                session_id,
                [
                    _row("user", older_content),
                    _row("assistant", "newer ordinary row"),
                ],
            )
            rows = seeded._conn.execute(
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            newest_id = int(rows[-1]["id"])

            def _seed_corrupt_probe(conn):
                conn.execute(
                    "DELETE FROM todo_authorities WHERE session_id = ?",
                    (session_id,),
                )
                conn.execute(
                    "INSERT INTO state_meta(key, value) VALUES "
                    "('todo_authority_legacy_high_water', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(newest_id),),
                )
                conn.execute(
                    "INSERT INTO todo_authority_migrations "
                    "(session_id, before_message_id, pending_results_json, "
                    " deferred_pending_results_json, phase, "
                    " authority_checked_message_id, complete) "
                    "VALUES (?, ?, '[]', '[]', 'authority_probe', ?, 0) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "before_message_id = excluded.before_message_id, "
                    "pending_results_json = '[]', "
                    "deferred_pending_results_json = '[]', "
                    "phase = 'authority_probe', "
                    "authority_checked_message_id = excluded.authority_checked_message_id, "
                    "complete = 0",
                    (session_id, newest_id + 1, newest_id + 100),
                )

            seeded._execute_write(_seed_corrupt_probe)
        finally:
            seeded.close()

        first_restart = SessionDB(db_path)
        try:
            assert first_restart.get_todo_state_messages(session_id) is None
            progress = first_restart._conn.execute(
                "SELECT before_message_id, pending_results_json, "
                "deferred_pending_results_json, phase, "
                "authority_checked_message_id, complete "
                "FROM todo_authority_migrations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert tuple(progress) == (newest_id + 1, "[]", "[]", "scan", None, 0)
        finally:
            first_restart.close()

        second_restart = SessionDB(db_path)
        try:
            projected = second_restart.get_todo_state_messages(session_id)
            assert [message["content"] for message in projected] == [older_content]
        finally:
            second_restart.close()

    @pytest.mark.asyncio
    async def test_todo_state_projection_retries_durable_migration_to_row_277757(
        self,
        tmp_path,
        adapter,
    ):
        db_path = tmp_path / "todo-http-migration-retry.db"
        seeded = SessionDB(db_path)
        session_id = seeded.create_session("todo-http-migration-retry", "api_server")
        try:
            seeded.replace_messages(
                session_id,
                [
                    _row("user", TODO_ROW_277757_CONTENT),
                    *[
                        _row("assistant", f"ordinary newer row {index}")
                        for index in range(2_048)
                    ],
                ],
            )
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE messages SET todo_pair_boundary_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        reopened = SessionDB(db_path)
        adapter._session_db = reopened
        statuses = []
        try:
            async with TestClient(TestServer(_messages_app(adapter))) as client:
                payload = None
                for _attempt in range(8):
                    response = await client.get(
                        f"/api/sessions/{session_id}/messages?projection=todo-state"
                    )
                    statuses.append(response.status)
                    payload = await response.json()
                    if response.status == 503:
                        assert payload["error"]["code"] == "todo_state_migration_pending"
                        continue
                    assert response.status == 200
                    break

            assert 503 in statuses
            assert statuses[-1] == 200
            assert [message["content"] for message in payload["data"]] == [
                TODO_ROW_277757_CONTENT
            ]
        finally:
            reopened.close()

    def test_todo_state_pending_migration_survives_compaction(self, tmp_path):
        db_path = tmp_path / "todo-pending-compaction.db"
        seeded = SessionDB(db_path)
        session_id = seeded.create_session("todo-pending-compaction", "api_server")
        try:
            seeded.replace_messages(
                session_id,
                [
                    _row("user", TODO_ROW_277757_CONTENT),
                    *[
                        _row("assistant", f"ordinary newer row {index}")
                        for index in range(2_048)
                    ],
                ],
            )
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE messages SET todo_pair_boundary_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        reopened = SessionDB(db_path)
        try:
            assert reopened.get_todo_state_messages(session_id) is None
            reopened.archive_and_compact(
                session_id,
                [_row("assistant", "compacted ordinary state")],
            )
        finally:
            reopened.close()

        projected = None
        for _attempt in range(8):
            reopened = SessionDB(db_path)
            try:
                projected = reopened.get_todo_state_messages(session_id)
            finally:
                reopened.close()
            if projected:
                break
        assert [message["content"] for message in projected] == [
            TODO_ROW_277757_CONTENT
        ]

    def test_todo_state_pending_migration_does_not_survive_destructive_rewrite(self, tmp_path):
        db_path = tmp_path / "todo-pending-rewrite.db"
        seeded = SessionDB(db_path)
        session_id = seeded.create_session("todo-pending-rewrite", "api_server")
        try:
            seeded.replace_messages(
                session_id,
                [
                    _row("user", TODO_ROW_277757_CONTENT),
                    *[
                        _row("assistant", f"ordinary newer row {index}")
                        for index in range(2_048)
                    ],
                ],
            )
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE messages SET todo_pair_boundary_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        reopened = SessionDB(db_path)
        try:
            assert reopened.get_todo_state_messages(session_id) is None
            reopened.replace_messages(
                session_id,
                [_row("assistant", "rewritten ordinary state")],
            )
            assert reopened.get_todo_state_messages(session_id) == []
        finally:
            reopened.close()

        reopened = SessionDB(db_path)
        try:
            assert reopened.get_todo_state_messages(session_id) == []
            progress = reopened._conn.execute(
                "SELECT complete FROM todo_authority_migrations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            assert progress is not None and progress["complete"] == 1
        finally:
            reopened.close()

    def test_todo_state_upgrade_materializes_preexisting_old_carrier(self, tmp_path):
        db_path = tmp_path / "todo-pre-materialization.db"
        db = SessionDB(db_path)
        session_id = db.create_session("todo-pre-materialization", "api_server")
        try:
            db.replace_messages(
                session_id,
                [
                    _row("user", TODO_ROW_277757_CONTENT),
                    *[
                        _row("assistant", f"ordinary newer row {index}")
                        for index in range(512)
                    ],
                ],
            )
        finally:
            db.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))

        reopened = SessionDB(db_path)
        try:
            metrics = self._todo_state_read_metrics(reopened, session_id)
            assert [message["content"] for message in metrics["messages"]] == [
                TODO_ROW_277757_CONTENT
            ]
            assert len(metrics["selects"]) == 1
            assert metrics["decoder_calls"] == 0
            assert metrics["vm_steps"] <= 200
        finally:
            reopened.close()

    def test_todo_state_upgrade_materializes_preexisting_paired_result(self, tmp_path):
        db_path = tmp_path / "todo-pre-materialization-pair.db"
        db = SessionDB(db_path)
        session_id = db.create_session("todo-pre-materialization-pair", "api_server")
        call = {
            "id": "old-pair",
            "type": "function",
            "function": {"name": "todo", "arguments": json.dumps({"todos": []})},
        }
        result = json.dumps(
            {"todos": [{"id": "old", "content": "old paired state", "status": "pending"}]}
        )
        try:
            db.replace_messages(
                session_id,
                [
                    _row("user", TODO_ROW_277757_CONTENT),
                    _row("assistant", "", tool_calls=[call]),
                    _row("tool", result, tool_call_id="old-pair"),
                ],
            )
        finally:
            db.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE messages SET todo_pair_boundary_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        reopened = SessionDB(db_path)
        try:
            projected = reopened.get_todo_state_messages(session_id)
            assert len(projected) == 2
            assert projected[0]["tool_calls"] == [call]
            assert projected[1]["content"] == result
        finally:
            reopened.close()

    @pytest.mark.asyncio
    async def test_todo_state_projection_bounds_payload_after_long_malformed_tool_history(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session("todo-state-bounded-session", "api_server")
        session_db.replace_messages(
            session_id,
            [
                _row("user", TODO_ROW_277757_CONTENT),
                *[
                    _row(
                        "assistant",
                        f"malformed tool-heavy candidate {index} " + "x" * 2_000,
                        tool_calls=[
                            {
                                "id": f"noise-{index}",
                                "type": "function",
                                "function": {"name": "not-todo", "arguments": "{}"},
                            }
                        ],
                    )
                    for index in range(256)
                ],
            ],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(
                f"/api/sessions/{session_id}/messages?projection=todo-state"
            )
            assert response.status == 200
            raw_payload = await response.read()
            payload = json.loads(raw_payload)

        assert len(raw_payload) < 7_000
        assert [message["content"] for message in payload["data"]] == [
            TODO_ROW_277757_CONTENT
        ]
        assert payload["pagination"] == {
            "exhausted": True,
            "has_more": False,
            "limit": 2,
            "next_before_id": None,
            "returned": 1,
        }

    @pytest.mark.asyncio
    async def test_todo_state_persistence_rejects_oversized_unpaired_evidence(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session("todo-state-oversized-session", "api_server")
        session_db.replace_messages(
            session_id,
            [
                _row(
                    "assistant",
                    "",
                    tool_calls=[
                        {
                            "id": "todo-call",
                            "type": "function",
                            "function": {
                                "name": "todo",
                                "arguments": json.dumps({"todos": []}),
                            },
                            "irrelevant": "x" * 1_200_000,
                        }
                    ],
                )
            ],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(
                f"/api/sessions/{session_id}/messages?projection=todo-state"
            )
            raw_payload = await response.read()

        assert response.status == 200
        assert len(raw_payload) < 1_000
        payload = json.loads(raw_payload)
        assert payload["data"] == []
        assert payload["pagination"]["returned"] == 0

    @pytest.mark.asyncio
    async def test_todo_state_projection_rejects_oversized_response(
        self,
        adapter,
        session_db,
        monkeypatch,
    ):
        session_id = session_db.create_session("todo-state-response-cap", "api_server")
        oversized = _row(
            "user",
            "界" * 400_000,
            display_kind=None,
            display_metadata=None,
        )
        oversized["session_id"] = session_id
        monkeypatch.setattr(
            session_db,
            "get_todo_state_messages",
            lambda _session_id: [oversized],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(
                f"/api/sessions/{session_id}/messages?projection=todo-state"
            )
            raw_payload = await response.read()

        assert response.status == 413
        assert len(raw_payload) < 1_000
        assert json.loads(raw_payload)["error"]["code"] == "todo_state_response_too_large"

    @pytest.mark.asyncio
    async def test_todo_state_projection_preserves_latest_paired_tool_result(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session("todo-state-tool-session", "api_server")
        tool_result = json.dumps(
            {"todos": [{"id": "tool", "content": "from tool", "status": "in_progress"}]}
        )
        session_db.replace_messages(
            session_id,
            [
                _row("user", TODO_ROW_277757_CONTENT),
                _row(
                    "assistant",
                    "",
                    tool_calls=[
                        {
                            "id": "todo-call",
                            "type": "function",
                            "function": {"name": "todo", "arguments": "{}"},
                        }
                    ],
                ),
                _row("tool", tool_result, tool_call_id="todo-call"),
                *[
                    _row(
                        "assistant",
                        f"later non-Todo tool call {index}",
                        tool_calls=[
                            {
                                "id": f"noise-{index}",
                                "type": "function",
                                "function": {"name": "not-todo", "arguments": "{}"},
                            }
                        ],
                    )
                    for index in range(64)
                ],
            ],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(
                f"/api/sessions/{session_id}/messages?projection=todo-state"
            )
            assert response.status == 200
            payload = await response.json()

        assert len(payload["data"]) == 2
        assert payload["data"][0]["tool_calls"][0]["function"]["name"] == "todo"
        assert payload["data"][1]["content"] == tool_result

    def test_todo_state_authority_requires_an_exact_complete_tool_pair(self, session_db):
        valid_result = json.dumps(
            {"todos": [{"id": "paired", "content": "paired", "status": "in_progress"}]}
        )
        def call(call_id="todo-call"):
            return _row(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "todo",
                            "arguments": json.dumps(
                                {
                                    "todos": [
                                        {
                                            "id": "args",
                                            "content": "args alone are not authority",
                                            "status": "pending",
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ],
            )

        cases = {
            "paired": (
                [call(), _row("tool", valid_result, tool_call_id="todo-call")],
                valid_result,
            ),
            "lone-named-result": (
                [_row("tool", valid_result, tool_name="todo")],
                TODO_ROW_277757_CONTENT,
            ),
            "lone-call": ([call()], TODO_ROW_277757_CONTENT),
            "wrong-call-id": (
                [call(), _row("tool", valid_result, tool_call_id="forged")],
                TODO_ROW_277757_CONTENT,
            ),
            "intervening-user-boundary": (
                [
                    call(),
                    _row("user", "interrupt the tool group"),
                    _row("tool", valid_result, tool_call_id="todo-call"),
                ],
                TODO_ROW_277757_CONTENT,
            ),
        }

        for label, (suffix, expected_content) in cases.items():
            session_id = session_db.create_session(f"todo-pair-{label}", "api_server")
            session_db.replace_messages(
                session_id,
                [_row("user", TODO_ROW_277757_CONTENT), *suffix],
            )
            projected = session_db.get_todo_state_messages(session_id)
            assert projected[-1]["content"] == expected_content, label
            if label == "paired":
                assert len(projected) == 2
                assert projected[0]["tool_calls"][0]["id"] == "todo-call"
            else:
                assert len(projected) == 1

    def test_first_matching_todo_result_becomes_authority(self, session_db):
        session_id = session_db.create_session("todo-first-result", "api_server")
        result = _todo_result("first")

        session_db.append_message(
            session_id,
            "assistant",
            "",
            tool_calls=[_todo_call("shared-call")],
        )
        session_db.append_message(
            session_id,
            "tool",
            result,
            tool_call_id="shared-call",
        )

        projected = session_db.get_todo_state_messages(session_id)
        assert projected is not None
        assert projected[-1]["content"] == result

    def test_later_duplicate_todo_result_cannot_supersede_first_after_restart(
        self, tmp_path
    ):
        db_path = tmp_path / "todo-duplicate-result-restart.db"
        db = SessionDB(db_path)
        session_id = db.create_session("todo-duplicate-result-restart", "api_server")
        first = _todo_result("first")
        duplicate = _todo_result("duplicate")
        try:
            db.append_message(
                session_id,
                "assistant",
                "",
                tool_calls=[_todo_call("shared-call")],
            )
            db.append_message(
                session_id,
                "tool",
                first,
                tool_call_id="shared-call",
            )
            duplicate_id = db.append_message(
                session_id,
                "tool",
                duplicate,
                tool_call_id="shared-call",
            )

            projected = db.get_todo_state_messages(session_id)
            assert projected is not None
            assert projected[-1]["content"] == first
            assert db._conn.execute(
                "SELECT todo_authority_json FROM messages WHERE id = ?",
                (duplicate_id,),
            ).fetchone()[0] is None
        finally:
            db.close()

        restarted = SessionDB(db_path)
        try:
            projected = restarted.get_todo_state_messages(session_id)
            assert projected is not None
            assert projected[-1]["content"] == first
        finally:
            restarted.close()

    @pytest.mark.parametrize(
        "first_result",
        ["{not-json", json.dumps({"todos": "not-a-list"})],
        ids=["malformed", "non-todo"],
    )
    def test_first_matching_malformed_result_consumes_todo_call(
        self, session_db, first_result
    ):
        session_id = session_db.create_session(
            f"todo-malformed-consumes-{first_result[:8]}", "api_server"
        )
        later = _todo_result("later")
        session_db.append_message(session_id, "user", TODO_ROW_277757_CONTENT)
        session_db.append_message(
            session_id,
            "assistant",
            "",
            tool_calls=[_todo_call("shared-call")],
        )
        session_db.append_message(
            session_id,
            "tool",
            first_result,
            tool_call_id="shared-call",
        )
        duplicate_id = session_db.append_message(
            session_id,
            "tool",
            later,
            tool_call_id="shared-call",
        )

        projected = session_db.get_todo_state_messages(session_id)
        assert projected is not None
        assert [message["content"] for message in projected] == [
            TODO_ROW_277757_CONTENT
        ]
        assert session_db._conn.execute(
            "SELECT todo_authority_json FROM messages WHERE id = ?",
            (duplicate_id,),
        ).fetchone()[0] is None

    def test_consuming_one_todo_call_preserves_distinct_sibling(self, session_db):
        session_id = session_db.create_session("todo-distinct-sibling", "api_server")
        first = _todo_result("first")
        sibling = _todo_result("sibling")
        assistant_id = session_db.append_message(
            session_id,
            "assistant",
            "",
            tool_calls=[_todo_call("call-a"), _todo_call("call-b")],
        )

        session_db.append_message(
            session_id,
            "tool",
            first,
            tool_call_id="call-a",
        )
        boundary_raw = session_db._conn.execute(
            "SELECT todo_pair_boundary_json FROM messages WHERE id = ?",
            (assistant_id,),
        ).fetchone()[0]
        assert [call["id"] for call in json.loads(boundary_raw)["tool_calls"]] == [
            "call-b"
        ]

        session_db.append_message(
            session_id,
            "tool",
            sibling,
            tool_call_id="call-b",
        )
        projected = session_db.get_todo_state_messages(session_id)
        assert projected is not None
        assert projected[-1]["content"] == sibling

    def test_legacy_migration_uses_first_result_for_reused_todo_call(self, tmp_path):
        db_path = tmp_path / "todo-duplicate-result-migration.db"
        seeded = SessionDB(db_path)
        session_id = seeded.create_session(
            "todo-duplicate-result-migration", "api_server"
        )
        first = _todo_result("first")
        duplicate = _todo_result("duplicate")
        try:
            seeded.replace_messages(
                session_id,
                [
                    _row("assistant", "", tool_calls=[_todo_call("shared-call")]),
                    _row("tool", first, tool_call_id="shared-call"),
                    _row("tool", duplicate, tool_call_id="shared-call"),
                ],
            )
        finally:
            seeded.close()

        with sqlite3.connect(db_path) as raw:
            raw.execute("UPDATE messages SET todo_authority_json = NULL")
            raw.execute("UPDATE messages SET todo_pair_boundary_json = NULL")
            raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION - 1,))
            raw.execute("DROP TABLE IF EXISTS todo_authorities")
            raw.execute("DROP TABLE IF EXISTS todo_authority_migrations")
            raw.execute("DROP TABLE IF EXISTS todo_pair_boundaries")

        migrated = SessionDB(db_path)
        try:
            projected = migrated.get_todo_state_messages(session_id)
            assert projected is not None
            assert projected[-1]["content"] == first
        finally:
            migrated.close()

        restarted = SessionDB(db_path)
        try:
            projected = restarted.get_todo_state_messages(session_id)
            assert projected is not None
            assert projected[-1]["content"] == first
        finally:
            restarted.close()

    def test_todo_state_paired_tail_survives_compaction_clone(self, session_db):
        session_id = session_db.create_session("todo-paired-tail", "api_server")
        session_db.append_message(session_id, "user", "prefix to compact")
        watermark = session_db.get_active_message_watermark(session_id)
        call = {
            "id": "todo-tail",
            "type": "function",
            "function": {"name": "todo", "arguments": json.dumps({"todos": []})},
        }
        result = json.dumps(
            {"todos": [{"id": "tail", "content": "tail survives", "status": "pending"}]}
        )
        session_db.append_message(session_id, "assistant", "", tool_calls=[call])
        session_db.append_message(
            session_id,
            "tool",
            result,
            tool_call_id="todo-tail",
        )

        session_db.archive_and_compact(
            session_id,
            [_row("user", TODO_ROW_277757_CONTENT)],
            watermark=watermark,
        )

        projected = session_db.get_todo_state_messages(session_id)
        assert len(projected) == 2
        assert projected[0]["tool_calls"] == [call]
        assert projected[1]["content"] == result

    def test_todo_state_rejects_each_malformed_list_as_a_whole(self, session_db):
        valid = {"id": "valid", "content": "must not survive alone", "status": "pending"}
        invalid_lists = [
            [valid, {"id": "bad", "content": "bad status", "status": "bogus"}],
            [{"id": "bad", "content": 42, "status": "pending"}],
            ["not-an-object"],
        ]

        for index, todos in enumerate(invalid_lists):
            structured_id = session_db.create_session(f"malformed-structured-{index}", "api_server")
            session_db.replace_messages(
                structured_id,
                [
                    _row("user", TODO_ROW_277757_CONTENT),
                    _row(
                        "user",
                        "opaque structured carrier",
                        display_kind="hidden",
                        display_metadata={"todo_snapshot": {"todos": todos}},
                    ),
                ],
            )
            assert session_db.get_todo_state_messages(structured_id)[0]["content"] == (
                TODO_ROW_277757_CONTENT
            )

        paired_id = session_db.create_session("malformed-paired-result", "api_server")
        call = {
            "id": "malformed-pair",
            "type": "function",
            "function": {"name": "todo", "arguments": json.dumps({"todos": []})},
        }
        session_db.replace_messages(
            paired_id,
            [
                _row("user", TODO_ROW_277757_CONTENT),
                _row("assistant", "", tool_calls=[call]),
                _row(
                    "tool",
                    json.dumps({"todos": invalid_lists[0]}),
                    tool_call_id="malformed-pair",
                ),
            ],
        )
        assert session_db.get_todo_state_messages(paired_id)[0]["content"] == (
            TODO_ROW_277757_CONTENT
        )

        empty_id = session_db.create_session("explicit-empty-state", "api_server")
        session_db.replace_messages(
            empty_id,
            [
                _row("user", TODO_ROW_277757_CONTENT),
                _row(
                    "user",
                    "opaque empty carrier",
                    display_kind="hidden",
                    display_metadata={"todo_snapshot": {"todos": []}},
                ),
            ],
        )
        projected = session_db.get_todo_state_messages(empty_id)
        assert projected[0]["content"] == ""
        assert projected[0]["display_metadata"] == {"todo_snapshot": {"todos": []}}

    @pytest.mark.asyncio
    async def test_todo_state_projection_skips_invalid_structured_candidate_for_older_state(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session("todo-state-invalid-boundary", "api_server")
        session_db.replace_messages(
            session_id,
            [
                _row("user", TODO_ROW_277757_CONTENT),
                _row(
                    "user",
                    "opaque invalid carrier",
                    display_kind="hidden",
                    display_metadata=json.dumps({"todo_snapshot": {"todos": "not-a-list"}}),
                ),
            ],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(
                f"/api/sessions/{session_id}/messages?projection=todo-state"
            )
            assert response.status == 200
            payload = await response.json()

        assert len(payload["data"]) == 1
        assert payload["data"][0]["content"] == TODO_ROW_277757_CONTENT
        assert payload["data"][0]["display_metadata"] is None

    @pytest.mark.asyncio
    async def test_legacy_todo_candidate_alias_uses_one_bounded_authority_read(
        self,
        adapter,
        session_db,
    ):
        assert len(TODO_ROW_277757_CONTENT) == 3067
        assert hashlib.sha256(TODO_ROW_277757_CONTENT.encode()).hexdigest() == (
            "8cb300e6a4e389ecd1facf179a69d081c7256bb4e376c00cbae5332b76cbeb75"
        )
        session_id = session_db.create_session("todo-candidate-session", "api_server")
        session_db.replace_messages(
            session_id,
            [
                _row("user", TODO_ROW_277757_CONTENT),
                *[
                    _row("assistant", f"ordinary trailing message {index}")
                    for index in range(121)
                ],
            ],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(
                f"/api/sessions/{session_id}/messages?projection=todo-state-candidates&limit=32"
            )
            assert response.status == 200
            payload = await response.json()

        assert [message["content"] for message in payload["data"]] == [
            TODO_ROW_277757_CONTENT
        ]
        assert payload["pagination"] == {
            "exhausted": True,
            "has_more": False,
            "limit": 2,
            "next_before_id": None,
            "returned": 1,
        }

    @pytest.mark.asyncio
    async def test_legacy_todo_candidate_alias_proves_bounded_absence(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session("todo-absence-session", "api_server")
        session_db.replace_messages(
            session_id,
            [_row("assistant", f"ordinary message {index}") for index in range(121)],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(
                f"/api/sessions/{session_id}/messages?projection=todo-state-candidates&limit=32"
            )
            assert response.status == 200
            payload = await response.json()

        assert payload["data"] == []
        assert payload["pagination"]["exhausted"] is True

    @pytest.mark.asyncio
    async def test_legacy_todo_candidate_alias_rejects_keyset_paging(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session("todo-candidate-no-paging", "api_server")
        session_db.replace_messages(
            session_id,
            [_row("user", TODO_ROW_277757_CONTENT)],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(
                f"/api/sessions/{session_id}/messages"
                "?projection=todo-state-candidates&limit=32&before_id=999"
            )
            payload = await response.json()

        assert response.status == 400
        assert payload["error"]["code"] == "invalid_pagination"

    @pytest.mark.asyncio
    async def test_messages_endpoint_preserves_persisted_todo_snapshot_metadata(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session("todo-projection-session", "api_server")
        session_db.replace_messages(
            session_id,
            [
                _row(
                    "user",
                    "opaque todo carrier",
                    display_kind="hidden",
                    display_metadata=json.dumps(TODO_SNAPSHOT_METADATA),
                )
            ],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(f"/api/sessions/{session_id}/messages")
            assert response.status == 200
            payload = await response.json()

        assert payload["data"][0]["display_metadata"] == TODO_SNAPSHOT_METADATA

    @pytest.mark.asyncio
    async def test_messages_endpoint_never_serves_compaction_scaffolding(
        self,
        adapter,
        session_db,
    ):
        session_id = session_db.create_session("projection-session", "api_server")
        session_db.replace_messages(
            session_id,
            [
                _row("user", STANDALONE_SUMMARY),
                _row("assistant", MERGED_CARRIER),
                _row("user", REAL_USER),
            ],
        )

        async with TestClient(TestServer(_messages_app(adapter))) as client:
            response = await client.get(f"/api/sessions/{session_id}/messages")
            assert response.status == 200
            payload = await response.json()

        messages = payload["data"]
        assert len(messages) == 3
        assert messages[0]["content"] == ""
        assert messages[0]["display_kind"] == "hidden"
        assert messages[1]["content"] == "Refactor complete."
        assert messages[2]["content"] == REAL_USER
        rendered = " ".join(str(message.get("content") or "") for message in messages)
        assert "PRIOR CONTEXT" not in rendered
        assert "CONTEXT COMPACTION" not in rendered
