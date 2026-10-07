"""Regression for #130961: compaction TODO scaffolding must not read as a human turn.

Producer (`_fold_todo_snapshot`) types standalone snapshots `display_kind=hidden`;
display projections (`project_compaction_message_for_display`,
`_project_for_display`) hide pure scaffolding and strip the snapshot suffix from
merged carriers, preserving authentic human text. The live TODO widget reads
tool results, never these user-role rows.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.compaction_display import is_todo_snapshot_message, project_compaction_message_for_display
from agent.session_persistence import _summary_display_kind
from hermes_state import SessionDB
from tools.todo_tool import TODO_INJECTION_HEADER

PURE_SNAPSHOT = f"{TODO_INJECTION_HEADER}\n- [>] demo. Internal task snapshot. (in_progress)"
MERGED = f"real user question\n\n{PURE_SNAPSHOT}"
GENUINE = "real user question"


def _build_agent_with_db(db: SessionDB, session_id: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform="cli",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    compressor = MagicMock()
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent.compression_in_place = False
    return agent


def _msgs(n: int = 20):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * 400}
        for i in range(n)
    ]


class TestTodoSnapshotRecognition:
    def test_pure_snapshot_recognized(self):
        assert is_todo_snapshot_message({"role": "user", "content": PURE_SNAPSHOT}) is True

    def test_flagged_row_recognized_without_header(self):
        assert is_todo_snapshot_message(
            {"role": "user", "content": "snapshot", "_todo_snapshot_synthetic": True}
        ) is True

    def test_genuine_user_not_recognized(self):
        assert is_todo_snapshot_message({"role": "user", "content": GENUINE}) is False
        assert is_todo_snapshot_message({"role": "assistant", "content": PURE_SNAPSHOT}) is False


class TestCompactionDisplayProjection:
    def test_pure_snapshot_projects_to_none(self):
        assert project_compaction_message_for_display({"role": "user", "content": PURE_SNAPSHOT}) is None

    def test_issue_fixture_projects_to_none(self):
        row = {
            "id": 1,
            "role": "user",
            "content": PURE_SNAPSHOT,
            "display_kind": None,
            "timestamp": 1790891125,
        }
        assert project_compaction_message_for_display(row) is None

    def test_merged_carrier_preserves_authentic_text(self):
        projected = project_compaction_message_for_display({"role": "user", "content": MERGED})
        assert projected is not None
        assert projected["content"] == "real user question"
        assert TODO_INJECTION_HEADER not in str(projected["content"])

    def test_genuine_message_untouched(self):
        message = {"role": "user", "content": GENUINE}
        projected = project_compaction_message_for_display(message)
        assert projected == message
        assert projected is not message

    def test_list_content_pure_hides(self):
        content = [{"type": "text", "text": PURE_SNAPSHOT}]
        assert project_compaction_message_for_display({"role": "user", "content": content}) is None

    def test_list_content_merged_strips_text_part_only(self):
        content = [
            {"type": "text", "text": f"hello\n\n{PURE_SNAPSHOT}"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ]
        projected = project_compaction_message_for_display({"role": "user", "content": content})
        assert projected is not None
        assert projected["content"][0] == {"type": "text", "text": "hello"}
        assert projected["content"][1]["type"] == "image_url"


class TestFoldTodoSnapshotProducer:
    def test_standalone_snapshot_typed_hidden(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "TODO_PRODUCER_HIDDEN"
        db.create_session(sid, source="cli")
        agent = _build_agent_with_db(db, sid)
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "acknowledged"},
        ]
        agent._todo_store._items = [{"id": "demo", "content": "Internal task snapshot.", "status": "in_progress"}]
        try:
            compressed, _ = agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        finally:
            db.close()
        snapshots = [m for m in compressed if TODO_INJECTION_HEADER in str(m.get("content") or "")]
        assert len(snapshots) == 1
        row = snapshots[0]
        assert row.get("_todo_snapshot_synthetic") is True
        assert row.get("display_kind") == "hidden"
        assert row.get("role") == "user"

    def test_merged_snapshot_stays_visible(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "TODO_PRODUCER_MERGED"
        db.create_session(sid, source="cli")
        agent = _build_agent_with_db(db, sid)
        original = _msgs()
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": original[-1]["content"]},
            {"role": "user", "content": "keep this human text"},
        ]
        agent._todo_store._items = [{"id": "t1", "content": "fresh task", "status": "pending"}]
        try:
            compressed, _ = agent._compress_context(original, "sys", approx_tokens=120_000)
        finally:
            db.close()
        tail = compressed[-1]
        assert "keep this human text" in str(tail.get("content"))
        assert TODO_INJECTION_HEADER in str(tail.get("content"))
        assert tail.get("display_kind") != "hidden"


class TestPersistenceDisplayKind:
    def test_pure_synthetic_persists_hidden(self):
        row = {"role": "user", "content": PURE_SNAPSHOT, "_todo_snapshot_synthetic": True}
        assert _summary_display_kind(row) == "hidden"

    def test_merged_row_keeps_visibility(self):
        assert _summary_display_kind({"role": "user", "content": MERGED}) is None

    def test_genuine_row_keeps_visibility(self):
        assert _summary_display_kind({"role": "user", "content": GENUINE}) is None

    def test_hidden_producer_row_round_trips(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            sid = db.create_session("TODO_PERSIST_ROUNDTRIP", "cli")
            db.append_message(sid, "user", GENUINE)
            from agent.session_persistence import _db_flush_row

            class FakeAgent:
                _persist_user_message_override = None

            row = _db_flush_row(
                FakeAgent(),
                {"role": "user", "content": PURE_SNAPSHOT, "_todo_snapshot_synthetic": True,
                 "display_kind": "hidden"},
                False,
            )
            assert row["display_kind"] == "hidden"
            db.append_messages_batch(sid, [row])
            stored = db.get_messages(sid)
            snapshot_rows = [m for m in stored if TODO_INJECTION_HEADER in str(m.get("content") or "")]
            assert len(snapshot_rows) == 1
            assert snapshot_rows[0].get("display_kind") == "hidden"
            assert not snapshot_rows[0].get("_compressed_summary")
        finally:
            db.close()


class TestRestDisplayProjection:
    def test_pure_snapshot_hidden_for_rest(self):
        from hermes_cli.web_routers.sessions import _project_for_display

        out = _project_for_display([{"role": "user", "content": PURE_SNAPSHOT, "display_kind": None}])
        assert len(out) == 1
        assert out[0].get("display_kind") == "hidden"

    def test_merged_carrier_shows_authentic_text(self):
        from hermes_cli.web_routers.sessions import _project_for_display

        out = _project_for_display([{"role": "user", "content": MERGED, "display_kind": None}])
        assert len(out) == 1
        assert out[0].get("display_kind") is None
        assert out[0].get("display_content") == "real user question"
        assert TODO_INJECTION_HEADER in str(out[0].get("content"))

    def test_genuine_message_never_hidden_or_altered(self):
        from hermes_cli.web_routers.sessions import _project_for_display

        out = _project_for_display([{"role": "user", "content": GENUINE, "display_kind": None}])
        assert len(out) == 1
        assert out[0].get("display_kind") is None
        assert "display_content" not in out[0]
        assert out[0].get("content") == GENUINE
