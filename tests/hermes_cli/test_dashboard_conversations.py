"""Focused tests for the dashboard Conversations API."""

from pathlib import Path

import pytest


UNSET = object()


def _set_session_fields(db, session_id: str, **fields) -> None:
    assignments = []
    params = []
    for key, value in fields.items():
        assignments.append(f"{key} = ?")
        params.append(value)
    params.append(session_id)

    def _do(conn):
        conn.execute(
            f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
            params,
        )

    db._execute_write(_do)


def _set_message_timestamp(db, message_id: int, timestamp: float) -> None:
    def _do(conn):
        conn.execute(
            "UPDATE messages SET timestamp = ? WHERE id = ?",
            (timestamp, message_id),
        )

    db._execute_write(_do)


def _append_message(db, session_id: str, role: str, content: str, timestamp: float, *, tool_name: str | None = None) -> int:
    message_id = db.append_message(session_id, role, content=content, tool_name=tool_name)
    _set_message_timestamp(db, message_id, timestamp)
    return message_id


def _seed_conversation_graph(db_path: Path) -> dict[str, str]:
    from agent.context_compressor import SUMMARY_PREFIX
    from hermes_state import SessionDB

    db = SessionDB(db_path=db_path)
    try:
        ids = {
            "root": "root-compress",
            "continuation": "cont-compress",
            "branch_keep": "branch-keep",
            "nested_delete": "nested-delete",
            "nested_grandchild": "nested-grandchild",
            "telegram": "telegram-root",
            "probe": "internal-probe",
        }

        db.create_session(ids["root"], "cli")
        _append_message(db, ids["root"], "user", "Alpha roadmap question", 100.1)
        _append_message(db, ids["root"], "assistant", "Alpha answer part 1", 100.2)
        _append_message(db, ids["root"], "user", "Middle compressed detail", 100.25)
        _append_message(db, ids["root"], "assistant", "Middle compressed answer", 100.28)
        _append_message(db, ids["root"], "user", "Recent carry-forward context", 100.3)
        _append_message(db, ids["root"], "assistant", "Recent carry-forward answer", 100.4)
        _append_message(db, ids["root"], "tool", "tool trace", 100.45, tool_name="search")
        _append_message(db, ids["root"], "user", "[system: compressed context]", 100.5)
        _set_session_fields(
            db,
            ids["root"],
            started_at=100.0,
            ended_at=101.0,
            end_reason="compression",
            title=None,
        )

        db.create_session(ids["continuation"], "cli", parent_session_id=ids["root"])
        _append_message(db, ids["continuation"], "user", "Alpha roadmap question", 102.1)
        _append_message(db, ids["continuation"], "assistant", "Alpha answer part 1", 102.2)
        _append_message(db, ids["continuation"], "user", "Middle compressed detail", 102.22)
        _append_message(db, ids["continuation"], "user", f"{SUMMARY_PREFIX}\nCompacted work so far", 102.25)
        _append_message(db, ids["continuation"], "user", "Recent carry-forward context", 102.3)
        _append_message(db, ids["continuation"], "assistant", "Recent carry-forward answer", 102.4)
        _append_message(db, ids["continuation"], "user", "Post-compression next step", 102.5)
        _append_message(db, ids["continuation"], "assistant", "Final insight from continuation", 102.6)
        _append_message(db, ids["continuation"], "tool", "continuation tool", 102.65, tool_name="delegate")
        _set_session_fields(
            db,
            ids["continuation"],
            started_at=102.0,
            ended_at=200.0,
            end_reason="branched",
            title=None,
        )

        db.create_session(ids["branch_keep"], "cli", parent_session_id=ids["continuation"])
        _append_message(db, ids["branch_keep"], "user", "Branch keep question", 201.1)
        _append_message(db, ids["branch_keep"], "assistant", "Branch keep answer", 201.2)
        _set_session_fields(db, ids["branch_keep"], started_at=201.0, ended_at=None, end_reason=None, title=None)

        db.create_session(ids["nested_delete"], "cli", parent_session_id=ids["continuation"])
        _append_message(db, ids["nested_delete"], "user", "Nested delete question", 250.1)
        _set_session_fields(db, ids["nested_delete"], started_at=250.0, ended_at=251.0, end_reason="complete", title=None)

        db.create_session(ids["nested_grandchild"], "cli", parent_session_id=ids["nested_delete"])
        _append_message(db, ids["nested_grandchild"], "user", "Nested grandchild question", 251.1)
        _set_session_fields(db, ids["nested_grandchild"], started_at=251.0, ended_at=None, end_reason=None, title=None)

        db.create_session(ids["telegram"], "telegram")
        _append_message(db, ids["telegram"], "user", "Telegram ping", 300.1)
        _append_message(db, ids["telegram"], "assistant", "Telegram pong", 300.2)
        _set_session_fields(db, ids["telegram"], started_at=300.0, ended_at=None, end_reason=None, title="Telegram thread")

        db.create_session(ids["probe"], "cli")
        _append_message(db, ids["probe"], "user", "reply exactly [silent]", 50.1)
        _set_session_fields(db, ids["probe"], started_at=50.0, ended_at=None, end_reason=None, title=None)

        return ids
    finally:
        db.close()


def test_strip_compaction_artifact_handles_merged_summary_prefix():
    from agent.context_compressor import SUMMARY_PREFIX
    from hermes_cli.dashboard_conversations import _strip_compaction_artifact, SUMMARY_MERGE_SEPARATOR

    merged = (
        f"{SUMMARY_PREFIX}\nCompacted work so far\n\n"
        f"{SUMMARY_MERGE_SEPARATOR}\n\n"
        "Real tail message"
    )

    assert _strip_compaction_artifact(merged) == "Real tail message"


def test_list_conversations_supports_legacy_compressed_end_reason(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "legacy-compressed.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("legacy-root", "cli")
        _append_message(db, "legacy-root", "user", "Legacy carry-forward", 10.1)
        _append_message(db, "legacy-root", "assistant", "Legacy answer", 10.2)
        _set_session_fields(db, "legacy-root", started_at=10.0, ended_at=11.0, end_reason="compressed", title=None)

        db.create_session("legacy-child", "cli", parent_session_id="legacy-root")
        _append_message(db, "legacy-child", "user", "Legacy carry-forward", 12.1)
        _append_message(db, "legacy-child", "assistant", "Legacy follow-up", 12.2)
        _set_session_fields(db, "legacy-child", started_at=12.0, ended_at=None, end_reason=None, title=None)
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    sessions_by_root = {item["thread_root_id"]: item for item in data["sessions"]}
    assert sessions_by_root["legacy-root"]["thread_session_count"] == 2



def test_list_conversations_keeps_resumed_compression_chains_together(monkeypatch, tmp_path):
    import hermes_state
    from agent.context_compressor import SUMMARY_PREFIX
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "resumed-compression.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("resumed-root", "cli")
        _append_message(db, "resumed-root", "user", "Original prompt", 15.1)
        _append_message(db, "resumed-root", "assistant", "Original answer", 15.2)
        _set_session_fields(db, "resumed-root", started_at=15.0, ended_at=16.0, end_reason="compression", title="Original prompt")

        db.create_session("resumed-child", "cli", parent_session_id="resumed-root")
        _append_message(db, "resumed-child", "user", f"{SUMMARY_PREFIX}\nCompacted work so far", 16.1)
        _append_message(db, "resumed-child", "assistant", "Continuation answer", 16.2)
        _set_session_fields(db, "resumed-child", started_at=16.0, ended_at=None, end_reason=None, title="Original prompt #2")

        db.reopen_session("resumed-root")
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    sessions_by_root = {item["thread_root_id"]: item for item in data["sessions"]}
    assert sessions_by_root["resumed-root"]["thread_session_count"] == 2
    assert "resumed-child" not in sessions_by_root



def test_list_conversations_keeps_legitimate_review_prompt_sessions(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "review-session.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("review-session", "cli")
        for idx in range(10):
            role = "user" if idx % 2 == 0 else "assistant"
            content = "Review this PR carefully" if idx == 0 else f"message {idx}"
            _append_message(db, "review-session", role, content, 20.0 + idx)
        _set_session_fields(db, "review-session", started_at=20.0, ended_at=None, end_reason=None, title=None)
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    roots = {item["thread_root_id"] for item in data["sessions"]}
    assert "review-session" in roots



def test_list_conversations_keeps_legitimate_exact_ok_prompt_sessions(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "exact-ok-session.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("exact-ok-session", "cli")
        _append_message(db, "exact-ok-session", "user", "Reply with exactly OK if you understand, then explain why.", 25.1)
        _append_message(db, "exact-ok-session", "assistant", "OK. Here is why.", 25.2)
        _set_session_fields(db, "exact-ok-session", started_at=25.0, ended_at=None, end_reason=None, title=None)
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    roots = {item["thread_root_id"] for item in data["sessions"]}
    assert "exact-ok-session" in roots



def test_list_conversations_hides_titled_internal_probe_sessions(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "titled-probe-session.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("titled-probe-session", "cli")
        _append_message(db, "titled-probe-session", "user", "reply exactly [silent]", 27.1)
        _append_message(db, "titled-probe-session", "assistant", "[silent]", 27.2)
        _set_session_fields(db, "titled-probe-session", started_at=27.0, ended_at=None, end_reason=None, title="Silent reply check")
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    roots = {item["thread_root_id"] for item in data["sessions"]}
    assert "titled-probe-session" not in roots



def test_list_conversations_hides_tool_source_sessions(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "tool-session.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("tool-session", "tool")
        _append_message(db, "tool-session", "user", "tool runner task", 30.1)
        _append_message(db, "tool-session", "assistant", "tool runner answer", 30.2)
        _set_session_fields(db, "tool-session", started_at=30.0, ended_at=None, end_reason=None, title=None)
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    roots = {item["thread_root_id"] for item in data["sessions"]}
    assert "tool-session" not in roots


def test_list_conversations_hides_branched_tool_child_sessions(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "branched-tool-child.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("parent", "cli")
        _append_message(db, "parent", "user", "Parent branch source", 40.1)
        _append_message(db, "parent", "assistant", "Parent answer", 40.2)
        _set_session_fields(db, "parent", started_at=40.0, ended_at=41.0, end_reason="branched", title=None)

        db.create_session("child-tool", "tool", parent_session_id="parent")
        _append_message(db, "child-tool", "user", "Tool child prompt", 41.1)
        _append_message(db, "child-tool", "assistant", "Tool child answer", 41.2)
        _set_session_fields(db, "child-tool", started_at=41.0, ended_at=None, end_reason=None, title=None)
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    roots = {item["thread_root_id"] for item in data["sessions"]}
    assert "child-tool" not in roots


def test_list_conversations_links_continuation_when_child_preview_differs(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "preview-mismatch.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("root", "cli")
        _append_message(db, "root", "user", "Original first prompt", 10.1)
        _append_message(db, "root", "assistant", "Original first answer", 10.2)
        _append_message(db, "root", "user", "Carry forward tail", 10.3)
        _append_message(db, "root", "assistant", "Carry forward answer", 10.4)
        _set_session_fields(db, "root", started_at=10.0, ended_at=11.0, end_reason="compression", title=None)

        db.create_session("child", "cli", parent_session_id="root")
        _append_message(db, "child", "user", "Rewritten first visible message after compression", 11.1)
        _append_message(db, "child", "assistant", "Fresh continuation answer", 11.2)
        _set_session_fields(db, "child", started_at=11.0, ended_at=None, end_reason=None, title=None)
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    sessions_by_root = {item["thread_root_id"]: item for item in data["sessions"]}
    assert sessions_by_root["root"]["thread_session_count"] == 2



def test_list_conversations_surfaces_gateway_created_branch_children(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "gateway-branch.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("parent-gateway", "telegram")
        _append_message(db, "parent-gateway", "user", "Gateway parent question", 20.1)
        _append_message(db, "parent-gateway", "assistant", "Gateway parent answer", 20.2)
        _set_session_fields(db, "parent-gateway", started_at=20.0, ended_at=None, end_reason=None, title="Gateway root")

        db.create_session("gateway-branch", "telegram", parent_session_id="parent-gateway")
        _append_message(db, "gateway-branch", "user", "Gateway branch question", 21.1)
        _append_message(db, "gateway-branch", "assistant", "Gateway branch answer", 21.2)
        _set_session_fields(db, "gateway-branch", started_at=21.0, ended_at=None, end_reason=None, title="Gateway branch")
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    roots = {item["thread_root_id"] for item in data["sessions"]}
    assert "gateway-branch" in roots



def test_list_conversations_surfaces_nested_gateway_branches(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations
    from hermes_state import SessionDB

    db_path = tmp_path / "nested-gateway-branch.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("gateway-root", "telegram")
        _append_message(db, "gateway-root", "user", "Gateway root question", 30.1)
        _append_message(db, "gateway-root", "assistant", "Gateway root answer", 30.2)
        _set_session_fields(db, "gateway-root", started_at=30.0, ended_at=None, end_reason=None, title="Gateway root")

        db.create_session("gateway-branch-1", "telegram", parent_session_id="gateway-root")
        _append_message(db, "gateway-branch-1", "user", "Gateway branch one question", 31.1)
        _append_message(db, "gateway-branch-1", "assistant", "Gateway branch one answer", 31.2)
        _set_session_fields(db, "gateway-branch-1", started_at=31.0, ended_at=None, end_reason=None, title="Gateway branch one")

        db.create_session("gateway-branch-2", "telegram", parent_session_id="gateway-branch-1")
        _append_message(db, "gateway-branch-2", "user", "Gateway branch two question", 32.1)
        _append_message(db, "gateway-branch-2", "assistant", "Gateway branch two answer", 32.2)
        _set_session_fields(db, "gateway-branch-2", started_at=32.0, ended_at=None, end_reason=None, title="Gateway branch two")
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    data = list_conversations(limit=10, offset=0)
    roots = {item["thread_root_id"] for item in data["sessions"]}
    assert {"gateway-root", "gateway-branch-1", "gateway-branch-2"}.issubset(roots)



def test_delete_conversation_orphans_nested_gateway_branches(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import delete_conversation
    from hermes_state import SessionDB

    db_path = tmp_path / "nested-gateway-delete.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("gateway-root", "telegram")
        _append_message(db, "gateway-root", "user", "Gateway root question", 40.1)
        _append_message(db, "gateway-root", "assistant", "Gateway root answer", 40.2)
        _set_session_fields(db, "gateway-root", started_at=40.0, ended_at=None, end_reason=None, title="Gateway root")

        db.create_session("gateway-branch-1", "telegram", parent_session_id="gateway-root")
        _append_message(db, "gateway-branch-1", "user", "Gateway branch one question", 41.1)
        _append_message(db, "gateway-branch-1", "assistant", "Gateway branch one answer", 41.2)
        _set_session_fields(db, "gateway-branch-1", started_at=41.0, ended_at=None, end_reason=None, title="Gateway branch one")

        db.create_session("gateway-branch-2", "telegram", parent_session_id="gateway-branch-1")
        _append_message(db, "gateway-branch-2", "user", "Gateway branch two question", 42.1)
        _append_message(db, "gateway-branch-2", "assistant", "Gateway branch two answer", 42.2)
        _set_session_fields(db, "gateway-branch-2", started_at=42.0, ended_at=None, end_reason=None, title="Gateway branch two")
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    result = delete_conversation("gateway-branch-1")
    assert result["ok"] is True

    db = SessionDB(db_path=db_path)
    try:
        assert db.get_session("gateway-branch-1") is None
        branch_two = db.get_session("gateway-branch-2")
        assert branch_two is not None
        assert branch_two["parent_session_id"] is None
    finally:
        db.close()



def test_delete_conversation_orphans_visible_descendants_under_hidden_intermediate(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import delete_conversation
    from hermes_state import SessionDB

    db_path = tmp_path / "hidden-intermediate-delete.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("gateway-root", "telegram")
        _append_message(db, "gateway-root", "user", "Gateway root question", 50.1)
        _append_message(db, "gateway-root", "assistant", "Gateway root answer", 50.2)
        _set_session_fields(db, "gateway-root", started_at=50.0, ended_at=None, end_reason=None, title="Gateway root")

        db.create_session("hidden-intermediate", "telegram", parent_session_id="gateway-root")
        _append_message(db, "hidden-intermediate", "user", "Hidden intermediate question", 51.1)
        _append_message(db, "hidden-intermediate", "assistant", "Hidden intermediate answer", 51.2)
        _set_session_fields(db, "hidden-intermediate", started_at=51.0, ended_at=None, end_reason=None, title=None)

        db.create_session("visible-leaf", "telegram", parent_session_id="hidden-intermediate")
        _append_message(db, "visible-leaf", "user", "Visible leaf question", 52.1)
        _append_message(db, "visible-leaf", "assistant", "Visible leaf answer", 52.2)
        _set_session_fields(db, "visible-leaf", started_at=52.0, ended_at=None, end_reason=None, title="Visible leaf")
    finally:
        db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    result = delete_conversation("gateway-root")
    assert result["ok"] is True

    db = SessionDB(db_path=db_path)
    try:
        assert db.get_session("gateway-root") is None
        assert db.get_session("hidden-intermediate") is None
        visible_leaf = db.get_session("visible-leaf")
        assert visible_leaf is not None
        assert visible_leaf["parent_session_id"] is None
    finally:
        db.close()



def test_list_conversations_clamps_direct_pagination_inputs(monkeypatch, tmp_path):
    import hermes_state
    from hermes_cli.dashboard_conversations import list_conversations

    db_path = tmp_path / "pagination-clamp.db"
    _seed_conversation_graph(db_path)

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

    clamped = list_conversations(limit=0, offset=-10)
    assert clamped["limit"] == 1
    assert clamped["offset"] == 0
    assert len(clamped["sessions"]) == 1

    capped = list_conversations(limit=999999, offset=-10)
    assert capped["limit"] == 500
    assert capped["offset"] == 0
    assert len(capped["sessions"]) == capped["total"] == 3


class TestDashboardConversationsAPI:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, tmp_path):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.db_path = tmp_path / "state.db"
        self.ids = _seed_conversation_graph(self.db_path)
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", self.db_path)

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    def test_list_conversations_supports_source_filters_and_chain_aggregation(self):
        resp = self.client.get(
            "/api/conversations",
            params={"source": "cli", "limit": 10, "offset": 0},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "cli"
        assert data["mode"] == "source-filtered-conversations"
        assert data["all_total"] == 3
        assert data["total"] == 2
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert data["sources"] == ["cli", "telegram"]

        sessions_by_root = {item["thread_root_id"]: item for item in data["sessions"]}
        assert set(sessions_by_root) == {self.ids["root"], self.ids["branch_keep"]}
        assert self.ids["probe"] not in sessions_by_root
        root = sessions_by_root[self.ids["root"]]
        assert root["thread_session_count"] == 2
        assert root["thread_message_count"] == 17
        assert root["title"] == "Alpha roadmap question"
        assert root["source"] == "cli"

    def test_list_conversations_searches_visible_transcript_content(self):
        resp = self.client.get(
            "/api/conversations",
            params={"q": "Final insight", "limit": 10, "offset": 0},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["sessions"][0]["thread_root_id"] == self.ids["root"]
        assert "Final insight" in data["sessions"][0]["snippet"]

        summary_resp = self.client.get(
            "/api/conversations",
            params={"q": "Compacted work", "limit": 10, "offset": 0},
        )
        assert summary_resp.status_code == 200
        assert summary_resp.json()["total"] == 0

    def test_list_conversations_paginates_without_overlap_and_keeps_metadata(self):
        pages = [
            self.client.get("/api/conversations", params={"limit": 1, "offset": offset})
            for offset in range(4)
        ]

        for response in pages:
            assert response.status_code == 200

        datasets = [response.json() for response in pages]
        expected_page_ids = [
            self.ids["telegram"],
            self.ids["branch_keep"],
            self.ids["root"],
        ]

        for offset, data in enumerate(datasets):
            assert data["total"] == 3
            assert data["all_total"] == 3
            assert data["sources"] == ["cli", "telegram"]
            assert data["offset"] == offset

        assert [data["sessions"][0]["thread_root_id"] for data in datasets[:3]] == expected_page_ids
        assert all(len(data["sessions"]) == 1 for data in datasets[:3])
        assert datasets[3]["sessions"] == []

    def test_list_conversations_keeps_filtered_pagination_totals(self):
        pages = [
            self.client.get(
                "/api/conversations",
                params={"source": "cli", "limit": 1, "offset": offset},
            )
            for offset in range(3)
        ]

        for response in pages:
            assert response.status_code == 200

        first_data, second_data, empty_data = [response.json() for response in pages]

        assert first_data["total"] == 2
        assert second_data["total"] == 2
        assert empty_data["total"] == 2
        assert first_data["all_total"] == 3
        assert second_data["all_total"] == 3
        assert empty_data["all_total"] == 3
        assert first_data["sources"] == ["cli", "telegram"]
        assert second_data["sources"] == ["cli", "telegram"]
        assert empty_data["sources"] == ["cli", "telegram"]
        assert first_data["offset"] == 0
        assert second_data["offset"] == 1
        assert empty_data["offset"] == 2
        assert len(first_data["sessions"]) == 1
        assert len(second_data["sessions"]) == 1
        assert empty_data["sessions"] == []
        assert [
            first_data["sessions"][0]["thread_root_id"],
            second_data["sessions"][0]["thread_root_id"],
        ] == [self.ids["branch_keep"], self.ids["root"]]

    @pytest.mark.parametrize(
        ("params", "field"),
        [
            ({"limit": 0}, "limit"),
            ({"limit": 999999}, "limit"),
            ({"offset": -10}, "offset"),
        ],
    )
    def test_list_conversations_rejects_invalid_pagination_params(self, params, field):
        resp = self.client.get("/api/conversations", params=params)

        assert resp.status_code == 422
        assert any(error["loc"][-1] == field for error in resp.json()["detail"])

    def test_get_conversation_messages_filters_hidden_messages_across_chain(self):
        resp = self.client.get(f"/api/conversations/{self.ids['root']}/messages")

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == self.ids["root"]
        assert data["thread_session_count"] == 2
        assert data["visible_count"] == 8
        assert [message["role"] for message in data["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert [message["content"] for message in data["messages"]] == [
            "Alpha roadmap question",
            "Alpha answer part 1",
            "Middle compressed detail",
            "Middle compressed answer",
            "Recent carry-forward context",
            "Recent carry-forward answer",
            "Post-compression next step",
            "Final insight from continuation",
        ]

    def test_get_conversation_messages_404_for_unknown_root(self):
        resp = self.client.get("/api/conversations/does-not-exist/messages")
        assert resp.status_code == 404

    def test_delete_conversation_404_for_unknown_root(self):
        resp = self.client.delete("/api/conversations/does-not-exist")
        assert resp.status_code == 404

    def test_delete_conversation_removes_nested_non_display_sessions_and_orphans_branches(self):
        resp = self.client.delete(f"/api/conversations/{self.ids['root']}")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"ok": True, "deleted_sessions": 4, "deleted_messages": 19}

        from hermes_state import SessionDB

        db = SessionDB(db_path=self.db_path)
        try:
            assert db.get_session(self.ids["root"]) is None
            assert db.get_session(self.ids["continuation"]) is None
            assert db.get_session(self.ids["nested_delete"]) is None
            assert db.get_session(self.ids["nested_grandchild"]) is None
            branch_keep = db.get_session(self.ids["branch_keep"])
            assert branch_keep is not None
            assert branch_keep["parent_session_id"] is None
        finally:
            db.close()

        list_resp = self.client.get("/api/conversations", params={"limit": 10, "offset": 0})
        assert list_resp.status_code == 200
        remaining_roots = {item["thread_root_id"] for item in list_resp.json()["sessions"]}
        assert self.ids["root"] not in remaining_roots
        assert self.ids["branch_keep"] in remaining_roots
        assert self.ids["telegram"] in remaining_roots
