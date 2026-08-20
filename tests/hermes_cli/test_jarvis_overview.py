from datetime import date
import os

import pytest
from fastapi.testclient import TestClient

from hermes_cli import jarvis_dashboard as jd
from hermes_cli import kanban_db


def _write_status_doc(path, phase="Phase one"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Status

## Current phase

{phase}

## Active priorities

1. Priority A
2. Priority B

## Known blockers / unknowns

- Blocker A
""",
        encoding="utf-8",
    )


def test_build_jarvis_overview_uses_explicit_boards(monkeypatch, tmp_path):
    kanban_db.init_db(board="jarvis-dashboard")
    conn = kanban_db.connect(board="jarvis-dashboard")
    try:
        task_id = kanban_db.create_task(
            conn,
            title="Owner approval needed",
            assignee="default",
            priority=7,
            created_by="test",
        )
        kanban_db.block_task(conn, task_id, reason="review-required: verify cockpit api_key remains hidden", kind="needs_input")
    finally:
        conn.close()

    # A worker may have HERMES_KANBAN_DB pinned to its own task board. The
    # overview must still read the explicit jarvis-dashboard board.
    wrong_db = tmp_path / "wrong-kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(wrong_db))

    club_status = tmp_path / "club-status.md"
    club_charter = tmp_path / "club-charter.md"
    cast_status = tmp_path / "cast-status.md"
    cast_charter = tmp_path / "cast-charter.md"
    _write_status_doc(club_status, phase="Club phase")
    _write_status_doc(cast_status, phase="Cast phase")
    club_charter.write_text("# Club charter\n", encoding="utf-8")
    cast_charter.write_text("# Cast charter\n", encoding="utf-8")
    monkeypatch.setitem(jd.PRODUCT_SOURCES["clubhub"], "status_path", str(club_status))
    monkeypatch.setitem(jd.PRODUCT_SOURCES["clubhub"], "charter_path", str(club_charter))
    monkeypatch.setitem(jd.PRODUCT_SOURCES["cast-and-tag"], "status_path", str(cast_status))
    monkeypatch.setitem(jd.PRODUCT_SOURCES["cast-and-tag"], "charter_path", str(cast_charter))

    overview = jd.build_jarvis_overview(
        {
            "overall": "ok",
            "gateway_running": True,
            "gateway_state": "running",
            "active_agents": 1,
            "active_sessions": 2,
            "auth_required": True,
            "profiles": ["default"],
            "components": {
                "gateway": {"status": "ok", "state": "running"},
                "dashboard": {"status": "ok"},
                "storage": {"status": "ok"},
                "platforms": {"status": "ok", "configured": 1, "connected": 1},
            },
        },
        {"psutil": False},
        [],
    )

    assert overview["todos"][0]["id"] == task_id
    assert overview["todos"][0]["board"] == "jarvis-dashboard"
    assert overview["todos"][0]["attention_action"] == "Answer needed"
    assert overview["todos"][0]["attention_since"] is not None
    assert overview["todos"][0]["task_href"].endswith(f"board=jarvis-dashboard&task={task_id}")
    assert "api_key" not in overview["todos"][0]["attention_reason"].lower()
    assert "redacted" in overview["todos"][0]["attention_reason"].lower()
    assert overview["agent_status"]["profiles"][0]["blocked_count"] == 1
    assert overview["agent_status"]["profiles"][0]["needs_attention"] is True
    assert overview["agent_status"]["profiles"][0]["blocked_tasks"][0]["id"] == task_id
    assert overview["products"][0]["phase"] == "Club phase"
    assert overview["products"][0]["last_updated"] is None
    assert overview["products"][0]["next_actions"] == []
    assert overview["products"][0]["owner_action"]["kind"] == "monitor"
    assert overview["products"][0]["primary_cta"]["label"] == "Open source doc"
    assert overview["products"][0]["freshness"]["status"] == "unavailable"
    assert overview["products"][0]["blocker_summary"]["total"] >= 0
    assert overview["products"][0]["approval_summary"]["total"] >= 0
    assert overview["products"][0]["safety_notes"] == []
    assert overview["memory_vault"]["obsidian"]["label"] == "Obsidian Memory"
    assert os.environ["HERMES_KANBAN_DB"] == str(wrong_db)
    assert not wrong_db.exists()


def test_memory_vault_uses_configured_obsidian_path(monkeypatch, tmp_path):
    vault = tmp_path / "ObsidianMemory"
    vault.mkdir()
    (vault / "00 Command Center").mkdir()
    (vault / "00 Command Center" / "Owner Cockpit.md").write_text("# Owner Cockpit\n", encoding="utf-8")
    (vault / "Decisions").mkdir()
    (vault / "Decisions" / "Decision Log.md").write_text("# Decision Log\n", encoding="utf-8")
    (vault / "Products" / "ClubHub").mkdir(parents=True)
    (vault / "Products" / "ClubHub" / "Research.md").write_text("# ClubHub Research\n", encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    memory_vault, source = jd._memory_vault_status()

    assert memory_vault["obsidian"]["configured"] is True
    assert memory_vault["obsidian"]["status"] == "available"
    assert memory_vault["obsidian"]["path"] == str(vault)
    assert memory_vault["obsidian"]["source"] == "OBSIDIAN_VAULT_PATH"
    assert memory_vault["obsidian"]["note_count"] == 3
    assert memory_vault["obsidian"]["decision_count"] == 1
    assert memory_vault["obsidian"]["product_note_count"] == 1
    assert memory_vault["obsidian"]["recent_notes"][0]["title"]
    assert memory_vault["obsidian"]["quick_links"][0]["label"] == "Owner Cockpit"
    assert source["status"] == "ok"


def test_memory_vault_setup_needed_when_missing(monkeypatch, tmp_path):
    missing = tmp_path / "missing-vault"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(missing))
    monkeypatch.setattr(jd, "DEFAULT_OBSIDIAN_VAULT_PATH", str(tmp_path / "default-missing"))
    monkeypatch.setattr(jd, "FALLBACK_OBSIDIAN_VAULT_PATH", str(tmp_path / "fallback-missing"))

    memory_vault, source = jd._memory_vault_status()

    assert memory_vault["obsidian"]["configured"] is False
    assert memory_vault["obsidian"]["status"] == "setup_needed"
    assert memory_vault["obsidian"]["path"] == str(missing)
    assert source["status"] == "unavailable"


def test_product_freshness_statuses_are_bounded_and_source_aware():
    ok_source = {"status": "ok", "path": "/tmp/status.md"}
    ok_charter = {"status": "ok", "path": "/tmp/charter.md"}
    board = {"available": True, "board": "clubhub", "source": {"status": "ok"}}

    assert jd._product_freshness("2026-07-23", ok_source, ok_charter, board, as_of=date(2026, 7, 26))["status"] == "fresh"
    assert jd._product_freshness("2026-07-20", ok_source, ok_charter, board, as_of=date(2026, 7, 26))["status"] == "aging"
    stale = jd._product_freshness("2026-07-18", ok_source, ok_charter, board, as_of=date(2026, 7, 26))
    assert stale["status"] == "stale"
    assert stale["age_days"] == 8
    assert "verify before acting" in stale["message"]
    assert jd._product_freshness(None, ok_source, ok_charter, board, as_of=date(2026, 7, 26))["status"] == "unknown"

    unavailable = jd._product_freshness("2026-07-26", {"status": "unavailable", "error": "file not found"}, ok_charter, board, as_of=date(2026, 7, 26))
    assert unavailable["status"] == "unavailable"
    assert unavailable["sources"][0]["message"] == "file not found"


def test_product_owner_action_priority_and_status_doc_fallback():
    board = {
        "board": "clubhub",
        "blocked_tasks": [
            {"id": "cap", "title": "Needs GitHub access", "status": "blocked", "block_kind": "capability", "priority": 9, "created_at": 10, "attention_since": 10, "attention_action": "Access needed", "task_href": "/cap"},
            {"id": "input", "title": "Approve PR", "status": "blocked", "block_kind": "needs_input", "priority": 1, "created_at": 20, "attention_since": 20, "attention_action": "Answer needed", "attention_reason": "review-required: approve without api_key leak", "task_href": "/input"},
        ],
        "review_tasks": [
            {"id": "review", "title": "Review copy", "status": "review", "block_kind": None, "priority": 10, "created_at": 5, "attention_since": 5, "attention_action": "Review changes", "task_href": "/review"},
            {"id": "ready", "title": "QA ready", "status": "ready", "block_kind": None, "priority": 10, "created_at": 1, "attention_since": 1, "attention_action": "Ready for next agent", "task_href": "/ready"},
        ],
    }

    action = jd._owner_action_from_product(board, ["Execute safe next work"])
    assert action["kind"] == "approval"
    assert action["label"] == "Answer needed"
    assert action["task_id"] == "input"
    assert "api_key" not in action["reason"].lower()
    assert "redacted" in action["reason"].lower()

    fallback = jd._owner_action_from_product({"board": "cast-and-tag", "blocked_tasks": [], "review_tasks": []}, ["Verify stale status doc"])
    assert fallback == {
        "kind": "next_work",
        "label": "Next safe work",
        "task_id": None,
        "title": "Verify stale status doc",
        "reason": None,
        "age_label": None,
        "href": None,
        "source": "status_doc",
    }


def test_product_primary_cta_labels_for_approval_source_and_fallback_states():
    approval = jd._product_primary_cta(
        {"kind": "approval", "label": "Owner decision", "href": "/plugins/kanban?board=clubhub&task=t_1"},
        {"status": "fresh"},
        {"board": "clubhub", "counts": {"blocked": 1}},
        "/tmp/status.md",
    )
    assert approval == {"label": "Open approval task", "href": "/plugins/kanban?board=clubhub&task=t_1", "kind": "approval_task"}

    source_doc = jd._product_primary_cta(
        {"kind": "monitor", "label": "Monitor", "href": "/plugins/kanban?board=cast-and-tag"},
        {"status": "stale"},
        {"board": "cast-and-tag", "counts": {}},
        "/tmp/cast-status.md",
    )
    assert source_doc == {"label": "Open source doc", "href": "/tmp/cast-status.md", "kind": "source_doc"}

    blocked_fallback = jd._product_primary_cta(
        {"kind": "access", "label": "Access needed", "href": "/plugins/kanban?board=clubhub&task=t_2"},
        {"status": "fresh"},
        {"board": "clubhub", "counts": {"blocked": 3}},
        "/tmp/status.md",
    )
    assert blocked_fallback == {"label": "Open blocked board", "href": "/plugins/kanban?board=clubhub", "kind": "blocked_board"}

    board_fallback = jd._product_primary_cta(
        {"kind": "monitor", "label": "Monitor", "href": "/plugins/kanban?board=clubhub"},
        {"status": "fresh"},
        {"board": "clubhub", "counts": {}},
        "/tmp/status.md",
    )
    assert board_fallback == {"label": "Open board", "href": "/plugins/kanban?board=clubhub", "kind": "board"}


def test_product_summaries_split_blockers_from_approvals():
    board = {
        "counts": {"blocked": 4, "review": 2, "ready": 1},
        "blocked_kind_counts": {"needs_input": 2, "capability": 1, "transient": 1},
        "review_status_counts": {"review": 2, "ready": 1},
        "blocked_tasks": [{"id": "b1"}, {"id": "b2"}, {"id": "b3"}],
        "review_tasks": [{"id": "r1"}, {"id": "r2"}, {"id": "q1"}],
    }

    blockers, approvals = jd._product_summaries(board, "Approval gated")
    assert blockers == {
        "total": 4,
        "needs_input": 2,
        "capability": 1,
        "transient": 1,
        "unknown": 0,
        "examples": [{"id": "b1"}, {"id": "b2"}],
    }
    assert approvals["total"] == 3
    assert approvals["review"] == 2
    assert approvals["ready"] == 1
    assert approvals["examples"] == [{"id": "r1"}, {"id": "r2"}]
    assert approvals["approval_note"] == "Approval gated"


@pytest.mark.parametrize("bad_key", ["token", "api_key", "password", "client_secret"])
def test_jarvis_overview_rejects_secret_bearing_keys(bad_key):
    with pytest.raises(ValueError):
        jd.assert_no_secret_keys({"safe": {bad_key: "redacted-or-not"}})


def test_assert_no_secret_keys_rejects_nested_secret_key():
    with pytest.raises(ValueError):
        jd.assert_no_secret_keys({"safe": [{"api_key": "nope"}]})


def test_sanitize_error_redacts_secret_words_case_insensitively():
    text = jd._sanitize_error("Api_Key=abc123 failed; Client_Secret: shh; Bearer live-token-123")

    assert "Api_Key" not in text
    assert "Client_Secret" not in text
    assert "abc123" not in text
    assert "shh" not in text
    assert "live-token-123" not in text
    assert text.count("redacted") >= 3


def test_jarvis_overview_endpoint_returns_safe_payload(monkeypatch):
    from hermes_cli import web_server

    async def fake_status(profile=None):
        return {
            "overall": "ok",
            "gateway_running": True,
            "gateway_state": "running",
            "active_agents": 0,
            "active_sessions": 1,
            "auth_required": True,
            "profiles": ["default"],
            "components": {
                "gateway": {"status": "ok", "state": "running"},
                "dashboard": {"status": "ok"},
                "storage": {"status": "ok"},
                "platforms": {"status": "ok", "configured": 1, "connected": 1},
            },
        }

    async def fake_system():
        return {"psutil": False}

    async def fake_cron(profile="all"):
        return []

    def fake_overview(status, system_stats, cron_jobs):
        return {
            "generated_at": "2026-07-23T00:00:00Z",
            "refresh_after_seconds": 15,
            "agent_status": {
                "overall": "ok",
                "gateway_state": "running",
                "active_agents": 0,
                "active_sessions": 1,
                "auth_required": True,
                "connected_platforms": 1,
                "configured_platforms": 1,
                "profiles": [],
                "components": {},
            },
            "todos": [],
            "products": [],
            "memory_vault": {
                "obsidian": {
                    "configured": False,
                    "status": "setup_needed",
                    "label": "Obsidian Memory",
                    "path": None,
                    "source": "not_configured",
                    "href": "/files",
                    "message": "Set OBSIDIAN_VAULT_PATH to enable vault browsing.",
                    "note_count": 0,
                    "decision_count": 0,
                    "product_note_count": 0,
                    "recent_notes": [],
                    "quick_links": [],
                }
            },
            "service_health": {
                "overall": "ok",
                "gateway": {"status": "ok"},
                "dashboard": {"status": "ok"},
                "storage": {"status": "ok"},
                "platforms": {"status": "ok"},
                "system": {"psutil": False},
                "cron": {"available": True, "total": 0, "enabled": 0, "paused": 0, "recent_failures": 0, "local_only": 0},
            },
            "sources": [],
        }

    monkeypatch.setattr(web_server, "get_status", fake_status)
    monkeypatch.setattr(web_server, "get_system_stats", fake_system)
    monkeypatch.setattr(web_server, "list_cron_jobs", fake_cron)
    monkeypatch.setattr(jd, "build_jarvis_overview", fake_overview)

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    resp = client.get("/api/jarvis/overview")

    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_status"]["auth_required"] is True
    assert "token" not in str(data).lower()
