"""Tests for durable background task and sidequest persistence."""

from gateway.sidequests import SidequestStore


def test_background_run_lifecycle_and_owner_scoped_lookup(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")

    store.create_background_run(
        bg_id="bg_120000_abcdef",
        prompt="research durable background tasks",
        platform="whatsapp",
        chat_id="chat-1",
        user_id="user-1",
        session_id="bg_120000_abcdef",
    )
    store.mark_running("bg_120000_abcdef")
    store.mark_completed(
        "bg_120000_abcdef",
        summary="Found a design for follow-up handles.",
        artifact_paths=["/tmp/report.md"],
    )

    run = store.get_background_run("bg_120000_abcdef", platform="whatsapp", chat_id="chat-1")
    assert run is not None
    assert run["status"] == "completed"
    assert run["latest_summary"] == "Found a design for follow-up handles."
    assert run["artifact_paths"] == ["/tmp/report.md"]

    assert store.get_background_run("bg_120000_abcdef", platform="whatsapp", chat_id="other") is None


def test_promote_background_run_creates_short_numbered_sidequest(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")
    store.create_background_run(
        bg_id="bg_120001_abcdef",
        prompt="audit the repo",
        platform="telegram",
        chat_id="chat-1",
        user_id="user-1",
        session_id="bg_120001_abcdef",
    )
    store.mark_completed("bg_120001_abcdef", summary="Repo audit summary")

    quest = store.promote_background_run("bg_120001_abcdef", platform="telegram", chat_id="chat-1")

    assert quest["quest_id"].startswith("sq_")
    assert quest["alias"] == 1
    assert quest["title"] == "audit the repo"
    assert quest["latest_summary"] == "Repo audit summary"
    assert store.resolve_quest("1", platform="telegram", chat_id="chat-1")["quest_id"] == quest["quest_id"]


def test_attached_background_completion_updates_sidequest_summary(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")
    quest = store.create_quest(
        title="durable quest",
        platform="whatsapp",
        chat_id="chat-1",
        user_id="user-1",
    )
    store.create_background_run(
        bg_id="bg_120003_abcdef",
        prompt="durable quest",
        platform="whatsapp",
        chat_id="chat-1",
        user_id="user-1",
        session_id="bg_120003_abcdef",
    )
    store.attach_background_to_quest(quest_id=quest["quest_id"], bg_id="bg_120003_abcdef")

    store.mark_completed("bg_120003_abcdef", summary="Quest run finished", artifact_paths=["/tmp/q.md"])

    refreshed = store.resolve_quest("1", platform="whatsapp", chat_id="chat-1")
    assert refreshed["status"] == "waiting"
    assert refreshed["latest_summary"] == "Quest run finished"
    assert refreshed["artifact_paths"] == ["/tmp/q.md"]


def test_followup_is_recorded_for_background_or_quest(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")
    store.create_background_run(
        bg_id="bg_120002_abcdef",
        prompt="first prompt",
        platform="whatsapp",
        chat_id="chat-1",
        user_id="user-1",
        session_id="bg_120002_abcdef",
    )

    followup = store.add_followup(
        target_id="bg_120002_abcdef",
        message="also check the docs",
        platform="whatsapp",
        chat_id="chat-1",
        user_id="user-1",
    )

    assert followup["target_type"] == "background"
    assert followup["status"] == "queued"
    assert followup["message"] == "also check the docs"
    assert store.list_followups("bg_120002_abcdef")[0]["message"] == "also check the docs"


def test_reconcile_incomplete_runs_blocks_linked_sidequests(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")
    quest = store.create_quest(
        title="interrupted quest",
        platform="whatsapp",
        chat_id="scope-1",
    )
    store.create_background_run(
        bg_id="bg_interrupted",
        prompt="long-running work",
        platform="whatsapp",
        chat_id="scope-1",
    )
    store.attach_background_to_quest(
        quest_id=quest["quest_id"],
        bg_id="bg_interrupted",
    )
    store.mark_running("bg_interrupted")

    assert store.reconcile_incomplete_runs() == 1

    run = store.get_background_run(
        "bg_interrupted",
        platform="whatsapp",
        chat_id="scope-1",
    )
    refreshed = store.resolve_quest("1", platform="whatsapp", chat_id="scope-1")
    assert run is not None
    assert refreshed is not None
    assert run["status"] == "failed"
    assert refreshed["status"] == "blocked"
    assert "gateway restarted" in refreshed["latest_summary"]
