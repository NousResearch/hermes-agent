from pathlib import Path

from agent.prompt_outbox import PromptDraftStore


def test_prompt_draft_store_creates_and_lists_drafts(tmp_path: Path):
    store = PromptDraftStore(tmp_path / "outbox.db")
    try:
        draft = store.create_prompt(
            title="Pause crons Graphit",
            content="Suspendre les crons Graphit si quota critique.",
            project="Graph'it",
            tags=["graphit", "quota"],
            send_condition={"mode": "quota_positive", "require_confirmation": True},
            priority=80,
        )

        drafts = store.list_prompts()

        assert len(drafts) == 1
        assert drafts[0].id == draft.id
        assert drafts[0].title == "Pause crons Graphit"
        assert drafts[0].status == "draft"
        assert drafts[0].project == "Graph'it"
        assert drafts[0].tags == ["graphit", "quota"]
        assert drafts[0].send_condition["mode"] == "quota_positive"
        assert drafts[0].priority == 80
        assert drafts[0].created_at.endswith("+00:00")
    finally:
        store.close()


def test_prompt_draft_store_updates_and_deletes_draft(tmp_path: Path):
    store = PromptDraftStore(tmp_path / "outbox.db")
    try:
        draft = store.create_prompt(title="Initial", content="First prompt")

        updated = store.update_prompt(
            draft.id,
            title="Updated",
            content="Updated prompt",
            status="queued",
            priority=10,
            project=None,
            send_condition={"mode": "quota_above_threshold", "threshold_percent": 25},
        )

        assert updated is not None
        assert updated.title == "Updated"
        assert updated.project is None
        assert updated.status == "queued"
        assert updated.priority == 10
        assert updated.send_condition["threshold_percent"] == 25
        assert updated.updated_at >= draft.updated_at

        assert store.delete_prompt(draft.id) is True
        assert store.get_prompt(draft.id) is None
        assert store.delete_prompt(draft.id) is False
    finally:
        store.close()


def test_prompt_draft_store_rejects_invalid_threshold(tmp_path: Path):
    store = PromptDraftStore(tmp_path / "outbox.db")
    try:
        try:
            store.create_prompt(
                title="Invalid",
                content="Invalid threshold",
                send_condition={"mode": "quota_above_threshold", "threshold_percent": "NaN"},
            )
        except ValueError as exc:
            assert "threshold_percent" in str(exc)
        else:
            raise AssertionError("Expected invalid threshold_percent to be rejected")
    finally:
        store.close()

def test_prompt_draft_store_filters_archived_by_default(tmp_path: Path):
    store = PromptDraftStore(tmp_path / "outbox.db")
    try:
        active = store.create_prompt(title="Active", content="Keep visible")
        archived = store.create_prompt(title="Archived", content="Hide by default")
        store.update_prompt(archived.id, status="archived")

        visible = store.list_prompts()
        all_prompts = store.list_prompts(include_archived=True)

        assert [draft.id for draft in visible] == [active.id]
        assert {draft.id for draft in all_prompts} == {active.id, archived.id}
    finally:
        store.close()
