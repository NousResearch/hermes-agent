from pathlib import Path

import pytest

from hermes_context_cards import ContextCardStore, handle_context_command


def test_context_new_writes_markdown_and_active_index(tmp_path: Path):
    store = ContextCardStore(tmp_path)

    card = store.create_card("hermes-agent", batch_id="B001-hermes-chat-context-control", session_id="sess-1")

    assert card.project_id == "hermes-agent"
    assert card.batch_id == "B001-hermes-chat-context-control"
    assert card.status == "active"
    card_path = tmp_path / "cards" / "b001-hermes-chat-context-control.md"
    assert card_path.exists()
    assert "project_id: hermes-agent" in card_path.read_text()
    assert "active_card_id: b001-hermes-chat-context-control" in (tmp_path / "_index.md").read_text()


def test_switch_pauses_previous_context_card(tmp_path: Path):
    store = ContextCardStore(tmp_path)
    first = store.create_card("MIM", batch_id="B003")

    second = store.switch_context("hermes-agent", batch_id="B001-hermes-chat-context-control")

    assert second.status == "active"
    reloaded_first = store.get_card(first.id)
    assert reloaded_first is not None
    assert reloaded_first.status == "paused"
    assert reloaded_first.paused_at
    assert store.active_card().id == second.id


def test_done_writes_resume_prompt_and_clears_active(tmp_path: Path):
    store = ContextCardStore(tmp_path)
    card = store.create_card("hermes-agent", batch_id="B001")

    done = store.mark_done(card.id, summary="구현 완료")

    assert done.status == "done"
    assert done.done_at
    assert done.resume_prompt
    assert "Resume Prompt" in done.body
    assert store.active_card() is None
    assert "active_card_id: null" in (tmp_path / "_index.md").read_text()


def test_handoff_generation_includes_spark_constraints(tmp_path: Path):
    store = ContextCardStore(tmp_path)
    card = store.create_card("hermes-agent", batch_id="B001-hermes-chat-context-control")

    handoff = store.generate_handoff(card.id, target="Spark")

    assert "Project: hermes-agent" in handoff
    assert "Batch: B001-hermes-chat-context-control" in handoff
    assert "Worker: Spark" in handoff
    assert "DB/schema" in handoff
    assert "secrets/API/auth" in handoff


def test_path_traversal_batch_id_is_sanitized(tmp_path: Path):
    store = ContextCardStore(tmp_path)

    card = store.create_card("../../evil", batch_id="../../escape")

    assert ".." not in card.id
    assert "/" not in card.id
    assert (tmp_path / "cards" / f"{card.id}.md").resolve().is_relative_to((tmp_path / "cards").resolve())


def test_context_command_new_and_status(tmp_path: Path):
    store = ContextCardStore(tmp_path)

    created = handle_context_command("new hermes-agent B001-hermes-chat-context-control", store=store, session_id="sess-1")
    status = handle_context_command("status", store=store, session_id="sess-1")

    assert "새 Context 고정" in created
    assert "Project: hermes-agent" in status
    assert "Batch: B001-hermes-chat-context-control" in status


def test_context_command_unknown_subcommand_raises_clear_error(tmp_path: Path):
    store = ContextCardStore(tmp_path)

    with pytest.raises(ValueError, match="Unknown context action"):
        handle_context_command("explode", store=store)
