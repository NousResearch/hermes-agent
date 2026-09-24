"""Kanban workflow definition: shape, derived copies, and kernel-event edges.

``DEFAULT_WORKFLOW`` is data the kernel does not consult yet (Phase 0 of #54818).
These tests pin it to what the kernel actually does, so a later phase that switches
the kernel over to reading the workflow cannot change behavior unnoticed, and a
kernel change that forgets the workflow fails here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_output
from hermes_cli import kanban_workflow as kw
from tools import kanban_tools_schemas


W = kw.DEFAULT_WORKFLOW


# --- shape + single source of truth ------------------------------------------------------


def test_default_workflow_is_valid_and_excludes_archived():
    W.validate()
    assert kw.ARCHIVED not in W.keys()


def test_status_copies_derive_from_the_workflow():
    assert kb.VALID_STATUSES == set(W.keys()) | {kw.ARCHIVED}
    # CLI: every listable status has a glyph (was missing triage + review).
    assert set(kanban_output._STATUS_ICONS) == kb.VALID_STATUSES


def test_agent_tool_status_enum_covers_every_status():
    # Was missing scheduled + review, so an agent could not filter on them.
    schema = next(s for s in _all_tool_schemas() if s["name"] == "kanban_list")
    enum = schema["parameters"]["properties"]["status"]["enum"]
    assert set(enum) == kb.VALID_STATUSES


def _all_tool_schemas():
    for value in vars(kanban_tools_schemas).values():
        if isinstance(value, dict) and "name" in value and "parameters" in value:
            yield value


def test_every_trait_and_event_is_used_by_the_default():
    # A trait or event no default column uses has no behavior to pin — it would be dead data.
    for trait in kw.TRAITS:
        assert W.keys_with(trait), trait
    assert set(W.defaults) == kw.EVENTS
    with pytest.raises(ValueError):
        W.keys_with("nope")


def test_on_event_prefers_column_edge_over_default():
    custom = replace(
        W,
        columns=tuple(replace(c, on=MappingProxyType({kw.EV_COMPLETE: "review"})) if c.key == "ready" else c
                      for c in W.columns),
    )
    custom.validate()
    assert custom.on_event("ready", kw.EV_COMPLETE) == "review"
    assert custom.on_event("todo", kw.EV_COMPLETE) == "done"
    with pytest.raises(ValueError):
        custom.on_event("ready", "nope")


def test_archive_is_always_a_manual_move():
    assert all(W.can_move(k, kw.ARCHIVED) for k in W.keys())
    assert not any(W.can_move(k, "running") for k in W.keys())


@pytest.mark.parametrize("mutate, message", [
    (lambda w: replace(w, columns=w.columns + (w.columns[0],)), "duplicate"),
    (lambda w: replace(w, columns=w.columns + (kw.Column(key=kw.ARCHIVED, label="A"),)), "reserved"),
    (lambda w: replace(w, defaults=MappingProxyType({k: v for k, v in w.defaults.items()
                                                    if k != kw.EV_BLOCK})), "no default"),
    (lambda w: replace(w, defaults=MappingProxyType({**w.defaults, kw.EV_BLOCK: "gone"})), "unknown column"),
    (lambda w: replace(w, manual=MappingProxyType({"ready": frozenset({"gone"})})), "manual"),
    (lambda w: replace(w, columns=tuple(replace(c, traits=frozenset()) if c.key == "done" else c
                                        for c in w.columns)), "terminal"),
    (lambda w: replace(w, columns=tuple(replace(c, traits=frozenset({"bogus"})) if c.key == "done" else c
                                        for c in w.columns)), "unknown traits"),
])
def test_validate_rejects_inconsistent_workflows(mutate, message):
    with pytest.raises(ValueError, match=message):
        mutate(W).validate()


# --- kernel events land where the workflow says --------------------------------------------


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _status(conn, task_id: str) -> str:
    return kb.get_task(conn, task_id).status


def test_edge_claim(conn):
    tid = kb.create_task(conn, title="t", assignee="builder")
    assert kb.claim_task(conn, tid) is not None
    assert _status(conn, tid) == W.on_event("ready", kw.EV_CLAIM)


def test_edge_complete_block_schedule(conn):
    for event, act in (
        (kw.EV_COMPLETE, lambda tid: kb.complete_task(conn, tid, summary="s")),
        (kw.EV_BLOCK, lambda tid: kb.block_task(conn, tid, reason="r")),
        (kw.EV_SCHEDULE, lambda tid: kb.schedule_task(conn, tid, reason="r")),
    ):
        tid = kb.create_task(conn, title=event, assignee="builder")
        assert kb.claim_task(conn, tid) is not None
        assert act(tid)
        assert _status(conn, tid) == W.on_event("running", event), event


def test_edge_review_and_changes(conn):
    tid = kb.create_task(conn, title="t", assignee="builder")
    assert kb.claim_task(conn, tid) is not None
    assert kb.request_review(conn, tid, summary="s", reviewer="reviewer")
    assert _status(conn, tid) == W.on_event("running", kw.EV_REVIEW)
    review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
    assert review is not None
    ok, _ = kb.request_changes(conn, tid, reason="fix it", expected_run_id=review.current_run_id)
    assert ok
    assert _status(conn, tid) == W.on_event("running", kw.EV_CHANGES)


def test_edge_parents_done(conn):
    parent = kb.create_task(conn, title="p", assignee="builder")
    child = kb.create_task(conn, title="c", assignee="builder", parents=[parent])
    waiting = _status(conn, child)
    assert waiting in W.keys_with(kw.WAIT_PARENTS)
    assert kb.complete_task(conn, parent, summary="s")
    kb.recompute_ready(conn)
    assert _status(conn, child) == W.on_event(waiting, kw.EV_PARENTS_DONE)
