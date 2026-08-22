"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim retained for call-site compatibility: extra_body plumbing
    # now lives inside call_llm, which _patch_aux_client already mocks.
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_decompose_skips_duplicate_child_and_links_existing_card(kanban_home):
    # Simulate #88656: a card produced by an earlier, unrelated decomposition
    # is already open on the board with the same observation title the new
    # decomposer is about to propose as a child.
    with kb.connect() as conn:
        existing_id = kb.create_task(conn, title="Verify two consecutive scheduled imports in production")
        tid = kb.create_task(conn, title="production incident", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "Investigate the failure signature", "body": "look it up", "assignee": "researcher", "parents": []},
            {
                "title": "verify two consecutive scheduled imports in production",
                "body": "watch the next two runs",
                "assignee": "engineer",
                "parents": [0],
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    # Only the non-duplicate child was created — no twin of the existing card.
    assert outcome.child_ids and len(outcome.child_ids) == 1

    with kb.connect() as conn:
        created_titles = {kb.get_task(conn, cid).title for cid in outcome.child_ids}
        assert created_titles == {"Investigate the failure signature"}
        # The existing card is now linked under the new root instead of duplicated.
        assert tid in kb.parent_ids(conn, existing_id)


def test_decompose_keeps_duplicate_with_dependent_as_twin(kanban_home):
    # #88656 follow-up: a proposed child that duplicates an existing open
    # card must NOT be deduped away if a sibling declares a dependency on
    # it — decompose_triage_task's children schema can only express
    # "depend on index N in this list", so a dropped duplicate has no way
    # to redirect that edge onto the existing card without data loss.
    # Refusing to dedup it (creating a twin, as before this PR) keeps the
    # sibling's dependency intact instead of silently promoting it to
    # `ready` ahead of the outstanding work it was supposed to wait on.
    with kb.connect() as conn:
        existing_id = kb.create_task(conn, title="Verify two consecutive scheduled imports in production")
        tid = kb.create_task(conn, title="production incident", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {
                "title": "verify two consecutive scheduled imports in production",
                "body": "watch the next two runs",
                "assignee": "engineer",
                "parents": [],
            },
            {
                "title": "Write followup report",
                "body": "summarize findings",
                "assignee": "researcher",
                "parents": [0],
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    # Both children were created — the duplicate became a twin instead of
    # being dropped, since "Write followup report" depends on it.
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        titles_by_id = {cid: kb.get_task(conn, cid).title for cid in outcome.child_ids}
        report_id = next(cid for cid, t in titles_by_id.items() if t == "Write followup report")
        twin_id = next(cid for cid, t in titles_by_id.items() if t != "Write followup report")
        # The dependency on the sibling survives — pointed at the twin,
        # not silently dropped.
        assert twin_id in kb.parent_ids(conn, report_id)
        assert kb.get_task(conn, report_id).status == "todo"
        # The pre-existing card is untouched: no twin duplication of it was
        # attempted, and it wasn't linked under the new root either, since
        # it was never in the dedup set.
        assert kb.parent_ids(conn, existing_id) == []


def test_decompose_dedup_picks_oldest_card_on_title_collision(kanban_home):
    # Reviewer nit on #88656: if several open cards share a normalized
    # title, the dedup must pick the same one every time rather than
    # whichever the board query happened to return last.
    with kb.connect() as conn:
        older_id = kb.create_task(conn, title="Verify two consecutive scheduled imports in production")
        conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (1000, older_id))
        newer_id = kb.create_task(conn, title="verify two consecutive scheduled imports in production")
        conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (2000, newer_id))
        tid = kb.create_task(conn, title="production incident", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "Investigate the failure signature", "body": "look it up", "assignee": "researcher", "parents": []},
            {
                "title": "verify two consecutive scheduled imports in production",
                "body": "watch the next two runs",
                "assignee": "engineer",
                "parents": [],
            },
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 1

    with kb.connect() as conn:
        assert tid in kb.parent_ids(conn, older_id)
        assert kb.parent_ids(conn, newer_id) == []


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not in triage" in outcome.reason


