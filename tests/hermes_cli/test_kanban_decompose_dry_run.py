"""Deterministic tests for ``kanban_decompose.dry_run_route``.

Covers the three required preview scenarios — single clear owner,
cross-functional fan-out, and Engineering-specific ownership — plus the
no-mutation / no-worker-invocation guarantee. The auxiliary LLM client is
mocked throughout: no network calls, no flakiness.
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


def _patch_aux_client(content: str):
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist, mirroring test_kanban_decompose.py."""
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
        patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value=names[0] if names else "default",
        ),
    ]


def _table_counts(conn):
    """Row counts for every table dry-run must never touch."""
    counts = {}
    for table in ("tasks", "task_links", "task_comments", "task_events"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def _run_dry_run_with_mutation_guards(**kwargs):
    """Call dry_run_route with kb's two mutating helpers spied on, and
    return (result, specify_mock, decompose_mock)."""
    with patch.object(
        kb, "specify_triage_task", wraps=kb.specify_triage_task
    ) as specify_spy, patch.object(
        kb, "decompose_triage_task", wraps=kb.decompose_triage_task
    ) as decompose_spy:
        result = decomp.dry_run_route(**kwargs)
    return result, specify_spy, decompose_spy


def test_dry_run_single_owner_no_mutation(kanban_home):
    """A request that resolves to one clear owning profile: dry-run must
    report the predicted owner + a well-formed envelope, and must not
    touch the DB or invoke either mutating write helper."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="fix the login bug", triage=True)
        before = _table_counts(conn)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single clear owner",
        "title": "Fix login bug",
        "body": "Investigate and patch the login regression.",
        "assignee": "backend_dev",
    })

    patches = _patch_list_profiles(["orchestrator", "backend_dev"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            result, specify_spy, decompose_spy = _run_dry_run_with_mutation_guards(
                task_id=tid,
            )
    finally:
        for p in patches:
            p.stop()

    assert result.ok, result.reason
    assert result.fanout is False
    assert result.predicted_owner == "backend_dev"

    envelope = result.context_envelope
    assert envelope is not None
    assert envelope["assignee"] == "backend_dev"
    assert envelope["title"] == "Fix login bug"
    assert envelope["body"]
    assert "worker_context" in envelope and envelope["worker_context"]
    assert envelope["roster"] is not None
    assert envelope["orchestrator"] == "orchestrator"

    assert result.dependency_graph is None

    specify_spy.assert_not_called()
    decompose_spy.assert_not_called()

    with kb.connect() as conn:
        after = _table_counts(conn)
        task = kb.get_task(conn, tid)
    assert after == before
    assert task.status == "triage"
    assert task.assignee is None


def test_dry_run_cross_functional_fanout_no_mutation(kanban_home):
    """A cross-functional request that would fan out to multiple
    profiles/child tasks: dry-run must return a valid proposed
    dependency graph without creating any card."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="launch new feature", triage=True)
        before = _table_counts(conn)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "needs design, build, and marketing",
        "tasks": [
            {"title": "design the UI", "body": "mock it up", "assignee": "designer", "parents": []},
            {"title": "build the API", "body": "implement endpoints", "assignee": "backend_dev", "parents": []},
            {"title": "write launch copy", "body": "draft announcement", "assignee": "marketing", "parents": [0, 1]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "designer", "backend_dev", "marketing"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            result, specify_spy, decompose_spy = _run_dry_run_with_mutation_guards(
                task_id=tid,
            )
    finally:
        for p in patches:
            p.stop()

    assert result.ok, result.reason
    assert result.fanout is True
    assert result.predicted_owner == "orchestrator"

    graph = result.dependency_graph
    assert graph is not None and len(graph) == 3

    assignees = {node["index"]: node["assignee"] for node in graph}
    assert assignees == {0: "designer", 1: "backend_dev", 2: "marketing"}

    # Every parent index must be a valid, distinct, earlier-or-other
    # in-graph node — i.e. a well-formed dependency graph.
    valid_indices = {node["index"] for node in graph}
    for node in graph:
        for parent_idx in node["parents"]:
            assert parent_idx in valid_indices
            assert parent_idx != node["index"]
    assert graph[2]["parents"] == [0, 1]
    assert graph[0]["parents"] == []
    assert graph[1]["parents"] == []

    envelope = result.context_envelope
    assert envelope is not None
    assert envelope["assignee"] == "orchestrator"
    assert envelope["roster"] is not None
    assert "worker_context" in envelope and envelope["worker_context"]

    specify_spy.assert_not_called()
    decompose_spy.assert_not_called()

    with kb.connect() as conn:
        after = _table_counts(conn)
        task = kb.get_task(conn, tid)
    assert after == before
    assert task.status == "triage"


def test_dry_run_engineering_owner_no_mutation(kanban_home):
    """A request that resolves to Engineering ownership specifically:
    dry-run must report 'engineering' as predicted owner without
    creating a card or invoking a worker."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="patch a SQL injection vulnerability",
            triage=True,
        )
        before = _table_counts(conn)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "security fix belongs to engineering",
        "title": "Patch SQL injection vulnerability",
        "body": "Parameterize the vulnerable query and add a regression test.",
        "assignee": "engineering",
    })

    patches = _patch_list_profiles(["orchestrator", "engineering", "marketing"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            result, specify_spy, decompose_spy = _run_dry_run_with_mutation_guards(
                task_id=tid,
            )
    finally:
        for p in patches:
            p.stop()

    assert result.ok, result.reason
    assert result.fanout is False
    assert result.predicted_owner == "engineering"

    envelope = result.context_envelope
    assert envelope is not None
    assert envelope["assignee"] == "engineering"
    assert envelope["title"] == "Patch SQL injection vulnerability"
    assert "worker_context" in envelope and envelope["worker_context"]

    assert result.dependency_graph is None

    specify_spy.assert_not_called()
    decompose_spy.assert_not_called()

    with kb.connect() as conn:
        after = _table_counts(conn)
        task = kb.get_task(conn, tid)
    assert after == before
    assert task.status == "triage"
    assert task.assignee is None


def test_dry_run_preview_with_no_backing_task_still_returns_full_prediction(kanban_home):
    """dry_run_route also supports an ad-hoc title/body preview with no
    backing task row at all — confirm it still returns owner + envelope
    + (for fanout) a graph, and never creates a row."""
    with kb.connect() as conn:
        before = _table_counts(conn)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit, ad-hoc preview",
        "title": "Draft preview title",
        "body": "Draft preview body.",
        "assignee": "engineering",
    })

    patches = _patch_list_profiles(["orchestrator", "engineering"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload):
            result, specify_spy, decompose_spy = _run_dry_run_with_mutation_guards(
                title="Raw idea needing routing",
                body="Some rough idea.",
            )
    finally:
        for p in patches:
            p.stop()

    assert result.ok, result.reason
    assert result.predicted_owner == "engineering"
    assert result.context_envelope is not None
    # No backing task_id means no worker_context key is attached.
    assert "worker_context" not in result.context_envelope

    specify_spy.assert_not_called()
    decompose_spy.assert_not_called()

    with kb.connect() as conn:
        after = _table_counts(conn)
    assert after == before


def test_dry_run_rejects_ambiguous_input(kanban_home):
    """Passing both task_id and title (or neither) is a usage error, not
    a silent guess — and must not touch the LLM or the DB."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="whatever", triage=True)

    result_neither = decomp.dry_run_route()
    assert result_neither.ok is False

    result_both = decomp.dry_run_route(task_id=tid, title="also set")
    assert result_both.ok is False
