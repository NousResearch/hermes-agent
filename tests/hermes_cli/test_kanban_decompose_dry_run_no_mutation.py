"""Failure-path proof that dry_run_route can NEVER write to the Kanban DB.

This deliberately tries to *break* the dry-run/live separation added in
commit f4e3e421f (docs/rfcs/yout-plus-dry-run-routing.md):

  * every write-capable ``kanban_db`` function is replaced with a spy that
    raises loudly if invoked, so any accidental write path — now or after
    a future refactor — fails the test immediately instead of silently
    passing;
  * the auxiliary LLM call is driven through several edge/error cases
    (mid-call exception, malformed JSON, empty fan-out list) that are the
    likeliest places a careless refactor would "fall through" into a
    write, since those are exactly the branches the live path turns into
    persisted DB state;
  * a full logical snapshot of every table is taken before and after each
    call and asserted byte-for-byte identical, not just a row count, so
    even an in-place UPDATE with no row-count change would be caught.

If a future change reintroduces a write path in dry-run mode (e.g. someone
"simplifies" by calling ``decompose_task`` internally, or touches
``kb.specify_triage_task`` / ``kb.decompose_triage_task`` from a new
branch), this file fails loudly.
"""

from __future__ import annotations

import inspect
import json as jsonlib
import sqlite3
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


def _patch_list_profiles(names: list[str]):
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


# Every function in kanban_db capable of mutating the DB. Deliberately a
# superset (includes helpers dry-run has no legitimate reason to ever call,
# like archive/delete/reassign) so an unexpected new call site is caught
# regardless of which write helper a future refactor reaches for.
_MUTATING_KB_FUNCS = [
    "create_task",
    "assign_task",
    "link_tasks",
    "unlink_tasks",
    "claim_task",
    "claim_review_task",
    "reclaim_task",
    "reassign_task",
    "complete_task",
    "edit_completed_task_result",
    "block_task",
    "promote_task",
    "unblock_task",
    "reopen_review_task",
    "specify_triage_task",
    "decompose_triage_task",
    "archive_task",
    "delete_archived_task",
    "delete_task",
    "schedule_task",
    "add_comment",
]


def _raising_spy(name: str):
    def _spy(*args, **kwargs):
        raise AssertionError(
            f"dry-run path illegally invoked mutating DB helper kb.{name}() "
            f"— dry-run must never write to the Kanban database"
        )
    return _spy


def _dump_all_tables(home: Path) -> dict:
    """Full logical snapshot of every kanban table: byte-for-byte diffable."""
    candidates = list(home.rglob("*.db"))
    assert candidates, f"no sqlite db found under {home}"
    db_path = candidates[0]
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        snapshot = {}
        for t in tables:
            rows = conn.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall()
            snapshot[t] = [dict(r) for r in rows]
        return snapshot
    finally:
        conn.close()


def _seed_triage_task(title="rough idea", body="do the thing"):
    with kb.connect() as conn:
        return kb.create_task(conn, title=title, body=body, triage=True)


def _mutation_guard(kanban_home):
    """Patch every mutating kb function to raise, and snapshot the DB.

    Call this AFTER any test setup (e.g. seeding the fixture task) is
    done, since setup legitimately uses kb.create_task — only the
    dry_run_route call itself must be write-free.

    Returns (assert_unchanged, patchers). The caller MUST stop the
    patchers (finally block) and call assert_unchanged() after invoking
    dry_run_route, proving both (a) no spy fired [no write call was even
    attempted] and (b) the DB content is byte-identical to before [belt
    and suspenders in case some future write path bypasses these named
    helpers entirely, e.g. a raw ``conn.execute("INSERT ...")``].
    """
    before = _dump_all_tables(kanban_home)
    patchers = [
        patch.object(kb, name, side_effect=_raising_spy(name))
        for name in _MUTATING_KB_FUNCS
    ]
    for p in patchers:
        p.start()

    def assert_unchanged():
        after = _dump_all_tables(kanban_home)
        assert after == before, (
            "Kanban DB content changed across a dry-run call — "
            "dry-run must be a pure read path"
        )

    return assert_unchanged, patchers


def test_dry_run_survives_llm_exception_mid_call_without_mutating(kanban_home):
    """Simulated network/API failure mid-processing must not leave a write."""
    tid = _seed_triage_task()
    assert_unchanged, guard_patches = _mutation_guard(kanban_home)

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=TimeoutError("simulated network failure mid-call"),
        ):
            result = decomp.dry_run_route(task_id=tid)
    finally:
        for p in patches:
            p.stop()
        for p in guard_patches:
            p.stop()

    assert result.ok is False
    assert "LLM error" in result.reason
    assert_unchanged()

    # Task must still be exactly where it started — untouched.
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "triage"


def test_dry_run_survives_malformed_llm_json_without_mutating(kanban_home):
    """Adversarial/garbage model output must not fall through to a write."""
    tid = _seed_triage_task()
    assert_unchanged, guard_patches = _mutation_guard(kanban_home)

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=_fake_aux_response("not json at all, just prose"),
        ):
            result = decomp.dry_run_route(task_id=tid)
    finally:
        for p in patches:
            p.stop()
        for p in guard_patches:
            p.stop()

    assert result.ok is False
    assert "malformed JSON" in result.reason
    assert_unchanged()


def test_dry_run_survives_empty_fanout_list_without_mutating(kanban_home):
    """fanout=true with an empty tasks[] is the exact shape live-path
    would otherwise try to persist via kb.decompose_triage_task — the
    edge case most likely to slip through a careless refactor."""
    tid = _seed_triage_task()
    assert_unchanged, guard_patches = _mutation_guard(kanban_home)

    llm_payload = jsonlib.dumps({"fanout": True, "rationale": "empty", "tasks": []})
    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=_fake_aux_response(llm_payload),
        ):
            result = decomp.dry_run_route(task_id=tid)
    finally:
        for p in patches:
            p.stop()
        for p in guard_patches:
            p.stop()

    assert result.ok is False
    assert "empty tasks list" in result.reason
    assert_unchanged()


def test_dry_run_successful_fanout_prediction_still_does_not_mutate(kanban_home):
    """The success path is the highest-risk one: it produces a fully
    valid child-task graph that *looks* exactly like what
    kb.decompose_triage_task would persist. Prove it never gets there."""
    tid = _seed_triage_task()
    assert_unchanged, guard_patches = _mutation_guard(kanban_home)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "split it",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })
    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=_fake_aux_response(llm_payload),
        ):
            result = decomp.dry_run_route(task_id=tid)
    finally:
        for p in patches:
            p.stop()
        for p in guard_patches:
            p.stop()

    # The prediction succeeds and looks like a real, promotable graph...
    assert result.ok is True
    assert result.fanout is True
    assert result.dependency_graph and len(result.dependency_graph) == 2
    assert result.predicted_owner == "orchestrator"
    # ...yet nothing was written.
    assert_unchanged()
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        all_tasks = kb.list_tasks(conn, tenant=None, limit=1000)
    assert root is not None
    assert root.status == "triage"  # never promoted to todo
    assert len(all_tasks) == 1  # no children ever created


def test_dry_run_ad_hoc_preview_with_no_backing_row_does_not_mutate(kanban_home):
    """title/body preview mode (no task_id) must also never write."""
    assert_unchanged, guard_patches = _mutation_guard(kanban_home)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "do it",
        "assignee": "engineer",
    })
    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=_fake_aux_response(llm_payload),
        ):
            result = decomp.dry_run_route(title="a rough idea", body="details")
    finally:
        for p in patches:
            p.stop()
        for p in guard_patches:
            p.stop()

    assert result.ok is True
    assert result.predicted_owner == "engineer"
    assert_unchanged()
    with kb.connect() as conn:
        all_tasks = kb.list_tasks(conn, tenant=None, limit=1000)
    assert len(all_tasks) == 0  # no row was ever created for the preview


def test_dry_run_route_source_never_calls_mutating_helpers():
    """Static guard: fail the build if dry_run_route's own source is ever
    edited to CALL either mutating helper directly, regardless of whether
    a runtime test happens to exercise that new line. Matches on the call
    form (name + open-paren) so the docstring's prose mention of the two
    names (explaining what it must never call) doesn't false-positive."""
    src = inspect.getsource(decomp.dry_run_route)
    assert "specify_triage_task(" not in src
    assert "decompose_triage_task(" not in src
