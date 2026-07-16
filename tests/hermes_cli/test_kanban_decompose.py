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
    client = _mock_client_returning(content)
    return patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(client, model),
    )


def _patch_extra_body():
    return patch(
        "agent.auxiliary_client.get_auxiliary_extra_body",
        return_value={},
    )


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


def test_decompose_fanout_false_assigns_default_when_unassigned(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="just one thing", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "**Goal**\nDo the thing.",
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
    assert outcome.fanout is False
    assert outcome.new_title == "Tightened title"
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    # specify path with no parents -> recompute_ready flips to 'ready'
    assert task.status == "ready"
    assert task.title == "Tightened title"
    assert task.assignee == "fallback"


def test_decompose_fanout_false_preserves_existing_assignee(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="already routed",
            assignee="engineer",
            triage=True,
        )

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Keep existing lane.",
        "assignee": "fallback",
    })

    patches = _patch_list_profiles(["orchestrator", "engineer", "fallback"])
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
    assert task.assignee == "engineer"
    assert task.title == "Tightened title"


def test_decompose_fanout_false_uses_valid_llm_assignee(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to specialist.",
        "assignee": "engineer",
    })

    patches = _patch_list_profiles(["orchestrator", "engineer", "fallback"])
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
    assert task.assignee == "engineer"


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


def test_decompose_unknown_assignee_falls_back_to_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)

    # Roster only has 'orchestrator' and 'fallback'; LLM picks 'made_up'.
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test",
        "tasks": [
            {"title": "do X", "body": "", "assignee": "made_up", "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with patch.dict(
            "os.environ", {}, clear=False,
        ), _patch_aux_client(llm_payload), _patch_extra_body(), \
            patch(
                "hermes_cli.kanban_decompose._load_config",
                return_value={
                    "kanban": {
                        "orchestrator_profile": "orchestrator",
                        "default_assignee": "fallback",
                    }
                },
            ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.child_ids and len(outcome.child_ids) == 1
    with kb.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    # 'made_up' wasn't in roster, so assignee rewritten to 'fallback'
    assert child.assignee == "fallback"


def test_decompose_handles_malformed_llm_json(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client("not json at all, sorry"), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "malformed JSON" in outcome.reason


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


def test_decompose_no_aux_client_configured(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", triage=True)

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(None, ""),
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "no auxiliary client" in outcome.reason


def test_decompose_rejects_brand_mismatch(kanban_home, monkeypatch):
    # Since the board-override fix (explicit/scoped board beats a stale
    # HERMES_KANBAN_DB pin), a cross-board leak is refused at connect time:
    # the foreign task is simply not visible on the active board. The brand
    # guard's specific audit message still covers IN-DB mismatches (e.g.
    # imported/migrated rows), staged here by rewriting the brand column.
    kb.create_board("brand-a")
    kb.create_board("brand-b")
    with kb.connect(board="brand-b") as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)
        conn.execute("UPDATE tasks SET brand = 'brand-a' WHERE id = ?", (tid,))
        conn.commit()

    with kb.scoped_current_board("brand-b"):
        outcome = decomp.decompose_task(tid, author="me")

    assert outcome.ok is False
    assert "does not match active board" in outcome.reason

    # The pre-fix leak scenario (task on another board's DB, env pinned to
    # that DB, different active board) now refuses even earlier.
    with kb.connect(board="brand-a") as conn:
        tid2 = kb.create_task(conn, title="ship another feature", triage=True)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board="brand-a")))
    with kb.scoped_current_board("brand-b"):
        outcome2 = decomp.decompose_task(tid2, author="me")

    assert outcome2.ok is False
    assert outcome2.reason == "unknown task id"


def test_decompose_allows_unknown_brand(kanban_home, monkeypatch):
    # Legacy/pre-migration rows have brand=NULL. The brand-mismatch gate must
    # NOT reject them (a null brand cannot prove a mismatch); they fall through
    # to normal decomposition. Regression guard for the intentional
    # transitional bypass in decompose_task.
    kb.create_board("brand-a")
    kb.create_board("brand-b")
    with kb.connect(board="brand-a") as conn:
        tid = kb.create_task(conn, title="legacy task", triage=True)
        # Simulate a pre-migration row whose brand was never populated.
        conn.execute("UPDATE tasks SET brand = NULL WHERE id = ?", (tid,))
        conn.commit()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board="brand-a")))
    with kb.scoped_current_board("brand-b"):
        outcome = decomp.decompose_task(tid, author="me")

    # It must get PAST the brand gate — i.e. it is not rejected for a brand
    # mismatch. It may still stop later (e.g. no auxiliary client in tests);
    # the point is the gate did not fire on a null brand.
    assert "does not match active board" not in (outcome.reason or "")


# ---------------------------------------------------------------------------
# Acceptance addendum (definition-of-done propagation into child bodies)
# ---------------------------------------------------------------------------

_CHARTER_TEXT = """# REVIEW charter

## 1. Quality bar
- generic quality prose

## 2. Always verify
- re-run the implementer's stated verification steps

## 3. Automatic FAIL
- secrets in any artifact
- unverifiable handoff

## 5. Tone of review
- strictness prose that must NOT reach workers
"""


def _write_default_charter(home: Path, text: str = _CHARTER_TEXT) -> Path:
    path = home / "kanban" / "charters" / "DEFAULT-REVIEW.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


_FANOUT_PAYLOAD = jsonlib.dumps({
    "fanout": True,
    "rationale": "test split",
    "tasks": [
        {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
        {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
    ],
})


def _run_decompose(tid, payload=_FANOUT_PAYLOAD):
    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(payload), _patch_extra_body():
            return decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()


def test_decompose_injects_acceptance_addendum_into_child_bodies(kanban_home):
    _write_default_charter(kanban_home)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)
    outcome = _run_decompose(tid)
    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        bodies = [kb.get_task(conn, cid).body for cid in outcome.child_ids]
    for body in bodies:
        assert decomp._ACCEPTANCE_HEADER in body
        assert "unverifiable handoff" in body
        assert "re-run the implementer" in body
        # Non-acceptance charter sections must not leak into worker bodies.
        assert "strictness prose" not in body
    # The original model-authored spec is preserved ahead of the addendum.
    assert bodies[0].startswith("look it up")


def test_decompose_acceptance_flag_off_leaves_bodies_untouched(kanban_home):
    _write_default_charter(kanban_home)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)
    with patch.object(
        decomp, "_load_config",
        return_value={"kanban": {"decompose_acceptance_addendum": False}},
    ):
        outcome = _run_decompose(tid)
    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        bodies = [kb.get_task(conn, cid).body for cid in outcome.child_ids]
    assert bodies == ["look it up", "code it"]


def test_decompose_acceptance_missing_source_is_noop(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)
    outcome = _run_decompose(tid)
    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        bodies = [kb.get_task(conn, cid).body for cid in outcome.child_ids]
    assert bodies == ["look it up", "code it"]


def test_decompose_acceptance_board_override_wins(kanban_home):
    _write_default_charter(kanban_home)
    board_dir = kb.boards_root() / kb.get_current_board()
    board_dir.mkdir(parents=True, exist_ok=True)
    (board_dir / "ACCEPTANCE.md").write_text(
        "board-specific definition of done", encoding="utf-8",
    )
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)
    outcome = _run_decompose(tid)
    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        body = kb.get_task(conn, outcome.child_ids[0]).body
    assert "board-specific definition of done" in body
    assert "unverifiable handoff" not in body


def test_decompose_fanout_false_injects_addendum(kanban_home):
    _write_default_charter(kanban_home)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="one unit", triage=True)
    payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single",
        "title": "tightened title",
        "body": "tightened spec",
        "assignee": "researcher",
    })
    outcome = _run_decompose(tid, payload)
    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body.startswith("tightened spec")
    assert decomp._ACCEPTANCE_HEADER in task.body


def test_append_addendum_is_idempotent():
    addendum = f"{decomp._ACCEPTANCE_HEADER}\nrules here"
    once = decomp._append_addendum("body", addendum)
    twice = decomp._append_addendum(once, addendum)
    assert once == twice
    assert once.count(decomp._ACCEPTANCE_HEADER) == 1


def test_extract_acceptance_sections_caps_length():
    text = "## 3. Automatic FAIL\n" + "\n".join(f"- rule {i}" for i in range(1000))
    out = decomp._extract_acceptance_sections(text)
    assert len(out) <= decomp._ACCEPTANCE_MAX_CHARS + len("\n[truncated]")
    assert out.endswith("[truncated]")


def test_extract_acceptance_sections_unstructured_file_used_whole():
    out = decomp._extract_acceptance_sections("just a plain definition of done")
    assert out == "just a plain definition of done"


def test_decompose_empty_child_body_stays_empty(kanban_home):
    _write_default_charter(kanban_home)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)
    payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "r",
        "tasks": [
            {"title": "bodyless", "body": "", "assignee": None, "parents": []},
            {"title": "specced", "body": "do it", "assignee": None, "parents": []},
        ],
    })
    outcome = _run_decompose(tid, payload)
    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        bodies = [kb.get_task(conn, cid).body for cid in outcome.child_ids]
    # No spec means no addendum — a pure definition-of-done body would
    # flip "has a body?" checks while describing nothing.
    assert (bodies[0] or "") == ""
    assert decomp._ACCEPTANCE_HEADER in bodies[1]
