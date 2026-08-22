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
from hermes_cli import kanban_decompose_templates as templates


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


# ---------------------------------------------------------------------------
# Template enforcement (task t_e36247bb): SL paths must emit a separate
# implementer + independent reviewer, never collapse review onto the
# implementer. Coverage: fanout with no reviewer, fanout=false single task,
# reviewer missing (skipped), and non-matching title (no template).
# ---------------------------------------------------------------------------

_SL_TEMPLATES = {
    "kanban": {
        "default_assignee": "academic",
        "orchestrator_profile": "default",
        "decompose_templates": [
            {
                "name": "sl_analysis_editorial",
                "match_title_requires_any": [
                    "suomen liittokunta", "sl", "pohjolan ihme",
                ],
                "match_title_contains": ["analysis", "editorial", "rothbard"],
                "enforce_review_child": True,
                "implementer": "suomen-liittokunta",
                "reviewer": "academic",
                "require_mention": True,
                "credential_isolation": True,
            },
            {
                "name": "sl_web_ship",
                "match_title_requires_any": [
                    "suomen liittokunta", "sl", "pohjolan ihme",
                ],
                "match_title_contains": ["website", "deploy", "ship", "vercel"],
                "enforce_review_child": True,
                "implementer": "websites",
                "reviewer": "academic",
                "require_mention": True,
                "credential_isolation": True,
            },
        ],
    }
}


def test_template_injects_reviewer_into_fanout(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="Suomen Liittokunta analysis on pensions", triage=True,
        )
    # LLM returns only a child under the wrong owner, forgetting the reviewer.
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "research then write",
        "tasks": [
            {"title": "Write analysis", "body": "do it",
             "assignee": "default", "parents": []},
        ],
    })
    patches = _patch_list_profiles(
        ["default", "suomen-liittokunta", "academic", "websites"],
    )
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value=_SL_TEMPLATES,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert len(outcome.child_ids) == 2
    with kb.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    # Implementer and reviewer are DISTINCT profiles.
    assert c0.assignee == "suomen-liittokunta"
    assert c1.assignee == "academic"
    assert c1.title.lower().startswith("review")
    assert "Credential isolation:" in c0.body
    assert "require_mention is enforced" in c1.body
    assert "Credential isolation:" in c1.body
    # Reviewer depends on the implementer; implementer runs first.
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c1.title.lower().startswith("review")


def test_template_upgrades_single_task_to_implementer_reviewer(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="SL analysis: tighten editorial copy", triage=True,
        )
    # LLM returned fanout=false (single unit) -> must be split.
    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened editorial",
        "body": "Write the copy.",
    })
    patches = _patch_list_profiles(
        ["default", "suomen-liittokunta", "academic", "websites"],
    )
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value=_SL_TEMPLATES,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert len(outcome.child_ids) == 2
    with kb.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0.assignee == "suomen-liittokunta"
    assert c1.assignee == "academic"
    assert c1.title.lower().startswith("review")


def test_template_skipped_when_reviewer_profile_missing(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="Suomen Liittokunta analysis on pensions", triage=True,
        )
    # Only implementer installed; reviewer 'academic' is NOT in the roster,
    # so the template must NOT fake a reviewer child.
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "research then write",
        "tasks": [
            {"title": "Write analysis", "body": "do it",
             "assignee": "suomen-liittokunta", "parents": []},
        ],
    })
    patches = _patch_list_profiles(["default", "suomen-liittokunta"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value=_SL_TEMPLATES,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    assert len(outcome.child_ids) == 1
    with kb.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
    assert c0.assignee == "suomen-liittokunta"


def test_template_no_match_for_non_sl_title(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="Deploy the finance website", triage=True,
        )
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "two parts",
        "tasks": [
            {"title": "Write fix", "body": "a", "assignee": "engineer", "parents": []},
            {"title": "Verify", "body": "b", "assignee": "engineer", "parents": [0]},
        ],
    })
    patches = _patch_list_profiles(
        ["default", "engineer", "suomen-liittokunta", "academic", "websites"],
    )
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value=_SL_TEMPLATES,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok, outcome.reason
    # No template matched -> LLM plan untouched, both children to 'engineer'.
    assert len(outcome.child_ids) == 2
    with kb.connect() as conn:
        for cid in outcome.child_ids:
            assert kb.get_task(conn, cid).assignee == "engineer"


def test_template_normalizes_existing_review_child_and_policy(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="SL website deploy",
            body="Acceptance: deployment receipt and smoke test.",
            triage=True,
        )
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "build then review",
        "tasks": [
            {"title": "Deploy website", "body": "ship it",
             "assignee": "websites", "parents": []},
            {"title": "Review deployment", "body": "check smoke evidence",
             "assignee": "websites", "parents": []},
        ],
    })
    patches = _patch_list_profiles(
        ["default", "suomen-liittokunta", "academic", "websites"],
    )
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value=_SL_TEMPLATES,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert len(outcome.child_ids) == 2
    with kb.connect() as conn:
        implementation = kb.get_task(conn, outcome.child_ids[0])
        review = kb.get_task(conn, outcome.child_ids[1])
    assert implementation.assignee == "websites"
    assert review.assignee == "academic"
    assert review.status == "todo"
    assert "check smoke evidence" in (review.body or "")
    assert "Original task acceptance context" in (review.body or "")
    assert "require_mention is enforced" in (review.body or "")
    assert "Credential isolation:" in (review.body or "")


def test_template_adds_implementer_when_llm_returns_only_review(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="SL analysis: review policy memo", triage=True,
        )
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "bad review-only plan",
        "tasks": [
            {"title": "Review memo", "body": "review it",
             "assignee": "academic", "parents": []},
        ],
    })
    patches = _patch_list_profiles(
        ["default", "suomen-liittokunta", "academic", "websites"],
    )
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value=_SL_TEMPLATES,
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        children = [kb.get_task(conn, cid) for cid in outcome.child_ids]
    by_assignee = {child.assignee: child for child in children}
    assert set(by_assignee) == {"suomen-liittokunta", "academic"}
    assert by_assignee["suomen-liittokunta"].status == "ready"
    assert by_assignee["academic"].status == "todo"


def test_template_policy_flags_fail_closed():
    template = dict(_SL_TEMPLATES["kanban"]["decompose_templates"][0])
    template["require_mention"] = False
    task = type("Task", (), {"title": "SL analysis", "body": ""})()
    plan = {"fanout": False, "title": "Write analysis", "body": "do it"}

    result, changed = templates._apply_to_plan(
        task,
        plan,
        template,
        {"suomen-liittokunta", "academic"},
    )

    assert changed is False
    assert result is plan

def test_template_breaks_implementer_reviewer_cycle():
    template = _SL_TEMPLATES["kanban"]["decompose_templates"][0]
    task = type("Task", (), {"title": "SL analysis", "body": ""})()
    plan = {
        "fanout": True,
        "tasks": [
            {
                "title": "Write analysis",
                "body": "implement",
                "assignee": "suomen-liittokunta",
                "parents": [1],
            },
            {
                "title": "Review analysis",
                "body": "review",
                "assignee": "academic",
                "parents": [0],
            },
        ],
    }

    result, changed = templates._apply_to_plan(
        task,
        plan,
        template,
        {"suomen-liittokunta", "academic"},
    )

    assert changed is True
    assert result["tasks"][0]["parents"] == []
    assert result["tasks"][1]["parents"] == [0]


def test_template_routes_every_implementation_child_to_implementer():
    template = _SL_TEMPLATES["kanban"]["decompose_templates"][0]
    task = type("Task", (), {"title": "SL editorial analysis", "body": ""})()
    plan = {
        "fanout": True,
        "tasks": [
            {
                "title": "Research sources",
                "body": "collect evidence",
                "assignee": "academic",
                "parents": [],
            },
            {
                "title": "Write editorial",
                "body": "write the artifact",
                "assignee": "default",
                "parents": [0],
            },
        ],
    }

    result, changed = templates._apply_to_plan(
        task,
        plan,
        template,
        {"default", "suomen-liittokunta", "academic"},
    )

    assert changed is True
    assert [entry["assignee"] for entry in result["tasks"]] == [
        "suomen-liittokunta",
        "suomen-liittokunta",
        "academic",
    ]


def test_template_policy_marker_cannot_suppress_credential_rule():
    template = _SL_TEMPLATES["kanban"]["decompose_templates"][0]
    task = type("Task", (), {"title": "SL analysis", "body": ""})()
    plan = {
        "fanout": False,
        "title": "Write analysis",
        "body": "Credential isolation: disabled for this task.",
    }

    result, changed = templates._apply_to_plan(
        task,
        plan,
        template,
        {"suomen-liittokunta", "academic"},
    )

    assert changed is True
    implementation_body = result["tasks"][0]["body"]
    assert "Credential isolation: disabled for this task." in implementation_body
    assert "never read, reuse, request, or pass another profile's secrets" in (
        implementation_body
    )


def test_template_runtime_error_fails_open(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="SL analysis: preserve original plan", triage=True,
        )
    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "one implementation child",
        "tasks": [
            {
                "title": "Write analysis",
                "body": "do it",
                "assignee": "suomen-liittokunta",
                "parents": [],
            },
        ],
    })
    patches = _patch_list_profiles(
        ["default", "suomen-liittokunta", "academic", "websites"],
    )
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value=_SL_TEMPLATES,
        ), patch(
            "hermes_cli.kanban_decompose_templates._apply_to_plan",
            side_effect=RuntimeError("template failure"),
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert len(outcome.child_ids) == 1
    with kb.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child.assignee == "suomen-liittokunta"
