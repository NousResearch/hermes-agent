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
from hermes_cli import kanban_db_connect as kbc
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
    with kbc.connect() as conn:
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

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kbc.connect() as conn:
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
            "hermes_cli.config.load_config_readonly",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_load_routing_falls_back_to_defaults_when_config_unreadable(kanban_home, monkeypatch):
    """decompose_task promises ok=False on expected failures; a config read that raises (missing
    profile home, HomeInitializationError) must not escape _load_routing as an exception."""
    from hermes_cli import config as config_mod

    def _boom():
        raise FileNotFoundError("profile home is gone")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    routing = decomp._load_routing()
    assert routing.default_assignee == "default" and routing.auto_promote is True


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kbc.connect() as conn:
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
        # call_llm raises RuntimeError when no provider is configured; the
        # decomposer must convert that into a failed outcome, not a crash.
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("No LLM provider configured"),
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    # call_llm's no-provider RuntimeError surfaces via the LLM-error branch.
    assert "LLM error" in outcome.reason


# ── auto_review policy + review-pairing transform ────────────────────────────

def test_review_policy_reads_valid_config():
    cfg = {"kanban": {"auto_review": {"review_roles": ["coder", " "], "reviewer": " reviewer "}}}
    assert decomp._review_policy(cfg) == {"review_roles": ["coder"], "reviewer": "reviewer"}


def test_review_policy_disabled_when_absent_or_malformed():
    assert decomp._review_policy({}) == {}
    assert decomp._review_policy({"kanban": {"auto_review": "on"}}) == {}
    assert decomp._review_policy({"kanban": {"auto_review": {"review_roles": []}}}) == {}
    assert decomp._review_policy({"kanban": {"auto_review": {"review_roles": ["c"]}}}) == {}
    assert decomp._review_policy(None) == {}


def test_pair_review_tasks_appends_reviewer_behind_impl():
    children = [
        {"title": "build api", "body": "b", "assignee": "coder", "parents": []},
        {"title": "write docs", "body": "d", "assignee": "writer", "parents": [0]},
    ]
    policy = {"review_roles": ["coder"], "reviewer": "reviewer"}
    out = decomp._pair_review_tasks(children, policy)
    # original two + one review child for the coder task (index 0)
    assert len(out) == 3
    review = out[2]
    assert review["assignee"] == "reviewer"
    assert review["parents"] == [0]
    assert review["title"].startswith("review: build api")
    # the writer task (not a review role) got no reviewer
    assert all(c["assignee"] != "reviewer" or c["parents"] == [0] for c in out)


def test_pair_review_tasks_noop_without_policy_or_reviewer_role():
    children = [{"title": "x", "body": "y", "assignee": "coder", "parents": []}]
    assert decomp._pair_review_tasks(children, {}) == children
    assert decomp._pair_review_tasks(children, None) == children
    # a child already assigned to the reviewer is not re-reviewed
    rev = [{"title": "r", "body": "y", "assignee": "reviewer", "parents": []}]
    assert decomp._pair_review_tasks(rev, {"review_roles": ["reviewer"], "reviewer": "reviewer"}) == rev


def test_decompose_wires_auto_review_end_to_end(kanban_home):
    """With auto_review configured, decompose_task appends a reviewer child
    gated behind the impl child (the wiring, not just the transform)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "one impl task",
        "tasks": [
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": []},
        ],
    })
    cfg = {"kanban": {"auto_review": {"review_roles": ["engineer"], "reviewer": "reviewer"}}}

    patches = _patch_list_profiles(["orchestrator", "engineer", "reviewer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), \
                patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    # 1 impl child + 1 auto-paired review child
    assert len(outcome.child_ids) == 2
    with kb.connect() as conn:
        kids = [kb.get_task(conn, cid) for cid in outcome.child_ids]
    assignees = sorted(k.assignee for k in kids)
    assert assignees == ["engineer", "reviewer"]
    review = next(k for k in kids if k.assignee == "reviewer")
    assert review.title.startswith("review: build")
