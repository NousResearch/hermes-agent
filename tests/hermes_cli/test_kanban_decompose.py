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


# ---------------------------------------------------------------------------
# Parent comments are decisions already taken (#126700)
# ---------------------------------------------------------------------------

def test_decompose_prompt_carries_parent_comments(kanban_home):
    """The decompose prompt includes the parent's comment thread with the
    decisions-already-taken rule; a scope ruling made after the body was
    written reaches the decomposer."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="triage card", body="target is 3 files", triage=True)
        kb.add_comment(conn, tid, author="maintainer", body="Ruling: target is widened to 6 files")

    captured = {}

    def _capture(*args, **kwargs):
        messages = kwargs.get("messages") or []
        for message in messages:
            if message.get("role") == "system":
                captured["system"] = message.get("content") or ""
            elif message.get("role") == "user":
                captured["user"] = message.get("content") or ""
        return _fake_aux_response(jsonlib.dumps({
            "fanout": True, "rationale": "split",
            "tasks": [{"title": "work", "body": "do it", "assignee": None, "parents": []}],
        }))

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=_capture), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    user_prompt = captured.get("user") or ""
    assert "Ruling: target is widened to 6 files" in user_prompt
    assert "maintainer" in user_prompt  # author rides verbatim
    assert "decisions already taken" in (captured.get("system") or "")


def test_decompose_children_carry_the_ruling_verbatim(kanban_home):
    """Every child body carries the same decision thread verbatim — a model that
    ignores the prompt cannot lose a ruling made after the body was written."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="triage card", body="target is 3 files", triage=True)
        kb.add_comment(conn, tid, author="maintainer", body="Ruling: target is widened to 6 files")

    llm_payload = jsonlib.dumps({
        "fanout": True, "rationale": "split",
        "tasks": [
            {"title": "a", "body": "stale scope spec", "assignee": None, "parents": []},
            {"title": "b", "body": "another stale spec", "assignee": None, "parents": []},
        ],
    })

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok and outcome.child_ids
    with kbc.connect() as conn:
        children = [kb.get_task(conn, cid) for cid in outcome.child_ids]
    for child in children:
        assert "Ruling: target is widened to 6 files" in (child.body or ""), child.body
        assert "maintainer" in (child.body or "")


def test_comment_block_clips_largest_first(kanban_home):
    """Largest-first with a floor: 'newest wins' dropped the ruling because the
    discussion after a decision is newer than the decision."""
    big_ruling = "Ruling: " + ("detail " * 200)  # the largest, and the OLDEST
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="card", triage=True)
        kb.add_comment(conn, tid, author="maintainer", body=big_ruling)
        kb.add_comment(conn, tid, author="chatter", body="btw nice card")  # newest, tiny

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
    block = decomp._decision_comments_block(comments, decomp._DECISION_COMMENTS_MAX_BYTES)

    assert "Ruling:" in block  # the largest decision rides, not the newest chatter
    assert "maintainer" in block


def test_child_body_block_stays_under_worker_cap(kanban_home):
    """The appended block is sized against build_worker_context's 8 KiB task.body
    cap — anything past the cap is invisible to the worker, so a body near the cap
    shrinks (or drops) the block instead of pushing it past the truncation point."""
    near_cap_body = "spec " * 1600  # ~8 KiB minus a little
    block = decomp._append_comments_to_body(near_cap_body, "Ruling: " + ("detail " * 2000))
    assert len(block.encode("utf-8")) <= decomp._WORKER_BODY_CAP_BYTES
    assert "Ruling:" in block  # the largest decision still rides, clipped to the room left

    over_cap_body = "spec " * 3000  # already past the cap: no room at all
    dropped = decomp._append_comments_to_body(over_cap_body, "Ruling: keep this visible")
    assert "Ruling:" not in dropped  # dropped rather than appended invisible

    small_body = "short spec"
    ok_block = decomp._append_comments_to_body(small_body, "Ruling: keep this visible")
    assert "Ruling: keep this visible" in ok_block
    assert len(ok_block.encode("utf-8")) <= decomp._WORKER_BODY_CAP_BYTES


def test_decompose_fanout_children_inherit_root_assignee_when_unrouted(kanban_home):
    """Unrouted children fall back to the ROOT task's assignee, not
    the decomposer's active profile (#114294). The active profile here is ``private``
    (an incognito profile with no credentials), so the old fallback spawned
    workers that deadlocked on capability blockers."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="zdr", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "made_up", "parents": []},
            {"title": "build", "body": "code it", "assignee": None, "parents": [0]},
        ],
    })

    # get_active_profile_name() is mocked to names[0] = "private" — the
    # global default chain would resolve there without kanban.default_assignee.
    patches = _patch_list_profiles(["private", "zdr"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0.assignee == "zdr"
    assert c1.assignee == "zdr"
    # Same class for the root: no ``orchestrator_profile`` must not hand the
    # orchestration card to the dispatcher's own (here: incognito) profile.
    assert root.assignee == "zdr"


def test_decompose_explicit_default_assignee_wins_over_root_assignee(kanban_home):
    """An explicitly configured ``kanban.default_assignee`` stays
    authoritative for unroutable children; the root task's assignee only
    fills in when no explicit default is set (explicit config → card
    assignee → active profile)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="engineer", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "made_up", "parents": []},
            {"title": "build", "body": "code it", "assignee": None, "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["engineer", "docs", "private"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"kanban": {"default_assignee": "docs"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert c0.assignee == "docs"
    assert c1.assignee == "docs"


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


