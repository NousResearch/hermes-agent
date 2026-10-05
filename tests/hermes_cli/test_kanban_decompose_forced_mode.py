"""Tests for --fanout / --no-fanout flags on `hermes kanban decompose`.

These tests verify that forced modes override the model's choice and that
the mode is recorded in the audit trail.
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


def _patch_aux_client(content: str):
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


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


def test_forced_fanout_against_single_task_reply(kanban_home):
    """--fanout forces children even when model returns fanout=false."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    # Model returns single-task (fanout=false)
    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Tightened body",
        "assignee": "engineer",
    })

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me", force_fanout=True)
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 1
    assert outcome.mode == "forced_fanout"

    # Verify the child was created with the model's tightened content
    with kbc.connect() as conn:
        child = kb.get_task(conn, outcome.child_ids[0])
    assert child.title == "Tightened title"
    assert child.body == "Tightened body"
    assert child.assignee == "engineer"


def test_forced_tighten_against_fanout_reply(kanban_home):
    """--no-fanout forces single-task when model returns fanout=true — typed failure."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    # Model returns fanout=true
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
            outcome = decomp.decompose_task(tid, author="me", force_fanout=False)
    finally:
        for p in patches:
            p.stop()

    # Forced tighten against model's fanout is a typed failure
    assert outcome.ok is False
    assert "forced tighten but model returned fanout=true" in outcome.reason
    assert outcome.fanout is False
    assert outcome.mode == "forced_tighten"

    # Verify no children were created
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "triage"  # still in triage, not promoted


def test_no_flag_control_model_chooses_fanout(kanban_home):
    """No flag: model chooses fanout=true → children created."""
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
            outcome = decomp.decompose_task(tid, author="me")  # no force_fanout
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2
    assert outcome.mode == "auto"


def test_no_flag_control_model_chooses_tighten(kanban_home):
    """No flag: model chooses fanout=false → single task promoted."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Tightened body",
        "assignee": "engineer",
    })

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")  # no force_fanout
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    assert outcome.new_title == "Tightened title"
    assert outcome.mode == "auto"


def test_mode_recorded_in_outcome(kanban_home):
    """Verify mode field is set correctly for all three cases."""
    with kbc.connect() as conn:
        tid1 = kb.create_task(conn, title="task 1", triage=True)
        tid2 = kb.create_task(conn, title="task 2", triage=True)
        tid3 = kb.create_task(conn, title="task 3", triage=True)

    # Case 1: forced_fanout
    llm_single = jsonlib.dumps({
        "fanout": False,
        "title": "Tight",
        "body": "Body",
        "assignee": "engineer",
    })
    llm_multi = jsonlib.dumps({
        "fanout": True,
        "tasks": [{"title": "t1", "assignee": "engineer"}],
    })

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_single), _patch_extra_body():
            out1 = decomp.decompose_task(tid1, author="me", force_fanout=True)
        with _patch_aux_client(llm_multi), _patch_extra_body():
            out2 = decomp.decompose_task(tid2, author="me", force_fanout=False)
        with _patch_aux_client(llm_multi), _patch_extra_body():
            out3 = decomp.decompose_task(tid3, author="me")  # auto
    finally:
        for p in patches:
            p.stop()

    assert out1.mode == "forced_fanout"
    assert out2.mode == "forced_tighten"
    assert out3.mode == "auto"
