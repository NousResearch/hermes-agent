"""Tests for the ``pre_kanban_decompose`` directive hook.

A plugin can ``skip`` an automatic decomposition (the task is promoted as one
unit with no LLM call) or ``route`` a single decomposer call to another model.
Invalid, failing and timed-out callbacks fail open: the task decomposes as
configured. An explicitly requested (manual) decomposition cannot be skipped.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS, get_plugin_manager
from hermes_cli.plugins_dispatch import _HOOK_TIMEOUT_BOUNDED_HOOKS

_FANOUT_REPLY = json.dumps({
    "fanout": True,
    "rationale": "split",
    "tasks": [
        {"title": "research", "body": "look", "assignee": "worker", "parents": []},
        {"title": "build", "body": "code", "assignee": "worker", "parents": [0]},
    ],
})


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    fake_profiles = [
        SimpleNamespace(name=n, description=f"desc for {n}") for n in ("orchestrator", "worker")
    ]
    names = {p.name for p in fake_profiles}
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda: fake_profiles)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda n: n in names)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "orchestrator")
    return home


@pytest.fixture
def hooks():
    """Register ``pre_kanban_decompose`` callbacks; restores the registry afterwards."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _register(*callbacks):
        mgr._hooks.setdefault("pre_kanban_decompose", []).extend(callbacks)

    try:
        yield _register
    finally:
        mgr._hooks = saved


@pytest.fixture
def call_llm():
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = _FANOUT_REPLY
    with patch("agent.auxiliary_client.call_llm", return_value=resp) as mock:
        yield mock


def _triage_task(title: str = "ship a feature", body: str = "details", assignee=None) -> str:
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title=title, body=body, assignee=assignee, triage=True)


def _task_and_events(tid: str):
    with kbc.connect_closing() as conn:
        return kb.get_task(conn, tid), kb.list_events(conn, tid), kb.list_comments(conn, tid)


def test_hook_is_registered_bounded_and_python_only():
    assert "pre_kanban_decompose" in VALID_HOOKS
    assert "pre_kanban_decompose" in _HOOK_TIMEOUT_BOUNDED_HOOKS
    assert "pre_kanban_decompose" in SHELL_UNSUPPORTED_HOOKS


def test_hook_receives_task_fields(kanban_home, hooks, call_llm):
    seen: list[dict] = []
    hooks(lambda **kw: seen.append(kw))
    tid = _triage_task(title="t1", body="b1", assignee="worker")

    decomp.decompose_task(tid, author="me", trigger="auto")

    assert len(seen) == 1
    kw = seen[0]
    assert (kw["task_id"], kw["title"], kw["body"], kw["assignee"]) == (tid, "t1", "b1", "worker")
    assert kw["board"] == kb.get_current_board()
    assert kw["profile_name"] == "orchestrator"
    assert kw["trigger"] == "auto"


def test_skip_promotes_single_task_without_llm_call(kanban_home, hooks, call_llm):
    hooks(lambda **kw: {"action": "skip", "reason": "one unit of work"})
    tid = _triage_task()

    outcome = decomp.decompose_task(tid, author="auto-decomposer", trigger="auto")

    call_llm.assert_not_called()
    assert outcome.ok and outcome.fanout is False and not outcome.child_ids
    assert "one unit of work" in outcome.reason
    task, events, comments = _task_and_events(tid)
    assert task.status in ("todo", "ready")
    assert task.title == "ship a feature" and task.body == "details"
    assert task.assignee == "orchestrator"  # unassigned task gets default_assignee
    specified = [e for e in events if e.kind == "specified"]
    assert specified and "one unit of work" in specified[-1].payload["reason"]
    assert any("one unit of work" in c.body and c.author == "auto-decomposer" for c in comments)
    with kbc.connect_closing() as conn:
        assert [t.id for t in kb.list_tasks(conn, limit=100)] == [tid]  # no children created


def test_skip_keeps_existing_assignee(kanban_home, hooks, call_llm):
    hooks(lambda **kw: {"action": "skip", "reason": "small"})
    tid = _triage_task(assignee="worker")

    decomp.decompose_task(tid, trigger="auto")

    assert _task_and_events(tid)[0].assignee == "worker"


def test_manual_decompose_ignores_skip(kanban_home, hooks, call_llm):
    hooks(lambda **kw: {"action": "skip", "reason": "small"})
    tid = _triage_task()

    outcome = decomp.decompose_task(tid, author="me")  # CLI / dashboard default

    call_llm.assert_called_once()
    assert outcome.ok and outcome.fanout is True


@pytest.mark.parametrize("trigger", ["auto", "manual"])
def test_route_overrides_reach_call_llm(kanban_home, hooks, call_llm, trigger):
    hooks(lambda **kw: {"action": "route", "provider": "openrouter", "model": "cheap/model",
                        "reasoning_effort": "low"})
    tid = _triage_task()

    outcome = decomp.decompose_task(tid, trigger=trigger)

    assert outcome.ok and outcome.fanout is True
    kwargs = call_llm.call_args.kwargs
    assert kwargs["task"] == "kanban_decomposer"
    assert kwargs["provider"] == "openrouter"
    assert kwargs["model"] == "cheap/model"
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_route_model_only_keeps_configured_provider(kanban_home, hooks, call_llm):
    hooks(lambda **kw: {"action": "route", "model": "mid/model"})

    decomp.decompose_task(_triage_task(), trigger="auto")

    kwargs = call_llm.call_args.kwargs
    assert kwargs["model"] == "mid/model"
    assert "provider" not in kwargs and "reasoning_config" not in kwargs


def test_no_directive_leaves_call_unchanged(kanban_home, hooks, call_llm):
    hooks(lambda **kw: None)

    outcome = decomp.decompose_task(_triage_task(), trigger="auto")

    assert outcome.ok and outcome.fanout is True
    kwargs = call_llm.call_args.kwargs
    assert not {"provider", "model", "reasoning_config"} & set(kwargs)


@pytest.mark.parametrize("bad", [
    "skip",
    {"action": "skip"},
    {"action": "skip", "reason": "   "},
    {"action": "route"},
    {"action": "route", "model": ""},
    {"action": "route", "model": "m", "provider": 3},
    {"action": "route", "model": "m", "reasoning_effort": "hgih"},
    {"action": "block", "reason": "no"},
])
def test_invalid_directive_is_ignored(kanban_home, hooks, call_llm, bad):
    hooks(lambda **kw: bad)

    outcome = decomp.decompose_task(_triage_task(), trigger="auto")

    call_llm.assert_called_once()
    assert not {"provider", "model", "reasoning_config"} & set(call_llm.call_args.kwargs)
    assert outcome.ok and outcome.fanout is True


def test_first_valid_directive_wins(kanban_home, hooks, call_llm):
    hooks(lambda **kw: {"action": "route"},  # invalid, ignored
          lambda **kw: {"action": "route", "model": "first"},
          lambda **kw: {"action": "skip", "reason": "loses"})

    decomp.decompose_task(_triage_task(), trigger="auto")

    assert call_llm.call_args.kwargs["model"] == "first"


def test_raising_callback_fails_open(kanban_home, hooks, call_llm):
    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    hooks(_boom)

    outcome = decomp.decompose_task(_triage_task(), trigger="auto")

    call_llm.assert_called_once()
    assert outcome.ok and outcome.fanout is True


def test_timed_out_callback_fails_open(kanban_home, hooks, call_llm, monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1)
    release = threading.Event()

    def _hang(**kw):
        release.wait(5)
        return {"action": "skip", "reason": "too late"}

    hooks(_hang)
    try:
        outcome = decomp.decompose_task(_triage_task(), trigger="auto")
    finally:
        release.set()

    call_llm.assert_called_once()
    assert outcome.ok and outcome.fanout is True


def test_dispatcher_auto_decompose_passes_auto_trigger(kanban_home, hooks, call_llm):
    from gateway.kanban_watchers_dispatcher import _KanbanDispatcher

    seen: list[str] = []
    hooks(lambda **kw: seen.append(kw["trigger"]) or {"action": "skip", "reason": "small"})
    tid = _triage_task()

    assert _KanbanDispatcher._decompose_one(decomp, "default", tid) == 1

    assert seen == ["auto"]
    call_llm.assert_not_called()
    assert _task_and_events(tid)[0].status in ("todo", "ready")
