"""Tests for the specifier module + `hermes kanban specify` CLI surface.

The auxiliary LLM client is mocked — these tests don't hit any network or
real provider. They exercise the prompt plumbing, response parsing, DB
writes, and CLI flag surface.
"""

from __future__ import annotations

import argparse
import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    """Build a minimal object shaped like an OpenAI chat.completions result.

    The specifier only reads ``resp.choices[0].message.content``, so we
    avoid importing the openai SDK and build the tree with MagicMock.
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    """Patch call_llm at its source module — specify_task now routes through
    it (#35566) instead of building a raw client. Returns (patcher, mock) so
    callers can still assert on the call.
    """
    mock_fn = MagicMock(return_value=_fake_aux_response(content))
    return patch("agent.auxiliary_client.call_llm", mock_fn), mock_fn


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# specify_task (module-level entry point)
# ---------------------------------------------------------------------------

def test_specify_task_happy_path(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal**\nA concrete goal.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    assert outcome.task_id == tid
    assert outcome.new_title == "Refined rough"

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # Parent-free → recompute_ready promotes to ready.
    assert task.status == "ready"
    assert task.title == "Refined rough"
    assert "**Goal**" in (task.body or "")


def _patch_router(*, configured=True, trivial=True, model="typesafe/jev-latest"):
    """Patch kanban_triage_router at its source module — specify_task calls
    it directly, so patching the source works regardless of import style."""
    return patch.multiple(
        "hermes_cli.kanban_triage_router",
        router_configured=MagicMock(return_value=configured),
        is_trivial=MagicMock(return_value=trivial),
        configured_model=MagicMock(return_value=model if configured else None),
    )


def test_specify_task_auto_promotes_on_trivial_verdict_skips_full_llm_call(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Fix a typo in the README", triage=True)

    mock_llm = MagicMock()  # must never be called — the router path skips it
    with _patch_router(trivial=True), patch("agent.auxiliary_client.call_llm", mock_llm):
        outcome = spec.specify_task(tid, author="ace")

    mock_llm.assert_not_called()
    assert outcome.ok is True
    assert outcome.task_id == tid

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
    assert task.status == "ready"  # no parents -> recompute_ready promotes past todo
    assert "**Goal**" in (task.body or "")
    assert len(events) == 1
    assert events[0].payload["auto_promoted"] is True
    assert events[0].payload["router_model"] == "typesafe/jev-latest"


def test_specify_task_falls_through_to_full_specify_when_not_trivial(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough idea", triage=True)

    content = jsonlib.dumps({"title": "Refined rough", "body": "**Goal**\nA concrete goal."})
    p, mock_llm = _patch_aux_client(content)
    with _patch_router(trivial=False), p:
        outcome = spec.specify_task(tid, author="ace")

    mock_llm.assert_called_once()  # full specify path ran as before
    assert outcome.ok is True
    assert outcome.new_title == "Refined rough"

    with kbc.connect() as conn:
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
    assert "auto_promoted" not in (events[0].payload or {})


def test_specify_task_full_path_unaffected_when_router_unconfigured(kanban_home):
    """Router unconfigured is functionally identical to it never having been
    added: is_trivial() would itself return False, but this exercises the
    real (unpatched) is_trivial() short-circuit end to end."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough idea", triage=True)

    content = jsonlib.dumps({"title": "Refined rough", "body": "**Goal**\nA concrete goal."})
    p, mock_llm = _patch_aux_client(content)
    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        lambda task: {},
    ), p:
        outcome = spec.specify_task(tid, author="ace")

    mock_llm.assert_called_once()
    assert outcome.ok is True
    assert outcome.new_title == "Refined rough"






# ---------------------------------------------------------------------------
# CLI wiring — argparse + _cmd_specify
# ---------------------------------------------------------------------------

def _run_cli(*argv: str) -> int:
    """Invoke the `hermes kanban …` argparse surface directly."""
    root = argparse.ArgumentParser()
    subp = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subp)
    ns = root.parse_args(["kanban", *argv])
    return kanban_cli.kanban_command(ns)




def test_cli_specify_tenant_filter(kanban_home, capsys):
    with kbc.connect() as conn:
        outside = kb.create_task(conn, title="outside", triage=True)
        inside = kb.create_task(
            conn, title="inside", triage=True, tenant="proj-a",
        )

    content = jsonlib.dumps({"title": "spec", "body": "body"})
    p, _ = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", "--all", "--tenant", "proj-a", "--json")
    assert rc == 0
    lines = [
        jsonlib.loads(l)
        for l in capsys.readouterr().out.strip().splitlines()
        if l
    ]
    ids = {row["task_id"] for row in lines}
    assert ids == {inside}

    # The outside task stays in triage.
    with kbc.connect() as conn:
        assert kb.get_task(conn, outside).status == "triage"
        # The inside task was promoted.
        assert kb.get_task(conn, inside).status in {"todo", "ready"}


