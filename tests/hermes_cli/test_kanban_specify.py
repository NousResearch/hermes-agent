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

def test_specify_task_promotes_title_without_rewriting_existing_body(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="rough",
            body="**Binding criteria**\n- Preserve this exact requirement.",
            triage=True,
        )

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
        events = kb.list_events(conn, tid)
        comments = kb.list_comments(conn, tid)
    # Parent-free → recompute_ready promotes to ready.
    assert task.status == "ready"
    assert task.title == "Refined rough"
    assert task.body == "**Binding criteria**\n- Preserve this exact requirement."
    specified = next(event for event in events if event.kind == "specified")
    assert specified.payload == {"changed_fields": ["title"]}
    assert "title" in comments[-1].body
    assert "body" not in comments[-1].body


def test_specify_task_rewrites_body_only_when_explicitly_requested(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", body="Original body", triage=True)

    content = jsonlib.dumps({"title": "Refined rough", "body": "**Goal**\nA concrete goal."})
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace", rewrite_body=True)

    assert outcome.ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.title == "Refined rough"
    assert task.body == "**Goal**\nA concrete goal."


def test_specify_task_malformed_response_preserves_existing_body(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", body="Must not be replaced", triage=True)

    p, _ = _patch_aux_client("This is not JSON")
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
    assert task.body == "Must not be replaced"
    specified = next(event for event in events if event.kind == "specified")
    assert specified.payload is None


def test_specify_task_empty_response_does_not_promote_or_change_body(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", body="Must not be replaced", triage=True)

    p, _ = _patch_aux_client("")
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is False
    assert outcome.reason == "LLM returned an empty response"
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "triage"
    assert task.body == "Must not be replaced"


def test_cli_rewrite_body_flag_is_explicit_opt_in(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", body="Original body", triage=True)

    content = jsonlib.dumps({"title": "Refined rough", "body": "Replacement body"})
    p, _ = _patch_aux_client(content)
    with p:
        assert _run_cli("specify", tid, "--rewrite-body") == 0

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body == "Replacement body"






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
