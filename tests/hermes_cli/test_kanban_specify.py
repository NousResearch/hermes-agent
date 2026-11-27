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


# ---------------------------------------------------------------------------
# Line-1 task_type marker carry-over (fleet dispatcher requirement)
# ---------------------------------------------------------------------------


def test_specify_task_carries_task_type_marker(kanban_home):
    """The old body's ``task_type:`` marker is authoritative: it survives the
    LLM rewrite even when the reply drops it entirely."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="rough", triage=True,
            body="task_type: code\nRough one-liner body.")

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal**\nShip the thing.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body.startswith("task_type: code\n")
    assert "**Goal**" in (task.body or "")


def test_specify_task_normalizes_drifted_task_type_marker(kanban_home):
    """A drifted first-line marker in the reply (wrong casing / spacing, or an
    attempted type change) is restored to the canonical old-body form; noise
    that merely looks like a marker further down is kept verbatim."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="rough", triage=True,
            body="task_type: review\nRough body.")

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "TASK_TYPE:  code\n**Goal**\nTaskType: OPS\nExtra tail.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # Old body wins: canonical marker restored, LLM's type override rejected.
    assert task.body.startswith("task_type: review\n")
    assert "task_type: code" not in (task.body or "")
    # Marker-like noise below line 1 is not touched.
    assert "TaskType: OPS" in (task.body or "")


def test_specify_task_without_old_marker_not_decorated(kanban_home):
    """No marker in the old body → none is invented in the new body."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="rough", triage=True, body="Rough body, no marker.")

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal**\nDo it.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert not task.body.startswith("task_type:")
    assert "task_type" not in (task.body or "")


def test_specify_task_canonical_echo_passes_through_unchanged(kanban_home):
    """A reply that already carries the exact canonical marker is kept
    byte-for-byte (no double-prefix, no rewrite)."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="rough", triage=True,
            body="task_type: ops\nRough body.")

    canonical_new = "task_type: ops\n**Goal**\nAlready canonical."
    content = jsonlib.dumps({"title": "Refined rough", "body": canonical_new})
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body == canonical_new


def test_specify_task_marker_survives_non_json_fallback_body(kanban_home):
    """The lenient parse fallback (whole reply becomes the body) still carries
    the marker — the reconcile runs on both reply shapes."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="rough", triage=True,
            body="task_type: research\nRough body.")

    p, _ = _patch_aux_client("not json at all — raw prose reply")
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body.startswith("task_type: research\n")
    assert "not json at all" in (task.body or "")


# ---------------------------------------------------------------------------
# _reconcile_task_type — pure-function invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["research", "code", "review", "ops"])
def test_reconcile_all_four_types_round_trip(marker):
    out = spec._reconcile_task_type(
        f"task_type: {marker}\nold", "**Goal**\nnew")
    assert out == f"task_type: {marker}\n**Goal**\nnew"


@pytest.mark.parametrize("drifted", [
    "TASK_TYPE: code", "task_type:CODE", "  task_type :  Code ", "task_type: Code",
])
def test_reconcile_normalizes_drifted_first_line(drifted):
    out = spec._reconcile_task_type(
        "task_type: code\nold", f"{drifted}\nrest")
    assert out == "task_type: code\nrest"


def test_reconcile_unknown_type_in_old_body_is_not_a_marker():
    """An old first line that is not one of the four types carries nothing —
    the specifier never invents a type."""
    body = "**Goal**\nnew"
    assert spec._reconcile_task_type("task_type: banana\nold", body) == body
    assert spec._reconcile_task_type(None, body) == body
    assert spec._reconcile_task_type("", body) == body


def test_reconcile_none_new_body_stays_none():
    """title-only replies (body=None) keep ``specify_triage_task``'s
    preserve-existing-body semantics: None in → None out, marker untouched."""
    assert spec._reconcile_task_type("task_type: code\nold", None) is None





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


