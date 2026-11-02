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
# Routing label preservation
#
# A leading `Word:` on a card title is a routing contract — the board's
# automation dispatches on it (`Spec:` -> spec, `Build:` -> build). The
# specifier asks an auxiliary LLM for a tightened title and writes it back,
# so before this fix the label was rewritten along with the prose: `Spec:
# 2026-10-31-money-tracker-stitch-design-of-record` came back as `Specify
# 2026-10-31-money-tracker-stitch-design-of-record` and the card silently
# stopped being dispatched.
# ---------------------------------------------------------------------------


def _specify_with_reply(task_id: str, title: str, body: str = "**Goal**\nA goal."):
    """Run the specifier against a mocked LLM that answers with ``title``."""
    content = jsonlib.dumps({"title": title, "body": body})
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(task_id, author="ace")
    assert outcome.ok is True, outcome.reason
    return outcome


def test_specify_task_keeps_leading_routing_label(kanban_home):
    """The live symptom: the model rewrote `Spec:` into the prose word
    `Specify`, and the card stopped being dispatched."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Spec: 2026-10-31-money-tracker-stitch-design-of-record", triage=True,
        )

    outcome = _specify_with_reply(tid, "Specify 2026-10-31-money-tracker-stitch-design-of-record")

    assert outcome.new_title == "Spec: 2026-10-31-money-tracker-stitch-design-of-record"
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.title == "Spec: 2026-10-31-money-tracker-stitch-design-of-record"


def test_specify_task_keeps_original_label_when_model_echoes_another(kanban_home):
    """The ORIGINAL label is authoritative — a model reply cannot re-route a
    card by answering with a different label."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Spec: stitch the money tracker", triage=True)

    outcome = _specify_with_reply(tid, "Review: stitch the money tracker")

    assert outcome.new_title == "Spec: stitch the money tracker"


def test_specify_task_leaves_unlabelled_title_unchanged(kanban_home):
    """A title with no routing label behaves exactly as before the fix."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough idea about money", triage=True)

    outcome = _specify_with_reply(tid, "Refined money idea")

    assert outcome.new_title == "Refined money idea"
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.title == "Refined money idea"


def test_specify_task_prompt_asks_for_the_label_verbatim(kanban_home):
    """The prompt is the first line of defence; the code path is the guarantee."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Spec: stitch the money tracker", triage=True)

    content = jsonlib.dumps({"title": "Spec: stitch the money tracker", "body": "body"})
    p, mock_fn = _patch_aux_client(content)
    with p:
        spec.specify_task(tid, author="ace")

    system = mock_fn.call_args.kwargs["messages"][0]["content"]
    assert "ROUTING LABEL" in system
    assert "Spec:" in system






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


