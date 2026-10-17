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




# ---------------------------------------------------------------------------
# PROMPT_BODY_LIMIT — refuse over-bound bodies; --force preserves the tail
# ---------------------------------------------------------------------------

def _over_bound_body(*, marker: str = "ACCEPTANCE_TAIL_MARKER") -> str:
    """Body just over PROMPT_BODY_LIMIT with a distinctive acceptance-criteria tail."""
    head = "H" * (spec.PROMPT_BODY_LIMIT - 40)
    return head + f"\n\n**Acceptance criteria**\n- {marker}\n"


def test_specify_refuses_over_bound_body_without_writing(kanban_home):
    """Over-bound triage body: refuse before the LLM call; DB untouched."""
    original = _over_bound_body()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="long", body=original, triage=True)

    called = MagicMock()
    with patch("agent.auxiliary_client.call_llm", called):
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is False
    assert "PROMPT_BODY_LIMIT" in outcome.reason
    assert outcome.body_length == len(original)
    assert outcome.prompt_body_limit == spec.PROMPT_BODY_LIMIT
    called.assert_not_called()

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "triage"
    assert task.body == original


def test_specify_force_appends_unseen_tail_exactly_once(kanban_home):
    """--force accepts the spec and appends the original unseen tail verbatim once."""
    marker = "ACCEPTANCE_TAIL_UNIQUE_9f3a"
    original = _over_bound_body(marker=marker)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="long", body=original, triage=True)

    new_body = "**Goal**\nRewritten goal.\n**Acceptance criteria**\n- done"
    content = jsonlib.dumps({"title": "Refined long", "body": new_body})
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace", force=True)

    assert outcome.ok is True
    assert outcome.body_truncated is True
    assert outcome.body_length == len(original)

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status in {"todo", "ready"}
    assert marker in (task.body or "")
    assert (task.body or "").count(marker) == 1
    assert "Original body tail" in (task.body or "")
    # The unseen complement starts at PROMPT_BODY_LIMIT - 1 (truncate keeps limit-1 + '…').
    assert original[spec.PROMPT_BODY_LIMIT - 1:] in (task.body or "")


def test_specify_body_exactly_at_bound_passes_through(kanban_home):
    """A body of exactly PROMPT_BODY_LIMIT chars is not refused."""
    body = "B" * spec.PROMPT_BODY_LIMIT
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="exact", body=body, triage=True)

    content = jsonlib.dumps({"title": "Exact", "body": "**Goal**\nok"})
    p, mock_fn = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    assert outcome.body_truncated is False
    mock_fn.assert_called_once()
    # Prompt received the full body (no ellipsis clamp).
    user_msg = mock_fn.call_args.kwargs["messages"][1]["content"]
    assert body in user_msg
    assert "…" not in user_msg.split("Current body:\n", 1)[1]


def test_specify_force_malformed_reply_does_not_overwrite_over_bound_body(kanban_home):
    """Forced over-bound + malformed reply: promote unchanged, never write raw text."""
    original = _over_bound_body()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="long", body=original, triage=True)

    p, _ = _patch_aux_client("this is not json at all")
    with p:
        outcome = spec.specify_task(tid, author="ace", force=True)

    assert outcome.ok is True
    assert "unchanged" in outcome.reason
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.body == original
    assert task.status in {"todo", "ready"}
