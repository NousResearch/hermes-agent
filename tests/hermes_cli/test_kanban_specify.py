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
    """Patch get_text_auxiliary_client at its source + at the module that
    imported it lazily inside specify_task. Both patches are needed
    because kanban_specify imports the function inside the function body.
    """
    client = _mock_client_returning(content)
    return patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(client, model),
    ), client


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def test_extract_json_blob_handles_plain_json():
    raw = '{"title": "T", "body": "B"}'
    assert spec._extract_json_blob(raw) == {"title": "T", "body": "B"}


def test_extract_json_blob_handles_fenced_json():
    raw = '```json\n{"title": "T", "body": "B"}\n```'
    assert spec._extract_json_blob(raw) == {"title": "T", "body": "B"}


def test_extract_json_blob_handles_prose_preamble():
    raw = 'Sure! Here you go:\n{"title": "T", "body": "B"}\nThanks.'
    assert spec._extract_json_blob(raw) == {"title": "T", "body": "B"}


def test_extract_json_blob_returns_none_for_unparseable():
    assert spec._extract_json_blob("no json here") is None
    assert spec._extract_json_blob("") is None
    assert spec._extract_json_blob("{not: valid}") is None


# ---------------------------------------------------------------------------
# specify_task (module-level entry point)
# ---------------------------------------------------------------------------

def test_specify_task_happy_path(kanban_home):
    with kb.connect() as conn:
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

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    # Parent-free → recompute_ready promotes to ready.
    assert task.status == "ready"
    assert task.title == "Refined rough"
    assert "**Goal**" in (task.body or "")


def test_specify_task_falls_back_to_body_only_on_bad_json(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="keep title", triage=True)

    # Model returned plain markdown, no JSON object.
    content = "Goal: Do a thing.\nApproach: Steps here."
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid)

    assert outcome.ok is True
    with kb.connect() as conn:
        t = kb.get_task(conn, tid)
    # Title preserved (no JSON with a title key).
    assert t.title == "keep title"
    # Body replaced with the raw response.
    assert "Goal:" in (t.body or "")


def test_specify_task_rejects_non_triage_task(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ready task")

    p, client = _patch_aux_client("unused")
    with p:
        outcome = spec.specify_task(tid)

    assert outcome.ok is False
    assert "not in triage" in outcome.reason
    # LLM must not be invoked for a non-triage task — fail cheap.
    assert client.chat.completions.create.call_count == 0


def test_specify_task_unknown_id(kanban_home):
    p, client = _patch_aux_client("unused")
    with p:
        outcome = spec.specify_task("t_nope")
    assert outcome.ok is False
    assert "unknown task" in outcome.reason
    assert client.chat.completions.create.call_count == 0


def test_specify_task_no_aux_client_configured(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    with patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(None, ""),
    ):
        outcome = spec.specify_task(tid)

    assert outcome.ok is False
    assert "auxiliary client" in outcome.reason
    # Task must stay in triage — we never touched it.
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_specify_task_llm_api_error_keeps_task_in_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=RuntimeError("429 rate limited"))
    with patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(client, "test-model"),
    ):
        outcome = spec.specify_task(tid)

    assert outcome.ok is False
    assert "LLM error" in outcome.reason
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_specify_task_empty_llm_response(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    p, _ = _patch_aux_client("")
    with p:
        outcome = spec.specify_task(tid)

    assert outcome.ok is False
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_list_triage_ids(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a", triage=True)
        b = kb.create_task(conn, title="b", triage=True, tenant="proj-1")
        kb.create_task(conn, title="c")  # not triage — excluded

    ids_all = spec.list_triage_ids()
    assert set(ids_all) == {a, b}
    ids_tenant = spec.list_triage_ids(tenant="proj-1")
    assert ids_tenant == [b]


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


def test_cli_specify_requires_id_or_all(kanban_home, capsys):
    rc = _run_cli("specify")
    assert rc == 2
    err = capsys.readouterr().err
    assert "requires a task id or --all" in err


def test_cli_specify_rejects_both_id_and_all(kanban_home, capsys):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)
    rc = _run_cli("specify", tid, "--all")
    assert rc == 2
    err = capsys.readouterr().err
    assert "either a task id OR --all" in err


def test_cli_specify_single_id_success(kanban_home, capsys):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    content = jsonlib.dumps({"title": "clean", "body": "body"})
    p, _ = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", tid)
    assert rc == 0
    out = capsys.readouterr().out
    assert tid in out
    assert "→ todo" in out or "-> todo" in out or "→" in out


def test_cli_specify_all_success_and_json(kanban_home, capsys):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a", triage=True)
        b = kb.create_task(conn, title="b", triage=True)

    content = jsonlib.dumps({"title": "spec", "body": "body"})
    p, _ = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", "--all", "--json")
    assert rc == 0
    lines = [l for l in capsys.readouterr().out.strip().splitlines() if l]
    # One JSON object per task + nothing else.
    assert len(lines) == 2
    parsed = [jsonlib.loads(l) for l in lines]
    ids = {row["task_id"] for row in parsed}
    assert ids == {a, b}
    assert all(row["ok"] for row in parsed)


def test_cli_specify_all_empty_triage_column(kanban_home, capsys):
    rc = _run_cli("specify", "--all")
    assert rc == 0
    assert "No triage tasks" in capsys.readouterr().out


def test_cli_specify_all_returns_1_when_every_task_fails(kanban_home, capsys):
    with kb.connect() as conn:
        kb.create_task(conn, title="a", triage=True)
        kb.create_task(conn, title="b", triage=True)

    with patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(None, ""),  # no aux client → every task fails
    ):
        rc = _run_cli("specify", "--all")

    assert rc == 1


def test_cli_specify_tenant_filter(kanban_home, capsys):
    with kb.connect() as conn:
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
    with kb.connect() as conn:
        assert kb.get_task(conn, outside).status == "triage"
        # The inside task was promoted.
        assert kb.get_task(conn, inside).status in {"todo", "ready"}


def test_cli_specify_author_passed_through(kanban_home, capsys):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    content = jsonlib.dumps({"title": "fresh title", "body": "fresh body"})
    p, _ = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", tid, "--author", "custom-agent")
    assert rc == 0
    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
    assert comments and comments[0].author == "custom-agent"
