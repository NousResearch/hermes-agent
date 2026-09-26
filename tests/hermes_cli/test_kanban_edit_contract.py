"""Invariant: a task's ``completion_contract`` can be changed after creation
(issue #121391).

A card created with an ``OWNER/REPO`` contract is otherwise stuck with it:
``kanban edit`` accepts only title/body/priority/result/summary/metadata, and
the only other writer (``prepare_acceptance``) narrows the contract toward the
published PR. The two honest no-PR closures — the change landed on the default
branch another way, and verify-only with no diff — then have no path to
``done``: ``--force`` does not bypass the missing ``published_pr`` gate, and
archiving misfiles the card as abandoned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _contract_events(conn, task_id):
    return [
        json.loads(row[0])
        for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'edited'"
            " AND payload LIKE '%completion_contract%'",
            (task_id,),
        )
    ]


def test_edit_relaxes_owner_repo_contract_to_local_only(kanban_home):
    """The verify-only escape hatch: downgrade to ``local-only`` on the record,
    then ``complete`` normally without a ``published_pr``."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="already correct", completion_contract="acme/repo")
        assert kb.edit_task(conn, tid, completion_contract="local-only") is True
        assert kb.get_task(conn, tid).completion_contract == "local-only"
        events = _contract_events(conn, tid)
        assert events and events[-1]["completion_contract_old"] == "acme/repo"
        assert events[-1]["completion_contract_new"] == "local-only"
        # No PR exists and none is needed anymore: plain completion lands done.
        assert kb.complete_task(conn, tid, result="verified already correct, no diff") is True
        assert kb.get_task(conn, tid).status == "done"


def test_edit_tightens_contract_and_validates(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="needs a gate")
        assert kb.edit_task(conn, tid, completion_contract="acme/repo") is True
        assert kb.get_task(conn, tid).completion_contract == "acme/repo"
        with pytest.raises(ValueError, match="completion_contract"):
            kb.edit_task(conn, tid, completion_contract="not a contract")
        assert kb.get_task(conn, tid).completion_contract == "acme/repo"


def test_edit_contract_is_a_no_op_when_unchanged(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="steady", completion_contract="acme/repo")
        assert kb.edit_task(conn, tid, completion_contract="acme/repo") is False


def test_edit_contract_refuses_live_claim(kanban_home):
    """Same fence as ``complete``: a running task under a live worker claim
    needs the worker's run to end (or a reclaim) before the gate changes."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="live", completion_contract="acme/repo")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        kbd._set_worker_pid(conn, tid, os.getpid())
        with pytest.raises(kb.LiveClaimError):
            kb.edit_task(conn, tid, completion_contract="local-only")
        assert kb.get_task(conn, tid).completion_contract == "acme/repo"


def test_edit_contract_without_live_worker_unchanged(kanban_home):
    """A claim whose worker never spawned protects no live run: the
    claim-then-edit flow keeps working."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="claimed", completion_contract="acme/repo")
        assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
        assert kb.edit_task(conn, tid, completion_contract="local-only") is True


def test_edit_contract_refuses_terminal_tasks(kanban_home):
    """A contract change on a ``done`` (or archived) card could never affect a
    gate again; refuse instead of recording a meaningless audit event."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="finished", completion_contract="local-only")
        assert kb.complete_task(conn, tid, result="done") is True
        assert kb.edit_task(conn, tid, completion_contract="acme/repo") is False
        assert kb.get_task(conn, tid).completion_contract == "local-only"


def test_cli_edit_accepts_completion_contract_flag(kanban_home):
    from hermes_cli import kanban as kc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="stuck", completion_contract="acme/repo")
    kc.run_slash(f"edit {tid} --completion-contract local-only")
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).completion_contract == "local-only"
