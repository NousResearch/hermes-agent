"""Dead-pin exits are booked as EX_CONFIG, not ladder-burning plain crashes.

The skill-pin gate refuses dead pins at create/dispatch/admission. When a pin
still dies inside the worker (a skill removed after admission, an assignee the
checker cannot enumerate), the death is deterministic — every retry dies
identically before the first model call. The worker therefore books the exit
with the existing terminal-configuration class (EX_CONFIG / 78) and its named
reason as the output, so the submitter exits 78 quoting the reason and the
dead-worker sweep parks the card on the FIRST occurrence instead of burning
the retry ladder. The board sentence names both causes of that exit; the
worker's own words (appended last output) carry the exact reason.

Breakthrough contract: a worker death on ``skill not found:`` books 78 with the
named output (any other death keeps the plain 1 booking and the usual budget).
"""
from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_client as wc
from gateway import session_kanban as sk

REASON = "skill not found: verification-before-ship (assignee=franky)"


def _pin_frame():
    return {"policy": {"kanban_json": json.dumps(
        {"db": "/unused/kanban.db", "profile": "franky", "skills": ["verification-before-ship"]})},
        "text": "work"}


def _run_worker_death(monkeypatch, exc: Exception) -> dict:
    booked: dict = {}

    def _boom(agent, frame, history, context):
        raise exc

    def _record(context, code, last_output):
        booked["code"] = code
        booked["last_output"] = last_output

    monkeypatch.setattr(sk, "_run_task_turns", _boom)
    monkeypatch.setattr(sk, "_record_worker_result", _record)
    with pytest.raises(type(exc)):
        sk._run_worker_turns(object(), _pin_frame(), [])
    return booked


def test_dead_pin_worker_death_books_ex_config_with_named_output(monkeypatch):
    booked = _run_worker_death(monkeypatch, ValueError(REASON))
    assert booked["code"] == kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE == 78
    assert booked["last_output"] == REASON


def test_other_worker_death_keeps_the_plain_crash_booking(monkeypatch):
    booked = _run_worker_death(monkeypatch, ValueError("transient upstream flake"))
    assert booked["code"] == 1
    assert booked["last_output"] == ""


def test_submitter_unknown_receipt_exits_with_the_booked_ex_config(monkeypatch, capsys):
    monkeypatch.setattr("gateway.session_kanban.worker_result",
                        lambda db, params: {"exit_code": 78, "last_output": REASON})
    code = wc._exit_code({"status": "unknown", "admission_id": "a1"},
                         {"board": "b", "task_id": "t", "run_id": 1, "claim_lock": "l"},
                         "/unused/kanban.db")
    captured = capsys.readouterr()
    assert code == 78
    assert REASON in captured.err
    assert "execution unknown" not in captured.err


def test_terminal_provider_board_line_names_both_causes_and_keeps_the_pinned_substring():
    dead = kbd._classify_dead_worker_exit(4321, "claimer", exit_code=78)
    assert dead.terminal_provider is True
    # The substring the credential-class booking is pinned on (test_kanban_db).
    assert "terminal provider error" in dead.error_text
    # ...and the second cause of the same exit: a worker-booked configuration death.
    assert "named configuration error" in dead.error_text
