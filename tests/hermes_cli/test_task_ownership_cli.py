"""CLI-level tests for ``hermes task``.

HERMES_HOME is isolated per test by the autouse ``_hermetic_environment``
fixture in tests/conftest.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli import task_ownership as tcli
from hermes_cli import task_ownership_db as tdb


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    task_parser = sub.add_parser("task", add_help=False)
    tcli.register_cli(task_parser)
    return parser


def run(argv, capsys=None):
    parser = _build_parser()
    args = parser.parse_args(["task", *argv])
    rc = args.func(args)
    if capsys is not None:
        # Drain output now so an unrelated later run_json() doesn't pick up
        # this call's stdout along with its own.
        capsys.readouterr()
    return rc


def run_json(capsys, argv):
    rc = run(argv)
    out = capsys.readouterr().out
    return rc, json.loads(out)


def _create(capsys, title="Sample task", extra=None) -> str:
    argv = ["create", title, "--json"]
    if extra:
        argv += extra
    rc, task = run_json(capsys, argv)
    assert rc == 0
    return task["id"]


# ── happy path CRUD ──────────────────────────────────────────────────────


def test_create_show_list(capsys):
    task_id = _create(capsys)
    rc, shown = run_json(capsys, ["show", task_id, "--json"])
    assert rc == 0
    assert shown["id"] == task_id
    assert shown["state"] == "NEW"

    rc, listed = run_json(capsys, ["list", "--json"])
    assert rc == 0
    assert any(t["id"] == task_id for t in listed)


def test_show_missing_task_returns_error(capsys):
    rc = run(["show", "t_nope"])
    assert rc == 1
    assert "no such task" in capsys.readouterr().err


def test_update_next_action_and_explicit_state(capsys):
    task_id = _create(capsys)
    rc, task = run_json(
        capsys,
        ["update", task_id, "--next-action", "write tests", "--state", "WORKING", "--json"],
    )
    assert rc == 0
    assert task["next_action"] == "write tests"
    assert task["state"] == "WORKING"


def test_update_invalid_transition_errors(capsys):
    task_id = _create(capsys)
    rc = run(["update", task_id, "--state", "DONE"])
    assert rc == 1
    assert "not a valid transition" in capsys.readouterr().err


# ── no-false-completion invariant ───────────────────────────────────────


def test_done_without_evidence_is_refused(capsys):
    task_id = _create(capsys)
    run(["update", task_id, "--state", "WORKING"])
    rc = run(["done", task_id])
    assert rc == 1
    assert "verification evidence" in capsys.readouterr().err
    _, task = run_json(capsys, ["show", task_id, "--json"])
    assert task["state"] != "DONE"


def test_verify_then_done_succeeds(capsys):
    task_id = _create(capsys)
    run(["update", task_id, "--state", "WORKING"], capsys)
    rc = run(["verify", task_id, "--evidence", "output diff is empty"], capsys)
    assert rc == 0
    rc, task = run_json(capsys, ["done", task_id, "--json"])
    assert rc == 0
    assert task["state"] == "DONE"


def test_approval_required_task_needs_approve_before_done(capsys):
    task_id = _create(capsys, extra=["--approval-required"])
    run(["update", task_id, "--state", "WORKING"])
    run(["verify", task_id, "--evidence", "checked"])

    rc = run(["done", task_id])
    assert rc == 1
    assert "approval" in capsys.readouterr().err.lower()

    run(["approve", task_id, "--by", "operator"], capsys)
    rc, task = run_json(capsys, ["done", task_id, "--json"])
    assert rc == 0
    assert task["state"] == "DONE"


# ── duplicate receipts ───────────────────────────────────────────────────


def test_duplicate_receipt_cli_is_reported_as_noop(capsys):
    task_id = _create(capsys)
    rc = run(["receipt", task_id, "--receipt-id", "ext-1", "--source", "stripe"])
    assert rc == 0
    assert "recorded" in capsys.readouterr().out

    rc = run(["receipt", task_id, "--receipt-id", "ext-1", "--source", "stripe"])
    assert rc == 0
    assert "no-op" in capsys.readouterr().out


# ── retries + fallback ───────────────────────────────────────────────────


def test_outcome_retry_exhausts_and_records_fallback(capsys):
    task_id = _create(capsys, extra=["--max-retries", "1"])
    run(["update", task_id, "--state", "WORKING"], capsys)

    rc, task = run_json(
        capsys, ["outcome", task_id, "--result", "failure", "--retry", "--json"]
    )
    assert task["state"] == "RETRYING"

    run(["update", task_id, "--state", "WORKING"], capsys)
    rc, task = run_json(
        capsys,
        [
            "outcome", task_id, "--result", "failure", "--retry",
            "--fallback", "notify on-call", "--json",
        ],
    )
    assert rc == 0
    assert task["state"] == "BLOCKED"
    assert task["fallback"] == "notify on-call"


def test_outcome_success_does_not_complete_task(capsys):
    task_id = _create(capsys)
    run(["update", task_id, "--state", "WORKING"], capsys)
    rc, task = run_json(
        capsys, ["outcome", task_id, "--result", "success", "--json"]
    )
    assert rc == 0
    assert task["state"] != "DONE"


# ── events / audit trail ─────────────────────────────────────────────────


def test_events_trail_shows_verification_before_completion(capsys):
    task_id = _create(capsys)
    run(["update", task_id, "--state", "WORKING"], capsys)
    run(["verify", task_id, "--evidence", "confirmed"], capsys)
    run(["done", task_id], capsys)

    rc, events = run_json(capsys, ["events", task_id, "--json"])
    assert rc == 0
    kinds = [e["event"] for e in events]
    assert kinds.index("verification_recorded") < kinds.index("completed")


# ── feature flag / shadow mode ───────────────────────────────────────────


def test_status_reports_disabled_by_default(capsys):
    rc, status = run_json(capsys, ["status", "--json"])
    assert rc == 0
    assert status["enabled"] is False


def test_age_check_is_silent_and_inert_while_disabled(capsys):
    task_id = _create(capsys)
    run(["update", task_id, "--state", "WORKING"], capsys)

    conn = tdb.connect()
    when = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    conn.execute(
        "UPDATE tasks SET state_changed_at = ?, updated_at = ? WHERE id = ?",
        (when, when, task_id),
    )
    conn.commit()

    rc = run(["age-check"])
    assert rc == 0
    assert capsys.readouterr().out == ""  # no notification spam while off

    _, task = run_json(capsys, ["show", task_id, "--json"])
    assert task["state"] == "WORKING"  # untouched — shadow mode is inert


def test_age_check_dry_run_works_regardless_of_flag(capsys):
    task_id = _create(capsys)
    run(["update", task_id, "--state", "WORKING"])
    conn = tdb.connect()
    when = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    conn.execute(
        "UPDATE tasks SET state_changed_at = ?, updated_at = ? WHERE id = ?",
        (when, when, task_id),
    )
    conn.commit()

    rc = run(["age-check", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert task_id in out
    assert "[DRY RUN]" in out

    _, task = run_json(capsys, ["show", task_id, "--json"])
    assert task["state"] == "WORKING"  # dry-run never mutates


def test_enable_disable_flip_age_check_behavior(capsys):
    task_id = _create(capsys)
    run(["update", task_id, "--state", "WORKING"], capsys)
    conn = tdb.connect()
    when = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    conn.execute(
        "UPDATE tasks SET state_changed_at = ?, updated_at = ? WHERE id = ?",
        (when, when, task_id),
    )
    conn.commit()

    rc = run(["enable"], capsys)
    assert rc == 0
    _, status = run_json(capsys, ["status", "--json"])
    assert status["enabled"] is True

    rc = run(["age-check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert task_id in out
    _, task = run_json(capsys, ["show", task_id, "--json"])
    assert task["state"] == "STALE"

    # Clean rollback: disable makes age-check inert again immediately.
    rc = run(["disable"], capsys)
    assert rc == 0
    _, status = run_json(capsys, ["status", "--json"])
    assert status["enabled"] is False


def test_disable_does_not_touch_existing_task_state(capsys):
    """Rollback (disabling the flag) must never mutate/delete durable state."""
    task_id = _create(capsys)
    run(["update", task_id, "--state", "WORKING"], capsys)
    run(["enable"], capsys)
    run(["disable"], capsys)
    _, task = run_json(capsys, ["show", task_id, "--json"])
    assert task["id"] == task_id
    assert task["state"] == "WORKING"
