"""Attributable disables: every transition that flips ``enabled`` true→false must
record why, in jobs.json itself (one read, not two).

Motivating incident: job ``f586101aac06`` was found disabled with no recorded
reason — the third silent disable that week. ``pause_job`` already writes
``paused_at``/``paused_reason`` next to ``enabled=false``, but the CLI never
sent a reason, and ``update_job`` (dashboard/API/``cron edit``) could flip
``enabled`` with nothing recorded at all.

Contract under test:
- ``pause_job``/``update_job`` stamp ``disabled_reason`` (+ ``disabled_at``)
  on every disabling transition; re-enable clears them (and records an optional
  ``enabled_reason``).
- ``hermes cron pause`` and ``hermes cron edit --disable`` refuse reason-less
  disables (same refusal shape both doors).
- Legacy jobs.json rows without the fields load unchanged.
"""

import json
from types import SimpleNamespace

from cron import jobs
from hermes_cli.cron import cron_command


def _store(tmp_path):
    """The single open-store block every operation in a test runs inside."""
    return jobs.use_cron_store(tmp_path)  # store file lands at <home>/cron/jobs.json


def _create():
    return jobs.create_job(prompt="watch the lane", schedule="every 1h")


def _raw_rows(tmp_path):
    data = json.loads((tmp_path / "cron" / "jobs.json").read_text())
    return data["jobs"] if isinstance(data, dict) else data


def _raw_row(tmp_path, job_id):
    return next(row for row in _raw_rows(tmp_path) if row["id"] == job_id)


# ---------------------------------------------------------------------------
# Store level: the reason lands in jobs.json next to the state it explains.
# ---------------------------------------------------------------------------

def test_pause_with_reason_persists_disable_fields(tmp_path):
    with _store(tmp_path):
        job = _create()
        paused = jobs.pause_job(job["id"], reason="watchdog false positive, triaging")
        assert paused["enabled"] is False
        assert paused["paused_reason"] == "watchdog false positive, triaging"
        # The required fields land in jobs.json itself — one read, not two.
        row = _raw_row(tmp_path, job["id"])
        assert row["disabled_reason"] == "watchdog false positive, triaging"
        assert row["disabled_at"]


def test_disable_via_update_persists_disable_fields(tmp_path):
    """The edit/update side door: flipping enabled true→false records why."""
    with _store(tmp_path):
        job = _create()
        updated = jobs.update_job(job["id"], {"enabled": False, "paused_reason": "cost freeze"})
        assert updated["enabled"] is False
        row = _raw_row(tmp_path, job["id"])
        assert row["disabled_reason"] == "cost freeze"
        assert row["disabled_at"]


def test_disable_without_reason_is_marked_unattributed(tmp_path):
    """A programmatic reason-less disable (dashboard toggle, API) still lands a
    marker naming the pattern rather than nothing at all."""
    with _store(tmp_path):
        job = _create()
        jobs.update_job(job["id"], {"enabled": False})
        row = _raw_row(tmp_path, job["id"])
        assert row["enabled"] is False
        assert "unattributable" in row["disabled_reason"]
        assert row["disabled_at"]


def test_reenable_clears_disable_fields(tmp_path):
    with _store(tmp_path):
        job = _create()
        jobs.pause_job(job["id"], reason="ops audit")
        resumed = jobs.resume_job(job["id"])
        assert resumed["enabled"] is True
        assert resumed["disabled_reason"] is None
        assert resumed["disabled_at"] is None
        row = _raw_row(tmp_path, job["id"])
        assert row["disabled_reason"] is None
        assert row["disabled_at"] is None


def test_enable_via_update_records_reason_and_clears_disable_fields(tmp_path):
    """Re-enable through the update path also clears the disable attribution
    and records why the job came back (cheap: same stamp shape)."""
    with _store(tmp_path):
        job = _create()
        jobs.pause_job(job["id"], reason="ops audit")
        updated = jobs.update_job(job["id"], {"enabled": True, "enabled_reason": "audit cleared"})
        assert updated["enabled"] is True
        assert updated["disabled_reason"] is None
        assert updated["disabled_at"] is None
        row = _raw_row(tmp_path, job["id"])
        assert row.get("enabled_reason") == "audit cleared"


# ---------------------------------------------------------------------------
# CLI level: both doors refuse a reason-less disable, same refusal shape.
# ---------------------------------------------------------------------------

def _pause_args(job_id, reason=None, via_flag=False):
    # The parser exposes --reason as ``reason_flag`` and the positional as
    # ``reason``; exactly one carries the value depending on the invocation.
    if via_flag:
        return SimpleNamespace(cron_command="pause", job_id=job_id, reason=None, reason_flag=reason)
    return SimpleNamespace(cron_command="pause", job_id=job_id, reason=reason, reason_flag=None)


def _edit_args(job_id, **overrides):
    base = dict(
        cron_command="edit", job_id=job_id, disable=None, reason=None, reason_flag=None,
        schedule=None, prompt=None, name=None, deliver=None, failure_deliver=None,
        repeat=None, catch_up=None, skill=None, skills=None, add_skills=None,
        remove_skills=None, clear_skills=False, script=None, no_agent=None,
        continuity=None, no_continuity=None, monitor_script=None, monitor_url=None,
        workdir=None, model=None, model_provider=None, pinned=None,
        reasoning_effort=None, interpreter=None, timezone=None, grace_minutes=None,
        enabled_reason=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cli_pause_without_reason_refuses(tmp_path, capsys):
    with _store(tmp_path):
        job = _create()
        assert cron_command(_pause_args(job["id"])) == 1
        out = capsys.readouterr().out
        assert "disables without reasons are unattributable" in out
        assert "--reason" in out
        # Refused: nothing changed on disk.
        assert jobs.get_job(job["id"])["enabled"] is True


def test_cli_pause_with_flag_reason_persists(tmp_path, capsys):
    with _store(tmp_path):
        job = _create()
        assert cron_command(_pause_args(job["id"], reason="watchdog retool", via_flag=True)) == 0
        out = capsys.readouterr().out
        assert "Paused job" in out
        row = _raw_row(tmp_path, job["id"])
        assert row["enabled"] is False
        assert row["paused_reason"] == "watchdog retool"
        assert row["disabled_reason"] == "watchdog retool"


def test_cli_pause_with_positional_reason_persists(tmp_path, capsys):
    """`hermes cron pause <id> "reason"` — positional, same effect as --reason."""
    with _store(tmp_path):
        job = _create()
        assert cron_command(_pause_args(job["id"], reason="false positive storm")) == 0
        row = _raw_row(tmp_path, job["id"])
        assert row["disabled_reason"] == "false positive storm"


def test_cli_edit_disable_flip_without_reason_refuses(tmp_path, capsys):
    with _store(tmp_path):
        job = _create()
        assert cron_command(_edit_args(job["id"], disable=True)) == 1
        out = capsys.readouterr().out
        assert "disables without reasons are unattributable" in out
        assert "--reason" in out
        assert jobs.get_job(job["id"])["enabled"] is True


def test_cli_edit_disable_flip_with_reason_persists(tmp_path, capsys):
    with _store(tmp_path):
        job = _create()
        assert cron_command(_edit_args(job["id"], disable=True, reason="retiring the lane watch")) == 0
        out = capsys.readouterr().out
        assert "Updated job" in out
        row = _raw_row(tmp_path, job["id"])
        assert row["enabled"] is False
        assert row["paused_reason"] == "retiring the lane watch"
        assert row["disabled_reason"] == "retiring the lane watch"
        assert row["disabled_at"]


def test_cli_edit_reenable_with_reason_clears_disable_fields(tmp_path, capsys):
    with _store(tmp_path):
        job = _create()
        jobs.pause_job(job["id"], reason="ops audit")
        assert cron_command(_edit_args(job["id"], disable=False, enabled_reason="audit cleared")) == 0
        row = _raw_row(tmp_path, job["id"])
        assert row["enabled"] is True
        assert row["disabled_reason"] is None
        assert row["disabled_at"] is None
        assert row.get("enabled_reason") == "audit cleared"


def test_cli_edit_schedule_only_unchanged_enabled(tmp_path, capsys):
    """A plain edit (no lifecycle flip) must not require or invent a reason."""
    with _store(tmp_path):
        job = _create()
        assert cron_command(_edit_args(job["id"], schedule="every 2h")) == 0
        row = _raw_row(tmp_path, job["id"])
        assert row["enabled"] is True
        assert row["schedule_display"] == "every 120m"
        assert "disabled_reason" not in row


# ---------------------------------------------------------------------------
# Backward compat: legacy rows load unchanged.
# ---------------------------------------------------------------------------

def test_legacy_jobs_json_without_disable_fields_loads_unchanged(tmp_path):
    """A pre-change jobs.json row has none of the new fields and loads as-is."""
    from datetime import datetime, UTC

    legacy = [{
        "id": "f586101aac06",
        "name": "ceo-lane-watchdog",
        "prompt": "watch the lane",
        "skills": [],
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        "schedule_display": "every 60m",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "next_run_at": datetime(2030, 1, 1, tzinfo=UTC).isoformat(),
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "last_delivery_error": None,
        "created_at": datetime(2025, 1, 1, tzinfo=UTC).isoformat(),
        "deliver": "local",
    }]
    store = tmp_path / "cron"
    store.mkdir(parents=True)
    (store / "jobs.json").write_text(json.dumps({"jobs": legacy, "updated_at": None}))
    with jobs.use_cron_store(tmp_path):
        loaded = jobs.load_jobs()
        assert len(loaded) == 1
        row = loaded[0]
        assert row["id"] == "f586101aac06"
        assert row["enabled"] is True
        assert row["state"] == "scheduled"
        assert "disabled_reason" not in row  # not invented for untouched rows
        assert "disabled_at" not in row


def test_cli_pause_refusal_names_the_motivating_pattern(tmp_path, capsys):
    """The refusal message names the pattern: 'disables without reasons are
    unattributable'."""
    with _store(tmp_path):
        job = _create()
        cron_command(_pause_args(job["id"]))
        out = capsys.readouterr().out
        assert "disables without reasons are unattributable" in out


# ---------------------------------------------------------------------------
# Real-parser end to end: argparse wiring (dest names, flag/positional) plus
# cmd_cron, the same way tests/hermes_cli/test_cron.py drives the edit path.
# ---------------------------------------------------------------------------

def _built_parser():
    import argparse
    from hermes_cli.subcommands.cron import build_cron_parser
    parser = argparse.ArgumentParser(prog="hermes")
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=cron_command)
    return parser


def test_real_parser_pause_positional_and_flag(tmp_path, capsys):
    with _store(tmp_path):
        job = _create()
        parser = _built_parser()
        # Positional reason: hermes cron pause <id> "why"
        args = parser.parse_args(["cron", "pause", job["id"], "storm of false positives"])
        assert cron_command(args) == 0
        row = _raw_row(tmp_path, job["id"])
        assert row["disabled_reason"] == "storm of false positives"
        # Flag reason on a second job: hermes cron pause <id> --reason "why"
        jobs.resume_job(job["id"])
        args = parser.parse_args(["cron", "pause", job["id"], "--reason", "flag form"])
        assert cron_command(args) == 0
        assert _raw_row(tmp_path, job["id"])["disabled_reason"] == "flag form"
        # Reason-less through the real parser still refuses.
        args = parser.parse_args(["cron", "pause", job["id"]])
        assert cron_command(args) == 1
        assert "disables without reasons are unattributable" in capsys.readouterr().out


def test_real_parser_edit_disable_gate(tmp_path, capsys):
    with _store(tmp_path):
        job = _create()
        parser = _built_parser()
        assert cron_command(parser.parse_args(["cron", "edit", job["id"], "--disable"])) == 1
        assert jobs.get_job(job["id"])["enabled"] is True
        assert "disables without reasons are unattributable" in capsys.readouterr().out
        args = parser.parse_args(
            ["cron", "edit", job["id"], "--disable", "--reason", "retiring the watch"])
        assert cron_command(args) == 0
        row = _raw_row(tmp_path, job["id"])
        assert row["enabled"] is False
        assert row["disabled_reason"] == "retiring the watch"


def test_preexisting_disable_without_reason_is_never_backfilled(tmp_path):
    """Incident shape f586101aac06: a job already disabled with NO recorded reason.

    The record must stay honest as a record — the gate requires reasons going
    forward but never invents history: no unrelated edit back-fills a
    disabled_reason onto a pre-existing reason-less disable. The historical
    mutation is made on disk between two store lifetimes, exactly like a
    legacy file the gate never saw."""
    with _store(tmp_path):
        legacy = jobs.create_job(prompt="heartbeat", schedule="*/20 * * * *",
                                 name="heartbeat")
        neighbor = jobs.create_job(prompt="other", schedule="*/5 * * * *")
    # Out-of-band historical write: disabled before the gate existed, no reason.
    store_file = tmp_path / "cron" / "jobs.json"
    data = json.loads(store_file.read_text())
    rows = data["jobs"] if isinstance(data, dict) else data
    for row in rows:
        if row["id"] == legacy["id"]:
            row["enabled"] = False
            row.pop("disabled_reason", None)
            row.pop("disabled_at", None)
            row.pop("paused_reason", None)
    store_file.write_text(json.dumps(data))
    with _store(tmp_path):
        row = _raw_row(tmp_path, legacy["id"])
        assert row["enabled"] is False
        assert "disabled_reason" not in row and "paused_reason" not in row
        # An unrelated schedule edit neither refuses nor invents a reason.
        args = _built_parser().parse_args(
            ["cron", "edit", legacy["id"], "--schedule", "10m"])
        assert cron_command(args) == 0
        row = _raw_row(tmp_path, legacy["id"])
        assert row["enabled"] is False
        assert "disabled_reason" not in row  # history not back-filled
        # A NEW disable through the gate still demands a reason — the historical
        # gap does not dilute the forward rule.
        args = _built_parser().parse_args(["cron", "pause", neighbor["id"]])
        assert cron_command(args) == 1
