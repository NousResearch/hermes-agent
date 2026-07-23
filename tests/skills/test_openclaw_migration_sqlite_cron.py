"""Regression tests for #64138 — hermes claw migrate must detect OpenClaw
cron jobs from the SQLite source-of-truth.

Background: OpenClaw moved its live cron store from a flat
``~/.openclaw/cron/jobs.json`` file to a SQLite database at
``~/.openclaw/state/openclaw.sqlite``, table ``cron_jobs``. The
migrator at
``optional-skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py``
still only reads the legacy JSON config and ``cron/`` directory, so it
silently reports "No cron configuration found" even when enabled cron
jobs exist.

These tests assert the new SQLite-source path detects and surfaces
those jobs so the operator at least knows they exist (and gets a record
in the migration report explaining how to recreate them via
``hermes cron``).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


# We import via the same loader pattern the existing tests use.
SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "migration"
    / "openclaw-migration"
    / "scripts"
    / "openclaw_to_hermes.py"
)


def _load_module():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("openclaw_to_hermes", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_source_with_sqlite_cron(tmp_path: Path, *, enabled_count: int = 2,
                                  disabled_count: int = 1) -> Path:
    """Build a fake OpenClaw source dir with a state/openclaw.sqlite
    containing both enabled and disabled cron_jobs rows."""
    source = tmp_path / ".openclaw"
    source.mkdir()
    (source / "openclaw.json").write_text(json.dumps({"channels": {}}), encoding="utf-8")

    state_dir = source / "state"
    state_dir.mkdir()
    db_path = state_dir / "openclaw.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE cron_jobs (
                job_id          TEXT PRIMARY KEY,
                name            TEXT,
                enabled         INTEGER NOT NULL,
                schedule_kind   TEXT,
                schedule_expr   TEXT,
                schedule_tz     TEXT,
                payload_message TEXT,
                payload_model   TEXT,
                delivery_channel TEXT,
                delivery_to     TEXT,
                job_json        TEXT,
                state_json      TEXT
            );
        """)
        for i in range(enabled_count):
            conn.execute(
                "INSERT INTO cron_jobs "
                "(job_id, name, enabled, schedule_kind, schedule_expr, schedule_tz, "
                " payload_message, payload_model, delivery_channel, delivery_to) "
                "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"job-en-{i}",
                    f"Enabled job {i}",
                    "cron",
                    "*/15 * * * *",
                    "UTC",
                    f"prompt {i}",
                    "deepseek-v3",
                    "telegram",
                    "@user",
                ),
            )
        for i in range(disabled_count):
            conn.execute(
                "INSERT INTO cron_jobs "
                "(job_id, name, enabled, schedule_kind, schedule_expr, schedule_tz, "
                " payload_message, payload_model, delivery_channel, delivery_to) "
                "VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"job-dis-{i}",
                    f"Disabled job {i}",
                    "cron",
                    "0 9 * * *",
                    "UTC",
                    f"disabled-prompt {i}",
                    "deepseek-v3",
                    "slack",
                    "#channel",
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return source


def _run_migrate_cron_jobs(tmp_path: Path, source: Path):
    """Run the migrator with only the cron-jobs option selected."""
    mod = _load_module()
    target = tmp_path / ".hermes"
    target.mkdir()
    output_dir = target / "migration-report"

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=output_dir,
        selected_options={"cron-jobs"},
    )
    return mod, migrator.migrate()


def test_sqlite_cron_jobs_are_detected_not_silent_skip(tmp_path: Path):
    """#64138: a source with only state/openclaw.sqlite cron jobs (no flat
    JSON, no cron/ directory) must be detected. The migrator currently
    records 'No cron configuration found' which is the bug."""
    source = _make_source_with_sqlite_cron(tmp_path, enabled_count=2, disabled_count=1)
    _mod, report = _run_migrate_cron_jobs(tmp_path, source)

    cron_items = [item for item in report["items"] if item["kind"] == "cron-jobs"]
    assert cron_items, "cron-jobs section must produce at least one record"

    reasons_blob = "\n".join(item.get("reason", "") for item in cron_items)
    assert "No cron configuration found" not in reasons_blob, (
        "the bug-shape (silent 'No cron configuration found') must NOT be "
        "the only record when the SQLite source has rows"
    )


def test_sqlite_cron_jobs_record_all_jobs_with_status(tmp_path: Path):
    """Both enabled and disabled jobs should be surfaced in the report
    so the operator can decide which to recreate via 'hermes cron'."""
    source = _make_source_with_sqlite_cron(tmp_path, enabled_count=2, disabled_count=1)
    _mod, report = _run_migrate_cron_jobs(tmp_path, source)

    cron_items = [item for item in report["items"] if item["kind"] == "cron-jobs"]
    # At least the enabled jobs should appear in the reasons blob.
    reasons_blob = "\n".join(item.get("reason", "") for item in cron_items)

    assert "Enabled job 0" in reasons_blob or "job-en-0" in reasons_blob, (
        "enabled job ID 0 must be visible in the cron-jobs reasons"
    )
    # F-C catch on the first revision: the report must include a copy-
    # pasteable `hermes cron add ...` command for each row, so the
    # operator can recreate without parsing the schedule by hand.
    assert "hermes cron add" in reasons_blob, (
        "each SQLite-sourced cron job must surface a copy-pasteable "
        "`hermes cron add ...` recreate command (F-C catch)"
    )


def test_sqlite_cron_jobs_do_not_break_when_no_flat_cron_dir(tmp_path: Path):
    """Regression guard: the fix must not require the flat cron/ directory
    to also exist. The SQLite source is the authoritative source for
    newer OpenClaw installs."""
    source = _make_source_with_sqlite_cron(tmp_path, enabled_count=1, disabled_count=0)
    # No source/cron/ directory created — only state/openclaw.sqlite.
    assert not (source / "cron").exists()

    _mod, report = _run_migrate_cron_jobs(tmp_path, source)

    cron_items = [item for item in report["items"] if item["kind"] == "cron-jobs"]
    reasons_blob = "\n".join(item.get("reason", "") for item in cron_items)
    assert "No cron configuration found" not in reasons_blob, (
        "SQLite-only source must not produce the legacy 'No cron "
        "configuration found' message"
    )


def test_no_sqlite_db_keeps_legacy_behavior(tmp_path: Path):
    """Regression guard: if state/openclaw.sqlite doesn't exist, the
    legacy behavior (config.cron + cron/ directory) must still work."""
    mod = _load_module()
    source = tmp_path / ".openclaw"
    source.mkdir()
    target = tmp_path / ".hermes"
    target.mkdir()
    output_dir = target / "migration-report"

    (source / "openclaw.json").write_text(
        json.dumps({
            "cron": {
                "version": 1,
                "jobs": [{"id": "legacy-job-1", "name": "legacy"}],
            }
        }),
        encoding="utf-8",
    )

    migrator = mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=False,
        migrate_secrets=False,
        output_dir=output_dir,
        selected_options={"cron-jobs"},
    )
    report = migrator.migrate()

    cron_items = [item for item in report["items"] if item["kind"] == "cron-jobs"]
    assert cron_items, "legacy config.cron path must still be detected"
    # The legacy path produces an "archived" record; the SQLite path
    # must NOT be needed for this to work. The reason wording is
    # implementation-specific — we just check the bug-shape is absent
    # and the section produced at least one record.
    reasons_blob = "\n".join(item.get("reason", "") for item in cron_items)
    assert "No cron configuration found" not in reasons_blob, (
        "legacy config.cron path must NOT silently report 'No cron "
        "configuration found' when jobs are defined in openclaw.json"
    )
