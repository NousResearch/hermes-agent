"""Regression: cron runtime churn must not fail the daily backup.

2026-10-04: the daily Hermes zip reported "Backup incomplete" with 6 files that no longer
existed when the writer reached them — ``cron/.jobs_<rand>.tmp`` (the jobs.json atomic-write
temp) and five ``cron/external-workers/*.stderr`` handoff files. The scan lists files once and
the archive writes them over minutes, so any runtime file created and removed in that window
turned into a hard failure and a red systemd timer (earlier runs: 8-48 such files).

Two defences, both asserted here:
  1. the scan never selects those paths (``_is_ephemeral_cron_path``), and
  2. a file that still vanishes mid-write is reported as skipped, not as an error.
"""

import zipfile
from pathlib import Path


def _fake_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron" / "external-workers").mkdir(parents=True)
    (home / "config.yaml").write_text("model: {}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_ephemeral_cron_paths_are_excluded(tmp_path, monkeypatch):
    _fake_home(tmp_path, monkeypatch)
    from hermes_cli import backup as backup_mod

    assert backup_mod._should_exclude(Path("cron/.jobs_abc123.tmp"))
    assert backup_mod._should_exclude(Path("cron/external-workers/abc.stderr"))
    assert backup_mod._should_exclude(Path("profiles/rdw/cron/external-workers/abc.json"))

    # Durable cron state and unrelated look-alikes stay in the backup.
    assert not backup_mod._should_exclude(Path("cron/jobs.json"))
    assert not backup_mod._should_exclude(Path("cron/output/8ca/2026-10-04_08-00.md"))
    assert not backup_mod._should_exclude(Path("projects/external-workers/notes.md"))
    assert not backup_mod._should_exclude(Path("foo/.jobs_keep.tmp"))


def test_walk_skips_cron_runtime_files(tmp_path, monkeypatch):
    home = _fake_home(tmp_path, monkeypatch)
    (home / "cron" / "jobs.json").write_text("{}")
    (home / "cron" / ".jobs_deadbeef.tmp").write_text("{}")
    (home / "cron" / "external-workers" / "abc.stderr").write_text("boom")
    (home / "cron" / "external-workers" / "abc.json").write_text("{}")

    from hermes_cli import backup as backup_mod

    rels = {str(rel) for _, rel in backup_mod._iter_backup_files(home, tmp_path / "out.zip")}
    assert "cron/jobs.json" in rels
    assert not any(r.startswith("cron/external-workers/") for r in rels)
    assert not any(r.startswith("cron/.jobs_") for r in rels)


def test_vanished_file_is_skipped_not_an_error(tmp_path):
    from hermes_cli import backup as backup_mod

    real = tmp_path / "real.txt"
    real.write_text("hoi")
    gone = tmp_path / "gone.stderr"  # listed by the scan, deleted before the write

    errors, vanished = [], []
    out = tmp_path / "out.zip"
    with zipfile.ZipFile(out, "w") as zf:
        backup_mod._write_zip_entries(
            zf, [(real, Path("real.txt")), (gone, Path("gone.stderr"))], out,
            on_db_failure=lambda rel: errors.append(f"{rel}: db"),
            on_error=lambda rel, exc: errors.append(f"{rel}: {exc}"),
            on_progress=lambda i: None,
            track_bytes=True,
            on_vanished=lambda rel: vanished.append(str(rel)))

    assert errors == []
    assert vanished == ["gone.stderr"]
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["real.txt"]


def test_vanished_file_without_callback_is_a_plain_skip(tmp_path):
    from hermes_cli import backup as backup_mod

    errors = []
    out = tmp_path / "out.zip"
    with zipfile.ZipFile(out, "w") as zf:
        backup_mod._write_zip_entries(
            zf, [(tmp_path / "gone.txt", Path("gone.txt"))], out,
            on_db_failure=lambda rel: errors.append(f"{rel}: db"),
            on_error=lambda rel, exc: errors.append(f"{rel}: {exc}"),
            on_progress=lambda i: None,
            track_bytes=True)

    assert errors == []