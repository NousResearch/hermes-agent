"""Consumed native handoffs and restart notices must not make full backups incomplete."""

from argparse import Namespace
from pathlib import Path
import sqlite3
import zipfile

from hermes_cli import backup


def _cron_tree(home: Path) -> tuple[set[str], list[Path]]:
    durable = set()
    handoffs = []
    for prefix in (Path(), Path("profiles/research")):
        cron = home / prefix / "cron"
        scratch = cron / "external-workers"
        scratch.mkdir(parents=True)
        marker = home / prefix / ".restart_pending.json"
        marker.write_text('{"delivered_targets": []}', encoding="utf-8")
        handoffs.append(marker)
        for name in ("attempt.json", "attempt.ready", "attempt.ready.tmp123", "attempt.stderr"):
            path = scratch / name
            path.write_text("synthetic handoff", encoding="utf-8")
            handoffs.append(path)
        for name in ("jobs.json", "output/job/run.stderr"):
            path = cron / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic durable data", encoding="utf-8")
            durable.add(path.relative_to(home).as_posix())
        for name in ("executions.db", "deliveries.db"):
            path = cron / name
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE evidence (value TEXT)")
                db.execute("INSERT INTO evidence VALUES ('synthetic')")
            durable.add(path.relative_to(home).as_posix())
    for name in ("skills/example/cron/external-workers/notes.stderr",
                 "skills/example/.restart_pending.json",
                 "profiles/research/skills/example/.restart_pending.json"):
        owner_path = home / name
        owner_path.parent.mkdir(parents=True, exist_ok=True)
        owner_path.write_text("owner artifact", encoding="utf-8")
        durable.add(owner_path.relative_to(home).as_posix())
    return durable, handoffs


def test_iterator_excludes_only_profile_root_cron_handoffs(tmp_path):
    home = tmp_path / "home"
    durable, _ = _cron_tree(home)
    selected = {rel.as_posix() for _, rel in backup._iter_backup_files(home, tmp_path / "out.zip")}
    assert selected == durable


def test_full_backup_succeeds_when_handoffs_disappear_after_scan(tmp_path, monkeypatch):
    home = tmp_path / "home"
    durable, handoffs = _cron_tree(home)
    original_walk = backup._iter_backup_files

    def scan_then_consume(*args, **kwargs):
        selected = list(original_walk(*args, **kwargs))
        for path in handoffs:
            path.unlink()
        return iter(selected)

    monkeypatch.setattr(backup, "_iter_backup_files", scan_then_consume)
    monkeypatch.setattr(backup, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(backup, "_collect_external_entries", lambda: ([], set()))
    output = tmp_path / "out.zip"
    assert backup.run_backup(Namespace(output=str(output), keep=0)) is True
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == durable
