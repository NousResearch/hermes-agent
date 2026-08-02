"""P13 isolation proof for scripts/system_health_daily.py.

Verifies:
- HERMES_HOME parameterisation: HERMES resolves under HERMES_HOME.
- --dry-run skips every network/subprocess probe (systemctl, sudo docker
  inspect, curl, free, df, pgrep) and suppresses every write path
  (create_task kanban filing, save_state, LOG_DIR json write).
- Read-only sqlite scans still run (drift detection) but nothing is filed.
- import-safe under a fake HERMES_HOME.
"""
import importlib.util
import json
import os
import sqlite3
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "system_health_daily.py"


def _load_module(monkeypatch, fake_home: Path, dry_run: bool = False):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    argv = ["system_health_daily.py"] + (["--dry-run"] if dry_run else [])
    monkeypatch.setattr(sys, "argv", argv)
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("shl_under_test", str(SCRIPT))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


def test_hermes_home_resolves_under_env(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert str(mod.HERMES).startswith(str(fake_home))


def test_dry_run_flag_set(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    assert mod._DRY_RUN is True


def test_dry_run_create_task_returns_none(monkeypatch, fake_home):
    """create_task under --dry-run must never file a kanban task."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    result = mod.create_task("title", "body", "wesker", "P1", "key")
    assert result is None


def test_dry_run_save_state_writes_nothing(monkeypatch, fake_home):
    """save_state under --dry-run must not write the state file."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    mod.save_state({"issues": {}, "last_run": "now"})
    state_file = fake_home / "governance" / "system-health-state.json"
    assert not state_file.exists(), "dry-run wrote the state file"


def test_dry_run_probes_skip_subprocess(monkeypatch, fake_home):
    """Every probe that shells out to a subprocess must return None/skip
    under --dry-run so docker/systemctl/network are never touched."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    assert mod.check_gateway() is None
    assert mod.check_web_backends() is None
    assert mod.check_memory() is None
    assert mod.check_disk() is None
    assert mod.check_swap() is None
    assert mod.check_discord_bots() is None


def test_web_backends_use_bounded_http_not_docker(monkeypatch, fake_home):
    """The web probe must not block on root-only Docker access."""
    mod = _load_module(monkeypatch, fake_home)
    calls = []

    class Response:
        def __init__(self, body):
            self.status = 200
            self._body = body

        def read(self, _limit=-1):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(url, timeout):
        assert timeout <= 10
        if ":8082/" in url:
            return Response(b'{"results": [{"title": "ok"}]}')
        return Response(b"ok")

    def fake_run(cmd, timeout=20):
        calls.append(cmd)
        assert cmd[0] != "sudo"
        return 0, "ok", ""

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod, "run", fake_run)
    assert mod.check_web_backends() is None
    assert len(calls) == 1
def test_check_kanban_accepts_pathlike_and_string_paths(monkeypatch, fake_home, tmp_path):
    """Board paths are normalised before Path methods are used.

    Covers the resolver's normal Path output plus absolute, relative, and
    network-style string paths accepted at the compatibility boundary.
    """
    db_path = tmp_path / "kanban.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE task_events (
                task_id TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            """
        )

    monkeypatch.chdir(tmp_path)
    board_paths = {
        "path": db_path,
        "absolute-string": str(db_path),
        "relative-string": db_path.name,
        "network-string": f"//{db_path.as_posix().lstrip('/')}",
    }
    monkeypatch.setitem(
        sys.modules,
        "_board_compat",
        types.SimpleNamespace(build_board_db_map=lambda _slugs: board_paths),
    )

    mod = _load_module(monkeypatch, fake_home)
    assert mod.check_kanban() is None


def test_dry_run_main_no_log_write(monkeypatch, fake_home):
    """main() under --dry-run must not write a JSON log to LOG_DIR."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    rc = mod.main()
    assert rc == 0
    logboard = fake_home / "governance" / "logboard"
    # No json log should have been written under the fake home.
    if logboard.exists():
        logs = list(logboard.glob("system-health-*.json"))
        assert logs == [], f"dry-run wrote log files: {logs}"


def test_cron_error_body_contains_every_failed_job(monkeypatch, fake_home):
    """The count in a cron-error alert must match every listed failure."""
    mod = _load_module(monkeypatch, fake_home)
    jobs_file = fake_home / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    jobs_file.write_text(
        json.dumps(
            {
                "jobs": [
                    {"name": f"failed-{i}", "enabled": True, "last_status": "error"}
                    for i in range(7)
                ]
                + [
                    {"name": "paused-error", "enabled": False, "last_status": "error"},
                    {"name": "healthy", "enabled": True, "last_status": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "JOBS_FILE", jobs_file)

    finding = mod.check_cron_errors()

    assert finding["title"] == "7 cron(s) failed last run"
    assert finding["body"].count("last_status error") == 7
    assert "paused-error" not in finding["body"]
    assert "healthy" not in finding["body"]
