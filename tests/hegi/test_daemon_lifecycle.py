from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest


def _empty_source(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, chat_id TEXT, chat_type TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            timestamp REAL, platform_message_id TEXT, active INTEGER,
            compacted INTEGER
        );
        """
    )
    connection.commit()
    connection.close()


def _runtime(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    home = tmp_path / "runtime"
    home.mkdir()
    env_path = home / ".env"
    env_path.write_text("TELEGRAM_BOT_TOKEN=fake-token\n", encoding="utf-8")
    first = tmp_path / "one.db"
    second = tmp_path / "two.db"
    _empty_source(first)
    _empty_source(second)
    hegi = home / "hegi"
    hegi.mkdir()
    (hegi / "config.yaml").write_text(
        f"""
enabled: true
state_db: "{hegi / 'state.db'}"
telegram:
  chat_id: "-1001"
  curator_env: "{env_path}"
  enabled: true
agents:
  - name: HeHe
    db_path: "{first}"
  - name: HeCo
    db_path: "{second}"
episode:
  quiet_minutes: 10
  max_gap_minutes: 30
  minimum_agents: 2
  minimum_messages: 4
daemon:
  poll_seconds: 1
archive:
  local_spool: "{hegi / 'archive'}"
memory:
  enabled: false
  auto_commit: false
  auto_draft: false
  require_professor_approval: true
  professor_user_ids: ["42"]
  default_project: research
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["HERMES_HOME"] = str(home)
    return home, environment


def _run(script: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


@pytest.mark.live_system_guard_bypass
def test_daemon_duplicate_start_status_stop_and_crash_recovery(tmp_path):
    home, environment = _runtime(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    start = repo / "hegi" / "scripts" / "start.sh"
    stop = repo / "hegi" / "scripts" / "stop.sh"
    status = repo / "hegi" / "scripts" / "status.sh"
    try:
        first = _run(start, environment)
        assert first.returncode == 0, first.stderr
        pidfile = home / "hegi" / "daemon.pid"
        first_pid = int(pidfile.read_text())

        duplicate = _run(start, environment)
        assert duplicate.returncode == 0
        assert "already running" in duplicate.stdout
        assert int(pidfile.read_text()) == first_pid

        state = _run(status, environment)
        assert state.returncode == 0, state.stderr
        assert json.loads(state.stdout)["daemon"]["alive"] is True

        os.kill(first_pid, signal.SIGKILL)
        for _ in range(50):
            try:
                os.kill(first_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)

        restarted = _run(start, environment)
        assert restarted.returncode == 0, restarted.stderr
        second_pid = int(pidfile.read_text())
        assert second_pid != first_pid
        assert (home / "hegi" / "daemon.ready").is_file()
    finally:
        _run(stop, environment)
    assert not (home / "hegi" / "daemon.pid").exists()
