from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from pathlib import Path

import yaml


def _database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        INSERT INTO sessions VALUES ('s', '-10077', 'group');
        INSERT INTO messages VALUES
        (1, 's', 'assistant', 'agent ready', 100, 'm1', 1, 0);
        """
    )
    connection.commit()
    connection.close()


def test_migrate_apply_stops_v1_backs_up_installs_and_starts_v2(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / ".hermes"
    curator = root / "profiles" / "memory-curator"
    curator.mkdir(parents=True)
    (curator / "config.yaml").write_text(
        """
mcp_servers:
  memory-forest-read:
    command: /bin/true
  memory-forest-curator-draft:
    command: /bin/true
plugins:
  enabled: []
""",
        encoding="utf-8",
    )
    (curator / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=fake-token\nHEGI_GROUP_CHAT_ID=-10077\n",
        encoding="utf-8",
    )
    (root / ".env").write_text(
        "TELEGRAM_ALLOWED_USERS=77\n", encoding="utf-8"
    )
    _database(root / "state.db")
    _database(root / "profiles" / "codex-test" / "state.db")
    _database(root / "profiles" / "heclaude-test" / "state.db")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    watcher = bin_dir / "hegi-memory-watch-loop"
    watcher.write_text(
        "#!/usr/bin/env bash\ntrap 'exit 0' TERM INT\nwhile true; do sleep 1; done\n",
        encoding="utf-8",
    )
    watcher.chmod(0o700)
    process = subprocess.Popen([str(watcher)])
    v1_state = root / "hegi-watch"
    v1_state.mkdir()
    (v1_state / "loop.pid").write_text(str(process.pid), encoding="ascii")
    (v1_state / "state.json").write_text('{"legacy": true}\n', encoding="utf-8")

    environment = os.environ.copy()
    environment.pop("HERMES_HOME", None)
    environment["HOME"] = str(tmp_path)
    try:
        result = subprocess.run(
            [
                str(repo / "hegi" / "scripts" / "migrate_v1.sh"),
                "--apply",
                "--no-systemd",
            ],
            cwd=repo,
            env=environment,
            text=True,
            capture_output=True,
            timeout=40,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        process.wait(timeout=5)
        assert list((root / "backups").glob("hegi-watch-*"))
        config = yaml.safe_load((curator / "hegi" / "config.yaml").read_text())
        assert config["enabled"] is True
        assert config["telegram"]["enabled"] is True
        assert (curator / "hegi" / "daemon.ready").is_file()
        assert "migration complete" in result.stdout
    finally:
        if process.poll() is None:
            process.kill()
        stop_environment = environment.copy()
        stop_environment["HERMES_HOME"] = str(curator)
        subprocess.run(
            [str(repo / "hegi" / "scripts" / "stop.sh")],
            env=stop_environment,
            capture_output=True,
            timeout=20,
            check=False,
        )
        time.sleep(0.05)
