from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path

import yaml

from hegi.bootstrap import discover_environment


def _source_database(path: Path, chat_id: str) -> None:
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
        """
    )
    connection.execute(
        "INSERT INTO sessions VALUES ('s', ?, 'group')", (chat_id,)
    )
    connection.execute(
        """
        INSERT INTO messages VALUES
        (1, 's', 'assistant', 'configured agent', 100, 'a-1', 1, 0)
        """
    )
    connection.commit()
    connection.close()


def _deployment(tmp_path: Path) -> tuple[Path, Path]:
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
        "TELEGRAM_BOT_TOKEN=fake-token\n"
        "HEGI_GROUP_CHAT_ID=-10042\n",
        encoding="utf-8",
    )
    (root / ".env").write_text(
        "TELEGRAM_ALLOWED_USERS=4242\n", encoding="utf-8"
    )
    _source_database(root / "state.db", "-10042")
    _source_database(root / "profiles" / "codex-test" / "state.db", "-10042")
    _source_database(
        root / "profiles" / "heclaude-test" / "state.db", "-10042"
    )
    return root, curator


def test_discovery_finds_operational_telegram_profile_and_agent_databases(tmp_path):
    root, curator = _deployment(tmp_path)
    result = discover_environment(hermes_root=root, runtime_home=curator)

    assert result.chat_id == "-10042"
    assert result.professor_user_ids == ("4242",)
    assert result.runtime_home == curator.resolve()
    assert {agent.name for agent in result.agents} == {
        "HeHe",
        "HeCo",
        "HeClaude",
    }


def test_install_script_writes_enabled_config_and_gateway_plugin(tmp_path):
    root, curator = _deployment(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("HERMES_HOME", None)
    environment["HOME"] = str(tmp_path)
    result = subprocess.run(
        [
            str(repo / "hegi" / "scripts" / "install.sh"),
            "--no-systemd",
            "--hermes-root",
            str(root),
            "--runtime-home",
            str(curator),
        ],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    config = yaml.safe_load((curator / "hegi" / "config.yaml").read_text())
    assert config["enabled"] is True
    assert config["telegram"]["enabled"] is True
    assert config["telegram"]["chat_id"] == "-10042"
    assert config["memory"]["professor_user_ids"] == ["4242"]
    assert len(config["agents"]) == 3
    assert (curator / "plugins" / "hegi-telegram" / "__init__.py").is_file()
    hermes_config = yaml.safe_load((curator / "config.yaml").read_text())
    assert "hegi-telegram" in hermes_config["plugins"]["enabled"]
    assert '"errors": []' in result.stdout
