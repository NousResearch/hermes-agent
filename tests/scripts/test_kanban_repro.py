from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli import kanban_db as kb


REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO_RUNNER = REPO_ROOT / "scripts" / "kanban_repro.py"


def _probe_script(tmp_path: Path) -> Path:
    probe = tmp_path / "restart_orphan_probe.py"
    probe.write_text(
        """
import json
from hermes_cli import kanban_db as kb

with kb.connect_closing() as conn:
    task_id = kb.create_task(conn, title="restart orphan repro", assignee="a")
print(json.dumps({"db": str(kb.kanban_db_path()), "task_id": task_id}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return probe


def _runner_env(configured_db: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_KANBAN_DB"] = str(configured_db)
    env["HERMES_KANBAN_BOARD"] = "default"
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO_ROOT),
        env.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)
    return env


def _synthetic_count(db_path: Path) -> int:
    with kb.connect(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM tasks "
            "WHERE title = 'restart orphan repro' AND assignee = 'a'"
        ).fetchone()[0]


def test_repro_runner_uses_temporary_db_instead_of_inherited_live_board(
    tmp_path: Path,
) -> None:
    live_db = tmp_path / "production" / "kanban.db"
    with kb.connect(live_db) as conn:
        kb.create_task(conn, title="real production task", assignee="coder")
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    result = subprocess.run(
        [sys.executable, str(REPRO_RUNNER), str(_probe_script(tmp_path))],
        cwd=REPO_ROOT,
        env=_runner_env(live_db),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(payload["db"]).resolve() != live_db.resolve()
    assert "hermes-kanban-repro-" in payload["db"]
    with kb.connect(live_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
    assert _synthetic_count(live_db) == 0


def test_repro_runner_refuses_configured_db_without_dangerous_opt_in(
    tmp_path: Path,
) -> None:
    live_db = tmp_path / "production" / "kanban.db"
    with kb.connect(live_db) as conn:
        kb.create_task(conn, title="real production task", assignee="coder")

    result = subprocess.run(
        [
            sys.executable,
            str(REPRO_RUNNER),
            "--db",
            str(live_db),
            str(_probe_script(tmp_path)),
        ],
        cwd=REPO_ROOT,
        env=_runner_env(live_db),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "refusing configured Kanban DB" in result.stderr
    assert "--allow-configured-db" in result.stderr
    assert _synthetic_count(live_db) == 0


def test_repro_runner_refuses_configured_db_even_before_it_exists(
    tmp_path: Path,
) -> None:
    live_db = tmp_path / "production" / "kanban.db"
    assert not live_db.exists()

    result = subprocess.run(
        [
            sys.executable,
            str(REPRO_RUNNER),
            "--db",
            str(live_db),
            str(_probe_script(tmp_path)),
        ],
        cwd=REPO_ROOT,
        env=_runner_env(live_db),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "refusing configured Kanban DB" in result.stderr
    assert not live_db.exists()


def test_repro_runner_allows_configured_db_only_with_explicit_dangerous_opt_in(
    tmp_path: Path,
) -> None:
    live_db = tmp_path / "production" / "kanban.db"

    result = subprocess.run(
        [
            sys.executable,
            str(REPRO_RUNNER),
            "--db",
            str(live_db),
            "--allow-configured-db",
            str(_probe_script(tmp_path)),
        ],
        cwd=REPO_ROOT,
        env=_runner_env(live_db),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _synthetic_count(live_db) == 1
