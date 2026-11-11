"""`hermes logs` across a log rotation and under a named profile, driven through the CLI entry point."""

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from tests.waiting import wait_until

REPO_ROOT = Path(__file__).resolve().parents[2]


def _agent_log_handler(log: Path, max_bytes: int):
    """The rotating handler Hermes writes agent.log with, at a small rollover size."""
    from hermes_logging import _new_file_handler

    return _new_file_handler(
        log, level=logging.INFO, max_bytes=max_bytes, backup_count=2,
        formatter=logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )


def _log(handler, message: str) -> None:
    handler.handle(logging.LogRecord("run_agent", logging.INFO, __file__, 0, message, None, None))


def _run_cli(monkeypatch, capsys, *argv: str) -> str:
    from hermes_cli.main import main

    monkeypatch.setattr(sys, "argv", ["hermes", *argv])
    main()
    return capsys.readouterr().out


def test_follow_continues_into_the_new_file_after_the_handler_rolls_over(tmp_path):
    home = tmp_path / "home"
    log = home / "logs" / "agent.log"
    handler = _agent_log_handler(log, max_bytes=1024)
    _log(handler, "written before following")
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.main", "logs", "-f"], cwd=REPO_ROOT, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    stdout = proc.stdout
    assert stdout is not None
    output: list[str] = []
    reader = threading.Thread(target=lambda: output.extend(iter(stdout.readline, "")), daemon=True)
    reader.start()
    try:
        wait_until(lambda: any("Ctrl+C to stop" in line for line in output), what="the follow header")
        _log(handler, "written before rotation")
        wait_until(lambda: any("written before rotation" in line for line in output), what="the live line")
        _log(handler, "written after rotation " + "x" * 2048)  # crosses max_bytes: agent.log -> agent.log.1
        assert (home / "logs" / "agent.log.1").exists()
        wait_until(lambda: any("written after rotation" in line for line in output),
                   what="the first line of the new agent.log")
    finally:
        handler.close()
        proc.kill()
        proc.wait()
        reader.join()


def test_list_shows_the_rotated_copies(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    handler = _agent_log_handler(home / "logs" / "agent.log", max_bytes=1024)
    _log(handler, "first record " + "x" * 2048)
    _log(handler, "second record " + "x" * 2048)
    handler.close()
    assert (home / "logs" / "agent.log.1").exists()

    listing = _run_cli(monkeypatch, capsys, "logs", "list")

    assert "agent.log.1" in listing


def test_update_log_header_names_the_root_home_under_a_named_profile(tmp_path, monkeypatch, capsys):
    from hermes_constants import display_hermes_home

    root = tmp_path / "root"
    profile = root / "profiles" / "coder"
    (root / "logs").mkdir(parents=True)
    (profile / "logs").mkdir(parents=True)
    (root / "logs" / "update.log").write_text("=== hermes update started 2026-10-03T10:00:00 ===\n")
    monkeypatch.setenv("HERMES_HOME", str(profile))

    out = _run_cli(monkeypatch, capsys, "logs", "update", "-n", "5")

    assert f"--- {display_hermes_home(root)}/logs/update.log (last 5) ---" in out.splitlines()
