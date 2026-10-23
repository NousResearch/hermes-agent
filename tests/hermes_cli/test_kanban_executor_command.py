"""Config-driven executor for thin-dispatcher kanban profiles.

A profile that sets ``kanban.executor_command`` is run directly by the
dispatcher (with the worker env) instead of spawning an LLM worker client.
"""

import sys

from hermes_cli import kanban_db_dispatch as kbd


def _write_profile(tmp_path, executor_command=None):
    d = tmp_path / "prof"
    d.mkdir()
    lines = ["kanban:"]
    if executor_command:
        lines.append(f"  executor_command: {executor_command}")
    lines.append("  failure_limit: 3")
    (d / "config.yaml").write_text("\n".join(lines) + "\n")
    return d


def test_executor_command_resolves_to_split_argv(tmp_path):
    d = _write_profile(tmp_path, "/venv/python /scripts/cursor-worker.py")
    assert kbd._profile_executor_command(str(d)) == [
        "/venv/python",
        "/scripts/cursor-worker.py",
    ]


def test_executor_command_quotes(tmp_path):
    d = _write_profile(tmp_path, '/venv/python /scripts/worker.py "arg with space"')
    assert kbd._profile_executor_command(str(d)) == [
        "/venv/python",
        "/scripts/worker.py",
        "arg with space",
    ]


def test_no_executor_returns_none(tmp_path):
    d = _write_profile(tmp_path)
    assert kbd._profile_executor_command(str(d)) is None


def test_missing_profile_returns_none(tmp_path):
    assert kbd._profile_executor_command(str(tmp_path / "nope")) is None


def test_worker_argv_falls_back_to_llm_client(tmp_path):
    d = _write_profile(tmp_path)
    assert kbd._worker_argv(object(), "default", str(d)) == [
        sys.executable,
        "-m",
        "hermes_cli.kanban_worker_client",
    ]


def test_worker_argv_uses_executor_when_configured(tmp_path):
    d = _write_profile(tmp_path, "/venv/python /scripts/cursor-worker.py")
    assert kbd._worker_argv(object(), "cursor-coder", str(d)) == [
        "/venv/python",
        "/scripts/cursor-worker.py",
    ]
