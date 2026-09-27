"""Regression tests for the attempt-scoped cron worker drain signal (#125513).

A terminal executions.db row or the outer run's return value do not prove the
attempt's worker finished: a timed-out run hands teardown to the inner future
(``defer_teardown_to_running_worker``). The drain record must appear only
AFTER that deferred teardown completes, and stay queryable across processes.
"""

from __future__ import annotations

import json
import threading
from argparse import Namespace
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

from cron import worker_drain
from cron.scheduler_detached_worker import defer_teardown_to_running_worker
from hermes_cli.cron import cron_command


_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


def _drain_log(home: Path) -> Path:
    return home / "cron" / "worker-drains.jsonl"


def test_detached_worker_records_drain_only_after_future_completes(tmp_path, monkeypatch):
    """The issue's core case: outer run already failed (timeout), inner worker live.

    Drain stays unproven while the future runs, then flips to drained once the
    deferred teardown (finalize + agent teardown) has completed.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    future = Future()
    with patch("cron.scheduler._finalize_cron_session") as finalize, \
         patch("cron.scheduler._teardown_cron_agent") as teardown_agent:
        assert defer_teardown_to_running_worker(
            future, MagicMock(), MagicMock(), "drain-job", "drain job",
            "cron_drain-job_20260928", execution_id="exec-drain-1") is True

        # Outer run has terminalized (simulated); inner worker still live:
        # no drain record may exist — completion unproven.
        assert worker_drain.drain_status("exec-drain-1")["status"] == "unknown"

        future.set_result({"final_response": "late"})

        finalize.assert_called_once()
        teardown_agent.assert_called_once()
    status = worker_drain.drain_status("exec-drain-1")
    assert status["status"] == "drained"
    assert status["drained"] is True
    assert status["job_id"] == "drain-job"


def test_run_job_inline_teardown_records_drain(tmp_path, monkeypatch):
    """The non-deferred path: run_job's inline teardown emits the record."""
    from tests.cron.test_cleanup_timeout import HangingSessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    release = threading.Event()
    fake_db = HangingSessionDB(release)
    job = {"id": "drain-inline", "name": "test", "prompt": "hello"}

    try:
        with patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state_registry.acquire", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, _output, final_response, error = __import__(
                "cron.scheduler", fromlist=["run_job"]).run_job(
                job, execution_id="exec-inline-1")

        assert success is True and error is None
        status = worker_drain.drain_status("exec-inline-1")
        assert status["status"] == "drained"
        assert status["job_id"] == "drain-inline"
        # Durable on disk, not just in-process (cross-restart query).
        rows = [json.loads(ln) for ln in _drain_log(tmp_path).read_text().splitlines()]
        assert any(r["execution_id"] == "exec-inline-1" for r in rows)
    finally:
        release.set()


def test_unknown_ids_and_missing_log_read_as_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # No log file at all (fresh home / pre-signal install).
    assert worker_drain.drain_status("never-recorded")["status"] == "unknown"
    assert worker_drain.drain_status("")["status"] == "unknown"
    # Corrupted trailing line must not break the scan.
    _drain_log(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _drain_log(tmp_path).write_text("not json\n", encoding="utf-8")
    assert worker_drain.drain_status("still-unknown")["status"] == "unknown"


def test_cli_drain_exit_codes(tmp_path, monkeypatch, capsys):
    """`hermes cron drain` maps drained -> 0, unknown -> 1 (the deploy gate)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    worker_drain.record_drain("exec-cli-1", job_id="cli-job")
    rc = cron_command(Namespace(cron_command="drain", target="exec-cli-1", json=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["drained"] is True
    rc = cron_command(Namespace(cron_command="drain", target="missing-id", json=True))
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["drained"] is False
