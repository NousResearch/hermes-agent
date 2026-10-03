"""Write-time redaction behavior for Kanban worker logs."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from io import BytesIO

from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_log


class _ChunkedReader:
    """A byte source that returns each supplied chunk in a separate read."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    def read(self, _size: int = -1) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


def test_worker_log_file_is_owner_only(tmp_path):
    log_path = tmp_path / "logs" / "worker.log"

    with kanban_worker_log.open_worker_log_file(log_path) as log_file:
        log_file.write(b"hello\n")

    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
    assert log_path.read_bytes() == b"hello\n"


def test_worker_log_redacts_secret_split_across_pipe_chunks():
    secret = b"sk-" + b"A" * 70000
    source = _ChunkedReader([b"prefix " + secret[:65530], secret[65530:] + b" suffix\n"])
    destination = BytesIO()

    kanban_worker_log.copy_redacted_worker_log_stream(source, destination)

    written = destination.getvalue()
    assert secret not in written
    assert b"prefix" in written
    assert b"suffix" in written


def test_worker_log_redacts_authorization_header_split_across_pipe_chunks():
    value = b"A" * 70000
    source = _ChunkedReader([b"Authorization: Bearer " + value[:65520], value[65520:] + b"\n"])
    destination = BytesIO()

    kanban_worker_log.copy_redacted_worker_log_stream(source, destination)

    assert value not in destination.getvalue()


def test_worker_log_forces_redaction_when_global_redaction_is_disabled(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    secret = b"sk-" + b"B" * 70000
    destination = BytesIO()

    kanban_worker_log.copy_redacted_worker_log_stream(BytesIO(secret + b"\n"), destination)

    assert secret not in destination.getvalue()


def test_worker_log_wrapper_redacts_child_output(tmp_path):
    log_path = tmp_path / "worker.log"

    returncode = kanban_worker_log.run_worker_with_redacted_log(
        log_path,
        [sys.executable, "-c", "print('token=sk-' + 'C' * 80)"],
    )

    assert returncode == 0
    assert b"C" * 80 not in log_path.read_bytes()
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


def test_worker_log_rotation_preserves_owner_only_mode(tmp_path):
    log_path = tmp_path / "worker.log"
    log_path.write_text("first log\n", encoding="utf-8")
    os.chmod(log_path, 0o644)

    kbd._rotate_worker_log(log_path, max_bytes=1, backup_count=2)

    first = tmp_path / "worker.log.1"
    assert stat.S_IMODE(first.stat().st_mode) == 0o600

    log_path.write_text("second log\n", encoding="utf-8")
    os.chmod(log_path, 0o644)
    kbd._rotate_worker_log(log_path, max_bytes=1, backup_count=2)

    assert stat.S_IMODE((tmp_path / "worker.log.2").stat().st_mode) == 0o600
    assert stat.S_IMODE(first.stat().st_mode) == 0o600


def test_worker_log_main_requires_a_child_command(tmp_path):
    assert kanban_worker_log.main([str(tmp_path / "worker.log")]) == 2


def test_worker_log_wrapper_propagates_child_exit_code(tmp_path):
    log_path = tmp_path / "worker.log"

    returncode = kanban_worker_log.run_worker_with_redacted_log(
        log_path,
        [sys.executable, "-c", "raise SystemExit(23)"],
    )

    assert returncode == 23
