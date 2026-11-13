"""Write-time redaction behavior for Kanban worker logs."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from io import BytesIO

import pytest

from agent.redact import SECRET_HEADER_NAME_LIST
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

    kanban_worker_log.rotate_worker_log(log_path, max_bytes=1, backup_count=2)

    first = tmp_path / "worker.log.1"
    assert stat.S_IMODE(first.stat().st_mode) == 0o600

    log_path.write_text("second log\n", encoding="utf-8")
    os.chmod(log_path, 0o644)
    kanban_worker_log.rotate_worker_log(log_path, max_bytes=1, backup_count=2)

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


def test_worker_log_wrapper_preserves_unexpected_failure_traceback(tmp_path, monkeypatch):
    def fail_open(_log_path):
        raise RuntimeError("unexpected log implementation failure")

    monkeypatch.setattr(kanban_worker_log, "open_worker_log_file", fail_open)
    with pytest.raises(RuntimeError, match="unexpected log implementation failure"):
        kanban_worker_log.run_worker_with_redacted_log(tmp_path / "worker.log", [sys.executable])


def test_worker_log_bootstraps_before_registration_import(tmp_path):
    from pathlib import Path

    script = """
import runpy, sys
repo, log_path = sys.argv[1:]
sys.path.insert(0, repo)
class BootstrapRequired:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tools.kanban_tools':
            assert 'hermes_bootstrap' in sys.modules, 'registration imported before dependency activation'
sys.meta_path.insert(0, BootstrapRequired())
sys.argv = ['hermes_cli.kanban_worker_log', log_path, '--', sys.executable, '-c', "print('bootstrapped child')"]
runpy.run_module('hermes_cli.kanban_worker_log', run_name='__main__')
"""
    log_path = tmp_path / "worker.log"
    env = {key: value for key, value in os.environ.items() if not key.startswith("HERMES_KANBAN_")}
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2]), str(log_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert log_path.read_text().strip() == "bootstrapped child"


@pytest.mark.parametrize("kind", ["dsn", "jwt", "api-header"])
@pytest.mark.parametrize("chunking", ["whole", "every-byte", "all-splits", "long"])
def test_worker_log_matches_canonical_redaction_for_complete_records(kind, chunking, monkeypatch):
    from agent.redact import redact_terminal_output

    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    value = b"aB8x" * (17500 if chunking == "long" else 75)
    records = {
        "dsn": b"postgres://user:" + value + b"@host/db\n",
        "jwt": b"eyJ" + value + b".cD7z.cD7z\n",
        "api-header": b"x-api-key: " + value + b"\n",
    }
    record = records[kind]
    expected = redact_terminal_output(record.decode(), force=True).encode()
    assert value[10:-10] not in expected
    if chunking == "all-splits":
        sources = ([record[:cut], record[cut:]] for cut in range(1, len(record)))
    elif chunking == "every-byte":
        sources = ([record[i:i + 1] for i in range(len(record))],)
    elif chunking == "long":
        sources = ([record[:65530], record[65530:]],)
    else:
        sources = ([record],)

    for chunks in sources:
        destination = BytesIO()
        kanban_worker_log.copy_redacted_worker_log_stream(_ChunkedReader(chunks), destination)
        written = destination.getvalue()
        assert value[10:-10] not in written
        assert value[-36:-4] not in written
        assert written == expected


@pytest.mark.parametrize("key_kind", ["", "RSA ", "EC ", "ENCRYPTED "])
def test_worker_log_preserves_public_output_after_every_pem_delimiter_split(key_kind):
    begin = f"-----BEGIN {key_kind}PRIVATE KEY-----".encode()
    end = f"-----END {key_kind}PRIVATE KEY-----".encode()
    before = b"public before\n"
    body = b"\nprivate-material\n"
    after = b"\npublic after\n[kanban-worker-exit] rc=0\n"
    record = before + begin + body + end + after
    expected = before + b"[REDACTED PRIVATE KEY]" + after
    cuts = [len(before) + cut for cut in range(1, len(begin) + 1)]
    cuts += [len(before + begin + body) + cut for cut in range(1, len(end) + 1)]
    sources = [[record], [record[i:i + 1] for i in range(len(record))]]
    sources += [[record[:cut], record[cut:]] for cut in cuts]

    for chunks in sources:
        destination = BytesIO()
        kanban_worker_log.copy_redacted_worker_log_stream(_ChunkedReader(chunks), destination)
        assert destination.getvalue() == expected


def test_worker_log_flushes_public_records_after_authorization_before_eof():
    value = b"dE9y" * 75
    public = b"public progress: ready\n" * 20
    destination = BytesIO()

    class CheckingReader(_ChunkedReader):
        def read(self, size=-1):
            if not self._chunks:
                assert public in destination.getvalue(), "public output withheld until EOF"
                assert value[10:-10] not in destination.getvalue()
            return super().read(size)

    source = CheckingReader([b"Authorization: Bearer " + value + b"\n", public])
    kanban_worker_log.copy_redacted_worker_log_stream(source, destination)


@pytest.mark.parametrize("header", SECRET_HEADER_NAME_LIST)
@pytest.mark.parametrize("whitespace", [b"\n", b" \r\n \t\n"])
def test_worker_log_retains_secret_headers_split_before_colon(header, whitespace):
    from agent.redact import redact_terminal_output

    value = b"dE9y" * 75
    public = b"public progress: ready\n[kanban-worker-exit] rc=0\n"
    record = header.upper().encode() + whitespace + b": " + value + b"\n" + public
    expected = redact_terminal_output(record.decode(), force=True).encode()
    assert value[10:-10] not in expected

    for chunks in ([record], [record[i:i + 1] for i in range(len(record))]):
        destination = BytesIO()

        class CheckingReader(_ChunkedReader):
            def read(self, size=-1):
                if not self._chunks:
                    assert public in destination.getvalue(), "public output withheld until EOF"
                return super().read(size)

        kanban_worker_log.copy_redacted_worker_log_stream(CheckingReader(chunks), destination)
        assert value[10:-10] not in destination.getvalue()
        assert destination.getvalue() == expected


def test_worker_log_preserves_utf8_split_at_every_byte():
    record = "public caf\u00e9 \u20ac \U0001f680\n".encode()
    for cut in range(1, len(record)):
        destination = BytesIO()
        kanban_worker_log.copy_redacted_worker_log_stream(
            _ChunkedReader([record[:cut], record[cut:]]), destination,
        )
        assert destination.getvalue() == record


@pytest.mark.parametrize("record", [
    b"Authorization:\nBearer " + b"aB8x" * 75 + b"\n",
    b"Proxy-Authorization: Bearer\n" + b"aB8x" * 75 + b"\n",
    b"x-api-key:\n" + b"aB8x" * 75 + b"\n",
    b'{"api_key":\n"ghp_ab\n' + b"aB8x" * 75 + b'"}\n',
    b"prefix ghp_ab\n\n" + b"aB8x" * 75 + b" suffix\n",
    b"prefix sk-ab\x1b" + b"aB8x" * 75 + b" suffix\n",
    b"prefix SG.ab\n" + b"aB8x" * 75 + b" suffix\n",
])
def test_worker_log_preserves_canonical_multiline_records(record):
    from agent.redact import redact_terminal_output

    expected = redact_terminal_output(record.decode(), force=True).encode()
    assert b"aB8x" * 20 not in expected
    for chunks in ([record], [record[i:i + 1] for i in range(len(record))]):
        destination = BytesIO()
        kanban_worker_log.copy_redacted_worker_log_stream(_ChunkedReader(chunks), destination)
        assert destination.getvalue() == expected


@pytest.mark.parametrize("record", [
    b"before -----BEGIN PRIVATE KEY-----\nprivate\n-----END PRIVATE KEY----- after\n",
    b"-----BEGIN PRIVATE KEY-----body-----END PRIVATE KEY-----\n",
    b"-----BEGIN PRIVATE KEY-----body-----END PRIVATE KEY-----",
])
def test_worker_log_preserves_text_adjacent_to_private_blocks(record):
    from agent.redact import redact_terminal_output

    expected = redact_terminal_output(record.decode(), force=True).encode()
    destination = BytesIO()
    kanban_worker_log.copy_redacted_worker_log_stream(
        _ChunkedReader([record[i:i + 1] for i in range(len(record))]), destination,
    )
    assert destination.getvalue() == expected


def test_worker_log_masks_unterminated_private_block():
    record = b"public before\n-----BEGIN PRIVATE KEY-----\n" + b"private material\n" * 10000
    destination = BytesIO()
    kanban_worker_log.copy_redacted_worker_log_stream(BytesIO(record), destination)
    written = destination.getvalue()
    assert written.startswith(b"public before\n[REDACTED PRIVATE KEY]")
    assert b"private material" not in written


def test_worker_log_replaces_incomplete_utf8_at_eof():
    destination = BytesIO()
    kanban_worker_log.copy_redacted_worker_log_stream(BytesIO(b"public \xc3"), destination)
    assert destination.getvalue() == "public \ufffd".encode()
