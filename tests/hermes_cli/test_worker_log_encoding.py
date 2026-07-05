"""Worker log encoding and diagnostic fallback tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


POLISH_TEXT = "Za\u017c\u00f3\u0142\u0107 g\u0119\u015bl\u0105 ja\u017a\u0144"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _write_log(task_id: str, payload: bytes) -> Path:
    path = kb.worker_log_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_cp1250_polish_worker_log_uses_diagnostic_fallback(kanban_home):
    _write_log("t_cp1250", POLISH_TEXT.encode("cp1250"))

    status = kb.read_worker_log_status("t_cp1250")

    assert status is not None
    assert status.content == POLISH_TEXT
    assert status.encoding.lower() == "cp1250"
    assert status.used_fallback is True
    assert status.had_replacement is False


def test_invalid_utf8_worker_log_preserves_replacement_marker(kanban_home):
    _write_log("t_invalid", b"before \xff\xfe after")

    status = kb.read_worker_log_status("t_invalid")

    assert status is not None
    assert status.encoding == "utf-8"
    assert status.used_fallback is False
    assert status.had_replacement is True
    assert "\ufffd" in status.content
    assert "before" in status.content


def test_utf8_worker_log_remains_unchanged(kanban_home):
    text = f"normal UTF-8 log: {POLISH_TEXT}"
    _write_log("t_utf8", text.encode("utf-8"))

    status = kb.read_worker_log_status("t_utf8")

    assert status is not None
    assert status.content == text
    assert status.encoding == "utf-8"
    assert status.used_fallback is False
    assert status.had_replacement is False


def test_worker_log_status_respects_tail_cap_before_decoding(kanban_home):
    raw_suffix = POLISH_TEXT.encode("cp1250")
    _write_log("t_tail", b"SECRET-PREFIX-SHOULD-NOT-LEAK\n" + raw_suffix)

    status = kb.read_worker_log_status("t_tail", tail_bytes=len(raw_suffix))

    assert status is not None
    assert status.content == POLISH_TEXT
    assert "SECRET-PREFIX" not in status.content
    assert status.truncated is True
    assert status.encoding.lower() == "cp1250"


def test_kanban_log_encoding_report_includes_fallback_metadata(kanban_home):
    _write_log("t_cli", POLISH_TEXT.encode("cp1250"))

    out = kc.run_slash("log t_cli --encoding-report")

    assert "encoding=cp1250" in out
    assert "fallback=True" in out
    assert "replacement=False" in out
    assert POLISH_TEXT in out


def test_default_spawn_forces_utf8_worker_python_env(kanban_home, monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="encoding spawn", assignee="worker")
        task = kb.claim_task(conn, tid)
    assert task is not None

    captured: dict[str, dict[str, str]] = {}

    class Proc:
        pid = 4242

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return Proc()

    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_WORKER_STARTUP_CHECK_SECONDS", "0")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    kb._default_spawn(task, str(workspace))

    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    assert captured["env"]["PYTHONUTF8"] == "1"


def test_default_spawn_preserves_explicit_utf8_pythonioencoding(
    kanban_home, monkeypatch, tmp_path
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="encoding spawn preserve", assignee="worker")
        task = kb.claim_task(conn, tid)
    assert task is not None

    captured: dict[str, dict[str, str]] = {}

    class Proc:
        pid = 4343

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return Proc()

    monkeypatch.setenv("PYTHONIOENCODING", "utf-8:backslashreplace")
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_WORKER_STARTUP_CHECK_SECONDS", "0")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    kb._default_spawn(task, str(workspace))

    assert captured["env"]["PYTHONIOENCODING"] == "utf-8:backslashreplace"
    assert captured["env"]["PYTHONUTF8"] == "1"


def test_kanban_subprocess_text_capture_recovers_cp1250_worker_output(tmp_path):
    script = tmp_path / "emit_cp1250.py"
    payload = POLISH_TEXT.encode("cp1250")
    script.write_bytes(
        b"import sys\n"
        b"sys.stdout.buffer.write(b'max-iteration summary: ' + "
        + repr(payload).encode("ascii")
        + b" + b'\\n')\n"
        b"sys.stderr.buffer.write(b'stderr tail: ' + "
        + repr(payload).encode("ascii")
        + b" + b'\\n')\n"
    )

    result = kb._run_text_subprocess(
        [sys.executable, str(script)],
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert "max-iteration summary:" in result.stdout
    assert "stderr tail:" in result.stderr
    assert POLISH_TEXT in result.stdout
    assert POLISH_TEXT in result.stderr
    assert "\ufffd" not in result.stdout
    assert "\ufffd" not in result.stderr


def test_default_spawn_overrides_disabled_pythonutf8(kanban_home, monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="encoding spawn pyutf8", assignee="worker")
        task = kb.claim_task(conn, tid)
    assert task is not None

    captured: dict[str, dict[str, str]] = {}

    class Proc:
        pid = 4444

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return Proc()

    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("HERMES_KANBAN_WORKER_STARTUP_CHECK_SECONDS", "0")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    kb._default_spawn(task, str(workspace))

    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    assert captured["env"]["PYTHONUTF8"] == "1"

