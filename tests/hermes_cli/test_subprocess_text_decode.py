from __future__ import annotations

import subprocess
import sys


POLISH_TEXT = "Za\u017c\u00f3\u0142\u0107 g\u0119\u015bl\u0105 ja\u017a\u0144"


def test_run_text_subprocess_recovers_cp1250_polish_output(tmp_path):
    from hermes_cli.subprocess_text import run_text_subprocess

    script = tmp_path / "emit_cp1250.py"
    payload = POLISH_TEXT.encode("cp1250")
    script.write_bytes(
        b"import sys\n"
        b"sys.stdout.buffer.write(b'out: ' + " + repr(payload).encode("ascii") + b")\n"
        b"sys.stderr.buffer.write(b'err: ' + " + repr(payload).encode("ascii") + b")\n"
    )

    result = run_text_subprocess(
        [sys.executable, str(script)],
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert "out:" in result.stdout
    assert "err:" in result.stderr
    assert POLISH_TEXT in result.stdout
    assert POLISH_TEXT in result.stderr
    assert "\ufffd" not in result.stdout
    assert "\ufffd" not in result.stderr


def test_decode_subprocess_bytes_keeps_replacement_for_unhelpful_binary():
    from hermes_cli.subprocess_text import decode_subprocess_bytes

    decoded = decode_subprocess_bytes(b"before \xff\xfe after", allow_fallback=True)

    assert decoded.text == "before \ufffd\ufffd after"
    assert decoded.encoding == "utf-8"
    assert decoded.used_fallback is False
    assert decoded.had_replacement is True


def test_popen_text_kwargs_force_lossy_utf8_text_mode():
    from hermes_cli.subprocess_text import popen_text_kwargs

    kwargs = popen_text_kwargs(bufsize=1)

    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["bufsize"] == 1


def test_install_lossy_text_subprocess_defaults_patches_text_popen(monkeypatch):
    from hermes_cli import subprocess_text

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    subprocess_text.install_lossy_text_subprocess_defaults()

    proc = subprocess.Popen(["cmd"], text=True)

    assert proc.kwargs["encoding"] == "utf-8"
    assert proc.kwargs["errors"] == "replace"


def test_tui_slash_worker_uses_lossy_utf8_popen(monkeypatch):
    from tui_gateway import server

    captured = {}

    class Proc:
        stdin = None
        stdout = []
        stderr = []
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    worker = server._SlashWorker("session-key", "gpt-5.5")
    worker.close()

    assert captured["text"] is True
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
