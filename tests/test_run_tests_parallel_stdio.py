"""The runner's stdio must survive native Windows console edge cases.

On native Windows, piped or legacy-console stdio defaults to cp1252, which
cannot encode the runner's ✓/✗ progress glyphs — before the fix, the first
per-file status line killed the whole run with UnicodeEncodeError. The
failure depends only on the stream's encoding, so these tests pin it on
every OS by building a cp1252 stream explicitly. Gateway/pythonw processes
also need pytest output captured to a real file because their inherited
console handles may be invalid.
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cp1252_stream() -> tuple[io.TextIOWrapper, io.BytesIO]:
    raw = io.BytesIO()
    return io.TextIOWrapper(raw, encoding="cp1252", errors="strict"), raw


def test_cp1252_stream_reproduces_the_crash_without_the_fix() -> None:
    # Baseline for the bug: a strict cp1252 stream cannot take the glyph.
    stream, _raw = _cp1252_stream()
    try:
        stream.write("✓")
    except UnicodeEncodeError:
        return
    raise AssertionError("expected UnicodeEncodeError on strict cp1252")


def test_glyph_safe_stdio_survives_cp1252(monkeypatch) -> None:
    mod = _load_runner()
    stream, raw = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)

    mod._make_stdio_glyph_safe()
    print("✓ tests/foo.py (3 tests, 1.2s) ✗")
    sys.stdout.flush()

    out = raw.getvalue()
    assert "✓".encode("utf-8") in out, "stream should now carry UTF-8 glyphs"
    assert b"tests/foo.py (3 tests, 1.2s)" in out, "line content must survive"


def test_glyph_safe_stdio_noop_without_reconfigure(monkeypatch) -> None:
    # Streams without .reconfigure (e.g. pytest's capture buffers, plain
    # StringIO) must pass through untouched instead of raising.
    mod = _load_runner()
    plain = io.StringIO()
    monkeypatch.setattr(sys, "stdout", plain)
    monkeypatch.setattr(sys, "stderr", plain)

    mod._make_stdio_glyph_safe()
    print("✓ still fine")

    assert "✓ still fine" in plain.getvalue()


def test_windows_pytest_output_uses_a_file_not_a_console_pipe(
    monkeypatch, tmp_path: Path,
) -> None:
    """A pythonw worker can expose an unusable inherited console pipe."""
    mod = _load_runner()
    test_file = tmp_path / "test_probe.py"
    test_file.write_text("def test_ok(): pass\n", encoding="utf-8")

    class FakeProcess:
        pid = 123
        returncode = 0

        def __init__(self, _cmd, **kwargs):
            output = kwargs["stdout"]
            if output == subprocess.PIPE:
                raise OSError(22, "Invalid argument")
            assert kwargs["stderr"] == subprocess.STDOUT
            output.write(b"1 passed in 0.01s\n")
            output.flush()

        def communicate(self, timeout=None):
            return None, None

        def kill(self):
            return None

    monkeypatch.setattr(mod.sys, "platform", "win32")
    monkeypatch.setattr(mod.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(mod, "_kill_tree", lambda *_args, **_kwargs: None)

    _file, rc, output, summary, _duration = mod._run_one_file_once(
        test_file, ["-q"], tmp_path, 10
    )

    assert rc == 0
    assert "1 passed" in output
    assert summary == {"passed": 1}
