"""uv --compile-bytecode is an optimization. A compile-step failure must not
fail the dependency sync (#124268).

Windows real-time AV can lock uv's temporary compile script (os error 1224)
after packages are already installed. The sync has to retry once without the
flag so pm repair/bootstrap still produce a usable environment.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from pm import environment
from pm.environment import PythonEnvironment
from pm.package import InstallError


def _env(tmp_path: Path, output=None) -> PythonEnvironment:
    return PythonEnvironment(
        uv=tmp_path / "uv",
        python=tmp_path / "python",
        destination=tmp_path / "venv",
        cache=tmp_path / "cache",
        env={},
        output=output,
    )


def test_streamed_sync_retries_without_bytecode_compile_and_reports_no_failure(tmp_path, monkeypatch):
    """Repair streams uv through LiveTail, and verbose output after the compile error can
    evict it from the 2000-character tail. The marker must survive, the retry must drop only
    --compile-bytecode, and the recovered sync must not paint a failure banner."""
    fake_uv = tmp_path / "fake_uv.py"
    fake_uv.write_text(
        "import sys\n"
        "if '--compile-bytecode' in sys.argv:\n"
        "    print('error: Failed to bytecode-compile Python file in: venv')\n"
        "    print('N' * 4000)\n"
        "    raise SystemExit(1)\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []
    real_stream = environment._run_streaming

    def spawn_fake_uv(command, **kwargs):
        calls.append(command[1:])
        return real_stream([sys.executable, str(fake_uv), *command[1:]], **kwargs)

    monkeypatch.setattr(environment, "_run_streaming", spawn_fake_uv)
    monkeypatch.setattr(environment, "verbose_output", lambda: False)
    notes = io.StringIO()
    _env(tmp_path, output=notes).sync(tmp_path, extras=("web",))

    assert len(calls) == 2
    assert "--compile-bytecode" in calls[0]
    assert calls[1] == [arg for arg in calls[0] if arg != "--compile-bytecode"]
    assert "✗" not in notes.getvalue()


def test_sync_does_not_retry_an_unrelated_uv_failure(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(self, args, *, cwd, timeout):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 1, stdout="",
                                           stderr="error: No solution found for the requested dependencies")

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    with pytest.raises(InstallError):
        _env(tmp_path).sync(tmp_path)
    assert len(calls) == 1
    assert "--compile-bytecode" in calls[0]
