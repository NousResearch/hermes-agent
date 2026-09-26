"""uv --compile-bytecode is an optimization. A compile-step failure must not
fail the dependency sync (#124268).

Windows real-time AV can lock uv's temporary compile script (os error 1224)
after packages are already installed. The sync has to retry once without the
flag so pm repair/bootstrap still produce a usable environment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pm.environment import PythonEnvironment
from pm.package import InstallError


_BYTECODE_STDERR = (
    "error: Failed to bytecode-compile Python file in: "
    "C:\\hermes\\cache\\uv\\venv\\Lib\\site-packages\n"
    "  Caused by: Failed to create temporary script file\n"
    "  Caused by: failed to write to file "
    "`C:\\hermes\\cache\\uv\\.tmpKKlVim\\pip_compileall.py`: "
    "The requested operation cannot be performed on a file with a "
    "user-mapped section open. (os error 1224)\n"
)


def _env(tmp_path: Path) -> PythonEnvironment:
    return PythonEnvironment(
        uv=tmp_path / "uv",
        python=tmp_path / "python",
        destination=tmp_path / "venv",
        cache=tmp_path / "cache",
        env={},
    )


def _completed(args: list[str], returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args, returncode, stdout="", stderr=stderr)


def test_sync_retries_without_bytecode_compile_when_that_step_fails(tmp_path, monkeypatch):
    """A bytecode-compile failure is not an install failure. The retry omits
    the flag and a clean second sync leaves the environment usable."""
    calls: list[list[str]] = []

    def fake_run(self, args, *, cwd, timeout):
        calls.append(list(args))
        if "--compile-bytecode" in args:
            return _completed(args, 1, _BYTECODE_STDERR)
        return _completed(args, 0)

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    _env(tmp_path).sync(tmp_path)

    assert len(calls) == 2
    assert "--compile-bytecode" in calls[0]
    assert "--compile-bytecode" not in calls[1]
    assert calls[1][0] == "sync"
    assert "--all-packages" in calls[1]
    assert "--frozen" in calls[1]


def test_sync_does_not_retry_an_unrelated_uv_failure(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(self, args, *, cwd, timeout):
        calls.append(list(args))
        return _completed(args, 1, "error: No solution found for the requested dependencies")

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    with pytest.raises(InstallError):
        _env(tmp_path).sync(tmp_path)
    assert len(calls) == 1
    assert "--compile-bytecode" in calls[0]


def test_sync_keeps_bytecode_compile_on_the_happy_path(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(self, args, *, cwd, timeout):
        calls.append(list(args))
        return _completed(args, 0)

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    _env(tmp_path).sync(tmp_path, extras=("web",), groups=("dev",))

    assert len(calls) == 1
    assert "--compile-bytecode" in calls[0]
    assert "--extra" in calls[0] and "web" in calls[0]
    assert "--group" in calls[0] and "dev" in calls[0]


def test_sync_recognizes_bytecode_failure_reported_only_on_stdout(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(self, args, *, cwd, timeout):
        calls.append(list(args))
        if "--compile-bytecode" in args:
            return subprocess.CompletedProcess(
                args, 1, stdout="error: Failed to bytecode-compile Python file in: venv\n", stderr=""
            )
        return _completed(args, 0)

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    _env(tmp_path).sync(tmp_path)
    assert len(calls) == 2
    assert "--compile-bytecode" not in calls[1]


def test_sync_fallback_keeps_selection_flags_and_warns(tmp_path, monkeypatch):
    import io

    calls: list[list[str]] = []

    def fake_run(self, args, *, cwd, timeout):
        calls.append(list(args))
        if "--compile-bytecode" in args:
            return _completed(args, 1, _BYTECODE_STDERR)
        return _completed(args, 0)

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    notes = io.StringIO()
    PythonEnvironment(
        uv=tmp_path / "uv",
        python=tmp_path / "python",
        destination=tmp_path / "venv",
        cache=tmp_path / "cache",
        env={},
        output=notes,
    ).sync(tmp_path, extras=("web",), no_default_groups=True)

    assert "--extra" in calls[1] and "web" in calls[1]
    assert "--no-default-groups" in calls[1]
    assert "--compile-bytecode" not in calls[1]
    assert "retrying the install without it" in notes.getvalue()


def test_bytecode_compile_failure_does_not_banner_the_status_as_failed(tmp_path, monkeypatch):
    """The non-verbose repair path wraps uv in LiveTail. Closing that tail as
    a failure before the retry paints a recovered sync as failed."""
    import io

    notes = io.StringIO()

    def fake_stream(command, *, cwd, env, timeout, output):
        output.write("error: Failed to bytecode-compile Python file in: venv\n")
        return subprocess.CompletedProcess(
            command, 1, "", "error: Failed to bytecode-compile Python file in: venv\n"
        )

    monkeypatch.setattr("pm.environment._run_streaming", fake_stream)
    monkeypatch.setattr("pm.environment.verbose_output", lambda: False)
    PythonEnvironment(
        uv=tmp_path / "uv",
        python=tmp_path / "python",
        destination=tmp_path / "venv",
        cache=tmp_path / "cache",
        env={},
        output=notes,
    )._run(["sync", "--frozen", "--compile-bytecode"], cwd=tmp_path, timeout=30)

    assert "✗" not in notes.getvalue()
    assert "failed" not in notes.getvalue().lower()


def test_streaming_tail_keeps_bytecode_marker_after_later_output(tmp_path):
    """Repair and bootstrap stream uv output. The diagnostic tail is 2000
    characters, and resolver markers are stitched back if verbose output
    evicts them. The bytecode marker has to survive the same way or the
    retry never runs on the path that actually failed (#124268)."""
    import sys

    from pm.environment import _run_streaming

    code = (
        "import sys\n"
        "sys.stdout.write('error: Failed to bytecode-compile Python file in: venv\\n')\n"
        "sys.stdout.write('N' * 4000 + '\\n')\n"
        "sys.stdout.flush()\n"
        "raise SystemExit(1)\n"
    )
    notes = __import__("io").StringIO()
    result = _run_streaming(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={},
        timeout=30,
        output=notes,
    )
    assert result.returncode == 1
    assert "failed to bytecode-compile" in (result.stderr or "").lower()


def test_sync_raises_when_the_fallback_sync_also_fails(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(self, args, *, cwd, timeout):
        calls.append(list(args))
        if "--compile-bytecode" in args:
            return _completed(args, 1, _BYTECODE_STDERR)
        return _completed(args, 1, "error: failed to download distribution")

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    with pytest.raises(InstallError, match="failed to download"):
        _env(tmp_path).sync(tmp_path)
    assert len(calls) == 2
    assert "--compile-bytecode" not in calls[1]
