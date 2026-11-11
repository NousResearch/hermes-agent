"""hermes -z must consume piped stdin without ever hanging on an idle pipe (#70647).

Two contracts under test:

1. Append: ``hermes -z "PROMPT" < file`` (or ``producer | hermes -z "PROMPT"``)
   must deliver ``PROMPT\\n\\n<stdin content>`` to the agent instead of silently
   discarding the piped data.
2. Never hang: a wrapper/daemon may launch ``hermes -z`` with stdin attached to
   an open pipe that is never written nor closed. The pre-fix behavior returns
   normally (stdin ignored); a naive ``sys.stdin.read()`` would block until EOF
   forever. The fix must keep the "returns normally" property: if no data (and
   no EOF) shows up within the first-byte window, stdin is treated as absent.

Timing contract ("first-byte handshake"): once ANY data — or EOF — arrives
within the window, the read commits to blocking until EOF, exactly like any
Unix filter (``cat``, ``grep``). Only a pipe that produced *nothing at all*
within the window is treated as absent. That window is the one deliberate
trade-off: a producer slower than the window to emit its FIRST byte is
indistinguishable from a never-writing daemon pipe and gets dropped.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from hermes_cli.oneshot import _read_stdin_prompt

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Wall-clock guard for tests that would deadlock the suite if the
# implementation regressed to an unconditional blocking read. Generous so
# slow CI never trips it; the implementation windows under test are <= 2s.
_SUITE_HANG_GUARD = 15.0


def _pipe_stdin(monkeypatch, data: bytes | None, *, keep_writer_open: bool = False):
    """Install a real OS pipe as ``sys.stdin`` and return the write fd (or None).

    ``data`` is written to the pipe up front. Unless ``keep_writer_open`` the
    write end is closed so readers see EOF — the ``hermes -z < file`` /
    ``echo x | hermes -z`` shape. With ``keep_writer_open=True`` the write end
    stays open (caller must close it), modeling the daemon-idle-pipe shape.
    """
    rfd, wfd = os.pipe()
    if data:
        os.write(wfd, data)
    if not keep_writer_open:
        os.close(wfd)
        wfd = None
    reader = os.fdopen(rfd, "r", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", reader)
    return wfd


class TestReadStdinPrompt:
    """Unit contract of ``_read_stdin_prompt`` (returns (prompt, error))."""

    def test_appends_piped_content_with_blank_line_separator(self, monkeypatch):
        _pipe_stdin(monkeypatch, b"line one\nline two\n")
        prompt, err = _read_stdin_prompt("summarize this")
        assert err is None
        assert prompt == "summarize this\n\nline one\nline two\n"

    def test_none_stdin_returns_prompt_unchanged(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", None)
        assert _read_stdin_prompt("p") == ("p", None)

    def test_tty_stdin_returns_prompt_unchanged(self, monkeypatch):
        class _Tty:
            closed = False

            def isatty(self):
                return True

            def read(self, *a):  # pragma: no cover - must never be called
                raise AssertionError("read() must not be called on a TTY stdin")

        monkeypatch.setattr(sys, "stdin", _Tty())
        assert _read_stdin_prompt("p") == ("p", None)

    def test_closed_stdin_returns_prompt_unchanged(self, monkeypatch):
        rfd, wfd = os.pipe()
        os.close(wfd)
        reader = os.fdopen(rfd, "r", encoding="utf-8")
        reader.close()
        monkeypatch.setattr(sys, "stdin", reader)
        assert _read_stdin_prompt("p") == ("p", None)

    def test_empty_piped_stdin_returns_prompt_unchanged(self, monkeypatch):
        # </dev/null and `: | hermes -z` — EOF with zero bytes.
        _pipe_stdin(monkeypatch, None)
        assert _read_stdin_prompt("p") == ("p", None)

    def test_read_failure_returns_original_prompt_and_error(self, monkeypatch):
        class _Broken:
            closed = False

            def isatty(self):
                return False

            def read(self, *a):
                raise OSError("boom")

        monkeypatch.setattr(sys, "stdin", _Broken())
        prompt, err = _read_stdin_prompt("p")
        assert prompt == "p"
        assert err == "hermes -z: failed to read piped stdin.\n"

    def test_open_pipe_never_written_does_not_hang(self, monkeypatch):
        """THE #70647 hang case: open pipe, no writer activity, no EOF.

        Run the call on a worker thread so a regression to unconditional
        blocking read fails this assertion instead of deadlocking pytest;
        the finally-close of the write end unblocks any leaked reader.
        """
        wfd = _pipe_stdin(monkeypatch, None, keep_writer_open=True)
        result: list = []
        worker = threading.Thread(
            target=lambda: result.append(_read_stdin_prompt("p", timeout=0.5)),
            daemon=True,
        )
        try:
            start = time.monotonic()
            worker.start()
            worker.join(_SUITE_HANG_GUARD)
            elapsed = time.monotonic() - start
            assert not worker.is_alive(), (
                "_read_stdin_prompt blocked on an idle open pipe — the #70647 "
                "hang the fix exists to prevent"
            )
            # Idle pipe == stdin absent: prompt unchanged, no error.
            assert result == [("p", None)]
            assert elapsed < _SUITE_HANG_GUARD / 2
        finally:
            os.close(wfd)

    def test_slow_writer_first_byte_within_window_is_read(self, monkeypatch):
        """A producer that needs a moment before its first byte still counts."""
        wfd = _pipe_stdin(monkeypatch, None, keep_writer_open=True)

        def _write_late():
            time.sleep(0.3)
            os.write(wfd, b"late data")
            os.close(wfd)

        writer = threading.Thread(target=_write_late, daemon=True)
        writer.start()
        prompt, err = _read_stdin_prompt("p", timeout=2.0)
        writer.join(_SUITE_HANG_GUARD)
        assert err is None
        assert prompt == "p\n\nlate data"

    def test_streaming_writer_commits_to_full_read_past_window(self, monkeypatch):
        """First byte inside the window commits to read-until-EOF, however
        long the rest of the stream takes — no truncation mid-stream."""
        wfd = _pipe_stdin(monkeypatch, None, keep_writer_open=True)

        def _stream():
            os.write(wfd, b"head ")
            time.sleep(0.8)  # well past the 0.2s decision window below
            os.write(wfd, b"tail")
            os.close(wfd)

        writer = threading.Thread(target=_stream, daemon=True)
        writer.start()
        prompt, err = _read_stdin_prompt("p", timeout=0.2)
        writer.join(_SUITE_HANG_GUARD)
        assert err is None
        assert prompt == "p\n\nhead tail"


def _oneshot_program() -> str:
    """A ``python -c`` body that stubs the agent to echo the prompt it got.

    Same isolation pattern as test_oneshot_surrogate.py: run_oneshot in a
    real subprocess so OS-level stdin plumbing (pipes, EOF, idle writers)
    is exercised for real, not simulated.
    """
    return textwrap.dedent(
        """
        import hermes_cli.oneshot as oneshot

        def fake_run_agent(prompt, **kwargs):
            return (
                "AGENT_GOT<<" + prompt + ">>",
                {"failed": False, "partial": False, "completed": True},
            )

        oneshot._run_agent = fake_run_agent
        raise SystemExit(oneshot.run_oneshot("base prompt"))
        """
    )


class TestOneshotStdinEndToEnd:
    def test_piped_stdin_reaches_the_agent(self):
        """The original #70647 report: -z with piped stdin must not drop it."""
        result = subprocess.run(
            [sys.executable, "-c", _oneshot_program()],
            cwd=_REPO_ROOT,
            input="piped payload\n",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "AGENT_GOT<<base prompt\n\npiped payload\n>>" in result.stdout

    def test_idle_open_pipe_exits_promptly_with_prompt_only(self):
        """Daemon shape: stdin is an open pipe nobody writes or closes.

        Pre-fix hermes returns normally here (stdin ignored); the fix must
        preserve that. A naive blocking-read patch makes this wait() raise
        TimeoutExpired — which is exactly the regression this test pins.
        """
        rfd, wfd = os.pipe()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", _oneshot_program()],
                cwd=_REPO_ROOT,
                stdin=rfd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                rc = proc.wait(timeout=_SUITE_HANG_GUARD)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                pytest.fail(
                    "hermes -z hung on an idle open stdin pipe (#70647 hang case)"
                )
            stdout = proc.stdout.read()
            assert rc == 0, proc.stderr.read()
            assert "AGENT_GOT<<base prompt>>" in stdout
        finally:
            os.close(rfd)
            os.close(wfd)

    def test_stdin_read_failure_fails_the_run_loudly(self):
        """If piped data exists but cannot be read, running with the bare
        prompt would silently reproduce #70647 on wrong input (and bill for
        it). Usage-error exit instead, matching run_oneshot's other guards."""
        program = textwrap.dedent(
            """
            import sys
            import hermes_cli.oneshot as oneshot

            class Broken:
                closed = False
                def isatty(self):
                    return False
                def read(self, *a):
                    raise OSError("boom")

            sys.stdin = Broken()
            oneshot._run_agent = lambda prompt, **kw: (
                "should never run", {"failed": False, "completed": True}
            )
            raise SystemExit(oneshot.run_oneshot("base prompt"))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert "failed to read piped stdin" in result.stderr
        assert "should never run" not in result.stdout
