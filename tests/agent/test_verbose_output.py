"""Verbose output preserves captured text while keeping TTY wrapping."""

import contextlib
import io
import os
from types import SimpleNamespace

from agent.tool_executor import _print_tool_completed
from run_agent import AIAgent


def test_redirected_verbose_result_preserves_text(tmp_path):
    result = "x" * 5000 + "MIDDLE_SENTINEL" + "─" * 200 + "\n\t spaced  text \n"
    log_path = tmp_path / "worker.log"
    agent = SimpleNamespace(verbose_logging=True, _wrap_verbose=AIAgent._wrap_verbose)
    with log_path.open("w", encoding="utf-8", newline="") as stream:
        with contextlib.redirect_stdout(stream):
            _print_tool_completed(agent, 1, 0.1, result)

    assert result.encode("utf-8") in log_path.read_bytes()


def test_verbose_tty_still_wraps_long_lines(monkeypatch):
    class TTY(io.StringIO):
        def isatty(self):
            return True

    columns = 60
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda *args: os.terminal_size((columns, 24))
    )
    result = "x" * 200
    with contextlib.redirect_stdout(TTY()):
        rendered = AIAgent._wrap_verbose("Result: ", result)

    lines = rendered.splitlines()
    assert len(lines) > 1
    assert "".join(line.strip() for line in lines).removeprefix("Result: ") == result
    assert all(len(line) <= columns for line in lines[1:])
