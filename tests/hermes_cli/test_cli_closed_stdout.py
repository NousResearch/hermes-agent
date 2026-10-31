"""`hermes <cmd> | head` must exit quietly, not dump a BrokenPipeError traceback."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix")  # Windows reports a closed pipe as EINVAL

_REPO = str(Path(__file__).resolve().parents[2])


def _run(body: str, *, close_reader: bool) -> subprocess.CompletedProcess:
    code = f"import sys; sys.path.insert(0, {_REPO!r})\nfrom hermes_cli.stdio import tolerate_closed_stdout\n{body}"
    argv = [sys.executable, "-c", code]
    if not close_reader:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8",
                              timeout=60, check=False)
    read_end, write_end = os.pipe()
    os.close(read_end)  # the reader already went away, like `head` after its lines
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=write_end, stderr=subprocess.PIPE,
                              encoding="utf-8", timeout=60, check=False)
    finally:
        os.close(write_end)


@pytest.mark.parametrize("body", [
    "tolerate_closed_stdout(lambda: print('row\\n' * 200000))()",  # large write, nothing left buffered
    "def entry():\n    print('row')\n    sys.exit(3)\ntolerate_closed_stdout(entry)()",  # buffered + sys.exit
])
def test_closed_stdout_exits_quietly(body):
    result = _run(body, close_reader=True)
    assert result.returncode == 1, result.stderr
    assert result.stderr == ""


def test_broken_pipe_elsewhere_still_raises():
    body = ("def entry():\n    print('ok')\n    raise BrokenPipeError(32, 'child pipe')\n"
            "tolerate_closed_stdout(entry)()")
    result = _run(body, close_reader=False)
    assert result.returncode == 1
    assert "BrokenPipeError: [Errno 32] child pipe" in result.stderr
    assert result.stdout == "ok\n"
