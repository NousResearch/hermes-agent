"""`hermes <cmd> | head` must exit quietly, not dump a BrokenPipeError traceback."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix")  # Windows reports a closed pipe as EINVAL

_REPO = str(Path(__file__).resolve().parents[2])


def _run(body: str, *, close_reader: bool) -> subprocess.CompletedProcess:
    code = f"import sys; sys.path.insert(0, {_REPO!r})\nfrom hermes_cli.stdio import tolerate_closed_stdout\n{body}"
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, encoding="utf-8")
    if close_reader:
        proc.stdout.close()  # the reader went away, like `head` after its lines
        err = proc.stderr.read()
        proc.wait(timeout=60)
        return subprocess.CompletedProcess(proc.args, proc.returncode, "", err)
    out, err = proc.communicate(timeout=60)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def test_closed_stdout_exits_quietly():
    result = _run("tolerate_closed_stdout(lambda: print('row\\n' * 200000))()", close_reader=True)
    assert result.returncode == 1, result.stderr
    assert result.stderr == ""


def test_broken_pipe_elsewhere_still_raises():
    body = ("def entry():\n    print('ok')\n    raise BrokenPipeError(32, 'child pipe')\n"
            "tolerate_closed_stdout(entry)()")
    result = _run(body, close_reader=False)
    assert result.returncode == 1
    assert "BrokenPipeError: [Errno 32] child pipe" in result.stderr
    assert result.stdout == "ok\n"
