"""Takeover completion failures must keep the build's stderr tail (#124040).

`run_contained` raises CalledProcessError with the failure tail in
output/stderr, but update_finish recorded only str(exc) (exit status + argv),
so receipts like historical_completion kept the exit code and lost the
diagnosable tail (npm exit 1 / 4294967295 with no actionable lines).
"""
from __future__ import annotations

import subprocess

from hermes_cli.update_finish import _completion_failure_detail


def _failure(output="", stderr=""):
    return subprocess.CalledProcessError(
        1, ["npm.CMD", "run", "build", "--", "--icons", "C:\\repo"],
        output=output, stderr=stderr)


def test_build_tail_survives_in_receipt_detail():
    tail = "\n".join(f"error TS2304: cannot find name '{name}'." for name in ("aaa", "bbb"))
    detail = _completion_failure_detail(_failure(output=f"building...\n{tail}"))
    assert "returned non-zero exit status 1" in detail
    assert "cannot find name 'aaa'" in detail
    assert "cannot find name 'bbb'" in detail


def test_stderr_fallback_when_stdout_empty():
    detail = _completion_failure_detail(_failure(stderr="EBUSY: resource busy or locked"))
    assert "EBUSY" in detail


def test_empty_output_leaves_plain_message():
    exc = _failure()
    assert _completion_failure_detail(exc) == str(exc)


def test_plain_exception_passes_through():
    exc = RuntimeError("cannot create takeover receipt")
    assert _completion_failure_detail(exc) == str(exc)


def test_tail_is_capped_to_recent_lines():
    lines = [f"line {n:02d}" for n in range(30)]
    detail = _completion_failure_detail(_failure(output="\n".join(lines)))
    assert "line 29" in detail
    assert "line 00" not in detail
