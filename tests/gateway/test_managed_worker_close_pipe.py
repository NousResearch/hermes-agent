"""An already-finished worker's broken stdin does not discard its result."""
from contextlib import suppress
import subprocess
import sys

import pytest

from gateway.session_managed_worker import ManagedWorker


@pytest.mark.platforms('linux')
def test_finished_worker_closes_remaining_pipes_after_real_broken_stdin():
    process = subprocess.Popen([sys.executable, '-c', 'pass'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=4096)
    try:
        process.stdin.write(b'buffered input')
        process.wait(timeout=10)
        ManagedWorker(process).close()
        assert process.stdin.closed and process.stdout.closed and process.stderr.closed
    finally:
        with suppress(BrokenPipeError):
            process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def test_finished_worker_preserves_other_stdin_close_errors():
    process = subprocess.Popen([sys.executable, '-c', 'pass'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    original = process.stdin
    failure = OSError('unexpected stdin close failure')

    class FailedClose:
        def close(self):
            raise failure

    try:
        process.wait(timeout=10)
        process.stdin = FailedClose()
        with pytest.raises(OSError) as caught:
            ManagedWorker(process).close()
        assert caught.value is failure
    finally:
        original.close()
        process.stdout.close()
        process.stderr.close()
