"""close() always finishes its teardown: a worker whose handle still has not exited five seconds
after SIGKILL (a Windows launcher, an uninterruptible exit) must not raise out of close() and
leave the pipes, reader and stderr drain open (N103). Real child process; only its handle's
wait() reports the second timeout."""
import subprocess
import sys

import pytest

from gateway.session_managed_worker import ManagedWorker


class _WedgedHandle(subprocess.Popen):
    """The child ignores SIGTERM; after SIGKILL the handle keeps reporting it alive."""
    killed = False

    def kill(self):
        self.killed = True
        super().kill()

    def wait(self, timeout=None):
        if self.killed:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return super().wait(timeout)


@pytest.mark.platforms("linux", "macos")
def test_close_completes_when_the_post_kill_wait_times_out():
    process = _WedgedHandle([sys.executable, '-c', 'import signal, time\n'
                             'signal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint(1, flush=True)\ntime.sleep(120)'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            close_fds=True)  # no own group: the handle alone is signalled
    try:
        assert process.stdout.readline() == b'1\n'
        ManagedWorker(process).close()
        assert process.killed and process.stdin.closed and process.stdout.closed
    finally:
        process.killed = False
        if process.poll() is None:
            process.kill()
        process.killed = False
        subprocess.Popen.wait(process, timeout=10)
