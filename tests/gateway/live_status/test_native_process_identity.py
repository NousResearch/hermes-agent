"""Exercise native process identity against a real child and its actual exit."""

from pathlib import Path
import subprocess
import sys

import psutil

from gateway.status import get_process_start_time
from hermes_cli._subprocess_compat import pid_exists_stdlib


def test_native_child_identity_and_exit():
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve())],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        assert child.stdout.readline() == b"ready\n"
        assert pid_exists_stdlib(child.pid)
        identity = get_process_start_time(child.pid)
        assert isinstance(identity, int) and identity > 0
        assert identity == get_process_start_time(child.pid)
        if sys.platform == "win32":
            assert abs(identity - round(psutil.Process(child.pid).create_time() * 100)) <= 1
        stdout, stderr = child.communicate(b"x", timeout=20)
        assert child.returncode == 0 and stdout == stderr == b""
        assert not pid_exists_stdlib(child.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)


if __name__ == "__main__":
    sys.stdout.buffer.write(b"ready\n")
    sys.stdout.buffer.flush()
    assert sys.stdin.buffer.read(1) == b"x"
