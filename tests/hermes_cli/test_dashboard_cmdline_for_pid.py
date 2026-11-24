"""``_dashboard_cmdline_for_pid`` returns the argv the kernel holds.

On macOS ``ps -o command=`` prints a newline inside one argument as ``\\012``, so a ``python -c``
bootstrap read back through ``shlex`` keeps only its first word and the post-update respawn dies
with ``SyntaxError: Expected one or more names after 'import'``.
"""

import subprocess
import sys

import pytest

from hermes_cli import main_dashboard


@pytest.mark.platforms("posix")
def test_python_c_argument_with_newlines_survives():
    argv = [sys.executable, "-I", "-c", "import sys\nprint('up', flush=True)\nsys.stdin.read()",
            "serve", "--host", "127.0.0.1", "--port", "9119"]
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline() == "up\n"  # exec done: the kernel holds the child's argv
        assert main_dashboard._dashboard_cmdline_for_pid(proc.pid) == argv
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
