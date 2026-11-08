"""Version metadata must finish while a Windows ACP-style stdin reader is pending.

The Git lookup became lazy in 68e4c2b31a6, so catalog prefetch can first resolve
the User-Agent after the ACP transport has started reading stdin. An inherited
console/pipe context can stall the Git launcher, and run()'s timeout cleanup
then joins its descendant's captured pipes forever.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli._subprocess_compat import kill_and_drain, windows_hide_flags
from hermes_cli.local_runtime.processes import spawn_server


@pytest.mark.platforms("windows")
def test_git_metadata_finishes_with_an_active_acp_stdin_reader(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True,
                   stdin=subprocess.DEVNULL, capture_output=True, creationflags=windows_hide_flags())
    root = Path(__file__).resolve().parents[2]
    program = tmp_path / "acp_reader.py"
    program.write_text(
        "import os, sys, json, threading\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from pathlib import Path\n"
        "from hermes_cli.version_info import _run_git\n"
        "ready = threading.Event()\n"
        "def reader():\n"
        "    sys.stdin.buffer.readline()\n"
        "    ready.set()\n"
        "    sys.stdin.buffer.readline()\n"
        "threading.Thread(target=reader, daemon=True).start()\n"
        "assert ready.wait(15)\n"
        f"result = _run_git(Path({str(repo)!r}), 'rev-parse', '--is-inside-work-tree')\n"
        "print(json.dumps({'result': result}), flush=True)\n"
        # Do not wait for an intentionally pending transport reader during interpreter shutdown.
        "os._exit(0)\n", encoding="utf8")
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    output = tmp_path / "result.txt"
    with output.open("w", encoding="utf8") as stream:
        proc, job = spawn_server([sys.executable, str(program)], stdin=subprocess.PIPE,
                                 stdout=stream, stderr=stream, startupinfo=startup)
        try:
            proc.stdin.write(b"initialize\n")
            proc.stdin.flush()  # deliberately keep the ACP input pipe OPEN
            proc.wait(timeout=15)
            assert proc.returncode == 0, output.read_text(encoding="utf8")
        finally:
            if job is not None:
                job.close()
            kill_and_drain(proc, 1)
    assert json.loads(output.read_text(encoding="utf8").strip())["result"] == "true"
