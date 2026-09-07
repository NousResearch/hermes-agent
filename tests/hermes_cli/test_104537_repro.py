"""Repro for #104537: _run_logged_subprocess must stream output into update.log
while the child is still running. The Desktop hand-off watchdog treats a
non-growing update.log as a stall and cancels a healthy build (exit 124).

Contract under test: child output reaches update.log *before* the child exits,
and still never touches the terminal.
"""

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from hermes_cli.main_dashboard import _UpdateOutputStream
from hermes_cli.update_cmd import _run_logged_subprocess
_CHILD_SRC = (
    "import pathlib, sys, time\n"
    "print('building', flush=True)\n"
    "release = pathlib.Path(sys.argv[1])\n"
    "deadline = time.monotonic() + 20\n"
    "while not release.exists() and time.monotonic() < deadline:\n"
    "    time.sleep(0.02)\n"
    "print('build finished', flush=True)\n"
    "sys.exit(int(sys.argv[2]))\n"
)


def test_build_progress_reaches_log_before_child_exits(tmp_path, monkeypatch):
    log_path = tmp_path / "update.log"
    release = tmp_path / "release"
    terminal = io.StringIO()
    with log_path.open("w", encoding="utf-8") as log:
        monkeypatch.setattr(sys, "stdout", _UpdateOutputStream(terminal, log))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _run_logged_subprocess,
                [sys.executable, "-c", _CHILD_SRC, str(release), "7"],
                cwd=tmp_path,
            )
            try:
                deadline = time.monotonic() + 20
                while (
                    "building" not in log_path.read_text(encoding="utf-8")
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                assert "building" in log_path.read_text(encoding="utf-8"), (
                    "update.log stayed empty while the child was alive - the "
                    "Desktop watchdog sees a stall and kills a healthy build"
                )
                assert not future.done(), "child exited before progress was observed"
            finally:
                release.touch()
                result = future.result(timeout=30)
    assert result.returncode == 7
    assert "build finished" in (result.stdout or "")
    assert terminal.getvalue() == ""  # still never echoed to the terminal
