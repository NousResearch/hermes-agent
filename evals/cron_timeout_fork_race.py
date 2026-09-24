"""Linux live cron race probe: real PID map, synchronized fork, owned-only cleanup.

Run with the repository's Python and a disposable HERMES_HOME. No provider calls.
The scheduling hook retains psutil's real snapshot; it never invents process state.
"""
import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import psutil

    # Confine adoption to this disposable probe, not the test runner or gateway.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
    with tempfile.TemporaryDirectory(prefix="cron-owned-fork-") as root:
        os.environ["HERMES_HOME"] = root
        os.environ["HERMES_CRON_SCRIPT_TIMEOUT"] = "2"
        from cron.scheduler_script import _run_job_script

        scripts = Path(root, "scripts")
        scripts.mkdir()
        release, marker = Path(root, "release"), Path(root, "pids.json")
        script = scripts / "spawner.py"
        script.write_text(
            "import subprocess,sys,time,os,json\nfrom pathlib import Path\n"
            "def spawn():\n"
            " return subprocess.Popen([sys.executable,'-c','import time;time.sleep(15)'],"
            "start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            "first=spawn()\n"
            f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),first.pid]))\n"
            f"while not Path({str(release)!r}).exists():time.sleep(0.001)\n"
            "late=spawn()\n"
            f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),first.pid,late.pid]))\n"
            "time.sleep(15)\n",
            encoding="utf-8",
        )
        original = psutil._ppid_map
        schedules = []

        def scheduling_map():
            mapping = original()
            release.touch()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    ids = json.loads(marker.read_text(encoding="utf-8"))
                    if len(ids) == 3 or psutil.Process(ids[0]).status() == psutil.STATUS_STOPPED:
                        break
                except (FileNotFoundError, ValueError, psutil.NoSuchProcess):
                    pass
                time.sleep(0.001)
            schedules.append(len(mapping))
            return mapping

        psutil._ppid_map = scheduling_map
        try:
            start = time.monotonic()
            result = _run_job_script(str(script), workdir=root)
            elapsed = time.monotonic() - start
        finally:
            psutil._ppid_map = original
        states = {}
        owned = psutil.Process().children(recursive=True)
        try:
            ids = json.loads(marker.read_text(encoding="utf-8"))
            for pid in ids:
                try:
                    states[pid] = psutil.Process(pid).status()
                except psutil.NoSuchProcess:
                    states[pid] = "gone"
            print(json.dumps({"result": result, "elapsed": elapsed, "states": states,
                              "real_snapshot_sizes": schedules}), flush=True)
            assert result[1].startswith("Script timed out after 2s:"), result
            assert len(ids) >= 2, "pre-existing detached child did not start"
            assert all(state in ("gone", psutil.STATUS_ZOMBIE) for state in states.values()), states
        finally:
            for process in owned:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except psutil.NoSuchProcess:
                    pass


if __name__ == "__main__":
    main()
