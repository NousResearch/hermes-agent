"""``save_trajectory()`` appends must be serialized across processes (#12684)."""
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from agent.trajectory import save_trajectory

_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def test_concurrent_process_appends_stay_parseable(tmp_path):
    """Payloads far larger than one atomic write() from several processes: every line parses."""
    target = tmp_path / "trajectory_samples.jsonl"
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {_REPO_ROOT!r})
        from agent.trajectory import save_trajectory
        big = "x" * 300_000
        for i in range(5):
            save_trajectory([{{"from": "human", "value": f"P{{sys.argv[1]}}-{{i}} " + big}}],
                            model="m", completed=True, filename={str(target)!r})
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script, str(n)], stdin=subprocess.DEVNULL) for n in range(6)]
    for p in procs:
        assert p.wait(timeout=120) == 0

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30
    tags = {json.loads(ln)["conversations"][0]["value"].split(" ", 1)[0] for ln in lines}
    assert tags == {f"P{n}-{i}" for n in range(6) for i in range(5)}


def test_append_honours_a_foreign_exclusive_lock(tmp_path):
    """While another process holds the file lock, save_trajectory() blocks instead of writing through."""
    target = tmp_path / "failed_trajectories.jsonl"
    target.write_text("", encoding="utf-8")
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import os, sys, time
            f = open({str(target)!r}, "a")
            if os.name == "nt":
                import msvcrt; f.write(" "); f.flush(); f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl; fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            print("locked", flush=True)
            time.sleep(1.5)
        """)],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"  # type: ignore[union-attr]
        started = time.monotonic()
        save_trajectory([{"from": "human", "value": "hi"}], model="m", completed=False, filename=str(target))
        waited = time.monotonic() - started
    finally:
        holder.kill()
        holder.wait()
    assert waited >= 1.0, f"append went through a held lock after {waited:.2f}s"
    assert json.loads(target.read_text(encoding="utf-8").strip().splitlines()[-1])["completed"] is False
