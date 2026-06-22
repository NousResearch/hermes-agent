"""Concurrency regression test for save_trajectory() file locking.

Salvage of #13218 by @bennytimz. In gateway mode many sessions (across
platforms) append to the SAME default trajectory file
(``trajectory_samples.jsonl`` / ``failed_trajectories.jsonl``). Without an
exclusive lock around the append, interleaved writes from concurrent threads
can corrupt the JSONL so individual lines no longer parse. save_trajectory()
now holds an exclusive advisory lock (POSIX ``fcntl.flock``) for the
single write+flush.
"""
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.trajectory import save_trajectory  # noqa: E402


def test_concurrent_save_trajectory_produces_valid_jsonl(tmp_path):
    """Every line written by concurrent appenders must be parseable JSON."""
    target = tmp_path / "trajectory_samples.jsonl"

    n_writers = 24
    # A reasonably large per-entry payload makes interleaving (and thus
    # corruption without locking) far more likely to manifest.
    big = "x" * 4096

    def _write(i: int) -> None:
        conv = [
            {"from": "human", "value": f"q{i} {big}"},
            {"from": "gpt", "value": f"a{i} {big}"},
        ]
        save_trajectory(conv, model=f"m{i}", completed=True, filename=str(target))

    with ThreadPoolExecutor(max_workers=n_writers) as pool:
        list(pool.map(_write, range(n_writers)))

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_writers, f"expected {n_writers} lines, got {len(lines)}"
    models = set()
    for ln in lines:
        obj = json.loads(ln)  # raises if a line was corrupted by interleaving
        assert obj["completed"] is True
        assert len(obj["conversations"]) == 2
        models.add(obj["model"])
    # No write was lost or overwritten: all distinct model tags survived.
    assert models == {f"m{i}" for i in range(n_writers)}


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl lock is POSIX-only")
def test_save_trajectory_uses_exclusive_lock(tmp_path, monkeypatch):
    """On POSIX the append is wrapped in LOCK_EX … LOCK_UN exactly once."""
    import agent.trajectory as traj

    calls = []
    real_flock = traj.fcntl.flock

    def _spy(fd, op):
        calls.append(op)
        return real_flock(fd, op)

    monkeypatch.setattr(traj.fcntl, "flock", _spy)

    save_trajectory([{"from": "human", "value": "hi"}], model="m", completed=False,
                    filename=str(tmp_path / "failed_trajectories.jsonl"))

    assert calls == [traj.fcntl.LOCK_EX, traj.fcntl.LOCK_UN]
