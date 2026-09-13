"""``save_trajectory()`` appends must be serialized across processes (#12684)."""
import gzip
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

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


def test_concurrent_gzip_appends_stay_decompressible(tmp_path):
    """Gzip member trailers are written before the next process may append."""
    target = tmp_path / "trajectory_samples.jsonl.gz"
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {_REPO_ROOT!r})
        from agent.trajectory import save_trajectory
        big = "x" * 150_000
        for i in range(3):
            save_trajectory([{{"from": "human", "value": f"P{{sys.argv[1]}}-{{i}} " + big}}],
                            model="m", completed=True, filename={str(target)!r})
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script, str(n)], stdin=subprocess.DEVNULL) for n in range(4)]
    for p in procs:
        assert p.wait(timeout=120) == 0

    with gzip.open(target, "rt", encoding="utf-8") as stream:
        entries = [json.loads(line) for line in stream]
    assert len(entries) == 12
    assert {entry["conversations"][0]["value"].split(" ", 1)[0] for entry in entries} == {
        f"P{n}-{i}" for n in range(4) for i in range(3)
    }


def test_interrupted_gzip_member_never_reaches_destination(tmp_path):
    """A worker killed while buffering cannot leave a truncated member behind."""
    target = tmp_path / "interrupted-trajectory.jsonl.gz"
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {_REPO_ROOT!r})
        import agent.trajectory as trajectory
        trajectory._build_gzip_member = lambda _line: os._exit(99)
        trajectory.save_trajectory([{{"from": "human", "value": "interrupted"}}],
                                   model="m", completed=True, filename={str(target)!r})
    """)
    process = subprocess.run([sys.executable, "-c", script], check=False)
    assert process.returncode == 99
    assert not target.exists() or target.stat().st_size == 0

    save_trajectory([{"from": "human", "value": "after-interruption"}], "m", True, str(target))
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["conversations"][0]["value"] == "after-interruption"


def test_failed_gzip_replace_leaves_previous_members_intact(tmp_path, monkeypatch):
    """A failed atomic publication must not damage the existing gzip stream."""
    target = tmp_path / "partial-trajectory.jsonl.gz"
    save_trajectory([{"from": "human", "value": "before-failure"}], "m", True, str(target))

    def fail_replace(_staged, _destination):
        raise OSError("simulated publication failure")

    monkeypatch.setattr("agent.trajectory.os.replace", fail_replace)
    save_trajectory([{"from": "human", "value": "not-published"}], "m", True, str(target))

    with gzip.open(target, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["conversations"][0]["value"] == "before-failure"

    monkeypatch.undo()
    save_trajectory([{"from": "human", "value": "after-failure"}], "m", True, str(target))
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        values = [json.loads(line)["conversations"][0]["value"] for line in stream]
    assert values == ["before-failure", "after-failure"]


@pytest.mark.windows_only
def test_concurrent_gzip_appends_are_serialized_on_windows(tmp_path):
    """Windows must serialize gzip members through one stable raw descriptor lock."""
    target = tmp_path / "windows-trajectory.jsonl.gz"
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {_REPO_ROOT!r})
        from agent.trajectory import save_trajectory
        save_trajectory([{{"from": "human", "value": "P" + sys.argv[1]}}],
                        model="m", completed=True, filename={str(target)!r})
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script, str(n)], stdin=subprocess.DEVNULL) for n in range(4)]
    for process in procs:
        assert process.wait(timeout=120) == 0

    with gzip.open(target, "rt", encoding="utf-8") as stream:
        entries = [json.loads(line) for line in stream]
    assert {entry["conversations"][0]["value"] for entry in entries} == {f"P{n}" for n in range(4)}


@pytest.mark.windows_only
def test_default_gzip_trajectory_saves_on_windows(tmp_path, monkeypatch):
    """The default gzip path must work with Windows' file-locking API."""
    monkeypatch.chdir(tmp_path)
    save_trajectory([{"from": "human", "value": "hello"}], model="m", completed=True)

    with gzip.open(tmp_path / "trajectory_samples.jsonl.gz", "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["completed"] is True


@pytest.mark.live_system_guard_bypass
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
        if holder.poll() is None:
            holder.kill()
            holder.wait()
    assert waited >= 1.0, f"append went through a held lock after {waited:.2f}s"
    assert json.loads(target.read_text(encoding="utf-8").strip().splitlines()[-1])["completed"] is False


def test_interrupted_append_recovers_without_reading_history(tmp_path, monkeypatch):
    import builtins
    from agent.trajectory import _append_gzip_member_atomically, _build_gzip_member
    target = tmp_path / "recover.gz"
    _append_gzip_member_atomically(str(target), _build_gzip_member("first\n"))
    script = textwrap.dedent(f"""
        import builtins, os, sys
        sys.path.insert(0, {_REPO_ROOT!r})
        from agent.trajectory import _append_gzip_member_atomically, _build_gzip_member
        real_open = builtins.open
        class Interrupted:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def __getattr__(self, key): return getattr(self.stream, key)
            def write(self, payload):
                self.stream.write(payload[:5]); self.stream.flush()
                os.fsync(self.stream.fileno()); os._exit(99)
        def interrupted_open(name, mode='r', *args, **kwargs):
            stream = real_open(name, mode, *args, **kwargs)
            return Interrupted(stream) if str(name) == {str(target)!r} else stream
        builtins.open = interrupted_open
        _append_gzip_member_atomically({str(target)!r}, _build_gzip_member('lost\\n'))
    """)
    assert subprocess.run([sys.executable, "-c", script]).returncode == 99
    real_open = builtins.open
    def guarded_open(name, mode="r", *args, **kwargs):
        if str(name) == str(target):
            assert mode == "a+b", "must not read/copy accumulated history"
        return real_open(name, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    _append_gzip_member_atomically(str(target), _build_gzip_member("second\n"))
    monkeypatch.undo()
    with gzip.open(target, "rt") as stream:
        assert stream.read() == "first\nsecond\n"


@pytest.mark.macos_only
def test_first_gzip_creation_honors_umask(tmp_path):
    import os
    import stat
    target = tmp_path / "shared.gz"
    old = os.umask(0o002)
    try:
        save_trajectory([], "m", True, str(target))
    finally:
        os.umask(old)
    assert stat.S_IMODE(target.stat().st_mode) == 0o664
