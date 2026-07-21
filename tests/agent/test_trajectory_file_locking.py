"""Reproduction test: save_trajectory() has no file locking.

When multiple processes call save_trajectory() concurrently, there is no
fcntl.flock() protecting the write.  On modern macOS/APFS, individual
write() syscalls happen to be atomic for reasonable sizes, so raw
concurrent writes rarely corrupt in practice.  However the *absence of
locking* means the code is fundamentally unsafe:

  - On Linux ext4/NFS, large writes can interleave.
  - If the write path ever changes (e.g. flush + fsync, buffered IO,
    or write size exceeds PIPE_BUF), data will corrupt.
  - Another process can write between open() and write().

This test uses two complementary strategies:

1. **Monkeypatch test**: Injects a small sleep inside the write path
   (simulating kernel-level write splitting on a loaded system).  Without
   file locking, the writes from concurrent processes interleave, producing
   corrupt JSONL lines.

2. **Lock-absence test**: One process holds an exclusive flock on the file;
   another process calls save_trajectory().  Without locking in the code,
   the second write goes through *ignoring* the lock — proving the file
   is not coordinated.

Expected result with bug present: FAIL (corruption / lock ignored).
Expected result after fix:        PASS (flock serialises all writes).
"""

import fcntl
import json
import multiprocessing
import os
import sys
import tempfile
import time


# ---------------------------------------------------------------------------
# Strategy 1: Monkeypatch write to simulate kernel-level splitting
# ---------------------------------------------------------------------------

def _slow_writer_process(filepath: str, process_id: int, num_writes: int,
                         barrier_path: str):
    """Child process: monkeypatch file.write to split into two chunks."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    import builtins
    _real_open = builtins.open

    class SlowWriteFile:
        """Wraps a real file but splits each write in half with a yield."""
        def __init__(self, fobj):
            self._f = fobj
        def write(self, data):
            mid = len(data) // 2
            self._f.write(data[:mid])
            self._f.flush()          # force partial data to disk
            time.sleep(0.001)        # yield to other processes
            self._f.write(data[mid:])
            self._f.flush()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self._f.__exit__(*args)
        def __getattr__(self, name):
            return getattr(self._f, name)

    def patched_open(*args, **kwargs):
        f = _real_open(*args, **kwargs)
        # Only wrap append-mode opens to our target file
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        fname = str(args[0]) if args else ""
        if "a" in mode and filepath in fname:
            return SlowWriteFile(f)
        return f

    builtins.open = patched_open

    from agent.trajectory import save_trajectory

    large_trajectory = [
        {"role": "user", "content": f"P{process_id}S{i}:" + "A" * 600}
        for i in range(4)
    ]

    while not os.path.exists(barrier_path):
        pass

    for seq in range(num_writes):
        save_trajectory(
            trajectory=large_trajectory,
            model=f"proc-{process_id}-seq-{seq}",
            completed=True,
            filename=filepath,
        )

    builtins.open = _real_open


def test_interleaved_writes_corrupt_jsonl():
    """Without file locking, split writes from concurrent processes
    interleave and produce corrupt JSONL lines.

    This test monkeypatches file.write() in child processes to write in two
    halves with a flush+sleep between them — simulating what happens on a
    loaded system or a filesystem that doesn't guarantee atomic large writes.
    """
    num_procs = 4
    writes_per_proc = 15

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        filepath = tmp.name
    barrier_path = filepath + ".go"

    try:
        processes = []
        for pid in range(num_procs):
            p = multiprocessing.Process(
                target=_slow_writer_process,
                args=(filepath, pid, writes_per_proc, barrier_path),
            )
            p.start()
            processes.append(p)

        time.sleep(0.3)
        with open(barrier_path, "w") as f:
            f.write("go")

        for p in processes:
            p.join(timeout=30)

        expected_lines = num_procs * writes_per_proc

        corrupt_lines = []
        total_lines = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, raw in enumerate(f, 1):
                line = raw.rstrip("\n")
                if not line:
                    continue
                total_lines += 1
                try:
                    obj = json.loads(line)
                    assert "conversations" in obj
                    assert "model" in obj
                except (json.JSONDecodeError, AssertionError) as exc:
                    corrupt_lines.append((line_num, str(exc), line[:100]))

        if corrupt_lines:
            print(f"\nBUG CONFIRMED: {len(corrupt_lines)}/{total_lines} "
                  f"lines corrupted by interleaved writes")
            for ln, err, preview in corrupt_lines[:3]:
                print(f"  Line {ln}: {err}\n    {preview!r}...")

        assert len(corrupt_lines) == 0, (
            f"{len(corrupt_lines)}/{total_lines} JSONL lines corrupted. "
            f"save_trajectory() needs fcntl.flock() to serialise writes.\n"
            f"First corrupt: {corrupt_lines[0] if corrupt_lines else 'N/A'}"
        )
        assert total_lines == expected_lines, (
            f"Expected {expected_lines} lines, got {total_lines}"
        )
    finally:
        for p in [filepath, barrier_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Strategy 2: Prove save_trajectory ignores existing exclusive locks
# ---------------------------------------------------------------------------

def _write_ignoring_lock(filepath: str, barrier_path: str):
    """Child: call save_trajectory while parent holds exclusive flock."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from agent.trajectory import save_trajectory

    while not os.path.exists(barrier_path):
        pass

    # If save_trajectory used flock internally, this would block until the
    # parent releases.  Without locking, it writes immediately.
    save_trajectory(
        trajectory=[{"role": "user", "content": "from child"}],
        model="child-model",
        completed=True,
        filename=filepath,
    )


def test_save_trajectory_ignores_exclusive_flock():
    """save_trajectory() does not acquire a file lock, so it writes
    right through an exclusive flock held by another process.

    After the fix, save_trajectory should use flock internally, causing the
    child's write to block until the parent releases.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        filepath = tmp.name
    barrier_path = filepath + ".go"

    try:
        # Parent acquires exclusive lock
        lock_fd = os.open(filepath, os.O_RDWR | os.O_CREAT)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        p = multiprocessing.Process(
            target=_write_ignoring_lock,
            args=(filepath, barrier_path),
        )
        p.start()

        # Signal child to go
        with open(barrier_path, "w") as f:
            f.write("go")

        # Give child time to attempt the write
        p.join(timeout=5)

        # Read what the child wrote (while we still hold the lock!)
        with open(filepath, "r") as f:
            content = f.read()

        # Release lock
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

        # If save_trajectory used flock, the child would have BLOCKED and
        # the file would be empty (child still waiting).  Without flock,
        # the child writes immediately — proving no lock coordination.
        lines = [l for l in content.strip().split("\n") if l.strip()]

        assert len(lines) == 0, (
            f"save_trajectory() wrote {len(lines)} line(s) while another process "
            f"held an exclusive flock — proving it does not use file locking. "
            f"Fix: add fcntl.flock(f, fcntl.LOCK_EX) inside save_trajectory()."
        )
    finally:
        for p_path in [filepath, barrier_path]:
            try:
                os.unlink(p_path)
            except OSError:
                pass
