import json
import os
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from agent.trajectory import save_trajectory

def test_save_trajectory_success(tmp_path):
    filename = str(tmp_path / "trajectory.jsonl")
    trajectory = [{"role": "user", "content": "hello"}]
    model = "test-model"
    completed = True

    save_trajectory(trajectory, model, completed, filename=filename)

    assert os.path.exists(filename)
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["conversations"] == trajectory
    assert data["model"] == model
    assert data["completed"] == completed
    assert "timestamp" in data

def test_save_trajectory_with_fcntl_mocked(tmp_path):
    filename = str(tmp_path / "trajectory.jsonl")
    trajectory = [{"role": "user", "content": "hello"}]
    model = "test-model"
    completed = True

    mock_fcntl = MagicMock()
    mock_fcntl.LOCK_EX = 1
    mock_fcntl.LOCK_UN = 2
    with patch("agent.trajectory.fcntl", mock_fcntl), patch("agent.trajectory.msvcrt", None):
        save_trajectory(trajectory, model, completed, filename=filename)

    assert mock_fcntl.flock.call_count == 2
    calls = mock_fcntl.flock.call_args_list
    assert calls[0][0][1] == mock_fcntl.LOCK_EX
    assert calls[1][0][1] == mock_fcntl.LOCK_UN

def test_save_trajectory_with_msvcrt_mocked(tmp_path):
    filename = str(tmp_path / "trajectory.jsonl")
    trajectory = [{"role": "user", "content": "hello"}]
    model = "test-model"
    completed = True

    mock_msvcrt = MagicMock()
    mock_msvcrt.LK_LOCK = 1
    mock_msvcrt.LK_UNLCK = 2
    with patch("agent.trajectory.fcntl", None), patch("agent.trajectory.msvcrt", mock_msvcrt):
        save_trajectory(trajectory, model, completed, filename=filename)

    assert mock_msvcrt.locking.call_count == 2
    calls = mock_msvcrt.locking.call_args_list
    assert calls[0][0][1] == mock_msvcrt.LK_LOCK
    assert calls[0][0][2] == 1
    assert calls[1][0][1] == mock_msvcrt.LK_UNLCK
    assert calls[1][0][2] == 1

def test_save_trajectory_fails_closed_on_msvcrt_oserror(tmp_path):
    filename = str(tmp_path / "trajectory.jsonl")
    trajectory = [{"role": "user", "content": "hello"}]
    model = "test-model"
    completed = True

    mock_msvcrt = MagicMock()
    mock_msvcrt.LK_LOCK = 1
    mock_msvcrt.LK_UNLCK = 2
    mock_msvcrt.locking.side_effect = OSError("locking error")
    with patch("agent.trajectory.fcntl", None), patch("agent.trajectory.msvcrt", mock_msvcrt):
        # Should fail closed and NOT write the trajectory
        save_trajectory(trajectory, model, completed, filename=filename)

    assert not os.path.exists(filename)


def test_save_trajectory_cross_process_locking(tmp_path):
    from agent.trajectory import fcntl, msvcrt
    if not fcntl and not msvcrt:
        pytest.skip("No platform-native locking support available")

    filename = str(tmp_path / "trajectory.jsonl")
    lock_filename = filename + ".lock"

    # Pre-populate lock file
    with open(lock_filename, "w", encoding="utf-8") as f:
        f.write(" ")

    # Acquire the exclusive lock in this process first
    lock_f = open(lock_filename, "r+" if msvcrt else "a+", encoding="utf-8")
    if fcntl:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
    elif msvcrt:
        lock_f.seek(0)
        msvcrt.locking(lock_f.fileno(), msvcrt.LK_LOCK, 1)

    # Spawn a child process to write to the trajectory
    repo_root = Path(__file__).parents[2]
    child_script = tmp_path / "child_writer.py"
    child_script.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(repo_root)!r})
            from agent.trajectory import save_trajectory
            save_trajectory([{{"role": "user", "content": "child"}}], "child-model", True, filename={filename!r})
            """
        )
    )

    import subprocess
    child = subprocess.Popen([sys.executable, str(child_script)])

    # The child writer should be blocked since we hold the lock.
    # Verify that the trajectory file has not been created or written to yet.
    time.sleep(0.5)
    assert not os.path.exists(filename) or os.path.getsize(filename) == 0

    # Release the lock
    if fcntl:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    elif msvcrt:
        lock_f.seek(0)
        msvcrt.locking(lock_f.fileno(), msvcrt.LK_UNLCK, 1)
    lock_f.close()

    # The child process should now unblock, write the trajectory, and exit successfully.
    child.wait(timeout=5)
    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 0
    with open(filename, "r", encoding="utf-8") as f:
        data = json.loads(f.read().strip())
    assert data["model"] == "child-model"
