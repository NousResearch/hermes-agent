import json
import os
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

def test_save_trajectory_handles_msvcrt_oserror(tmp_path):
    filename = str(tmp_path / "trajectory.jsonl")
    trajectory = [{"role": "user", "content": "hello"}]
    model = "test-model"
    completed = True

    mock_msvcrt = MagicMock()
    mock_msvcrt.LK_LOCK = 1
    mock_msvcrt.LK_UNLCK = 2
    mock_msvcrt.locking.side_effect = OSError("locking error")
    with patch("agent.trajectory.fcntl", None), patch("agent.trajectory.msvcrt", mock_msvcrt):
        # Should not raise exception because it's caught and ignored
        save_trajectory(trajectory, model, completed, filename=filename)

    assert os.path.exists(filename)
