"""Tests for kanban-dispatch.py."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\kanban-dispatch.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("kd", SCRIPT)
kd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kd)


# --- list_ready_tasks ---

def test_list_ready_tasks_parses_json():
    """Returns ready tasks regardless of assignee status."""
    tasks = [
        {"id": "t1", "title": "Task 1", "status": "ready", "assignee": None},
        {"id": "t2", "title": "Task 2", "status": "ready", "assignee": "alice"},
        {"id": "t3", "title": "Task 3", "status": "done", "assignee": None},
    ]
    with patch.object(kd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(tasks), stderr="")
        ready = kd.list_ready_tasks()
    # Should return ready tasks (regardless of assignee)
    assert len(ready) == 2
    assert {t["id"] for t in ready} == {"t1", "t2"}


def test_list_ready_tasks_handles_failure():
    with patch.object(kd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="auth failed")
        ready = kd.list_ready_tasks()
    assert ready == []


def test_list_ready_tasks_handles_invalid_json():
    with patch.object(kd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")
        ready = kd.list_ready_tasks()
    assert ready == []


def test_list_ready_tasks_empty():
    with patch.object(kd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")
        ready = kd.list_ready_tasks()
    assert ready == []


# --- dispatch_tasks ---

def test_dispatch_tasks_parses_json_spawned():
    output = json.dumps({"spawned": 3, "failed": 0})
    with patch.object(kd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=output, stderr="")
        count, out = kd.dispatch_tasks(5)
    assert count == 3
    assert out == output


def test_dispatch_tasks_parses_json_list():
    output = json.dumps([{"id": "t1"}, {"id": "t2"}])
    with patch.object(kd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=output, stderr="")
        count, _ = kd.dispatch_tasks(5)
    assert count == 2


def test_dispatch_tasks_parses_non_json():
    output = "Dispatched 2 task(s). spawned."  # 1 "dispatched" + 1 "spawned" = 2
    with patch.object(kd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=output, stderr="")
        count, _ = kd.dispatch_tasks(5)
    assert count == 2


# --- main flow ---

def test_main_no_tasks_exits_0(tmp_path, monkeypatch):
    """Direct call to main() — patches work."""
    monkeypatch.setattr(kd, "LOG_FILE", tmp_path / "logs" / "kanban-dispatch.log")
    with patch.object(kd, "list_ready_tasks", return_value=[]):
        r = kd.main()
    assert r == 0
    log_file = tmp_path / "logs" / "kanban-dispatch.log"
    assert log_file.exists()
    assert "no ready" in log_file.read_text()


def test_main_with_tasks_dispatches(tmp_path, monkeypatch):
    """Direct call — patches apply."""
    monkeypatch.setattr(kd, "LOG_FILE", tmp_path / "logs" / "kanban-dispatch.log")
    tasks = [{"id": "t1", "title": "Task 1", "status": "ready", "assignee": None}]
    with patch.object(kd, "list_ready_tasks", return_value=tasks):
        with patch.object(kd, "dispatch_tasks", return_value=(1, "dispatched 1")):
            with patch.object(kd, "send_telegram", return_value=True):
                r = kd.main()
    assert r == 0
    log_text = (tmp_path / "logs" / "kanban-dispatch.log").read_text()
    assert "1 spawned" in log_text


def test_main_respects_max(monkeypatch):
    tasks = [{"id": f"t{i}", "title": f"T{i}", "status": "ready", "assignee": None}
             for i in range(10)]
    with patch.object(kd, "list_ready_tasks", return_value=tasks):
        with patch.object(kd, "dispatch_tasks", return_value=(3, "ok")) as mock_disp:
            with patch.object(kd, "send_telegram", return_value=True):
                kd.main()
    # Should call dispatch with min(MAX_PER_RUN, len(tasks)) = 2
    args = mock_disp.call_args[0]
    assert args[0] == 2