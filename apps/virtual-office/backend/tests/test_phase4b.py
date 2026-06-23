from pathlib import Path

from backend.models.task import Task
from backend.services import handoff_manager, json_store, log_store, settings_store
from backend.services.handoff_manager import HandoffManager


class FakeCodexBridge:
    def exec(self, prompt: str, workdir: str = r"D:\Codex", timeout: int = 120) -> dict:
        return {
            "output": f"HANDLED::{prompt}",
            "session_id": "session-123",
            "exit_code": 0,
            "tokens_used": 42,
            "workdir": workdir,
        }


def _configure_temp_stores(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path]:
    tasks_path = tmp_path / "tasks.json"
    handoffs_path = tmp_path / "handoffs.json"
    logs_path = tmp_path / "events.jsonl"
    settings_path = tmp_path / "settings.json"

    monkeypatch.setattr(json_store, "TASKS_PATH", tasks_path)
    monkeypatch.setattr(json_store, "HANDOFFS_PATH", handoffs_path)
    monkeypatch.setattr(handoff_manager, "TASKS_PATH", tasks_path)
    monkeypatch.setattr(handoff_manager, "HANDOFFS_PATH", handoffs_path)
    monkeypatch.setattr(log_store, "LOGS_PATH", logs_path)
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", settings_path)
    return tasks_path, handoffs_path, logs_path, settings_path


def test_task_defaults_include_room() -> None:
    task = Task(id="task-1", title="Test", goal="Ship")
    assert task.status == "pending"
    assert task.agent == "codex"
    assert task.tags == []
    assert task.room == "main-office"


def test_settings_store_round_trip(tmp_path: Path, monkeypatch) -> None:
    _tasks_path, _handoffs_path, _logs_path, settings_path = _configure_temp_stores(tmp_path, monkeypatch)

    defaults = settings_store.get_settings()
    assert defaults["codex_workdir"] == r"D:\Codex"

    saved = settings_store.save_settings({"codex_workdir": r"D:\TradeDesk", "backend_port": 9000})
    assert saved["codex_workdir"] == r"D:\TradeDesk"
    assert saved["backend_port"] == 9000
    assert settings_path.exists()

    reread = settings_store.get_settings()
    assert reread["codex_workdir"] == r"D:\TradeDesk"
    assert reread["backend_port"] == 9000


def test_run_task_uses_configured_workdir_and_updates_logs(tmp_path: Path, monkeypatch) -> None:
    tasks_path, handoffs_path, _logs_path, _settings_path = _configure_temp_stores(tmp_path, monkeypatch)
    monkeypatch.setattr(HandoffManager, "__init__", lambda self: setattr(self, "codex_bridge", FakeCodexBridge()))
    settings_store.save_settings({"codex_workdir": r"D:\TradeDesk"})

    json_store.write_list_store(
        tasks_path,
        [
            {
                "id": "task-1",
                "title": "Review setup",
                "goal": "Review EURUSD setup",
                "context": "Use the latest notes",
                "agent": "codex",
                "room": "trade-room",
                "priority": "high",
                "status": "pending",
                "created_at": "2026-06-23T00:00:00+00:00",
                "updated_at": "2026-06-23T00:00:00+00:00",
                "tags": [],
            }
        ],
    )

    result = HandoffManager().run_task("task-1")

    assert result["status"] == "completed"
    assert result["result"]["task_id"] == "task-1"

    tasks = json_store.read_list_store(tasks_path)
    assert tasks[0]["status"] == "completed"
    assert tasks[0]["handoff_id"] == result["id"]
    assert tasks[0]["result"].startswith("HANDLED::Review EURUSD setup")

    handoffs = json_store.read_list_store(handoffs_path)
    assert len(handoffs) == 1
    assert handoffs[0]["payload"]["room"] == "trade-room"
    assert handoffs[0]["payload"]["prompt"] == "Review EURUSD setup"
    assert handoffs[0]["payload"]["workdir"] == r"D:\TradeDesk"

    events = log_store.list_events(limit=10, task_id="task-1")
    assert len(events) == 2
    assert events[0]["metadata"]["handoff_id"] == result["id"]


def test_list_events_filters_by_task_and_handoff(tmp_path: Path, monkeypatch) -> None:
    _tasks_path, _handoffs_path, _logs_path, _settings_path = _configure_temp_stores(tmp_path, monkeypatch)

    first = log_store.append_event("INFO", "Started", agent="codex", task_id="task-1", handoff_id="handoff-1")
    log_store.append_event("ERROR", "Failed", agent="codex", task_id="task-2", handoff_id="handoff-2")

    task_events = log_store.list_events(limit=10, task_id="task-1")
    handoff_events = log_store.list_events(limit=10, handoff_id="handoff-1")
    single = log_store.get_event(first["id"])

    assert len(task_events) == 1
    assert task_events[0]["id"] == first["id"]
    assert len(handoff_events) == 1
    assert handoff_events[0]["id"] == first["id"]
    assert single is not None
    assert single["message"] == "Started"
