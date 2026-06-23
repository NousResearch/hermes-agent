import importlib
import os
import subprocess
from pathlib import Path

from backend.services.codex_bridge import CodexBridge


APP_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = APP_ROOT / "frontend"


def test_codex_bridge_supports_fake_exec_mode(monkeypatch):
    monkeypatch.setenv("VIRTUAL_OFFICE_FAKE_CODEX", "1")

    result = CodexBridge().exec(prompt="Reply with exactly FAKE_OK", workdir=r"D:\\Fake", timeout=7)

    assert result["output"] == "FAKE_OK"
    assert result["exit_code"] == 0
    assert result["workdir"] == r"D:\\Fake"
    assert result["session_id"].startswith("fake-codex-")
    assert result["timeout"] == 7


def test_frontend_exposes_browser_e2e_harness():
    result = subprocess.run(
        ["npm", "run", "test:e2e", "--", "--list"],
        cwd=FRONTEND_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CI": "1"},
    )

    assert result.returncode == 0, result.stderr or result.stdout
    combined = f"{result.stdout}\n{result.stderr}"
    assert "virtual-office.e2e.spec.ts" in combined


def test_data_root_can_be_overridden_for_isolated_e2e_runs(monkeypatch, tmp_path):
    custom_root = tmp_path / "vo-e2e-data"
    monkeypatch.setenv("VIRTUAL_OFFICE_DATA_ROOT", str(custom_root))

    json_store = importlib.import_module("backend.services.json_store")
    log_store = importlib.import_module("backend.services.log_store")
    settings_store = importlib.import_module("backend.services.settings_store")

    json_store = importlib.reload(json_store)
    log_store = importlib.reload(log_store)
    settings_store = importlib.reload(settings_store)

    assert json_store.TASKS_PATH == custom_root / "tasks" / "tasks.json"
    assert json_store.HANDOFFS_PATH == custom_root / "handoffs" / "handoffs.json"
    assert log_store.LOGS_PATH == custom_root / "logs" / "events.jsonl"
    assert settings_store.SETTINGS_PATH == custom_root / "config" / "settings.json"
