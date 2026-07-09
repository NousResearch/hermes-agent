import asyncio
import json

from capabilities.moneyprinter import adapter
from capabilities.moneyprinter.mcp import tools as mp_tools


def test_default_create_video_body_requires_subject():
    try:
        adapter._default_create_video_body({})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "video_subject" in str(exc)


def test_default_create_video_body_fills_defaults():
    payload = adapter._default_create_video_body({"video_subject": "  上海咖啡  "})
    assert payload["video_subject"] == "上海咖啡"
    assert payload["video_aspect"] == "9:16"
    assert payload["video_count"] == 1
    assert payload["subtitle_enabled"] is True
    assert payload["voice_name"]


def test_build_service_env_is_isolated(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/evil/site-packages")
    monkeypatch.setenv("VIRTUAL_ENV", "/evil/venv")
    monkeypatch.setenv("HOME", "/tmp/home")
    env = adapter._build_service_env()
    assert "PYTHONPATH" not in env
    assert "VIRTUAL_ENV" not in env
    assert env.get("PYTHONNOUSERSITE") == "1"
    assert env.get("PYTHONUNBUFFERED") == "1"
    assert env.get("HOME") == "/tmp/home"


def test_list_outputs_data_flattens_videos(tmp_path, monkeypatch):
    task_dir = tmp_path / "task-out"
    task_dir.mkdir()
    (task_dir / "final-1.mp4").write_bytes(b"fake")
    (task_dir / "script.json").write_text(
        json.dumps({"params": {"video_subject": "demo"}, "script": "hi"}),
        encoding="utf-8",
    )

    async def fake_proxy_json(method, upstream_path, body=None):
        return 503, {"message": "down"}

    monkeypatch.setattr(adapter, "TASKS_DIR", tmp_path)
    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy_json)

    status, payload = asyncio.run(adapter.list_outputs_data())
    assert status == 200
    assert payload["ok"] is True
    outputs = payload["data"]["outputs"]
    assert len(outputs) == 1
    assert outputs[0]["taskId"] == "task-out"
    assert "streamUrl" in outputs[0]


def test_mcp_generate_video_tool_uses_adapter(monkeypatch):
    async def fake_create(body):
        assert body["video_subject"] == "测试主题"
        return 200, {
            "ok": True,
            "data": {
                "task": {
                    "id": "t-1",
                    "state": "queued",
                    "progress": 0,
                    "videos": [],
                }
            },
            "error": None,
        }

    async def fake_ensure():
        return {"ok": True, "data": {"serviceRunning": True}}

    monkeypatch.setattr(adapter, "create_video_data", fake_create)
    monkeypatch.setattr(mp_tools, "_ensure_service_if_needed", fake_ensure)

    raw = mp_tools.moneyprinter_generate_video(video_subject="测试主题", auto_start=True)
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["data"]["task_id"] == "t-1"
    assert "Poll moneyprinter_get_task" in payload["data"]["message"]


def test_mcp_tool_specs_cover_phase3_and_phase2_names():
    names = {spec["name"] for spec in mp_tools.TOOL_SPECS}
    required = {
        "moneyprinter_health_check",
        "moneyprinter_generate_video",
        "moneyprinter_get_task",
        "moneyprinter_list_tasks",
        "moneyprinter_list_outputs",
        "moneyprinter_generate_script",
        "moneyprinter_generate_terms",
        "moneyprinter_delete_task",
    }
    assert required.issubset(names)
