import asyncio
import json
from types import SimpleNamespace

from capabilities.moneyprinter import adapter


def _response_body(response):
    return json.loads(response.body.decode("utf-8"))


def test_list_tasks_recovers_completed_disk_outputs(tmp_path, monkeypatch):
    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "final-1.mp4").write_bytes(b"fake mp4")
    (task_dir / "final-1TEMP_MPY_wvf_snd.mp4").write_bytes(b"moviepy temp audio")
    (task_dir / "script.json").write_text(
        json.dumps(
            {
                "params": {"video_subject": "Hermes Video Studio"},
                "script": "A short Hermes demo.",
                "search_terms": ["desktop agent", "video studio"],
            }
        ),
        encoding="utf-8",
    )

    async def fake_proxy_json(method, upstream_path, body=None):
        return 200, {"status": 200, "data": {"tasks": []}}

    monkeypatch.setattr(adapter, "TASKS_DIR", tmp_path)
    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy_json)
    monkeypatch.setattr(
        adapter,
        "_json",
        lambda data, status=200: SimpleNamespace(body=json.dumps(data).encode("utf-8"), status=status),
    )

    response = asyncio.run(adapter.list_tasks())
    body = _response_body(response)

    assert body["ok"] is True
    tasks = body["data"]["tasks"]
    task = next(item for item in tasks if item["id"] == "task-1")
    assert task["state"] == "complete"
    assert task["progress"] == 100
    assert task["subject"] == "Hermes Video Studio"
    assert task["videos"][0]["streamUrl"] == "/api/capabilities/moneyprinter/stream/task-1/final-1.mp4"
    assert all("TEMP" not in video["name"] for video in task["videos"])


def test_get_task_falls_back_to_disk_when_upstream_lost_state(tmp_path, monkeypatch):
    task_dir = tmp_path / "task-2"
    task_dir.mkdir()
    (task_dir / "combined-1.mp4").write_bytes(b"fake mp4")
    (task_dir / "script.json").write_text(
        json.dumps({"params": {"video_subject": "Recovered task"}, "script": "Recovered script."}),
        encoding="utf-8",
    )

    async def fake_proxy_json(method, upstream_path, body=None):
        return 404, {"status": 404, "message": "task not found"}

    monkeypatch.setattr(adapter, "TASKS_DIR", tmp_path)
    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy_json)
    monkeypatch.setattr(
        adapter,
        "_json",
        lambda data, status=200: SimpleNamespace(body=json.dumps(data).encode("utf-8"), status=status),
    )

    response = asyncio.run(adapter.get_task("task-2"))
    body = _response_body(response)

    assert body["ok"] is True
    assert body["data"]["id"] == "task-2"
    assert body["data"]["script"] == "Recovered script."
    assert body["data"]["videos"][0]["downloadUrl"] == "/api/capabilities/moneyprinter/download/task-2/combined-1.mp4"


def test_as_task_normalizes_upstream_tasks_media_prefix():
    task = adapter._as_task(
        {
            "task_id": "task-3",
            "state": "1",
            "videos": ["/tasks/task-3/final-1.mp4"],
        }
    )

    assert task["videos"][0]["file"] == "task-3/final-1.mp4"
    assert task["videos"][0]["streamUrl"] == "/api/capabilities/moneyprinter/stream/task-3/final-1.mp4"
    assert task["videos"][0]["downloadUrl"] == "/api/capabilities/moneyprinter/download/task-3/final-1.mp4"


def test_normalize_media_path_accepts_desktop_and_upstream_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "TASKS_DIR", tmp_path)

    assert adapter._normalize_media_path("/tasks/t/final-1.mp4") == "t/final-1.mp4"
    assert (
        adapter._normalize_media_path("/api/capabilities/moneyprinter/stream/tasks/t/final-1.mp4")
        == "t/final-1.mp4"
    )
    assert adapter._normalize_media_path("http://127.0.0.1:8080/api/v1/download/tasks/t/final-1.mp4") == "t/final-1.mp4"
    assert adapter._normalize_media_path(str(tmp_path / "t" / "final-1.mp4")) == "t/final-1.mp4"


def test_as_task_normalizes_upstream_numeric_state():
    assert adapter._as_task({"task_id": "t1", "state": 1})["state"] == "complete"
    assert adapter._as_task({"task_id": "t2", "state": 4})["state"] == "processing"
    assert adapter._as_task({"task_id": "t3", "state": -1})["state"] == "failed"


def test_media_proxy_forwards_range_headers():
    assert "content-range" in adapter.MEDIA_PROXY_HEADERS


def test_upload_local_material_copies_into_moneyprinter_whitelist(tmp_path, monkeypatch):
    local_dir = tmp_path / "local_videos"
    source = tmp_path / "source clip.mp4"
    source.write_bytes(b"fake material")
    monkeypatch.setattr(adapter, "LOCAL_MATERIALS_DIR", local_dir)

    status, payload = adapter.upload_local_material_data({"filename": "../source clip.mp4", "sourcePath": str(source)})

    assert status == 200
    assert payload["ok"] is True
    material = payload["data"]["material"]
    assert material["file"] == "source clip.mp4"
    assert material["kind"] == "video"
    assert (local_dir / "source clip.mp4").read_bytes() == b"fake material"

    status, payload = adapter.list_local_materials_data()
    assert status == 200
    assert [item["file"] for item in payload["data"]["materials"]] == ["source clip.mp4"]


def test_default_create_video_body_normalizes_local_materials(tmp_path, monkeypatch):
    local_dir = tmp_path / "local_videos"
    local_dir.mkdir()
    (local_dir / "clip-a.mp4").write_bytes(b"fake material")
    monkeypatch.setattr(adapter, "LOCAL_MATERIALS_DIR", local_dir)

    payload = adapter._default_create_video_body(
        {
            "video_materials": [{"duration": "2.5", "url": "../clip-a.mp4"}],
            "video_source": "local",
            "video_subject": "本地素材测试",
        }
    )

    assert payload["video_materials"] == [{"duration": 2.5, "provider": "local", "url": "clip-a.mp4"}]
