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
    assert env.get("MONEYPRINTER_HERMES_TOKEN") == adapter._SIDECAR_TOKEN


def test_managed_sidecar_command_binds_to_loopback(monkeypatch):
    monkeypatch.setattr(adapter, "DEFAULT_BASE_URL", "http://127.0.0.1:18080")
    monkeypatch.setattr(adapter, "_moneyprinter_python", lambda: "/tmp/moneyprinter-python")

    command = adapter._managed_sidecar_command()

    assert command[:4] == ("/tmp/moneyprinter-python", "-m", "uvicorn", "app.asgi:app")
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "18080"


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


def test_mcp_cache_local_material_uses_existing_adapter(monkeypatch):
    seen = {}

    def fake_upload(body):
        seen.update(body)
        return 200, {
            "ok": True,
            "data": {"material": {"file": body["filename"]}},
            "error": None,
        }

    monkeypatch.setattr(adapter, "upload_local_material_data", fake_upload)

    payload = json.loads(
        mp_tools.moneyprinter_cache_local_material(
            "/vault/02_精选镜头/clip.mp4",
            "beef-noodle-asset-clip.mp4",
        )
    )

    assert payload["data"]["material"]["file"] == "beef-noodle-asset-clip.mp4"
    assert seen == {
        "filename": "beef-noodle-asset-clip.mp4",
        "sourcePath": "/vault/02_精选镜头/clip.mp4",
    }


def test_mcp_generate_video_accepts_cached_local_materials(monkeypatch):
    seen = {}

    async def fake_create(body):
        seen.update(body)
        return 200, {
            "ok": True,
            "data": {"task": {"id": "task-local", "state": "queued"}},
            "error": None,
        }

    monkeypatch.setattr(adapter, "create_video_data", fake_create)

    payload = json.loads(
        mp_tools.moneyprinter_generate_video(
            video_subject="牛肉面门店",
            video_script="顾客吃面。员工端碗。成品面特写。",
            video_source="local",
            local_materials=["one.mp4", "two.mp4", "three.mp4"],
            custom_audio_file="acceptance.mp3",
            match_materials_to_script=True,
            auto_start=False,
            bgm_type="none",
        )
    )

    assert payload["data"]["task_id"] == "task-local"
    assert seen["video_source"] == "local"
    assert [item["url"] for item in seen["video_materials"]] == ["one.mp4", "two.mp4", "three.mp4"]
    assert seen["custom_audio_file"] == "acceptance.mp3"
    assert seen["match_materials_to_script"] is True


def test_mcp_tool_specs_cover_phase3_and_phase2_names():
    names = {spec["name"] for spec in mp_tools.TOOL_SPECS}
    required = {
        "moneyprinter_health_check",
        "moneyprinter_generate_video",
        "moneyprinter_cache_local_material",
        "moneyprinter_get_task",
        "moneyprinter_list_tasks",
        "moneyprinter_list_outputs",
        "moneyprinter_generate_script",
        "moneyprinter_generate_terms",
        "moneyprinter_delete_task",
        "moneyprinter_minimax_list_voices",
        "moneyprinter_minimax_clone_voice",
        "moneyprinter_minimax_generate_tts",
        "moneyprinter_minimax_generate_lyrics",
        "moneyprinter_minimax_generate_music",
        "video_library_import_asset",
        "video_library_get_status",
        "video_library_scan_library",
        "video_library_analyze_asset",
        "video_library_search_clips",
        "video_library_create_timeline",
    }
    assert required.issubset(names)


def test_mcp_video_library_import_uses_adapter(monkeypatch):
    from capabilities.video_library import adapter as video_library_adapter

    monkeypatch.setattr(
        video_library_adapter,
        "import_asset_data",
        lambda body: (
            200,
            {
                "ok": True,
                "data": {"asset": {"id": "asset-1", "source_path": body["sourcePath"]}},
                "error": None,
            },
        ),
    )

    payload = json.loads(mp_tools.video_library_import_asset("/tmp/source.mp4"))

    assert payload["data"]["asset"] == {"id": "asset-1", "source_path": "/tmp/source.mp4"}


def test_mcp_video_library_named_import_forwards_library_id(monkeypatch):
    from capabilities.video_library import adapter as video_library_adapter

    seen = {}

    def fake_import(body):
        seen.update(body)
        return 200, {"ok": True, "data": {"asset": {"id": "asset-1"}}, "error": None}

    monkeypatch.setattr(video_library_adapter, "import_asset_data", fake_import)

    payload = json.loads(mp_tools.video_library_import_asset("/tmp/source.mp4", library_id="beef-noodle"))

    assert payload["ok"] is True
    assert seen == {"sourcePath": "/tmp/source.mp4", "libraryId": "beef-noodle"}


def test_mcp_video_library_named_analyze_forwards_library_id(monkeypatch):
    from capabilities.video_library import adapter as video_library_adapter

    seen = {}

    def fake_analyze(asset_id, body):
        seen.update({"asset_id": asset_id, **body})
        return 200, {"ok": True, "data": {"clips": []}, "error": None}

    monkeypatch.setattr(video_library_adapter, "analyze_asset_data", fake_analyze)

    payload = json.loads(mp_tools.video_library_analyze_asset("asset-1", library_id="beef-noodle"))

    assert payload["ok"] is True
    assert seen["asset_id"] == "asset-1"
    assert seen["libraryId"] == "beef-noodle"


def test_mcp_video_library_status_uses_named_adapter(monkeypatch):
    from capabilities.video_library import adapter as video_library_adapter

    monkeypatch.setattr(
        video_library_adapter,
        "library_status_data",
        lambda library_id: (
            200,
            {"ok": True, "data": {"library_id": library_id, "clips": 3}, "error": None},
        ),
    )

    payload = json.loads(mp_tools.video_library_get_status("beef-noodle"))

    assert payload["data"] == {"library_id": "beef-noodle", "clips": 3}


def test_mcp_video_library_named_timeline_forwards_script(monkeypatch):
    from capabilities.video_library import adapter as video_library_adapter

    seen = {}

    def fake_timeline(body):
        seen.update(body)
        return 200, {"ok": True, "data": {"id": "timeline-1"}, "error": None}

    monkeypatch.setattr(video_library_adapter, "create_timeline_data", fake_timeline)
    script = [{"id": "segment-1", "text": "顾客吃面"}]

    payload = json.loads(
        mp_tools.video_library_create_timeline(
            ["clip-1"],
            library_id="beef-noodle",
            script=script,
        )
    )

    assert payload["ok"] is True
    assert seen == {
        "aspect": "9:16",
        "clipIds": ["clip-1"],
        "libraryId": "beef-noodle",
        "script": script,
    }


def test_mcp_video_library_scan_uses_named_adapter(monkeypatch):
    from capabilities.video_library import adapter as video_library_adapter

    monkeypatch.setattr(
        video_library_adapter,
        "scan_library_data",
        lambda library_id, body: (
            200,
            {
                "data": {"dry_run": body["dryRun"], "library_id": library_id},
                "error": None,
                "ok": True,
            },
        ),
    )

    payload = json.loads(mp_tools.video_library_scan_library("beef-noodle", dry_run=True))

    assert payload["data"] == {"dry_run": True, "library_id": "beef-noodle"}


def test_mcp_minimax_generate_music_uses_adapter(monkeypatch):
    service_started = False

    async def fake_ensure():
        nonlocal service_started
        service_started = True
        return {"ok": True}

    async def fake_music(body):
        assert body["prompt"] == "科技感短视频开场"
        assert body["lyrics_optimizer"] is True
        assert body["save_as_bgm"] is True
        return 200, {
            "ok": True,
            "data": {"bgm": {"file": "minimax-music-demo.mp3", "name": "minimax-music-demo.mp3"}},
            "error": None,
        }

    monkeypatch.setattr(adapter, "generate_minimax_music_data", fake_music)
    monkeypatch.setattr(mp_tools, "_ensure_service_if_needed", fake_ensure)

    raw = mp_tools.moneyprinter_minimax_generate_music(prompt="科技感短视频开场", save_as_bgm=True)
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["data"]["bgm"]["file"] == "minimax-music-demo.mp3"
    assert service_started is True


def test_mcp_minimax_list_voices_uses_adapter(monkeypatch):
    async def fake_list_voices():
        return 200, {
            "ok": True,
            "data": {
                "voices": [
                    {
                        "category": "system",
                        "id": "Korean_GentleBoss",
                        "name": "Gentle Boss",
                        "providerConfirmed": True,
                    }
                ]
            },
            "error": None,
        }

    monkeypatch.setattr(
        adapter,
        "list_minimax_voices_data",
        fake_list_voices,
    )

    payload = json.loads(mp_tools.moneyprinter_minimax_list_voices())

    assert payload["data"]["voices"][0]["id"] == "Korean_GentleBoss"


def test_mcp_minimax_clone_voice_uses_local_source_and_configured_model(monkeypatch):
    service_started = False

    async def fake_ensure():
        nonlocal service_started
        service_started = True
        return {"ok": True}

    async def fake_clone(body):
        assert body["activate"] is False
        assert body["clone_audio"] == {"filename": "/tmp/clone.wav", "sourcePath": "/tmp/clone.wav"}
        assert body["model"] == ""
        return 200, {"ok": True, "data": {"voice_id": body["voice_id"]}, "error": None}

    monkeypatch.setattr(adapter, "clone_minimax_voice_data", fake_clone)
    monkeypatch.setattr(mp_tools, "_ensure_service_if_needed", fake_ensure)

    payload = json.loads(
        mp_tools.moneyprinter_minimax_clone_voice(
            voice_id="MiniMaxDemo001",
            clone_audio_source_path="/tmp/clone.wav",
        )
    )

    assert payload["data"]["voice_id"] == "MiniMaxDemo001"
    assert service_started is True


def test_mcp_minimax_tts_uses_adapter(monkeypatch):
    async def fake_ensure():
        return {"ok": True}

    async def fake_tts(body):
        assert body == {
            "model": "",
            "save_as_custom_audio": True,
            "text": "Hello Hermes",
            "voice_id": "MiniMaxDemo001",
        }
        return 200, {"ok": True, "data": {"audio": {"file": "tts.mp3"}}, "error": None}

    monkeypatch.setattr(adapter, "generate_minimax_tts_data", fake_tts)
    monkeypatch.setattr(mp_tools, "_ensure_service_if_needed", fake_ensure)

    payload = json.loads(mp_tools.moneyprinter_minimax_generate_tts("Hello Hermes", "MiniMaxDemo001"))

    assert payload["data"]["audio"]["file"] == "tts.mp3"


def test_mcp_minimax_lyrics_uses_adapter(monkeypatch):
    async def fake_ensure():
        return {"ok": True}

    async def fake_lyrics(body):
        assert body == {
            "lyrics": "",
            "mode": "write_full_song",
            "prompt": "A concise technology song",
            "title": "",
        }
        return 200, {"ok": True, "data": {"lyrics": "[Verse]"}, "error": None}

    monkeypatch.setattr(adapter, "generate_minimax_lyrics_data", fake_lyrics)
    monkeypatch.setattr(mp_tools, "_ensure_service_if_needed", fake_ensure)

    payload = json.loads(mp_tools.moneyprinter_minimax_generate_lyrics("A concise technology song"))

    assert payload["data"]["lyrics"] == "[Verse]"
