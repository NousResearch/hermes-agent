import asyncio
import base64
import json
import tomllib
from types import SimpleNamespace

from capabilities.moneyprinter import adapter


def _response_body(response):
    return json.loads(response.body.decode("utf-8"))


def test_managed_sidecar_identity_requires_expected_marker():
    assert adapter._managed_identity_valid(
        {
            "status": 200,
            "data": {
                "managed": True,
                "protocol_version": 1,
                "service": "moneyprinterturbo",
            },
        }
    ) is True
    assert adapter._managed_identity_valid({"status": 200, "data": {"service": "other"}}) is False
    assert adapter._managed_identity_valid({"status": 200, "data": {"managed": False, "service": "moneyprinterturbo"}}) is False


def test_media_proxy_request_headers_forward_range_and_sidecar_authentication():
    request = SimpleNamespace(headers={"Range": "bytes=786432-"})

    headers = adapter._media_proxy_request_headers(request)

    assert headers["Range"] == "bytes=786432-"
    assert headers[adapter.SIDECAR_TOKEN_HEADER] == adapter._SIDECAR_TOKEN


def test_proxy_media_serves_completed_local_video_with_matching_range(tmp_path, monkeypatch):
    task_dir = tmp_path / "task-1"
    task_dir.mkdir()
    (task_dir / "final-1.mp4").write_bytes(b"0123456789")
    monkeypatch.setattr(adapter, "TASKS_DIR", tmp_path)

    request = SimpleNamespace(headers={"Range": "bytes=4-7"})
    response = asyncio.run(adapter.proxy_media("stream", "task-1/final-1.mp4", request))

    assert response.status == 206
    assert response.body == b"4567"
    assert response.headers["Content-Range"] == "bytes 4-7/10"
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.headers["Content-Type"] == "video/mp4"


def test_proxy_media_serves_allowlisted_minimax_audio_without_sidecar(tmp_path, monkeypatch):
    audio = tmp_path / "storage" / "custom_audio" / "preview.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"0123456789")
    monkeypatch.setattr(adapter, "MONEYPRINTER_ROOT", tmp_path)
    monkeypatch.setattr(adapter, "ClientSession", None)

    response = asyncio.run(
        adapter.proxy_media(
            "stream",
            "storage/custom_audio/preview.mp3",
            SimpleNamespace(headers={"Range": "bytes=2-5"}),
        )
    )

    assert response.status == 206
    assert response.body == b"2345"
    assert response.headers["Content-Type"] == "audio/mpeg"


def test_moneyprinter_runtime_status_reports_missing_dependencies(monkeypatch):
    adapter._moneyprinter_runtime_status.cache_clear()
    monkeypatch.setattr(adapter, "_moneyprinter_python", lambda: "/tmp/moneyprinter-python")
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=3,
            stdout=json.dumps({"ffmpeg": "", "missing": ["moviepy", "imageio_ffmpeg"]}),
            stderr="",
        ),
    )

    status = adapter._moneyprinter_runtime_status()

    assert status == {
        "ffmpegPath": "",
        "missingDependencies": ["moviepy", "imageio_ffmpeg"],
        "runtimePython": "/tmp/moneyprinter-python",
        "runtimeReady": False,
    }
    adapter._moneyprinter_runtime_status.cache_clear()


def test_start_service_rejects_unrelated_process_on_sidecar_port(monkeypatch):
    async def not_managed():
        return False

    async def port_in_use():
        return True

    monkeypatch.setattr(
        adapter,
        "_moneyprinter_runtime_status",
        lambda: {
            "ffmpegPath": "/tmp/ffmpeg",
            "missingDependencies": [],
            "runtimePython": "/tmp/python",
            "runtimeReady": True,
        },
    )
    monkeypatch.setattr(adapter, "_probe_managed_sidecar", not_managed)
    monkeypatch.setattr(adapter, "_sidecar_port_in_use", port_in_use)
    monkeypatch.setattr(adapter, "_is_installed", lambda: True)

    response = asyncio.run(adapter.start_service())
    payload = _response_body(response)

    assert response.status == 409
    assert payload["error"]["code"] == "MONEYPRINTER_PORT_CONFLICT"


def test_config_summary_returns_visible_saved_fields_without_secret(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '\n'.join(
            [
                'llm_provider = "openai"',
                'openai_api_key = "dummy-saved-key"',
                'openai_base_url = "https://openrouter.ai/api/v1"',
                'openai_model_name = "openrouter/auto"',
                'pexels_api_keys = ["dummy-pexels-key"]',
                'pixabay_api_keys = []',
                'coverr_api_keys = []',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "CONFIG_PATH", config_path)
    monkeypatch.setattr(adapter, "CONFIG_EXAMPLE_PATH", tmp_path / "missing.toml")

    summary = adapter._config_summary()

    assert summary["llmProvider"] == "openai"
    assert summary["modelName"] == "openrouter/auto"
    assert summary["baseUrl"] == "https://openrouter.ai/api/v1"
    assert summary["apiKeyConfigured"] is True
    assert summary["materialProviders"] == {"coverr": False, "pexels": True, "pixabay": False}
    assert "dummy-saved-key" not in json.dumps(summary)
    assert "dummy-pexels-key" not in json.dumps(summary)


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

    async def fake_proxy_json(method, upstream_path, body=None, **kwargs):
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

    async def fake_proxy_json(method, upstream_path, body=None, **kwargs):
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


def test_config_summary_includes_minimax_without_secret(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '\n'.join(
            [
                'llm_provider = "openai"',
                'openai_api_key = "dummy-saved-key"',
                '[minimax]',
                'api_key = "dummy-minimax-key"',
                'base_url = "https://api.minimaxi.com"',
                't2a_model = "speech-2.8-hd"',
                'voice_clone_model = "speech-2.8-hd"',
                'music_model = "music-2.6-free"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "CONFIG_PATH", config_path)
    monkeypatch.setattr(adapter, "CONFIG_EXAMPLE_PATH", tmp_path / "missing.toml")

    summary = adapter._config_summary()

    assert summary["minimax"] == {
        "apiKeyConfigured": True,
        "baseUrl": "https://api.minimaxi.com",
        "musicModel": "music-2.6-free",
        "t2aModel": "speech-2.8-hd",
        "voiceCloneModel": "speech-2.8-hd",
    }
    assert "dummy-minimax-key" not in json.dumps(summary)


def test_save_config_updates_minimax_credentials_without_returning_secret(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '\n'.join(
            [
                '[app]',
                'llm_provider = "deepseek"',
                'minimax_api_key = ""',
                'minimax_base_url = "https://api.minimax.io/v1"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "CONFIG_PATH", config_path)
    monkeypatch.setattr(adapter, "CONFIG_EXAMPLE_PATH", tmp_path / "missing.toml")

    class Request:
        async def json(self):
            return {
                "llmProvider": "deepseek",
                "minimaxApiKey": "new-minimax-secret",
                "minimaxBaseUrl": "https://api.minimaxi.com",
                "minimaxMusicModel": "music-2.6",
                "minimaxT2aModel": "speech-2.6-hd",
                "minimaxVoiceCloneModel": "speech-2.6-hd",
            }

    response = asyncio.run(adapter.save_config(Request()))
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    payload = _response_body(response)

    assert parsed["minimax"]["api_key"] == "new-minimax-secret"
    assert parsed["minimax"]["base_url"] == "https://api.minimaxi.com"
    assert parsed["minimax"]["music_model"] == "music-2.6"
    assert parsed["minimax"]["t2a_model"] == "speech-2.6-hd"
    assert parsed["minimax"]["voice_clone_model"] == "speech-2.6-hd"
    assert payload["data"]["minimax"]["apiKeyConfigured"] is True
    assert "new-minimax-secret" not in json.dumps(payload)


def test_save_config_inserts_missing_minimax_fields_into_canonical_section(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '\n'.join(
            [
                '[app]',
                'llm_provider = "deepseek"',
                '',
                '[chatterbox]',
                'base_url = "http://127.0.0.1:8000"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "CONFIG_PATH", config_path)
    monkeypatch.setattr(adapter, "CONFIG_EXAMPLE_PATH", tmp_path / "missing.toml")

    class Request:
        async def json(self):
            return {
                "llmProvider": "deepseek",
                "minimaxApiKey": "new-minimax-secret",
                "minimaxBaseUrl": "https://api.minimaxi.com",
            }

    response = asyncio.run(adapter.save_config(Request()))
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert response.status == 200
    assert parsed["minimax"]["api_key"] == "new-minimax-secret"
    assert parsed["minimax"]["base_url"] == "https://api.minimaxi.com"
    assert parsed["chatterbox"] == {"base_url": "http://127.0.0.1:8000"}


def test_save_config_updates_existing_minimax_section_without_touching_other_api_keys(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '\n'.join(
            [
                '[app]',
                'llm_provider = "deepseek"',
                '',
                '[elevenlabs]',
                'api_key = "keep-elevenlabs-secret"',
                '',
                '[minimax]',
                'api_key = "old-minimax-secret"',
                'base_url = "https://api.minimax.io/v1"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "CONFIG_PATH", config_path)
    monkeypatch.setattr(adapter, "CONFIG_EXAMPLE_PATH", tmp_path / "missing.toml")

    class Request:
        async def json(self):
            return {
                "llmProvider": "deepseek",
                "minimaxApiKey": "new-minimax-secret",
                "minimaxBaseUrl": "https://api.minimaxi.com",
            }

    response = asyncio.run(adapter.save_config(Request()))
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert response.status == 200
    assert parsed["elevenlabs"]["api_key"] == "keep-elevenlabs-secret"
    assert parsed["minimax"]["api_key"] == "new-minimax-secret"
    assert parsed["minimax"]["base_url"] == "https://api.minimaxi.com"


def test_save_config_rejects_multiline_scalar_injection(tmp_path, monkeypatch):
    original = '[app]\nllm_provider = "openai"\nminimax_base_url = "https://api.minimax.io/v1"\n'
    config_path = tmp_path / "config.toml"
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(adapter, "CONFIG_PATH", config_path)
    monkeypatch.setattr(adapter, "CONFIG_EXAMPLE_PATH", tmp_path / "missing.toml")

    class Request:
        async def json(self):
            return {
                "llmProvider": "openai",
                "minimaxBaseUrl": 'https://api.example.test\nlisten_host = "0.0.0.0"',
            }

    response = asyncio.run(adapter.save_config(Request()))
    payload = _response_body(response)

    assert response.status == 400
    assert payload["error"]["code"] == "MONEYPRINTER_INVALID_CONFIG"
    assert config_path.read_text(encoding="utf-8") == original


def test_list_assets_includes_local_minimax_voice_metadata(tmp_path, monkeypatch):
    voice_dir = tmp_path / "MiniMaxDemo001"
    voice_dir.mkdir(parents=True)
    (voice_dir / "metadata.json").write_text(
        json.dumps({"voice_id": "MiniMaxDemo001", "display_name": "演示音色"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "MINIMAX_VOICES_DIR", tmp_path)
    monkeypatch.setattr(adapter, "SONGS_DIR", tmp_path / "songs")
    monkeypatch.setattr(adapter, "CUSTOM_AUDIO_DIR", tmp_path / "custom_audio")
    monkeypatch.setattr(adapter, "FONTS_DIR", tmp_path / "fonts")

    status, payload = adapter.list_assets_data()

    assert status == 200
    assert "minimax:MiniMaxDemo001:演示音色" in payload["data"]["voices"]


def test_list_minimax_voices_proxies_structured_provider_records(monkeypatch):
    async def fake_proxy(method, path, body=None, **kwargs):
        assert method == "GET"
        assert path == "/api/v1/minimax/voices?voice_type=all"
        return 200, {
            "status": 200,
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
        }

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy)

    status, payload = asyncio.run(adapter.list_minimax_voices_data())

    assert status == 200
    assert payload["data"]["voices"][0]["id"] == "Korean_GentleBoss"


def test_clone_duplicate_id_returns_actionable_error(monkeypatch):
    async def fake_proxy(method, path, body=None, **kwargs):
        return 502, {"status": 502, "message": "voice clone voice id duplicate"}

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy)

    status, payload = asyncio.run(
        adapter.clone_minimax_voice_data(
            {
                "clone_audio": {"filename": "clone.wav", "contentBase64": "YQ=="},
                "voice_id": "Korean_GentleBoss",
            }
        )
    )

    assert status == 409
    assert payload["error"]["code"] == "MONEYPRINTER_MINIMAX_VOICE_ID_DUPLICATE"
    assert "已有音色" in payload["error"]["message"]


def test_minimax_proxy_uses_operation_specific_timeout(monkeypatch):
    captured = {}

    async def fake_proxy(method, path, body=None, *, timeout_seconds):
        captured.update({"method": method, "path": path, "timeout_seconds": timeout_seconds})
        return 200, {"status": 200, "data": {"audio": {"file": "generated.mp3"}}}

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy)

    status, payload = asyncio.run(adapter.generate_minimax_music_data({"prompt": "technology intro"}))

    assert status == 200
    assert payload["ok"] is True
    assert captured == {
        "method": "POST",
        "path": "/api/v1/minimax/music",
        "timeout_seconds": adapter.MINIMAX_PROXY_TIMEOUT_SECONDS["/api/v1/minimax/music"],
    }


def test_minimax_tts_response_includes_authenticated_audio_stream(monkeypatch):
    async def fake_proxy(method, path, body=None, **kwargs):
        return 200, {
            "status": 200,
            "data": {
                "audio": {
                    "file": "storage/custom_audio/minimax-preview.mp3",
                    "name": "minimax-preview.mp3",
                },
                "voice_id": "Korean_GentleBoss",
            },
        }

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy)

    status, payload = asyncio.run(
        adapter.generate_minimax_tts_data(
            {"text": "试听", "voice_id": "Korean_GentleBoss"}
        )
    )

    assert status == 200
    assert payload["data"]["audio"]["streamUrl"] == (
        "/api/capabilities/moneyprinter/stream/"
        "storage/custom_audio/minimax-preview.mp3"
    )


def test_minimax_clone_response_includes_authenticated_trial_stream(monkeypatch):
    async def fake_proxy(method, path, body=None, **kwargs):
        return 200, {
            "status": 200,
            "data": {
                "activated": False,
                "trialAudioFile": (
                    str(adapter.MONEYPRINTER_ROOT)
                    + "/storage/minimax/voices/HermesClone001/trial.mp3"
                ),
                "voice_id": "HermesClone001",
            },
        }

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy)

    status, payload = asyncio.run(
        adapter.clone_minimax_voice_data(
            {
                "clone_audio": {"filename": "clone.wav", "contentBase64": "YQ=="},
                "trial_text": "试听",
                "voice_id": "HermesClone001",
            }
        )
    )

    assert status == 200
    assert payload["data"]["trialAudio"]["streamUrl"].endswith(
        "/storage/minimax/voices/HermesClone001/trial.mp3"
    )


def test_generate_minimax_music_data_proxies_to_sidecar(monkeypatch):
    async def fake_proxy_json(method, upstream_path, body=None, **kwargs):
        assert method == "POST"
        assert upstream_path == "/api/v1/minimax/music"
        assert body == {"prompt": "科技感短视频开场", "save_as_bgm": True}
        return 200, {
            "status": 200,
            "data": {"bgm": {"file": "minimax-music-demo.mp3", "name": "minimax-music-demo.mp3"}},
        }

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy_json)

    status, payload = asyncio.run(
        adapter.generate_minimax_music_data({"prompt": "科技感短视频开场", "save_as_bgm": True})
    )

    assert status == 200
    assert payload["ok"] is True
    assert payload["data"]["bgm"]["file"] == "minimax-music-demo.mp3"


def test_clone_minimax_voice_materializes_local_audio_before_proxy(tmp_path, monkeypatch):
    clone_audio = tmp_path / "clone.wav"
    prompt_audio = tmp_path / "prompt.m4a"
    clone_audio.write_bytes(b"clone-audio")
    prompt_audio.write_bytes(b"prompt-audio")

    async def fake_proxy_json(method, upstream_path, body=None, **kwargs):
        assert method == "POST"
        assert upstream_path == "/api/v1/minimax/voices/clone"
        assert "sourcePath" not in body["clone_audio"]
        assert "sourcePath" not in body["prompt_audio"]
        assert base64.b64decode(body["clone_audio"]["contentBase64"]) == b"clone-audio"
        assert base64.b64decode(body["prompt_audio"]["contentBase64"]) == b"prompt-audio"
        return 200, {"status": 200, "data": {"voice_id": "MiniMaxDemo001"}}

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy_json)

    status, payload = asyncio.run(
        adapter.clone_minimax_voice_data(
            {
                "clone_audio": {"filename": "clone.wav", "sourcePath": str(clone_audio)},
                "prompt_audio": {"filename": "prompt.m4a", "sourcePath": str(prompt_audio)},
                "voice_id": "MiniMaxDemo001",
            }
        )
    )

    assert status == 200
    assert payload["data"]["voice_id"] == "MiniMaxDemo001"


def test_clone_minimax_voice_rejects_unsupported_local_audio(tmp_path, monkeypatch):
    clone_audio = tmp_path / "secret.txt"
    clone_audio.write_text("not audio", encoding="utf-8")
    called = False

    async def fake_proxy_json(method, upstream_path, body=None):
        nonlocal called
        called = True
        return 200, {}

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy_json)

    status, payload = asyncio.run(
        adapter.clone_minimax_voice_data(
            {
                "clone_audio": {"filename": "secret.txt", "sourcePath": str(clone_audio)},
                "voice_id": "MiniMaxDemo001",
            }
        )
    )

    assert status == 400
    assert payload["error"]["code"] == "MONEYPRINTER_MINIMAX_AUDIO_INVALID"
    assert called is False


def test_create_subtitle_data_stringifies_subtitle_enabled_for_upstream(monkeypatch):
    captured = {}

    async def fake_proxy_json(method, upstream_path, body=None):
        captured["method"] = method
        captured["upstream_path"] = upstream_path
        captured["body"] = body
        return 200, {"status": 200, "data": {"task_id": "subtitle-task", "state": "4"}}

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy_json)

    status, payload = asyncio.run(adapter.create_subtitle_data({"subtitle_enabled": False, "video_script": "Hello"}))

    assert status == 200
    assert payload["ok"] is True
    assert captured["method"] == "POST"
    assert captured["upstream_path"] == "/api/v1/subtitle"
    assert captured["body"]["subtitle_enabled"] == "false"
