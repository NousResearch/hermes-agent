import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from pydantic import ValidationError

from app.controllers.v1 import minimax as minimax_controller
from app.models.exception import HttpException
from app.models.schema import MiniMaxMusicRequest, MiniMaxTtsRequest, MiniMaxUploadedAsset, MiniMaxVoiceCloneRequest


def _storage_dir(tmp_path, sub_dir="", create=False):
    target = tmp_path / "storage" / sub_dir
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return str(target.resolve())


def test_uploaded_file_rejects_source_path_and_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(
        minimax_controller.utils,
        "storage_dir",
        lambda sub_dir="", create=False: _storage_dir(tmp_path, sub_dir, create),
    )

    with pytest.raises(HttpException) as missing_content:
        minimax_controller._uploaded_file(
            SimpleNamespace(filename="clone.wav", contentBase64=None, sourcePath="/tmp/secret.wav"),
            "request-1",
            "clone_audio",
        )
    assert "contentBase64" in missing_content.value.message

    with pytest.raises(HttpException) as unsupported_extension:
        minimax_controller._uploaded_file(
            MiniMaxUploadedAsset(
                contentBase64=base64.b64encode(b"not audio").decode("ascii"),
                filename="clone.txt",
            ),
            "request-1",
            "clone_audio",
        )
    assert "mp3, m4a or wav" in unsupported_extension.value.message

    with pytest.raises(ValidationError):
        MiniMaxUploadedAsset(filename="clone.wav")


def test_clone_voice_cleans_up_decoded_uploads(tmp_path, monkeypatch):
    seen_paths = []
    monkeypatch.setattr(minimax_controller.base, "get_task_id", lambda request: "request-2")
    monkeypatch.setattr(
        minimax_controller.utils,
        "storage_dir",
        lambda sub_dir="", create=False: _storage_dir(tmp_path, sub_dir, create),
    )

    def fake_clone_voice(**kwargs):
        seen_paths.extend([kwargs["clone_audio_file"], kwargs["prompt_audio_file"]])
        assert Path(kwargs["clone_audio_file"]).read_bytes() == b"clone-audio"
        assert Path(kwargs["prompt_audio_file"]).read_bytes() == b"prompt-audio"
        return {"voice_id": kwargs["voice_id"]}

    monkeypatch.setattr(minimax_controller.minimax, "clone_voice", fake_clone_voice)
    body = MiniMaxVoiceCloneRequest(
        clone_audio=MiniMaxUploadedAsset(
            contentBase64=base64.b64encode(b"clone-audio").decode("ascii"),
            filename="clone.wav",
        ),
        prompt_audio=MiniMaxUploadedAsset(
            contentBase64=base64.b64encode(b"prompt-audio").decode("ascii"),
            filename="prompt.mp3",
        ),
        voice_id="MiniMaxDemo001",
    )

    response = minimax_controller.clone_voice(object(), body)

    assert response["status"] == 200
    assert seen_paths
    assert all(not Path(path).exists() for path in seen_paths)


def test_clone_preview_passes_trial_text_without_activation(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(minimax_controller.base, "get_task_id", lambda request: "request-preview")
    monkeypatch.setattr(
        minimax_controller.utils,
        "storage_dir",
        lambda sub_dir="", create=False: _storage_dir(tmp_path, sub_dir, create),
    )

    def fake_clone_voice(**kwargs):
        captured.update(kwargs)
        return {"voice_id": kwargs["voice_id"], "activated": False}

    monkeypatch.setattr(minimax_controller.minimax, "clone_voice", fake_clone_voice)
    body = MiniMaxVoiceCloneRequest(
        activate=False,
        clone_audio=MiniMaxUploadedAsset(
            contentBase64=base64.b64encode(b"audio").decode(),
            filename="clone.wav",
        ),
        trial_text="试听文本",
        voice_id="MiniMaxDemo001",
    )

    response = minimax_controller.clone_voice(object(), body)

    assert response["status"] == 200
    assert captured["trial_text"] == "试听文本"
    assert response["data"]["activated"] is False


def test_tts_honors_non_library_output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(minimax_controller.base, "get_task_id", lambda request: "request-3")
    monkeypatch.setattr(
        minimax_controller.utils,
        "storage_dir",
        lambda sub_dir="", create=False: _storage_dir(tmp_path, sub_dir, create),
    )

    def fake_t2a(text, voice_id, output_file, **kwargs):
        Path(output_file).write_bytes(b"tts-audio")
        return {"file": output_file, "voice_id": voice_id}

    monkeypatch.setattr(minimax_controller.minimax, "t2a_sync", fake_t2a)

    response = minimax_controller.generate_tts(
        object(),
        MiniMaxTtsRequest(
            save_as_custom_audio=False,
            text="Hermes MiniMax TTS",
            voice_id="MiniMaxDemo001",
        ),
    )

    assert response["status"] == 200
    assert response["data"]["audio"]["file"].startswith("storage/minimax/tts/")
    assert Path(response["data"]["file"]).read_bytes() == b"tts-audio"


def test_tts_marks_matching_local_clone_activated(tmp_path, monkeypatch):
    monkeypatch.setattr(minimax_controller.base, "get_task_id", lambda request: "request-activate-clone")
    monkeypatch.setattr(
        minimax_controller.utils,
        "storage_dir",
        lambda sub_dir="", create=False: _storage_dir(tmp_path, sub_dir, create),
    )
    metadata_path = tmp_path / "storage" / "minimax" / "voices" / "MiniMaxDemo001" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        '{"activated": false, "voice_id": "MiniMaxDemo001"}',
        encoding="utf-8",
    )

    def fake_t2a(text, voice_id, output_file, **kwargs):
        Path(output_file).write_bytes(b"activated-audio")
        return {"file": output_file, "voice_id": voice_id}

    monkeypatch.setattr(minimax_controller.minimax, "t2a_sync", fake_t2a)

    response = minimax_controller.generate_tts(
        object(),
        MiniMaxTtsRequest(text="正式激活", voice_id="MiniMaxDemo001"),
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert response["status"] == 200
    assert metadata["activated"] is True


def test_request_models_defer_model_defaults_to_service_config():
    clone = MiniMaxVoiceCloneRequest(
        clone_audio=MiniMaxUploadedAsset(contentBase64="YQ==", filename="clone.wav"),
        voice_id="MiniMaxDemo001",
    )
    tts = MiniMaxTtsRequest(text="hello", voice_id="MiniMaxDemo001")
    music = MiniMaxMusicRequest(prompt="instrumental", is_instrumental=True)

    assert clone.activate is False
    assert clone.model == ""
    assert tts.model == ""
    assert music.model == ""


def test_clone_legacy_activate_flag_does_not_force_trial_tts(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(minimax_controller.base, "get_task_id", lambda request: "request-activation")
    monkeypatch.setattr(
        minimax_controller.utils,
        "storage_dir",
        lambda sub_dir="", create=False: _storage_dir(tmp_path, sub_dir, create),
    )
    monkeypatch.setattr(
        minimax_controller.minimax,
        "clone_voice",
        lambda **kwargs: captured.update(kwargs) or {"voice_id": kwargs["voice_id"], "activated": False},
    )
    body = MiniMaxVoiceCloneRequest(
        activate=True,
        clone_audio=MiniMaxUploadedAsset(contentBase64="YQ==", filename="clone.wav"),
        voice_id="MiniMaxDemo001",
    )

    response = minimax_controller.clone_voice(object(), body)

    assert response["status"] == 200
    assert response["data"]["activated"] is False
    assert captured["trial_text"] == ""


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (ValueError("bad input"), 400),
        (requests.Timeout("upstream timeout"), 504),
        (requests.ConnectionError("upstream unavailable"), 502),
        (RuntimeError("provider rejected request"), 502),
        (OSError("disk full"), 500),
    ],
)
def test_operation_error_classifies_client_dependency_and_internal_failures(exc, expected_status):
    error = minimax_controller._operation_error("request-4", exc)

    assert error.status_code == expected_status


def test_minimax_routes_require_managed_sidecar_token(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.asgi import app

    monkeypatch.setenv("MONEYPRINTER_HERMES_TOKEN", "sidecar-test-token")
    monkeypatch.setattr(
        minimax_controller.utils,
        "storage_dir",
        lambda sub_dir="", create=False: _storage_dir(tmp_path, sub_dir, create),
    )
    client = TestClient(app)

    assert client.get("/api/v1/minimax/voices").status_code == 401
    authenticated = client.get(
        "/api/v1/minimax/voices",
        headers={"X-Hermes-MoneyPrinter-Token": "sidecar-test-token"},
    )

    assert authenticated.status_code == 200


def test_minimax_routes_fail_closed_without_sidecar_auth(monkeypatch):
    from fastapi.testclient import TestClient

    from app.asgi import app

    monkeypatch.delenv("MONEYPRINTER_HERMES_TOKEN", raising=False)
    monkeypatch.setattr(minimax_controller.config, "app", {}, raising=False)

    response = TestClient(app).get("/api/v1/minimax/voices")

    assert response.status_code == 503
