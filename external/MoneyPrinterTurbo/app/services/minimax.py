import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import toml

from app.config import config
from app.utils import utils


VOICE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{6,254}[A-Za-z0-9]$")
MAX_MINIMAX_AUDIO_BYTES = 20 * 1024 * 1024
MINIMAX_AUDIO_EXTS = frozenset({".m4a", ".mp3", ".wav"})


def get_minimax_config() -> dict[str, Any]:
    minimax_config = getattr(config, "minimax", {}) or {}
    app_config = getattr(config, "app", {}) or {}
    try:
        live_config = toml.load(config.config_file)
        if isinstance(live_config.get("app"), dict):
            app_config = live_config["app"]
        if isinstance(live_config.get("minimax"), dict):
            minimax_config = live_config["minimax"]
    except (OSError, toml.TomlDecodeError):
        pass

    return {
        "api_key": minimax_config.get("api_key") or app_config.get("minimax_api_key", ""),
        "base_url": minimax_config.get("base_url") or app_config.get("minimax_base_url", "https://api.minimaxi.com"),
        "music_model": minimax_config.get("music_model", "music-2.6-free"),
        "t2a_model": minimax_config.get("t2a_model", "speech-2.8-hd"),
        "voice_clone_model": minimax_config.get("voice_clone_model", "speech-2.8-hd"),
    }


def _headers() -> dict[str, str]:
    api_key = str(get_minimax_config().get("api_key") or "").strip()
    if not api_key:
        raise ValueError("MiniMax API key is not configured")
    return {"Authorization": f"Bearer {api_key}"}


def _endpoint(path: str) -> str:
    base_url = str(get_minimax_config().get("base_url") or "https://api.minimaxi.com").rstrip("/")
    if base_url.endswith("/v1") and path.startswith("/v1/"):
        return f"{base_url}{path[3:]}"
    return f"{base_url}{path}"


def _ensure_success(payload: dict[str, Any]) -> None:
    base_resp = payload.get("base_resp")
    if not isinstance(base_resp, dict):
        return
    status_code = base_resp.get("status_code")
    if status_code in (None, 0, "0"):
        return
    message = base_resp.get("status_msg") or base_resp.get("message") or "MiniMax request failed"
    raise RuntimeError(str(message))


def validate_voice_id(voice_id: str) -> str:
    value = str(voice_id or "").strip()
    if not 8 <= len(value) <= 256:
        raise ValueError("voice_id 长度必须为 8-256")
    if not value[0].isalpha() or not value[0].isascii():
        raise ValueError("voice_id 首字符必须是英文字母")
    if value[-1] in {"-", "_"}:
        raise ValueError("voice_id 末位不能是 - 或 _")
    if not VOICE_ID_RE.match(value):
        raise ValueError("voice_id 只能包含英文字母、数字、- 和 _")
    return value


def list_voices(voice_type: str = "all") -> dict[str, Any]:
    response = requests.post(
        _endpoint("/v1/get_voice"),
        json={"voice_type": str(voice_type or "all")},
        headers=_headers(),
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    _ensure_success(payload)
    voices = []
    for field, category in (
        ("system_voice", "system"),
        ("voice_cloning", "voice_cloning"),
        ("voice_generation", "voice_generation"),
    ):
        for raw in payload.get(field) or []:
            if not isinstance(raw, dict) or not raw.get("voice_id"):
                continue
            voice_id = str(raw["voice_id"])
            voices.append(
                {
                    "category": category,
                    "id": voice_id,
                    "name": str(raw.get("voice_name") or voice_id),
                    "providerConfirmed": True,
                }
            )
    return {"voice_type": str(voice_type or "all"), "voices": voices}


def _write_audio_payload(payload: dict[str, Any], output_file: str) -> str:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    audio = data.get("audio") or data.get("audio_content")
    url = data.get("audio_url") or data.get("url")
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, str) and audio:
        target.write_bytes(bytes.fromhex(audio))
        return str(target)

    if isinstance(url, str) and url:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)
        return str(target)

    raise RuntimeError("MiniMax response did not include audio data")


def _download_public_audio(url: str, output_file: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return str(target)


def upload_file(path: str, purpose: str) -> int:
    source = Path(path).expanduser().resolve(strict=True)
    if purpose not in {"prompt_audio", "voice_clone"}:
        raise ValueError(f"Unsupported MiniMax upload purpose: {purpose}")
    if source.suffix.lower() not in MINIMAX_AUDIO_EXTS:
        raise ValueError("MiniMax clone audio must be mp3, m4a or wav")
    if source.stat().st_size > MAX_MINIMAX_AUDIO_BYTES:
        raise ValueError("MiniMax clone audio exceeds the 20 MB limit")
    with source.open("rb") as fp:
        response = requests.post(
            _endpoint("/v1/files/upload"),
            data={"purpose": purpose},
            files={"file": (source.name, fp)},
            headers=_headers(),
            timeout=60,
        )
    response.raise_for_status()
    payload = response.json()
    _ensure_success(payload)
    file_data = payload.get("file") if isinstance(payload.get("file"), dict) else {}
    file_id = file_data.get("file_id") or payload.get("file_id")
    if file_id is None:
        raise RuntimeError("MiniMax upload response did not include file_id")
    return int(file_id)


def t2a_sync(
    text: str,
    voice_id: str,
    output_file: str,
    *,
    model: str = "",
    pitch: int = 0,
    speed: float = 1.0,
    vol: float = 1.0,
) -> dict[str, Any]:
    voice_id = validate_voice_id(voice_id)
    text = str(text or "").strip()
    if not text:
        raise ValueError("MiniMax TTS text is required")
    if len(text) > 10000:
        raise ValueError("MiniMax TTS text exceeds 10000 characters")
    speed = float(speed)
    vol = float(vol)
    if not 0.5 <= speed <= 2.0:
        raise ValueError("MiniMax TTS speed must be between 0.5 and 2.0")
    if not 0.01 <= vol <= 10.0:
        raise ValueError("MiniMax TTS volume must be between 0.01 and 10.0")
    model_name = model or str(get_minimax_config().get("t2a_model") or "speech-2.8-hd")
    payload = {
        "model": model_name,
        "text": text,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": speed,
            "vol": vol,
            "pitch": pitch,
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1,
        },
        "output_format": "hex",
    }
    response = requests.post(_endpoint("/v1/t2a_v2"), json=payload, headers=_headers(), timeout=90)
    response.raise_for_status()
    result = response.json()
    _ensure_success(result)
    file_path = _write_audio_payload(result, output_file)
    return {
        "file": file_path,
        "trace_id": result.get("trace_id"),
        "voice_id": voice_id,
    }


def clone_voice(
    *,
    voice_id: str,
    clone_audio_file: str,
    output_dir: str,
    model: str = "",
    prompt_audio_file: str = "",
    prompt_text: str = "",
    trial_text: str = "",
    **extra: Any,
) -> dict[str, Any]:
    voice_id = validate_voice_id(voice_id)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = str(prompt_text or "").strip()
    if prompt_audio_file and not prompt_text:
        raise ValueError("MiniMax prompt audio requires prompt_text")
    if prompt_text and not prompt_audio_file:
        raise ValueError("MiniMax prompt_text requires prompt audio")
    clone_file_id = upload_file(clone_audio_file, "voice_clone")

    payload: dict[str, Any] = {
        "file_id": clone_file_id,
        "model": model or str(get_minimax_config().get("voice_clone_model") or "speech-2.8-hd"),
        "voice_id": voice_id,
    }
    if prompt_audio_file:
        prompt_file_id = upload_file(prompt_audio_file, "prompt_audio")
        payload["clone_prompt"] = {
            "prompt_audio": prompt_file_id,
            "prompt_text": prompt_text,
        }
    if trial_text:
        payload["text"] = str(trial_text).strip()
    for key in ("language_boost", "need_noise_reduction", "need_volume_normalization"):
        if key in extra and extra[key] is not None:
            payload[key] = extra[key]

    response = requests.post(_endpoint("/v1/voice_clone"), json=payload, headers=_headers(), timeout=90)
    response.raise_for_status()
    result = response.json()
    _ensure_success(result)

    metadata = {
        "activated": False,
        "display_name": extra.get("display_name") or voice_id,
        "model": payload["model"],
        "trace_id": result.get("trace_id"),
        "voice_id": voice_id,
        "voiceNameForVideo": f"minimax:{voice_id}",
    }
    metadata_path = target_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if trial_text:
        preview_url = str(result.get("demo_audio") or "").strip()
        try:
            if not preview_url:
                raise RuntimeError("MiniMax clone response did not include demo_audio")
            metadata["trialAudioFile"] = _download_public_audio(
                preview_url,
                str(target_dir / "trial.mp3"),
            )
        except Exception as exc:
            metadata["previewError"] = str(exc)
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return metadata


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return (slug or "music")[:80].rstrip("-._") or "music"


def generate_music(
    *,
    prompt: str,
    is_instrumental: bool = False,
    lyrics: str = "",
    lyrics_optimizer: bool = True,
    model: str = "",
    save_as_bgm: bool = False,
    filename_slug: str = "",
) -> dict[str, Any]:
    prompt = str(prompt or "").strip()
    lyrics = str(lyrics or "").strip()
    if len(prompt) > 2000:
        raise ValueError("MiniMax music prompt exceeds 2000 characters")
    if len(lyrics) > 3500:
        raise ValueError("MiniMax music lyrics exceed 3500 characters")
    if is_instrumental and not prompt:
        raise ValueError("MiniMax instrumental music requires a prompt")
    if not is_instrumental and not lyrics and not lyrics_optimizer:
        raise ValueError("MiniMax vocal music requires lyrics or lyrics_optimizer")
    if lyrics_optimizer and not lyrics and not prompt:
        raise ValueError("MiniMax lyrics_optimizer requires a prompt")
    model_name = model or str(get_minimax_config().get("music_model") or "music-2.6-free")
    payload = {
        "is_instrumental": is_instrumental,
        "lyrics": lyrics,
        "lyrics_optimizer": lyrics_optimizer,
        "model": model_name,
        "audio_setting": {
            "sample_rate": 44100,
            "bitrate": 256000,
            "format": "mp3",
        },
        "output_format": "hex",
        "prompt": prompt,
    }
    response = requests.post(_endpoint("/v1/music_generation"), json=payload, headers=_headers(), timeout=120)
    response.raise_for_status()
    result = response.json()
    _ensure_success(result)

    request_id = utils.get_uuid(remove_hyphen=True)[:12]
    filename = f"minimax-music-{_safe_slug(filename_slug or prompt)}-{request_id}.mp3"
    output_dir = Path(utils.song_dir()) if save_as_bgm else Path(utils.storage_dir("custom_audio", create=True))
    output_file = output_dir / filename
    file_path = _write_audio_payload(result, str(output_file))
    data = {
        "file": file_path,
        "model": model_name,
        "trace_id": result.get("trace_id"),
    }
    if save_as_bgm:
        data["bgm"] = {"file": filename, "name": filename}
    else:
        data["audio"] = {"file": f"storage/custom_audio/{filename}", "name": filename}
    return data


def generate_lyrics(
    *,
    mode: str = "write_full_song",
    prompt: str = "",
    lyrics: str = "",
    title: str = "",
) -> dict[str, Any]:
    mode = str(mode or "").strip()
    prompt = str(prompt or "").strip()
    lyrics = str(lyrics or "").strip()
    title = str(title or "").strip()
    if mode not in {"edit", "write_full_song"}:
        raise ValueError("MiniMax lyrics mode must be edit or write_full_song")
    if len(prompt) > 2000:
        raise ValueError("MiniMax lyrics prompt exceeds 2000 characters")
    if len(lyrics) > 3500:
        raise ValueError("MiniMax lyrics input exceeds 3500 characters")
    if mode == "edit" and not lyrics:
        raise ValueError("MiniMax lyrics edit mode requires existing lyrics")
    payload = {
        "lyrics": lyrics,
        "mode": mode,
        "prompt": prompt,
        "title": title,
    }
    response = requests.post(_endpoint("/v1/lyrics_generation"), json=payload, headers=_headers(), timeout=60)
    response.raise_for_status()
    result = response.json()
    _ensure_success(result)
    return {
        "lyrics": result.get("lyrics"),
        "song_title": result.get("song_title"),
        "style_tags": result.get("style_tags"),
        "trace_id": result.get("trace_id"),
    }
