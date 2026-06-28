from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from agent.tts_provider import TTSProvider


PLUGIN_DIR = Path(__file__).resolve().parent
HERMES_ROOT = PLUGIN_DIR.parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:17493"
DEFAULT_CLIENT_ID = "hermes"
DEFAULT_PROFILE = "hakua"
DEFAULT_LANGUAGE = "ja"
DEFAULT_ENGINE = ""
DEFAULT_TIMEOUT = 600.0
DEFAULT_MODEL = "voicebox"
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_IRODORI_HAKUA_REF = HERMES_ROOT.parent / "irodori-tts-server" / "voices" / "hakua.ogg"
DEFAULT_HAKUA_REFERENCE_TEXT = "はくあの参考音声です。"

VOICEBOX_ENGINES = (
    "qwen",
    "qwen_custom_voice",
    "luxtts",
    "chatterbox",
    "chatterbox_turbo",
    "tada",
    "kokoro",
)

TERMINAL_GENERATION_STATUSES = frozenset({"completed", "failed", "not_found"})


@dataclass(frozen=True)
class VoiceboxSettings:
    base_url: str
    client_id: str
    profile: str
    language: str
    engine: str
    timeout: float
    personality: bool
    poll_interval: float
    irodori_ref_audio: str
    auto_import_profile: bool
    reference_text: str
    irodori_ref_audio: str
    auto_import_profile: bool
    reference_text: str


def _load_tts_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return {}
    tts = config.get("tts", {})
    return tts if isinstance(tts, dict) else {}


def _voicebox_config(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    tts = tts_config if isinstance(tts_config, dict) else _load_tts_section()
    section = tts.get("voicebox", {})
    return section if isinstance(section, dict) else {}


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def settings(tts_config: dict[str, Any] | None = None) -> VoiceboxSettings:
    cfg = _voicebox_config(tts_config)
    base_url = (
        cfg.get("base_url")
        or cfg.get("url")
        or os.environ.get("VOICEBOX_BASE_URL")
        or DEFAULT_BASE_URL
    )
    client_id = (
        cfg.get("client_id")
        or os.environ.get("VOICEBOX_CLIENT_ID")
        or DEFAULT_CLIENT_ID
    )
    language = (
        cfg.get("language")
        or os.environ.get("VOICEBOX_LANGUAGE")
        or DEFAULT_LANGUAGE
    )
    engine = (
        cfg.get("engine")
        or os.environ.get("VOICEBOX_ENGINE")
        or DEFAULT_ENGINE
    )
    timeout = _float_value(
        cfg.get("timeout", os.environ.get("VOICEBOX_TIMEOUT")),
        DEFAULT_TIMEOUT,
    )
    personality = _bool_value(
        cfg.get("personality", os.environ.get("VOICEBOX_PERSONALITY")),
        False,
    )
    poll_interval = _float_value(
        cfg.get("poll_interval", os.environ.get("VOICEBOX_POLL_INTERVAL")),
        DEFAULT_POLL_INTERVAL,
    )
    irodori_ref = (
        cfg.get("irodori_ref_audio")
        or os.environ.get("VOICEBOX_IRODORI_REF_AUDIO")
        or str(DEFAULT_IRODORI_HAKUA_REF)
    )
    auto_import = _bool_value(
        cfg.get("auto_import_profile", os.environ.get("VOICEBOX_AUTO_IMPORT_PROFILE")),
        True,
    )
    reference_text = (
        cfg.get("reference_text")
        or os.environ.get("VOICEBOX_REFERENCE_TEXT")
        or DEFAULT_HAKUA_REFERENCE_TEXT
    )
    default_profile = (
        cfg.get("profile")
        or cfg.get("voice")
        or os.environ.get("VOICEBOX_PROFILE")
        or DEFAULT_PROFILE
    )
    return VoiceboxSettings(
        base_url=str(base_url).rstrip("/"),
        client_id=str(client_id).strip() or DEFAULT_CLIENT_ID,
        profile=str(default_profile).strip(),
        language=str(language).strip() or DEFAULT_LANGUAGE,
        engine=str(engine).strip(),
        timeout=timeout,
        personality=personality,
        poll_interval=max(0.25, poll_interval),
        irodori_ref_audio=str(irodori_ref),
        auto_import_profile=auto_import,
        reference_text=str(reference_text),
    )


def _request_headers(cfg: VoiceboxSettings) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if cfg.client_id:
        headers["X-Voicebox-Client-Id"] = cfg.client_id
    return headers


def _request_json(
    method: str,
    path: str,
    *,
    cfg: VoiceboxSettings | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    resolved = cfg or settings()
    url = f"{resolved.base_url}{path}"
    headers = kwargs.pop("headers", {})
    merged_headers = {**_request_headers(resolved), **headers}
    response = requests.request(
        method,
        url,
        headers=merged_headers,
        timeout=timeout if timeout is not None else min(resolved.timeout, 30.0),
        **kwargs,
    )
    response.raise_for_status()
    if not response.content:
        return {}
    return response.json()


def status_payload(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = settings(tts_config)
    endpoint = f"{cfg.base_url}/health"
    profiles_endpoint = f"{cfg.base_url}/profiles"
    ref_path = Path(cfg.irodori_ref_audio).expanduser()
    try:
        health = _request_json("GET", "/health", cfg=cfg, timeout=min(cfg.timeout, 10.0))
        reachable = True
        health_error = ""
    except Exception as exc:
        health = {}
        reachable = False
        health_error = str(exc)

    profile_count = 0
    profiles_error = ""
    if reachable:
        try:
            profiles = _request_json(
                "GET",
                "/profiles",
                cfg=cfg,
                timeout=min(cfg.timeout, 10.0),
            )
            profile_count = len(profiles) if isinstance(profiles, list) else 0
        except Exception as exc:
            profiles_error = str(exc)

    return {
        "ok": reachable,
        "provider": "voicebox",
        "available": reachable,
        "server": {
            "endpoint": endpoint,
            "reachable": reachable,
            "health": health,
            "error": health_error,
        },
        "profiles": {
            "endpoint": profiles_endpoint,
            "count": profile_count,
            "error": profiles_error,
        },
        "defaults": {
            "base_url": cfg.base_url,
            "client_id": cfg.client_id,
            "profile": cfg.profile,
            "language": cfg.language,
            "engine": cfg.engine or None,
            "timeout": cfg.timeout,
            "personality": cfg.personality,
            "auto_import_profile": cfg.auto_import_profile,
        },
        "paths": {
            "irodori_ref_audio": str(ref_path),
            "irodori_ref_present": ref_path.is_file(),
        },
    }


def list_profiles(tts_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = settings(tts_config)
    try:
        payload = _request_json("GET", "/profiles", cfg=cfg, timeout=min(cfg.timeout, 15.0))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def list_voices(tts_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    voices: list[dict[str, Any]] = []
    for profile in list_profiles(tts_config):
        profile_id = str(profile.get("id") or "").strip()
        if not profile_id:
            continue
        name = str(profile.get("name") or profile_id)
        language = str(profile.get("language") or "")
        description = str(profile.get("description") or "").strip()
        display = name if not description else f"{name} — {description}"
        voices.append(
            {
                "id": profile_id,
                "display": display,
                "language": language,
                "profile_name": name,
            }
        )
    return voices


def list_models() -> list[dict[str, Any]]:
    return [
        {
            "id": engine,
            "display": engine.replace("_", " ").title(),
            "languages": ["multilingual"],
        }
        for engine in VOICEBOX_ENGINES
    ]


def _find_profile(name: str, tts_config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    target = name.strip().lower()
    if not target:
        return None
    for profile in list_profiles(tts_config):
        profile_name = str(profile.get("name") or "").lower()
        profile_id = str(profile.get("id") or "").lower()
        if target in {profile_name, profile_id}:
            return profile
    return None


def ensure_hakua_profile(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a Voicebox clone profile from Irodori hakua.ogg when missing."""
    cfg = settings(tts_config)
    profile_name = cfg.profile or "hakua"
    existing = _find_profile(profile_name, tts_config)
    if existing is not None:
        return {
            "ok": True,
            "created": False,
            "profile": str(existing.get("name") or existing.get("id") or profile_name),
            "profile_id": existing.get("id"),
        }

    ref_path = Path(cfg.irodori_ref_audio).expanduser()
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Irodori Hakua reference audio not found: {ref_path}. "
            "Set tts.voicebox.irodori_ref_audio or place hakua.ogg under irodori-tts-server/voices/."
        )

    created = _request_json(
        "POST",
        "/profiles",
        cfg=cfg,
        json={
            "name": profile_name,
            "language": cfg.language,
            "description": "Imported from Irodori TTS hakua.ogg reference audio.",
        },
        timeout=min(cfg.timeout, 30.0),
    )
    if not isinstance(created, dict) or not created.get("id"):
        raise RuntimeError(f"Voicebox profile creation failed: {created}")

    profile_id = str(created["id"])
    with ref_path.open("rb") as handle:
        response = requests.post(
            f"{cfg.base_url}/profiles/{profile_id}/samples",
            headers=_request_headers(cfg),
            files={"file": (ref_path.name, handle, "audio/ogg")},
            data={"reference_text": cfg.reference_text},
            timeout=cfg.timeout,
        )
    response.raise_for_status()

    return {
        "ok": True,
        "created": True,
        "profile": profile_name,
        "profile_id": profile_id,
        "reference_audio": str(ref_path),
    }


def _resolved_output_path(output_path: str | Path | None) -> Path:
    if output_path:
        path = Path(output_path).expanduser()
    else:
        cache_dir = Path(
            os.environ.get(
                "HERMES_AUDIO_CACHE_DIR",
                Path.home() / "AppData" / "Local" / "hermes" / "audio_cache",
            )
        )
        path = cache_dir / f"voicebox_plugin_{time.strftime('%Y%m%d-%H%M%S')}.wav"
    if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        path = path.with_suffix(".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _wait_for_generation(
    generation_id: str,
    *,
    cfg: VoiceboxSettings,
) -> dict[str, Any]:
    deadline = time.monotonic() + cfg.timeout
    last_payload: dict[str, Any] = {"id": generation_id, "status": "generating"}
    while time.monotonic() < deadline:
        payload = _request_json(
            "GET",
            f"/history/{generation_id}",
            cfg=cfg,
            timeout=min(cfg.timeout, 15.0),
        )
        if not isinstance(payload, dict):
            raise RuntimeError(f"Voicebox returned invalid generation payload for {generation_id}")
        last_payload = payload
        status = str(payload.get("status") or "completed")
        if status in TERMINAL_GENERATION_STATUSES:
            if status == "failed":
                error = payload.get("error") or "Voicebox generation failed"
                raise RuntimeError(str(error))
            if status == "not_found":
                raise RuntimeError(f"Voicebox generation not found: {generation_id}")
            return payload
        time.sleep(cfg.poll_interval)
    raise TimeoutError(
        f"Voicebox generation timed out after {cfg.timeout:.0f}s (last status={last_payload.get('status')})"
    )


def _download_generation_audio(
    generation_id: str,
    destination: Path,
    *,
    cfg: VoiceboxSettings,
) -> None:
    url = f"{cfg.base_url}/audio/{generation_id}"
    response = requests.get(
        url,
        headers=_request_headers(cfg),
        timeout=cfg.timeout,
        stream=True,
    )
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                handle.write(chunk)
    if destination.stat().st_size <= 0:
        raise RuntimeError(f"Voicebox created empty audio: {destination}")


def _find_profile(name: str, tts_config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    target = name.strip().lower()
    if not target:
        return None
    for profile in list_profiles(tts_config):
        profile_name = str(profile.get("name") or "").lower()
        profile_id = str(profile.get("id") or "").lower()
        if target in {profile_name, profile_id}:
            return profile
    return None


def ensure_hakua_profile(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a Voicebox profile from Irodori hakua.ogg when missing."""
    cfg = settings(tts_config)
    profile_name = cfg.profile or "hakua"
    existing = _find_profile(profile_name, tts_config)
    if existing is not None:
        return {
            "ok": True,
            "created": False,
            "profile": str(existing.get("name") or existing.get("id") or profile_name),
            "profile_id": existing.get("id"),
        }

    ref_path = Path(cfg.irodori_ref_audio).expanduser()
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Irodori Hakua reference audio not found: {ref_path}. "
            "Place hakua.ogg under irodori-tts-server/voices/ or set tts.voicebox.irodori_ref_audio."
        )

    created = _request_json(
        "POST",
        "/profiles",
        cfg=cfg,
        json={
            "name": profile_name,
            "language": cfg.language,
            "description": "Imported from Irodori TTS hakua.ogg reference audio.",
        },
        timeout=min(cfg.timeout, 30.0),
    )
    if not isinstance(created, dict) or not created.get("id"):
        raise RuntimeError(f"Voicebox profile creation failed: {created}")

    profile_id = str(created["id"])
    with ref_path.open("rb") as handle:
        response = requests.post(
            f"{cfg.base_url}/profiles/{profile_id}/samples",
            headers=_request_headers(cfg),
            files={"file": (ref_path.name, handle, "audio/ogg")},
            data={"reference_text": cfg.reference_text},
            timeout=cfg.timeout,
        )
    response.raise_for_status()

    return {
        "ok": True,
        "created": True,
        "profile": profile_name,
        "profile_id": profile_id,
        "reference_audio": str(ref_path),
    }


def synthesize_text(
    text: str,
    output_path: str | Path | None = None,
    voice: str | None = None,
    model: str | None = None,
    speed: float | None = None,
    language: str | None = None,
    personality: bool | None = None,
    tts_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del speed

    if not text or not text.strip():
        raise ValueError("text must not be empty")

    cfg = settings(tts_config)
    if cfg.auto_import_profile:
        ensure_hakua_profile(tts_config)

    profile = (voice or cfg.profile or "").strip()
    if not profile:
        profiles = list_profiles(tts_config)
        if len(profiles) == 1:
            profile = str(profiles[0].get("name") or profiles[0].get("id") or "")
        if not profile:
            raise ValueError(
                "No Voicebox profile configured. Set tts.voicebox.profile or pass voice=<name or id>."
            )

    body: dict[str, Any] = {
        "text": text,
        "profile": profile,
        "language": language or cfg.language,
    }
    engine = (model or cfg.engine or "").strip()
    if engine:
        body["engine"] = engine
    personality_flag = cfg.personality if personality is None else bool(personality)
    if personality_flag:
        body["personality"] = True

    speak_payload = _request_json(
        "POST",
        "/speak",
        cfg=cfg,
        json=body,
        timeout=min(cfg.timeout, 30.0),
    )
    if not isinstance(speak_payload, dict):
        raise RuntimeError("Voicebox /speak returned a non-object response")
    generation_id = str(speak_payload.get("id") or "").strip()
    if not generation_id:
        raise RuntimeError(f"Voicebox /speak did not return a generation id: {speak_payload}")

    completed = _wait_for_generation(generation_id, cfg=cfg)
    destination = _resolved_output_path(output_path)
    _download_generation_audio(generation_id, destination, cfg=cfg)

    resolved_engine = str(completed.get("engine") or engine or DEFAULT_MODEL)
    return {
        "ok": True,
        "provider": "voicebox",
        "file_path": str(destination),
        "format": destination.suffix.lstrip(".").lower() or "wav",
        "voice": profile,
        "model": resolved_engine,
        "generation_id": generation_id,
        "duration": completed.get("duration"),
        "media_tag": f"MEDIA:{destination}",
    }


def transcribe_audio(
    audio_path: str | Path,
    *,
    language: str | None = None,
    model: str | None = None,
    tts_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = settings(tts_config)
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    data: dict[str, Any] = {}
    if language:
        data["language"] = language
    if model:
        data["model"] = model

    with path.open("rb") as handle:
        response = requests.post(
            f"{cfg.base_url}/transcribe",
            headers=_request_headers(cfg),
            files={"file": (path.name, handle, "application/octet-stream")},
            data=data,
            timeout=cfg.timeout,
        )
    if response.status_code >= 400:
        detail = response.text.strip()
        try:
            detail = json.dumps(response.json(), ensure_ascii=False)
        except Exception:
            pass
        raise RuntimeError(f"Voicebox transcribe failed ({response.status_code}): {detail}")

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Voicebox /transcribe returned a non-object response")
    return {
        "ok": True,
        "provider": "voicebox",
        "text": payload.get("text", ""),
        "duration": payload.get("duration"),
        "model": model,
        "language": language,
    }


class VoiceboxTTSProvider(TTSProvider):
    name = "voicebox"
    display_name = "Voicebox"
    voice_compatible = True

    def is_available(self) -> bool:
        return bool(status_payload()["available"])

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "Voicebox",
            "badge": "local · clone · dictate",
            "tag": "Local AI voice studio via Voicebox REST API (https://github.com/zapabob/voicebox)",
            "env_vars": [],
        }

    def list_voices(self) -> list[dict[str, Any]]:
        return list_voices()

    def default_voice(self) -> str | None:
        profile = settings().profile
        return profile or None

    def list_models(self) -> list[dict[str, Any]]:
        return list_models()

    def default_model(self) -> str | None:
        engine = settings().engine
        return engine or DEFAULT_MODEL

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float | None = None,
        format: str | None = None,
        **_: Any,
    ) -> str:
        del format
        result = synthesize_text(
            text=text,
            output_path=output_path,
            voice=voice,
            model=model,
            speed=speed,
        )
        return str(result["file_path"])
