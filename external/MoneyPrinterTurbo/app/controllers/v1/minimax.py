import base64
import binascii
import hmac
import json
import os
import re
from pathlib import Path

import requests
from fastapi import Depends, Query, Request

from app.controllers import base
from app.controllers.v1.base import new_router
from app.config import config
from app.models.exception import HttpException
from app.models.schema import MiniMaxLyricsRequest, MiniMaxMusicRequest, MiniMaxTtsRequest, MiniMaxVoiceCloneRequest
from app.services import minimax
from app.utils import utils

def _verify_hermes_token(request: Request) -> None:
    managed_token = os.getenv("MONEYPRINTER_HERMES_TOKEN", "")
    expected = managed_token or str(config.app.get("api_key") or "")
    if not expected:
        raise HttpException(
            task_id=base.get_task_id(request),
            status_code=503,
            message="MiniMax sidecar authentication is not configured",
        )
    header_name = "X-Hermes-MoneyPrinter-Token" if managed_token else "X-Api-Key"
    supplied = request.headers.get(header_name, "")
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HttpException(
            task_id=base.get_task_id(request),
            status_code=401,
            message="invalid Hermes sidecar token",
        )


router = new_router(dependencies=[Depends(_verify_hermes_token)])


def _safe_filename(filename: str, fallback: str) -> str:
    name = Path(str(filename or fallback).replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return name or fallback


def _operation_error(request_id: str, exc: Exception) -> HttpException:
    if isinstance(exc, ValueError):
        status_code = 400
    elif isinstance(exc, requests.Timeout):
        status_code = 504
    elif isinstance(exc, (requests.RequestException, RuntimeError)):
        status_code = 502
    else:
        status_code = 500
    return HttpException(task_id=request_id, status_code=status_code, message=f"{request_id}: {str(exc)}")


def _uploaded_file(input_data, request_id: str, label: str) -> str:
    if not input_data:
        raise HttpException(task_id=request_id, status_code=400, message=f"{request_id}: {label} is required")
    if input_data.contentBase64:
        content = str(input_data.contentBase64)
        if "," in content and content.lstrip().lower().startswith("data:"):
            content = content.split(",", 1)[1]
        try:
            raw = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError):
            raise HttpException(task_id=request_id, status_code=400, message=f"{request_id}: {label} contentBase64 is invalid")
        if len(raw) > minimax.MAX_MINIMAX_AUDIO_BYTES:
            raise HttpException(task_id=request_id, status_code=400, message=f"{request_id}: {label} exceeds the MiniMax 20 MB limit")
        filename = _safe_filename(input_data.filename, f"{label}.mp3")
        if Path(filename).suffix.lower() not in minimax.MINIMAX_AUDIO_EXTS:
            raise HttpException(task_id=request_id, status_code=400, message=f"{request_id}: {label} must be mp3, m4a or wav")
        upload_dir = Path(utils.storage_dir("minimax/uploads", create=True))
        target = upload_dir / f"{utils.get_uuid(remove_hyphen=True)}-{filename}"
        target.write_bytes(raw)
        target.chmod(0o600)
        return str(target)
    raise HttpException(task_id=request_id, status_code=400, message=f"{request_id}: {label} contentBase64 is required")


def _mark_local_voice_activated(voice_id: str, activation_audio_file: str) -> None:
    metadata_path = Path(
        utils.storage_dir(f"minimax/voices/{minimax.validate_voice_id(voice_id)}/metadata.json")
    )
    if not metadata_path.is_file():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(metadata, dict):
        return
    metadata["activated"] = True
    metadata["activationAudioFile"] = activation_audio_file
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/minimax/voices")
def list_voices(voice_type: str = Query("all")):
    request_id = utils.get_uuid(remove_hyphen=False)
    try:
        result = minimax.list_voices(voice_type)
        provider_ids = {str(voice.get("id") or "") for voice in result["voices"]}
        voices_dir = Path(utils.storage_dir("minimax/voices", create=True))
        for metadata_path in sorted(voices_dir.glob("*/metadata.json"), key=lambda p: p.parent.name.lower()):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(metadata, dict):
                continue
            voice_id = str(metadata.get("voice_id") or metadata_path.parent.name).strip()
            if not voice_id or voice_id in provider_ids:
                continue
            result["voices"].append(
                {
                    "category": "local_preview",
                    "id": voice_id,
                    "name": str(metadata.get("display_name") or voice_id),
                    "providerConfirmed": False,
                }
            )
        return utils.get_response(200, result)
    except Exception as exc:
        raise _operation_error(request_id, exc)


@router.post("/minimax/voices/clone")
def clone_voice(request: Request, body: MiniMaxVoiceCloneRequest):
    request_id = base.get_task_id(request)
    temporary_files = []
    try:
        voice_id = minimax.validate_voice_id(body.voice_id)
        output_dir = Path(utils.storage_dir(f"minimax/voices/{voice_id}", create=True))
        clone_audio_file = _uploaded_file(body.clone_audio, request_id, "clone_audio")
        temporary_files.append(clone_audio_file)
        prompt_audio_file = ""
        if body.prompt_audio:
            prompt_audio_file = _uploaded_file(body.prompt_audio, request_id, "prompt_audio")
            temporary_files.append(prompt_audio_file)
        result = minimax.clone_voice(
            voice_id=voice_id,
            clone_audio_file=clone_audio_file,
            prompt_audio_file=prompt_audio_file,
            prompt_text=body.prompt_text,
            trial_text=body.trial_text,
            output_dir=str(output_dir),
            model=body.model,
        )
        return utils.get_response(200, result)
    except HttpException:
        raise
    except Exception as exc:
        raise _operation_error(request_id, exc)
    finally:
        for temporary_file in temporary_files:
            Path(temporary_file).unlink(missing_ok=True)


@router.post("/minimax/tts")
def generate_tts(request: Request, body: MiniMaxTtsRequest):
    request_id = base.get_task_id(request)
    try:
        filename = f"minimax-tts-{utils.get_uuid(remove_hyphen=True)}.mp3"
        storage_subdir = "custom_audio" if body.save_as_custom_audio else "minimax/tts"
        output_file = Path(utils.storage_dir(storage_subdir, create=True)) / filename
        result = minimax.t2a_sync(
            body.text,
            body.voice_id,
            str(output_file),
            model=body.model,
            speed=body.speed,
            vol=body.volume,
        )
        _mark_local_voice_activated(body.voice_id, str(result.get("file") or output_file))
        result["audio"] = {"file": f"storage/{storage_subdir}/{filename}", "name": filename}
        return utils.get_response(200, result)
    except Exception as exc:
        raise _operation_error(request_id, exc)


@router.post("/minimax/lyrics")
def generate_lyrics(request: Request, body: MiniMaxLyricsRequest):
    request_id = base.get_task_id(request)
    try:
        result = minimax.generate_lyrics(mode=body.mode, prompt=body.prompt, lyrics=body.lyrics, title=body.title)
        return utils.get_response(200, result)
    except Exception as exc:
        raise _operation_error(request_id, exc)


@router.post("/minimax/music")
def generate_music(request: Request, body: MiniMaxMusicRequest):
    request_id = base.get_task_id(request)
    try:
        result = minimax.generate_music(
            prompt=body.prompt,
            is_instrumental=body.is_instrumental,
            lyrics=body.lyrics,
            lyrics_optimizer=body.lyrics_optimizer,
            model=body.model,
            save_as_bgm=body.save_as_bgm,
            filename_slug=body.prompt,
        )
        return utils.get_response(200, result)
    except Exception as exc:
        raise _operation_error(request_id, exc)
