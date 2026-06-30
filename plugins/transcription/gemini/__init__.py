"""Gemini speech-to-text plugin."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import requests

from agent.transcription_provider import TranscriptionProvider
from hermes_cli.config import get_env_value


DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
INLINE_AUDIO_LIMIT_BYTES = 20 * 1024 * 1024


def _api_key() -> str | None:
    value = (
        get_env_value("GEMINI_STT_API_KEY")
        or get_env_value("GEMINI_API_KEY")
        or get_env_value("GOOGLE_API_KEY")
    )
    return value.strip() if isinstance(value, str) and value.strip() else None


def _base_url() -> str:
    value = (
        get_env_value("GEMINI_STT_BASE_URL")
        or get_env_value("GEMINI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return str(value).strip().rstrip("/") or DEFAULT_BASE_URL


def _failure(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "transcript": "",
        "provider": "gemini",
        "error": message,
    }


def _extract_text(payload: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "".join(texts).strip()


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    except Exception:
        pass
    return response.text[:300].strip() or f"HTTP {response.status_code}"


class GeminiTranscriptionProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Gemini STT"

    def is_available(self) -> bool:
        return bool(_api_key())

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "Gemini 3.1 Flash Lite",
            },
            {
                "id": "gemini-3-flash-preview",
                "display": "Gemini 3 Flash",
            },
            {
                "id": "gemini-2.5-flash",
                "display": "Gemini 2.5 Flash",
            },
        ]

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "cloud",
            "tag": "Google Gemini audio transcription",
            "env_vars": [
                {
                    "key": "GEMINI_API_KEY",
                    "prompt": "Google AI Studio API key",
                    "url": "https://aistudio.google.com/app/apikey",
                }
            ],
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        del extra
        api_key = _api_key()
        if not api_key:
            return _failure("Gemini STT requires GEMINI_API_KEY or GOOGLE_API_KEY.")

        audio_path = Path(file_path)
        try:
            audio_size = audio_path.stat().st_size
        except OSError as exc:
            return _failure(f"Gemini STT could not read audio file: {exc}")
        if audio_size > INLINE_AUDIO_LIMIT_BYTES:
            return _failure(
                "Gemini STT inline audio is limited to 20 MB; use a shorter audio file."
            )

        mime_type = mimetypes.guess_type(audio_path.name)[0] or "audio/mpeg"
        prompt = "Transcribe this audio. Return only the transcript text."
        if language:
            prompt += f" The expected language is {language}."

        try:
            audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
            selected_model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
            response = requests.post(
                f"{_base_url()}/models/{selected_model}:generateContent",
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": audio_b64,
                                    }
                                },
                            ],
                        }
                    ]
                },
                timeout=120,
            )
            if response.status_code != 200:
                return _failure(
                    f"Gemini STT API error (HTTP {response.status_code}): {_error_detail(response)}"
                )
            transcript = _extract_text(response.json())
        except Exception as exc:
            return _failure(f"Gemini STT failed: {exc}")

        if not transcript:
            return _failure("Gemini returned an empty transcript.")

        return {
            "success": True,
            "transcript": transcript,
            "provider": self.name,
        }


def register(ctx: Any) -> None:
    ctx.register_transcription_provider(GeminiTranscriptionProvider())
