"""Multipart audio transcription route for the API server adapter."""

import asyncio
import logging
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Dict, Optional

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - mirrors the adapter's optional dependency
    web = None

logger = logging.getLogger("gateway.platforms.api_server")


class AudioTranscriptionsMixin:
    async def _handle_audio_transcriptions(self, request: "web.Request") -> "web.Response":
        """POST /v1/audio/transcriptions: OpenAI-compatible speech to text."""
        from gateway.platforms.api_server import (
            MAX_REQUEST_BYTES, _openai_error, _redact_api_error_text,
        )

        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if not request.content_type.startswith("multipart/"):
            return web.json_response(
                _openai_error(
                    "Request body must be multipart/form-data",
                    code="invalid_content_type",
                ),
                status=400,
            )

        upload_path: Optional[str] = None
        upload_suffix = ""
        upload_size = 0
        multipart_size = 0
        fields: Dict[str, str] = {}

        def multipart_too_large() -> bool:
            buffered_size = int(getattr(request.content, "total_bytes", 0) or 0)
            return max(multipart_size, buffered_size) > MAX_REQUEST_BYTES

        def body_too_large_response() -> "web.Response":
            return web.json_response(
                _openai_error("Request body too large.", code="body_too_large"),
                status=413,
            )

        try:
            try:
                reader = await request.multipart()
                while True:
                    part = await reader.next()
                    if multipart_too_large():
                        return body_too_large_response()
                    if part is None:
                        break
                    if part.name == "file":
                        if upload_path is not None:
                            return web.json_response(
                                _openai_error(
                                    "Only one 'file' field is supported",
                                    param="file",
                                    code="duplicate_file",
                                ),
                                status=400,
                            )
                        suffix = Path(part.filename or "").suffix.lower()
                        if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
                            suffix = ""
                        upload_suffix = suffix
                        with tempfile.NamedTemporaryFile(
                            prefix="hermes-api-stt-", suffix=suffix, delete=False,
                        ) as upload:
                            upload_path = upload.name
                            while True:
                                chunk = await part.read_chunk(size=64 * 1024)
                                if not chunk:
                                    break
                                multipart_size += len(chunk)
                                if multipart_too_large():
                                    return body_too_large_response()
                                upload_size += len(chunk)
                                upload.write(chunk)
                    elif part.name:
                        value = bytearray()
                        while True:
                            chunk = await part.read_chunk(size=64 * 1024)
                            if not chunk:
                                break
                            multipart_size += len(chunk)
                            if multipart_too_large():
                                return body_too_large_response()
                            value.extend(chunk)
                        fields[part.name] = value.decode(
                            part.get_charset(default="utf-8"), errors="replace",
                        ).strip()
            except (AssertionError, KeyError, ValueError, web.HTTPBadRequest):
                return web.json_response(
                    _openai_error("Invalid multipart request body", code="invalid_multipart"),
                    status=400,
                )

            if upload_path is None:
                return web.json_response(
                    _openai_error("Missing required 'file' field", param="file", code="missing_file"),
                    status=400,
                )
            if upload_size == 0:
                return web.json_response(
                    _openai_error("Uploaded audio file is empty", param="file", code="empty_file"),
                    status=400,
                )
            model = self._clean_runtime_id(fields.get("model"))
            if not model:
                return web.json_response(
                    _openai_error("Missing required 'model' field", param="model", code="missing_model"),
                    status=400,
                )
            response_format = fields.get("response_format", "json").lower()
            if response_format not in {"json", "text", "verbose_json"}:
                return web.json_response(
                    _openai_error(
                        "response_format must be one of: json, text, verbose_json",
                        param="response_format", code="unsupported_response_format",
                    ),
                    status=400,
                )
            language = self._clean_runtime_id(fields.get("language"), max_len=32) or None
            prompt = fields.get("prompt") or None
            if prompt is not None and len(prompt) > 10_000:
                return web.json_response(
                    _openai_error("prompt is too long", param="prompt", code="invalid_prompt"),
                    status=400,
                )

            from tools.transcription_common import SUPPORTED_FORMATS
            from tools.transcription_audio import _probe_audio_duration
            from tools.transcription_tools import transcribe_audio

            if upload_suffix not in SUPPORTED_FORMATS and upload_suffix != ".silk":
                return web.json_response(
                    _openai_error(
                        f"Unsupported audio format: {upload_suffix or '(none)'}",
                        param="file", code="unsupported_audio_format",
                    ),
                    status=400,
                )
            result = await asyncio.to_thread(
                transcribe_audio, upload_path, model=model, language=language,
                prompt=prompt, source="api_server",
            )
            if not result.get("success"):
                message = result.get("error") or "Audio transcription failed"
                return web.json_response(
                    _openai_error(message, err_type="api_error", code="transcription_failed"),
                    status=502,
                )
            transcript = str(result.get("transcript") or "")
            if response_format == "text":
                return web.Response(text=transcript, content_type="text/plain")
            if response_format == "verbose_json":
                result_segments = result.get("segments")
                duration = await asyncio.to_thread(_probe_audio_duration, upload_path)
                return web.json_response({
                    "task": "transcribe",
                    "language": str(result.get("language") or language or "unknown"),
                    "duration": float(result.get("duration") or duration or 0.0),
                    "text": transcript,
                    "segments": result_segments if isinstance(result_segments, list) else [],
                })
            return web.json_response({"text": transcript})
        except web.HTTPRequestEntityTooLarge:
            raise
        except Exception as exc:
            logger.exception("API audio transcription failed")
            return web.json_response(
                _openai_error(
                    _redact_api_error_text(exc), err_type="server_error", code="transcription_error",
                ),
                status=500,
            )
        finally:
            if upload_path:
                with suppress(OSError):
                    os.unlink(upload_path)
