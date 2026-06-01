from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import wave
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from .tracing import NativeCallTraceWriter


Transcriber = Callable[[str], Mapping[str, Any]]
Responder = Callable[[str], str | Awaitable[str]]
Synthesizer = Callable[[str, str], str]

_DEFAULT_CALL_SYSTEM_PROMPT = (
    "You are Hermes on a live phone call. Reply in short, natural spoken "
    "sentences. Ask one concise follow-up if you need more information. "
    "Do not mention internal traces, transcripts, or implementation details."
)
_VALID_API_MODES = {
    "chat_completions",
    "codex_responses",
    "anthropic_messages",
    "bedrock_converse",
    "codex_app_server",
}


@dataclass(frozen=True)
class VoiceTurnResult:
    ok: bool
    code: str
    message: str
    transcript: str = ""
    response_text: str = ""
    audio_path: Path | None = None
    stt_provider: str = ""
    tts_provider: str = ""


@dataclass(frozen=True)
class VoiceDebugTracePolicy:
    transcript_previews: bool = False
    max_preview_chars: int = 240


def _safe_call_id(call_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(call_id or "unknown"))
    return safe[:128] or "unknown"


def _default_audio_dir() -> Path:
    return get_hermes_home() / "cache" / "calls"


def _write_pcm16_wav(path: Path, pcm16: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)


def _default_transcriber(audio_path: str) -> Mapping[str, Any]:
    from tools.transcription_tools import transcribe_audio

    return transcribe_audio(audio_path)


def _default_synthesizer(text: str, output_path: str) -> str:
    from tools.tts_tool import text_to_speech_tool

    return text_to_speech_tool(text=text, output_path=output_path)


def _default_agent_response(call_id: str, transcript: str) -> str:
    from run_agent import AIAgent

    agent = AIAgent(**_call_agent_kwargs(call_id))
    return str(agent.chat(transcript) or "")


def _load_call_agent_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        return {}
    if not isinstance(cfg, dict):
        return {}

    merged: dict[str, Any] = {}
    calls_cfg = cfg.get("calls")
    if isinstance(calls_cfg, dict):
        native_cfg = calls_cfg.get("native")
        if isinstance(native_cfg, dict):
            agent_cfg = native_cfg.get("agent")
            if isinstance(agent_cfg, dict):
                merged.update(agent_cfg)

    platforms_cfg = cfg.get("platforms")
    if isinstance(platforms_cfg, dict):
        simplex_cfg = platforms_cfg.get("simplex")
        if isinstance(simplex_cfg, dict):
            extra_cfg = simplex_cfg.get("extra")
            if isinstance(extra_cfg, dict):
                native_cfg = extra_cfg.get("native_calls")
                if isinstance(native_cfg, dict):
                    agent_cfg = native_cfg.get("agent")
                    if isinstance(agent_cfg, dict):
                        merged.update(agent_cfg)

    return merged


def _load_call_debug_policy() -> VoiceDebugTracePolicy:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        return VoiceDebugTracePolicy()
    if not isinstance(cfg, dict):
        return VoiceDebugTracePolicy()

    merged: dict[str, Any] = {}
    calls_cfg = cfg.get("calls")
    if isinstance(calls_cfg, dict):
        native_cfg = calls_cfg.get("native")
        if isinstance(native_cfg, dict):
            debug_cfg = native_cfg.get("debug")
            if isinstance(debug_cfg, dict):
                merged.update(debug_cfg)

    platforms_cfg = cfg.get("platforms")
    if isinstance(platforms_cfg, dict):
        simplex_cfg = platforms_cfg.get("simplex")
        if isinstance(simplex_cfg, dict):
            extra_cfg = simplex_cfg.get("extra")
            if isinstance(extra_cfg, dict):
                native_cfg = extra_cfg.get("native_calls")
                if isinstance(native_cfg, dict):
                    debug_cfg = native_cfg.get("debug")
                    if isinstance(debug_cfg, dict):
                        merged.update(debug_cfg)

    return VoiceDebugTracePolicy(
        transcript_previews=_coerce_bool(merged.get("transcript_previews")),
        max_preview_chars=_preview_char_limit(merged.get("max_preview_chars")),
    )


def _clean_string(value: Any) -> str:
    return str(value or "").strip() if value is not None else ""


def _clean_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        return None
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _preview_char_limit(value: Any) -> int:
    parsed = _positive_int(value)
    if parsed is None:
        return 240
    return max(16, min(parsed, 2000))


def _preview_text(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized

    boundary = max_chars
    next_space = normalized.find(" ", max_chars)
    if 0 <= next_space <= max_chars + max(8, max_chars // 3):
        boundary = next_space
    else:
        prev_space = normalized.rfind(" ", 0, max_chars)
        if prev_space > max_chars // 2:
            boundary = prev_space

    preview = normalized[:boundary].rstrip(" \t\r\n.,;:!?")
    if not preview:
        preview = normalized[:max_chars].rstrip(" \t\r\n.,;:!?")
    return f"{preview}..."


def _speech_tool_intents(transcript: str) -> list[str]:
    checks = (
        ("weather", r"\b(weather|forecast|temperature|rain|snow|humidity)\b"),
        ("calendar", r"\b(calendar|schedule|meeting|appointment|availability)\b"),
        ("email", r"\b(email|gmail|inbox|message|messages)\b"),
    )
    normalized = transcript.lower()
    return [intent for intent, pattern in checks if re.search(pattern, normalized)]


def _call_agent_kwargs(call_id: str) -> dict[str, Any]:
    cfg = _load_call_agent_config()
    provider = _clean_string(cfg.get("provider"))
    model = _clean_string(cfg.get("model"))
    base_url = _clean_string(cfg.get("base_url"))

    runtime: dict[str, Any] = {}
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(
            requested=provider or None,
            explicit_base_url=base_url or None,
            target_model=model or None,
        )
    except Exception:
        if provider or base_url:
            raise

    if not model:
        model = _clean_string(runtime.get("model"))

    kwargs: dict[str, Any] = {
        "platform": "simplex_call",
        "session_id": f"simplex-native-call:{_safe_call_id(call_id)}",
        "quiet_mode": True,
        "skip_context_files": bool(cfg.get("skip_context_files", True)),
        "skip_memory": bool(cfg.get("skip_memory", False)),
        "ephemeral_system_prompt": (
            _clean_string(cfg.get("system_prompt")) or _DEFAULT_CALL_SYSTEM_PROMPT
        ),
    }

    for key in (
        "provider",
        "api_key",
        "base_url",
        "api_mode",
        "command",
        "credential_pool",
    ):
        value = runtime.get(key)
        if value:
            kwargs[key] = value
    args = runtime.get("args")
    if args:
        kwargs["args"] = list(args)

    configured_api_mode = _clean_string(cfg.get("api_mode"))
    if configured_api_mode in _VALID_API_MODES:
        kwargs["api_mode"] = configured_api_mode

    if model:
        kwargs["model"] = model

    max_iterations = _positive_int(cfg.get("max_iterations"))
    if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
    max_tokens = _positive_int(cfg.get("max_tokens"))
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    enabled_toolsets = _clean_list(cfg.get("enabled_toolsets"))
    if enabled_toolsets is not None:
        kwargs["enabled_toolsets"] = enabled_toolsets
    disabled_toolsets = _clean_list(cfg.get("disabled_toolsets"))
    if disabled_toolsets is not None:
        kwargs["disabled_toolsets"] = disabled_toolsets

    return kwargs


def _parse_tts_result(raw: str) -> tuple[bool, str, Path | None, str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False, "TTS returned invalid JSON.", None, ""
    if not isinstance(payload, dict):
        return False, "TTS returned an invalid response.", None, ""
    if not payload.get("success"):
        return False, str(payload.get("error") or "TTS generation failed."), None, ""
    file_path = payload.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return False, "TTS did not return an audio file path.", None, str(payload.get("provider") or "")
    return True, "", Path(file_path), str(payload.get("provider") or "")


class HermesVoiceTurnPipeline:
    """File-backed voice turn bridge for native calls.

    The native media layer can feed mono PCM16 audio here without knowing
    which STT, agent, or TTS backend is configured. Blocking backends are
    isolated with ``asyncio.to_thread`` so the WebRTC event loop stays alive.
    """

    def __init__(
        self,
        *,
        audio_dir: Path | None = None,
        transcriber: Transcriber | None = None,
        responder: Responder | None = None,
        synthesizer: Synthesizer | None = None,
        tracer: NativeCallTraceWriter | None = None,
        debug_policy: VoiceDebugTracePolicy | None = None,
    ) -> None:
        self.audio_dir = audio_dir or _default_audio_dir()
        self.transcriber = transcriber or _default_transcriber
        self.responder = responder
        self.synthesizer = synthesizer or _default_synthesizer
        self.tracer = tracer or NativeCallTraceWriter()
        self.debug_policy = debug_policy or _load_call_debug_policy()

    async def process_pcm16(
        self,
        *,
        call_id: str,
        pcm16: bytes,
        sample_rate: int,
    ) -> VoiceTurnResult:
        safe_call_id = _safe_call_id(call_id)
        turn_id = f"{int(time.time() * 1000)}"
        call_dir = self.audio_dir / safe_call_id
        input_path = call_dir / f"input_{turn_id}.wav"
        tts_path = call_dir / f"reply_{turn_id}.mp3"

        if not pcm16 or len(pcm16) % 2 != 0:
            return self._failure(
                call_id,
                "call_voice_audio_invalid",
                "Call audio must be non-empty PCM16 data.",
                pcm_bytes=len(pcm16 or b""),
            )
        if sample_rate <= 0:
            return self._failure(
                call_id,
                "call_voice_audio_invalid",
                "Call audio sample rate must be positive.",
                sample_rate=sample_rate,
            )

        self._trace(
            call_id,
            "voice_turn_started",
            sample_rate=sample_rate,
            pcm_bytes=len(pcm16),
        )
        _write_pcm16_wav(input_path, pcm16, sample_rate)

        stt_result = await asyncio.to_thread(self.transcriber, str(input_path))
        if not bool(stt_result.get("success")):
            return self._failure(
                call_id,
                "call_voice_stt_failed",
                str(stt_result.get("error") or "Speech transcription failed."),
                stt_provider=str(stt_result.get("provider") or ""),
            )

        transcript = str(stt_result.get("transcript") or "").strip()
        stt_provider = str(stt_result.get("provider") or "")
        if not transcript:
            return self._failure(
                call_id,
                "call_voice_transcript_empty",
                "Speech transcription produced no usable text.",
                stt_provider=stt_provider,
            )

        self._trace(
            call_id,
            "voice_turn_transcribed",
            transcript_chars=len(transcript),
            stt_provider=stt_provider,
        )
        self._trace_observed_text(
            call_id,
            "voice_turn_transcript_observed",
            transcript,
            chars=len(transcript),
            stt_provider=stt_provider,
        )
        for intent in _speech_tool_intents(transcript):
            self._trace(
                call_id,
                "tool_intent_observed",
                intent=intent,
                source="speech_transcript",
                needs_tool=True,
            )
        response_text = await self._respond(call_id, transcript)
        response_text = response_text.strip()
        if not response_text:
            return self._failure(
                call_id,
                "call_voice_agent_empty",
                "Hermes produced no voice response.",
                stt_provider=stt_provider,
                transcript_chars=len(transcript),
            )

        self._trace(
            call_id,
            "voice_turn_agent_responded",
            response_chars=len(response_text),
        )
        self._trace_observed_text(
            call_id,
            "voice_turn_agent_response_observed",
            response_text,
            chars=len(response_text),
        )
        tts_raw = await asyncio.to_thread(
            self.synthesizer,
            response_text,
            str(tts_path),
        )
        tts_ok, tts_error, audio_path, tts_provider = _parse_tts_result(tts_raw)
        if not tts_ok or audio_path is None:
            return self._failure(
                call_id,
                "call_voice_tts_failed",
                tts_error,
                stt_provider=stt_provider,
                transcript_chars=len(transcript),
                response_chars=len(response_text),
            )

        self._trace(
            call_id,
            "voice_turn_tts_ready",
            audio_path=str(audio_path),
            tts_provider=tts_provider,
        )
        return VoiceTurnResult(
            ok=True,
            code="call_voice_turn_completed",
            message="Voice turn completed.",
            transcript=transcript,
            response_text=response_text,
            audio_path=audio_path,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
        )

    async def _respond(self, call_id: str, transcript: str) -> str:
        if self.responder is None:
            return await asyncio.to_thread(_default_agent_response, call_id, transcript)
        response = self.responder(transcript)
        if inspect.isawaitable(response):
            response = await response
        return str(response or "")

    def _failure(self, call_id: str, code: str, message: str, **fields: Any) -> VoiceTurnResult:
        self._trace(call_id, "voice_turn_failed", code=code, message=message, **fields)
        return VoiceTurnResult(ok=False, code=code, message=message)

    def _trace_observed_text(
        self,
        call_id: str,
        event: str,
        text: str,
        **fields: Any,
    ) -> None:
        if not self.debug_policy.transcript_previews:
            return
        max_chars = self.debug_policy.max_preview_chars
        self._trace(
            call_id,
            event,
            preview=_preview_text(text, max_chars),
            preview_chars=max_chars,
            sensitive=True,
            **fields,
        )

    def _trace(self, call_id: str, event: str, **fields: Any) -> None:
        try:
            self.tracer.record(call_id, event, **fields)
        except Exception:
            pass
