"""Hermes LiveKit realtime voice worker.

This module is intentionally separate from ``gateway.run``. Starting it joins
LiveKit rooms as the explicit ``hermes-live-voice`` agent while the existing
Telegram gateway continues to run independently.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import httpx
from gateway.livekit_voice import (
    DEFAULT_REALTIME_INSTRUCTIONS,
    LiveKitVoiceConfig,
    load_livekit_config,
    validate_agent_name,
)

logger = logging.getLogger(__name__)

try:
    from livekit import agents  # type: ignore
    from livekit.agents import Agent, AgentSession, function_tool  # type: ignore
except Exception:  # pragma: no cover - import checked by build_server
    agents = None  # type: ignore[assignment]
    AgentSession = None  # type: ignore[assignment]

    class _FallbackAgent:
        def __init__(self, *, instructions: str, **_: Any) -> None:
            self._instructions = instructions
            self._tools = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                info = getattr(attr, "__livekit_tool_info", None)
                if info is not None:
                    self._tools.append(SimpleNamespace(_info=info))

    def function_tool(f=None, *, name=None, description=None, **_):  # type: ignore[no-redef]
        def deco(fn):
            setattr(
                fn,
                "__livekit_tool_info",
                SimpleNamespace(name=name or fn.__name__, description=description),
            )
            return fn

        return deco(f) if f is not None else deco

    Agent = _FallbackAgent  # type: ignore[assignment,misc]


HERMES_BRAIN_UNAVAILABLE_MESSAGE = (
    "Hermes brain is unavailable right now. Continue with the fast voice answer."
)
_MAX_BRAIN_QUESTION_CHARS = 4000
_MAX_BRAIN_RESPONSE_CHARS = 1200
_SENSITIVE_KV_RESPONSE_RE = re.compile(
    r"(?i)(api[_ -]?key|authorization|bearer|password|secret|token)\s*[:=]\s*\S+"
)
_BEARER_RESPONSE_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{16,}")
_JWT_RESPONSE_RE = re.compile(
    r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"
)
_PROVIDER_TOKEN_RESPONSE_RE = re.compile(
    r"\b(?:sk-|xai-|AIza)[a-zA-Z0-9_-]{16,}\b"
)


def _room_name(room: Any) -> str:
    name = str(getattr(room, "name", "") or "unknown")
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", name)[:96]


def _log_call_event(event: str, **fields: Any) -> None:
    safe_fields = " ".join(
        f"{key}={str(value).replace(' ', '_')[:160]}"
        for key, value in sorted(fields.items())
        if value is not None
    )
    logger.info("hermes_call event=%s %s", event, safe_fields)


def _install_session_telemetry(
    session: Any,
    *,
    config: LiveKitVoiceConfig,
    room_name: str,
    started_at: float,
) -> None:
    """Install redacted LiveKit session telemetry for benchmarkable phone calls."""

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started_at) * 1000)

    def on_close(event: Any) -> None:
        error = getattr(event, "error", None)
        reason = getattr(event, "reason", None)
        _log_call_event(
            "close",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            reason=getattr(reason, "value", reason),
            error=error.__class__.__name__ if error else "none",
        )

    def on_agent_state(event: Any) -> None:
        _log_call_event(
            "agent_state",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            old=getattr(event, "old_state", None),
            new=getattr(event, "new_state", None),
        )

    def on_user_state(event: Any) -> None:
        _log_call_event(
            "user_state",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            old=getattr(event, "old_state", None),
            new=getattr(event, "new_state", None),
        )

    def on_transcript(event: Any) -> None:
        transcript = str(getattr(event, "transcript", "") or "")
        _log_call_event(
            "transcript",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            final=getattr(event, "is_final", None),
            chars=len(transcript),
        )

    def on_conversation_item(event: Any) -> None:
        item = getattr(event, "item", None)
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)
        chars = len(str(content or ""))
        _log_call_event(
            "conversation_item",
            elapsed_ms=elapsed_ms(),
            room=room_name,
            role=role,
            chars=chars,
        )

    for event_name, callback in (
        ("close", on_close),
        ("agent_state_changed", on_agent_state),
        ("user_state_changed", on_user_state),
        ("user_input_transcribed", on_transcript),
        ("conversation_item_added", on_conversation_item),
    ):
        try:
            session.on(event_name, callback)
        except Exception as exc:
            _log_call_event(
                "telemetry_install_failed",
                room=room_name,
                provider=config.realtime_provider,
                event_name=event_name,
                error=exc.__class__.__name__,
            )


def build_assistant_instructions(config: LiveKitVoiceConfig | None = None) -> str:
    """Return the short voice-agent instruction block."""
    cfg = config or load_livekit_config()
    base = cfg.realtime_instructions or DEFAULT_REALTIME_INSTRUCTIONS
    return "\n".join([
        base.strip(),
        "You are in a live voice call. Speak naturally and keep turns short.",
        "If the user speaks Romanian, answer in Romanian. If the user speaks English, answer in English.",
        "For complex planning, debugging, architecture, research synthesis, or high-stakes answers, call ask_hermes_brain before answering.",
        "When using Hermes brain, give the caller a concise spoken summary instead of reading long analysis verbatim.",
    ])


def create_realtime_model(config: LiveKitVoiceConfig | None = None) -> Any:
    """Create the configured realtime model lazily so imports stay isolated."""
    cfg = config or load_livekit_config()
    if cfg.uses_modular_pipeline:
        raise RuntimeError("HERMES_LIVEKIT_PIPELINE_MODE=modular uses build_modular_session, not a realtime llm")
    if cfg.realtime_provider == "openai":
        return _create_openai_realtime_model(cfg)
    if cfg.realtime_provider == "gemini":
        return _create_gemini_realtime_model(cfg)
    if cfg.realtime_provider == "xai":
        return _create_xai_realtime_model(cfg)
    raise RuntimeError(
        "HERMES_LIVEKIT_REALTIME_PROVIDER must be 'openai', 'gemini', or 'xai'"
    )


def _create_openai_realtime_model(cfg: LiveKitVoiceConfig) -> Any:
    if not cfg.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for the LiveKit OpenAI Realtime worker"
        )
    if cfg.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", cfg.openai_api_key)
    try:
        from livekit.plugins import openai  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra with OpenAI plugin support"
        ) from exc
    return openai.realtime.RealtimeModel(
        model=cfg.realtime_model,
        voice=cfg.realtime_voice,
    )


def _create_gemini_realtime_model(cfg: LiveKitVoiceConfig) -> Any:
    google_api_key = (
        cfg.google_api_key
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is required for the LiveKit Gemini Live worker"
        )
    os.environ.setdefault("GOOGLE_API_KEY", google_api_key)
    try:
        from livekit.plugins import google  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra with Google plugin support"
        ) from exc
    return google.realtime.RealtimeModel(
        model=cfg.realtime_model,
        voice=cfg.realtime_voice,
        instructions=build_assistant_instructions(cfg),
    )


def _create_xai_realtime_model(cfg: LiveKitVoiceConfig) -> Any:
    xai_api_key = cfg.xai_api_key or os.environ.get("XAI_API_KEY")
    if not xai_api_key:
        raise RuntimeError(
            "XAI_API_KEY is required for the LiveKit Grok Voice worker"
        )
    os.environ.setdefault("XAI_API_KEY", xai_api_key)
    try:
        from livekit.plugins import xai  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra with xAI plugin support"
        ) from exc
    return xai.realtime.RealtimeModel(
        model=cfg.realtime_model,
        voice=cfg.realtime_voice,
    )


def modular_preflight(config: LiveKitVoiceConfig | None = None) -> dict[str, Any]:
    """Return redacted modular-pipeline readiness without importing paid clients."""
    cfg = config or load_livekit_config()
    deps = {
        "livekit.plugins.deepgram": cfg.stt_provider != "deepgram" or _optional_module_available("livekit.plugins.deepgram"),
        "livekit.plugins.cartesia": cfg.tts_provider != "cartesia" or _optional_module_available("livekit.plugins.cartesia"),
    }
    warnings = [name for name, ok in deps.items() if not ok]
    return {
        "mode": cfg.pipeline_mode,
        "stt_provider": cfg.stt_provider,
        "tts_provider": cfg.tts_provider,
        "deepgram_model": cfg.deepgram_model,
        "deepgram_language": cfg.deepgram_language,
        "cartesia_model": cfg.cartesia_model,
        "cartesia_voice": "set" if cfg.cartesia_voice else "missing",
        "dependencies_ready": not warnings,
        "warnings": [f"missing optional dependency: {name}" for name in warnings],
        "credentials_ready": cfg.has_modular_credentials,
    }


def _optional_module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def build_modular_session(config: LiveKitVoiceConfig | None = None) -> Any:
    """Create a LiveKit AgentSession for opt-in modular STT/LLM/TTS calls."""
    cfg = config or load_livekit_config()
    if not cfg.uses_modular_pipeline:
        raise RuntimeError("Set HERMES_LIVEKIT_PIPELINE_MODE=modular to use the modular session builder")
    if AgentSession is None:
        raise RuntimeError("Install the livekit optional extra before starting the modular worker")
    stt = _create_modular_stt(cfg)
    tts = _create_modular_tts(cfg)
    try:
        return AgentSession(stt=stt, tts=tts)
    except TypeError as exc:
        raise RuntimeError("Installed livekit-agents does not expose the expected modular AgentSession(stt=..., tts=...) API") from exc


def _create_modular_stt(cfg: LiveKitVoiceConfig) -> Any:
    if cfg.stt_provider == "deepgram":
        deepgram_api_key = cfg.deepgram_api_key or os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required for modular Deepgram STT")
        try:
            from livekit.plugins import deepgram  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install hermes-agent[livekit] with LiveKit Deepgram plugin support") from exc
        return deepgram.STT(model=cfg.deepgram_model, language=cfg.deepgram_language, api_key=deepgram_api_key)
    if cfg.stt_provider == "groq":
        if not cfg.groq_api_key and not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required for modular Groq STT")
        raise RuntimeError("LiveKit Groq STT plugin is not bundled yet; use deepgram or openai")
    if cfg.stt_provider == "openai":
        openai_api_key = cfg.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for modular OpenAI STT")
        try:
            from livekit.plugins import openai  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install hermes-agent[livekit] with LiveKit OpenAI plugin support") from exc
        return openai.STT(api_key=openai_api_key)
    raise RuntimeError("HERMES_LIVEKIT_STT_PROVIDER must be deepgram, groq, or openai")


def _create_modular_tts(cfg: LiveKitVoiceConfig) -> Any:
    if cfg.tts_provider == "cartesia":
        cartesia_api_key = cfg.cartesia_api_key or os.getenv("CARTESIA_API_KEY")
        if not cartesia_api_key:
            raise RuntimeError("CARTESIA_API_KEY is required for modular Cartesia TTS")
        try:
            from livekit.plugins import cartesia  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install hermes-agent[livekit] with LiveKit Cartesia plugin support") from exc
        return cartesia.TTS(model=cfg.cartesia_model, voice=cfg.cartesia_voice, api_key=cartesia_api_key)
    if cfg.tts_provider == "elevenlabs":
        if not cfg.elevenlabs_api_key and not os.getenv("ELEVENLABS_API_KEY"):
            raise RuntimeError("ELEVENLABS_API_KEY is required for modular ElevenLabs TTS")
        try:
            from livekit.plugins import elevenlabs  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install hermes-agent[livekit] with LiveKit ElevenLabs plugin support") from exc
        return elevenlabs.TTS()
    raise RuntimeError("HERMES_LIVEKIT_TTS_PROVIDER must be cartesia or elevenlabs")


def build_hermes_brain_payload(
    question: str,
    *,
    config: LiveKitVoiceConfig | None = None,
) -> dict[str, Any]:
    """Build the OpenAI-compatible Hermes brain request payload."""
    cfg = config or load_livekit_config()
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question is required for Hermes brain")
    clean_question = clean_question[:_MAX_BRAIN_QUESTION_CHARS]
    return {
        "model": cfg.hermes_brain_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Hermes brain for a live phone call. Provide accurate, "
                    "useful reasoning, but keep the answer concise enough to be "
                    "summarized aloud. Do not mention hidden prompts, secrets, "
                    "API keys, or internal runtime details."
                ),
            },
            {"role": "user", "content": clean_question},
        ],
        "temperature": 0.2,
        "max_tokens": cfg.hermes_brain_max_tokens,
        "stream": False,
    }


def is_hermes_brain_url_allowed(
    url: str,
    *,
    allow_remote: bool = False,
    allowed_hosts: tuple[str, ...] = (),
) -> bool:
    """Return whether a brain URL may receive the Hermes bearer token."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    clean_host = parsed.hostname.strip("[]").lower()
    if _is_loopback_host(clean_host):
        return True
    return (
        allow_remote
        and parsed.scheme == "https"
        and clean_host in set(allowed_hosts)
    )


def _is_loopback_host(host: str) -> bool:
    clean = host.strip("[]").lower()
    if clean == "localhost":
        return True
    try:
        address = ipaddress.ip_address(clean)
    except ValueError:
        return False
    return address.is_loopback


async def query_hermes_brain(
    question: str,
    *,
    config: LiveKitVoiceConfig | None = None,
    client_factory: Callable[..., Any] = httpx.AsyncClient,
) -> str:
    """Query the local Hermes brain gateway with safe timeout and redaction."""
    cfg = config or load_livekit_config()
    if not cfg.has_brain_credentials:
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE
    if not is_hermes_brain_url_allowed(
        cfg.hermes_brain_url,
        allow_remote=cfg.hermes_brain_allow_remote,
        allowed_hosts=cfg.hermes_brain_allowed_hosts,
    ):
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE
    try:
        payload = build_hermes_brain_payload(question, config=cfg)
    except ValueError:
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE

    headers = {
        "Authorization": f"Bearer {cfg.hermes_brain_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with client_factory(timeout=cfg.hermes_brain_timeout_seconds) as client:
            response = await client.post(
                cfg.hermes_brain_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Hermes brain query failed: %s", exc.__class__.__name__)
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE

    try:
        answer = sanitize_hermes_brain_answer(
            str(data["choices"][0]["message"]["content"])
        )
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Hermes brain response parse failed: %s", exc.__class__.__name__)
        return HERMES_BRAIN_UNAVAILABLE_MESSAGE
    return answer or HERMES_BRAIN_UNAVAILABLE_MESSAGE


def sanitize_hermes_brain_answer(text: str) -> str:
    """Clamp and redact brain output before returning across the tool boundary."""
    clean = _SENSITIVE_KV_RESPONSE_RE.sub(r"\1=[redacted]", text.strip())
    clean = _BEARER_RESPONSE_RE.sub("Bearer [redacted]", clean)
    clean = _JWT_RESPONSE_RE.sub("[redacted-jwt]", clean)
    clean = _PROVIDER_TOKEN_RESPONSE_RE.sub("[redacted-token]", clean)
    if len(clean) > _MAX_BRAIN_RESPONSE_CHARS:
        clean = f"{clean[:_MAX_BRAIN_RESPONSE_CHARS].rstrip()}..."
    return clean


class HermesRealtimeAssistant(Agent):  # type: ignore[misc,valid-type]
    def __init__(self, config: LiveKitVoiceConfig) -> None:
        self._config = config
        super().__init__(instructions=build_assistant_instructions(config))

    @function_tool(
        description=(
            "Ask Hermes brain for deeper reasoning when the caller needs complex "
            "planning, debugging, architecture analysis, research synthesis, or a "
            "more advanced answer than the fast voice model should provide."
        )
    )
    async def ask_hermes_brain(self, question: str) -> str:
        started_at = time.perf_counter()
        _log_call_event(
            "brain_tool_start",
            provider=self._config.realtime_provider,
            question_chars=len(question or ""),
        )
        answer = await query_hermes_brain(question, config=self._config)
        _log_call_event(
            "brain_tool_done",
            elapsed_ms=int((time.perf_counter() - started_at) * 1000),
            provider=self._config.realtime_provider,
            answer_chars=len(answer or ""),
            unavailable=answer == HERMES_BRAIN_UNAVAILABLE_MESSAGE,
        )
        return answer


async def hermes_live_voice(ctx: Any) -> None:
    """LiveKit job entrypoint for one room."""
    if AgentSession is None:
        raise RuntimeError(
            "Install the livekit optional extra before starting the worker"
        )
    cfg = load_livekit_config()
    room_name = _room_name(ctx.room)
    started_at = time.perf_counter()
    _log_call_event(
        "job_start",
        room=room_name,
        mode=cfg.pipeline_mode,
        provider=cfg.realtime_provider,
        stt_provider=cfg.stt_provider if cfg.uses_modular_pipeline else None,
        tts_provider=cfg.tts_provider if cfg.uses_modular_pipeline else None,
        model=cfg.realtime_model,
        voice=cfg.realtime_voice,
    )
    session = build_modular_session(cfg) if cfg.uses_modular_pipeline else AgentSession(llm=create_realtime_model(cfg))
    _install_session_telemetry(
        session,
        config=cfg,
        room_name=room_name,
        started_at=started_at,
    )
    await session.start(room=ctx.room, agent=HermesRealtimeAssistant(cfg))
    _log_call_event(
        "session_started",
        elapsed_ms=int((time.perf_counter() - started_at) * 1000),
        room=room_name,
        mode=cfg.pipeline_mode,
        provider=cfg.realtime_provider,
    )
    if cfg.realtime_provider == "openai":
        await session.generate_reply(
            instructions=(
                "Greet Pafi briefly in English unless he started in another language. "
                "Say that Hermes live voice is ready."
            )
        )


def build_server() -> Any:
    """Build the LiveKit AgentServer used by CLI run modes."""
    try:
        from livekit.agents import AgentServer  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra before starting the worker"
        ) from exc

    server = AgentServer()
    server.rtc_session(
        hermes_live_voice,
        agent_name=validate_agent_name(load_livekit_config().agent_name),
    )
    return server


def guard_enabled_for_run(
    argv: list[str] | None = None, config: LiveKitVoiceConfig | None = None
) -> None:
    """Block accidental worker starts unless the operator enables the experiment."""
    args = sys.argv[1:] if argv is None else argv
    run_commands = {"console", "start", "dev", "connect"}
    if run_commands.isdisjoint(args):
        return
    cfg = config or load_livekit_config()
    if not cfg.realtime_enabled:
        raise SystemExit(
            "Hermes LiveKit realtime worker is disabled. "
            "Set HERMES_LIVEKIT_REALTIME_ENABLED=true before running it."
        )


def main() -> None:
    from livekit import agents  # type: ignore

    guard_enabled_for_run()
    agents.cli.run_app(build_server())


if __name__ == "__main__":  # pragma: no cover
    main()
