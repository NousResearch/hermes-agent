import asyncio
import json
import logging
import os
import sys
import time
import types

import pytest

from gateway.livekit_realtime_agent import (
    HERMES_BRAIN_UNAVAILABLE_MESSAGE,
    HermesRealtimeAssistant,
    build_server,
    build_modular_session,
    build_hermes_brain_payload,
    build_assistant_instructions,
    create_realtime_model,
    modular_preflight,
    guard_enabled_for_run,
    hermes_live_voice,
    _install_session_telemetry,
    is_hermes_brain_url_allowed,
    query_hermes_brain,
    sanitize_hermes_brain_answer,
)
from gateway.livekit_voice import (
    DEFAULT_GEMINI_REALTIME_MODEL,
    DEFAULT_HERMES_BRAIN_MODEL,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_DEEPGRAM_MODEL,
    DEFAULT_CARTESIA_MODEL,
    DEFAULT_XAI_REALTIME_MODEL,
    build_dispatch_rule_payload,
    build_inbound_trunk_payload,
    build_livekit_preflight,
    build_realtime_room_metadata,
    build_realtime_worker_status,
    build_room_name,
    build_room_token_output,
    load_livekit_config,
)


def test_preflight_reports_missing_number_without_blocking_web_mvp():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env)
    assert report["ok"] is True
    assert report["ready"]["web_mvp"] is True
    assert report["ready"]["sip_phone"] is False
    assert any(issue["code"] == "missing_phone_number" for issue in report["issues"])


def test_preflight_can_require_phone_number_for_sip_gate():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env, require_phone_number=True)
    assert report["ok"] is False
    assert any(
        issue["code"] == "missing_phone_number" and issue["severity"] == "error"
        for issue in report["issues"]
    )


def test_preflight_redacts_secret_values():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "lk_API_KEY_SECRET_VALUE",
        "LIVEKIT_API_SECRET": "lk_API_SECRET_VALUE",
        "OPENAI_API_KEY": "sk-test-secret-value",
        "HERMES_LIVEKIT_AGENT_NAME": "hermes-live-voice",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    rendered = json.dumps(report, sort_keys=True)
    assert "lk_API_KEY_SECRET_VALUE" not in rendered
    assert "lk_API_SECRET_VALUE" not in rendered
    assert "sk-test-secret-value" not in rendered
    assert report["config"]["livekit_api_key"] == "set"
    assert report["config"]["livekit_api_secret"] == "set"
    assert report["config"]["openai_api_key"] == "set"
    assert report["config"]["google_api_key"] == "missing"
    assert report["config"]["xai_api_key"] == "missing"


def test_hermes_brain_config_is_loaded_and_redacted():
    env = {
        "HERMES_LIVEKIT_HERMES_URL": "http://127.0.0.1:8646/v1/chat/completions",
        "HERMES_LIVEKIT_HERMES_API_KEY": "hermes-brain-secret",
        "HERMES_LIVEKIT_HERMES_MODEL": "voice",
        "HERMES_LIVEKIT_HERMES_TIMEOUT_SECONDS": "9.5",
        "HERMES_LIVEKIT_HERMES_MAX_TOKENS": "320",
        "HERMES_LIVEKIT_HERMES_ALLOWED_HOSTS": "brain.example.com, api.example.net",
    }
    cfg = load_livekit_config(env)
    rendered = json.dumps(cfg.public_dict(), sort_keys=True)
    assert cfg.hermes_brain_url.endswith("/v1/chat/completions")
    assert cfg.hermes_brain_api_key == "hermes-brain-secret"
    assert cfg.hermes_brain_model == "voice"
    assert cfg.hermes_brain_timeout_seconds == 9.5
    assert cfg.hermes_brain_max_tokens == 320
    assert cfg.hermes_brain_allowed_hosts == (
        "brain.example.com",
        "api.example.net",
    )
    assert cfg.has_brain_credentials is True
    assert cfg.public_dict()["hermes_brain_api_key"] == "set"
    assert "hermes-brain-secret" not in rendered


def test_hermes_brain_config_defaults_are_phone_safe():
    cfg = load_livekit_config({})
    assert cfg.hermes_brain_model == DEFAULT_HERMES_BRAIN_MODEL
    assert cfg.hermes_brain_timeout_seconds <= 10
    assert cfg.hermes_brain_max_tokens <= 500
    assert cfg.hermes_brain_allow_remote is False
    assert cfg.has_brain_credentials is False


def test_hermes_brain_url_allows_only_trusted_hosts_by_default():
    assert is_hermes_brain_url_allowed("http://127.0.0.1:8646/v1/chat/completions")
    assert not is_hermes_brain_url_allowed("http://10.0.0.5:8646/v1/chat/completions")
    assert not is_hermes_brain_url_allowed("https://brain.example.com/v1/chat/completions")
    assert not is_hermes_brain_url_allowed(
        "https://brain.example.com/v1/chat/completions",
        allow_remote=True,
    )
    assert is_hermes_brain_url_allowed(
        "https://brain.example.com/v1/chat/completions",
        allow_remote=True,
        allowed_hosts=("brain.example.com",),
    )
    assert not is_hermes_brain_url_allowed("http://brain.example.com/v1/chat/completions", allow_remote=True)


def test_hermes_brain_payload_is_concise_and_non_streaming():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_HERMES_MODEL": "voice",
        "HERMES_LIVEKIT_HERMES_MAX_TOKENS": "321",
    })
    payload = build_hermes_brain_payload(
        "Explain the Hermes phone architecture in depth.",
        config=cfg,
    )
    assert payload["model"] == "voice"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 321
    assert payload["temperature"] <= 0.3
    assert "live phone call" in payload["messages"][0]["content"]
    assert "Hermes phone architecture" in payload["messages"][1]["content"]


def test_hermes_brain_payload_rejects_empty_questions():
    cfg = load_livekit_config({})
    with pytest.raises(ValueError, match="question"):
        build_hermes_brain_payload("   ", config=cfg)


def test_query_hermes_brain_returns_assistant_text():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_HERMES_API_KEY": "fake-brain-key",
        "HERMES_LIVEKIT_HERMES_MODEL": "voice",
    })

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "Use the fast voice model first."}}
                ]
            }

    class FakeClient:
        def __init__(self):
            self.posted = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, headers, json):
            self.posted = (url, headers, json)
            return FakeResponse()

    fake_client = FakeClient()
    answer = asyncio.run(
        query_hermes_brain(
            "Should this use deeper reasoning?",
            config=cfg,
            client_factory=lambda **_: fake_client,
        )
    )
    assert answer == "Use the fast voice model first."
    assert fake_client.posted[1]["Authorization"] == "Bearer fake-brain-key"


def test_sanitize_hermes_brain_answer_redacts_and_clamps():
    raw = (
        "Here is the answer. API_KEY=secret-value "
        "Bearer abcdefghijklmnopqrstuvwxyz "
        "eyJaaaaaaaaaaa.bbbbbbbbbbbb.cccccccccccc "
        "xai-abcdefghijklmnopqrstuvwxyz "
        + ("x" * 2000)
    )
    clean = sanitize_hermes_brain_answer(raw)
    assert "secret-value" not in clean
    assert "abcdefghijklmnopqrstuvwxyz" not in clean
    assert "API_KEY=[redacted]" in clean
    assert "Bearer [redacted]" in clean
    assert "[redacted-jwt]" in clean
    assert "[redacted-token]" in clean
    assert len(clean) <= 1203
    assert clean.endswith("...")


def test_query_hermes_brain_returns_safe_message_on_error(caplog):
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_HERMES_API_KEY": "fake-brain-key",
    })

    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, *, headers, json):
            raise RuntimeError("upstream leaked detail")

    answer = asyncio.run(
        query_hermes_brain(
            "Need deep answer.",
            config=cfg,
            client_factory=lambda **_: FailingClient(),
        )
    )
    assert answer == HERMES_BRAIN_UNAVAILABLE_MESSAGE
    assert "Hermes brain query failed: RuntimeError" in caplog.text
    assert "fake-brain-key" not in caplog.text


def test_query_hermes_brain_does_not_send_key_to_untrusted_url():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_HERMES_URL": "https://brain.example.com/v1/chat/completions",
        "HERMES_LIVEKIT_HERMES_API_KEY": "fake-brain-key",
    })

    class FailingIfCalledClient:
        async def __aenter__(self):
            raise AssertionError("client must not be opened for untrusted brain URL")

    answer = asyncio.run(
        query_hermes_brain(
            "Need deep answer.",
            config=cfg,
            client_factory=lambda **_: FailingIfCalledClient(),
        )
    )
    assert answer == HERMES_BRAIN_UNAVAILABLE_MESSAGE


def test_realtime_assistant_registers_hermes_brain_tool():
    cfg = load_livekit_config({})
    assistant = HermesRealtimeAssistant(cfg)
    tool_names = {
        getattr(getattr(tool, "_info", None), "name", None)
        for tool in assistant._tools
    }
    assert "ask_hermes_brain" in tool_names


def test_session_telemetry_logs_redacted_events(caplog):
    caplog.set_level(logging.INFO, logger="gateway.livekit_realtime_agent")

    class FakeSession:
        def __init__(self):
            self.callbacks = {}

        def on(self, event_name, callback):
            self.callbacks[event_name] = callback

    session = FakeSession()
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    _install_session_telemetry(
        session,
        config=cfg,
        room_name="room one",
        started_at=time.monotonic(),
    )

    session.callbacks["user_input_transcribed"](
        types.SimpleNamespace(transcript="secret user words", is_final=True)
    )
    session.callbacks["agent_state_changed"](
        types.SimpleNamespace(old_state="thinking", new_state="speaking")
    )
    session.callbacks["close"](
        types.SimpleNamespace(reason=types.SimpleNamespace(value="done"), error=None)
    )

    assert "hermes_call event=transcript" in caplog.text
    assert "chars=17" in caplog.text
    assert "secret user words" not in caplog.text
    assert "hermes_call event=close" in caplog.text


def test_brain_tool_logs_start_and_done(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="gateway.livekit_realtime_agent")
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    assistant = HermesRealtimeAssistant(cfg)

    async def fake_query(question, *, config):
        return "answer"

    monkeypatch.setattr(
        "gateway.livekit_realtime_agent.query_hermes_brain",
        fake_query,
    )

    answer = asyncio.run(assistant.ask_hermes_brain("question"))

    assert answer == "answer"
    assert "hermes_call event=brain_tool_start" in caplog.text
    assert "hermes_call event=brain_tool_done" in caplog.text


def test_hermes_live_voice_logs_job_and_session(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="gateway.livekit_realtime_agent")

    class FakeSession:
        def __init__(self, *, llm):
            self.llm = llm
            self.callbacks = {}

        def on(self, event_name, callback):
            self.callbacks[event_name] = callback

        async def start(self, *, room, agent):
            self.room = room
            self.agent = agent

    monkeypatch.setattr("gateway.livekit_realtime_agent.AgentSession", FakeSession)
    monkeypatch.setattr(
        "gateway.livekit_realtime_agent.create_realtime_model",
        lambda cfg: object(),
    )
    monkeypatch.setenv("HERMES_LIVEKIT_REALTIME_PROVIDER", "xai")
    ctx = types.SimpleNamespace(room=types.SimpleNamespace(name="bench-room"))

    asyncio.run(hermes_live_voice(ctx))

    assert "hermes_call event=job_start" in caplog.text
    assert "room=bench-room" in caplog.text
    assert "hermes_call event=session_started" in caplog.text


def test_realtime_preflight_reports_missing_gemini_key_by_default():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    assert report["ok"] is False
    assert report["ready"]["realtime_agent"] is False
    assert report["config"]["pipeline_mode"] == "realtime"
    assert report["config"]["realtime_provider"] == "gemini"
    assert any(issue["code"] == "missing_google_api_key" for issue in report["issues"])


def test_livekit_preflight_rejects_remote_ws_url():
    env = {
        "LIVEKIT_URL": "ws://livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env)
    assert report["ok"] is False
    assert any(issue["code"] == "invalid_livekit_url" for issue in report["issues"])


def test_livekit_preflight_allows_loopback_ws_url():
    env = {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
    }
    report = build_livekit_preflight(env)
    assert report["ok"] is True
    assert not any(issue["code"] == "invalid_livekit_url" for issue in report["issues"])


def test_livekit_preflight_rejects_invalid_agent_name():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_AGENT_NAME": "../bad agent",
    }
    report = build_livekit_preflight(env)
    assert report["ok"] is False
    assert any(issue["code"] == "invalid_agent_name" for issue in report["issues"])


def test_gemini_realtime_preflight_accepts_gemini_api_key():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "gemini",
        "GEMINI_API_KEY": "gemini-secret-value",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    rendered = json.dumps(report, sort_keys=True)
    assert report["ok"] is True
    assert report["ready"]["realtime_agent"] is True
    assert report["config"]["google_api_key"] == "set"
    assert "gemini-secret-value" not in rendered
    assert report["worker"]["model"] == DEFAULT_GEMINI_REALTIME_MODEL


def test_gemini_realtime_preflight_reports_missing_google_key():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "gemini",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    assert report["ok"] is False
    assert report["ready"]["realtime_agent"] is False
    assert any(issue["code"] == "missing_google_api_key" for issue in report["issues"])


def test_xai_realtime_preflight_accepts_xai_api_key():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "xai",
        "XAI_API_KEY": "xai-secret-value",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    rendered = json.dumps(report, sort_keys=True)
    assert report["ok"] is True
    assert report["ready"]["realtime_agent"] is True
    assert report["config"]["xai_api_key"] == "set"
    assert "xai-secret-value" not in rendered
    assert report["worker"]["model"] == DEFAULT_XAI_REALTIME_MODEL
    assert report["worker"]["voice"] == "ara"


def test_xai_realtime_preflight_reports_missing_xai_key():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "xai",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    assert report["ok"] is False
    assert report["ready"]["realtime_agent"] is False
    assert any(issue["code"] == "missing_xai_api_key" for issue in report["issues"])


def test_realtime_config_defaults_and_status_are_operator_safe():
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "GEMINI_API_KEY": "gemini-secret-value",
        "HERMES_LIVEKIT_REALTIME_ENABLED": "true",
    }
    cfg = load_livekit_config(env)
    status = build_realtime_worker_status(config=cfg)
    assert cfg.pipeline_mode == "realtime"
    assert cfg.realtime_provider == "gemini"
    assert cfg.realtime_model == DEFAULT_GEMINI_REALTIME_MODEL
    assert cfg.realtime_voice == "Puck"
    assert status["enabled"] is True
    assert status["mode"] == "manual"
    assert "gateway.livekit_realtime_agent" in status["run"]


def test_realtime_room_metadata_is_stable_for_webrtc_dispatch():
    cfg = load_livekit_config({})
    assert build_realtime_room_metadata(mode="webrtc", config=cfg) == {
        "mode": "webrtc",
        "route": "hermes-main",
        "voice_version": "v02",
        "pipeline_mode": "realtime",
        "realtime_provider": "gemini",
        "stt_provider": "none",
        "tts_provider": "none",
    }


def test_realtime_room_metadata_uses_configured_provider():
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "gemini"})
    assert build_realtime_room_metadata(mode="sip", config=cfg)["realtime_provider"] == "gemini"

    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "xai"})
    assert build_realtime_room_metadata(mode="sip", config=cfg)["realtime_provider"] == "xai"


def test_assistant_instructions_are_short_and_language_aware():
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_INSTRUCTIONS": "Be concise."})
    instructions = build_assistant_instructions(cfg)
    assert "Be concise." in instructions
    assert "Romanian" in instructions
    assert "English" in instructions


def test_realtime_worker_start_guard_requires_explicit_enable():
    cfg = load_livekit_config({})

    with pytest.raises(SystemExit, match="disabled"):
        guard_enabled_for_run(["dev"], cfg)

    guard_enabled_for_run(["--help"], cfg)
    enabled_cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_ENABLED": "true"})
    guard_enabled_for_run(["dev"], enabled_cfg)


def test_create_realtime_model_rejects_unknown_provider():
    cfg = load_livekit_config({"HERMES_LIVEKIT_REALTIME_PROVIDER": "bogus"})
    with pytest.raises(RuntimeError, match="openai.*gemini.*xai"):
        create_realtime_model(cfg)


def test_openai_realtime_model_uses_config_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeRealtimeModel:
        def __init__(self, *, model, voice):
            self.model = model
            self.voice = voice

    fake_openai = types.SimpleNamespace(
        realtime=types.SimpleNamespace(RealtimeModel=FakeRealtimeModel)
    )
    fake_plugins = types.SimpleNamespace(openai=fake_openai)
    monkeypatch.setitem(sys.modules, "livekit.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.openai", fake_openai)

    cfg = load_livekit_config({
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "openai",
        "OPENAI_API_KEY": "cfg-openai-key",
    })
    model = create_realtime_model(cfg)

    assert os.environ["OPENAI_API_KEY"] == "cfg-openai-key"
    assert model.model == DEFAULT_REALTIME_MODEL
    assert model.voice == "coral"


def test_gemini_realtime_model_uses_config_key_and_instructions(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    class FakeRealtimeModel:
        def __init__(self, *, model, voice, instructions):
            self.model = model
            self.voice = voice
            self.instructions = instructions

    fake_google = types.SimpleNamespace(
        realtime=types.SimpleNamespace(RealtimeModel=FakeRealtimeModel)
    )
    fake_plugins = types.SimpleNamespace(google=fake_google)
    monkeypatch.setitem(sys.modules, "livekit.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.google", fake_google)

    cfg = load_livekit_config({
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "gemini",
        "GEMINI_API_KEY": "cfg-gemini-key",
    })
    model = create_realtime_model(cfg)

    assert os.environ["GOOGLE_API_KEY"] == "cfg-gemini-key"
    assert model.model == DEFAULT_GEMINI_REALTIME_MODEL
    assert model.voice == "Puck"
    assert "live voice call" in model.instructions


def test_xai_realtime_model_uses_config_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    class FakeRealtimeModel:
        def __init__(self, *, model, voice):
            self.model = model
            self.voice = voice

    fake_xai = types.SimpleNamespace(
        realtime=types.SimpleNamespace(RealtimeModel=FakeRealtimeModel)
    )
    fake_plugins = types.SimpleNamespace(xai=fake_xai)
    monkeypatch.setitem(sys.modules, "livekit.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.xai", fake_xai)

    cfg = load_livekit_config({
        "HERMES_LIVEKIT_REALTIME_PROVIDER": "xai",
        "XAI_API_KEY": "cfg-xai-key",
    })
    model = create_realtime_model(cfg)

    assert os.environ["XAI_API_KEY"] == "cfg-xai-key"
    assert model.model == DEFAULT_XAI_REALTIME_MODEL
    assert model.voice == "ara"



def test_modular_provider_env_parsing_and_public_dict_redaction():
    cfg = load_livekit_config({
        "HERMES_LIVEKIT_PIPELINE_MODE": "modular",
        "HERMES_LIVEKIT_STT_PROVIDER": "deepgram",
        "HERMES_LIVEKIT_TTS_PROVIDER": "cartesia",
        "HERMES_LIVEKIT_DEEPGRAM_MODEL": "nova-3",
        "HERMES_LIVEKIT_DEEPGRAM_LANGUAGE": "ro",
        "HERMES_LIVEKIT_CARTESIA_MODEL": "sonic-2",
        "HERMES_LIVEKIT_CARTESIA_VOICE": "voice-id",
        "DEEPGRAM_API_KEY": "deepgram-secret",
        "CARTESIA_API_KEY": "cartesia-secret",
    })
    public = cfg.public_dict()
    rendered = json.dumps(public, sort_keys=True)
    assert cfg.uses_modular_pipeline is True
    assert cfg.has_modular_credentials is True
    assert cfg.deepgram_model == "nova-3"
    assert cfg.deepgram_language == "ro"
    assert cfg.cartesia_model == "sonic-2"
    assert public["deepgram_api_key"] == "set"
    assert public["cartesia_api_key"] == "set"
    assert "deepgram-secret" not in rendered
    assert "cartesia-secret" not in rendered


def test_modular_preflight_reports_missing_dependency_or_key(monkeypatch):
    env = {
        "LIVEKIT_URL": "wss://pafi-livekit.example.com",
        "LIVEKIT_API_KEY": "livekit-key",
        "LIVEKIT_API_SECRET": "livekit-secret",
        "HERMES_LIVEKIT_PIPELINE_MODE": "modular",
    }
    report = build_livekit_preflight(env, include_realtime=True)
    assert report["ok"] is False
    assert any(issue["code"] == "missing_deepgram_api_key" for issue in report["issues"])
    assert any(issue["code"] == "missing_cartesia_api_key" for issue in report["issues"])

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    preflight = modular_preflight(load_livekit_config({"HERMES_LIVEKIT_PIPELINE_MODE": "modular"}))
    assert preflight["dependencies_ready"] is False
    assert any("deepgram" in warning for warning in preflight["warnings"])
    assert preflight["cartesia_voice"] in {"set", "missing"}


def test_build_modular_session_uses_lazy_livekit_plugin_apis(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)

    class FakeSession:
        def __init__(self, *, stt, tts):
            self.stt = stt
            self.tts = tts

    class FakeSTT:
        def __init__(self, *, model, language, **_kwargs):
            self.model = model
            self.language = language

    class FakeTTS:
        def __init__(self, *, model, voice, **_kwargs):
            self.model = model
            self.voice = voice

    monkeypatch.setattr("gateway.livekit_realtime_agent.AgentSession", FakeSession)
    fake_deepgram = types.SimpleNamespace(STT=FakeSTT)
    fake_cartesia = types.SimpleNamespace(TTS=FakeTTS)
    fake_plugins = types.SimpleNamespace(deepgram=fake_deepgram, cartesia=fake_cartesia)
    monkeypatch.setitem(sys.modules, "livekit.plugins", fake_plugins)
    monkeypatch.setitem(sys.modules, "livekit.plugins.deepgram", fake_deepgram)
    monkeypatch.setitem(sys.modules, "livekit.plugins.cartesia", fake_cartesia)

    cfg = load_livekit_config({
        "HERMES_LIVEKIT_PIPELINE_MODE": "modular",
        "DEEPGRAM_API_KEY": "deepgram-key",
        "CARTESIA_API_KEY": "cartesia-key",
    })
    session = build_modular_session(cfg)

    assert session.stt.model == DEFAULT_DEEPGRAM_MODEL
    assert session.tts.model == DEFAULT_CARTESIA_MODEL

def test_dispatch_rule_payload_uses_explicit_agent_dispatch():
    payload = build_dispatch_rule_payload(
        agent_name="hermes-live-voice",
        room_prefix="hermes-call-",
        metadata={"route": "hermes-main", "mode": "sip"},
        trunk_ids=["ST_123"],
    )
    assert payload == {
        "name": "Hermes live voice dispatch",
        "trunkIds": ["ST_123"],
        "rule": {"dispatchRuleIndividual": {"roomPrefix": "hermes-call-"}},
        "roomConfig": {
            "agents": [
                {
                    "agentName": "hermes-live-voice",
                    "metadata": '{"mode":"sip","route":"hermes-main"}',
                }
            ]
        },
    }


def test_dispatch_rule_payload_rejects_invalid_agent_name():
    with pytest.raises(ValueError, match="agent_name"):
        build_dispatch_rule_payload(agent_name="../bad agent")


def test_dispatch_rule_payload_rejects_invalid_trunk_id():
    with pytest.raises(ValueError, match="trunk_ids"):
        build_dispatch_rule_payload(trunk_ids=["ST_good", "../bad"])


def test_build_server_registers_validated_agent_name(monkeypatch):
    calls = []

    class FakeAgentServer:
        def rtc_session(self, entrypoint, *, agent_name):
            calls.append((entrypoint, agent_name))

    fake_agents = types.SimpleNamespace(AgentServer=FakeAgentServer)
    monkeypatch.setitem(sys.modules, "livekit.agents", fake_agents)
    monkeypatch.setenv("HERMES_LIVEKIT_AGENT_NAME", "safe-agent")

    server = build_server()

    assert isinstance(server, FakeAgentServer)
    assert calls[0][1] == "safe-agent"


def test_build_server_rejects_invalid_agent_name(monkeypatch):
    class FakeAgentServer:
        def rtc_session(self, entrypoint, *, agent_name):
            raise AssertionError("invalid agent name should fail before registration")

    fake_agents = types.SimpleNamespace(AgentServer=FakeAgentServer)
    monkeypatch.setitem(sys.modules, "livekit.agents", fake_agents)
    monkeypatch.setenv("HERMES_LIVEKIT_AGENT_NAME", "../bad agent")

    with pytest.raises(ValueError, match="agent_name"):
        build_server()


def test_inbound_trunk_payload_requires_e164_number():
    with pytest.raises(ValueError, match="E.164"):
        build_inbound_trunk_payload("0740000000")
    payload = build_inbound_trunk_payload(
        "+40740000000", allowed_numbers=["+40741111111"]
    )
    assert payload == {
        "trunk": {
            "name": "Hermes live voice inbound trunk",
            "numbers": ["+40740000000"],
            "krispEnabled": True,
            "allowedNumbers": ["+40741111111"],
        }
    }


def test_room_name_is_stable_safe_and_prefixed():
    assert (
        build_room_name("Hermes Call ", "Pafi Main Chat", suffix="abc123")
        == "hermes-call-pafi-main-chat-abc123"
    )


def test_room_name_preserves_suffix_when_prefix_is_long():
    room = build_room_name("x" * 200, "session", suffix="abc123")
    assert len(room) <= 96
    assert room.endswith("-abc123")


def test_room_token_output_redacts_jwt_by_default():
    output = build_room_token_output(
        livekit_url="wss://livekit.example.com",
        room="room",
        identity="pafi",
        token="jwt-secret",
    )
    assert output["token"] == "redacted"
    assert output["token_sensitive"] is True
    assert "jwt-secret" not in json.dumps(output)


def test_room_token_output_can_show_jwt_explicitly():
    output = build_room_token_output(
        livekit_url="wss://livekit.example.com",
        room="room",
        identity="pafi",
        token="jwt-secret",
        show_token=True,
    )
    assert output["token"] == "jwt-secret"
