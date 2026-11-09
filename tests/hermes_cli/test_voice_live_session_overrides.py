"""Regression for #151774: profile-local GPT-Live overrides use the host-owned exchange."""

import asyncio
import contextvars
import json
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from hermes_cli.web_models import VoiceLiveSessionRequest
from hermes_cli.web_routers import audio
from tools import voice_live


def test_concurrent_profile_scoped_voice_sessions_keep_overrides_local(monkeypatch):
    selected = contextvars.ContextVar("selected_voice_profile")
    configured = {
        "alpha": {"voice": "willow", "instructions": "Profile alpha."},
        "beta": {"voice": "quartz", "instructions": "Profile beta."},
    }

    @contextmanager
    def scope(profile):
        token = selected.set(profile)
        try:
            yield
        finally:
            selected.reset(token)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"session":{"id":"live"},"transport":{"type":"webrtc","sdp":"answer"}}'

    captured = {}

    def fake_urlopen(request, timeout=0):
        captured[selected.get()] = {
            "url": request.full_url,
            "auth": request.get_header("Authorization"),
            "body": json.loads(request.data),
        }
        return Response()

    monkeypatch.setattr(audio, "_config_profile_scope", scope)
    monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: configured[selected.get()])
    monkeypatch.setattr(
        voice_live, "_resolve_credentials",
        lambda live: (f"sk-{selected.get()}", "https://api.example/v1"),
    )
    monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)

    history = [{"type": "message", "role": "user", "content": []}]
    first = VoiceLiveSessionRequest(
        sdp="v=0 alpha\r\n", history=history, instructions="Call alpha.", voice="marin",
    )
    second = VoiceLiveSessionRequest(
        sdp="v=0 beta\r\n", history=history, instructions="Call beta.", voice="cedar",
    )

    async def exchange_both():
        return await asyncio.gather(
            audio.create_voice_live_session(first, profile="alpha"),
            audio.create_voice_live_session(second, profile="beta"),
        )

    results = asyncio.run(exchange_both())
    answer = {"ok": True, "session": {"id": "live"}, "transport": {"type": "webrtc", "sdp": "answer"}}
    assert results == [answer, answer]
    for name, own, other, expected_voice in (
        ("alpha", "Call alpha.", "Call beta.", "marin"),
        ("beta", "Call beta.", "Call alpha.", "cedar"),
    ):
        exchange = captured[name]
        session = exchange["body"]["session"]
        assert exchange["url"] == "https://api.example/v1/live/sessions"
        assert exchange["auth"] == f"Bearer sk-{name}"
        assert session["instructions"].startswith(voice_live.LIVE_PERSONA)
        assert configured[name]["instructions"] in session["instructions"]
        assert own in session["instructions"]
        assert other not in session["instructions"]
        assert session["audio"]["output"]["voice"] == expected_voice
        assert session["delegation"] == {"type": "client"}
        assert session["input"] == history
        assert exchange["body"]["transport"] == {"type": "webrtc", "sdp": f"v=0 {name}\r\n"}

    assert configured == {
        "alpha": {"voice": "willow", "instructions": "Profile alpha."},
        "beta": {"voice": "quartz", "instructions": "Profile beta."},
    }
    for invalid in (
        {"instructions": "   "}, {"voice": " "}, {"voice": "bad\nvoice"},
        {"voice": "v" * 129}, {"instructions": "x" * 32769},
    ):
        with pytest.raises(ValidationError):
            VoiceLiveSessionRequest(sdp="v=0", **invalid)
