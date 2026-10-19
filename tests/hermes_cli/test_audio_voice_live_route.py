"""The voice-live HTTP boundary preserves profile and billing selection without exposing secrets."""

import pytest
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_voice_live_session_route_pins_auth_and_returns_only_public_fields(monkeypatch):
    from hermes_cli.web_models import VoiceLiveSessionRequest
    from hermes_cli.web_routers import audio
    from tools import voice_live

    calls = []

    async def scoped(profile, func, *args, **kwargs):
        calls.append((profile, args, kwargs))
        return func(*args, **kwargs)

    def create(sdp, history, *, expected_auth=None):
        calls.append((sdp, history, expected_auth))
        return {
            "auth": "subscription",
            "session": {"id": "rtc_fixture"},
            "transport": {"type": "webrtc", "sdp": "v=0 answer"},
        }

    monkeypatch.setattr(audio, "_run_config_scoped", scoped)
    monkeypatch.setattr(voice_live, "create_webrtc_session", create)
    result = await audio.create_voice_live_session(
        VoiceLiveSessionRequest(
            sdp="v=0 offer\r\n",
            history=[{"type": "message", "role": "user", "content": []}],
            expected_auth="subscription",
        ),
        profile="bot-profile",
    )

    assert calls[0][0] == "bot-profile"
    assert calls[1] == (
        "v=0 offer\r\n",
        [{"type": "message", "role": "user", "content": []}],
        "subscription",
    )
    assert result == {
        "ok": True,
        "auth": "subscription",
        "session": {"id": "rtc_fixture"},
        "transport": {"type": "webrtc", "sdp": "v=0 answer"},
    }


def test_voice_live_session_request_rejects_invalid_expected_auth():
    from hermes_cli.web_models import VoiceLiveSessionRequest

    with pytest.raises(ValidationError):
        VoiceLiveSessionRequest.model_validate({"sdp": "v=0 offer\r\n", "expected_auth": "automatic"})
