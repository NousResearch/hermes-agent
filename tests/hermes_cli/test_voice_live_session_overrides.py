"""Per-call GPT-Live settings use the host-owned, profile-scoped exchange (#151774)."""

import asyncio
import io
import json
import threading

import httpx
import pytest

from hermes_cli import profiles, web_server
from tools import voice_live


@pytest.fixture
def voice_profiles():
    originals = {}
    for name in ("alpha", "beta"):
        home = profiles.get_profile_dir(name)
        home.mkdir(parents=True)
        config = home / "config.yaml"
        config.write_text(json.dumps({"voice": {"gpt_live": {
            "instructions": f"Profile {name}.", "voice": f"configured-{name}",
            "model": f"model-{name}", "base_url": f"https://{name}.example/v1",
        }}}), encoding="utf-8")
        env = home / ".env"
        env.write_text(f"OPENAI_API_KEY=test-{name}-key\n", encoding="utf-8")
        originals.update({path: path.read_bytes() for path in (config, env)})
    return originals


@pytest.mark.asyncio
async def test_concurrent_sessions_keep_settings_and_credentials_local(voice_profiles, monkeypatch):
    """Exercise HTTP parsing, real profile/config/key resolution and the vendor request."""
    # Two overlapping calls within each profile also catch a temporary config-write workaround.
    calls = [(name, f"{name}-{index}", voice)
             for name in ("alpha", "beta")
             for index, voice in enumerate(("marin", "cedar"))]
    rendezvous = threading.Barrier(len(calls), timeout=10)
    captured = {}

    def exchange(request, timeout):
        body = json.loads(request.data)
        sdp = body["transport"]["sdp"]
        captured[sdp] = (request.full_url, request.get_header("Authorization"), body)
        rendezvous.wait()
        return io.BytesIO(json.dumps({
            "session": {"id": sdp}, "transport": {"type": "webrtc", "sdp": "answer"},
        }).encode())

    monkeypatch.setattr(voice_live.urllib.request, "urlopen", exchange)
    headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_server.app), base_url="http://test", headers=headers,
    ) as client:
        responses = await asyncio.gather(*(
            client.post("/api/audio/voice-live/session", params={"profile": name}, json={
                "sdp": f"v=0 {call}\r\n", "instructions": f"Call {call}.", "voice": voice,
                "history": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": f"History {call}."},
                ]}],
            }) for name, call, voice in calls
        ))

    for (name, call, voice), response in zip(calls, responses):
        assert response.status_code == 200, response.text
        sdp = f"v=0 {call}\r\n"
        assert response.json() == {
            "ok": True, "session": {"id": sdp}, "transport": {"type": "webrtc", "sdp": "answer"},
        }
        url, authorization, body = captured[sdp]
        assert url == f"https://{name}.example/v1/live/sessions"
        assert authorization == f"Bearer test-{name}-key"
        assert body == {
            "session": {
                "model": f"model-{name}",
                "instructions": f"{voice_live.LIVE_PERSONA}\n\nProfile {name}.\n\nCall {call}.",
                "audio": {"output": {"voice": voice}}, "delegation": {"type": "client"},
                "input": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": f"History {call}."},
                ]}],
            },
            "transport": {"type": "webrtc", "sdp": sdp},
        }
    assert {path: path.read_bytes() for path in voice_profiles} == voice_profiles


@pytest.mark.asyncio
@pytest.mark.parametrize("settings, expected_voice, extra", [
    ({}, "configured-alpha", ""),
    ({"instructions": None, "voice": None}, "configured-alpha", ""),
    ({"voice": "cedar"}, "cedar", ""),
    ({"instructions": "Call context."}, "configured-alpha", "\n\nCall context."),
    ({"instructions": " \nCall context.\n ", "voice": "custom voice"},
     "custom voice", "\n\n \nCall context.\n "),
    ({"instructions": "x" * 32768, "voice": "v" * 128}, "v" * 128, "\n\n" + "x" * 32768),
])
async def test_optional_settings_keep_profile_defaults(voice_profiles, monkeypatch, settings, expected_voice, extra):
    captured = {}

    def exchange(request, timeout):
        captured.update(json.loads(request.data))
        return io.BytesIO(b'{"session":{"id":"live"},"transport":{"type":"webrtc","sdp":"answer"}}')

    monkeypatch.setattr(voice_live.urllib.request, "urlopen", exchange)
    headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_server.app), base_url="http://test", headers=headers,
    ) as client:
        response = await client.post(
            "/api/audio/voice-live/session", params={"profile": "alpha"}, json={"sdp": "v=0\r\n", **settings},
        )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "ok": True, "session": {"id": "live"}, "transport": {"type": "webrtc", "sdp": "answer"},
    }
    assert captured["session"] == {
        "model": "model-alpha", "instructions": voice_live.LIVE_PERSONA + "\n\nProfile alpha." + extra,
        "audio": {"output": {"voice": expected_voice}}, "delegation": {"type": "client"},
    }
    assert captured["transport"] == {"type": "webrtc", "sdp": "v=0\r\n"}
    assert {path: path.read_bytes() for path in voice_profiles} == voice_profiles


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [
    {"instructions": ""}, {"instructions": " \n "}, {"instructions": "x" * 32769},
    {"instructions": 42}, {"voice": ""}, {"voice": " "}, {"voice": " marin"},
    {"voice": "marin "}, {"voice": "bad\nvoice"}, {"voice": "bad\x7fvoice"},
    {"voice": "v" * 129}, {"voice": ["marin"]},
])
async def test_invalid_overrides_are_rejected_before_exchange(monkeypatch, settings):
    def no_exchange(*args, **kwargs):
        pytest.fail("invalid overrides must not reach credential resolution or the provider")

    monkeypatch.setattr(voice_live, "_resolve_credentials", no_exchange)
    monkeypatch.setattr(voice_live.urllib.request, "urlopen", no_exchange)
    headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_server.app), base_url="http://test", headers=headers,
    ) as client:
        response = await client.post("/api/audio/voice-live/session", json={"sdp": "v=0\r\n", **settings})
    assert response.status_code == 422, response.text
    with pytest.raises(ValueError):
        voice_live.build_session_config(**settings)
    with pytest.raises(ValueError):
        voice_live.create_webrtc_session("v=0\r\n", **settings)
