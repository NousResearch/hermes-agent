"""Audio requests retain their profile through real config, SDK and HTTP paths."""

import io
import tempfile
import wave
from types import SimpleNamespace

import pytest
import yaml
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from agent import secret_scope
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, body_limit_middleware


@pytest.mark.asyncio
async def test_audio_profile_hops_keep_config_and_credentials_isolated(tmp_path, monkeypatch):
    homes = {name: tmp_path / "profiles" / name for name in ("alpha", "beta")}
    for home in homes.values():
        home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: tmp_path if name == "default" else homes[name],
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", tmp_path), *homes.items()],
    )
    received = []

    async def transcribe(request):
        reader = await request.multipart()
        fields = {}
        while (part := await reader.next()) is not None:
            fields[part.name] = bytes(await part.read())
        received.append((request.headers["Authorization"], fields))
        return web.Response(text="profile transcript", content_type="text/plain")

    backend = web.Application()
    backend.router.add_post("/v1/audio/transcriptions", transcribe)
    async with TestServer(backend) as backend_server:
        for name, home in homes.items():
            (home / ".env").write_text(f"API_SERVER_KEY=api-{name}-key-1234567890123456\n")
            (home / "config.yaml").write_text(yaml.safe_dump({
                "stt": {
                    "provider": "openai",
                    "openai": {
                        "api_key": f"stt-{name}-key",
                        "base_url": str(backend_server.make_url("/v1")),
                    },
                },
            }))
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter.gateway_runner = SimpleNamespace(config=GatewayConfig(multiplex_profiles=True))
        app = web.Application(middlewares=[
            adapter._make_profile_prefix_middleware(), body_limit_middleware,
        ])
        app.router.add_post(
            "/p/{profile}/v1/audio/transcriptions", adapter._handle_audio_transcriptions,
        )
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 16000)

        def form():
            body = FormData()
            body.add_field("file", audio.getvalue(), filename="voice.wav", content_type="audio/wav")
            body.add_field("model", "whisper-1")
            body.add_field("language", "zh")
            body.add_field("prompt", "FunASR SenseVoice")
            return body

        secret_scope.set_multiplex_active(True)
        try:
            async with TestClient(TestServer(app)) as client:
                for name in ("alpha", "beta", "alpha"):
                    response = await client.post(
                        f"/p/{name}/v1/audio/transcriptions", data=form(),
                        headers={"Authorization": f"Bearer api-{name}-key-1234567890123456"},
                    )
                    result = await response.json()
                    assert response.status == 200, result
                    assert result == {"text": "profile transcript"}
                denied = await client.post(
                    "/p/beta/v1/audio/transcriptions", data=form(),
                    headers={"Authorization": "Bearer api-alpha-key-1234567890123456"},
                )
                assert denied.status == 401
        finally:
            secret_scope.set_multiplex_active(False)

    assert [auth for auth, _ in received] == [
        "Bearer stt-alpha-key", "Bearer stt-beta-key", "Bearer stt-alpha-key",
    ]
    for _, fields in received:
        assert fields["model"] == b"whisper-1"
        assert fields["language"] == b"zh"
        assert fields["prompt"] == b"FunASR SenseVoice"
        assert fields["file"].startswith(b"RIFF")
    assert not list(tmp_path.glob("hermes-api-stt-*"))
