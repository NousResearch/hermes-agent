"""Voice replies through real extraction, sidecar HTTP and Spectrum's iMessage sender.

Only provider speech generation and the remote iMessage client are synthetic. A converted
M4A uploaded with the original MP3 filename is not a valid voice-upload identity.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from gateway.session_context import _SESSION_PLATFORM
from plugins.platforms.photon.adapter import PhotonAdapter
from tests.waiting import wait_until
from tools import tts_tool
from tools.tts_tool_delivery import _wrap_pcm_as_wav, _write_wav_bytes_as

SIDECAR = Path("plugins/platforms/photon/sidecar").resolve()
SDK = SIDECAR / "node_modules"
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or shutil.which("ffmpeg") is None
    or not (SDK / "@spectrum-ts/imessage/dist/index.js").is_file(),
    reason="Photon sidecar dependencies and ffmpeg are required",
)


def _write_sdk_fixture(root: Path) -> None:
    """Keep real builders/provider send logic, replace only application and remote I/O."""
    spectrum = root / "node_modules/spectrum-ts"
    (spectrum / "providers").mkdir(parents=True)
    (spectrum / "package.json").write_text(json.dumps({
        "type": "module", "exports": {".": "./index.js", "./providers/imessage": "./providers/imessage.js"},
    }))
    (spectrum / "index.js").write_text(
        f"export * from {json.dumps((SDK / 'spectrum-ts/dist/index.js').as_uri())}\n"
        "export async function Spectrum() { return {\n"
        "  messages: { [Symbol.asyncIterator]() { return { next: () => new Promise(() => {}) } } },\n"
        "  stop: async () => undefined\n"
        "} }\n")
    (root / "node_modules/@spectrum-ts").mkdir()
    (root / "node_modules/@spectrum-ts/core").symlink_to(SDK / "@spectrum-ts/core", target_is_directory=True)
    provider = f"""
import {{ imessage as real }} from {json.dumps((SDK / '@spectrum-ts/imessage/dist/index.js').as_uri())}
import {{ writeFile }} from 'node:fs/promises'
const records = []
let pending
let duration
const remote = {{
  attachments: {{ upload: async (input) => {{
    await writeFile('upload-' + records.length, input.data)
    pending = {{ fileName: input.fileName, bytes: input.data.length, duration }}
    return {{ attachment: {{ guid: 'synthetic-upload' }} }}
  }} }},
  messages: {{ sendAttachment: async (chat, guid, options) => {{
    records.push({{ ...pending, chat, guid, options }})
    await writeFile('capture.json', JSON.stringify(records))
    return {{ guid: '00000000-0000-4000-8000-000000000001', dateCreated: new Date() }}
  }} }}
}}
const space = {{ id: 'any;-;+15555550100', phone: 'shared', type: 'dm',
  send: async (builder) => {{
    const content = await builder.build()
    duration = content.duration
    return real.config().__definition.send({{ space, content, client: [{{ phone: 'shared', client: remote }}] }})
  }}
}}
export function imessage() {{ return {{ space: {{ create: async () => space, get: async () => space }} }} }}
imessage.config = () => ({{}})
"""
    (spectrum / "providers/imessage.js").write_text(provider)


@pytest.fixture
def sidecar(tmp_path, monkeypatch, request):
    root = tmp_path / "sidecar"
    root.mkdir()
    for source in SIDECAR.glob("*.mjs"):
        shutil.copyfile(source, root / source.name)
    _write_sdk_fixture(root)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("PHOTON_PROJECT_ID", "synthetic-project")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "synthetic-secret")
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "synthetic-token")
    monkeypatch.setenv("PHOTON_SIDECAR_PORT", str(port))
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    if getattr(request, "param", None) == "no_ffmpeg":
        env["PATH"] = str(tmp_path / "empty-bin")
    proc = subprocess.Popen(
        [shutil.which("node"), "index.mjs"], cwd=root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))
    adapter._http_client = object()

    def ready():
        import httpx

        if proc.poll() is not None:
            raise AssertionError(proc.communicate()[1])
        try:
            response = httpx.post(f"http://127.0.0.1:{port}/healthz", json={},
                                  headers={"X-Hermes-Sidecar-Token": "synthetic-token"},
                                  timeout=0.5, trust_env=False)
            return response.status_code == 200
        except httpx.TransportError:
            return False

    try:
        wait_until(ready, timeout=10, what="synthetic sidecar ready")
        yield adapter, root
    finally:
        proc.terminate()
        _, stderr = proc.communicate(timeout=10)
        print(stderr)
        assert not list(tmp_path.glob("hermes-photon-voice-*"))


def _probe_audio(path):
    return subprocess.run(
        ["ffmpeg", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, check=False, timeout=10,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("container", ["mp3", "wav", "m4a", "wav_with_mp3_suffix"])
async def test_tts_reply_uploads_real_m4a_voice(sidecar, tmp_path, monkeypatch, container):
    adapter, root = sidecar
    cache = Path(os.environ["HERMES_HOME"]) / "cache/audio"
    cache.mkdir(parents=True, exist_ok=True)
    wav = _wrap_pcm_as_wav(b"\x01\x00" * 4800)

    def generate(text, path, config):
        if container == "wav_with_mp3_suffix":
            Path(path).write_bytes(wav)
            return path
        return _write_wav_bytes_as(wav, path)

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
    monkeypatch.setattr(tts_tool, "_generate_gemini_tts", generate)
    monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(cache))
    token = _SESSION_PLATFORM.set("photon")
    try:
        output = json.loads(tts_tool.text_to_speech_tool(
            "Synthetic voice reply.",
            **({"output_path": str(cache / f"synthetic.{container}")}
               if container in {"wav", "m4a"} else {})))
    finally:
        _SESSION_PLATFORM.reset(token)
    assert output["success"], output
    # This false flag on the pin must NOT prevent Photon audio routing.
    assert output["voice_compatible"] is False
    source = SessionSource(platform=adapter.platform, chat_id="any;-;+15555550100", user_id="synthetic-owner")
    event = MessageEvent(text="Reply with a voice note", source=source, message_id="synthetic-request")
    response = await adapter._extract_response_content(
        output["media_tag"], event, "synthetic-session", is_ephemeral_response=False)
    assert response.media_files == [(output["file_path"], False)]
    assert response.local_files == []
    results = []
    await adapter._deliver_attachments(event, response, {}, anything_sent=False, record_delivery=results.append)
    assert len(results) == 1 and results[0].success, results
    records = json.loads((root / "capture.json").read_text())
    assert len(records) == 1
    record = records[0]
    assert record["options"]["isAudioMessage"] is True
    assert 0 < record["duration"] <= 300
    assert record["fileName"].endswith(".m4a"), record
    uploaded = (root / "upload-0").read_bytes()
    assert uploaded[4:8] == b"ftyp"
    probe = await asyncio.to_thread(_probe_audio, root / "upload-0")
    assert probe.returncode == 0, probe.stderr
    assert "Audio: aac" in probe.stderr
    assert "Duration: 00:00:00.20" in probe.stderr
    assert not adapter._should_auto_tts_for_chat(source.chat_id)
    ordinary = await adapter._extract_response_content(
        "Ordinary text reply.", event, "synthetic-next-turn", is_ephemeral_response=False)
    assert ordinary.media_files == [] and ordinary.local_files == []
    await adapter._deliver_attachments(event, ordinary, {}, anything_sent=True, record_delivery=results.append)
    assert len(json.loads((root / "capture.json").read_text())) == 1


@pytest.mark.asyncio
async def test_invalid_voice_never_reaches_upload(sidecar, tmp_path):
    from plugins.platforms.photon.adapter import PhotonSidecarError

    adapter, root = sidecar
    source = tmp_path / "invalid.mp3"
    source.write_bytes(b"not audio")
    with pytest.raises(PhotonSidecarError):
        await adapter._sidecar_call("/send-attachment", {
            "spaceId": "any;-;+15555550100", "path": str(source), "kind": "voice",
            "mimeType": "audio/mp4", "name": "misleading.m4a",
        })
    assert not (root / "capture.json").exists()
    assert not (root / "upload-0").exists()


@pytest.mark.asyncio
async def test_voice_upload_identity_is_unique_but_documents_are_unchanged(sidecar, tmp_path):
    adapter, root = sidecar
    source = tmp_path / "synthetic.wav"
    wav = _wrap_pcm_as_wav(b"\x01\x00" * 4800)
    source.write_bytes(wav)
    for _ in range(2):
        await adapter._sidecar_call("/send-attachment", {
            "spaceId": "any;-;+15555550100", "path": str(source), "kind": "voice",
        })
    await adapter._sidecar_call("/send-attachment", {
        "spaceId": "any;-;+15555550100", "path": str(source), "kind": "attachment",
    })
    records = json.loads((root / "capture.json").read_text())
    assert len(records) == 3
    assert records[0]["fileName"] != records[1]["fileName"]
    assert all(record["fileName"].endswith(".m4a") for record in records[:2])
    assert records[2]["fileName"] == source.name
    assert records[2]["options"].get("isAudioMessage") is not True
    assert (root / "upload-2").read_bytes() == wav


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar", ["no_ffmpeg"], indirect=True)
@pytest.mark.parametrize("container", ["wav_with_mp3_suffix", "m4a"])
async def test_missing_encoder_never_uploads_voice(sidecar, tmp_path, container):
    from plugins.platforms.photon.adapter import PhotonSidecarError

    adapter, root = sidecar
    source = tmp_path / ("valid.m4a" if container == "m4a" else "wav-disguised-as.mp3")
    wav = _wrap_pcm_as_wav(b"\x01\x00" * 4800)
    if container == "m4a":
        await asyncio.to_thread(_write_wav_bytes_as, wav, str(source))
    else:
        source.write_bytes(wav)
    with pytest.raises(PhotonSidecarError):
        await adapter._sidecar_call("/send-attachment", {
            "spaceId": "any;-;+15555550100", "path": str(source), "kind": "voice",
            "mimeType": "audio/mp4",
        })
    assert not (root / "capture.json").exists()
    assert not (root / "upload-0").exists()


def _unusable_m4a(root, case):
    source = root / "unusable.m4a"
    if case == "empty_audio":
        source.write_bytes(_wrap_pcm_as_wav(b""))
        return source
    if case == "overlong":
        source.write_bytes(_wrap_pcm_as_wav(b"\x01\x00" * (24000 * 301)))
        return source
    if case == "too_large":
        with source.open("wb") as handle:
            handle.seek(32 * 1024 * 1024)
            handle.write(b"x")
        return source
    if case == "header_only":
        source.write_bytes(b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00M4A isom")
        return source
    if case == "video_only":
        command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=16x16:r=10",
                   "-t", "0.2", "-an", "-c:v", "mpeg4", "-f", "mp4", str(source)]
    else:
        wav = root / "input.wav"
        wav.write_bytes(_wrap_pcm_as_wav(b"\x01\x00" * 4800))
        command = ["ffmpeg", "-v", "error", "-i", str(wav), "-c:a", "aac",
                   "-movflags", "+faststart", "-f", "ipod", str(source)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr
    if case == "truncated":
        assert _probe_audio(source).returncode == 0
        data = source.read_bytes()
        start = data.index(b"mdat") + 4
        source.write_bytes(data[:start + max(1, (len(data) - start) // 2)])
    return source


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["header_only", "truncated", "video_only", "empty_audio", "overlong", "too_large"])
async def test_unacceptable_audio_fails_before_remote_upload(sidecar, tmp_path, case):
    from plugins.platforms.photon.adapter import PhotonSidecarError

    adapter, root = sidecar
    source = await asyncio.to_thread(_unusable_m4a, tmp_path, case)
    with pytest.raises(PhotonSidecarError):
        await adapter._sidecar_call("/send-attachment", {
            "spaceId": "any;-;+15555550100", "path": str(source), "kind": "voice",
            "mimeType": "audio/mp4",
        })
    assert not (root / "capture.json").exists()
    assert not (root / "upload-0").exists()
