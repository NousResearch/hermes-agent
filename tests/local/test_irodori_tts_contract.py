from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_irodori():
    path = REPO_ROOT / "skills" / "audio" / "irodori-tts" / "scripts" / "irodori_tts.py"
    spec = importlib.util.spec_from_file_location("local_secretary_irodori_tts", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_irodori = _load_irodori()
chunk_sentences = _irodori.chunk_sentences
synthesize_speech = _irodori.synthesize_speech


class _FakeAudioResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return b"RIFF"


def test_chunk_sentences_splits_japanese():
    text = (
        "\u304a\u306f\u3088\u3046\u3002"
        "\u4eca\u65e5\u306e\u4e88\u5b9a\u3067\u3059\u3002"
        "\u5915\u65b9\u306b\u78ba\u8a8d\u3002"
    )
    chunks = chunk_sentences(text, max_chars=20)
    assert len(chunks) >= 2


def test_synthesize_speech_dry_run_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("IRODORI_TTS_OUTPUT_DIR", str(tmp_path))
    out = tmp_path / "voice.wav"
    payload = json.loads(
        synthesize_speech(
            "Dry run only.",
            output_path=out,
            dry_run=True,
        )
    )
    assert payload["success"] is True
    assert payload["dry_run"] is True
    assert out.exists()


def test_synthesize_speech_requires_confirmation_outside_safe_root(tmp_path, monkeypatch):
    safe_root = tmp_path / "safe"
    unsafe_root = tmp_path / "unsafe"
    monkeypatch.setenv("IRODORI_TTS_OUTPUT_DIR", str(safe_root))

    blocked = json.loads(
        synthesize_speech(
            "Dry run only.",
            output_path=unsafe_root / "voice.wav",
            dry_run=True,
        )
    )
    assert blocked["success"] is False
    assert blocked["confirmation_required"] is True

    allowed = json.loads(
        synthesize_speech(
            "Dry run only.",
            output_path=unsafe_root / "voice.wav",
            dry_run=True,
            confirmed=True,
        )
    )
    assert allowed["success"] is True


def test_irodori_api_key_only_sent_to_loopback_by_default(monkeypatch):
    seen_authorization: list[str | None] = []

    def fake_urlopen(req, timeout: float = 300):
        seen_authorization.append(req.headers.get("Authorization"))
        return _FakeAudioResponse()

    monkeypatch.setenv("IRODORI_API_KEY", "secret-token")
    monkeypatch.delenv("IRODORI_TTS_ALLOW_REMOTE_API_KEY", raising=False)
    monkeypatch.setattr(_irodori, "urlopen", fake_urlopen)

    _irodori._synthesize_chunk(
        base_url="https://tts.example.invalid",
        text="hello",
        voice="none",
        response_format="wav",
        speed=1.0,
        seed=None,
    )
    _irodori._synthesize_chunk(
        base_url="http://127.0.0.1:8088",
        text="hello",
        voice="none",
        response_format="wav",
        speed=1.0,
        seed=None,
    )

    assert seen_authorization == [None, "Bearer secret-token"]


def test_health_contract_helper_import():
    from agent.local_secretary.llama_contract import MIN_CONTEXT_SIZE

    assert MIN_CONTEXT_SIZE == 64000
