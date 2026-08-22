import base64

from agent.gemini_native_adapter import _extract_multimodal_parts
from agent.media_routing import build_native_media_content_parts, supported_input_modalities


def test_active_gemini_flash_lite_declares_all_multimodal_inputs(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch
    fake_info = SimpleNamespace(input_modalities=["text", "image", "pdf", "video", "audio"])
    with patch("agent.models_dev.get_model_info", return_value=fake_info):
        modalities = supported_input_modalities("openrouter", "google/gemini-3.5-flash-lite")
        assert {"text", "image", "pdf", "video", "audio"} <= modalities


def test_builds_openrouter_native_parts_for_all_media(tmp_path):
    files = {
        "image": (tmp_path / "photo.png", "image/png"),
        "pdf": (tmp_path / "notes.pdf", "application/pdf"),
        "audio": (tmp_path / "sample.mp3", "audio/mpeg"),
        "video": (tmp_path / "clip.mp4", "video/mp4"),
    }
    for path, _mime in files.values():
        path.write_bytes(b"test-bytes")

    parts, skipped = build_native_media_content_parts(
        "analyze everything",
        [
            {"path": str(path), "mime_type": mime, "modality": modality}
            for modality, (path, mime) in files.items()
        ],
    )

    assert skipped == []
    assert [part["type"] for part in parts] == [
        "text", "image_url", "file", "input_audio", "video_url",
    ]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[2]["file"]["file_data"].startswith("data:application/pdf;base64,")
    assert parts[3]["input_audio"]["format"] == "mp3"
    assert parts[3]["input_audio"]["data"] == base64.b64encode(b"test-bytes").decode()
    assert parts[4]["video_url"]["url"].startswith("data:video/mp4;base64,")


def test_gemini_native_adapter_converts_non_image_media_to_inline_data(tmp_path):
    pdf = tmp_path / "notes.pdf"
    audio = tmp_path / "sample.mp3"
    video = tmp_path / "clip.mp4"
    for path in (pdf, audio, video):
        path.write_bytes(b"native-data")

    parts, skipped = build_native_media_content_parts(
        "analyze",
        [
            {"path": str(pdf), "mime_type": "application/pdf", "modality": "pdf"},
            {"path": str(audio), "mime_type": "audio/mpeg", "modality": "audio"},
            {"path": str(video), "mime_type": "video/mp4", "modality": "video"},
        ],
    )
    assert skipped == []

    native = _extract_multimodal_parts(parts)
    assert native[0] == {"text": parts[0]["text"]}
    assert [part["inlineData"]["mimeType"] for part in native[1:]] == [
        "application/pdf", "audio/mpeg", "video/mp4",
    ]


def test_unreadable_attachment_falls_back_without_fake_media(tmp_path):
    missing = tmp_path / "missing.pdf"
    parts, skipped = build_native_media_content_parts(
        "read this",
        [{"path": str(missing), "mime_type": "application/pdf", "modality": "pdf"}],
    )
    assert parts == [{"type": "text", "text": "read this"}]
    assert skipped == [str(missing)]


def test_attachment_exceeding_size_ceiling_is_skipped(tmp_path):
    oversized = tmp_path / "large_voice.ogg"
    oversized.write_bytes(b"x" * 200)

    # Test with custom max_size_bytes
    parts, skipped = build_native_media_content_parts(
        "listen",
        [{"path": str(oversized), "mime_type": "audio/ogg", "modality": "audio"}],
        max_size_bytes=100,
    )
    assert parts == [{"type": "text", "text": "listen"}]
    assert skipped == [str(oversized)]


def test_audio_format_and_mime_normalization():
    from agent.media_routing import normalize_audio_format, normalize_audio_mime
    from pathlib import Path

    assert normalize_audio_format(Path("audio.mp3")) == "mp3"
    assert normalize_audio_format(Path("audio.ogg")) == "ogg"
    assert normalize_audio_format(Path("audio.opus")) == "ogg"
    assert normalize_audio_format(Path("audio.wav")) == "wav"
    assert normalize_audio_format(Path("audio.flac")) == "flac"
    assert normalize_audio_format(mime="audio/mpeg") == "mp3"
    assert normalize_audio_format(mime="audio/ogg") == "ogg"

    assert normalize_audio_mime("mp3") == "audio/mpeg"
    assert normalize_audio_mime("mpeg") == "audio/mpeg"
    assert normalize_audio_mime("ogg") == "audio/ogg"
    assert normalize_audio_mime("opus") == "audio/ogg"
    assert normalize_audio_mime("wav") == "audio/wav"
    assert normalize_audio_mime("flac") == "audio/flac"
    assert normalize_audio_mime("audio/custom") == "audio/custom"


def test_modality_auto_derived_when_omitted(tmp_path):
    img = tmp_path / "screenshot.png"
    img.write_bytes(b"png-data")
    audio = tmp_path / "memo.ogg"
    audio.write_bytes(b"ogg-data")

    parts, skipped = build_native_media_content_parts(
        "check",
        [
            {"path": str(img), "mime_type": "image/png"},
            {"path": str(audio), "mime_type": "audio/ogg"},
        ],
    )
    assert skipped == []
    assert [part["type"] for part in parts] == ["text", "image_url", "input_audio"]

