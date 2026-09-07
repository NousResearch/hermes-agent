"""Invariants for conservative native voice-note payload construction."""

import base64


def test_builder_sniffs_audio_and_enforces_the_encoded_ceiling(tmp_path):
    from agent.audio_routing import build_native_audio_content_parts

    voice = tmp_path / "voice.bin"
    raw = b"OggS" + b"voice-bytes"
    voice.write_bytes(raw)
    attachment = {"path": str(voice), "mime_type": "audio/wav"}

    parts, skipped = build_native_audio_content_parts("listen", [attachment])

    assert skipped == []
    audio = next(part["input_audio"] for part in parts if part["type"] == "input_audio")
    assert audio["format"] == "ogg"
    assert base64.b64decode(audio["data"]) == raw
    assert "audio/ogg" in parts[0]["text"]

    limited_parts, skipped = build_native_audio_content_parts(
        "listen", [attachment], max_encoded_bytes=4,
    )
    assert limited_parts == [{"type": "text", "text": "listen"}]
    assert skipped == [str(voice)]


def test_routing_is_capability_driven_but_endpoint_denials_are_hard(monkeypatch):
    import agent.audio_routing as audio_routing

    assert audio_routing.normalize_audio_mime("m4a") == "audio/m4a"
    monkeypatch.setattr(audio_routing, "supported_input_modalities", lambda *_: {"text", "audio"})
    assert audio_routing.decide_audio_input_mode("openrouter", "google/gemini", "auto") == "native"
    assert audio_routing.decide_audio_input_mode("openrouter", "unknown", "stt") == "stt"
    assert audio_routing.decide_audio_input_mode("meta", "llama-audio", "native") == "stt"
    assert audio_routing.decide_audio_input_mode("openrouter", "vendor/muse-spark", "native") == "stt"


def test_direct_openai_aliases_reject_incompatible_container_when_conversion_fails(monkeypatch, tmp_path):
    import agent.audio_routing as audio_routing

    voice = tmp_path / "voice.ogg"
    voice.write_bytes(b"OggSvoice-bytes")
    monkeypatch.setattr(audio_routing, "transcode_audio_to_supported_format", lambda *_args: None)

    for provider in ("openai", "openai-api", "azure-foundry"):
        parts, skipped = audio_routing.build_native_audio_content_parts(
            "listen", [{"path": str(voice), "mime_type": "audio/ogg"}], target_provider=provider,
        )

        assert parts == [{"type": "text", "text": "listen"}]
        assert skipped == [str(voice)]
