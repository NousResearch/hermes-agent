"""Tests for tts.<provider>.streaming config field.

Per-provider streaming flag controls whether CLI voice mode plays audio
in real-time (chunk-by-chunk via sounddevice) instead of synthesizing a
complete file first. Defaults preserve existing behaviour:

- elevenlabs: streaming = true  (already wired in stream_tts_to_speaker)
- edge:       streaming = false (real-time path added in this PR)
- openai:     streaming = false (real-time path added in this PR)

The flag ships as an additive default and reaches users via the existing
deep-merge path, so no version-pinned snapshot assertion is needed — the
migration tests below cover the deep-merge invariant instead.
"""

import yaml


def test_default_config_has_streaming_fields():
    """DEFAULT_CONFIG declares streaming for all three real-time providers."""
    from hermes_cli.config import DEFAULT_CONFIG

    tts = DEFAULT_CONFIG["tts"]
    assert tts["edge"]["streaming"] is False
    assert tts["openai"]["streaming"] is False
    assert tts["elevenlabs"]["streaming"] is True


def test_v31_config_migrates_streaming_fields():
    """A v31 config gets streaming fields visible at runtime via deep-merge."""
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "_config_version": 31,
        "tts": {
            "provider": "edge",
            "edge": {"voice": "en-US-AriaNeural"},
            "openai": {"model": "gpt-4o-mini-tts", "voice": "alloy"},
            "elevenlabs": {
                "voice_id": "pNInz6obpgDQGcFmaJgB",
                "model_id": "eleven_multilingual_v2",
            },
        },
    }), encoding="utf-8")

    from hermes_cli.config import migrate_config, load_config
    migrate_config(interactive=False, quiet=True)

    # load_config deep-merges DEFAULT_CONFIG into user yaml, so the new
    # streaming fields are visible at runtime even if not written to disk.
    cfg = load_config()
    assert cfg["tts"]["edge"]["streaming"] is False
    assert cfg["tts"]["openai"]["streaming"] is False
    assert cfg["tts"]["elevenlabs"]["streaming"] is True


def test_v31_config_preserves_user_set_streaming():
    """User-set streaming=true must survive migration (deep-merge keeps user values)."""
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "_config_version": 31,
        "tts": {
            "edge": {"voice": "en-US-AriaNeural", "streaming": True},
            "openai": {"model": "gpt-4o-mini-tts", "voice": "alloy", "streaming": True},
        },
    }), encoding="utf-8")

    from hermes_cli.config import migrate_config, load_config
    migrate_config(interactive=False, quiet=True)

    cfg = load_config()
    assert cfg["tts"]["edge"]["streaming"] is True
    assert cfg["tts"]["openai"]["streaming"] is True
