"""Cross-layer contract for the OpenRouter voice providers (STT + TTS).

The three layers that must agree for a provider to be selectable AND to show its options in the
GUI — runtime built-in sets, the served config schema, and the config defaults — drifted apart
twice while this provider was added (a provider absent from the schema renders no row at all, and
a key the schema advertises but DEFAULT_CONFIG doesn't seed used to be reported by the CLI as
"not a recognized config key"). These assertions are the cheap guard against a repeat.
"""

from __future__ import annotations

import pytest


def test_provider_is_selectable_in_the_served_schema():
    """The dropdowns are schema/ENUM-driven: no option = no way to pick it."""
    from hermes_cli.web_server_config import _schema_with_dynamic_provider_options

    merged = _schema_with_dynamic_provider_options()
    assert "openrouter" in merged["stt.provider"]["options"]


@pytest.mark.parametrize("key", [
    "stt.provider", "stt.openrouter.model", "stt.openrouter.language",
])
def test_option_rows_exist_in_the_schema(key):
    """Every key the Voice tab renders must be in the served schema, with a category."""
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    entry = CONFIG_SCHEMA.get(key)
    assert entry is not None, f"{key} missing from CONFIG_SCHEMA — the row will not render"
    assert isinstance(entry.get("description"), str) and entry["description"].strip()
    assert entry.get("category"), f"{key} has no category (breaks test_no_single_field_categories)"


def test_schema_advertised_keys_are_recognized_by_the_config_cli():
    """`hermes config set` must not call a schema-advertised key "not a recognized config key"."""
    from hermes_cli.config import _SCHEMA_ONLY_LEAF_KEYS, _validate_config_key
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    for key in _SCHEMA_ONLY_LEAF_KEYS:
        assert key in CONFIG_SCHEMA, (
            f"{key} is exempted from the DEFAULT_CONFIG walk but is not advertised by the schema")
        assert _validate_config_key(key) == (True, None)


def test_openrouter_is_registered_as_an_stt_builtin():
    """The built-in set and its registry mirror (drift throws at dispatch time)."""
    from agent.transcription_registry import _BUILTIN_NAMES as stt_registry
    from tools.transcription_common import BUILTIN_STT_PROVIDERS

    assert "openrouter" in BUILTIN_STT_PROVIDERS == stt_registry


def test_default_config_ships_the_stt_provider_block():
    """The block gives the schema its per-provider row and the GUI its starting values."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    stt_block = DEFAULT_CONFIG["stt"]["openrouter"]
    assert stt_block["model"] and stt_block["language"] == ""
    # No seeded stt.provider: a stored value counts as an explicit pick and disables auto-detect.
    assert "provider" not in DEFAULT_CONFIG["stt"]


def test_client_direct_speaks_the_openai_wire(monkeypatch):
    """The desktop skips the relay hop only when a wire is resolvable for the selected provider."""
    from tools import transcription_tools
    from tools import voice_client_config as vcc

    monkeypatch.setattr(transcription_tools, "_load_stt_config",
                        lambda: {"provider": "openrouter", "openrouter": {}})
    monkeypatch.setattr(transcription_tools, "_resolve_provider_key", lambda *a, **kw: "test-key")

    stt = vcc._resolve_stt_client_config()

    assert (stt["mode"], stt["wire"]) == ("direct", vcc.STT_WIRE_OPENAI)


def _desktop_constant_list(key: str) -> list[str]:
    """Slugs inside a desktop ``ENUM_OPTIONS`` list (the desktop cannot import Python constants)."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2]
            / "apps/desktop/src/app/settings/constants.ts").read_text()
    block = re.search(rf"'{re.escape(key)}': \[(.*?)\n  \]", text, re.S)
    assert block is not None, f"{key} missing from the desktop constants"
    return re.findall(r"'([^']+)'", block.group(1))


def test_desktop_stt_suggestions_cover_the_shared_catalog():
    """The CLI picker imports OPENROUTER_STT_MODELS; the desktop keeps a hand-copied list for the
    same field. They had already drifted (15 entries vs 20), so pin containment."""
    from tools.transcription_common import OPENROUTER_STT_MODELS

    desktop = _desktop_constant_list("stt.openrouter.model")
    missing = [slug for slug in OPENROUTER_STT_MODELS if slug not in desktop]
    assert not missing, f"desktop catalog is missing {missing} from OPENROUTER_STT_MODELS"


def test_desktop_renders_the_openrouter_model_row():
    """A schema-advertised row the desktop never lists stays invisible: ``sectionFieldEntries``
    walks the static ``SECTIONS`` list, not the served schema. Pin the row, its free-input escape
    hatch, and its copy (the fallback label is prettyName(lastSegment) => a bare "Model")."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2]
            / "apps/desktop/src/app/settings/constants.ts").read_text()

    voice_section = text.split("id: 'voice'", 1)[1].split("id: '", 1)[0]
    assert "'stt.openrouter.model'" in voice_section, (
        "row missing from the Voice section - sectionFieldEntries only walks SECTIONS, so the "
        "field would never render however the schema advertises it")

    free_input = text.split("export const FREE_INPUT_KEYS", 1)[1]
    assert "'stt.openrouter.model'" in free_input, "row must stay free-input (live catalog moves)"

    assert "'stt.openrouter.model': 'OpenRouter Model'" in text, "FIELD_LABELS copy missing"
    assert "'stt.openrouter.model': 'Vendor-prefixed OpenRouter slug" in text, (
        "FIELD_DESCRIPTIONS copy missing")


def test_cli_picker_catalog_is_the_shared_stt_catalog():
    """No second hand-typed copy in the CLI picker."""
    from hermes_cli.tools_config_providers import STT_MODEL_CATALOG
    from tools.transcription_common import OPENROUTER_STT_MODELS

    assert STT_MODEL_CATALOG["openrouter"] == list(OPENROUTER_STT_MODELS)


def test_desktop_defaults_match_the_runtime_defaults():
    """A default changed in one layer must not leave the other advertising the old model."""
    from tools.transcription_common import DEFAULT_OPENROUTER_STT_MODEL

    assert DEFAULT_OPENROUTER_STT_MODEL in _desktop_constant_list("stt.openrouter.model")


def test_wav_only_models_are_a_probed_capability_not_a_guess():
    """Probed live 2026-09-14 with an MP3: 19 of 20 OpenRouter transcription models accept it, and
    meta/muse-voice-transcribe-1.0 alone answers "requires WAV audio (input is not a RIFF/WAVE
    container)" — then rejects 22.05/44.1 kHz WAV, accepting only 16 or 24 kHz."""
    from tools.transcription_common import (STT_WAV_SAMPLE_RATES, stt_requires_wav)

    assert stt_requires_wav("meta/muse-voice-transcribe-1.0") is True
    assert stt_requires_wav("META/Muse-Voice-Transcribe-1.0") is True
    assert stt_requires_wav("openai/whisper-large-v3") is False
    assert stt_requires_wav(None) is False
    assert STT_WAV_SAMPLE_RATES == (16000, 24000)


def test_client_direct_advertises_the_container_the_model_needs(monkeypatch):
    """The desktop converts WebM/Opus to WAV only when the resolver says the model requires it."""
    from tools import transcription_tools, voice_client_config as vcc

    def resolve(model: str) -> dict:
        monkeypatch.setattr(transcription_tools, "_load_stt_config",
                            lambda: {"provider": "openrouter", "openrouter": {"model": model}})
        monkeypatch.setattr(transcription_tools, "_resolve_provider_key", lambda *a, **kw: "test-key")
        return vcc._resolve_stt_client_config()

    wav_model = resolve("meta/muse-voice-transcribe-1.0")
    assert wav_model["audio_format"] == "wav"
    assert wav_model["model"] == "meta/muse-voice-transcribe-1.0"
    assert "audio_format" not in resolve("openai/whisper-large-v3")


@pytest.mark.parametrize("error_text, expected", [
    ('Meta transcription requires WAV audio (input is not a RIFF/WAVE container)', "wav"),
    ('requires a 16000 Hz or 24000 Hz WAV sample rate (received 22050 Hz)', "wav"),
    ("Unsupported file format", "m4a"),
    ("audio is corrupted", "m4a"),
    ("invalid file: no audio stream", "m4a"),
    ("OpenRouter is rate limiting you", None),
])
def test_retry_container_is_read_from_the_error(error_text, expected):
    """Endpoints name the container they want; the retry must not keep its own model list."""
    from tools.transcription_cloud import _stt_retry_target

    assert _stt_retry_target(error_text) == expected


def test_wav_transcode_targets_the_required_rate(monkeypatch, tmp_path):
    """target='wav' produces mono 16-bit PCM at the requested rate (no real ffmpeg in tests)."""
    from tools import transcription_audio as ta

    calls = []
    monkeypatch.setattr(ta, "_find_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ta, "_run_ffmpeg_wav_encode",
                        lambda ffmpeg, src, dst, *, sample_rate: calls.append((src, dst, sample_rate)))

    converted, error = ta._transcode_audio_for_stt(
        str(tmp_path / "voice.webm"), str(tmp_path), target="wav", sample_rate=16000)

    assert error is None
    assert converted.endswith("-stt.wav")
    assert calls == [(str(tmp_path / "voice.webm"), converted, 16000)]


def test_stt_response_format_matches_what_each_provider_accepts(monkeypatch):
    """A live voice note proved this matters: OpenRouter rejects response_format="text" with HTTP
    400 ("Only json and verbose_json are supported"), while OpenAI/Groq return a bare string for it.
    The desktop sends whatever the resolver puts in the wire payload."""
    from tools import transcription_tools, voice_client_config as vcc

    def resolve(provider: str) -> dict:
        monkeypatch.setattr(transcription_tools, "_load_stt_config",
                            lambda: {"provider": provider, provider: {}})
        monkeypatch.setattr(transcription_tools, "_resolve_provider_key", lambda *a, **kw: "test-key")
        # openai resolves through its own audio chain, not _resolve_provider_key — without this the
        # case degrades to a relay verdict (no key in the test env) and has no wire payload at all.
        monkeypatch.setattr(transcription_tools, "_resolve_openai_audio_client_config",
                            lambda: ("test-key", "https://api.openai.com/v1"))
        return vcc._resolve_stt_client_config()

    assert resolve("openrouter")["response_format"] == "json"
    assert resolve("groq")["response_format"] == "text"
    assert resolve("openai")["response_format"] == "text"


@pytest.mark.parametrize("expected_attr", ["OPENROUTER_STT_BASE_URL"])
def test_base_url_override_is_honoured_by_the_shared_reader(expected_attr):
    """``stt.openrouter.base_url`` must beat the env default — the shipped config comment and the
    docs both promise this key, and it must apply to relay and client-direct alike."""
    from tools import transcription_common as tc

    reader, module = tc.openrouter_stt_base_url, tc

    assert reader(None) == getattr(module, expected_attr)
    assert reader({}) == getattr(module, expected_attr)
    assert reader({"base_url": "https://stt.example/v1/"}) == "https://stt.example/v1"
    assert reader({"base_url": "   "}) == getattr(module, expected_attr)
