"""Contract tests for OpenRouter as a TTS provider (issue #15726).

This provider has to agree across four layers: the runtime registry, the shipped config defaults,
the served schema the desktop renders its rows from, and the desktop's own hand-copied suggestion
lists. Each test pins one seam, so a change in one layer cannot silently strand another.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def test_provider_is_selectable_in_the_served_schema():
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    assert "openrouter" in CONFIG_SCHEMA["tts.provider"]["options"]


@pytest.mark.parametrize("key", ["tts.provider", "tts.openrouter.model", "tts.openrouter.voice"])
def test_option_rows_exist_in_the_schema(key):
    """Every key the Voice tab renders must be in the served schema, with a description."""
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    entry = CONFIG_SCHEMA.get(key)
    assert entry is not None, f"{key} missing from CONFIG_SCHEMA - the row will not render"
    assert isinstance(entry.get("description"), str) and entry["description"].strip()


def test_openrouter_is_registered_as_a_tts_builtin():
    """The built-in set and its registry mirror (drift throws at dispatch time)."""
    from agent.tts_registry import _BUILTIN_NAMES as tts_registry
    from tools.tts_command_provider import BUILTIN_TTS_PROVIDERS

    assert "openrouter" in BUILTIN_TTS_PROVIDERS == tts_registry


def test_dispatch_resolves_a_generator_for_openrouter():
    """A registered built-in with no dispatch entry would raise only at call time."""
    from tools import tts_tool

    entry = tts_tool._BUILTIN_DISPATCH["openrouter"]
    names = [item for item in entry if isinstance(item, str)]
    assert any(getattr(tts_tool, name, None) is not None for name in names), entry


def test_default_config_ships_the_provider_block():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    block = DEFAULT_CONFIG["tts"]["openrouter"]
    assert block["model"] and block["voice"]


def test_shipped_defaults_match_the_runtime_fallbacks():
    """Defaults live in two places; a change to one must not strand the other.

    The runtime side honours TTS_OPENROUTER_MODEL / TTS_OPENROUTER_VOICE, so this compares the
    fallbacks - unset those variables if this fails locally with them exported.
    """
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from tools import tts_tool_openai as t

    block = DEFAULT_CONFIG["tts"]["openrouter"]
    assert block["model"] == t.DEFAULT_OPENROUTER_TTS_MODEL
    assert block["voice"] == t.DEFAULT_OPENROUTER_TTS_VOICE


def test_openrouter_is_transcoded_to_opus_for_voice_bubbles():
    """The endpoint never returns Ogg, so delivery must convert rather than pass it through."""
    from tools import tts_tool

    assert "openrouter" in tts_tool._FFMPEG_OPUS_PROVIDERS
    assert "openrouter" not in tts_tool._NATIVE_OPUS_PROVIDERS


def test_client_direct_speaks_the_openai_wire(monkeypatch):
    """The desktop skips the relay hop only when a wire is resolvable for the selected provider."""
    from tools import tts_tool
    from tools import voice_client_config as vcc

    monkeypatch.setattr(tts_tool, "_load_tts_config",
                        lambda: {"provider": "openrouter", "openrouter": {}})
    monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *a, **kw: "test-key")

    tts = vcc._resolve_tts_client_config()

    assert (tts["mode"], tts["wire"]) == ("direct", vcc.TTS_WIRE_OPENAI)


def test_client_direct_falls_back_to_relay_without_credentials(monkeypatch):
    """No key must degrade to the relay, not to a broken direct request."""
    from tools import tts_tool
    from tools import voice_client_config as vcc

    monkeypatch.setattr(tts_tool, "_load_tts_config",
                        lambda: {"provider": "openrouter", "openrouter": {}})
    monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *a, **kw: None)

    assert vcc._resolve_tts_client_config()["mode"] == "relay"


def test_base_url_override_is_honoured_by_the_shared_reader():
    """``tts.openrouter.base_url`` beats the env default for relay and client-direct alike."""
    from tools import tts_tool_openai

    reader = tts_tool_openai.openrouter_tts_base_url

    assert reader(None) == tts_tool_openai.OPENROUTER_TTS_BASE_URL
    assert reader({}) == tts_tool_openai.OPENROUTER_TTS_BASE_URL
    assert reader({"base_url": "https://speech.example/v1/"}) == "https://speech.example/v1"
    assert reader({"base_url": "   "}) == tts_tool_openai.OPENROUTER_TTS_BASE_URL


def _desktop_constant_list(key: str) -> list[str]:
    """Slugs inside a desktop ``ENUM_OPTIONS`` list (the desktop cannot import Python constants)."""
    text = (Path(__file__).resolve().parents[2]
            / "apps/desktop/src/app/settings/constants.ts").read_text()
    block = re.search(rf"'{re.escape(key)}': \[(.*?)\n  \]", text, re.S)
    assert block is not None, f"{key} missing from the desktop constants"
    return re.findall(r"'([^']+)'", block.group(1))


def test_desktop_suggestions_include_the_runtime_defaults():
    """The desktop keeps a hand-copied list for a free-input field; pin the default at least."""
    from tools.tts_tool_openai import DEFAULT_OPENROUTER_TTS_MODEL, DEFAULT_OPENROUTER_TTS_VOICE

    assert DEFAULT_OPENROUTER_TTS_MODEL in _desktop_constant_list("tts.openrouter.model")
    assert DEFAULT_OPENROUTER_TTS_VOICE in _desktop_constant_list("tts.openrouter.voice")
