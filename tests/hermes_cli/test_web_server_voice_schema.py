"""Plugin-registered voice providers reach the config schema (#136485).

Desktop's Settings > Voice offers the ``tts.provider`` / ``stt.provider`` options the
schema serves, so plugin providers must be there on the first request, and
``stt.provider`` (no seeded default; unset = autodetect) must be served once it is set.
"""

import pytest

import hermes_cli.config as _cfg_mod
import hermes_cli.web_server_config as _web_server_config
from agent import transcription_registry, tts_registry
from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider
from hermes_cli import plugins


class _FakeTTSProvider(TTSProvider):
    @property
    def name(self):
        return "schema-plugin-tts"

    def synthesize(self, text, output_path, **kwargs):
        return output_path


class _FakeSTTProvider(TranscriptionProvider):
    @property
    def name(self):
        return "schema-plugin-stt"

    def transcribe(self, file_path, **kwargs):
        return {"success": True, "transcript": "", "provider": self.name}


@pytest.fixture(autouse=True)
def _voice_plugins(monkeypatch):
    """Registers the fake providers only when discovery runs."""

    def discover_voice_plugins():
        tts_registry.register_provider(_FakeTTSProvider())
        transcription_registry.register_provider(_FakeSTTProvider())

    monkeypatch.setattr(plugins, "discover_plugins", discover_voice_plugins)
    tts_registry._reset_for_tests()
    transcription_registry._reset_for_tests()
    yield
    tts_registry._reset_for_tests()
    transcription_registry._reset_for_tests()


def test_first_schema_request_offers_plugin_voice_providers(monkeypatch):
    monkeypatch.setattr(_cfg_mod, "load_config", lambda: {"stt": {"provider": "local"}})

    fields = _web_server_config._schema_with_dynamic_provider_options()

    assert "schema-plugin-tts" in fields["tts.provider"]["options"]
    assert fields["stt.provider"]["type"] == "select"
    assert fields["stt.provider"]["options"][:2] == ["local", "groq"]
    assert "schema-plugin-stt" in fields["stt.provider"]["options"]


def test_unset_stt_provider_adds_no_schema_field(monkeypatch):
    monkeypatch.setattr(_cfg_mod, "load_config", dict)

    assert "stt.provider" not in _web_server_config._schema_with_dynamic_provider_options()
