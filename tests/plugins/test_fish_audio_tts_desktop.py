from __future__ import annotations

from hermes_cli import web_server


def test_tts_plugin_credentials_are_grouped_for_desktop_keys(monkeypatch):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "test-secret")
    metadata = web_server._catalog_provider_env_metadata()
    row = metadata["FISH_AUDIO_API_KEY"]
    assert row["provider"] == "tts:fishaudio"
    assert row["provider_label"] == "Fish Audio"
    assert row["category"] == "provider"
    assert row["is_password"] is True
    assert "test-secret" not in str(metadata)
