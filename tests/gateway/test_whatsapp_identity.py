"""Tests for gateway.whatsapp_identity alias resolution path."""

import json

from gateway.whatsapp_identity import expand_whatsapp_aliases


def test_aliases_resolve_on_modern_platforms_layout(tmp_path, monkeypatch):
    tmp_home = tmp_path / "hermes-home"
    mapping_dir = tmp_home / "platforms" / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "lid-mapping-999999999999999.json").write_text(
        json.dumps("15551234567@s.whatsapp.net"),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    assert expand_whatsapp_aliases("999999999999999@lid") == {
        "999999999999999",
        "15551234567",
    }


def test_aliases_resolve_on_legacy_layout(tmp_path, monkeypatch):
    tmp_home = tmp_path / "hermes-home"
    mapping_dir = tmp_home / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "lid-mapping-999999999999999.json").write_text(
        json.dumps("15551234567@s.whatsapp.net"),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    assert expand_whatsapp_aliases("999999999999999@lid") == {
        "999999999999999",
        "15551234567",
    }
