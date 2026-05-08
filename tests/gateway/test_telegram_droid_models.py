"""Tests for Telegram Factory Droid model routing picker helpers."""

import json
from pathlib import Path

from gateway.config import Platform, PlatformConfig


def _make_adapter(tmp_path, monkeypatch):
    from gateway.platforms.telegram import TelegramAdapter

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.config = config
    adapter._droid_model_picker_state = {}
    return adapter


def _write_droid(path: Path, model: str = "inherit"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"name: {path.stem}\n"
        "description: test droid\n"
        f"model: {model}\n"
        "---\n"
        "body\n"
    )


def test_load_droid_byok_models_reads_factory_settings_id(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch)
    settings = tmp_path / ".factory" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({
        "customModels": [
            {"id": "custom:ONE", "displayName": "One"},
            {"modelId": "custom:IGNORED", "displayName": "Ignored legacy key"},
            {"id": "not-custom", "displayName": "Not custom"},
        ]
    }))

    assert adapter._load_droid_byok_models() == [{"id": "custom:ONE", "displayName": "One"}]


def test_apply_droid_inherit_updates_all_role_files_and_creates_backups(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch)
    droid_dir = tmp_path / ".factory" / "droids"
    for name in ["worker.md", "user-testing-flow-validator.md", "scrutiny-feature-reviewer.md"]:
        _write_droid(droid_dir / name, "custom:OLD")

    result = adapter._apply_droid_inherit()

    assert "Inherit mode applied" in result
    for name in ["worker.md", "user-testing-flow-validator.md", "scrutiny-feature-reviewer.md"]:
        text = (droid_dir / name).read_text()
        assert "model: inherit" in text
        assert list(droid_dir.glob(f"{name}.bak-*")), name


def test_write_droid_models_updates_single_role(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch)
    droid_dir = tmp_path / ".factory" / "droids"
    _write_droid(droid_dir / "worker.md", "inherit")

    result = adapter._write_droid_models({"worker.md": "custom:GLM-5.1-[Z.AI-BYOK]-2"})

    assert "worker.md" in result
    assert "custom:GLM-5.1-[Z.AI-BYOK]-2" in (droid_dir / "worker.md").read_text()
    assert list(droid_dir.glob("worker.md.bak-*"))
