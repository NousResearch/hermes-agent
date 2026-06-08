"""Tests for persist-last-model feature (#42111).

Verifies that ``model.persist_last`` config option saves and restores
the last-used model/provider across CLI restarts.
"""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────

def _make_cli_module():
    """Import cli module functions without triggering full CLI init."""
    import importlib
    import cli as _cli
    importlib.reload(_cli)
    return _cli


# ── _save_persisted_model ──────────────────────────────────────────────

class TestSavePersistedModel:
    """Tests for _save_persisted_model()."""

    def test_creates_file_with_model_and_provider(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            cli._save_persisted_model(model="gpt-4", provider="openai")
        data = json.loads(target.read_text())
        assert data["model"] == "gpt-4"
        assert data["provider"] == "openai"
        assert "timestamp" in data

    def test_creates_file_with_empty_values(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            cli._save_persisted_model()
        data = json.loads(target.read_text())
        assert data["model"] == ""
        assert data["provider"] == ""

    def test_overwrites_existing_file(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        target.write_text('{"model": "old", "provider": "old", "timestamp": "x"}')
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            cli._save_persisted_model(model="new-model", provider="new-provider")
        data = json.loads(target.read_text())
        assert data["model"] == "new-model"
        assert data["provider"] == "new-provider"

    def test_no_exception_on_write_failure(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "nonexistent_dir" / "last_model.json"
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            # Should not raise
            cli._save_persisted_model(model="x", provider="y")


# ── _load_persisted_model ──────────────────────────────────────────────

class TestLoadPersistedModel:
    """Tests for _load_persisted_model()."""

    def test_returns_empty_when_no_file(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            result = cli._load_persisted_model()
        assert result == {}

    def test_returns_model_and_provider(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        target.write_text(json.dumps({
            "model": "claude-sonnet-4-20250514",
            "provider": "anthropic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            result = cli._load_persisted_model()
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["provider"] == "anthropic"

    def test_skips_empty_model_and_provider(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        target.write_text(json.dumps({
            "model": "",
            "provider": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            result = cli._load_persisted_model()
        assert result == {}

    def test_skips_stale_data_older_than_30_days(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        stale_ts = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        target.write_text(json.dumps({
            "model": "gpt-4",
            "provider": "openai",
            "timestamp": stale_ts,
        }))
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            result = cli._load_persisted_model()
        assert result == {}

    def test_accepts_data_within_30_days(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        recent_ts = (datetime.now(timezone.utc) - timedelta(days=29)).isoformat()
        target.write_text(json.dumps({
            "model": "gpt-4",
            "provider": "openai",
            "timestamp": recent_ts,
        }))
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            result = cli._load_persisted_model()
        assert result["model"] == "gpt-4"
        assert result["provider"] == "openai"

    def test_handles_corrupt_json(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        target.write_text("not valid json {{{")
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            result = cli._load_persisted_model()
        assert result == {}

    def test_handles_missing_timestamp(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        target.write_text(json.dumps({"model": "gpt-4", "provider": "openai"}))
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            result = cli._load_persisted_model()
        # No timestamp → no staleness check → should return data
        assert result["model"] == "gpt-4"
        assert result["provider"] == "openai"

    def test_roundtrip_save_then_load(self, tmp_path):
        cli = _make_cli_module()
        target = tmp_path / "last_model.json"
        with patch.object(cli, "_LAST_MODEL_FILE", target):
            cli._save_persisted_model(model="gemini-2.5-pro", provider="gemini")
            result = cli._load_persisted_model()
        assert result["model"] == "gemini-2.5-pro"
        assert result["provider"] == "gemini"
