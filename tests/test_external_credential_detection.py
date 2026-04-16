"""Tests for detect_external_credentials() -- Phase 2 credential sync."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.auth import detect_external_credentials


class TestDetectCodexCLI:
    def test_detects_valid_codex_auth(self, tmp_path, monkeypatch):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        auth = codex_dir / "auth.json"
        auth.write_text(json.dumps({
            "tokens": {"access_token": "tok-123", "refresh_token": "ref-456"}
        }))
        monkeypatch.setenv("CODEX_HOME", str(codex_dir))
        result = detect_external_credentials()
        codex_hits = [c for c in result if c["provider"] == "openai-codex"]
        assert len(codex_hits) == 1
        assert "Codex CLI" in codex_hits[0]["label"]

    def test_skips_codex_without_access_token(self, tmp_path, monkeypatch):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(json.dumps({"tokens": {}}))
        monkeypatch.setenv("CODEX_HOME", str(codex_dir))
        result = detect_external_credentials()
        assert not any(c["provider"] == "openai-codex" for c in result)

    def test_skips_missing_codex_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nonexistent"))
        result = detect_external_credentials()
        assert not any(c["provider"] == "openai-codex" for c in result)

    def test_skips_malformed_codex_auth(self, tmp_path, monkeypatch):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{bad json")
        monkeypatch.setenv("CODEX_HOME", str(codex_dir))
        result = detect_external_credentials()
        assert not any(c["provider"] == "openai-codex" for c in result)

    def test_returns_empty_when_nothing_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nonexistent"))
        result = detect_external_credentials()
        assert result == []
