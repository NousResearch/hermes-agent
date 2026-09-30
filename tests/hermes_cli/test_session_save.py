"""Tests for the /save current-session export helpers."""

import json

import pytest

from hermes_cli.session_export import (
    SAVE_FORMATS,
    default_save_filename,
    normalize_save_format,
    render_session_for_save,
)

SESSION = {
    "id": "20260814_abc123",
    "title": "Test Session",
    "model": "test-model",
    "started_at": 1755100000,
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "tool", "content": "tool output"},
    ],
}

class TestNormalizeSaveFormat:

    def test_aliases(self):
        assert normalize_save_format("markdown") == "md"
        assert normalize_save_format("MD") == "md"
        assert normalize_save_format("HTML") == "html"
        assert normalize_save_format("snapshot") == "json"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            normalize_save_format("pdf")

    def test_all_declared_formats_normalize_to_themselves(self):
        for fmt in SAVE_FORMATS:
            assert normalize_save_format(fmt) == fmt

class TestRenderSessionForSave:
    def test_json_round_trips(self):
        out = render_session_for_save(SESSION, "json")
        data = json.loads(out)
        assert data["id"] == SESSION["id"]
        assert len(data["messages"]) == 3

    def test_markdown_contains_messages(self):
        out = render_session_for_save(SESSION, "md")
        assert "Hello" in out
        assert "Hi there" in out

    def test_html_is_standalone_document(self):
        out = render_session_for_save(SESSION, "html")
        assert out.lstrip().lower().startswith("<!doctype html")
        assert "Hello" in out

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            render_session_for_save(SESSION, "pdf")

class TestDefaultSaveFilename:

    def test_hostile_session_id_sanitized(self):
        name = default_save_filename("../../etc/passwd", "json")
        assert "/" not in name
        assert ".." not in name.replace("etcpasswd", "")
