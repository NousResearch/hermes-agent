"""Epoch zero is a timestamp; absent time fields must remain absent."""

from datetime import datetime

import pytest
from ruamel.yaml import YAML

from hermes_cli.session_export_html import generate_multi_session_html_export
from hermes_cli.session_export_md import render_session_markdown
from hermes_state import SessionDB


@pytest.mark.parametrize("fmt", ["html", "md", "qmd"])
def test_epoch_zero_survives_session_store_and_export(tmp_path, monkeypatch, fmt):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    try:
        assert db.import_sessions([{
            "id": "epoch-zero", "source": "cli", "started_at": 0,
            "messages": [{"role": "user", "content": "epoch content", "timestamp": 0}],
        }])["imported"] == 1
        session = db.export_session("epoch-zero")
        assert session["started_at"] == session["messages"][0]["timestamp"] == 0
        if fmt == "html":
            # Multi-session output exercises both the sidebar and body timestamps.
            text = generate_multi_session_html_export([session, {"id": "other", "messages": []}])
            local = datetime.fromtimestamp(0).strftime("%Y-%m-%d %H:%M:%S")
            assert f"<strong>Started:</strong> {local}" in text
            assert f'<div class="timestamp">{local}</div>' in text
            assert f"<span>{local.split(' ')[0]}</span>" in text
        else:
            text = render_session_markdown(session, fmt=fmt)
            frontmatter = YAML(typ="safe").load(text.split("---", 2)[1])
            assert frontmatter["created_at"] == "1970-01-01T00:00:00Z"
            assert "### User — 1970-01-01T00:00:00Z" in text
        assert "epoch content" in text
    finally:
        db.close()


@pytest.mark.parametrize("value", [0, None, ""])
def test_fallback_time_fields_distinguish_zero_from_unset(value):
    session = {
        "id": "fallback", "started_at": value, "created_at": 60,
        "last_active": value, "updated_at": 60,
        "messages": [{"role": "user", "content": "body", "created_at": value, "timestamp": 60}],
    }
    text = render_session_markdown(session)
    expected = "1970-01-01T00:00:00Z" if value == 0 else "1970-01-01T00:01:00Z"
    frontmatter = YAML(typ="safe").load(text.split("---", 2)[1])
    assert frontmatter["created_at"] == frontmatter["updated_at"] == expected
    assert f"### User — {expected}" in text
    missing = generate_multi_session_html_export([{"id": "missing", "messages": [{"role": "user", "content": "body"}]}])
    assert "<strong>Started:</strong> N/A" in missing
    assert '<div class="timestamp">N/A</div>' in missing
