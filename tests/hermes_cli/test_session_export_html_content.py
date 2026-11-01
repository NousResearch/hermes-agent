"""HTML exports preserve structured content accepted by the session store."""

import html
import json
import re
from argparse import Namespace

import pytest

from hermes_cli.sessions_cmd import cmd_sessions
from hermes_state import SessionDB


def _export_content(tmp_path, monkeypatch, content):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = SessionDB()
    try:
        db.create_session("structured-export", source="cli")
        db.append_message("structured-export", "user", content)
        assert db.export_session("structured-export")["messages"][0]["content"] == content
    finally:
        db.close()
    path = tmp_path / "export.html"
    args = Namespace(
        sessions_action="export", format="html", output=str(path),
        session_id="structured-export", redact=False, dry_run=False, only=None,
    )
    cmd_sessions(args)
    document = path.read_text(encoding="utf-8-sig")
    bodies = re.findall(r'<div class="content">(.*?)</div>', document, re.DOTALL)
    return [html.unescape(body) for body in bodies], document


@pytest.mark.parametrize("part", [
    {"file": "报告.md", "matches": 0, "details": {"found": False}},
    {"type": "input_text", "text": "Read <report> & its details"},
    {"type": "output_text", "text": "The result is <empty> & valid"},
])
def test_structured_parts_survive_public_html_export(tmp_path, monkeypatch, part):
    bodies, document = _export_content(tmp_path, monkeypatch, [part])
    assert len(bodies) == 1
    if isinstance(part.get("text"), str):
        assert bodies[0] == part["text"]
        assert "&lt;" in document and "&amp;" in document
    else:
        assert json.loads(bodies[0]) == part


def test_existing_multimodal_text_and_image_placeholder_remain(tmp_path, monkeypatch):
    content = [
        {"type": "text", "text": "Before <image> & after"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,cGF5bG9hZA=="}},
        "plain tail",
    ]
    bodies, document = _export_content(tmp_path, monkeypatch, content)
    assert bodies == ["Before <image> & after\n[Image Attachment]\nplain tail"]
    assert "&lt;image&gt; &amp;" in document
    assert "cGF5bG9hZA==" not in document
