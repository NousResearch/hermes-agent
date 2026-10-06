"""Tests for gmail_get MIME extraction (body_text/body_html/attachments)."""

from __future__ import annotations

import base64
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)


@pytest.fixture(scope="module")
def api():
    spec = importlib.util.spec_from_file_location("google_api_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode()


def _nested_message() -> dict:
    """multipart/mixed wrapping multipart/alternative, plus a PDF attachment."""
    return {
        "id": "m1",
        "threadId": "t1",
        "labelIds": ["INBOX"],
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [
                {"name": "From", "value": "a@example.com"},
                {"name": "To", "value": "b@example.com"},
                {"name": "Subject", "value": "Hi"},
                {"name": "Date", "value": "Mon, 1 Jan 2026 10:00:00 +0000"},
            ],
            "body": {},
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "filename": "",
                    "body": {},
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": _b64("plain body")}},
                        {"mimeType": "text/html", "body": {"data": _b64("<p>html body</p>")}},
                    ],
                },
                {
                    "mimeType": "application/pdf",
                    "filename": "doc.pdf",
                    "body": {"attachmentId": "att1", "size": 1234},
                },
            ],
        },
    }


def test_extract_bodies_nested(api):
    bodies = api._extract_bodies(_nested_message())
    assert bodies == {"text": "plain body", "html": "<p>html body</p>"}


def test_extract_message_body_prefers_text_then_html(api):
    assert api._extract_message_body(_nested_message()) == "plain body"
    html_only = {"payload": {"mimeType": "text/html", "body": {"data": _b64("<b>x</b>")}}}
    assert api._extract_message_body(html_only) == "<b>x</b>"


def test_extract_attachments_skips_structural_parts(api):
    atts = api._extract_attachments(_nested_message())
    assert atts == [
        {
            "filename": "doc.pdf",
            "mime_type": "application/pdf",
            "attachment_id": "att1",
            "size": 1234,
        }
    ]


def test_extract_handles_empty_payload(api):
    assert api._extract_bodies({}) == {"text": "", "html": ""}
    assert api._extract_attachments({}) == []


@pytest.mark.parametrize("use_gws", [True, False])
def test_gmail_get_output_keys(api, monkeypatch, capsys, use_gws):
    msg = _nested_message()
    if use_gws:
        monkeypatch.setattr(api, "_gws_binary", lambda: "/usr/bin/gws")
        monkeypatch.setattr(api, "_run_gws", lambda *a, **k: msg)
    else:
        monkeypatch.setattr(api, "_gws_binary", lambda: None)

        class _Exec:
            def execute(self):
                return msg

        class _Svc:
            def users(self):
                return self

            def messages(self):
                return self

            def get(self, **kw):
                return _Exec()

        monkeypatch.setattr(api, "build_service", lambda *a, **k: _Svc())

    api.gmail_get(Namespace(message_id="m1"))
    out = json.loads(capsys.readouterr().out)
    assert out["body"] == "plain body"  # backwards-compatible key
    assert out["body_text"] == "plain body"
    assert out["body_html"] == "<p>html body</p>"
    assert [a["filename"] for a in out["attachments"]] == ["doc.pdf"]
    assert out["subject"] == "Hi"
