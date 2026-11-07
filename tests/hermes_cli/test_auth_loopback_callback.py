"""Unit tests for OAuth loopback callback handler and HTML rendering."""

from __future__ import annotations

import urllib.request
from http.server import HTTPServer
import threading

import pytest

from hermes_cli.auth_device_flow import (
    _make_loopback_callback_handler,
    _render_loopback_callback_html,
)


def test_render_loopback_callback_html_success():
    html_bytes = _render_loopback_callback_html("OpenAI Codex", "received")
    text = html_bytes.decode("utf-8")

    assert "OpenAI Codex authorization received." in text
    assert "Hermes Agent" in text
    assert "Successfully authenticated with <strong>OpenAI Codex</strong>." in text
    assert "You can close this tab and return to your terminal." in text
    assert "Close This Tab" in text
    assert "window.close()" in text
    assert "/portal-art.svg" in text


def test_render_loopback_callback_html_failure():
    html_bytes = _render_loopback_callback_html(
        "OpenAI Codex",
        "failed",
        error="access_denied",
        error_description="The user cancelled the login request.",
    )
    text = html_bytes.decode("utf-8")

    assert "OpenAI Codex authorization failed." in text
    assert "Failed to authenticate with <strong>OpenAI Codex</strong>." in text
    assert "The user cancelled the login request." in text
    assert "Please return to your terminal and try signing in again." in text


def test_render_loopback_callback_html_escaping():
    html_bytes = _render_loopback_callback_html(
        '<script>alert("xss")</script>',
        "failed",
        error="<b>danger</b>",
        error_description="<img src=x onerror=alert(1)>",
    )
    text = html_bytes.decode("utf-8")

    assert '<script>alert("xss")</script>' not in text
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in text
    assert "<img src=x" not in text
    assert "&lt;img src=x onerror=alert(1)&gt;" in text


def test_loopback_callback_server_e2e():
    handler_cls, result = _make_loopback_callback_handler(
        "/auth/callback", display_name="Test Provider"
    )
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/auth/callback?code=mock_auth_code&state=mock_state"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        assert "text/html; charset=utf-8" in resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8")
        assert "Test Provider authorization received." in body
        assert "Close This Tab" in body

    thread.join(timeout=2)
    server.server_close()

    assert result["code"] == "mock_auth_code"
    assert result["state"] == "mock_state"
    assert result["error"] is None


def test_loopback_callback_server_serves_portal_art_svg():
    handler_cls, _ = _make_loopback_callback_handler(
        "/auth/callback", display_name="Test Provider"
    )
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/portal-art.svg"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        assert "image/svg+xml" in resp.headers.get("Content-Type", "")
        content = resp.read()
        assert b"<svg" in content

    thread.join(timeout=2)
    server.server_close()


def test_loopback_callback_server_serves_favicon_png():
    handler_cls, _ = _make_loopback_callback_handler(
        "/auth/callback", display_name="Test Provider"
    )
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/favicon.png"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        assert "image/png" in resp.headers.get("Content-Type", "")
        content = resp.read()
        assert len(content) > 100

    thread.join(timeout=2)
    server.server_close()
