"""Regression tests for the dashboard attachment upload route.

[CN-fork] P-002 — ``POST /api/upload`` is a downstream-only endpoint the
desktop composer depends on for pasting/dropping images. It has been silently
dropped by an upstream sync before (the handler vanished while its helpers
``_parse_multipart_form`` / ``_safe_upload_filename`` / ``_unique_upload_path``
stayed behind as dead code), which made the desktop GUI's Ctrl+V image paste
fail with ``HTTP 405 Method Not Allowed`` — the SPA catch-all answers the path
on GET only, so a POST falls through to 405. See Desktop issue #306.

These tests fail loudly if the route disappears again.
"""

from starlette.testclient import TestClient

from hermes_cli import web_server


def _client_with_app_state():
    prev_auth_required = getattr(web_server.app.state, "auth_required", None)
    prev_bound_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client, prev_auth_required, prev_bound_host


def _restore_app_state(prev_auth_required, prev_bound_host):
    if prev_auth_required is None:
        delattr(web_server.app.state, "auth_required")
    else:
        web_server.app.state.auth_required = prev_auth_required
    if prev_bound_host is None:
        if hasattr(web_server.app.state, "bound_host"):
            delattr(web_server.app.state, "bound_host")
    else:
        web_server.app.state.bound_host = prev_bound_host


def _upload_client(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))

    client, prev_auth_required, prev_bound_host = _client_with_app_state()
    return client, home, prev_auth_required, prev_bound_host


def test_post_api_upload_route_exists_and_accepts_post(monkeypatch, tmp_path):
    """The core regression: POST /api/upload must be routed to the handler,
    never fall through to the SPA catch-all and answer 405 (issue #306)."""
    client, _home, prev_auth, prev_host = _upload_client(monkeypatch, tmp_path)
    try:
        resp = client.post(
            "/api/upload",
            data={"session_id": "sess-1"},
            files={"file": ("shot.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        )
        assert resp.status_code != 405, "POST /api/upload regressed to 405 — the P-002 route was dropped"
        assert resp.status_code == 200, resp.text
    finally:
        client.close()
        _restore_app_state(prev_auth, prev_host)


def test_upload_writes_file_and_returns_expected_shape(monkeypatch, tmp_path):
    """The desktop's AttachmentUploadResult schema requires filename/path/size
    and reads back the absolute path; assert the response shape stays stable."""
    client, home, prev_auth, prev_host = _upload_client(monkeypatch, tmp_path)
    try:
        payload = b"\x89PNG\r\n\x1a\nclipboard-bytes"
        resp = client.post(
            "/api/upload",
            data={"session_id": "abc_123"},
            files={"file": ("clipboard.png", payload, "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["filename"] == "clipboard.png"
        assert body["size"] == len(payload)
        assert body["mime_type"] == "image/png"

        written = home / "uploads" / "abc_123" / "clipboard.png"
        assert written.read_bytes() == payload
        assert body["path"] == str(written)
    finally:
        client.close()
        _restore_app_state(prev_auth, prev_host)


def test_upload_rejects_invalid_session_id(monkeypatch, tmp_path):
    client, _home, prev_auth, prev_host = _upload_client(monkeypatch, tmp_path)
    try:
        resp = client.post(
            "/api/upload",
            data={"session_id": "../escape"},
            files={"file": ("x.png", b"data", "image/png")},
        )
        assert resp.status_code == 400
    finally:
        client.close()
        _restore_app_state(prev_auth, prev_host)
