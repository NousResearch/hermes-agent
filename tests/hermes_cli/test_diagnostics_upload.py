"""Tests for ``hermes_cli.diagnostics_upload`` — the Nous-S3 upload client.

All network I/O is mocked at ``urllib.request.urlopen``; no real requests
are made.
"""

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest


def _resp(*, status=200, body=b""):
    """Build a context-manager mock mimicking ``urllib.request.urlopen``."""
    m = MagicMock()
    m.status = status
    m.getcode.return_value = status
    m.read.return_value = body
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    return m


# ---------------------------------------------------------------------------
# request_upload_url
# ---------------------------------------------------------------------------

class TestRequestUploadUrl:
    def test_happy_path_posts_json_and_returns_dict(self):
        from hermes_cli.diagnostics_upload import request_upload_url

        payload = {
            "success": True,
            "id": "abc-123",
            "uploadUrl": "https://bucket.s3.amazonaws.com/uploads/abc-123.json.gz?sig",
            "viewUrl": "https://support.example.com/diagnostics/abc-123",
            "uploadExpiresInSeconds": 900,
        }
        resp = _resp(status=200, body=json.dumps(payload).encode())

        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            return_value=resp,
        ) as urlopen:
            result = request_upload_url(content_type="application/gzip", size_bytes=512)

        assert result == payload

        # The request object passed to urlopen carries our JSON body + headers.
        req = urlopen.call_args[0][0]
        assert req.method == "POST"
        assert req.full_url.endswith("/api/diagnostics/upload-url")
        sent = json.loads(req.data.decode())
        assert sent["contentType"] == "application/gzip"
        assert sent["sizeBytes"] == 512
        # urllib lower-cases header keys.
        assert req.headers["Content-type"] == "application/json"

    def test_non_2xx_raises(self):
        from hermes_cli.diagnostics_upload import request_upload_url

        resp = _resp(status=500, body=b"boom")
        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            return_value=resp,
        ):
            with pytest.raises(RuntimeError):
                request_upload_url()

    def test_missing_upload_url_raises(self):
        from hermes_cli.diagnostics_upload import request_upload_url

        resp = _resp(status=200, body=json.dumps({"id": "x"}).encode())
        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            return_value=resp,
        ):
            with pytest.raises(RuntimeError):
                request_upload_url()

    def test_non_json_raises(self):
        from hermes_cli.diagnostics_upload import request_upload_url

        resp = _resp(status=200, body=b"<html>not json</html>")
        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            return_value=resp,
        ):
            with pytest.raises(RuntimeError):
                request_upload_url()

    def test_base_url_env_override(self, monkeypatch):
        # NAS_BASE is read at import time; re-import the module under the
        # patched env to confirm the override is honoured.
        import importlib

        monkeypatch.setenv("HERMES_DIAGNOSTICS_BASE_URL", "https://staging.example.com")
        import hermes_cli.diagnostics_upload as mod

        mod = importlib.reload(mod)
        try:
            assert mod.NAS_BASE == "https://staging.example.com"
            resp = _resp(
                status=200,
                body=json.dumps({"uploadUrl": "u", "id": "i", "viewUrl": "v"}).encode(),
            )
            with patch(
                "hermes_cli.diagnostics_upload.urllib.request.urlopen",
                return_value=resp,
            ) as urlopen:
                mod.request_upload_url()
            req = urlopen.call_args[0][0]
            assert req.full_url == "https://staging.example.com/api/diagnostics/upload-url"
        finally:
            monkeypatch.delenv("HERMES_DIAGNOSTICS_BASE_URL", raising=False)
            importlib.reload(mod)


# ---------------------------------------------------------------------------
# put_bundle
# ---------------------------------------------------------------------------

class TestPutBundle:
    def test_put_sends_exact_body_and_content_type(self):
        from hermes_cli.diagnostics_upload import put_bundle

        data = b"\x1f\x8b\x08gzipped-bytes"
        resp = _resp(status=200, body=b"")

        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            return_value=resp,
        ) as urlopen:
            put_bundle("https://bucket.s3.amazonaws.com/uploads/x.json.gz?sig", data)

        req = urlopen.call_args[0][0]
        assert req.method == "PUT"
        # PUT body must be the bundle bytes, unchanged.
        assert req.data == data
        assert req.headers["Content-type"] == "application/gzip"

    def test_custom_content_type(self):
        from hermes_cli.diagnostics_upload import put_bundle

        resp = _resp(status=204, body=b"")
        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            return_value=resp,
        ) as urlopen:
            put_bundle("https://u", b"data", content_type="application/json")
        req = urlopen.call_args[0][0]
        assert req.headers["Content-type"] == "application/json"

    def test_non_2xx_raises(self):
        from hermes_cli.diagnostics_upload import put_bundle

        resp = _resp(status=403, body=b"AccessDenied")
        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            return_value=resp,
        ):
            with pytest.raises(RuntimeError):
                put_bundle("https://u", b"data")

    def test_http_error_propagates(self):
        from hermes_cli.diagnostics_upload import put_bundle

        err = urllib.error.HTTPError("https://u", 500, "err", {}, io.BytesIO(b""))
        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            side_effect=err,
        ):
            with pytest.raises(urllib.error.HTTPError):
                put_bundle("https://u", b"data")


# ---------------------------------------------------------------------------
# confirm_upload
# ---------------------------------------------------------------------------

class TestConfirmUpload:
    def test_confirm_posts(self):
        from hermes_cli.diagnostics_upload import confirm_upload

        resp = _resp(status=200, body=b"")
        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            return_value=resp,
        ) as urlopen:
            confirm_upload("abc-123", 1024)
        req = urlopen.call_args[0][0]
        assert req.method == "POST"
        assert req.full_url.endswith("/api/diagnostics/confirm")
        sent = json.loads(req.data.decode())
        assert sent == {"id": "abc-123", "sizeBytes": 1024}

    def test_confirm_failure_is_swallowed(self):
        from hermes_cli.diagnostics_upload import confirm_upload

        with patch(
            "hermes_cli.diagnostics_upload.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            # Must NOT raise.
            confirm_upload("abc-123", 1024)


# ---------------------------------------------------------------------------
# share_to_nous (orchestration)
# ---------------------------------------------------------------------------

class TestShareToNous:
    def test_orchestrates_request_put_confirm(self):
        from hermes_cli import diagnostics_upload as mod

        info = {
            "id": "id-9",
            "uploadUrl": "https://bucket/uploads/id-9.json.gz?sig",
            "viewUrl": "https://support/diagnostics/id-9",
            "expiresAt": "2026-06-20T00:00:00Z",
        }
        blob = b"gzipped-bundle"

        with patch.object(mod, "request_upload_url", return_value=info) as req, \
             patch.object(mod, "put_bundle") as put, \
             patch.object(mod, "confirm_upload") as confirm:
            result = mod.share_to_nous(blob)

        assert result == info
        req.assert_called_once()
        # request was told the real byte size
        assert req.call_args.kwargs["size_bytes"] == len(blob)
        # PUT got the signed URL + the exact blob
        put.assert_called_once_with(
            info["uploadUrl"], blob, content_type="application/gzip"
        )
        confirm.assert_called_once_with("id-9", len(blob))

    def test_put_failure_propagates(self):
        from hermes_cli import diagnostics_upload as mod

        info = {"id": "id-9", "uploadUrl": "https://u", "viewUrl": "v"}
        with patch.object(mod, "request_upload_url", return_value=info), \
             patch.object(mod, "put_bundle", side_effect=RuntimeError("PUT failed")), \
             patch.object(mod, "confirm_upload") as confirm:
            with pytest.raises(RuntimeError):
                mod.share_to_nous(b"data")
        # confirm never runs if PUT failed
        confirm.assert_not_called()

    def test_confirm_skipped_without_id(self):
        from hermes_cli import diagnostics_upload as mod

        info = {"uploadUrl": "https://u", "viewUrl": "v"}  # no id
        with patch.object(mod, "request_upload_url", return_value=info), \
             patch.object(mod, "put_bundle"), \
             patch.object(mod, "confirm_upload") as confirm:
            mod.share_to_nous(b"data")
        confirm.assert_not_called()
