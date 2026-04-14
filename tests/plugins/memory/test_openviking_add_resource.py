"""Tests for OpenViking memory provider - add_resource functionality.

Tests the fix for local file uploads and remote URL handling in viking_add_resource.
"""

import os
import io
import json
import pytest
from unittest.mock import MagicMock, patch

from plugins.memory.openviking import OpenVikingMemoryProvider, _VikingClient


class FakeOpenVikingClient:
    """Fake OpenViking client that simulates API responses."""

    def __init__(self):
        self.post_calls = []
        self.temp_uploads = {}

    def post(self, endpoint, json=None, files=None):
        """Mock POST request handler."""
        self.post_calls.append({
            "endpoint": endpoint,
            "json": json,
            "files": files is not None
        })

        if endpoint == "/api/v1/resources/temp_upload" and files:
            # Simulate temp upload
            temp_id = f"upload_test_{len(self.temp_uploads)}.md"
            self.temp_uploads[temp_id] = files
            return {
                "status": "ok",
                "result": {
                    "temp_file_id": temp_id
                }
            }

        elif endpoint == "/api/v1/resources":
            # Simulate resource addition
            if json and ("path" in json or "temp_file_id" in json):
                return {
                    "status": "ok",
                    "result": {
                        "status": "success",
                        "root_uri": f"viking://resources/test_{len(self.post_calls)}",
                        "errors": []
                    }
                }
            else:
                return {
                    "status": "error",
                    "result": {
                        "errors": ["Missing path or temp_file_id"]
                    }
                }

        return {"status": "error", "result": {}}


class TestOpenVikingAddResource:
    """Test suite for viking_add_resource tool."""

    def _make_provider(self):
        """Create a test provider with fake client."""
        provider = OpenVikingMemoryProvider()
        provider._client = FakeOpenVikingClient()
        provider._endpoint = "http://test:1933"
        provider._api_key = "test-key"
        return provider

    # ---------------------------------------------------------------------------
    # Test 1: Remote URL handling
    # ---------------------------------------------------------------------------

    def test_add_remote_url_uses_path_field(self):
        """Remote URLs should be sent with 'path' field."""
        provider = self._make_provider()

        result = provider._tool_add_resource({
            "path": "https://example.com/document.md",
            "reason": "Test remote URL"
        })

        result_json = json.loads(result)
        assert result_json["status"] == "added"
        assert result_json["root_uri"].startswith("viking://resources/")

        # Verify the client received 'path' not 'url'
        last_call = provider._client.post_calls[-1]
        assert "path" in last_call["json"]
        assert last_call["json"]["path"] == "https://example.com/document.md"
        assert last_call["json"]["reason"] == "Test remote URL"

    # ---------------------------------------------------------------------------
    # Test 2: Local file upload via temp_upload
    # ---------------------------------------------------------------------------

    def test_add_local_file_uploads_first(self, tmp_path):
        """Local files should be uploaded via temp_upload endpoint first."""
        # Create a temporary test file
        test_file = tmp_path / "test_document.md"
        test_file.write_text("# Test Document\n\nThis is test content.")

        provider = self._make_provider()

        result = provider._tool_add_resource({
            "path": str(test_file),
            "reason": "Test local file upload"
        })

        result_json = json.loads(result)
        assert result_json["status"] == "added"
        assert result_json["root_uri"].startswith("viking://resources/")

        # Verify temp_upload was called first
        assert len(provider._client.post_calls) == 2
        assert provider._client.post_calls[0]["endpoint"] == "/api/v1/resources/temp_upload"
        assert provider._client.post_calls[0]["files"] is True

        # Verify add_resource was called with temp_file_id
        assert provider._client.post_calls[1]["endpoint"] == "/api/v1/resources"
        assert "temp_file_id" in provider._client.post_calls[1]["json"]
        assert provider._client.post_calls[1]["json"]["reason"] == "Test local file upload"

    # ---------------------------------------------------------------------------
    # Test 3: Backward compatibility with 'url' field
    # ---------------------------------------------------------------------------

    def test_add_resource_supports_url_field_for_backward_compat(self):
        """Should support 'url' field for backward compatibility."""
        provider = self._make_provider()

        result = provider._tool_add_resource({
            "url": "https://example.com/old-format.md",
            "reason": "Test backward compatibility"
        })

        result_json = json.loads(result)
        assert result_json["status"] == "added"

        # The tool should map 'url' to 'path' internally
        last_call = provider._client.post_calls[-1]
        assert "path" in last_call["json"]
        assert last_call["json"]["path"] == "https://example.com/old-format.md"

    # ---------------------------------------------------------------------------
    # Test 4: Error handling - missing path/url
    # ---------------------------------------------------------------------------

    def test_add_resource_requires_path_or_url(self):
        """Should return error when neither path nor url is provided."""
        provider = self._make_provider()

        result = provider._tool_add_resource({
            "reason": "Missing path test"
        })

        assert "error" in result.lower() or "required" in result.lower()

    # ---------------------------------------------------------------------------
    # Test 5: Support for wait parameter
    # ---------------------------------------------------------------------------

    def test_add_resource_supports_wait_parameter(self):
        """Should pass wait parameter to the API."""
        provider = self._make_provider()

        result = provider._tool_add_resource({
            "path": "https://example.com/doc.md",
            "reason": "Test wait param",
            "wait": True
        })

        last_call = provider._client.post_calls[-1]
        assert last_call["json"]["wait"] is True

    # ---------------------------------------------------------------------------
    # Test 7: Support for optional parameters
    # ---------------------------------------------------------------------------

    def test_add_resource_passes_reason_and_wait(self):
        """Should pass reason and wait parameters."""
        provider = self._make_provider()

        result = provider._tool_add_resource({
            "path": "https://example.com/doc.md",
            "reason": "Custom reason",
            "wait": True
        })

        last_call = provider._client.post_calls[-1]
        assert last_call["json"]["reason"] == "Custom reason"
        assert last_call["json"]["wait"] is True


class TestVikingClientFilesKwarg:
    """Integration tests for _VikingClient.post() files= kwarg handling.

    These tests exercise the real _VikingClient code path with a mocked httpx layer,
    verifying the json/files conflict fix from the PR review.
    """

    def test_viking_client_post_with_files_omits_json(self):
        """_VikingClient.post(files=) must not pass json= to httpx.

        httpx raises ValueError when both json= and files= are passed simultaneously
        ('Multipart payloads cannot be combined with json'). The fix routes around
        this by skipping json= when files= is present.
        """
        client = _VikingClient("https://example.com", api_key="test-key")

        captured_kwargs = {}
        def capture_httpx_post(url, **kwargs):
            captured_kwargs.update(kwargs)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"temp_file_id": "test123"}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch.object(client._httpx, "post", side_effect=capture_httpx_post):
            result = client.post(
                "/api/v1/resources/temp_upload",
                files={"file": ("test.txt", io.BytesIO(b"hello"))}
            )

        # httpx.post was called with files= present
        assert "files" in captured_kwargs
        # json= must NOT be present -- that was the bug
        assert "json" not in captured_kwargs, (
            "json= should not be passed when files= is present; "
            "httpx raises ValueError: Multipart payloads cannot be combined with json"
        )
        assert result == {"temp_file_id": "test123"}

    def test_viking_client_post_with_files_removes_content_type_header(self):
        """When files= is present, Content-Type header should not be set.

        httpx sets multipart Content-Type automatically with the correct boundary.
        Manually setting it causes a mismatch.
        """
        client = _VikingClient("https://example.com", api_key="test-key")

        captured_headers = {}
        def capture_httpx_post(url, **kwargs):
            if "headers" in kwargs:
                captured_headers.update(kwargs["headers"])
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch.object(client._httpx, "post", side_effect=capture_httpx_post):
            client.post(
                "/api/v1/resources/temp_upload",
                files={"file": ("test.txt", io.BytesIO(b"hello"))}
            )

        # Content-Type should not be manually set for multipart
        assert "Content-Type" not in captured_headers, (
            "Content-Type header should not be set manually for multipart uploads; "
            "httpx sets it automatically with the correct boundary"
        )

    def test_viking_client_post_without_files_uses_json(self):
        """_VikingClient.post() without files= should still pass json= normally."""
        client = _VikingClient("https://example.com", api_key="test-key")

        captured_kwargs = {}
        def capture_httpx_post(url, **kwargs):
            captured_kwargs.update(kwargs)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "ok"}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch.object(client._httpx, "post", side_effect=capture_httpx_post):
            result = client.post(
                "/api/v1/resources",
                payload={"path": "https://example.com/doc.md", "reason": "test"}
            )

        # json= should be present (not files=)
        assert "json" in captured_kwargs
        assert "files" not in captured_kwargs
        assert captured_kwargs["json"]["path"] == "https://example.com/doc.md"
