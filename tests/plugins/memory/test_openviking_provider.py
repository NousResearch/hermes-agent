import json
from unittest.mock import MagicMock, mock_open, patch

from plugins.memory.openviking import OpenVikingMemoryProvider


def test_tool_search_sorts_by_raw_score_across_buckets():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [
                {"uri": "viking://memories/1", "score": 0.9003, "abstract": "memory result"},
            ],
            "resources": [
                {"uri": "viking://resources/1", "score": 0.9004, "abstract": "resource result"},
            ],
            "skills": [
                {"uri": "viking://skills/1", "score": 0.8999, "abstract": "skill result"},
            ],
            "total": 3,
        }
    }

    result = json.loads(provider._tool_search({"query": "ranking"}))

    assert [entry["uri"] for entry in result["results"]] == [
        "viking://resources/1",
        "viking://memories/1",
        "viking://skills/1",
    ]
    assert [entry["score"] for entry in result["results"]] == [0.9, 0.9, 0.9]
    assert result["total"] == 3


def test_tool_search_sorts_missing_raw_score_after_negative_scores():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [
                {"uri": "viking://memories/missing", "abstract": "missing score"},
            ],
            "resources": [
                {"uri": "viking://resources/negative", "score": -0.25, "abstract": "negative score"},
            ],
            "skills": [
                {"uri": "viking://skills/positive", "score": 0.1, "abstract": "positive score"},
            ],
            "total": 3,
        }
    }

    result = json.loads(provider._tool_search({"query": "ranking"}))

    assert [entry["uri"] for entry in result["results"]] == [
        "viking://skills/positive",
        "viking://memories/missing",
        "viking://resources/negative",
    ]
    assert [entry["score"] for entry in result["results"]] == [0.1, 0.0, -0.25]
    assert result["total"] == 3


def test_tool_add_resource_remote_url_passes_path_directly():
    """Remote URLs should pass 'path' directly to /api/v1/resources (unchanged)."""
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client._url.return_value = "http://127.0.0.1:1933/api/v1/resources"
    provider._client.post.return_value = {
        "result": {"root_uri": "viking://resources/docs/my-doc.md"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": "https://example.com/doc.md",
        "reason": "test reason",
    }))

    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/docs/my-doc.md"

    # Should call post with 'path' field, NOT temp_file_id
    provider._client.post.assert_called_once_with(
        "/api/v1/resources",
        {"path": "https://example.com/doc.md", "reason": "test reason"},
    )
    # _httpx.post should NOT be called for remote URLs
    provider._client._httpx.post.assert_not_called()


def test_tool_add_resource_local_file_uploads_via_temp_upload():
    """Local file paths should upload via temp_upload, then pass temp_file_id."""
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client._headers.return_value = {
        "Content-Type": "application/json",
        "X-OpenViking-Account": "default",
        "X-OpenViking-User": "default",
        "X-OpenViking-Agent": "hermes",
    }
    provider._client._url.return_value = "http://127.0.0.1:1933/api/v1/resources/temp_upload"

    # Mock httpx temp_upload response
    mock_httpx_resp = MagicMock()
    mock_httpx_resp.json.return_value = {
        "result": {"temp_file_id": "upload_abc123.md"},
    }
    provider._client._httpx.post.return_value = mock_httpx_resp

    # Mock final resource registration response
    provider._client.post.return_value = {
        "result": {"root_uri": "viking://resources/docs/my-local.md"},
    }

    with patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=b"file content")):
        result = json.loads(provider._tool_add_resource({
            "url": "/home/user/notes.md",
            "reason": "local test",
        }))

    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/docs/my-local.md"

    # Should have called temp_upload (multipart upload via _httpx.post)
    provider._client._httpx.post.assert_called_once()
    call_args = provider._client._httpx.post.call_args
    assert call_args[0] == ("http://127.0.0.1:1933/api/v1/resources/temp_upload",)
    assert "files" in call_args[1]
    assert call_args[1]["files"]["file"][0] == "notes.md"

    # Final post should use temp_file_id, NOT path
    provider._client.post.assert_called_once_with(
        "/api/v1/resources",
        {"temp_file_id": "upload_abc123.md", "reason": "local test"},
    )


def test_tool_add_resource_local_file_not_found():
    """Local file that doesn't exist should return an error."""
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()

    with patch("pathlib.Path.exists", return_value=False):
        result = json.loads(provider._tool_add_resource({
            "url": "/nonexistent/file.md",
        }))

    assert "error" in result
    assert "not found" in result["error"]
    # _httpx.post should NOT be called for missing files
    provider._client._httpx.post.assert_not_called()
