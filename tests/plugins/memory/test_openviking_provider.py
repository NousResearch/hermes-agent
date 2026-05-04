import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.openviking import OpenVikingMemoryProvider, _VikingClient


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


def test_tool_add_resource_uploads_existing_local_file(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("# Local resource\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.upload_temp_file.return_value = "upload_sample.md"
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/sample"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": str(sample),
        "reason": "local test",
        "wait": True,
    }))

    provider._client.upload_temp_file.assert_called_once_with(sample)
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "reason": "local test",
        "wait": True,
        "source_name": "sample.md",
        "temp_file_id": "upload_sample.md",
    })
    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/sample"


def test_tool_add_resource_uploads_file_uri(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("# Local resource\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.upload_temp_file.return_value = "upload_sample.md"
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/sample"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": sample.as_uri(),
        "reason": "file uri test",
    }))

    provider._client.upload_temp_file.assert_called_once_with(sample)
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "reason": "file uri test",
        "source_name": "sample.md",
        "temp_file_id": "upload_sample.md",
    })
    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/sample"


def test_tool_add_resource_uploads_existing_local_directory_and_cleans_zip(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    nested = docs / "nested"
    nested.mkdir()
    (nested / "api.md").write_text("# API\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    uploaded_paths = []
    provider._client.upload_temp_file.side_effect = (
        lambda path: uploaded_paths.append(path) or "upload_docs.zip"
    )
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/docs"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": str(docs),
        "reason": "directory test",
        "wait": True,
    }))

    assert uploaded_paths
    assert uploaded_paths[0].suffix == ".zip"
    assert not uploaded_paths[0].exists()
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "reason": "directory test",
        "wait": True,
        "source_name": "docs",
        "temp_file_id": "upload_docs.zip",
    })
    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/docs"


def test_tool_add_resource_cleans_local_directory_zip_when_add_fails(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    uploaded_paths = []
    provider._client.upload_temp_file.side_effect = (
        lambda path: uploaded_paths.append(path) or "upload_docs.zip"
    )
    provider._client.post.side_effect = RuntimeError("add failed")

    with pytest.raises(RuntimeError, match="add failed"):
        provider._tool_add_resource({"url": str(docs)})

    assert uploaded_paths
    assert not uploaded_paths[0].exists()


def test_tool_add_resource_cleans_local_directory_zip_when_upload_fails(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    uploaded_paths = []

    def fail_upload(path):
        uploaded_paths.append(path)
        raise RuntimeError("upload failed")

    provider._client.upload_temp_file.side_effect = fail_upload

    with pytest.raises(RuntimeError, match="upload failed"):
        provider._tool_add_resource({"url": str(docs)})

    assert uploaded_paths
    assert not uploaded_paths[0].exists()
    provider._client.post.assert_not_called()


def test_tool_add_resource_rejects_missing_local_path(tmp_path):
    missing = tmp_path / "missing.md"
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()

    result = json.loads(provider._tool_add_resource({"url": str(missing)}))

    assert result["error"] == f"Local resource path does not exist: {missing}"
    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_not_called()


def test_tool_add_resource_sends_remote_url_as_path():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/remote"},
    }

    provider._tool_add_resource({"url": "https://example.com/doc.md"})

    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "path": "https://example.com/doc.md",
    })


@pytest.mark.parametrize("url", [
    "git@github.com:org/repo.git",
    "git@ssh.dev.azure.com:v3/org/project/repo",
    "ssh://git@github.com/org/repo.git",
    "git://github.com/org/repo.git",
])
def test_tool_add_resource_sends_git_remote_sources_as_path(url):
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/repo"},
    }

    provider._tool_add_resource({"url": url})

    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "path": url,
    })


def test_viking_client_upload_temp_file_uses_multipart_identity_headers(tmp_path, monkeypatch):
    sample = tmp_path / "sample.md"
    sample.write_text("# Local resource\n", encoding="utf-8")
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="test-account",
        user="test-user",
        agent="test-agent",
    )
    captured_kwargs = {}

    def capture_httpx_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "ok", "result": {"temp_file_id": "upload_sample.md"}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(client._httpx, "post", capture_httpx_post)

    assert client.upload_temp_file(sample) == "upload_sample.md"

    assert "files" in captured_kwargs
    assert "json" not in captured_kwargs
    headers = captured_kwargs["headers"]
    assert headers["X-OpenViking-Account"] == "test-account"
    assert headers["X-OpenViking-User"] == "test-user"
    assert headers["X-OpenViking-Agent"] == "test-agent"
    assert headers["X-API-Key"] == "test-key"
    assert "Content-Type" not in headers


def test_viking_client_raises_structured_server_error():
    client = _VikingClient.__new__(_VikingClient)
    response = SimpleNamespace(
        status_code=403,
        text='{"status":"error"}',
        json=lambda: {
            "status": "error",
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "direct host filesystem paths are not allowed",
            },
        },
        raise_for_status=lambda: None,
    )

    with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
        client._parse_response(response)
