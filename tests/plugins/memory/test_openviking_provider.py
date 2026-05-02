import json
from unittest.mock import MagicMock

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


def test_tool_delete_file_returns_ok():
    """viking_delete on a file URI should call DELETE /api/v1/fs and return ok."""
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.delete.return_value = {
        "result": {"uri": "viking://user/test.md"},
    }

    result = json.loads(provider._tool_delete({
        "uri": "viking://user/test.md",
    }))

    assert result["status"] == "ok"
    assert result["deleted"] == "viking://user/test.md"

    provider._client.delete.assert_called_once_with(
        "/api/v1/fs",
        params={"uri": "viking://user/test.md"},
    )


def test_tool_delete_directory_recursive():
    """viking_delete with recursive=true should pass the flag."""
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.delete.return_value = {
        "result": {"uri": "viking://user/dir"},
    }

    result = json.loads(provider._tool_delete({
        "uri": "viking://user/dir",
        "recursive": True,
    }))

    assert result["status"] == "ok"
    assert result["deleted"] == "viking://user/dir"

    provider._client.delete.assert_called_once_with(
        "/api/v1/fs",
        params={"uri": "viking://user/dir", "recursive": "true"},
    )


def test_tool_delete_missing_uri():
    """viking_delete without uri should return an error."""
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()

    result = json.loads(provider._tool_delete({}))

    assert "error" in result
    assert "uri is required" in result["error"]
    provider._client.delete.assert_not_called()


def test_tool_delete_server_error():
    """viking_delete should surface server errors gracefully."""
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.delete.side_effect = RuntimeError("500 Internal Server Error")

    result = json.loads(provider._tool_delete({
        "uri": "viking://user/test.md",
    }))

    assert "error" in result
    assert "500 Internal Server Error" in result["error"]
