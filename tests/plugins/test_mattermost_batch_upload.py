"""Standalone Mattermost deliveries upload attachments in one multipart request."""
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from plugins.platforms.mattermost.adapter import _standalone_send


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("outcome", ["ok", "empty", "http_error", "network_error"])
async def test_standalone_batches_uploads(tmp_path, monkeypatch, count, outcome):
    monkeypatch.setenv("MATTERMOST_URL", "https://mattermost.invalid")
    monkeypatch.setenv("MATTERMOST_TOKEN", "fixture-token")
    monkeypatch.setenv("MATTERMOST_PROXY", "http://proxy.invalid:8080")
    paths = []
    for i in range(count):
        path = tmp_path / f"file-{i}.txt"
        path.write_bytes(f"content-{i}".encode())
        paths.append(str(path))
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        response = MagicMock(status=201)
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        if url.endswith("/files"):
            # Exercise aiohttp's actual multipart construction, not a fake form.
            payload = kwargs["data"]()
            assert payload.size > 0
            if outcome == "network_error":
                raise aiohttp.ClientError("fixture disconnect")
            if outcome == "http_error":
                response.status = 413
                response.text = AsyncMock(return_value="too large")
            response.json = AsyncMock(return_value={"file_infos": [] if outcome == "empty" else [{"id": f"id-{i}"} for i in range(count)]})
        else:
            response.json = AsyncMock(return_value={"id": "post-id"})
        return response

    session.post.side_effect = post
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: session)
    result = await _standalone_send(None, "channel", "caption", thread_id="thread", media_files=paths)
    if count and outcome in {"http_error", "network_error"}:
        assert not result.get("success"), result
        assert len(calls) == 1  # no post or ambiguous upload retry
        return
    assert result["success"], result
    uploads = [call for call in calls if call[0].endswith("/files")]
    assert len(uploads) == (1 if count else 0)
    payload = calls[-1][1]["json"]
    assert payload["channel_id"] == "channel"
    assert payload["message"] == "caption"
    assert payload["root_id"] == "thread"
    expected_ids = [] if outcome == "empty" else [f"id-{i}" for i in range(count)]
    assert payload.get("file_ids", []) == expected_ids
    assert all(kwargs["proxy"] == "http://proxy.invalid:8080" for _, kwargs in calls)
