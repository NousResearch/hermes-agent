"""Live batch upload contracts exercised through send_multiple_images."""
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.mattermost.adapter import MattermostAdapter


def response(status=201, ids=()):
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value={"file_infos": [{"id": fid} for fid in ids]})
    resp.text = AsyncMock(return_value="rejected")
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def adapter(tmp_path, responses, count=3):
    obj = MattermostAdapter(PlatformConfig(token="fixture", extra={"url": "https://mm.example"}))
    obj._session = MagicMock()
    obj._session.post.side_effect = responses
    obj._post_message = AsyncMock(return_value={"id": "post"})
    images = []
    for index in range(count):
        path = tmp_path / f"image{index}.png"
        path.write_bytes(f"data{index}".encode())
        images.append(("file://" + str(path), f"caption{index}"))
    return obj, images


@pytest.mark.asyncio
async def test_multiple_images_use_one_upload(tmp_path):
    obj, images = adapter(tmp_path, [response(ids=["a", "b", "c"])] * 3)
    result = await obj.send_multiple_images("channel", images, metadata={"thread_id": "root"})
    assert result.success
    assert obj._session.post.call_count == 1
    obj._post_message.assert_awaited_once_with(
        "channel", "caption0\ncaption1\ncaption2", None, {"thread_id": "root"}, ["a", "b", "c"])


@pytest.mark.asyncio
async def test_rejected_file_does_not_drop_good_files(tmp_path):
    # Server rejects any form containing the middle file. Both the old single
    # path and a new batch-then-single path must retain the good attachments.
    obj, images = adapter(tmp_path, [])
    def upload(url, **kwargs):
        fields = kwargs["data"]._fields
        names = [field[0].get("filename") for field in fields if field[0]["name"] == "files"]
        return response(413) if "image1.png" in names else response(ids=[name for name in names])
    obj._session.post.side_effect = upload
    result = await obj.send_multiple_images("channel", images)
    assert result.success
    assert obj._post_message.await_count == 1
    assert obj._post_message.await_args.args[-1] == ["image0.png", "image2.png"]


@pytest.mark.asyncio
async def test_empty_upload_result_does_not_create_post(tmp_path):
    obj, images = adapter(tmp_path, [response()] * 3)
    result = await obj.send_multiple_images("channel", images)
    assert not result.success
    obj._post_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_file_uses_existing_upload_owner(tmp_path):
    obj, images = adapter(tmp_path, [], count=1)
    obj._upload_file = AsyncMock(return_value="single")
    result = await obj.send_multiple_images("channel", images)
    assert result.success
    obj._upload_file.assert_awaited_once_with("channel", b"data0", "image0.png", "image/png")
    obj._session.post.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_cap_and_proxy_are_preserved(tmp_path):
    obj, images = adapter(tmp_path, [response(ids=["a", "b", "c", "d", "e"]), response(ids=["f", "g"])], count=7)
    obj._proxy_req_kw = {"proxy": "http://proxy.example:8080"}
    result = await obj.send_multiple_images("channel", images)
    assert result.success
    assert obj._session.post.call_count == 2
    assert obj._post_message.await_count == 2
    assert [len(c.args[-1]) for c in obj._post_message.await_args_list] == [5, 2]
    for call in obj._session.post.call_args_list:
        assert call.kwargs["proxy"] == "http://proxy.example:8080"
        assert call.kwargs["headers"] == {"Authorization": "Bearer fixture"}
        assert call.kwargs["timeout"].total == 60


@pytest.mark.asyncio
async def test_upload_exception_preserves_existing_fallback(tmp_path, monkeypatch):
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.base import SendResult
    obj, images = adapter(tmp_path, [aiohttp.ClientConnectionError("fixture")])
    fallback = AsyncMock(return_value=SendResult(success=True))
    monkeypatch.setattr(BasePlatformAdapter, "send_multiple_images", fallback)
    result = await obj.send_multiple_images("channel", images, metadata={"thread_id": "root"})
    assert result.success
    obj._post_message.assert_not_awaited()
    fallback.assert_awaited_once_with("channel", images, {"thread_id": "root"}, human_delay=0.0)


@pytest.mark.asyncio
async def test_missing_images_do_not_upload(tmp_path):
    obj, images = adapter(tmp_path, [])
    for path in tmp_path.glob("image*.png"):
        path.unlink()
    result = await obj.send_multiple_images("channel", images)
    assert not result.success
    obj._session.post.assert_not_called()
    obj._post_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_rejected_images_do_not_post(tmp_path):
    obj, images = adapter(tmp_path, [response(413)] * 4)
    result = await obj.send_multiple_images("channel", images)
    assert not result.success
    assert obj._session.post.call_count == 4
    obj._post_message.assert_not_awaited()
