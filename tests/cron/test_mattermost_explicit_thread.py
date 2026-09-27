"""Cron's selected Mattermost thread applies to both text and attached files."""

import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import Platform, PlatformConfig
from plugins.platforms.mattermost.adapter import MattermostAdapter


@pytest.mark.parametrize("target_kind", ["explicit", "origin", "channel-only"])
def test_live_cron_thread_routes_text_and_media_with_replies_off(tmp_path, target_kind):
    channel, thread = "c" * 26, "r" * 26
    pconfig = PlatformConfig(enabled=True, token="fixture-token", extra={
        "url": "https://mattermost.example", "reply_mode": "off",
    })
    adapter = MattermostAdapter(pconfig)
    adapter._api_get = AsyncMock(return_value={"id": thread})
    adapter._session = MagicMock()
    config = MagicMock()
    config.platforms = {Platform.MATTERMOST: pconfig}
    config.get_home_channel.return_value = None
    posts = []

    def post(url, **kwargs):
        response = AsyncMock()
        response.__aenter__.return_value = response
        response.__aexit__.return_value = False
        response.status = 201
        if url.endswith("/files"):
            response.json.return_value = {"file_infos": [{"id": "attachment"}]}
        else:
            assert url.endswith("/posts")
            posts.append(kwargs["json"])
            response.json.return_value = {"id": f"post-{len(posts)}"}
        return response

    adapter._session.post.side_effect = post
    loop = MagicMock()
    loop.is_running.return_value = True

    def run_scheduled(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    job: dict = {"id": "thread-report", "name": "Report", "deliver": f"mattermost:{channel}:{thread}"}
    if target_kind == "origin":
        job.update(deliver="origin", origin={
            "platform": "mattermost", "chat_id": channel, "thread_id": thread,
        })
    elif target_kind == "channel-only":
        job["deliver"] = f"mattermost:{channel}"
    attachment = tmp_path / "chart.png"
    attachment.write_bytes(b"image")
    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}
    ), patch("asyncio.run_coroutine_threadsafe", side_effect=run_scheduled), patch(
        "cron.jobs.update_job"
    ), patch("cron.scheduler_delivery._seed_live_delivery_sessions"), patch(
        "cron.scheduler_delivery._maybe_mirror_cron_delivery"
    ):
        error = _deliver_result(job, f"Report\nMEDIA:{attachment}",
                                adapters={Platform.MATTERMOST: adapter}, loop=loop)

    assert error is None
    assert len(posts) == 2
    assert posts[0]["message"] == "Report"
    assert posts[1]["file_ids"] == ["attachment"]
    assert all(post["channel_id"] == channel for post in posts)
    if target_kind == "channel-only":
        assert all("root_id" not in post for post in posts)
    else:
        assert all(post.get("root_id") == thread for post in posts)
