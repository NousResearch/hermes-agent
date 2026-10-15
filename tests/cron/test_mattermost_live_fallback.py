"""Mattermost cron must preserve independent files without replaying uncertain text."""

import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import Platform, PlatformConfig
from plugins.platforms.mattermost.adapter import MattermostAdapter


def test_warning_only_ack_keeps_cron_standalone_text_fallback_eligible(tmp_path, monkeypatch):
    from tools import send_message_tool as tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tool.prepare_send_message_platforms()
    pconfig = PlatformConfig(enabled=True, token="fixture-token", extra={
        "url": "https://mattermost.example", "max_post_length": 500, "reply_mode": "thread",
    })
    adapter = MattermostAdapter(pconfig)
    posts = []

    async def post(path, payload):
        posts.append(payload)
        if len(posts) == 1:
            return {"_post_rejection_status": 400, "_post_broken_thread_root": True}
        if len(posts) == 2:
            return {"id": "warning-only"}
        return {"_post_rejection_status": 400}

    adapter._api_post = post
    adapter._api_get = AsyncMock(return_value={"id": "root", "root_id": ""})
    content = "x" * 500
    receipt = asyncio.run(adapter.send("channel", content, metadata={"notify": True, "thread_id": "root"}))
    assert [p["message"] for p in posts] == [content, posts[1]["message"], content]
    assert receipt.message_id == "warning-only"
    assert receipt.raw_response["source_confirmed_prefix"] == ""
    assert receipt.raw_response["source_attempted_prefix"] == ""
    assert not receipt.raw_response.get("partial_overflow")

    adapter.send = AsyncMock(return_value=receipt)
    config = MagicMock()
    config.platforms = {Platform.MATTERMOST: pconfig}
    config.get_home_channel.return_value = None
    loop = MagicMock()
    loop.is_running.return_value = True

    def run_scheduled(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    standalone = MagicMock(return_value=({"success": True, "message_id": "recovered"}, None))
    job = {"id": "warning-only", "name": "Report", "deliver": "origin",
           "origin": {"platform": "mattermost", "chat_id": "channel"}}
    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}
    ), patch("asyncio.run_coroutine_threadsafe", side_effect=run_scheduled), patch(
        "cron.jobs.update_job"
    ), patch("cron.scheduler_delivery._standalone_send", standalone), patch(
        "cron.scheduler_delivery._seed_live_delivery_sessions"
    ), patch("cron.scheduler_delivery._maybe_mirror_cron_delivery"):
        assert _deliver_result(job, content, adapters={Platform.MATTERMOST: adapter}, loop=loop) is None
    standalone.assert_called_once()
    assert not job.get("last_delivery_unverified")


@pytest.mark.parametrize(("lost_post", "rejected"), [
    pytest.param(1, False, id="first-post"),
    pytest.param(2, False, id="confirmed-prefix"),
    pytest.param(1, True, id="zero-delivery-rejected"),
    pytest.param(2, True, id="confirmed-prefix-rejected"),
])
def test_uncertain_live_post_does_not_replay_via_standalone(
    tmp_path, monkeypatch, caplog, lost_post, rejected,
):
    import aiohttp
    from tools import send_message_tool as tool

    caplog.set_level("INFO", logger="cron.scheduler")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.platforms import base as platform_base
    monkeypatch.setattr(platform_base, "_HERMES_ROOT", tmp_path)
    monkeypatch.setattr(platform_base, "MEDIA_DELIVERY_SAFE_ROOTS",
                        tuple(tmp_path / "cache" / name for name in platform_base._MEDIA_DELIVERY_CACHE_SUBDIRS))
    monkeypatch.setenv("MATTERMOST_URL", "https://mattermost.example")
    monkeypatch.setenv("MATTERMOST_TOKEN", "fixture-token")
    tool.prepare_send_message_platforms()
    # If cron wrongly falls through, exercise the actual standalone sender too.
    monkeypatch.setattr(tool, "_live_adapter", lambda platform: (None, None))
    pconfig = PlatformConfig(enabled=True, token="fixture-token", extra={
        "url": "https://mattermost.example", "max_post_length": 500, "reply_mode": "off",
    })
    adapter = MattermostAdapter(pconfig)
    config = MagicMock()
    config.platforms = {Platform.MATTERMOST: pconfig}
    config.get_home_channel.return_value = None
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    posts, uploads = [], []

    def post(url, **kwargs):
        response = AsyncMock()
        response.__aenter__.return_value = response
        response.__aexit__.return_value = False
        response.status = 201
        if url == "https://mattermost.example/api/v4/files":
            uploads.append(kwargs["data"])
            response.json.return_value = {"file_infos": [{"id": "chart-file"}]}
            return response
        assert url == "https://mattermost.example/api/v4/posts"
        posts.append(kwargs["json"])
        response.json.return_value = {"id": f"post-{len(posts)}"}
        if len(posts) == lost_post:
            if rejected:
                response.status = 400
                response.text.return_value = "post rejected"
            else:
                # The server accepted the POST; only reading its receipt fails.
                response.json.side_effect = TimeoutError()
        return response

    session.post.side_effect = post
    adapter._session = session
    loop = MagicMock()
    loop.is_running.return_value = True

    def run_scheduled(coro, _loop):
        # Existing cron confirmation scaffold: execute the real router coroutine.
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    job = {"id": "uncertain-mattermost", "name": "Report", "deliver": "origin",
           "origin": {"platform": "mattermost", "chat_id": "channel"}}
    content = "a" * 500 + "b" * 100
    attachment = tmp_path / "chart.png"
    attachment.write_bytes(b"chart fixture")
    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}
    ), patch("asyncio.run_coroutine_threadsafe", side_effect=run_scheduled), patch(
        "cron.jobs.update_job"
    ), patch(
        "cron.scheduler_delivery._seed_live_delivery_sessions"
    ) as seed_live_session, patch(
        "cron.scheduler_delivery._maybe_mirror_cron_delivery"
    ), patch.object(aiohttp, "ClientSession", return_value=session) as standalone_session:
        error = _deliver_result(job, f"{content}\nMEDIA:{attachment}",
                                adapters={Platform.MATTERMOST: adapter}, loop=loop)

    seed_live_session.assert_not_called()
    assert "via live adapter thread=" not in caplog.text
    assert all(p["props"]["disable_mentions"] for p in posts)
    file_posts = [p for p in posts if p.get("file_ids")]
    if rejected and lost_post == 1:
        # No content was accepted: a real standalone fallback remains permitted.
        assert [p["message"] for p in posts] == [content[:500], content[:500], content[500:]]
        standalone_session.assert_called_once()
        assert not job.get("last_delivery_unverified")
        assert error is None
        assert len(uploads) == len(file_posts) == 1
        assert file_posts[0]["file_ids"] == ["chart-file"]
        return

    assert [p["message"] for p in posts if not p.get("file_ids")] == [content[:500], content[500:]][:lost_post], (
        "cron replayed an accepted but unacknowledged POST through standalone delivery"
    )
    standalone_session.assert_not_called()
    assert job.get("last_delivery_unverified") == ["mattermost:channel"]
    assert error is not None, "uncertain delivery must not be reported as confirmed success"
    assert len(uploads) == len(file_posts) == 1, "text uncertainty abandoned an independent attachment"
    assert file_posts[0]["file_ids"] == ["chart-file"]
    assert file_posts[0]["message"] in ("", "📎 chart.png"), "attachment delivery must not replay the text as a caption"
