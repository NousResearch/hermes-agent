"""Discord hides a ``||![alt](url)||`` image: the attachment is named ``SPOILER_…`` (the only
signal Discord blurs on) and its caption goes inside spoiler bars. A failed download posts the
link hidden and without a preview, in a reply's batch as in a single send, and a new forum post
never takes a spoiler as its title."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.base_image_tags import SpoilerAlt
from plugins.platforms.discord.adapter import DiscordAdapter, _derive_forum_thread_name

URL = "https://cdn.example.com/boss.png"


GOOD, BAD = "https://e.com/map.png", "https://e.com/missing-ending.png"


def _adapter(monkeypatch, channel, *, failing=(), safe=True, forum=False):
    """A Discord adapter whose downloads answer 404 for the ``failing`` URLs and 200 otherwise."""
    async def download(_session, url, **_kw):
        return (404, b"", {}) if url in failing else (200, b"\x89PNG", {"content-type": "image/png"})

    monkeypatch.setattr("plugins.platforms.discord.adapter.is_safe_url", lambda _url: safe)
    monkeypatch.setattr("plugins.platforms.discord.adapter._read_url_image_with_redirect_guard", download)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._is_forum_parent = lambda _ch: forum  # type: ignore[method-assign]
    adapter._client = SimpleNamespace(get_channel=lambda _cid: channel, fetch_channel=AsyncMock())
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text"))  # type: ignore[method-assign]
    return adapter


def _channel():
    return SimpleNamespace(id=5, guild=None, send=AsyncMock(return_value=SimpleNamespace(id=42)))


@pytest.mark.asyncio
@pytest.mark.parametrize("spoiler, filename, content", [
    (True, "SPOILER_image.png", "||Final boss||"),
    (False, "image.png", "Final boss"),
])
async def test_only_a_spoiler_image_is_a_spoiler_attachment_with_a_hidden_caption(monkeypatch, spoiler, filename, content):
    channel = _channel()
    result = await _adapter(monkeypatch, channel).send_image("5", URL, caption="Final boss", spoiler=spoiler)

    assert result.success
    kwargs = channel.send.await_args.kwargs
    assert kwargs["file"].filename == filename and kwargs["content"] == content


@pytest.mark.asyncio
@pytest.mark.parametrize("spoiler, failing, safe, content", [
    (True, (URL,), True, f"||Final boss||\n||<{URL}>||"),
    (True, (), False, f"||Final boss||\n||<{URL}>||"),
    (False, (URL,), True, f"Final boss\n{URL}"),
])
async def test_only_a_spoiler_fallback_link_is_hidden_and_has_no_preview(monkeypatch, spoiler, failing, safe, content):
    """A failed download (or a URL the SSRF guard refuses) posts the link instead; a bare URL
    would unfurl into the image itself, so a spoiler's goes as ``||<url>||``."""
    channel = _channel()
    adapter = _adapter(monkeypatch, channel, failing=failing, safe=safe)

    await adapter.send_image("5", URL, caption="Final boss", spoiler=spoiler)

    channel.send.assert_not_awaited()
    assert adapter.send.await_args.kwargs["content"] == content


@pytest.mark.asyncio
async def test_batch_hides_only_the_marked_image(monkeypatch):
    channel = _channel()
    images, _ = BasePlatformAdapter.extract_images(
        "||![The ending](https://e.com/end.png)|| ![World map](https://e.com/map.png)")

    result = await _adapter(monkeypatch, channel).send_multiple_images("5", images)

    assert result.success
    kwargs = channel.send.await_args.kwargs
    assert [f.filename for f in kwargs["files"]] == ["SPOILER_image_0.png", "image_1.png"]
    assert kwargs["content"] == "||The ending||"


@pytest.mark.asyncio
async def test_batch_posts_a_failed_spoiler_as_a_hidden_link_beside_the_images_that_worked(monkeypatch):
    """Every reply goes through the batch: a spoiler that 404s must not vanish, nor unfurl."""
    channel = _channel()
    images, _ = BasePlatformAdapter.extract_images(f"||![The ending]({BAD})|| ![World map]({GOOD})")

    result = await _adapter(monkeypatch, channel, failing=(BAD,)).send_multiple_images("5", images)

    assert result.success
    kwargs = channel.send.await_args.kwargs
    assert [f.filename for f in kwargs["files"]] == ["image_0.png"]
    assert kwargs["content"] == f"||The ending||\n||<{BAD}>||"


@pytest.mark.asyncio
async def test_batch_whose_images_all_fail_still_delivers_the_hidden_links(monkeypatch):
    channel = _channel()
    other = "https://e.com/missing-boss.png"
    images, _ = BasePlatformAdapter.extract_images(f"||![The ending]({BAD})|| ||![Boss]({other})||")
    adapter = _adapter(monkeypatch, channel, failing=(BAD, other))

    result = await adapter.send_multiple_images("5", images)

    assert result.success
    channel.send.assert_not_awaited()
    adapter.send.assert_awaited_once_with(
        chat_id="5", content=f"||The ending||\n||<{BAD}>||\n||<{other}>||", metadata=None)


LONG = [f"https://e.com/{i}/{'x' * 300}.png" for i in range(9)]  # one chunk: nine links, about 2,900 characters


@pytest.mark.asyncio
@pytest.mark.parametrize("images, caption, links", [
    ([(GOOD, "c" * 1980), (BAD, SpoilerAlt(""))], "c" * 1980, [BAD]),  # with the link: 2,020 characters
    ([(GOOD, "")] + [(url, SpoilerAlt("")) for url in LONG], None, LONG),
])
async def test_hidden_links_that_would_pass_the_message_limit_follow_in_their_own_message(
        monkeypatch, images, caption, links):
    """Discord refuses a message over 2,000 characters: the files keep the caption, the links
    follow through ``send``, which splits a long text."""
    channel = _channel()
    adapter = _adapter(monkeypatch, channel, failing=(BAD, *LONG))

    result = await adapter.send_multiple_images("5", images)

    assert result.success
    kwargs = channel.send.await_args.kwargs
    assert kwargs["content"] == caption and len(kwargs["files"]) == 1
    adapter.send.assert_awaited_once_with(
        chat_id="5", content="\n".join(f"||<{url}>||" for url in links), metadata=None)


@pytest.mark.asyncio
async def test_hidden_links_that_follow_a_forum_post_go_into_its_thread(monkeypatch):
    forum = SimpleNamespace(id=5, guild=None, send=AsyncMock())
    adapter = _adapter(monkeypatch, forum, failing=(BAD,), forum=True)
    adapter._forum_post_file = AsyncMock(  # type: ignore[method-assign]
        return_value=SendResult(success=True, message_id="8", raw_response={"thread_id": "7"}))

    result = await adapter.send_multiple_images("5", [(GOOD, "c" * 1980), (BAD, SpoilerAlt(""))])

    assert result.success
    assert adapter._forum_post_file.await_args.kwargs["content"] == "c" * 1980
    adapter.send.assert_awaited_once_with(chat_id="5", content=f"||<{BAD}>||", metadata={"thread_id": "7"})
    forum.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_unmarked_and_local_images_are_dropped_as_before(monkeypatch, tmp_path):
    """Only a remote spoiler gets the link: a plain image never did, and a local path is no link."""
    channel = _channel()
    missing = (tmp_path / "gone.png").as_uri()
    images = [(BAD, "Plain"), (missing, SpoilerAlt("Local")), (GOOD, "")]

    result = await _adapter(monkeypatch, channel, failing=(BAD,)).send_multiple_images("5", images)

    assert result.success
    kwargs = channel.send.await_args.kwargs
    assert kwargs["content"] == "Plain" and [f.filename for f in kwargs["files"]] == ["image_0.png"]


@pytest.mark.asyncio
async def test_spoiler_image_in_a_forum_channel_does_not_title_the_post_with_its_caption(monkeypatch):
    thread = SimpleNamespace(id=7, message=SimpleNamespace(id=8, attachments=[SimpleNamespace(filename="x")]),
                             thread=SimpleNamespace(id=7, send=AsyncMock()))
    forum = SimpleNamespace(id=5, guild=None, create_thread=AsyncMock(return_value=thread))

    result = await _adapter(monkeypatch, forum, forum=True).send_image("5", URL, caption="Final boss", spoiler=True)

    assert result.success
    kwargs = forum.create_thread.await_args.kwargs
    assert kwargs["file"].filename == "SPOILER_image.png" and kwargs["content"] == "||Final boss||"
    assert "Final boss" not in kwargs["name"]


def test_forum_title_is_never_a_spoiler():
    assert _derive_forum_thread_name("||Snape kills Dumbledore||\nmore") == _derive_forum_thread_name("")
    assert _derive_forum_thread_name("Spoilers below\n||x||") == "Spoilers below"
