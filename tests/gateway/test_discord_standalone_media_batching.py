"""Phase-2A standalone Discord multipart batching + one-429 retry regression.

Hermetic: no real network, no hardcoded /tmp paths. Calls the real
``_standalone_send(pconfig, chat_id, message, media_files=...)`` signature
and asserts two properties:

1. 26 media files are chunked into [10, 10, 6] batches (3 multipart POSTs).
2. One 429 on a media batch retries once inline and still succeeds.

The aiohttp.ClientSession is fully faked; ``_standalone_read_text_limited``
and ``_standalone_read_json_limited`` are patched to return canned bodies so
no real HTTP streams are read.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.discord import adapter as discord_adapter


def _make_pconfig():
    return SimpleNamespace(token="fake-token")


class _FakeMultipartResponse:
    """A single ``async with session.post(...)`` context-manager response."""

    def __init__(self, status, body_text='{"id": "msg1"}', json_body=None):
        self.status = status
        self._body_text = body_text
        self._json_body = json_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeJsonResponse:
    """Response for the text-message JSON POST (first call)."""

    def __init__(self, status=200, json_body=None):
        self.status = status
        self._json_body = json_body or {"id": "txt1"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _build_fake_session(media_post_plan):
    """Build a fake aiohttp.ClientSession.

    ``media_post_plan`` is a list of statuses to return for each *media*
    multipart POST (consumed in order). Text-message JSON POSTs always get a
    200. The session tracks every POST's ``data`` kwarg so the test can
    inspect batch sizes via the FormData field count.
    """
    media_calls = []
    json_calls = []
    plan_iter = iter(media_post_plan)

    def _post(url, headers=None, json=None, data=None, **kw):
        # JSON POSTs pass ``json=``; multipart POSTs pass ``data=<FormData>``.
        if json is not None:
            json_calls.append(json)
            return _FakeJsonResponse()
        # multipart path
        n_fields = len(getattr(data, "_fields", []))
        media_calls.append(n_fields)
        try:
            status = next(plan_iter)
        except StopIteration:
            status = 200
        if status == 429:
            return _FakeMultipartResponse(429, body_text='{"retry_after": 0.2}')
        return _FakeMultipartResponse(200, json_body={"id": "msg-x"})

    session = MagicMock()
    session.post = MagicMock(side_effect=_post)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session, media_calls


@pytest.mark.asyncio
async def test_26_media_files_chunked_into_10_10_6(tmp_path):
    """26 media files produce 3 multipart batches of [10, 10, 6]."""
    media = []
    for i in range(26):
        p = tmp_path / f"doc-{i:02d}.html"
        p.write_text(f"<html>doc {i}</html>")
        media.append((str(p), False))

    # All media batches return 200 (no 429).
    session, media_calls = _build_fake_session([200, 200, 200])

    close_spy = MagicMock(wraps=discord_adapter._standalone_close_handles)
    with patch("aiohttp.ClientSession", return_value=session), \
         patch.object(
             discord_adapter, "_standalone_read_json_limited",
             AsyncMock(return_value={"id": "msg-x"}),
         ), \
         patch.object(
             discord_adapter, "_standalone_read_text_limited",
             AsyncMock(return_value=""),
         ), \
         patch.object(
             discord_adapter, "_standalone_close_handles", close_spy,
         ), \
         patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        result = await discord_adapter._standalone_send(
            _make_pconfig(), chat_id="123", message="", media_files=media,
        )

    assert result.get("success") is True, f"unexpected result: {result}"
    # Exactly 3 media multipart POSTs, sized [10, 10, 6].
    assert media_calls == [10, 10, 6], f"batch sizes wrong: {media_calls}"
    # Deterministic handle close: one close per batch (no 429 retries).
    assert close_spy.call_count == 3, (
        f"expected 3 handle-close calls (one per batch), got {close_spy.call_count}"
    )
    # 2 inter-batch sleeps (between batch 1->2 and 2->3); text message uses
    # _standalone_post_json_with_429_retry which does NOT sleep on 200.
    sleep_calls = [c.args[0] if c.args else c[0][0] for c in mock_sleep.await_args_list]
    assert len(sleep_calls) == 2, f"expected 2 inter-batch sleeps, got {len(sleep_calls)}: {sleep_calls}"
    assert all(abs(s - 1.0) < 1e-9 for s in sleep_calls), f"expected 1.0s delays, got {sleep_calls}"


@pytest.mark.asyncio
async def test_one_429_on_media_batch_retries_and_succeeds(tmp_path):
    """First media batch returns 429 then 200 on retry; overall send succeeds."""
    media = []
    for i in range(26):
        p = tmp_path / f"doc-{i:02d}.html"
        p.write_text(f"<html>doc {i}</html>")
        media.append((str(p), False))

    # Plan: batch1 -> 429 then 200(retry); batch2 -> 200; batch3 -> 200.
    # That's 4 media POSTs total (1 retry).
    session, media_calls = _build_fake_session([429, 200, 200, 200])

    with patch("aiohttp.ClientSession", return_value=session), \
         patch.object(
             discord_adapter, "_standalone_read_json_limited",
             AsyncMock(return_value={"id": "msg-x"}),
         ), \
         patch.object(
             discord_adapter, "_standalone_read_text_limited",
             AsyncMock(return_value='{"retry_after": 0.2}'),
         ), \
         patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        result = await discord_adapter._standalone_send(
            _make_pconfig(), chat_id="123", message="", media_files=media,
        )

    assert result.get("success") is True, f"unexpected result: {result}"
    # 4 media POSTs: 429 + retry(200) for batch1, then 200, 200.
    assert media_calls == [10, 10, 10, 6], (
        f"expected [10,10,10,6] (retry of batch1 doubles its count), got {media_calls}"
    )
    # At least one sleep is the 429 retry_after (0.2s); 2 are inter-batch (1.0s).
    sleep_calls = [c.args[0] if c.args else c[0][0] for c in mock_sleep.await_args_list]
    retry_sleeps = [s for s in sleep_calls if abs(s - 0.2) < 1e-6]
    batch_sleeps = [s for s in sleep_calls if abs(s - 1.0) < 1e-9]
    assert len(retry_sleeps) == 1, f"expected one 429-retry sleep, got {sleep_calls}"
    assert len(batch_sleeps) == 2, f"expected two inter-batch sleeps, got {sleep_calls}"
