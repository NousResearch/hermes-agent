"""Every relay attempt (retries, recovery rungs, fallbacks) streams through the progress hook (#98466).

Before the seam default, only the primary attempt passed ``create=``; the 18 unwrapped relay sites went
out non-streaming and ticked the compression watchdog zero times, so a healthy still-generating summary
was killed at the idle deadline ("timed out after 120.0s with no output from the summary model").
"""
import asyncio
from types import SimpleNamespace

from agent import auxiliary_client as aux


def _chunk(text):
    return SimpleNamespace(id="r1", model="m", usage=None, choices=[SimpleNamespace(
        finish_reason=None, delta=SimpleNamespace(content=text, reasoning=None, reasoning_content=None,
                                                  reasoning_details=None, tool_calls=None))])


class _SyncClient:
    def __init__(self):
        self.wire = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.base_url = "https://example.test/v1"

    def _create(self, **kwargs):
        self.wire.append(kwargs.get("stream"))
        if kwargs.get("stream"):
            return iter([_chunk("hello"), _chunk(" world")])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="plain"))])


class _AsyncClient(_SyncClient):
    async def _create(self, **kwargs):
        self.wire.append(kwargs.get("stream"))
        if kwargs.get("stream"):
            async def agen():
                yield _chunk("hello")
                yield _chunk(" world")
            return agen()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="plain"))])


def test_relay_default_callback_streams_and_ticks_hook_sync_and_async():
    ticks = []
    with aux.aux_progress_hook(lambda: ticks.append(1)):
        sync_client = _SyncClient()
        resp = aux._relay_sync_completion(sync_client, {"model": "m", "messages": []})
        assert sync_client.wire == [True]
        assert resp.choices[0].message.content == "hello world"
        sync_ticks = len(ticks)
        assert sync_ticks >= 2  # one per substantive chunk, on top of the dispatch tick

        async_client = _AsyncClient()
        resp = asyncio.run(aux._relay_async_completion(async_client, {"model": "m", "messages": []}))
        assert async_client.wire == [True]
        assert resp.choices[0].message.content == "hello world"
        assert len(ticks) - sync_ticks >= 2


def test_relay_default_callback_is_plain_create_without_hook():
    client = _SyncClient()
    resp = aux._relay_sync_completion(client, {"model": "m", "messages": []})
    assert client.wire == [None]
    assert resp.choices[0].message.content == "plain"


def test_async_midstream_failure_is_not_retried_non_streaming():
    """Plain-create fallback covers stream NEGOTIATION rejections only; a failure after content has
    streamed must surface to the classified recovery ladder, not silently re-send the prompt."""
    class _Client(_AsyncClient):
        async def _create(self, **kwargs):
            self.wire.append(kwargs.get("stream"))
            if kwargs.get("stream"):
                async def agen():
                    yield _chunk("partial")
                    raise ValueError("bad frame")
                return agen()
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="plain"))])

    client = _Client()
    with aux.aux_progress_hook(lambda: None):
        try:
            asyncio.run(aux._acreate_with_progress(client, {"model": "m", "messages": []}))
        except ValueError:
            pass
        else:
            raise AssertionError("midstream failure must propagate")
    assert client.wire == [True]
