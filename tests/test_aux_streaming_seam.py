"""Seam tests for the R5 C-B extraction: agent/aux_streaming.py.

The streaming cluster (window 8221-8558 of ``agent/auxiliary_client.py``)
moved into ``agent/aux_streaming.py``. The godfile keeps an identity
re-export block, so every moved name must resolve to the SAME object through
both modules — not a copy, not a wrapper. Plus a streaming behavior smoke:
accumulator append/read and sync/async stream aggregation.
"""

import pytest
from types import SimpleNamespace

from agent import auxiliary_client as godfile
from agent import aux_streaming as seam

# Every name moved by the R5 C-B extraction, in window order.
REEXPORTED_NAMES = [
    "_AUX_STREAM_CEILING_FLOOR_SECONDS",
    "_AUX_STREAM_CEILING_MULTIPLIER",
    "_aux_stream_total_ceiling",
    "_client_streams_internally",
    "_is_streaming_rejected_error",
    "_provider_requires_stream",
    "_create_with_progress",
    "_aggregate_chat_stream",
    "_ChatStreamAccumulator",
    "_aggregate_chat_stream_async",
    "_acreate_with_stream",
]


def test_every_reexported_name_is_identical():
    """The godfile re-export must resolve `is`-identical, never a copy."""
    assert len(REEXPORTED_NAMES) == 11
    for name in REEXPORTED_NAMES:
        seam_obj = getattr(seam, name)
        god_obj = getattr(godfile, name)
        assert seam_obj is god_obj, (
            f"{name}: aux_streaming.{name} is not auxiliary_client.{name}"
        )


def test_moved_names_live_in_aux_streaming_module():
    """Sanity: the definitions really moved (module identity, not re-import)."""
    for name in REEXPORTED_NAMES:
        obj = getattr(seam, name)
        if callable(obj) or isinstance(obj, type):
            assert obj.__module__ == "agent.aux_streaming", name


def _chunk(content=None, reasoning=None, tool_calls=None, finish_reason=None,
           chunk_id="c1", model="m1", usage=None):
    delta = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(id=chunk_id, model=model, choices=[choice], usage=usage)


class TestAccumulatorAppendRead:
    def test_append_content_and_finish_aggregates(self):
        acc = seam._ChatStreamAccumulator(model="m1")
        acc.feed(_chunk(content="Hel"))
        acc.feed(_chunk(content="lo "))
        acc.feed(_chunk(content="world", finish_reason="stop"))
        resp = acc.finish()
        assert resp.model == "m1"
        assert resp.choices[0].message.content == "Hello world"
        assert resp.choices[0].finish_reason == "stop"

    def test_append_reasoning_and_tool_calls(self):
        acc = seam._ChatStreamAccumulator(model="m2")
        acc.feed(_chunk(reasoning="think"))
        acc.feed(_chunk(reasoning="ing"))
        acc.feed(_chunk(tool_calls=[
            SimpleNamespace(
                index=0, id="call_1",
                function=SimpleNamespace(name="search", arguments='{"q":'),
            ),
        ]))
        acc.feed(_chunk(tool_calls=[
            SimpleNamespace(
                index=0, id=None,
                function=SimpleNamespace(name=None, arguments='"x"}'),
            ),
        ]))
        resp = acc.finish()
        msg = resp.choices[0].message
        assert msg.reasoning == "thinking"
        assert msg.tool_calls[0].id == "call_1"
        assert msg.tool_calls[0].function.name == "search"
        assert msg.tool_calls[0].function.arguments == '{"q":"x"}'

    def test_ceiling_times_out(self):
        acc = seam._ChatStreamAccumulator(total_ceiling=-1.0)
        with pytest.raises(TimeoutError, match="timed out"):
            acc.feed(_chunk(content="x"))


class TestStreamAggregation:
    def test_sync_aggregation(self):
        chunks = iter([
            _chunk(content="a"),
            _chunk(content="b", finish_reason="stop"),
        ])
        resp = seam._aggregate_chat_stream(chunks, model="m3")
        assert resp.choices[0].message.content == "ab"
        assert resp.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_async_aggregation(self):
        async def stream():
            yield _chunk(content="a")
            yield _chunk(content="b", finish_reason="stop")

        resp = await seam._aggregate_chat_stream_async(stream(), model="m4")
        assert resp.choices[0].message.content == "ab"
        assert resp.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_acreate_with_stream_aggregates(self):
        class _FakeAsyncClient:
            def __init__(self):
                self.calls = []
                self.chat = SimpleNamespace(completions=SimpleNamespace(
                    create=self._create,
                ))

            async def _create(self, **kwargs):
                self.calls.append(kwargs)
                async def stream():
                    yield _chunk(content="async ", model="m5")
                    yield _chunk(content="ok", model="m5", finish_reason="stop")
                return stream()

        client = _FakeAsyncClient()
        resp = await seam._acreate_with_stream(
            client, {"model": "m5", "timeout": 30.0}, task="t",
        )
        assert resp.choices[0].message.content == "async ok"
        assert client.calls[0]["stream"] is True
        assert client.calls[0]["stream_options"] == {"include_usage": True}

    def test_total_ceiling_floor(self):
        assert seam._aux_stream_total_ceiling(30) == 600.0
        assert seam._aux_stream_total_ceiling(None) == 600.0
        assert seam._aux_stream_total_ceiling(0) == 600.0
        assert seam._aux_stream_total_ceiling(1000) == 4000.0
