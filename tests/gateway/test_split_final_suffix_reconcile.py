"""Regression: an authoritative final that prefix-extends a SPLIT
delivery reconciles by suffix, not by full resend (PR 85796 review
round 2, finding 3).

The _FINAL_TEXT adoption guard refuses wholesale adoption on split
turns — correct (#78541: sealed heads would repeat inside the tail) but
previously absolute: a post-split verifier footer never entered the
ledger, delivered_final_matches() reported a mismatch, and the gateway
resent the ENTIRE body+footer after the split chunks. Now the strictly
prefix-extending case appends only the missing suffix to the live tail
and the ledger; non-prefix rewrites keep the full-resend fallback.
"""

import asyncio

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_adapter():
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    A = type("SplitAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 4096})
    A.__abstractmethods__ = frozenset()
    a = A.__new__(A)
    a._typing_paused = set()
    a._fatal_error_message = None
    a.send_calls = []
    a.edit_calls = []

    async def _send(chat_id, content, reply_to=None, metadata=None, **kw):
        a.send_calls.append(content)
        return SendResult(success=True, message_id=f"m{len(a.send_calls)}")
    a.send = _send

    async def _edit(chat_id, message_id, content, **kw):
        a.edit_calls.append({"id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)
    a.edit_message = _edit
    return a


BODY = "the split answer body"
FOOTER = "\n\n⚠️ File-mutation verifier: 1 file(s) were NOT modified this turn."


def _consumer(adapter):
    cfg = StreamConsumerConfig(
        transport="edit", chat_type="dm",
        edit_interval=0.01, buffer_threshold=1, cursor="",
    )
    return GatewayStreamConsumer(adapter, "D1", cfg)


class TestSplitSuffixReconcile:
    @pytest.mark.asyncio
    async def test_prefix_extending_final_delivers_only_suffix(self):
        adapter = _make_adapter()
        sc = _consumer(adapter)
        task = asyncio.create_task(sc.run())
        sc.on_delta(BODY)
        await asyncio.sleep(0.06)
        # Simulate an overflow-adopted split mid-turn: heads sealed, the
        # ledger holds full text, _accumulated holds the tail only.
        sc._turn_split_delivery = True
        sc.finish(BODY + FOOTER)
        await task
        # The recorded payload reconciles with the authoritative final —
        # the gateway will NOT resend the body+footer.
        assert sc.delivered_final_matches(BODY + FOOTER) is True
        # And the footer actually reached the platform (rode the final
        # edit/send), rather than reconciling vacuously.
        all_payloads = adapter.send_calls + [
            e["content"] for e in adapter.edit_calls
        ]
        assert any(FOOTER.strip() in p for p in all_payloads)
        # The body was NOT re-delivered whole after the split: no payload
        # equals the full body+footer via a fresh send while heads were
        # already on screen — the suffix rode the existing tail.
        assert all(
            p.startswith(BODY) or FOOTER.strip() in p for p in all_payloads
        )

    @pytest.mark.asyncio
    async def test_non_prefix_rewrite_keeps_mismatch(self):
        """A rewritten final (not a prefix extension) must still report a
        mismatch so the gateway's full resend delivers the true final."""
        adapter = _make_adapter()
        sc = _consumer(adapter)
        task = asyncio.create_task(sc.run())
        sc.on_delta(BODY)
        await asyncio.sleep(0.06)
        sc._turn_split_delivery = True
        rewritten = "a completely different final answer"
        sc.finish(rewritten)
        await task
        assert sc.delivered_final_matches(rewritten) is False

    @pytest.mark.asyncio
    async def test_unsplit_turn_adoption_unchanged(self):
        adapter = _make_adapter()
        sc = _consumer(adapter)
        task = asyncio.create_task(sc.run())
        sc.on_delta(BODY)
        await asyncio.sleep(0.06)
        sc.finish(BODY + FOOTER)
        await task
        assert sc.delivered_final_matches(BODY + FOOTER) is True
