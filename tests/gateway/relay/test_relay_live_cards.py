"""Relay Slack live cards — gateway half (NS-658).

The relay adapter emits three new additive ops when the connector's
negotiated descriptor advertises them:

  {op: "draft", chat_id, draft_id, content, final, metadata}
  {op: "task_card", chat_id, card_id, chunks, metadata}
  {op: "task_card_stop", chat_id, card_id, metadata}

Gateway side is deliberately dumb: no Slack API knowledge, no new config
keys. Capability is descriptor-driven — an old connector that never
advertises "draft"/"task_card" gets byte-identical behavior to today.

Semantic bridge: the base send_draft contract is Telegram-shaped (draft
clears; final answer is a separate send). Slack native streaming makes the
stream THE message. The adapter tracks the open draft per chat; the
turn-final regular send() for that chat converts to draft(final=true) so
the connector seals the stream instead of posting a duplicate.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from tests.gateway.relay.stub_connector import StubConnector


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="slack",
        label="Slack",
        max_message_length=39000,
        supports_draft_streaming=True,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="slack",
        len_unit="chars",
        emoji="\U0001f4ac",
        platform_hint="",
        pii_safe=False,
        supported_ops=("send", "edit", "typing", "draft", "task_card"),
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _connected_adapter(**desc_kw):
    desc = make_desc(**desc_kw)
    stub = StubConnector(desc)
    adapter = RelayAdapter(PlatformConfig(), desc, transport=stub)
    return adapter, stub


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Draft streaming op
# ---------------------------------------------------------------------------


class TestDraftOp:
    def test_send_draft_emits_draft_op(self, loop):
        adapter, stub = _connected_adapter()
        result = loop.run_until_complete(
            adapter.send_draft(chat_id="C1", draft_id=7, content="hel")
        )
        assert result.success
        frames = [s for s in stub.sent if s.get("op") == "draft"]
        assert len(frames) == 1
        frame = frames[0]
        assert frame["chat_id"] == "C1"
        assert frame["draft_id"] == 7
        assert frame["content"] == "hel"
        assert frame["final"] is False

    def test_send_draft_requires_descriptor_op(self, loop):
        # supports_draft_streaming True but op NOT advertised: old connector.
        adapter, stub = _connected_adapter(
            supported_ops=("send", "edit", "typing")
        )
        with pytest.raises(NotImplementedError):
            loop.run_until_complete(
                adapter.send_draft(chat_id="C1", draft_id=1, content="x")
            )
        assert not [s for s in stub.sent if s.get("op") == "draft"]

    def test_supports_draft_streaming_requires_both_flag_and_op(self):
        both, _ = _connected_adapter()
        assert both.supports_draft_streaming() is True
        flag_only, _ = _connected_adapter(
            supported_ops=("send", "edit", "typing")
        )
        assert flag_only.supports_draft_streaming() is False
        # Legacy connector with EMPTY supported_ops: fail-open elsewhere, but
        # draft must NOT fail open — the op did not exist pre-contract.
        legacy, _ = _connected_adapter(supported_ops=())
        assert legacy.supports_draft_streaming() is False

    def test_final_send_seals_open_draft(self, loop):
        adapter, stub = _connected_adapter()
        loop.run_until_complete(
            adapter.send_draft(chat_id="C1", draft_id=7, content="hel")
        )
        loop.run_until_complete(
            adapter.send_draft(chat_id="C1", draft_id=7, content="hello wor")
        )
        stub.next_draft_result = {"success": True, "message_id": "1723600000.1"}
        result = loop.run_until_complete(
            adapter.send("C1", "hello world", reply_to=None)
        )
        assert result.success
        # No regular send op — the final frame rides the draft stream.
        ops = [s["op"] for s in stub.sent]
        assert "send" not in ops, "final must seal the stream, not post anew"
        finals = [
            s for s in stub.sent if s.get("op") == "draft" and s.get("final")
        ]
        assert len(finals) == 1
        assert finals[0]["content"] == "hello world"
        # Stream ts comes back as the message identity.
        assert result.message_id == "1723600000.1"

    def test_send_without_open_draft_is_normal(self, loop):
        adapter, stub = _connected_adapter()
        loop.run_until_complete(adapter.send("C1", "plain", reply_to=None))
        assert [s["op"] for s in stub.sent] == ["send"]

    def test_draft_state_clears_after_seal(self, loop):
        adapter, stub = _connected_adapter()
        loop.run_until_complete(
            adapter.send_draft(chat_id="C1", draft_id=7, content="a")
        )
        loop.run_until_complete(adapter.send("C1", "a!", reply_to=None))
        # Second send on the same chat is a NORMAL send again.
        loop.run_until_complete(adapter.send("C1", "followup", reply_to=None))
        ops = [s["op"] for s in stub.sent]
        assert ops == ["draft", "draft", "send"]

    def test_draft_failure_result_propagates(self, loop):
        adapter, stub = _connected_adapter()
        stub.next_draft_result = {"success": False, "error": "stream_gone"}
        result = loop.run_until_complete(
            adapter.send_draft(chat_id="C1", draft_id=1, content="x")
        )
        assert not result.success
        # DEFINITE connector rejection disarms seal-interception: the
        # stream consumer falls back to edit-based streaming on this
        # failure, and its turn-final must go out as a REAL send — never
        # a seal on a stream the connector just rejected.
        stub.next_send_result = {"success": True, "message_id": "m2"}
        loop.run_until_complete(adapter.send("C1", "final", reply_to=None))
        assert [s["op"] for s in stub.sent][-1] == "send"

    def test_draft_transport_exception_keeps_interception_armed(self, loop):
        """Ambiguity contract (G-D1): a transport EXCEPTION — as opposed
        to an explicit rejection — may mean the frame was delivered, so
        interception stays armed and the turn-final still seals."""
        adapter, _ = _connected_adapter()

        class _FlakyOnce:
            def __init__(self):
                self.calls = 0

            async def send_outbound(self, payload, platform=None):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionError("mid-write drop")
                return {"success": True, "message_id": "ts.9"}

        adapter._transport = _FlakyOnce()
        result = loop.run_until_complete(
            adapter.send_draft(chat_id="C1", draft_id=2, content="x")
        )
        assert not result.success
        # Still armed: the turn-final converts to the sealing frame.
        final = loop.run_until_complete(adapter.send("C1", "final", reply_to=None))
        assert final.success
        assert final.message_id == "ts.9"

    def test_drafts_are_per_chat(self, loop):
        adapter, stub = _connected_adapter()
        loop.run_until_complete(
            adapter.send_draft(chat_id="C1", draft_id=1, content="a")
        )
        # A send to a DIFFERENT chat is untouched.
        loop.run_until_complete(adapter.send("D9", "other", reply_to=None))
        by_op = [(s["op"], s["chat_id"]) for s in stub.sent]
        assert ("send", "D9") in by_op


# ---------------------------------------------------------------------------
# Task-card ops (#85476 TurnRunner seam, relay leg)
# ---------------------------------------------------------------------------


def _tasks():
    return [
        {
            "id": "call_1",
            "title": "terminal",
            "status": "in_progress",
            "details": "ls -la",
        },
        {
            "id": "call_2",
            "title": "web_search",
            "status": "complete",
            "details": "slack startStream",
        },
    ]


class TestTaskCardOps:
    def test_progress_emits_task_card_op(self, loop):
        adapter, stub = _connected_adapter()
        result = loop.run_until_complete(
            adapter.send_native_task_card_progress(
                chat_id="C1", tasks=_tasks(), reply_to="1700.100"
            )
        )
        assert result.success
        frames = [s for s in stub.sent if s.get("op") == "task_card"]
        assert len(frames) == 1
        assert frames[0]["card_id"] == "turn:1700.100"
        chunk_ids = [c["id"] for c in frames[0]["chunks"]]
        assert chunk_ids == ["call_1", "call_2"]
        statuses = {c["id"]: c["status"] for c in frames[0]["chunks"]}
        assert statuses["call_2"] == "complete"

    def test_stop_emits_task_card_stop(self, loop):
        adapter, stub = _connected_adapter()
        loop.run_until_complete(
            adapter.send_native_task_card_progress(
                chat_id="C1", tasks=_tasks(), reply_to="1700.100"
            )
        )
        loop.run_until_complete(
            adapter.stop_native_task_card_progress(chat_id="C1", reply_to="1700.100")
        )
        assert [s["op"] for s in stub.sent] == ["task_card", "task_card_stop"]

    def test_task_card_gated_on_descriptor(self, loop):
        adapter, stub = _connected_adapter(
            supported_ops=("send", "edit", "typing", "draft")
        )
        # No "task_card" in supported_ops: the TurnRunner's hasattr gate must
        # see NO capability — expose it via a probe method returning False,
        # and the send must be a clean no-op failure (never an exception on
        # the turn path).
        assert adapter.supports_native_task_cards() is False
        result = loop.run_until_complete(
            adapter.send_native_task_card_progress(
                chat_id="C1", tasks=_tasks(), reply_to="t"
            )
        )
        assert not result.success
        assert not stub.sent

    def test_task_card_legacy_empty_ops_not_fail_open(self):
        legacy, _ = _connected_adapter(supported_ops=())
        assert legacy.supports_native_task_cards() is False
