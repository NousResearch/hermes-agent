"""CardKit streaming-card adapter integration: draft contract, send routing,
cron result cards and topic isolation. Pure-logic SDK stubs (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any


# --------------------------------------------------------------------------- #
# SDK stub: records every CardKit/IM call the adapter makes.
# --------------------------------------------------------------------------- #

class _RecordingLark:
    def __init__(self) -> None:
        self.created_cards: list[dict] = []
        self.updated_cards: list[dict] = []
        self.closed: list[str] = []
        self.sent_cards: list[dict] = []
        self.replied_to: list[str] = []
        self._seq = 0

    @staticmethod
    def _ok(**data: Any) -> Any:
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(**data))

    def _card_api(self) -> Any:
        stub = self

        def create(request: Any) -> Any:
            stub._seq += 1
            stub.created_cards.append(request)
            return stub._ok(card_id=f"card_{stub._seq}")

        def update(request: Any) -> Any:
            stub.updated_cards.append(request)
            return stub._ok()

        def batch_update(request: Any) -> Any:
            return stub._ok()

        def settings(request: Any) -> Any:
            stub.closed.append(request)
            return stub._ok()

        return SimpleNamespace(create=create, update=update, batch_update=batch_update, settings=settings)

    def _element_api(self) -> Any:
        return SimpleNamespace(content=lambda request: self._ok())

    def _message_api(self) -> Any:
        stub = self

        async def areply(request: Any) -> Any:
            stub.replied_to.append(str(request))
            return stub._ok(message_id="om_reply")

        async def acreate(request: Any) -> Any:
            stub.sent_cards.append(request)
            return stub._ok(message_id="om_send")

        return SimpleNamespace(areply=areply, acreate=acreate)

    def __init_subclass__(cls) -> None:  # pragma: no cover - not subclassed
        raise TypeError

    # NS tree assembled lazily so __init__ ordering above stays simple
    @property
    def cardkit(self) -> Any:
        if not hasattr(self, "_cardkit_ns"):
            self._cardkit_ns = SimpleNamespace(
                v1=SimpleNamespace(card=self._card_api(), card_element=self._element_api()))
        return self._cardkit_ns

    @property
    def im(self) -> Any:
        if not hasattr(self, "_im_ns"):
            self._im_ns = SimpleNamespace(v1=SimpleNamespace(message=self._message_api()))
        return self._im_ns


def make_streaming_adapter() -> tuple[Any, _RecordingLark]:
    from plugins.platforms.feishu.adapter import FeishuAdapter

    lark = _RecordingLark()

    async def run_blocking(func: Any, *args: Any) -> Any:
        result = func(*args)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    adapter = object.__new__(FeishuAdapter)
    adapter._client = lark
    adapter._run_blocking = run_blocking
    return adapter, lark


async def _settle(rounds: int = 3) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0.4)
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _update_card_payload(request: Any) -> dict:
    body = request.request_body
    card_model = body.card
    import json
    return json.loads(card_model.data)


CRON_FAILURE = (
    "Cronjob Response: sentinel\n"
    "(job_id: j1)\n"
    "-------------\n\n"
    "**Status:** script failed\n\n{output}\n"
    '\n\nTo stop or manage this job, send me a new message (e.g. "stop reminder sentinel").'
)


# --------------------------------------------------------------------------- #
# Draft-streaming contract lifecycle
# --------------------------------------------------------------------------- #

def test_probe_draft_final_lifecycle() -> None:
    async def _t() -> None:
        adapter, lark = make_streaming_adapter()
        chat = "oc_" + "1" * 20
        assert adapter.supports_draft_streaming(chat_id=chat, metadata={}) is True
        await _settle(2)
        assert len(lark.created_cards) == 1, "probe opens the card eagerly"

        result = await adapter.send_draft(chat, 1, "streaming body", {"reply_to_message_id": "om_user"})
        assert result.success is True
        await _settle()
        assert adapter._card_engine().session_for(chat).answer_seg.text == "streaming body"

        final = await adapter.send(chat, "streaming body final")
        assert final.success is True and final.message_id
        await _settle(2)
        assert lark.closed, "completion closes streaming mode"
        assert lark.updated_cards, "completion re-renders the full card"
        payload = _update_card_payload(lark.updated_cards[-1])
        body = "".join(e.get("content", "") for e in payload["body"]["elements"]
                       if isinstance(e, dict) and e.get("tag") == "markdown")
        assert "streaming body final" in body

    asyncio.run(_t())


def test_probe_refuses_without_client() -> None:
    async def _t() -> None:
        adapter, _ = make_streaming_adapter()
        adapter._client = None
        assert adapter.supports_draft_streaming(chat_id="oc_x", metadata={}) is False

    asyncio.run(_t())


# --------------------------------------------------------------------------- #
# send() routing classification
# --------------------------------------------------------------------------- #

def test_interim_send_routes_to_heartbeat_not_native() -> None:
    async def _t() -> None:
        adapter, lark = make_streaming_adapter()
        chat = "oc_" + "2" * 20
        adapter.supports_draft_streaming(chat_id=chat, metadata={})
        await _settle(2)
        sent_before = len(lark.sent_cards)

        result = await adapter.send(chat, "...working", metadata={"_interim_send": True})
        assert result.success is True
        assert len(lark.sent_cards) == sent_before, "interim must not create a native message"
        session = adapter._card_engine().session_for(chat)
        assert session.heartbeat_text == "...working"

    asyncio.run(_t())


def test_busy_ack_routes_to_heartbeat() -> None:
    async def _t() -> None:
        adapter, lark = make_streaming_adapter()
        chat = "oc_" + "3" * 20
        adapter.supports_draft_streaming(chat_id=chat, metadata={})
        await _settle(2)
        sent_before = len(lark.sent_cards)

        result = await adapter.send(chat, "\u21aa redirected the current run")
        assert result.success is True
        assert len(lark.sent_cards) == sent_before

    asyncio.run(_t())


def test_unrelated_send_falls_through_to_native() -> None:
    async def _t() -> None:
        adapter, _ = make_streaming_adapter()
        routed = await adapter._streaming_cards_route("oc_" + "4" * 20, "plain text", None, None)
        assert routed is None, "no session / no markers: the native path owns it"

    asyncio.run(_t())


# --------------------------------------------------------------------------- #
# Cron result cards
# --------------------------------------------------------------------------- #

def test_cron_failure_renders_red_header_card() -> None:
    async def _t() -> None:
        adapter, lark = make_streaming_adapter()
        chat = "oc_" + "5" * 20
        result = await adapter.send(chat, CRON_FAILURE, metadata={"job_id": "j1", "notify": True})
        assert result.success is True
        assert lark.sent_cards, "cron results send a standalone card message"
        assert lark.created_cards, "the card entity is created first"
        import json
        # sent message carries a card_id reference; the schema lives on the create call
        content = json.loads(lark.created_cards[-1].request_body.data)
        assert content["header"]["template"] == "red", "failure notices render red"
        assert "sentinel" in content["header"]["title"]["content"]

    asyncio.run(_t())


def test_cron_success_renders_blue_card_unwrapped() -> None:
    async def _t() -> None:
        adapter, lark = make_streaming_adapter()
        chat = "oc_" + "6" * 20
        wrapped = (
            "Cronjob Response: daily digest\n"
            "(job_id: j2)\n"
            "-------------\n\n"
            "Here is your digest.\n"
            '\n\nTo stop or manage this job, send me a new message (e.g. "stop reminder daily digest").'
        )
        result = await adapter.send(chat, wrapped, metadata={"job_id": "j2", "notify": True})
        assert result.success is True
        import json
        content = json.loads(lark.created_cards[-1].request_body.data)
        assert content["header"]["template"] == "blue"
        body = "".join(e.get("content", "") for e in content["body"]["elements"]
                       if isinstance(e, dict) and e.get("tag") == "markdown")
        assert "Here is your digest." in body
        assert "Cronjob Response:" not in body, "envelope header is unwrapped into the card header"

    asyncio.run(_t())


# --------------------------------------------------------------------------- #
# Topic isolation
# --------------------------------------------------------------------------- #

def test_topic_session_defers_card_until_anchor_and_replies_in_thread() -> None:
    async def _t() -> None:
        adapter, lark = make_streaming_adapter()
        chat = "oc_" + "7" * 20
        topic = "omt_" + "t" * 16

        adapter.supports_draft_streaming(chat_id=chat, metadata={"thread_id": topic})
        await _settle(2)
        assert len(lark.created_cards) == 0, "topic card creation waits for the reply anchor"

        await adapter.send_draft(chat, 1, "topic body",
                                 {"reply_to_message_id": "om_topic_user", "thread_id": topic})
        await _settle()
        assert len(lark.created_cards) == 1
        assert lark.replied_to, "anchored creation replies inside the thread"

        final = await adapter.send(chat, "topic body final", metadata={"thread_id": topic})
        assert final.success is True and final.message_id == "om_reply"

    asyncio.run(_t())


def test_main_and_topic_sessions_isolated() -> None:
    async def _t() -> None:
        adapter, lark = make_streaming_adapter()
        chat = "oc_" + "8" * 20
        topic = "omt_" + "u" * 16

        adapter.supports_draft_streaming(chat_id=chat, metadata={})
        adapter.supports_draft_streaming(chat_id=chat, metadata={"thread_id": topic})
        await _settle(2)
        engine = adapter._card_engine()
        assert len(engine._sessions) == 2

        await adapter.send_draft(chat, 1, "main body", {"reply_to_message_id": "om_main"})
        await adapter.send_draft(chat, 2, "topic body",
                                 {"reply_to_message_id": "om_topic", "thread_id": topic})
        await _settle()
        assert engine.session_for(chat).answer_seg.text == "main body"
        assert engine.session_for(chat, topic).answer_seg.text == "topic body"

    asyncio.run(_t())
