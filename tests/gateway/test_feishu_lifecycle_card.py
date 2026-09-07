"""Lifecycle status card: one interactive card per turn, patched in place."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from gateway.platforms.base import ProcessingOutcome
from plugins.platforms.feishu.lifecycle_card import (
    LifecycleCard,
    make_title,
    split_entry_ref,
    to_lark_md,
)


def _ok_response(message_id="om_card1"):
    return SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(message_id=message_id),
        code=0,
        msg="ok",
    )


CHAT_KEY = "oc_1\x00om_ask"


def make_lifecycle_adapter():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(patch=lambda req: _ok_response())))
    )
    adapter._lifecycle_cards = {}
    adapter._lifecycle_card_chats = {}
    adapter._lifecycle_title_tasks = set()
    adapter._spawn_lifecycle_title = lambda *a, **k: None
    adapter._lifecycle_lock = asyncio.Lock()
    adapter._lifecycle_cards_enabled = True
    adapter._sends = []
    adapter._patches = []

    async def _fake_send(**kwargs):
        adapter._sends.append(kwargs)
        return _ok_response()

    async def _fake_run_blocking(fn, request):
        adapter._patches.append(request)
        return _ok_response()

    adapter._feishu_send_with_retry = _fake_send
    adapter._run_blocking = _fake_run_blocking
    return adapter


class TestCardModel:
    def test_split_entry_ref(self):
        assert split_entry_ref("om_x#e3") == ("om_x", 3)
        assert split_entry_ref("om_x") == ("om_x", None)
        assert split_entry_ref("om_x#enope") == ("om_x#enope", None)

    def test_state_colors(self):
        card = LifecycleCard(chat_id="oc_1", title="Check logs")
        assert card.build_card()["header"]["template"] == "blue"
        card.finalize("done")
        assert card.build_card()["header"]["template"] == "green"
        card.finalize("failed")
        assert card.build_card()["header"]["template"] == "red"

    def test_entry_cap_evicts_stalest_not_oldest(self):
        card = LifecycleCard(chat_id="oc_1")
        heartbeat_idx = card.add_entry("heartbeat v1")
        for n in range(20):
            card.update_entry(heartbeat_idx, f"heartbeat v{n}")
            card.add_entry(f"noise {n}")
        assert heartbeat_idx in card.entries
        assert "heartbeat v19" in card.entries[heartbeat_idx].text

    def test_entry_clips_to_tail(self):
        card = LifecycleCard(chat_id="oc_1")
        idx = card.add_entry("\n".join(f"line {n}" for n in range(50)))
        text = card.entries[idx].text
        assert text.startswith("…")
        assert "line 49" in text
        assert "line 0" not in text

    def test_answer_replaces_body(self):
        card = LifecycleCard(chat_id="oc_1", title="Q", requester="Test User")
        card.add_entry("working on it")
        card.finalize("done", answer="**TL;DR** all good")
        built = card.build_card()
        markdowns = [e for e in built["elements"] if e.get("tag") == "markdown"]
        assert markdowns == [{"tag": "markdown", "content": "**TL;DR** all good"}]
        note = built["elements"][-1]
        assert note["tag"] == "note"
        assert "requested by Test User" in note["elements"][0]["content"]

    def test_make_title_collapses_and_caps(self):
        assert make_title("  check   the\nlogs ") == "check the logs"
        assert len(make_title("x" * 500)) <= 80

    def test_make_title_strips_mention_markup(self):
        raw = (
            "[Mentioned: Test User (open_id=ou_1a2b3c4d)]\n\n"
            "@test\\_bot @Test User\nUpdated, with notes"
        )
        assert make_title(raw) == "Updated, with notes"
        assert make_title("@test\\_bot approve the MR then") == "approve the MR then"

    def test_to_lark_md_remaps_unrenderable_syntax(self):
        src = "## Cause\nfailing on `initiate_payment` (warn 5%)\n```py\nx = `raw`\n```\nafter"
        assert to_lark_md(src) == (
            "**Cause**\nfailing on **initiate_payment** (warn 5%)\n```py\nx = `raw`\n```\nafter"
        )

    def test_to_lark_md_flattens_pipe_tables(self):
        src = (
            "Noise:\n"
            "| Count | Share | Signature |\n"
            "|------:|------:|---|\n"
            "| 156 | 53% | iText missing PDF image |\n"
            "| 24 | | `Invoice` idempotent return |\n"
            "after"
        )
        assert to_lark_md(src) == (
            "Noise:\n"
            "**Count · Share · Signature**\n"
            "- 156 · 53% · iText missing PDF image\n"
            "- 24 · **Invoice** idempotent return\n"
            "after"
        )

    def test_to_lark_md_leaves_lone_pipe_line_alone(self):
        assert to_lark_md("| just a pipe line |") == "| just a pipe line |"

    def test_card_body_is_lark_md_with_divider_before_footer(self):
        card = LifecycleCard(chat_id="oc_1", title="Q")
        card.finalize("done", answer="see `errorCode`")
        elements = card.build_card()["elements"]
        assert elements[0] == {"tag": "markdown", "content": "see **errorCode**"}
        assert elements[1] == {"tag": "hr"}
        assert elements[-1]["tag"] == "note"

    def test_matches_anchor_accepts_thread_root(self):
        card = LifecycleCard(
            chat_id="oc_1", anchor_message_id="om_ask", thread_anchor_id="om_root"
        )
        assert card.matches_anchor("om_ask")
        assert card.matches_anchor("om_root")
        assert not card.matches_anchor("om_other")
        assert not card.matches_anchor("")
        assert not LifecycleCard(chat_id="oc_1").matches_anchor("")


class TestAdapterRouting:
    def test_interim_creates_card_then_patches(self):
        adapter = make_lifecycle_adapter()

        async def scenario():
            first = await adapter._lifecycle_interim_send("oc_1", "⏳ Working — 1 min", "om_ask", {})
            second = await adapter._lifecycle_interim_send("oc_1", "↪ Redirected", "om_ask", {})
            return first, second

        first, second = asyncio.run(scenario())
        assert first.success and first.message_id == "om_card1#e0"
        assert second.success and second.message_id == "om_card1#e1"
        assert len(adapter._sends) == 1
        assert adapter._sends[0]["msg_type"] == "interactive"
        assert len(adapter._patches) == 1

    def test_edit_entry_updates_in_place(self):
        adapter = make_lifecycle_adapter()

        async def scenario():
            sent = await adapter._lifecycle_interim_send("oc_1", "⏳ Working — 1 min", "om_ask", {})
            edited = await adapter._lifecycle_edit_entry("om_card1", 0, "⏳ Working — 4 min")
            return sent, edited

        _, edited = asyncio.run(scenario())
        assert edited.success
        card = adapter._lifecycle_cards[CHAT_KEY]
        assert "4 min" in card.entries[0].text
        assert len(adapter._sends) == 1

    def test_absorb_final_requires_anchor_match(self):
        adapter = make_lifecycle_adapter()

        async def scenario():
            async with adapter._lifecycle_lock:
                card = LifecycleCard(
                    chat_id="oc_1",
                    anchor_message_id="om_ask",
                    thread_anchor_id="om_root",
                )
                card.message_id = "om_card1"
                adapter._register_lifecycle_card(CHAT_KEY, card)
                adapter._lifecycle_card_chats["om_card1"] = CHAT_KEY
            wrong = await adapter._lifecycle_absorb_final(
                "oc_1", "answer", reply_to="om_other", metadata=None, chunk_count=1
            )
            multi = await adapter._lifecycle_absorb_final(
                "oc_1", "answer", reply_to="om_ask", metadata=None, chunk_count=2
            )
            thread = await adapter._lifecycle_absorb_final(
                "oc_1", "answer", reply_to="om_root", metadata=None, chunk_count=1
            )
            return wrong, multi, thread

        wrong, multi, right = asyncio.run(scenario())
        assert wrong is None and multi is None
        assert right.success and right.message_id == "om_card1"
        card = adapter._lifecycle_cards[CHAT_KEY]
        assert card.answer_absorbed and card.state == "done"
        patched = json.loads(adapter._patches[-1].request_body.content)
        assert patched["header"]["template"] == "green"

    def test_close_marks_outcome_and_drops_state(self):
        adapter = make_lifecycle_adapter()

        async def scenario():
            await adapter._lifecycle_interim_send("oc_1", "⏳ Working — 1 min", "om_ask", {})
            await adapter._lifecycle_close("oc_1", "om_ask", ProcessingOutcome.FAILURE)

        asyncio.run(scenario())
        assert CHAT_KEY not in adapter._lifecycle_cards
        assert not adapter._lifecycle_card_chats
        patched = json.loads(adapter._patches[-1].request_body.content)
        assert patched["header"]["template"] == "red"

    def test_close_after_absorb_skips_extra_patch(self):
        adapter = make_lifecycle_adapter()

        async def scenario():
            async with adapter._lifecycle_lock:
                card = LifecycleCard(chat_id="oc_1", anchor_message_id="om_ask")
                card.message_id = "om_card1"
                adapter._register_lifecycle_card(CHAT_KEY, card)
                adapter._lifecycle_card_chats["om_card1"] = CHAT_KEY
            await adapter._lifecycle_absorb_final(
                "oc_1", "answer", reply_to="om_ask", metadata=None, chunk_count=1
            )
            patch_count = len(adapter._patches)
            await adapter._lifecycle_close("oc_1", "om_ask", ProcessingOutcome.SUCCESS)
            return patch_count

        patch_count = asyncio.run(scenario())
        assert len(adapter._patches) == patch_count
        assert CHAT_KEY not in adapter._lifecycle_cards

    def test_failed_card_send_falls_back_to_none(self):
        adapter = make_lifecycle_adapter()

        async def _failing_send(**kwargs):
            raise RuntimeError("boom")

        adapter._feishu_send_with_retry = _failing_send
        result = asyncio.run(adapter._lifecycle_interim_send("oc_1", "⏳ Working", "om_ask", {}))
        assert result is None


class TestConcurrentTurns:
    """A second question sent mid-run is its own turn, so it gets its own
    titled card — sharing one card stranded the slower turn (live 2026-09-01:
    turn A's answer sealed the card, turn B fell out to a plain message)."""

    @staticmethod
    def _event(chat_id, message_id, text):
        return SimpleNamespace(
            source=SimpleNamespace(chat_id=chat_id),
            message_id=message_id,
            text=text,
            user_name="Test User",
            reply_to_message_id=None,
        )

    def _adapter(self):
        adapter = make_lifecycle_adapter()
        adapter._reactions_enabled = lambda: False
        sends = iter(["om_cardA", "om_cardB"])

        async def _fake_send(**kwargs):
            adapter._sends.append(kwargs)
            return _ok_response(next(sends))

        adapter._feishu_send_with_retry = _fake_send
        return adapter

    def test_each_turn_gets_its_own_card(self):
        adapter = self._adapter()

        async def scenario():
            await adapter.on_processing_start(self._event("oc_1", "om_a", "check PCS shutdown"))
            await adapter.on_processing_start(self._event("oc_1", "om_b", "do what vivek says"))
            a = await adapter._lifecycle_interim_send("oc_1", "reading", "om_a", {})
            b = await adapter._lifecycle_interim_send("oc_1", "terminal", "om_b", {})
            return a, b

        a, b = asyncio.run(scenario())
        assert a.message_id == "om_cardA#e0"
        assert b.message_id == "om_cardB#e0"
        titles = {c.title for c in adapter._lifecycle_cards.values()}
        assert titles == {"check PCS shutdown", "do what vivek says"}

    def test_first_answer_does_not_strand_the_other_turn(self):
        adapter = self._adapter()

        async def scenario():
            await adapter.on_processing_start(self._event("oc_1", "om_a", "check PCS shutdown"))
            await adapter.on_processing_start(self._event("oc_1", "om_b", "do what vivek says"))
            await adapter._lifecycle_interim_send("oc_1", "reading", "om_a", {})
            await adapter._lifecycle_interim_send("oc_1", "terminal", "om_b", {})
            first = await adapter._lifecycle_absorb_final(
                "oc_1", "answer A", reply_to="om_a", metadata=None, chunk_count=1
            )
            await adapter.on_processing_complete(
                self._event("oc_1", "om_a", "check PCS shutdown"), ProcessingOutcome.SUCCESS
            )
            second = await adapter._lifecycle_absorb_final(
                "oc_1", "answer B", reply_to="om_b", metadata=None, chunk_count=1
            )
            return first, second

        first, second = asyncio.run(scenario())
        assert first.message_id == "om_cardA"
        assert second.message_id == "om_cardB", "turn B's answer must land in turn B's card"
        assert len(adapter._sends) == 2, "no third, untitled card"
        card_b = adapter._lifecycle_cards[adapter._lifecycle_card_key("oc_1", "om_b")]
        assert card_b.answer == "answer B" and card_b.title == "do what vivek says"


class TestFastTurnStillGetsACard:
    """A turn that finishes before any status update (no tools, under the
    heartbeat threshold) still lands in a card, not a plain message.
    Live 2026-08-31 13:22: 38.2s / api_calls=2 posted as bare text."""

    def test_answer_creates_the_card_when_no_progress_ran(self):
        adapter = make_lifecycle_adapter()

        async def scenario():
            async with adapter._lifecycle_lock:
                adapter._register_lifecycle_card(
                    CHAT_KEY, LifecycleCard(chat_id="oc_1", anchor_message_id="om_ask")
                )
            return await adapter._lifecycle_absorb_final(
                "oc_1", "the answer", reply_to="om_ask", metadata=None, chunk_count=1
            )

        result = asyncio.run(scenario())
        assert result is not None and result.success
        assert result.message_id == "om_card1"
        assert len(adapter._sends) == 1
        assert adapter._sends[0]["msg_type"] == "interactive"
        payload = json.loads(adapter._sends[0]["payload"])
        assert payload["header"]["template"] == "green"
        assert "the answer" in payload["elements"][0]["content"]

    def test_send_failure_leaves_the_card_reusable(self):
        adapter = make_lifecycle_adapter()

        async def _failing_send(**kwargs):
            raise RuntimeError("boom")

        adapter._feishu_send_with_retry = _failing_send

        async def scenario():
            async with adapter._lifecycle_lock:
                adapter._register_lifecycle_card(
                    CHAT_KEY, LifecycleCard(chat_id="oc_1", anchor_message_id="om_ask")
                )
            return await adapter._lifecycle_absorb_final(
                "oc_1", "the answer", reply_to="om_ask", metadata=None, chunk_count=1
            )

        assert asyncio.run(scenario()) is None
        card = adapter._lifecycle_cards[CHAT_KEY]
        assert card.state == "working" and not card.answer_absorbed


class TestGeneratedCardTitle:
    """The card ships with the trimmed prompt, then upgrades to the auxiliary
    titler's output — which takes seconds, so it must never block the turn."""

    def test_generated_title_replaces_the_raw_prompt(self, monkeypatch):
        import agent.title_generator as tg

        adapter = make_lifecycle_adapter()
        del adapter._spawn_lifecycle_title  # exercise the real one
        monkeypatch.setattr(
            tg, "generate_title", lambda *a, **k: "Count August pending by payment method"
        )

        async def scenario():
            card = LifecycleCard(chat_id="oc_1", title="can you also show me how many of…")
            async with adapter._lifecycle_lock:
                adapter._register_lifecycle_card(CHAT_KEY, card)
            await adapter._lifecycle_interim_send("oc_1", "working", "om_ask", {})
            await adapter._apply_lifecycle_title(card, "can you also show me how many…")
            return card

        card = asyncio.run(scenario())
        assert card.title == "Count August pending by payment method"
        assert json.loads(adapter._sends[0]["payload"])["header"]["title"]["content"].startswith("⏳ can you")
        assert len(adapter._patches) == 1

    def test_title_is_dropped_when_the_card_was_replaced(self, monkeypatch):
        import agent.title_generator as tg

        adapter = make_lifecycle_adapter()
        monkeypatch.setattr(tg, "generate_title", lambda *a, **k: "A better title")

        async def scenario():
            stale = LifecycleCard(chat_id="oc_1", title="old")
            async with adapter._lifecycle_lock:
                adapter._register_lifecycle_card(CHAT_KEY, LifecycleCard(chat_id="oc_1", title="live"))
            await adapter._apply_lifecycle_title(stale, "prompt")
            return stale, adapter._lifecycle_cards[CHAT_KEY]

        stale, live = asyncio.run(scenario())
        assert stale.title == "old" and live.title == "live"

    def test_titler_failure_keeps_the_raw_prompt(self, monkeypatch):
        import agent.title_generator as tg

        adapter = make_lifecycle_adapter()

        def _boom(*a, **k):
            raise RuntimeError("no provider configured")

        monkeypatch.setattr(tg, "generate_title", _boom)

        async def scenario():
            card = LifecycleCard(chat_id="oc_1", title="raw prompt")
            async with adapter._lifecycle_lock:
                adapter._register_lifecycle_card(CHAT_KEY, card)
            await adapter._apply_lifecycle_title(card, "prompt")
            return card

        assert asyncio.run(scenario()).title == "raw prompt"
