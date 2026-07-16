"""Tests for Telegram Business Mode (Secretary Bots).

Covers two layers:
  - ``hermes_state.SessionDB`` business-connection + draft CRUD
  - ``plugins.platforms.telegram.telegram_business.BusinessModeManager`` orchestration

The manager tests use lightweight async stubs for the adapter callbacks
(``send_message``, ``draft_generator``) so the agent loop / network are
never touched — pure unit-level coverage of the state machine.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Fresh SessionDB with the business migration already applied."""
    db_path = tmp_path / "biz_state.db"
    sdb = SessionDB(db_path=db_path)
    sdb.apply_telegram_business_migration()
    yield sdb
    sdb.close()


# =========================================================================
# Schema / CRUD
# =========================================================================


class TestBusinessConnectionPersistence:
    def test_upsert_and_get(self, db):
        db.upsert_telegram_business_connection(
            connection_id="c1", owner_user_id="42",
            owner_chat_id="100", can_reply=True, is_enabled=True,
        )
        row = db.get_telegram_business_connection("c1")
        assert row is not None
        assert row["connection_id"] == "c1"
        assert row["owner_user_id"] == "42"
        assert row["owner_chat_id"] == "100"
        assert row["can_reply"] is True
        assert row["is_enabled"] is True
        assert row["auto_draft"] is True
        assert row["paused_chats"] == []

    def test_upsert_preserves_auto_draft_and_paused(self, db):
        db.upsert_telegram_business_connection(
            connection_id="c1", owner_user_id="42",
            owner_chat_id="100", can_reply=True, is_enabled=True,
        )
        db.set_telegram_business_auto_draft("c1", auto_draft=False)
        db.set_telegram_business_paused_chats("c1", ["200", "300"])
        # Simulate Telegram re-sending the BusinessConnection
        db.upsert_telegram_business_connection(
            connection_id="c1", owner_user_id="42",
            owner_chat_id="100", can_reply=False, is_enabled=True,
        )
        row = db.get_telegram_business_connection("c1")
        # can_reply updated, but owner preferences kept
        assert row["can_reply"] is False
        assert row["auto_draft"] is False
        assert sorted(row["paused_chats"]) == ["200", "300"]

    def test_list_enabled_filter(self, db):
        db.upsert_telegram_business_connection(
            connection_id="c1", owner_user_id="42",
            owner_chat_id="100", can_reply=True, is_enabled=True,
        )
        db.upsert_telegram_business_connection(
            connection_id="c2", owner_user_id="42",
            owner_chat_id="100", can_reply=False, is_enabled=False,
        )
        active = db.list_telegram_business_connections(
            owner_user_id="42", enabled_only=True,
        )
        all_rows = db.list_telegram_business_connections(
            owner_user_id="42", enabled_only=False,
        )
        assert len(active) == 1 and active[0]["connection_id"] == "c1"
        assert len(all_rows) == 2

    def test_get_returns_none_for_unknown(self, db):
        assert db.get_telegram_business_connection("missing") is None

    def test_migration_idempotent(self, db):
        # Calling twice must not raise; subsequent CRUD still works.
        db.apply_telegram_business_migration()
        db.apply_telegram_business_migration()
        db.upsert_telegram_business_connection(
            connection_id="c1", owner_user_id="42",
            owner_chat_id="100", can_reply=True, is_enabled=True,
        )
        assert db.get_telegram_business_connection("c1") is not None


class TestBusinessDraftLifecycle:
    @pytest.fixture(autouse=True)
    def _conn(self, db):
        db.upsert_telegram_business_connection(
            connection_id="c1", owner_user_id="42",
            owner_chat_id="100", can_reply=True, is_enabled=True,
        )

    def test_create_and_get(self, db):
        did = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="200",
            customer_msg_id="m1", customer_text="hi", draft_text="hello",
        )
        row = db.get_telegram_business_draft(did)
        assert row is not None
        assert row["customer_text"] == "hi"
        assert row["draft_text"] == "hello"
        assert row["status"] == "pending"
        assert row["owner_message_id"] is None
        assert row["final_sent_text"] is None
        assert row["expires_at"] > row["created_at"]

    def test_new_draft_supersedes_prior_pending(self, db):
        d1 = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="200",
            customer_msg_id="m1", customer_text="hi", draft_text="hello",
        )
        d2 = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="200",
            customer_msg_id="m2", customer_text="hi again", draft_text="hello again",
        )
        assert d1 != d2
        # Prior draft should be marked superseded.
        prior = db.get_telegram_business_draft(d1)
        new = db.get_telegram_business_draft(d2)
        assert prior["status"] == "superseded"
        assert new["status"] == "pending"

    def test_supersede_only_affects_same_customer_chat(self, db):
        d1 = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="200",
            customer_msg_id="m1", customer_text="A", draft_text="a",
        )
        d2 = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="999",
            customer_msg_id="m2", customer_text="B", draft_text="b",
        )
        # Different customer chats — d1 should still be pending.
        assert db.get_telegram_business_draft(d1)["status"] == "pending"
        assert db.get_telegram_business_draft(d2)["status"] == "pending"

    def test_resolve_sent_returns_prior_and_blocks_double_resolve(self, db):
        did = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="200",
            customer_msg_id="m1", customer_text="hi", draft_text="hello",
        )
        first = db.resolve_telegram_business_draft(
            did, status="sent", final_sent_text="hello",
        )
        assert first is not None
        # Status now committed
        row = db.get_telegram_business_draft(did)
        assert row["status"] == "sent"
        assert row["final_sent_text"] == "hello"
        # Second resolution must be a no-op.
        second = db.resolve_telegram_business_draft(did, status="discarded")
        assert second is None
        assert db.get_telegram_business_draft(did)["status"] == "sent"

    def test_resolve_invalid_status_raises(self, db):
        did = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="200",
            customer_msg_id="m1", customer_text="hi", draft_text="hello",
        )
        with pytest.raises(ValueError):
            db.resolve_telegram_business_draft(did, status="bogus")

    def test_owner_message_id_round_trip(self, db):
        did = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="200",
            customer_msg_id="m1", customer_text="hi", draft_text="hello",
        )
        db.set_telegram_business_draft_owner_message(did, "owner_msg_42")
        row = db.get_telegram_business_draft(did)
        assert row["owner_message_id"] == "owner_msg_42"

    def test_expire_only_affects_overdue_pending(self, db):
        d_old = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="200",
            customer_msg_id="m_old", customer_text="old", draft_text="old reply",
            ttl_seconds=60.0,
        )
        # Different customer chat so it doesn't supersede d_old.
        d_new = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="201",
            customer_msg_id="m_new", customer_text="new", draft_text="new reply",
            ttl_seconds=999_999.0,
        )
        # Run expiry just past d_old's expiry but well before d_new's.
        d_old_row = db.get_telegram_business_draft(d_old)
        affected = db.expire_telegram_business_drafts(now=d_old_row["expires_at"] + 1.0)
        assert affected == 1
        assert db.get_telegram_business_draft(d_old)["status"] == "expired"
        assert db.get_telegram_business_draft(d_new)["status"] == "pending"

    def test_get_pending_for_owner(self, db):
        did = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="200",
            customer_msg_id="m1", customer_text="hi", draft_text="hello",
        )
        # Resolved draft must not appear.
        d2 = db.create_telegram_business_draft(
            connection_id="c1", owner_chat_id="100", customer_chat_id="999",
            customer_msg_id="m2", customer_text="hi2", draft_text="hello2",
        )
        db.resolve_telegram_business_draft(d2, status="discarded")
        pending = db.get_pending_telegram_business_drafts_for_owner("100")
        ids = {row["draft_id"] for row in pending}
        assert did in ids
        assert d2 not in ids


# =========================================================================
# Manager orchestration
# =========================================================================


def _fake_business_connection(
    *, conn_id="conn1", owner_id=42, owner_chat=100,
    is_enabled=True, can_reply=True,
) -> Any:
    """Build a duck-typed BusinessConnection good enough for the manager."""
    rights = SimpleNamespace(can_reply=can_reply) if can_reply is not None else None
    return SimpleNamespace(
        id=conn_id,
        user=SimpleNamespace(id=owner_id, full_name="Owner"),
        user_chat_id=owner_chat,
        is_enabled=is_enabled,
        rights=rights,
    )


def _fake_business_message(
    *, conn_id="conn1", customer_chat_id=200, customer_id=999,
    text="Hi there", msg_id=42,
) -> Any:
    """Build a duck-typed business_message-style PTB Message."""
    return SimpleNamespace(
        business_connection_id=conn_id,
        chat=SimpleNamespace(id=customer_chat_id, type="private"),
        from_user=SimpleNamespace(
            id=customer_id, full_name="Customer Carol",
            first_name="Carol", username="carol",
        ),
        text=text,
        caption=None,
        message_id=msg_id,
    )


class _SentRecorder:
    """Capture all send_message kwargs the manager produces."""

    def __init__(self, *, fail: bool = False):
        self.calls: List[Dict[str, Any]] = []
        self.fail = fail
        self._next_id = 1000

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("simulated send failure")
        sent = SimpleNamespace(message_id=self._next_id)
        self._next_id += 1
        return sent


def _make_manager(db, *, draft_text="Draft reply!", draft_fails=False,
                  send_recorder=None, debounce=0.0):
    from plugins.platforms.telegram.telegram_business import BusinessModeManager

    async def _draft(customer_text: str, customer_chat_id: str) -> str:
        if draft_fails:
            raise RuntimeError("model boom")
        return draft_text

    sender = send_recorder if send_recorder is not None else _SentRecorder()
    mgr = BusinessModeManager(
        session_db=db,
        send_message=sender,
        draft_generator=_draft,
        debounce_seconds=debounce,
    )
    return mgr, sender


class TestConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_first_connection_persists_and_sends_onboarding(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        row = db.get_telegram_business_connection("conn1")
        assert row is not None
        assert row["is_enabled"] is True
        assert row["can_reply"] is True
        # Owner got the onboarding DM
        assert len(sender.calls) == 1
        assert sender.calls[0]["chat_id"] == 100
        assert "Business assistant" in sender.calls[0]["text"]

    @pytest.mark.asyncio
    async def test_repeat_connection_does_not_resend_onboarding(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        sender.calls.clear()
        await mgr.handle_connection_update(_fake_business_connection())
        assert sender.calls == []

    @pytest.mark.asyncio
    async def test_disconnection_dms_owner(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        sender.calls.clear()
        await mgr.handle_connection_update(
            _fake_business_connection(is_enabled=False)
        )
        assert any("ended" in c["text"].lower() for c in sender.calls)

    @pytest.mark.asyncio
    async def test_can_reply_toggle_notifies_owner(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection(can_reply=True))
        sender.calls.clear()
        await mgr.handle_connection_update(_fake_business_connection(can_reply=False))
        assert sender.calls, "owner should be told send permission flipped"
        assert "OFF" in sender.calls[0]["text"]


class TestBusinessMessageDraftFlow:
    @pytest.mark.asyncio
    async def test_incoming_customer_message_drafts_and_dms_owner(self, db):
        mgr, sender = _make_manager(db, draft_text="Hey, thanks for reaching out!")
        await mgr.handle_connection_update(_fake_business_connection())
        sender.calls.clear()
        await mgr.handle_business_message(_fake_business_message(text="Hi!"))
        assert sender.calls, "draft should have been DM'd to the owner"
        owner_msg = sender.calls[0]
        assert owner_msg["chat_id"] == 100
        assert "Hey, thanks for reaching out!" in owner_msg["text"]
        assert "Hi!" in owner_msg["text"]
        # Inline keyboard rendered (under the test telegram mock,
        # InlineKeyboardMarkup is a MagicMock — we just verify a keyboard
        # was attached and that three callback_datas were produced).
        assert owner_msg.get("reply_markup") is not None
        # One draft row recorded
        drafts = db.get_pending_telegram_business_drafts_for_owner("100")
        assert len(drafts) == 1
        assert drafts[0]["draft_text"] == "Hey, thanks for reaching out!"

    @pytest.mark.asyncio
    async def test_no_can_reply_drops_send_button(self, db, monkeypatch):
        # Capture callback_data passed to InlineKeyboardButton to verify
        # the Send button is omitted when can_reply is False.  We wrap
        # _build_draft_keyboard so we can inspect the choices it produced
        # — the underlying telegram mock makes the resulting keyboard
        # object opaque.
        from plugins.platforms.telegram import telegram_business as biz_mod

        original = biz_mod.BusinessModeManager._build_draft_keyboard
        captured: List[str] = []

        def _spy(draft_id, *, can_reply):
            if can_reply:
                captured.append("send")
            captured.append("edit")
            captured.append("discard")
            return original(draft_id, can_reply=can_reply)

        monkeypatch.setattr(
            biz_mod.BusinessModeManager,
            "_build_draft_keyboard",
            staticmethod(_spy),
        )

        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection(can_reply=False))
        sender.calls.clear()
        await mgr.handle_business_message(_fake_business_message())
        assert captured == ["edit", "discard"]

    @pytest.mark.asyncio
    async def test_unknown_connection_silently_ignored(self, db):
        mgr, sender = _make_manager(db)
        # No prior handle_connection_update — should silently skip.
        await mgr.handle_business_message(_fake_business_message())
        assert sender.calls == []
        assert db.get_pending_telegram_business_drafts_for_owner("100") == []

    @pytest.mark.asyncio
    async def test_auto_draft_paused_skips_drafting(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        db.set_telegram_business_auto_draft("conn1", auto_draft=False)
        sender.calls.clear()
        await mgr.handle_business_message(_fake_business_message())
        assert sender.calls == []
        assert db.get_pending_telegram_business_drafts_for_owner("100") == []

    @pytest.mark.asyncio
    async def test_paused_customer_chat_skipped(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        db.set_telegram_business_paused_chats("conn1", ["200"])
        sender.calls.clear()
        await mgr.handle_business_message(_fake_business_message(customer_chat_id=200))
        assert sender.calls == []
        # Other chat still drafts.
        await mgr.handle_business_message(_fake_business_message(customer_chat_id=300))
        assert sender.calls

    @pytest.mark.asyncio
    async def test_empty_text_skipped(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        sender.calls.clear()
        await mgr.handle_business_message(_fake_business_message(text=""))
        assert sender.calls == []

    @pytest.mark.asyncio
    async def test_draft_failure_reports_to_owner(self, db):
        mgr, sender = _make_manager(db, draft_fails=True)
        await mgr.handle_connection_update(_fake_business_connection())
        sender.calls.clear()
        await mgr.handle_business_message(_fake_business_message(text="Hi"))
        assert sender.calls
        assert "couldn't draft" in sender.calls[0]["text"]

    @pytest.mark.asyncio
    async def test_debounce_coalesces_burst(self, db):
        # Use a real debounce window and fire 3 messages in quick succession.
        mgr, sender = _make_manager(db, debounce=0.05,
                                    draft_text="single draft")
        await mgr.handle_connection_update(_fake_business_connection())
        sender.calls.clear()
        for i in range(3):
            await mgr.handle_business_message(
                _fake_business_message(text=f"part {i}", msg_id=i)
            )
        # Let the debounce fire.
        await asyncio.sleep(0.2)
        # Only one draft should have been generated and one owner DM sent.
        owner_dms = [c for c in sender.calls if c.get("chat_id") == 100]
        assert len(owner_dms) == 1


class TestCallbackDispatch:
    @pytest.mark.asyncio
    async def test_send_button_delivers_to_customer_chat(self, db):
        mgr, sender = _make_manager(db, draft_text="hello there")
        await mgr.handle_connection_update(_fake_business_connection())
        await mgr.handle_business_message(_fake_business_message())
        sender.calls.clear()
        draft = db.get_pending_telegram_business_drafts_for_owner("100")[0]
        did = draft["draft_id"]
        # Fake the inline-button click
        answered: List[Dict[str, Any]] = []
        edited: List[Dict[str, Any]] = []

        async def _answer(**kw): answered.append(kw)
        async def _edit(**kw): edited.append(kw)

        dispatched = await mgr.handle_callback(
            data=f"bd:send:{did}", caller_user_id="42",
            answer=_answer, edit_message_text=_edit,
        )
        assert dispatched is True
        # Sent to customer chat with business_connection_id
        sends_to_customer = [
            c for c in sender.calls if c.get("chat_id") == 200
        ]
        assert sends_to_customer
        assert sends_to_customer[0]["business_connection_id"] == "conn1"
        assert sends_to_customer[0]["text"] == "hello there"
        # Draft now sent
        row = db.get_telegram_business_draft(did)
        assert row["status"] == "sent"
        assert row["final_sent_text"] == "hello there"
        # Owner DM was edited to show resolution
        assert edited and "Sent" in edited[0]["text"]

    @pytest.mark.asyncio
    async def test_discard_marks_status(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        await mgr.handle_business_message(_fake_business_message())
        draft = db.get_pending_telegram_business_drafts_for_owner("100")[0]
        did = draft["draft_id"]

        async def _answer(**kw): pass
        async def _edit(**kw): pass

        await mgr.handle_callback(
            data=f"bd:discard:{did}", caller_user_id="42",
            answer=_answer, edit_message_text=_edit,
        )
        assert db.get_telegram_business_draft(did)["status"] == "discarded"

    @pytest.mark.asyncio
    async def test_callback_for_unknown_draft_no_ops(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        sender.calls.clear()

        answered: List[Dict[str, Any]] = []

        async def _answer(**kw): answered.append(kw)
        async def _edit(**kw): pass

        await mgr.handle_callback(
            data="bd:send:999999", caller_user_id="42",
            answer=_answer, edit_message_text=_edit,
        )
        assert answered and "expired" in answered[0]["text"].lower()
        # No customer-chat sends.
        assert not any(c.get("chat_id") == 200 for c in sender.calls)

    @pytest.mark.asyncio
    async def test_callback_rejects_non_owner(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        await mgr.handle_business_message(_fake_business_message())
        draft = db.get_pending_telegram_business_drafts_for_owner("100")[0]
        did = draft["draft_id"]
        answered: List[Dict[str, Any]] = []

        async def _answer(**kw): answered.append(kw)
        async def _edit(**kw): pass

        # Different user — should be rejected.
        await mgr.handle_callback(
            data=f"bd:send:{did}", caller_user_id="9999",
            answer=_answer, edit_message_text=_edit,
        )
        assert answered and "Only the connected account owner" in answered[0]["text"]
        assert db.get_telegram_business_draft(did)["status"] == "pending"

    @pytest.mark.asyncio
    async def test_send_blocked_when_can_reply_false(self, db):
        mgr, sender = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection(can_reply=False))
        await mgr.handle_business_message(_fake_business_message())
        draft = db.get_pending_telegram_business_drafts_for_owner("100")[0]
        did = draft["draft_id"]
        answered: List[Dict[str, Any]] = []

        async def _answer(**kw): answered.append(kw)
        async def _edit(**kw): pass

        await mgr.handle_callback(
            data=f"bd:send:{did}", caller_user_id="42",
            answer=_answer, edit_message_text=_edit,
        )
        # Send-on-your-behalf is OFF → reject with explanation, no customer send.
        assert any("send-on-your-behalf is off" in (a.get("text") or "").lower()
                   for a in answered)
        assert not any(c.get("chat_id") == 200 for c in sender.calls)
        assert db.get_telegram_business_draft(did)["status"] == "pending"

    @pytest.mark.asyncio
    async def test_non_bd_callback_returns_false(self, db):
        mgr, _ = _make_manager(db)

        async def _noop(**kw):
            pass

        dispatched = await mgr.handle_callback(
            data="ea:once:1", caller_user_id="42",
            answer=_noop, edit_message_text=_noop,
        )
        assert dispatched is False


class TestEditCapture:
    @pytest.mark.asyncio
    async def test_edit_then_text_sends_override(self, db):
        mgr, sender = _make_manager(db, draft_text="original draft")
        await mgr.handle_connection_update(_fake_business_connection())
        await mgr.handle_business_message(_fake_business_message())
        draft = db.get_pending_telegram_business_drafts_for_owner("100")[0]
        did = draft["draft_id"]
        # Tap Edit
        async def _answer(**kw): pass
        async def _edit(**kw): pass

        await mgr.handle_callback(
            data=f"bd:edit:{did}", caller_user_id="42",
            answer=_answer, edit_message_text=_edit,
        )
        sender.calls.clear()
        # Owner sends override text
        consumed = await mgr.maybe_handle_edit_capture(
            owner_chat_id="100", text="my custom reply",
        )
        assert consumed is True
        # Customer received the override
        customer_sends = [c for c in sender.calls if c.get("chat_id") == 200]
        assert customer_sends
        assert customer_sends[0]["text"] == "my custom reply"
        assert customer_sends[0]["business_connection_id"] == "conn1"
        # Draft now resolved as edited
        row = db.get_telegram_business_draft(did)
        assert row["status"] == "edited"
        assert row["final_sent_text"] == "my custom reply"

    @pytest.mark.asyncio
    async def test_edit_capture_idle_returns_false(self, db):
        mgr, _ = _make_manager(db)
        consumed = await mgr.maybe_handle_edit_capture(
            owner_chat_id="100", text="hello",
        )
        assert consumed is False


class TestBizCommand:
    @pytest.mark.asyncio
    async def test_status_with_no_connections(self, db):
        mgr, _ = _make_manager(db)
        reply = await mgr.handle_biz_command(
            owner_user_id="42", owner_chat_id="100", args=[],
        )
        assert "haven't connected" in reply.lower()

    @pytest.mark.asyncio
    async def test_status_with_connection(self, db):
        mgr, _ = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        reply = await mgr.handle_biz_command(
            owner_user_id="42", owner_chat_id="100", args=[],
        )
        assert "active" in reply
        assert "drafting ON" in reply

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, db):
        mgr, _ = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        reply = await mgr.handle_biz_command(
            owner_user_id="42", owner_chat_id="100", args=["pause"],
        )
        assert "Paused" in reply
        assert db.get_telegram_business_connection("conn1")["auto_draft"] is False
        reply = await mgr.handle_biz_command(
            owner_user_id="42", owner_chat_id="100", args=["resume"],
        )
        assert "Resumed" in reply
        assert db.get_telegram_business_connection("conn1")["auto_draft"] is True

    @pytest.mark.asyncio
    async def test_per_chat_off_and_on(self, db):
        mgr, _ = _make_manager(db)
        await mgr.handle_connection_update(_fake_business_connection())
        reply = await mgr.handle_biz_command(
            owner_user_id="42", owner_chat_id="100", args=["off", "200"],
        )
        assert "200" in reply
        assert "200" in db.get_telegram_business_connection("conn1")["paused_chats"]
        reply = await mgr.handle_biz_command(
            owner_user_id="42", owner_chat_id="100", args=["on", "200"],
        )
        assert "200" in reply
        assert "200" not in db.get_telegram_business_connection("conn1")["paused_chats"]

    @pytest.mark.asyncio
    async def test_unknown_subcommand_returns_help(self, db):
        mgr, _ = _make_manager(db)
        reply = await mgr.handle_biz_command(
            owner_user_id="42", owner_chat_id="100", args=["unknown"],
        )
        assert "Usage" in reply
