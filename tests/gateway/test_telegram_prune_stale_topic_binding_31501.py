"""Regression tests for #31501 — prune stale Telegram DM topic bindings.

When a Telegram user deletes a DM topic in the client, the Bot API
responds to the gateway's next send with ``Thread not found``.  The
adapter falls back to a plain send (no ``message_thread_id``), but
prior to this fix it left the corresponding row in
``telegram_dm_topic_bindings`` untouched.
``gateway.run._recover_telegram_topic_thread_id`` then walked the
user's bindings newest-first on every later inbound message and
cheerfully redirected them back to the deleted topic — tool
progress, approvals and replies all silently landed in the wrong
place until the operator manually ran ``DELETE`` on ``state.db``.

The fix has three pieces — these tests pin all three:

1. ``SessionDB.delete_telegram_topic_binding`` — the targeted
   prune helper (new public API).
2. ``TelegramAdapter._prune_stale_dm_topic_binding`` — the
   adapter glue that calls the helper from a send-fallback hot
   path without raising on cleanup failure.
3. The two "Thread not found" call sites in the streaming send
   loop and the control-message helper now invoke (2) — we pin
   this with a source-level guard rather than spinning the full
   send pipeline.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# SessionDB.delete_telegram_topic_binding
# ---------------------------------------------------------------------------


def _seed_binding(
    db: SessionDB,
    *,
    chat_id: str = "5595856929",
    thread_id: str = "15287",
    user_id: str = "5595856929",
    session_id: str = "sess-target",
) -> None:
    db.create_session(
        session_id=session_id,
        source="telegram",
        user_id=user_id,
    )
    db.bind_telegram_topic(
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=user_id,
        session_key=f"agent:main:telegram:dm:{chat_id}:{thread_id}",
        session_id=session_id,
    )


class TestDeleteTelegramTopicBinding:
    def test_removes_matching_row_and_returns_count(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed_binding(db, thread_id="15287")
        # Sanity check — binding present before prune.
        assert db.get_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        ) is not None

        removed = db.delete_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        )

        assert removed == 1
        assert db.get_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        ) is None
        db.close()

    def test_does_not_touch_unrelated_bindings(self, tmp_path):
        # Critical for the fix: a chat with multiple topics must
        # only lose the one Telegram confirmed deleted, never the
        # rest.  Otherwise the user's healthy topics also vanish
        # from recovery's view.
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed_binding(db, thread_id="15287", session_id="sess-stale")
        _seed_binding(db, thread_id="15418", session_id="sess-fresh")

        removed = db.delete_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        )
        assert removed == 1

        # Stale binding is gone; the fresh one survives.
        assert db.get_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        ) is None
        assert db.get_telegram_topic_binding(
            chat_id="5595856929", thread_id="15418",
        ) is not None
        db.close()

    def test_missing_row_returns_zero_silently(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed_binding(db, thread_id="15287")

        # Different thread_id — must not raise, just report 0.
        removed = db.delete_telegram_topic_binding(
            chat_id="5595856929", thread_id="99999",
        )
        assert removed == 0
        # Original binding still intact.
        assert db.get_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        ) is not None
        db.close()

    def test_pristine_database_with_no_topic_tables_is_silent_noop(self, tmp_path):
        # Fresh profile that has never run /topic — the topic-mode
        # tables don't exist yet.  The send-fallback hot path can
        # still hit this code, so we must not crash.
        db = SessionDB(db_path=tmp_path / "state.db")
        # Confirm precondition: tables really aren't there.
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'telegram_dm%'"
            ).fetchall()
        }
        assert "telegram_dm_topic_bindings" not in tables

        removed = db.delete_telegram_topic_binding(
            chat_id="any", thread_id="any",
        )
        assert removed == 0
        db.close()

    def test_idempotent_under_repeated_calls(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed_binding(db, thread_id="15287")

        first = db.delete_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        )
        second = db.delete_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        )

        assert first == 1
        assert second == 0  # already gone, no spurious "1"
        db.close()


# ---------------------------------------------------------------------------
# Adapter glue — _prune_stale_dm_topic_binding
# ---------------------------------------------------------------------------


def _bare_adapter(db: SessionDB | None = None):
    # The adapter accesses the SessionDB via
    # ``self._session_store._db`` (set by GatewayRunner via
    # ``set_session_store``).  Build a minimal stand-in with just
    # the surface the prune helper touches; we don't need the
    # python-telegram-bot import-graph here.  ``name`` is a
    # property that delegates to ``platform.value.title()``, so
    # we set ``platform`` rather than poking ``name`` directly.
    from gateway.config import Platform
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    if db is not None:
        adapter._session_store = SimpleNamespace(_db=db)
    return adapter


class TestPruneStaleDmTopicBindingHelper:
    def test_drops_binding_when_session_store_db_is_present(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed_binding(db, thread_id="15287")

        adapter = _bare_adapter(db)
        adapter._prune_stale_dm_topic_binding("5595856929", 15287)

        assert db.get_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        ) is None
        db.close()

    def test_silent_when_session_store_unavailable(self):
        # No ``_session_store`` attribute — the helper must not
        # explode (the streaming send path hits this in tests
        # that bypass the gateway runner).
        adapter = _bare_adapter()
        adapter._prune_stale_dm_topic_binding("123", "456")

    def test_silent_when_db_lacks_helper(self):
        # Old SessionDB without the new method (e.g. running
        # against an older state.db schema).  Must be a no-op
        # rather than AttributeError.
        adapter = _bare_adapter()
        adapter._session_store = SimpleNamespace(
            _db=SimpleNamespace(),  # no methods at all
        )
        adapter._prune_stale_dm_topic_binding("123", "456")

    def test_swallows_db_exceptions_so_send_continues(self):
        class ExplodingDb:
            def delete_telegram_topic_binding(self, **_):
                raise RuntimeError("disk full or whatever")

        adapter = _bare_adapter()
        adapter._session_store = SimpleNamespace(_db=ExplodingDb())

        # The point of the helper is that a failed cleanup must
        # NEVER turn into a failed user-facing send.  No exception
        # should escape.
        adapter._prune_stale_dm_topic_binding("123", "456")

    def test_skips_when_chat_or_thread_missing(self, tmp_path):
        # Defensive — control-message paths sometimes call us
        # with chat_id=None when kwargs lack the key.  We must
        # not produce a spurious DELETE that matches every row
        # with a NULL chat_id.
        db = SessionDB(db_path=tmp_path / "state.db")
        _seed_binding(db, thread_id="15287")

        adapter = _bare_adapter(db)

        adapter._prune_stale_dm_topic_binding(None, "15287")
        adapter._prune_stale_dm_topic_binding("5595856929", None)

        # Still there — neither call generated a DELETE.
        assert db.get_telegram_topic_binding(
            chat_id="5595856929", thread_id="15287",
        ) is not None
        db.close()


# ---------------------------------------------------------------------------
# Source-level wiring guards — both fallback sites must call the helper
# ---------------------------------------------------------------------------


class TestThreadNotFoundFallbackSitesPruneBinding:
    """Pin that the two ``Thread not found`` warning sites in the
    Telegram adapter actually invoke ``_prune_stale_dm_topic_binding``.
    These guards stop a future refactor from quietly losing the
    cleanup wire — re-opening #31501.
    """

    def test_streaming_send_fallback_calls_prune(self):
        from plugins.platforms.telegram import adapter as telegram_mod

        src = inspect.getsource(telegram_mod.TelegramAdapter.send)
        # Locate the second-failure branch (the one that flips
        # ``used_thread_fallback``).  It must invoke the prune
        # helper before flipping the flag.
        marker = "retrying without message_thread_id"
        idx = src.find(marker)
        assert idx != -1, (
            "Streaming send must keep its 'thread not found' "
            "fallback log line — the prune wiring is anchored "
            "next to it."
        )
        # 600 char window is enough to cover the warning, the
        # prune call, and the ``used_thread_fallback = True``
        # assignment that follows.
        window = src[idx:idx + 600]
        assert "_prune_stale_dm_topic_binding" in window, (
            "Streaming send 'Thread not found' fallback must call "
            "_prune_stale_dm_topic_binding so the stale row in "
            "telegram_dm_topic_bindings doesn't keep redirecting "
            "future inbound messages to the deleted topic (#31501)."
        )

    def test_control_message_helper_calls_prune(self):
        from plugins.platforms.telegram import adapter as telegram_mod

        src = inspect.getsource(
            telegram_mod.TelegramAdapter._send_message_with_thread_fallback
        )
        # The helper has a single retry path; the prune call
        # must sit inside it, not in dead code outside the
        # ``if message_thread_id is not None and …`` guard.
        assert "_prune_stale_dm_topic_binding" in src, (
            "_send_message_with_thread_fallback must call "
            "_prune_stale_dm_topic_binding when Telegram returns "
            "BadRequest('Thread not found') for a control message "
            "(#31501)."
        )
        # Belt-and-braces: the call must precede the retry
        # ``send_message`` so the prune happens whether or not
        # the retry itself succeeds.
        prune_idx = src.find("_prune_stale_dm_topic_binding")
        retry_idx = src.find("send_message(**retry_kwargs)")
        assert 0 <= prune_idx < retry_idx, (
            "_prune_stale_dm_topic_binding must run before the "
            "fallback send_message retry."
        )


# ---------------------------------------------------------------------------
# End-to-end semantic — prune + recovery returns None for deleted topic
# ---------------------------------------------------------------------------


class TestRecoveryAfterPrune:
    """The whole point of the fix: once a topic is pruned, the
    GatewayRunner's ``_recover_telegram_topic_thread_id`` must no
    longer steer future inbound messages to it.
    """

    def test_recovery_no_longer_returns_pruned_topic(self, tmp_path):
        # Build the same fixture used elsewhere: two topic bindings
        # for the same user, then prune the most-recent one.
        # ``_recover_telegram_topic_thread_id`` walks bindings
        # newest-first, so without the prune it would pick the
        # one we just removed.
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource, build_session_key

        db = SessionDB(db_path=tmp_path / "state.db")
        db.enable_telegram_topic_mode(
            chat_id="5595856929", user_id="5595856929",
        )

        for sid, thread in (("sess-A", "111"), ("sess-B", "222")):
            db.create_session(
                session_id=sid, source="telegram",
                user_id="5595856929",
            )
            db.bind_telegram_topic(
                chat_id="5595856929",
                thread_id=thread,
                user_id="5595856929",
                session_key=build_session_key(SessionSource(
                    platform=Platform.TELEGRAM,
                    user_id="5595856929",
                    chat_id="5595856929",
                    user_name="tester",
                    chat_type="dm",
                    thread_id=thread,
                )),
                session_id=sid,
            )

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            }
        )
        runner._session_db = db
        runner._telegram_topic_mode_enabled = lambda _src: True

        # Sanity: before the prune, recovery picks "222" (newest).
        # Recovery only fires for a lobby-shaped inbound (omitted
        # message_thread_id or General topic "1"); a non-lobby
        # unknown thread is preserved as a brand-new topic. Use the
        # General topic id so the recovery walk actually runs.
        before = runner._recover_telegram_topic_thread_id(SessionSource(
            platform=Platform.TELEGRAM,
            user_id="5595856929",
            chat_id="5595856929",
            user_name="tester",
            chat_type="dm",
            thread_id="1",  # General/stripped reply — triggers recovery
        ))
        assert before == "222"

        # User deletes topic 222 in Telegram → adapter prunes.
        db.delete_telegram_topic_binding(
            chat_id="5595856929", thread_id="222",
        )

        # Now recovery falls back to topic 111 (the surviving
        # binding) instead of the dead one.  This is the exact
        # behaviour change the bug report asks for.
        after = runner._recover_telegram_topic_thread_id(SessionSource(
            platform=Platform.TELEGRAM,
            user_id="5595856929",
            chat_id="5595856929",
            user_name="tester",
            chat_type="dm",
            thread_id="1",
        ))
        assert after == "111"
        db.close()
