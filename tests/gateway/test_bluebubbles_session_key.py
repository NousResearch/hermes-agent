"""BlueBubbles DM session-key stability across chatGuid variants.

BlueBubbles delivers the same 1:1 conversation under several chat_id forms:
the service-prefixed GUID (``iMessage;-;+1555…``, ``any;-;+1555…``) when the
webhook carries one, and the bare handle (``+1555…``) when it does not (the
``chat_identifier`` fallback in the adapter).  Keying sessions on the raw
chat_id splits one conversation across several SessionEntries; a message
landing on a long-untouched variant then trips the idle reset and wipes an
actively-used conversation.
"""

from datetime import datetime, timedelta

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import (
    SessionEntry,
    SessionSource,
    SessionStore,
    build_session_key,
    canonical_bluebubbles_identifier,
)


HANDLE = "+15551234567"


class TestCanonicalBlueBubblesIdentifier:
    @pytest.mark.parametrize(
        "raw",
        [
            f"iMessage;-;{HANDLE}",
            f"any;-;{HANDLE}",
            f"SMS;-;{HANDLE}",
            HANDLE,
        ],
    )
    def test_dm_guid_variants_collapse_to_bare_handle(self, raw):
        assert canonical_bluebubbles_identifier(raw) == HANDLE

    def test_email_handle_is_preserved(self):
        assert (
            canonical_bluebubbles_identifier("iMessage;-;user@example.com")
            == "user@example.com"
        )

    def test_group_guid_is_left_alone(self):
        """Group GUIDs use ``;+;`` and are opaque — never rewrite them."""
        guid = "iMessage;+;chat9876543210"
        assert canonical_bluebubbles_identifier(guid) == guid

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_values_pass_through(self, value):
        assert canonical_bluebubbles_identifier(value) == value


class TestBlueBubblesDMSessionKey:
    def _key(self, chat_id):
        return build_session_key(
            SessionSource(
                platform=Platform.BLUEBUBBLES,
                chat_id=chat_id,
                chat_type="dm",
                user_id=HANDLE,
            )
        )

    def test_all_dm_variants_share_one_session_key(self):
        """The regression: three forms of one conversation, one key."""
        keys = {
            self._key(f"iMessage;-;{HANDLE}"),
            self._key(f"any;-;{HANDLE}"),
            self._key(HANDLE),
        }
        assert keys == {f"agent:main:bluebubbles:dm:{HANDLE}"}

    def test_distinct_contacts_stay_isolated(self):
        assert self._key(f"any;-;{HANDLE}") != self._key("any;-;+15559999999")

    def test_group_chats_keep_their_raw_guid(self):
        """Groups must be untouched: their GUID is not a participant handle."""
        source = SessionSource(
            platform=Platform.BLUEBUBBLES,
            chat_id="iMessage;+;chat9876543210",
            chat_type="group",
            user_id=HANDLE,
        )
        key = build_session_key(source)
        assert "iMessage;+;chat9876543210" in key

    def test_threaded_dm_variants_share_one_key(self):
        def keyed(chat_id):
            return build_session_key(
                SessionSource(
                    platform=Platform.BLUEBUBBLES,
                    chat_id=chat_id,
                    chat_type="dm",
                    user_id=HANDLE,
                    thread_id="t1",
                )
            )

        assert keyed(f"any;-;{HANDLE}") == keyed(HANDLE)
        assert keyed(HANDLE) == f"agent:main:bluebubbles:dm:{HANDLE}:t1"

    def test_other_platforms_unaffected(self):
        """Only BlueBubbles is canonicalized; a lookalike id elsewhere is raw."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=f"iMessage;-;{HANDLE}",
            chat_type="dm",
            user_id="u1",
        )
        assert build_session_key(source) == (
            f"agent:main:telegram:dm:iMessage;-;{HANDLE}"
        )


class TestBlueBubblesRoutingMigration:
    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        # Each test gets its own state.db — DEFAULT_DB_PATH is module-level and
        # would otherwise be shared by every SessionDB() in this file's
        # subprocess, leaking sessions between tests.
        import hermes_state

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    @staticmethod
    def _seed(store, *, chat_id, session_id, updated_at, created_at=None, handle=HANDLE):
        """Seed one legacy route: index entry + durable session + a message."""
        source = SessionSource(
            platform=Platform.BLUEBUBBLES,
            chat_id=chat_id,
            chat_type="dm",
            user_id=handle,
        )
        session_key = f"agent:main:bluebubbles:dm:{chat_id}"
        store._entries[session_key] = SessionEntry(
            session_key=session_key,
            session_id=session_id,
            created_at=created_at or updated_at,
            updated_at=updated_at,
            origin=source,
            platform=Platform.BLUEBUBBLES,
        )
        store._db.create_session(
            session_id,
            "bluebubbles",
            user_id=handle,
            session_key=session_key,
            chat_id=chat_id,
            chat_type="dm",
        )
        store._db.append_message(session_id, "user", "hello")
        return source

    def test_loaded_legacy_alias_is_replaced_by_canonical_key(self, tmp_path):
        """An upgrade leaves one durable route and one restart candidate."""
        sessions_dir = tmp_path / "sessions"
        config = GatewayConfig()
        now = datetime.now()

        initial = SessionStore(sessions_dir=sessions_dir, config=config)
        initial._loaded = True
        source = self._seed(
            initial,
            chat_id=f"any;-;{HANDLE}",
            session_id="legacy-bb-session",
            updated_at=now,
        )
        initial._save_entries()

        restarted = SessionStore(sessions_dir=sessions_dir, config=config)
        recovered = restarted.get_or_create_session(source)
        canonical_key = build_session_key(source)

        assert recovered.session_id == "legacy-bb-session"
        aliases = {
            key
            for key, entry in restarted._entries.items()
            if entry.session_id == "legacy-bb-session"
        }
        assert aliases == {canonical_key}
        assert restarted.suspend_recently_active(max_age_seconds=120) == 1
        assert set(
            restarted._db.load_gateway_routing_entries(
                scope=restarted._routing_scope()
            )
        ) == {canonical_key}
        assert restarted._db.get_session("legacy-bb-session")["session_key"] == (
            canonical_key
        )

    def test_loaded_alias_collision_keeps_most_recent_session(self, tmp_path):
        """Multiple old GUID variants collapse without reviving a stale one."""
        sessions_dir = tmp_path / "sessions"
        config = GatewayConfig()
        now = datetime.now()

        initial = SessionStore(sessions_dir=sessions_dir, config=config)
        initial._loaded = True
        self._seed(
            initial,
            chat_id=f"iMessage;-;{HANDLE}",
            session_id="older-bb-session",
            updated_at=now - timedelta(minutes=5),
        )
        source = self._seed(
            initial,
            chat_id=f"any;-;{HANDLE}",
            session_id="newer-bb-session",
            updated_at=now,
        )
        initial._save_entries()

        restarted = SessionStore(sessions_dir=sessions_dir, config=config)
        recovered = restarted.get_or_create_session(source)
        canonical_key = build_session_key(source)

        assert recovered.session_id == "newer-bb-session"
        assert set(restarted._entries) == {canonical_key}

    def test_retired_alias_sibling_cannot_strand_the_live_transcript(self, tmp_path):
        """A retired sibling must be ended, not merely dropped from the index.

        The stray bare-handle session is typically created *later* than the real
        conversation (a GUID-less webhook lands mid-thread), so a peer lookup
        ordered by ``started_at`` prefers it.  If the migration only drops it
        from the routing index, it stays live in state.db under the canonical
        key and a later recovery reopens the near-empty stray, orphaning the
        real transcript.
        """
        sessions_dir = tmp_path / "sessions"
        config = GatewayConfig()
        now = datetime.now()

        store = SessionStore(sessions_dir=sessions_dir, config=config)
        store._loaded = True
        real_src = self._seed(
            store,
            chat_id=f"any;-;{HANDLE}",
            session_id="real-session",
            created_at=now - timedelta(hours=2),
            updated_at=now,
        )
        self._seed(
            store,
            chat_id=HANDLE,
            session_id="stray-session",
            created_at=now,
            updated_at=now - timedelta(hours=1),
        )
        # Pin the started_at ordering the bug depends on: the stray is newer.
        store._db._conn.execute(
            "UPDATE sessions SET started_at = 1000 WHERE id = 'real-session'"
        )
        store._db._conn.execute(
            "UPDATE sessions SET started_at = 2000 WHERE id = 'stray-session'"
        )
        store._db._conn.commit()
        store._save_entries()

        restarted = SessionStore(sessions_dir=sessions_dir, config=config)
        resumed = restarted.get_or_create_session(real_src)
        assert resumed.session_id == "real-session"

        stray_row = restarted._db.get_session("stray-session")
        assert stray_row["ended_at"] is not None, "retired sibling left live"

        recovered = restarted._db.find_latest_gateway_session_for_peer(
            source="bluebubbles",
            user_id=HANDLE,
            session_key=build_session_key(real_src),
            chat_id=f"any;-;{HANDLE}",
            chat_type="dm",
            match_by_participant_identity=True,
        )
        assert recovered is not None
        assert recovered["id"] == "real-session", (
            "recovery reopened the retired stray and orphaned the transcript"
        )


