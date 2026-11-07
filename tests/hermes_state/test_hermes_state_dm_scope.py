"""Own-DM scoping for the Telegram Mini App's non-admin tier: session listing, counts and row checks."""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


class TestSessionCountDmScope:
    def test_session_count_scope_own_matches_list_sessions_rich(self, db):
        db.create_session("s1", "telegram", user_id="42", chat_id="42", chat_type="dm")
        db.create_session("s2", "telegram", user_id="99", chat_id="99", chat_type="dm")
        db.create_session("group1", "telegram", chat_id="42", chat_type="group")
        assert db.session_count(scope="own", requester_user_id="42") == 1

    def test_session_count_scope_own_requires_requester_user_id(self, db):
        with pytest.raises(ValueError):
            db.session_count(scope="own")

    def test_session_count_scope_rejects_invalid_value(self, db):
        with pytest.raises(ValueError):
            db.session_count(scope="everything")


class TestListSessionsRichDmScope:
    """scope="own" — the Telegram Mini App dashboard's non-admin DM scoping."""

    def test_scope_own_returns_only_requesters_dm_sessions(self, db):
        db.create_session("s1", "telegram", user_id="42", chat_id="42", chat_type="dm")
        db.create_session("s2", "telegram", user_id="99", chat_id="99", chat_type="dm")
        sessions = db.list_sessions_rich(scope="own", requester_user_id="42")
        assert [s["id"] for s in sessions] == ["s1"]

    def test_scope_own_excludes_group_sessions_even_for_same_chat_id(self, db):
        # A group chat_id could numerically collide with some other user's DM
        # user id; chat_type='dm' is required in addition to chat_id match.
        db.create_session("group1", "telegram", chat_id="42", chat_type="group")
        db.create_session("dm1", "telegram", user_id="42", chat_id="42", chat_type="dm")
        sessions = db.list_sessions_rich(scope="own", requester_user_id="42")
        assert [s["id"] for s in sessions] == ["dm1"]

    def test_scope_own_excludes_other_platforms(self, db):
        db.create_session("s1", "cli", chat_id="42", chat_type="dm")
        sessions = db.list_sessions_rich(scope="own", requester_user_id="42")
        assert sessions == []

    def test_scope_own_excludes_legacy_rows_with_no_chat_id(self, db):
        # Predates chat/thread capture — cannot prove ownership, fails closed.
        db.create_session("s1", "telegram", user_id="42", chat_type="dm")
        sessions = db.list_sessions_rich(scope="own", requester_user_id="42")
        assert sessions == []

    def test_scope_own_requires_requester_user_id(self, db):
        with pytest.raises(ValueError):
            db.list_sessions_rich(scope="own")

    def test_scope_own_pinned_backfill_excludes_other_users_pins(self, db):
        # The pinned back-fill builds its own WHERE (pins ignore the archived
        # filter), so it must re-apply the own-DM scope as well.
        db.create_session("mine", "telegram", user_id="42", chat_id="42", chat_type="dm")
        db.create_session("theirs", "telegram", user_id="99", chat_id="99", chat_type="dm")
        db.set_session_pinned("theirs", True)
        sessions = db.list_sessions_rich(scope="own", requester_user_id="42", include_pinned=True)
        assert [s["id"] for s in sessions] == ["mine"]

    def test_scope_rejects_invalid_value(self, db):
        with pytest.raises(ValueError):
            db.list_sessions_rich(scope="everything")

    def test_scope_none_is_unfiltered_default(self, db):
        db.create_session("s1", "telegram", user_id="42", chat_id="42", chat_type="dm")
        db.create_session("s2", "telegram", user_id="99", chat_id="99", chat_type="dm")
        sessions = db.list_sessions_rich()
        assert {s["id"] for s in sessions} == {"s1", "s2"}

    def test_scope_admin_is_unfiltered(self, db):
        db.create_session("s1", "telegram", user_id="42", chat_id="42", chat_type="dm")
        db.create_session("s2", "telegram", user_id="99", chat_id="99", chat_type="dm")
        sessions = db.list_sessions_rich(scope="admin")
        assert {s["id"] for s in sessions} == {"s1", "s2"}


class TestSessionRowIsOwnDm:
    """session_row_is_own_dm — the single-row counterpart to
    _dm_own_scope_clause, used by the dashboard's per-session ownership
    check (GET /api/sessions/{id} and .../messages) where the row is
    already fetched and a second filtered query isn't needed. Same rule as
    TestListSessionsRichDmScope above, evaluated in Python instead of SQL —
    the parity tests at the bottom of this class pin that the two shapes
    agree on every case exercised there.
    """

    def test_own_dm_row_is_owned(self):
        from hermes_state_session_scope import session_row_is_own_dm

        row = {"source": "telegram", "chat_type": "dm", "chat_id": "42"}
        assert session_row_is_own_dm(row, "42") is True

    def test_other_users_dm_row_is_not_owned(self):
        from hermes_state_session_scope import session_row_is_own_dm

        row = {"source": "telegram", "chat_type": "dm", "chat_id": "99"}
        assert session_row_is_own_dm(row, "42") is False

    def test_group_session_with_matching_chat_id_is_not_owned(self):
        from hermes_state_session_scope import session_row_is_own_dm

        row = {"source": "telegram", "chat_type": "group", "chat_id": "42"}
        assert session_row_is_own_dm(row, "42") is False

    def test_other_platform_is_not_owned(self):
        from hermes_state_session_scope import session_row_is_own_dm

        row = {"source": "cli", "chat_type": "dm", "chat_id": "42"}
        assert session_row_is_own_dm(row, "42") is False

    def test_legacy_row_with_no_chat_id_fails_closed(self):
        from hermes_state_session_scope import session_row_is_own_dm

        row = {"source": "telegram", "chat_type": "dm", "chat_id": None}
        assert session_row_is_own_dm(row, "42") is False

    def test_blank_chat_id_fails_closed(self):
        from hermes_state_session_scope import session_row_is_own_dm

        row = {"source": "telegram", "chat_type": "dm", "chat_id": ""}
        assert session_row_is_own_dm(row, "42") is False

    def test_missing_requester_user_id_fails_closed(self):
        from hermes_state_session_scope import session_row_is_own_dm

        row = {"source": "telegram", "chat_type": "dm", "chat_id": "42"}
        assert session_row_is_own_dm(row, None) is False
        assert session_row_is_own_dm(row, "") is False

    def test_numeric_and_string_chat_id_types_both_match(self):
        # sqlite3.Row values come back as whatever type the column stored;
        # the comparison must not be sensitive to int vs. str.
        from hermes_state_session_scope import session_row_is_own_dm

        row = {"source": "telegram", "chat_type": "dm", "chat_id": 42}
        assert session_row_is_own_dm(row, "42") is True
        assert session_row_is_own_dm(row, 42) is True

    # ---- parity with _dm_own_scope_clause / list_sessions_rich(scope="own") --

    def test_parity_with_list_sessions_rich_scope_own(self, db):
        """Every session list_sessions_rich(scope="own") returns for a given
        requester must also be judged "owned" by session_row_is_own_dm on
        the same fetched row, and vice versa for excluded rows — the SQL
        filter and the Python predicate must agree, not just individually
        look plausible.
        """
        from hermes_state_session_scope import session_row_is_own_dm

        db.create_session("owned-dm", "telegram", user_id="42", chat_id="42", chat_type="dm")
        db.create_session("other-dm", "telegram", user_id="99", chat_id="99", chat_type="dm")
        db.create_session("group-same-chat-id", "telegram", chat_id="42", chat_type="group")
        db.create_session("legacy-no-chat-id", "telegram", user_id="42", chat_type="dm")
        db.create_session("other-platform", "cli", chat_id="42", chat_type="dm")

        included_ids = {s["id"] for s in db.list_sessions_rich(scope="own", requester_user_id="42")}
        all_ids = [
            "owned-dm", "other-dm", "group-same-chat-id",
            "legacy-no-chat-id", "other-platform",
        ]
        for sid in all_ids:
            row = db.get_session(sid)
            predicate_says_owned = session_row_is_own_dm(row, "42")
            sql_says_owned = sid in included_ids
            assert predicate_says_owned == sql_says_owned, sid
