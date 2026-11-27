"""The SQLite maintenance repository's prune verbs (``prune_sessions``, ``list_prune_candidates``, ``count_prune_matches``, ``count_open_prune_matches``) and the routing repository's never-active verbs: the age window, the open-session guard, the filter surface and its LIKE escaping."""

import time

import pytest

from persistence.housekeeping.maintenance import PruneCriteria
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


@pytest.fixture
def media_root(tmp_path):
    return str(tmp_path / "media")


# =========================================================================
# Prune
# =========================================================================

class TestPruneSessions:
    def test_prune_old_ended_sessions(self, db, store, media_root):
        # Create and end an "old" session
        create_session(db, session_id="old", source="cli")
        end_session(db, "old", end_reason="done")
        # Manually backdate started_at
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 100 * 86400, "old"),
        )
        db._conn.commit()

        # Create a recent session
        create_session(db, session_id="new", source="cli")

        pruned = store.maintenance.prune_sessions(older_than_days=90, media_root=media_root).count
        assert pruned == 1
        assert get_session(db, "old") is None
        session = get_session(db, "new")
        assert session is not None
        assert session["id"] == "new"

    def test_prune_skips_active_sessions(self, db, store, media_root):
        create_session(db, session_id="active", source="cli")
        # Backdate but don't end
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 200 * 86400, "active"),
        )
        db._conn.commit()

        pruned = store.maintenance.prune_sessions(older_than_days=90, media_root=media_root).count
        assert pruned == 0
        assert get_session(db, "active") is not None
        assert store.maintenance.count_open_prune_matches(older_than_days=90) == 1

    def test_open_prune_match_count_applies_other_filters(self, db, store):
        create_session(db, session_id="matching-open", source="cron")
        create_session(db, session_id="other-source", source="cli")
        create_session(db, session_id="ended", source="cron")
        end_session(db, "ended", "completed")
        old = time.time() - 200 * 86400
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id IN (?, ?, ?)",
            (old, "matching-open", "other-source", "ended"),
        )
        db._conn.commit()

        assert store.maintenance.count_open_prune_matches(
            older_than_days=90, source="cron", criteria=PruneCriteria(archived=False)
        ) == 1
        assert {row.session_id for row in store.maintenance.list_prune_candidates(
            older_than_days=90, source="cron", criteria=PruneCriteria(archived=False)
        )} == {"ended"}

    def test_negative_older_than_days_rejected_at_every_prune_boundary(self, db, store, media_root):
        """A negative bound builds a FUTURE cutoff that matches every ended session (and every
        never-active keyed row) — the repository verbs must raise, naming the allowed range, instead
        of mass-deleting; ``sessions.retention_days: -1`` reaches these paths from config (#116361)."""
        create_session(db, session_id="ended", source="cli")
        end_session(db, "ended", "done")
        create_session(db, session_id="keyed", source="telegram", session_key="telegram:dm:1")
        for call in (lambda **kw: store.maintenance.prune_sessions(**kw, media_root=media_root),
                     store.maintenance.list_prune_candidates, store.maintenance.count_prune_matches,
                     store.routing.never_active_keyed_sessions,
                     lambda **kw: store.routing.prune_never_active(**kw, media_root=media_root)):
            with pytest.raises(ValueError, match=">= 0"):
                call(older_than_days=-1)
        assert get_session(db, "ended") is not None
        assert get_session(db, "keyed") is not None

class TestPruneSessionFilters:
    """Extended filter surface shared by prune/archive/list_prune_candidates."""

    @staticmethod
    def _mk(db, sid, *, source="cli", age_seconds=0, title=None,
            end_reason="done", message_count=0, cwd=None):
        create_session(db, session_id=sid, source=source, cwd=cwd)
        end_session(db, sid, end_reason=end_reason)
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, message_count = ?, title = ? "
            "WHERE id = ?",
            (time.time() - age_seconds, message_count, title, sid),
        )
        db._conn.commit()

    def test_started_after_window_prunes_only_recent(self, db, store, media_root):
        self._mk(db, "recent1", age_seconds=3600)       # 1h ago
        self._mk(db, "recent2", age_seconds=2 * 3600)   # 2h ago
        self._mk(db, "old", age_seconds=10 * 3600)      # 10h ago

        cutoff = time.time() - 5 * 3600
        pruned = store.maintenance.prune_sessions(
            older_than_days=None, criteria=PruneCriteria(started_after=cutoff), media_root=media_root
        ).count
        assert pruned == 2
        assert get_session(db, "old") is not None
        assert get_session(db, "recent1") is None

    def test_title_and_message_count_filters(self, db, store, media_root):
        self._mk(db, "smoke1", age_seconds=60, title="Codex Smoke Test 1",
                 message_count=2)
        self._mk(db, "smoke2", age_seconds=60, title="codex smoke test 2",
                 message_count=8)
        self._mk(db, "real", age_seconds=60, title="Debugging auth",
                 message_count=8)

        rows = store.maintenance.list_prune_candidates(criteria=PruneCriteria(title_like="smoke"))
        assert {r.session_id for r in rows} == {"smoke1", "smoke2"}

        pruned = store.maintenance.prune_sessions(
            older_than_days=None, criteria=PruneCriteria(title_like="Smoke", max_messages=3), media_root=media_root
        ).count
        assert pruned == 1
        assert get_session(db, "smoke1") is None
        assert get_session(db, "smoke2") is not None
        assert get_session(db, "real") is not None

    @staticmethod
    def _mk_rich(db, sid, **cols):
        """Create an ended session then set arbitrary sessions columns."""
        create_session(db, session_id=sid, source=cols.pop("source", "cli"))
        end_session(db, sid, end_reason=cols.pop("end_reason", "done"))
        cols.setdefault("started_at", time.time() - 60)
        sets = ", ".join(f"{k} = ?" for k in cols)
        db._conn.execute(
            f"UPDATE sessions SET {sets} WHERE id = ?", (*cols.values(), sid)
        )
        db._conn.commit()

    def test_title_like_underscore_is_literal_not_a_wildcard(self, db, store, media_root):
        """``_`` is a single-character wildcard in SQL LIKE, so an unescaped
        filter deletes sessions the operator never selected. The filters are
        documented (and shown in the CLI confirmation) as substring matches.
        """
        self._mk(db, "target", title="user_auth refactor")
        self._mk(db, "bystander1", title="user-auth review")
        self._mk(db, "bystander2", title="userXauth notes")
        self._mk(db, "bystander3", title="user auth meeting")

        rows = store.maintenance.list_prune_candidates(criteria=PruneCriteria(title_like="user_auth"))
        assert {r.session_id for r in rows} == {"target"}

        pruned = store.maintenance.prune_sessions(
            older_than_days=None, criteria=PruneCriteria(title_like="user_auth"), media_root=media_root
        ).count
        assert pruned == 1
        for survivor in ("bystander1", "bystander2", "bystander3"):
            assert get_session(db, survivor) is not None

    def test_percent_in_filter_does_not_select_everything(self, db, store):
        """``%`` matches any run of characters — a bare one would delete the
        whole table."""
        self._mk(db, "a", title="alpha")
        self._mk(db, "b", title="beta")
        self._mk(db, "pct", title="100% coverage run")

        # Only the title that really contains a percent sign matches.
        assert {r.session_id for r in store.maintenance.list_prune_candidates(
            criteria=PruneCriteria(title_like="%"))} == {"pct"}
        assert {r.session_id for r in store.maintenance.list_prune_candidates(
            criteria=PruneCriteria(title_like="100%"))} == {"pct"}

    def test_branch_like_underscore_is_literal(self, db, store):
        """Branch names carry underscores routinely."""
        self._mk_rich(db, "want", git_branch="fix/session_prune")
        self._mk_rich(db, "other", git_branch="fix/session-prune")

        rows = store.maintenance.list_prune_candidates(criteria=PruneCriteria(branch_like="session_prune"))
        assert {r.session_id for r in rows} == {"want"}

    def test_model_like_underscore_is_literal(self, db, store):
        self._mk_rich(db, "want", model="vendor/model_mini")
        self._mk_rich(db, "other", model="vendor/model-mini")

        rows = store.maintenance.list_prune_candidates(criteria=PruneCriteria(model_like="model_mini"))
        assert {r.session_id for r in rows} == {"want"}

    def test_plain_substring_filters_still_match(self, db, store):
        """Guard against over-escaping: ordinary filters keep working, and a
        literal backslash in the needle is matched as itself."""
        self._mk(db, "smoke", title="Codex Smoke Test")
        self._mk_rich(db, "winpath", title=r"build C:\tmp artifacts")

        assert {r.session_id for r in store.maintenance.list_prune_candidates(
            criteria=PruneCriteria(title_like="smoke"))} == {"smoke"}
        assert {r.session_id for r in store.maintenance.list_prune_candidates(
            criteria=PruneCriteria(title_like=r"c:\tmp"))} == {"winpath"}

    def test_cwd_prefix_underscore_is_literal_not_a_wildcard(self, db, store, media_root):
        """``_`` is a LIKE wildcard but an ordinary character in a path, so an
        unescaped prefix also matched a same-length sibling directory — and
        prune_sessions deletes what it matches."""
        self._mk(db, "target", cwd="/home/me/my_project/src")
        self._mk(db, "sibling", cwd="/home/me/myXproject/src")

        rows = store.maintenance.list_prune_candidates(criteria=PruneCriteria(cwd_prefix="/home/me/my_project"))
        assert {r.session_id for r in rows} == {"target"}

        pruned = store.maintenance.prune_sessions(
            older_than_days=None, criteria=PruneCriteria(cwd_prefix="/home/me/my_project"), media_root=media_root
        ).count
        assert pruned == 1
        assert get_session(db, "sibling") is not None

    def test_cwd_prefix_percent_does_not_select_everything(self, db, store):
        self._mk(db, "a", cwd="/home/me/one")
        self._mk(db, "b", cwd="/home/me/two")

        assert store.maintenance.list_prune_candidates(criteria=PruneCriteria(cwd_prefix="/home/me/%")) == ()

    def test_cwd_prefix_still_matches_the_directory_and_its_children(self, db, store):
        """Control: the prefix must keep matching itself and anything under it."""
        self._mk(db, "root", cwd="/home/me/proj")
        self._mk(db, "child", cwd="/home/me/proj/src")
        self._mk(db, "outside", cwd="/home/me/other")

        rows = store.maintenance.list_prune_candidates(criteria=PruneCriteria(cwd_prefix="/home/me/proj"))
        assert {r.session_id for r in rows} == {"root", "child"}

    def test_cwd_prefix_windows_separator_arm(self, db, store):
        """The backslash child arm (``{esc}\\\\\\\\%`` in the pattern) must keep
        matching Windows children while ``_`` stays literal — a guard against
        'simplifying' the quadruple backslash."""
        self._mk(db, "win_root", cwd=r"C:\Users\me\my_project")
        self._mk(db, "win_child", cwd=r"C:\Users\me\my_project\src")
        self._mk(db, "win_sibling", cwd=r"C:\Users\me\myXproject\src")

        rows = store.maintenance.list_prune_candidates(criteria=PruneCriteria(cwd_prefix=r"C:\Users\me\my_project"))
        assert {r.session_id for r in rows} == {"win_root", "win_child"}

    def test_unknown_filter_rejected(self, store, media_root):
        import pytest as _pytest
        with _pytest.raises(TypeError):
            store.maintenance.prune_sessions(
                older_than_days=None, criteria=PruneCriteria(bogus_filter="x"), media_root=media_root
            )
