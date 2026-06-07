"""State-of-the-art inactivity-based chat retention.

A completed (ended, non-archived) chat is auto-deleted only after it has been
untouched for N days, measured from its completion time (``ended_at`` = last
activity). Active and archived chats are never deleted; continuing a chat
refreshes ``ended_at`` and resets the clock. Backs
config.sessions.delete_inactive_after_days via
SessionDB.prune_inactive_sessions, wired into
SessionDB.maybe_auto_prune_and_vacuum(inactive_days=...).
"""

import time
import pytest

from hermes_state import RetentionPolicy, SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


def _ended(db, sid, source="cli", parent=None):
    db.create_session(session_id=sid, source=source, parent_session_id=parent)
    db.end_session(sid, end_reason="done")


def _backdate_ended(db, sid, days_ago):
    db._conn.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        (time.time() - days_ago * 86400, sid),
    )
    db._conn.commit()


def _backdate_started(db, sid, days_ago):
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (time.time() - days_ago * 86400, sid),
    )
    db._conn.commit()


class TestPruneInactiveSessions:
    def test_deletes_inactive_completed(self, db):
        _ended(db, "old")
        _backdate_ended(db, "old", 100)  # finished 100 days ago, untouched
        assert db.prune_inactive_sessions(older_than_days=90) == 1
        assert db.get_session("old") is None

    def test_keeps_recently_completed(self, db):
        _ended(db, "recent")  # finished just now
        assert db.prune_inactive_sessions(older_than_days=90) == 0
        assert db.get_session("recent") is not None

    def test_keeps_active_even_if_old(self, db):
        db.create_session(session_id="active", source="cli")  # never ended
        _backdate_started(db, "active", 200)
        assert db.prune_inactive_sessions(older_than_days=90) == 0
        assert db.get_session("active") is not None

    def test_keeps_archived_even_if_old(self, db):
        _ended(db, "arch")
        _backdate_ended(db, "arch", 200)
        db.set_session_archived("arch", True)
        assert db.prune_inactive_sessions(older_than_days=90) == 0
        assert db.get_session("arch") is not None

    def test_measures_from_ended_not_started(self, db):
        # Started long ago but finished recently -> KEPT. This is the core SOTA
        # improvement over the legacy started_at-based prune.
        _ended(db, "longrun")
        _backdate_started(db, "longrun", 300)  # ended_at stays ~now
        assert db.prune_inactive_sessions(older_than_days=90) == 0
        assert db.get_session("longrun") is not None

    def test_orphans_children_instead_of_cascade(self, db):
        _ended(db, "parent")
        _ended(db, "child", parent="parent")  # child finished just now
        _backdate_ended(db, "parent", 100)
        assert db.prune_inactive_sessions(older_than_days=90) == 1
        assert db.get_session("parent") is None
        child = db.get_session("child")
        assert child is not None
        assert child["parent_session_id"] is None


class TestMaybeAutoPruneInactivePolicy:
    def test_inactive_policy_used_when_set(self, db):
        _ended(db, "stale")
        _backdate_ended(db, "stale", 100)
        _ended(db, "fresh")
        res = db.maybe_auto_prune_and_vacuum(inactive_days=90, vacuum=False)
        assert res["pruned"] == 1
        assert db.get_session("stale") is None
        assert db.get_session("fresh") is not None

    def test_age_policy_unchanged_when_inactive_unset(self, db):
        # Legacy behaviour: measured from started_at.
        _ended(db, "oldstart")
        _backdate_started(db, "oldstart", 100)
        res = db.maybe_auto_prune_and_vacuum(retention_days=90, vacuum=False)
        assert res["pruned"] == 1
        assert db.get_session("oldstart") is None


def _insert_orphan_message(db, session_id):
    db._conn.execute("PRAGMA foreign_keys=OFF")
    db._conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
        (session_id, "user", "x", time.time()),
    )
    db._conn.commit()


class TestOrphanMessageSweep:
    def test_deletes_only_orphans(self, db):
        db.create_session(session_id="s1", source="cli")
        _insert_orphan_message(db, "ghost")   # no such session -> orphan
        _insert_orphan_message(db, "s1")       # real session -> keep
        removed = db.delete_orphan_messages()
        assert removed == 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='ghost'").fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='s1'").fetchone()[0] == 1

    def test_no_orphans_is_noop(self, db):
        db.create_session(session_id="s1", source="cli")
        _insert_orphan_message(db, "s1")
        assert db.delete_orphan_messages() == 0

    def test_auto_prune_sweeps_orphans(self, db):
        db.create_session(session_id="s1", source="cli")
        _insert_orphan_message(db, "ghost")
        res = db.maybe_auto_prune_and_vacuum(inactive_days=90, vacuum=False)
        assert res["orphan_messages"] == 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='ghost'").fetchone()[0] == 0


class TestTieredRetention:
    def test_source_filter_only_deletes_matching_source(self, db):
        _ended(db, "cron1", source="cron")
        _ended(db, "web1", source="webui")
        _backdate_ended(db, "cron1", 40)
        _backdate_ended(db, "web1", 40)
        assert db.prune_inactive_sessions(older_than_days=30, source="cron") == 1
        assert db.get_session("cron1") is None
        assert db.get_session("web1") is not None

    def test_tiered_auto_prune_cron_shorter_than_interactive(self, db):
        _ended(db, "cron_old", source="cron")
        _backdate_ended(db, "cron_old", 40)    # cron 40d -> deleted (>30)
        _ended(db, "cron_fresh", source="cron")
        _backdate_ended(db, "cron_fresh", 10)  # cron 10d -> kept (<30)
        _ended(db, "web_mid", source="webui")
        _backdate_ended(db, "web_mid", 40)     # webui 40d -> kept (<90)
        _ended(db, "web_old", source="webui")
        _backdate_ended(db, "web_old", 100)    # webui 100d -> deleted (>90)
        res = db.maybe_auto_prune_and_vacuum(
            inactive_days=90, automated_inactive_days=30,
            automated_source="cron", vacuum=False)
        assert res["pruned"] == 2
        assert db.get_session("cron_old") is None
        assert db.get_session("web_old") is None
        assert db.get_session("cron_fresh") is not None
        assert db.get_session("web_mid") is not None


class TestSoftDeleteTrash:
    def test_soft_delete_moves_to_trash_and_hides(self, db):
        _ended(db, "old", source="webui")
        _backdate_ended(db, "old", 100)
        _ended(db, "fresh", source="webui")
        assert db.soft_delete_inactive_sessions(older_than_days=90) == 1
        # row still present (restorable) but flagged deleted_at
        assert db.get_session("old")["deleted_at"] is not None
        # hidden from listing + count, fresh still visible
        ids = [s["id"] for s in db.list_sessions_rich(limit=50)]
        assert "old" not in ids and "fresh" in ids
        assert db.count_trashed_sessions() == 1
        assert db.session_count() == 1  # only 'fresh' counted

    def test_archived_not_trashed(self, db):
        _ended(db, "arch")
        _backdate_ended(db, "arch", 100)
        db.set_session_archived("arch", True)
        assert db.soft_delete_inactive_sessions(older_than_days=90) == 0
        assert db.get_session("arch")["deleted_at"] is None

    def test_purge_after_grace_only(self, db):
        _ended(db, "old")
        _backdate_ended(db, "old", 100)
        db.soft_delete_inactive_sessions(older_than_days=90)
        # just trashed -> within grace -> not purged
        assert db.purge_trashed_sessions(grace_days=30) == 0
        assert db.get_session("old") is not None
        # backdate trash timestamp beyond grace -> purged
        db._conn.execute("UPDATE sessions SET deleted_at=? WHERE id=?",
                         (time.time() - 40 * 86400, "old"))
        db._conn.commit()
        assert db.purge_trashed_sessions(grace_days=30) == 1
        assert db.get_session("old") is None

    def test_restore(self, db):
        _ended(db, "old")
        _backdate_ended(db, "old", 100)
        db.soft_delete_inactive_sessions(older_than_days=90)
        assert db.restore_session("old") is True
        assert db.get_session("old")["deleted_at"] is None
        assert "old" in [s["id"] for s in db.list_sessions_rich(limit=50)]
        assert db.restore_session("old") is False  # not trashed anymore

    def test_auto_prune_soft_then_purge(self, db):
        _ended(db, "to_trash")
        _backdate_ended(db, "to_trash", 100)   # inactive -> trashed this run
        _ended(db, "old_trash")
        db._conn.execute("UPDATE sessions SET deleted_at=? WHERE id=?",
                         (time.time() - 40 * 86400, "old_trash"))  # already in trash >grace
        db._conn.commit()
        res = db.maybe_auto_prune_and_vacuum(
            inactive_days=90, trash_grace_days=30, vacuum=False)
        assert res.get("trashed") == 1
        assert res["pruned"] == 1
        assert db.get_session("old_trash") is None
        t = db.get_session("to_trash")
        assert t is not None and t["deleted_at"] is not None

    def test_trashed_excluded_from_search(self, db):
        db.create_session(session_id="s1", source="webui")
        db.append_message("s1", "user", "uniquemagicword apple")
        db.end_session("s1", end_reason="done")
        # visible in search before trashing
        hits = db.search_messages("uniquemagicword", limit=10)
        assert any(h["session_id"] == "s1" for h in hits)
        _backdate_ended(db, "s1", 100)
        db.soft_delete_inactive_sessions(older_than_days=90)
        hits2 = db.search_messages("uniquemagicword", limit=10)
        assert all(h["session_id"] != "s1" for h in hits2)


class TestRetentionPolicy:
    def test_from_config_all_keys(self):
        p = RetentionPolicy.from_config({
            "retention_days": 60, "min_interval_hours": 12, "vacuum_after_prune": False,
            "delete_inactive_after_days": 30, "delete_automated_inactive_after_days": 7,
            "automated_source": "telegram", "trash_grace_days": 14,
        })
        assert (p.retention_days, p.min_interval_hours, p.vacuum_after_prune) == (60, 12, False)
        assert (p.inactive_days, p.automated_inactive_days) == (30, 7)
        assert p.automated_source == "telegram" and p.trash_grace_days == 14

    def test_from_config_defaults_and_none(self):
        for cfg in ({}, None):
            p = RetentionPolicy.from_config(cfg)
            assert p.retention_days == 90 and p.inactive_days is None
            assert p.automated_source == "cron" and p.trash_grace_days is None

    def test_policy_object_matches_kwargs(self, db):
        _ended(db, "old")
        _backdate_ended(db, "old", 100)
        _ended(db, "fresh")
        res = db.maybe_auto_prune_and_vacuum(RetentionPolicy(inactive_days=90), vacuum=False)
        assert res["pruned"] == 1
        assert db.get_session("old") is None and db.get_session("fresh") is not None


class TestActiveSessionsView:
    def test_view_excludes_trashed(self, db):
        _ended(db, "keep")
        _ended(db, "trash")
        _backdate_ended(db, "trash", 100)
        db.soft_delete_inactive_sessions(older_than_days=90)
        view_ids = {r[0] for r in db._conn.execute("SELECT id FROM active_sessions")}
        assert "keep" in view_ids and "trash" not in view_ids

    def test_no_query_surfaces_trashed(self, db):
        # One trashed chat must not leak through ANY listing/count/search path.
        db.create_session(session_id="leaky", source="webui")
        db.append_message("leaky", "user", "zzqq_secret_token apple")
        db.end_session("leaky", end_reason="done")
        _backdate_ended(db, "leaky", 100)
        before = db.session_count()
        db.soft_delete_inactive_sessions(older_than_days=90)
        assert db.session_count() == before - 1
        assert "leaky" not in {s["id"] for s in db.list_sessions_rich(limit=100)}
        assert "leaky" not in {s["id"] for s in db.list_sessions_rich(limit=100, order_by_last_active=True)}
        assert "leaky" not in {s["id"] for s in db.search_sessions(limit=100)}
        assert all(h["session_id"] != "leaky"
                   for h in db.search_messages("zzqq_secret_token", limit=10))
        # ...yet still reachable by id and restorable.
        assert db.get_session("leaky") is not None
        assert db.restore_session("leaky") is True
        assert "leaky" in {s["id"] for s in db.list_sessions_rich(limit=100)}
