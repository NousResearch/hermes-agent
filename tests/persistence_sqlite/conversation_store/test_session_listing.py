"""The conversation store's session listing on a backend store's StoreHolder: ``list_sessions_rich`` previews, last-active from activity heartbeats, session-key scoping, reset and branch children, delegate rows, compact rows, the gateway peer recording the listing reads through the routing repository, and one cron job's run history."""

import json

import pytest

from agent.session_activity import ActivityProvenance, build_activity_snapshot
from persistence.sqlite.conversation._store.compression import get_compression_chain
from persistence.sqlite.conversation._store.gateway import list_gateway_sessions
from persistence.sqlite.conversation._store.messages import append_message, resolve_resume_session_id
from persistence.sqlite.conversation._store.portability import _get_session_rich_rows_batch, list_cron_job_runs
from persistence.sqlite.conversation._store.sessions import (
    clear_session_activity_labels,
    create_session,
    delete_session,
    end_session,
    get_session,
    get_session_delete_targets,
    list_sessions_rich,
    reopen_session,
    session_count,
    session_count_by_source,
    touch_session_activity,
    update_session_cwd,
    update_system_prompt,
)
from persistence.sqlite.conversation._store.titles import set_session_title


def _activity_snapshot(db, session_id):
    """Durable activity snapshot for *session_id* (what gateway/delegate readers build from the row)."""
    row = get_session(db, session_id)
    return build_activity_snapshot(
        last_activity_at=row.get("last_activity_at"),
        last_activity_description=row.get("last_activity_description"),
        last_activity_provenance=row.get("last_activity_provenance"),
    )


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


class TestListSessionsRich:
    """Tests for enhanced session listing with preview and last_active."""

    def test_preview_from_first_user_message(self, db):
        create_session(db, "s1", "cli")
        append_message(db, "s1", "system", "You are a helpful assistant.")
        append_message(db, "s1", "user", "Help me refactor the auth module please")
        append_message(db, "s1", "assistant", "Sure, let me look at it.")
        sessions = list_sessions_rich(db)
        assert len(sessions) == 1
        assert "Help me refactor the auth module" in sessions[0]["preview"]

    @pytest.mark.parametrize(
        "unsafe_model_config",
        ["{not-json", "[]", '"scalar"', "5", "null"],
    )
    def test_unsafe_model_config_does_not_break_session_surfaces(
        self, db, store, unsafe_model_config
    ):
        create_session(db, "root", "telegram")
        append_message(db, "root", "user", "root message")
        create_session(db, "compression-parent", "telegram")
        end_session(db, "compression-parent", "compression")
        create_session(db,
            "compression-child",
            "telegram",
            parent_session_id="compression-parent",
        )
        append_message(db, "compression-child", "user", "child message")
        create_session(db, "routing-orphan", "telegram")
        append_message(db, "routing-orphan", "user", "orphan message")
        db._conn.execute(
            "UPDATE sessions SET model_config = ? "
            "WHERE id IN (?, ?, ?)",
            (unsafe_model_config, "root", "compression-child", "routing-orphan"),
        )
        db._conn.commit()

        listed = list_sessions_rich(db, source="telegram")
        ordered = list_sessions_rich(db,
            source="telegram", order_by_last_active=True
        )

        assert "root" in {row["id"] for row in listed}
        assert "root" in {row["id"] for row in ordered}
        assert session_count(db, source="telegram", exclude_children=True) == 3
        assert session_count_by_source(db, exclude_children=True)["telegram"] == 3
        assert get_compression_chain(db, "compression-parent") == [
            "compression-parent",
            "compression-child",
        ]
        store.routing.record_session_peer(
            "compression-child",
            source="telegram",
            session_key="agent:main:telegram:dm:recovered",
            include_compression_ancestors=True,
        )
        assert get_session(db, "compression-parent")["session_key"] == (
            "agent:main:telegram:dm:recovered"
        )
        assert any(
            row.orphan_id == "routing-orphan"
            for row in store.routing.orphaned_sessions()
        )

    def test_created_source_preserved_across_cross_platform_resume(self, db, store):
        """``created_source`` is immutable provenance (#56439): stamped at creation and never
        rewritten by gateway peer recording, which must keep ``source`` as live routing state."""
        create_session(db, "tui-sess", "tui")
        append_message(db, "tui-sess", "user", "created on desktop")

        # /resume from Telegram: routing state moves, provenance does not.
        store.routing.record_session_peer(
            "tui-sess", source="telegram", session_key="agent:main:telegram:dm:1", chat_id="1"
        )
        row = get_session(db, "tui-sess")
        assert row["source"] == "telegram"
        assert row["created_source"] == "tui"

        # Later upserts (any surface) never clobber the stamped provenance.
        create_session(db, "tui-sess", "discord")
        assert get_session(db, "tui-sess")["created_source"] == "tui"

        # Self-healing insert stamps provenance from the first writer.
        store.routing.record_session_peer(
            "slack-sess", source="slack", session_key="agent:main:slack:ch:2", chat_id="2"
        )
        assert get_session(db, "slack-sess")["created_source"] == "slack"

    def test_last_active_prefers_session_activity_heartbeat(self, db):
        """Mid-turn agent heartbeats must advance last_active without new messages (#72016)."""
        create_session(db, "s1", "cli")
        append_message(db, "s1", "user", "hello")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=? AND role=?",
                (1_700_000_000.0, "s1", "user"),
            )
            db._conn.commit()

        before = list_sessions_rich(db)[0]["last_active"]
        heartbeat = 1_700_000_500.0
        touch_session_activity(db,
            "s1",
            heartbeat,
            description="starting API call #1",
            provenance=ActivityProvenance.UNKNOWN,
        )
        after = list_sessions_rich(db)[0]["last_active"]
        assert after == heartbeat
        assert after > before

        row = get_session(db, "s1")
        assert row["last_activity_at"] == heartbeat
        assert row["last_activity_description"] == "starting API call #1"
        assert row["last_activity_provenance"] == "unknown"

        activity = _activity_snapshot(db, "s1")
        assert activity["last_activity_at"] == heartbeat
        assert activity["last_activity_description"] == "starting API call #1"
        assert "phase" not in activity

        # Never move last_activity_at backwards.
        touch_session_activity(db, "s1", heartbeat - 100, description="ignored")
        assert get_session(db, "s1")["last_activity_at"] == heartbeat
        assert get_session(db, "s1")["last_activity_description"] == "starting API call #1"

    def test_clear_session_activity_labels_keeps_timestamp(self, db):
        """Turn-end label clear must wipe desc/provenance without moving ts."""
        create_session(db, "s1", "cli")
        heartbeat = 1_700_000_500.0
        touch_session_activity(db,
            "s1",
            heartbeat,
            description="compressing context",
            provenance=ActivityProvenance.AGENT_COMPRESSION,
        )
        row = get_session(db, "s1")
        assert row["last_activity_at"] == heartbeat
        assert row["last_activity_description"] == "compressing context"
        assert row["last_activity_provenance"] == "agent.compression"

        clear_session_activity_labels(db, "s1")
        row = get_session(db, "s1")
        assert row["last_activity_at"] == heartbeat
        assert row["last_activity_description"] == ""
        assert row["last_activity_provenance"] == "unknown"
        activity = _activity_snapshot(db, "s1")
        assert activity["last_activity_at"] == heartbeat
        assert activity["last_activity_description"] == ""
        assert activity["last_activity_provenance"] == "unknown"

    def test_last_active_uses_newer_message_over_stale_heartbeat(self, db):
        """Rate-limited heartbeats can lag message writes; last_active must take max."""
        create_session(db, "s1", "cli")
        append_message(db, "s1", "user", "hello")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=?",
                (1_700_000_800.0, "s1"),
            )
            db._conn.commit()
        touch_session_activity(db, "s1", 1_700_000_500.0, description="api")  # older than message
        assert list_sessions_rich(db)[0]["last_active"] == 1_700_000_800.0

    def test_list_gateway_sessions_last_active_uses_activity_heartbeat(self, db):
        create_session(db,
            "gw-1",
            "telegram",
            session_key="agent:main:telegram:dm:c1",
            chat_id="c1",
            chat_type="dm",
        )
        append_message(db, "gw-1", "user", "ping")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=?",
                (1_700_000_000.0, "gw-1"),
            )
            db._conn.commit()

        heartbeat = 1_700_000_900.0
        touch_session_activity(db,
            "gw-1",
            heartbeat,
            description="compressing context",
        )
        rows = list_gateway_sessions(db, active_only=True)
        assert len(rows) == 1
        assert rows[0]["last_active"] == heartbeat
        activity = _activity_snapshot(db, "gw-1")
        assert activity["last_activity_description"] == "compressing context"

    def test_rich_list_session_key_filter_precedes_limit(self, db):
        lane_key = "agent:main:telegram:dm:lane"
        create_session(db,
            "lane_oldest", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        create_session(db,
            "lane_newest", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        for i in range(60):
            create_session(db,
                f"foreign_{i}", "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
        create_session(db,
            "legacy_null_key", "telegram", user_id="lane-user", chat_id="lane"
        )

        sessions = list_sessions_rich(db,
            source="telegram", session_key=lane_key, limit=2
        )

        assert [session["id"] for session in sessions] == [
            "lane_newest", "lane_oldest",
        ]

    def test_rich_list_session_key_scopes_search_and_projects_compression(self, db):
        lane_key = "agent:main:telegram:dm:lane"
        create_session(db,
            "lane_root", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        set_session_title(db, "lane_root", "Needle root")
        end_session(db, "lane_root", "compression")
        create_session(db,
            "lane_tip", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane", parent_session_id="lane_root",
        )
        set_session_title(db, "lane_tip", "Needle continuation")
        append_message(db, "lane_tip", "user", "latest lane activity")
        create_session(db,
            "foreign_match", "telegram",
            session_key="agent:main:telegram:dm:foreign",
            user_id="foreign-user", chat_id="foreign",
        )
        set_session_title(db, "foreign_match", "Needle foreign")

        sessions = list_sessions_rich(db,
            source="telegram",
            session_key=lane_key,
            search_query="needle",
            order_by_last_active=True,
            limit=1,
        )

        assert [session["id"] for session in sessions] == ["lane_tip"]
        assert sessions[0]["_lineage_root_id"] == "lane_root"

    @pytest.mark.parametrize(
        "end_reason",
        [
            "session_reset",
            "session_switch",
            "idle",
            "daily",
            "suspended",
            "resume_pending_expired",
        ],
    )
    def test_rich_list_keeps_legacy_reset_children_visible(self, db, end_reason):
        from persistence.sqlite._schema_ddl import _ephemeral_child_sql

        lane_key = "agent:main:telegram:dm:lane"
        parent_id = f"parent_{end_reason}"
        child_id = f"child_{end_reason}"
        create_session(db, parent_id, "telegram", session_key=lane_key)
        end_session(db, parent_id, end_reason)
        # No _reset_from marker: this is the on-disk shape written before the
        # marker existed. The unchanged routing key proves a reset boundary.
        create_session(db,
            child_id,
            "telegram",
            session_key=lane_key,
            parent_session_id=parent_id,
        )

        listed = [row["id"] for row in list_sessions_rich(db, source="telegram")]
        assert {parent_id, child_id}.issubset(listed)
        assert session_count(db, source="telegram", exclude_children=True) == 2
        assert session_count_by_source(db, exclude_children=True)["telegram"] == 2
        ephemeral = db._conn.execute(
            f"SELECT s.id FROM sessions s WHERE {_ephemeral_child_sql('s')}"
        ).fetchall()
        assert child_id not in {row["id"] for row in ephemeral}

    def test_reopen_keeps_branch_provenance_out_of_legacy_reset_backfill(self, db):
        """A same-key branch is never rewritten as a reset successor on reopen."""
        lane_key = "agent:main:telegram:dm:branch"
        create_session(db, "branch_parent", "telegram", session_key=lane_key)
        create_session(db,
            "branch_child",
            "telegram",
            session_key=lane_key,
            parent_session_id="branch_parent",
            model_config={"_branched_from": "branch_parent"},
        )
        end_session(db, "branch_parent", "session_switch")

        reopen_session(db, "branch_parent")

        child = get_session(db, "branch_child")
        assert child is not None
        assert json.loads(child["model_config"]) == {"_branched_from": "branch_parent"}
        assert "branch_child" in [row["id"] for row in list_sessions_rich(db, source="telegram")]

    def test_reopen_does_not_backfill_child_that_precedes_reset_boundary(self, db):
        """A pre-marker branch cannot become a reset child after a later reopen cycle."""
        lane_key = "agent:main:telegram:dm:legacy-branch"
        create_session(db, "legacy_branch_parent", "telegram", session_key=lane_key)
        create_session(db,
            "legacy_branch_child",
            "telegram",
            session_key=lane_key,
            parent_session_id="legacy_branch_parent",
        )
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?", (100.0, "legacy_branch_child")
        )
        db._conn.commit()
        end_session(db, "legacy_branch_parent", "branched")
        reopen_session(db, "legacy_branch_parent")
        end_session(db, "legacy_branch_parent", "session_switch")
        db._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?", (200.0, "legacy_branch_parent")
        )
        db._conn.commit()

        reopen_session(db, "legacy_branch_parent")

        child = get_session(db, "legacy_branch_child")
        assert child is not None
        assert child["model_config"] is None

    def test_reopen_backfills_legacy_reset_child_of_cycled_parent(self, db):
        """A markerless reset child from an earlier boundary is still frozen after the parent was
        reopened and re-ended later (its started_at precedes the parent's current ended_at)."""
        lane_key = "agent:main:telegram:dm:cycled"
        create_session(db, "cycled_parent", "telegram", session_key=lane_key)
        end_session(db, "cycled_parent", "session_reset")
        create_session(db,
            "cycled_reset_child", "telegram", session_key=lane_key, parent_session_id="cycled_parent"
        )
        db._conn.execute(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?", ("cycled_parent",)
        )
        db._conn.commit()
        end_session(db, "cycled_parent", "session_switch")
        db._conn.execute(
            "UPDATE sessions SET ended_at = ended_at + 100 WHERE id = ?", ("cycled_parent",)
        )
        db._conn.commit()

        reopen_session(db, "cycled_parent")

        child = get_session(db, "cycled_reset_child")
        assert json.loads(child["model_config"]) == {"_reset_from": "cycled_parent"}
        assert "cycled_reset_child" in [row["id"] for row in list_sessions_rich(db, source="telegram")]

    def test_reset_parent_does_not_surface_unrelated_child(self, db):
        create_session(db,
            "reset_parent",
            "telegram",
            session_key="agent:main:telegram:dm:lane",
        )
        end_session(db, "reset_parent", "session_reset")
        create_session(db,
            "unrelated_child",
            "tool",
            session_key="delegate:other",
            parent_session_id="reset_parent",
        )

        listed = [row["id"] for row in list_sessions_rich(db)]
        assert "unrelated_child" not in listed
        assert session_count(db, exclude_children=True) == 1

    def test_resume_walker_does_not_cross_reset_boundary(self, db):
        """resolve_resume_session_id must not redirect a reset parent's resume
        into the post-reset conversation — that would restore the exact
        context the user reset away. Covers both the durable marker and the
        legacy markerless shape."""
        lane_key = "agent:main:telegram:dm:lane"
        # Marker shape (rows written by current gateway code).
        create_session(db, "walk_parent", "telegram", session_key=lane_key)
        append_message(db, "walk_parent", "user", "before reset")
        end_session(db, "walk_parent", "session_reset")
        create_session(db,
            "walk_child",
            "telegram",
            session_key=lane_key,
            parent_session_id="walk_parent",
            model_config={"_reset_from": "walk_parent"},
        )
        append_message(db, "walk_child", "user", "after reset")
        assert resolve_resume_session_id(db, "walk_parent") == "walk_parent"

        # Legacy markerless shape (pre-marker on-disk rows).
        lane2 = "agent:main:telegram:dm:lane2"
        create_session(db, "legacy_parent", "telegram", session_key=lane2)
        append_message(db, "legacy_parent", "user", "before reset")
        end_session(db, "legacy_parent", "session_reset")
        create_session(db,
            "legacy_child",
            "telegram",
            session_key=lane2,
            parent_session_id="legacy_parent",
        )
        append_message(db, "legacy_child", "user", "after reset")
        assert resolve_resume_session_id(db, "legacy_parent") == "legacy_parent"

    # Compression-tip following (the walker's original purpose) is pinned by
    # test_follows_compression_tip_when_parent_retains_messages.

    def test_delegate_subagent_marker_hides_orphaned_row(self, db):
        """``_delegate_from`` keeps delegate rows out of pickers after orphaning."""
        create_session(db, "parent", "cli")
        create_session(db,
            "delegate",
            "cli",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        append_message(db, "delegate", "user", "scan the repo")

        assert "delegate" not in [s["id"] for s in list_sessions_rich(db)]

        db._conn.execute(
            "UPDATE sessions SET parent_session_id = NULL WHERE id = ?", ("delegate",)
        )
        db._conn.commit()

        assert "delegate" not in [s["id"] for s in list_sessions_rich(db)]

    def test_delete_session_expected_targets_fail_closed_on_new_delegate(self, db, media_root):
        create_session(db, "parent", "cli")
        create_session(db,
            "delegate",
            "cli",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        create_session(db,
            "branch",
            "cli",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )

        expected_ids = get_session_delete_targets(db, "parent")
        assert expected_ids == ["parent", "delegate"]

        create_session(db,
            "late-delegate",
            "cli",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )

        assert (
            delete_session(db, "parent", expected_delete_ids=expected_ids, media_root=media_root) == ()
        )
        assert get_session(db, "parent") is not None
        assert get_session(db, "delegate") is not None
        assert get_session(db, "late-delegate") is not None
        assert get_session(db, "branch") is not None

    def test_subagent_session_still_hidden(self, db):
        """Sub-agent children (parent NOT ended with 'branched') remain hidden."""
        create_session(db, "root", "cli")
        create_session(db, "delegate", "cli", parent_session_id="root")

        sessions = list_sessions_rich(db)
        ids = [s["id"] for s in sessions]
        assert "delegate" not in ids, "Delegate sub-agent should not appear in default list"
        assert "root" in ids

    def test_rich_list_promotes_reset_and_branch_markers(self, db):
        """List rows expose _reset_from / _branched_from so UIs can tell a
        /new reset from a genuine /branch without reading model_config."""
        create_session(db, "parent", "cli")
        create_session(db,
            "reset_child",
            "cli",
            parent_session_id="parent",
            model_config={"_reset_from": "parent"},
        )
        create_session(db,
            "branch_child",
            "cli",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )

        by_id = {row["id"]: row for row in list_sessions_rich(db)}
        assert by_id["reset_child"]["_reset_from"] == "parent"
        assert not by_id["reset_child"].get("_branched_from")
        assert by_id["branch_child"]["_branched_from"] == "parent"
        assert not by_id["branch_child"].get("_reset_from")
        compact = {row["id"]: row for row in list_sessions_rich(db, compact_rows=True)}
        assert compact["reset_child"]["_reset_from"] == "parent"
        assert compact["branch_child"]["_branched_from"] == "parent"


# =========================================================================
# Concurrent write safety / lock contention fixes (#3139)
# =========================================================================

class TestConcurrentWriteSafety:
    def test_create_session_insert_or_ignore_is_idempotent(self, db):
        """create_session with the same ID twice must not raise (INSERT OR IGNORE)."""
        create_session(db, session_id="dup-1", source="cli", model="m")
        # Second call should be silent — no IntegrityError
        create_session(db, session_id="dup-1", source="gateway", model="m2")
        session = get_session(db, "dup-1")
        # Row should exist (first write wins with OR IGNORE)
        assert session is not None
        assert session["source"] == "cli"



# =========================================================================
# compact_rows — lightweight column projection (issue #47414)
# =========================================================================

class TestCompactRows:
    """list_sessions_rich and _get_session_rich_row with compact_rows=True
    must omit system_prompt but return all other metadata fields."""

    def _create(self, db, sid, *, system_prompt="big blob " * 500):
        create_session(db, session_id=sid, source="cli", model="m")
        update_system_prompt(db, sid, system_prompt)
        return sid

    def test_compact_rows_omits_system_prompt(self, db):
        self._create(db, "s1")
        rows = list_sessions_rich(db, compact_rows=True)
        assert len(rows) == 1
        assert "system_prompt" not in rows[0]

    def test_batch_compact_rows_omits_system_prompt_keeps_git_fields(self, db):
        """_get_session_rich_rows_batch(compact_rows=True) must apply the same
        schema-derived compact projection as the single-row path: no
        system_prompt blob, but git_branch/git_repo_root still present."""
        self._create(db, "s1", system_prompt="should be gone")
        update_session_cwd(db, "s1", "/tmp/w1", git_branch="main", git_repo_root="/tmp/w1")
        rows = _get_session_rich_rows_batch(db, ["s1"], compact_rows=True)
        assert set(rows) == {"s1"}
        row = rows["s1"]
        assert "system_prompt" not in row
        assert row["git_branch"] == "main"
        assert row["git_repo_root"] == "/tmp/w1"

    def test_compression_tip_projection_threads_compact_rows(self, db):
        """list_sessions_rich(compact_rows=True) must thread compact_rows
        through the batched tip-row fetch: the projected tip row must lack
        system_prompt but keep git metadata (guards the call site at the
        projection loop, not just the batch helper)."""
        import time as _time

        t0 = _time.time() - 3600
        create_session(db, "rootc", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "rootc"))
        append_message(db, "rootc", "user", "start")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 100, "compression", "rootc"),
        )
        create_session(db, "tipc", "cli", parent_session_id="rootc")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 101, "tipc"))
        append_message(db, "tipc", "user", "continuation")
        update_system_prompt(db, "tipc", "big blob " * 500)
        update_session_cwd(db, "tipc", "/tmp/w2", git_branch="dev", git_repo_root="/tmp/w2")
        db._conn.commit()

        rows = list_sessions_rich(db, source="cli", compact_rows=True)
        tip = next(s for s in rows if s["id"] == "tipc")
        assert tip["_lineage_root_id"] == "rootc"
        assert "system_prompt" not in tip
        assert tip["git_branch"] == "dev"
        assert tip["git_repo_root"] == "/tmp/w2"


class TestListCronJobRuns:
    """``list_cron_job_runs`` powers the desktop cron run-history endpoint.

    It must scope to exactly one job's runs via an id prefix range (not a
    substring), order newest-first, enrich with preview/last_active, and stay
    bounded by the requested window rather than the whole cron history.
    """

    def _seed_run(self, db, job_id: str, idx: int, started_at: float):
        sid = f"cron_{job_id}_20261011_{idx:06d}"  # the scheduler's cron_{job}_{YYYYmmdd_HHMMSS}
        create_session(db, session_id=sid, source="cron")
        append_message(db, sid, role="user", content=f"run {idx} for {job_id}")
        append_message(db, sid, role="assistant", content="done")
        end_session(db, sid, "completed")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?", (started_at, sid)
        )
        db._conn.commit()
        return sid

    def test_scopes_to_job_newest_first_and_enriched(self, db):
        base = 1_700_000_000.0
        # Target job: 5 runs, ascending started_at.
        for i in range(5):
            self._seed_run(db, "alpha", i, base + i * 60)
        # A different job that must not leak in.
        for i in range(3):
            self._seed_run(db, "beta", i, base + i * 60)

        runs = list_cron_job_runs(db, "alpha", limit=20)

        assert len(runs) == 5
        assert all(r["id"].startswith("cron_alpha_") for r in runs)
        # Newest started_at first.
        sts = [r["started_at"] for r in runs]
        assert sts == sorted(sts, reverse=True)
        # Enriched like list_sessions_rich.
        assert runs[0]["preview"].startswith("run 4 for alpha")
        assert runs[0]["last_active"] >= runs[0]["started_at"]

    def test_limit_and_offset_paging(self, db):
        base = 1_700_000_000.0
        for i in range(10):
            self._seed_run(db, "alpha", i, base + i * 60)

        page1 = list_cron_job_runs(db, "alpha", limit=4, offset=0)
        page2 = list_cron_job_runs(db, "alpha", limit=4, offset=4)

        assert len(page1) == 4
        assert len(page2) == 4
        assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})
        # Combined window is still newest-first and contiguous.
        combined = [r["started_at"] for r in page1 + page2]
        assert combined == sorted(combined, reverse=True)
