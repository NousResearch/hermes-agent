from hermes_state import SessionDB


def test_gateway_topic_context_upsert_and_fetch_is_thread_scoped(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    ctx = db.upsert_gateway_topic_context(
        platform="telegram",
        chat_id="-1001",
        thread_id="205",
        profile="default",
        chat_name="Hermes",
        topic_name="Hermes docs patch",
        purpose="Keep topic context after /new.",
        skills=["hermes-agent", "obsidian"],
    )

    assert ctx["topic_name"] == "Hermes docs patch"
    assert ctx["purpose"] == "Keep topic context after /new."
    assert ctx["skills"] == ["hermes-agent", "obsidian"]

    fetched = db.get_gateway_topic_context(
        platform="telegram",
        chat_id="-1001",
        thread_id="205",
        profile="default",
    )
    assert fetched == ctx

    assert db.get_gateway_topic_context(
        platform="telegram",
        chat_id="-1001",
        thread_id="206",
        profile="default",
    ) is None


def test_gateway_topic_context_partial_update_preserves_existing_lists(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.upsert_gateway_topic_context(
        platform="telegram",
        chat_id="-1001",
        thread_id="205",
        topic_name="Old",
        purpose="Old purpose",
        skills=["hermes-agent"],
    )

    updated = db.upsert_gateway_topic_context(
        platform="telegram",
        chat_id="-1001",
        thread_id="205",
        topic_name="New",
        purpose="New purpose",
    )

    assert updated["topic_name"] == "New"
    assert updated["purpose"] == "New purpose"
    assert updated["skills"] == ["hermes-agent"]


def test_gateway_topic_context_workdir_set_and_clear(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    # Setter creates the row when absent.
    ctx = db.set_gateway_topic_context_workdir(
        platform="telegram",
        chat_id="-1001",
        thread_id="205",
        profile="default",
        workdir="/srv/project",
    )
    assert ctx["workdir"] == "/srv/project"

    # Clearing writes NULL (COALESCE upsert could not express this).
    cleared = db.set_gateway_topic_context_workdir(
        platform="telegram",
        chat_id="-1001",
        thread_id="205",
        profile="default",
        workdir=None,
    )
    assert cleared["workdir"] is None


def test_gateway_topic_context_workdir_preserved_across_upsert(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_gateway_topic_context_workdir(
        platform="telegram",
        chat_id="-1001",
        thread_id="205",
        workdir="/srv/project",
    )

    updated = db.upsert_gateway_topic_context(
        platform="telegram",
        chat_id="-1001",
        thread_id="205",
        topic_name="Notes",
        purpose="Project workspace.",
    )

    assert updated["topic_name"] == "Notes"
    assert updated["workdir"] == "/srv/project"


def test_gateway_topic_context_migration_adds_workdir_column(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    # Hand-create the table with the OLD schema (no workdir column).
    with db._lock:
        db._conn.execute(
            """
            CREATE TABLE gateway_topic_contexts (
                platform TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                profile TEXT NOT NULL DEFAULT 'default',
                chat_name TEXT,
                topic_name TEXT,
                purpose TEXT,
                skills_json TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (platform, chat_id, thread_id, profile)
            )
            """
        )
        db._conn.execute(
            """
            INSERT INTO gateway_topic_contexts (
                platform, chat_id, thread_id, profile, topic_name,
                created_at, updated_at
            ) VALUES ('telegram', '-1001', '205', 'default', 'Old', 1, 1)
            """
        )
        db._conn.commit()

    db.apply_gateway_topic_context_migration()

    with db._lock:
        cols = {row[1] for row in db._conn.execute(
            "PRAGMA table_info(gateway_topic_contexts)"
        ).fetchall()}
    assert "workdir" in cols

    ctx = db.get_gateway_topic_context(
        platform="telegram", chat_id="-1001", thread_id="205"
    )
    assert "workdir" in ctx
    assert ctx["workdir"] is None

    # Idempotent: a second call must not raise or duplicate the column.
    db.apply_gateway_topic_context_migration()


def test_gateway_topic_context_handles_corrupt_json_as_empty_lists(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_gateway_topic_context_migration()

    for thread_id, corrupt_skills in (("205", "not-json"), ("206", "{}")):
        with db._lock:
            db._conn.execute(
                """
                INSERT INTO gateway_topic_contexts (
                    platform, chat_id, thread_id, profile, skills_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, 1)
                """,
                ("telegram", "-1001", thread_id, "default", corrupt_skills),
            )
            db._conn.commit()

        ctx = db.get_gateway_topic_context(
            platform="telegram",
            chat_id="-1001",
            thread_id=thread_id,
        )
        assert ctx["skills"] == []
