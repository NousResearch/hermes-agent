"""Test schema migration on GET /api/profiles/sessions."""
import sqlite3
import pytest


class TestProfileSessionSchemaMigration:
    """Verify that lagging profile DBs are auto-migrated instead of
    silently returning zero sessions.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        self.monkeypatch = monkeypatch
        self.hermes_home = get_hermes_home()

    def _create_lagging_db(self, db_path):
        """Create a state.db whose sessions table lacks the ``archived``
        column, simulating a profile DB from before the feature was added.

        Strategy: create the DB with full schema via SessionDB, add a session,
        then DROP the archived column.
        """
        from hermes_state import SessionDB

        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = SessionDB(db_path=db_path, read_only=False)
        try:
            db.create_session(session_id="laggy-session-1", source="cli")
            db.append_message(
                session_id="laggy-session-1", role="user", content="hello from laggy"
            )
        finally:
            db.close()

        # Drop the archived column to simulate an older schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("ALTER TABLE sessions DROP COLUMN archived")
        conn.commit()
        conn.close()

    def test_profiles_sessions_auto_migrates_lagging_schema(self):
        """When a profile DB is missing the ``archived`` column, the
        cross-profile endpoint auto-migrates it instead of silently
        returning zero sessions for that profile.
        """
        from hermes_cli import profiles as profiles_mod

        # Create a fake "laggy" profile with an outdated DB
        profile_dir = self.hermes_home / "profiles" / "laggy"
        db_path = profile_dir / "state.db"
        self._create_lagging_db(db_path)

        # Monkeypatch list_profiles to include our laggy profile
        self.monkeypatch.setattr(
            profiles_mod,
            "list_profiles",
            lambda: [
                profiles_mod.ProfileInfo(
                    name="default",
                    path=self.hermes_home,
                    is_default=True,
                    gateway_running=False,
                ),
                profiles_mod.ProfileInfo(
                    name="laggy",
                    path=profile_dir,
                    is_default=False,
                    gateway_running=False,
                ),
            ],
        )

        # Call the endpoint — should trigger migration and return the session
        resp = self.client.get("/api/profiles/sessions?limit=20&min_messages=0")
        assert resp.status_code == 200
        data = resp.json()

        # No errors for the laggy profile
        laggy_errors = [
            e for e in data.get("errors", []) if e.get("profile") == "laggy"
        ]
        assert len(laggy_errors) == 0, f"Unexpected errors: {laggy_errors}"

        # The laggy profile's session should appear (migration happened)
        sessions = data["sessions"]
        laggy = [s for s in sessions if s.get("profile") == "laggy"]
        assert len(laggy) == 1
        assert laggy[0]["id"] == "laggy-session-1"

    def test_profiles_sessions_migration_is_idempotent(self):
        """Second call to the endpoint should work without re-migrating."""
        from hermes_cli import profiles as profiles_mod

        profile_dir = self.hermes_home / "profiles" / "laggy3"
        db_path = profile_dir / "state.db"
        self._create_lagging_db(db_path)

        self.monkeypatch.setattr(
            profiles_mod,
            "list_profiles",
            lambda: [
                profiles_mod.ProfileInfo(
                    name="default",
                    path=self.hermes_home,
                    is_default=True,
                    gateway_running=False,
                ),
                profiles_mod.ProfileInfo(
                    name="laggy3",
                    path=profile_dir,
                    is_default=False,
                    gateway_running=False,
                ),
            ],
        )

        # First call triggers migration
        resp1 = self.client.get("/api/profiles/sessions?limit=20&min_messages=0")
        assert resp1.status_code == 200

        # Second call should also work (no re-migration needed)
        resp2 = self.client.get("/api/profiles/sessions?limit=20&min_messages=0")
        assert resp2.status_code == 200
        data2 = resp2.json()
        laggy = [s for s in data2["sessions"] if s.get("profile") == "laggy3"]
        assert len(laggy) == 1
        assert len(data2.get("errors", [])) == 0