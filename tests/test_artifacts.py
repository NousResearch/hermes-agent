"""Tests for Phase 1 Hermes Code Mode: artifacts/diffs per session."""

import time
import pytest
from pathlib import Path

from hermes_state import SessionDB, count_diff_changes


# =============================================================================
# count_diff_changes
# =============================================================================

SAMPLE_DIFF = """\
diff --git a/test.ts b/test.ts
index abc..def 100644
--- a/test.ts
+++ b/test.ts
@@ -1,3 +1,4 @@
 const a = 1;
-const b = 2;
+const b = 3;
+const c = 4;
"""


class TestCountDiffChanges:
    def test_headers_not_counted(self):
        additions, deletions = count_diff_changes(SAMPLE_DIFF)
        assert additions == 2
        assert deletions == 1

    def test_empty_diff(self):
        assert count_diff_changes("") == (0, 0)

    def test_plus_plus_plus_skipped(self):
        diff = "+++ b/file.py\n+actual add\n"
        additions, deletions = count_diff_changes(diff)
        assert additions == 1  # only the real '+' line
        assert deletions == 0

    def test_minus_minus_minus_skipped(self):
        diff = "--- a/file.py\n-actual remove\n"
        additions, deletions = count_diff_changes(diff)
        assert additions == 0
        assert deletions == 1  # only the real '-' line

    def test_context_lines_not_counted(self):
        diff = " context line\n+add\n-remove\n"
        additions, deletions = count_diff_changes(diff)
        assert additions == 1
        assert deletions == 1

    def test_no_changes(self):
        diff = "diff --git a/f b/f\nindex abc 100644\n--- a/f\n+++ b/f\n"
        assert count_diff_changes(diff) == (0, 0)


# =============================================================================
# SessionDB.create_artifact + get_artifacts_by_session
# =============================================================================


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_artifacts.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


class TestArtifactPersistence:
    def test_create_and_retrieve_artifact(self, db):
        db.create_session("s1", "cli")
        artifact = db.create_artifact(
            session_id="s1",
            tool_name="patch",
            path="src/example.ts",
            status="modified",
            diff=SAMPLE_DIFF,
        )
        assert artifact["id"]
        assert artifact["tool_name"] == "patch"
        assert artifact["path"] == "src/example.ts"
        assert artifact["status"] == "modified"
        assert artifact["additions"] == 2
        assert artifact["deletions"] == 1

    def test_get_artifacts_by_session_returns_from_table(self, db):
        db.create_session("s1", "cli")
        db.create_artifact(
            session_id="s1",
            tool_name="patch",
            path="foo.py",
            status="modified",
            diff=SAMPLE_DIFF,
        )
        artifacts = db.get_artifacts_by_session("s1")
        assert len(artifacts) == 1
        assert artifacts[0]["path"] == "foo.py"
        assert artifacts[0]["additions"] == 2
        assert artifacts[0]["deletions"] == 1

    def test_empty_session_returns_empty_list(self, db):
        db.create_session("s1", "cli")
        artifacts = db.get_artifacts_by_session("s1")
        assert artifacts == []

    def test_multiple_artifacts_ordered_by_timestamp(self, db):
        db.create_session("s1", "cli")
        db.create_artifact(
            session_id="s1", tool_name="patch", path="a.py", status="modified"
        )
        time.sleep(0.01)
        db.create_artifact(
            session_id="s1", tool_name="patch", path="b.py", status="added"
        )
        artifacts = db.get_artifacts_by_session("s1")
        assert len(artifacts) == 2
        assert artifacts[0]["path"] == "a.py"
        assert artifacts[1]["path"] == "b.py"

    def test_artifacts_isolated_per_session(self, db):
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")
        db.create_artifact(
            session_id="s1", tool_name="write_file", path="x.py", status="added"
        )
        db.create_artifact(
            session_id="s2", tool_name="write_file", path="y.py", status="added"
        )
        assert len(db.get_artifacts_by_session("s1")) == 1
        assert len(db.get_artifacts_by_session("s2")) == 1
        assert db.get_artifacts_by_session("s1")[0]["path"] == "x.py"

    def test_explicit_additions_deletions_override_diff_count(self, db):
        db.create_session("s1", "cli")
        artifact = db.create_artifact(
            session_id="s1",
            tool_name="patch",
            path="f.py",
            status="modified",
            diff=SAMPLE_DIFF,
            additions=99,
            deletions=88,
        )
        assert artifact["additions"] == 99
        assert artifact["deletions"] == 88

    def test_write_file_artifact_added_status(self, db):
        db.create_session("s1", "cli")
        artifact = db.create_artifact(
            session_id="s1",
            tool_name="write_file",
            path="new_file.py",
            status="added",
            diff="",
        )
        assert artifact["status"] == "added"
        assert artifact["additions"] == 0
        assert artifact["deletions"] == 0

    def test_tool_call_id_stored(self, db):
        db.create_session("s1", "cli")
        artifact = db.create_artifact(
            session_id="s1",
            tool_name="patch",
            path="f.py",
            status="modified",
            tool_call_id="call_abc123",
        )
        fetched = db.get_artifacts_by_session("s1")
        assert fetched[0]["tool_call_id"] == "call_abc123"


# =============================================================================
# Legacy fallback: artifacts from messages
# =============================================================================


class TestLegacyArtifactFallback:
    def test_fallback_to_messages_when_no_artifacts_table_data(self, db):
        """Sessions with no rows in artifacts table fall back to message extraction."""
        db.create_session("s1", "cli")

        # Insert a fake tool result message simulating a legacy patch result
        legacy_content = {
            "success": True,
            "diff": SAMPLE_DIFF,
            "files_modified": ["legacy.py"],
        }
        import json

        db.append_message(
            "s1",
            role="tool",
            content=json.dumps(legacy_content),
            tool_name="patch",
            tool_call_id="call_legacy_1",
        )

        artifacts = db.get_artifacts_by_session("s1")
        assert len(artifacts) == 1
        assert artifacts[0]["path"] == "legacy.py"
        assert artifacts[0]["status"] == "modified"
        assert artifacts[0]["additions"] == 2
        assert artifacts[0]["deletions"] == 1

    def test_new_table_takes_precedence_over_messages(self, db):
        """If artifacts table has data, messages are NOT used as fallback."""
        db.create_session("s1", "cli")

        # Insert legacy message data
        import json

        db.append_message(
            "s1",
            role="tool",
            content=json.dumps(
                {"success": True, "diff": "", "files_modified": ["msg.py"]}
            ),
            tool_name="patch",
        )

        # Also insert real artifact
        db.create_artifact(
            session_id="s1",
            tool_name="patch",
            path="real.py",
            status="modified",
        )

        artifacts = db.get_artifacts_by_session("s1")
        # Should only return the real artifact, not the message-extracted one
        assert len(artifacts) == 1
        assert artifacts[0]["path"] == "real.py"


# =============================================================================
# Schema migration
# =============================================================================


class TestArtifactsSchema:
    def test_artifacts_table_exists(self, db):
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artifacts'"
        )
        assert cursor.fetchone() is not None

    def test_artifacts_session_index_exists(self, db):
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_artifacts_session_id'"
        )
        assert cursor.fetchone() is not None

    def test_schema_version_is_18(self, db):
        cursor = db._conn.execute("SELECT version FROM schema_version")
        assert cursor.fetchone()[0] == 18
