"""C5 adversarial tests — shadow classifier human provenance fail-closed."""

from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import shadow_classifier as sc


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kb.connect()
    yield conn, home
    conn.close()


def _mk(conn, *, created_by, title="t"):
    return kb.create_task(
        conn, title=title, assignee="octacon",
        body="## Problem\nx\n## Success Criteria\ny", triage=True,
        created_by=created_by,
    )


class TestC5Provenance:
    def test_unknown_automation_tokens_not_human(self, env):
        """RED: webhook-7831 / cron-runner must not classify as human."""
        conn, home = env
        for token in ("webhook-7831", "cron-runner", "automation-bot",
                      "feature-pipeline", "swarm-7", "job-4242"):
            tid = _mk(conn, created_by=token)
            assert sc.suggest(conn, tid) is None, token

    def test_registry_verified_profile_author_is_human(self, env, tmp_path):
        conn, home = env
        # Deploy a registry containing the author profile
        gov = home / "governance"
        gov.mkdir(parents=True, exist_ok=True)
        import yaml
        (gov / "profile-registry.yaml").write_text(
            json.dumps({
                "schema_version": 1,
                "root": {"name": "KENSEI", "description": "root"},
                "profiles": [
                    {"name": "misa-misa", "kind": "lead", "parent": "KENSEI",
                     "lifecycle": "active", "domains": [], "gateway_unit": None},
                ],
            })
        )
        tid = _mk(conn, created_by="misa-misa")
        s = sc.suggest(conn, tid)
        assert s is not None  # registry-verified interactive profile author

    def test_unknown_profile_author_not_human(self, env):
        conn, home = env
        # No registry deployed → profile-author stamp cannot be verified
        tid = _mk(conn, created_by="some-unregistered-profile")
        assert sc.suggest(conn, tid) is None

    def test_trusted_seams_still_human(self, env):
        conn, home = env
        for token in ("dashboard", "ideabox", "cli"):
            tid = _mk(conn, created_by=token)
            assert sc.suggest(conn, tid) is not None, token

    def test_missing_identity_no_suggestion(self, env):
        conn, home = env
        tid = _mk(conn, created_by=None)
        assert sc.suggest(conn, tid) is None


class TestC5IdempotenceAllEvents:
    def test_idempotence_checks_all_prior_events(self, env):
        """RED: version idempotence must search ALL prior events, not just latest."""
        conn, home = env
        tid = _mk(conn, created_by="dashboard")
        s1 = sc.suggest(conn, tid)
        sc.insert_shadow_event(conn, tid, s1)
        # Simulate a later unrelated event of the same kind with a different
        # classifier version landing after ours (e.g. a newer classifier run).
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, ?, ?, strftime('%s','now'))",
            (tid, sc.EVENT_KIND,
             json.dumps({"classifier_version": "shadow-classifier-9"})),
        )
        conn.commit()
        # Same version re-insert must still be idempotent (existing found)
        assert sc.insert_shadow_event(conn, tid, s1) is None


class TestC5ReasonCleanup:
    def test_no_duplicate_kind_reasons(self, env):
        conn, home = env
        tid = _mk(conn, created_by="dashboard")
        s = sc.suggest(conn, tid)
        assert len([r for r in s["reasons"] if r.startswith("kind:")]) <= 1
        # bug/gate tasks previously emitted duplicate kind:* tokens
        tid_bug = _mk(conn, created_by="dashboard")
        conn.execute("UPDATE tasks SET task_kind = 'bug' WHERE id = ?", (tid,))
        conn.commit()
        s2 = sc.suggest(conn, tid)
        assert [r for r in s2["reasons"] if r.startswith("kind:")] == ["kind:bug"]