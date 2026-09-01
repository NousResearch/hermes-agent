"""P4.1 tests — deterministic shadow risk classification."""

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
    yield conn
    conn.close()


def _mk(conn, *, title="t", created_by="dashboard", tier=None, kind="task",
        priority=0, budget=None):
    return kb.create_task(
        conn, title=title, assignee="octacon",
        body="## Problem\nx\n## Success Criteria\ny", triage=True,
        created_by=created_by, tier=tier, task_kind=kind, priority=priority,
        max_runtime_seconds=budget,
    )


class TestSuggestion:
    def test_deterministic_suggestions(self, env):
        tid_fast = _mk(env, created_by="dashboard")
        tid_full = _mk(env, created_by="dashboard", priority=3)
        s1 = sc.suggest(env, tid_fast)
        s2 = sc.suggest(env, tid_fast)
        assert s1["suggested_tier"] == "fast"
        assert s1 == s2  # deterministic
        s3 = sc.suggest(env, tid_full)
        assert s3["suggested_tier"] == "full"
        assert set(s3) >= {"kind", "suggested_tier", "suggested_task_kind",
                           "reasons", "classifier_version"}

    def test_system_task_not_eligible(self, env):
        tid = _mk(env, created_by="feature-pipeline")
        assert sc.suggest(env, tid) is None
        tid2 = _mk(env, created_by=None)
        assert sc.suggest(env, tid2) is None

    def test_missing_task_none(self, env):
        assert sc.suggest(env, "t_nonexistent") is None


class TestInsertNonMutation:
    def test_insert_idempotent(self, env):
        tid = _mk(env)
        s = sc.suggest(env, tid)
        row1 = sc.insert_shadow_event(env, tid, s)
        row2 = sc.insert_shadow_event(env, tid, s)
        assert row1 is not None
        assert row2 is None  # idempotent per task+version

    def test_tier_status_assignee_review_path_unchanged(self, env):
        tid = _mk(env, tier="full")
        before = env.execute(
            "SELECT tier, status, assignee, task_kind FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        s = sc.suggest(env, tid)
        sc.insert_shadow_event(env, tid, s)
        after = env.execute(
            "SELECT tier, status, assignee, task_kind FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert tuple(before) == tuple(after)  # byte-for-data unchanged

    def test_human_override_retained(self, env):
        tid = _mk(env, tier="full")
        s = sc.suggest(env, tid)
        if s["suggested_tier"] != "full":
            # Suggestion said fast, human said full → human wins (nothing written)
            row = env.execute("SELECT tier FROM tasks WHERE id = ?", (tid,)).fetchone()
            assert row["tier"] == "full"
        d = sc.classification_disagreement(env.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone(), s)
        if s["suggested_tier"] != "full":
            assert d["material_disagreement"] is True
            assert d["human_tier"] == "full"

    def test_unclassified_task_unchanged(self, env):
        tid = _mk(env, tier=None)
        s = sc.suggest(env, tid)
        sc.insert_shadow_event(env, tid, s)
        d = sc.classification_disagreement(env.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone(), s)
        assert d is None  # no human choice yet → no disagreement claimed


class TestShadowDisabled:
    def test_disabled_means_no_event(self, env):
        """Shadow integration is caller-driven; absent caller → no event."""
        tid = _mk(env)
        events = kb.list_events(env, tid)
        assert not [e for e in events if e.kind == sc.EVENT_KIND]


class TestHygiene:
    def test_no_prompt_body_leakage(self, env):
        tid = _mk(env, title="SECRET-INTERNAL-body-leak-check",
                  created_by="ideabox")
        s = sc.suggest(env, tid)
        sc.insert_shadow_event(env, tid, s)
        events = [e for e in kb.list_events(env, tid) if e.kind == sc.EVENT_KIND]
        assert events
        payload = json.dumps(events[-1].payload)
        assert "SECRET-INTERNAL" not in payload  # no title/body text copied
        for reason in events[-1].payload["reasons"]:
            assert sc._BOUNDARY_TOKEN_RE.match(reason)

    def test_unguarded_tokens_rejected(self):
        with pytest.raises(ValueError):
            sc._bounded("bad token with spaces")