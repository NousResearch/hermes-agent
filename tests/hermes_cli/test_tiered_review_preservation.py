"""P4.2 — tiered review preservation regression tests.

Locks the existing ``_should_review`` contract on the corrected phase 2/3
base so P4.3/P4.4 work cannot silently weaken it:

  * ``full`` → mandatory review;
  * ``fast`` → deterministic 1-in-N sampling + ALWAYS review on last-run
    failure;
  * unclassified/NULL tier → no review;
  * ``kanban.tiered_review`` config gate default OFF (legacy review-
    everything behaviour for fast tier when off);
  * no second review-policy engine introduced.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kb.connect()
    yield conn
    conn.close()


def _task(conn, tier=None):
    return kb.create_task(
        conn, title="t", assignee="octacon",
        body="## Problem\nx\n## Success Criteria\ny", triage=True, tier=tier,
    )


def test_full_tier_always_reviewed_with_gate_on(env):
    tid = _task(env, tier="full")
    assert kb._should_review(env, "full", tid, kanban_cfg={"tiered_review": True})


def test_fast_sampled_bucket_stable(env):
    cfg = {"tiered_review": True, "review_sample_rate": 5}
    tid = _task(env, tier="fast")
    first = kb._should_review(env, "fast", tid, kanban_cfg=cfg)
    # Deterministic across repeated calls (sha256 bucket, not salted hash)
    for _ in range(3):
        assert kb._should_review(env, "fast", tid, kanban_cfg=cfg) is first


def test_fast_failed_run_always_reviewed(env):
    cfg = {"tiered_review": True, "review_sample_rate": 5}
    tid = _task(env, tier="fast")
    # Insert a non-completed run outcome
    env.execute(
        "INSERT INTO task_runs (task_id, profile, status, outcome, started_at) "
        "VALUES (?, 'octacon', 'failed', 'crashed', strftime('%s','now'))",
        (tid,),
    )
    env.commit()
    # Regardless of sample bucket, a failed run MUST be reviewed
    assert kb._should_review(env, "fast", tid, kanban_cfg=cfg) is True


def test_unclassified_no_review(env):
    tid = _task(env, tier=None)
    assert kb._should_review(env, None, tid, kanban_cfg={"tiered_review": True}) is False
    assert kb._should_review(env, "unclassified", tid, kanban_cfg={"tiered_review": True}) is False


def test_config_gate_defaults_off(env):
    """Default config (no kanban.tiered_review key) keeps legacy behaviour:
    fast-tier tasks are reviewed; only the sampling changes with the gate ON."""
    tid = _task(env, tier="fast")
    assert kb._should_review(env, "fast", tid, kanban_cfg={}) is True


def test_sampling_off_means_review_all_fast(env):
    tid = _task(env, tier="fast")
    assert kb._should_review(env, "fast", tid, kanban_cfg={}) is True


def test_unknown_tier_not_reviewed(env):
    tid = _task(env, tier="banana")
    assert kb._should_review(env, "banana", tid, kanban_cfg={"tiered_review": True}) is False