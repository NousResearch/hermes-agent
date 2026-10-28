"""Usage attribution of the review fork through the keyword-only recorder.

``_record_review_usage_to_parent(parent, usage, *, task, session_id)`` is the one chokepoint
every cache-parity fork bills through: the automatic review as ``task="background_review"``
under ``review_session_id`` (a cold ``/refine`` runs under its own session id, not the parent's
current one), ``/btw`` and plugin side questions under their own tasks. The review bills from
``_run_review_fork``'s ``finally``, outside the recorder's own try/except: a call that does not
bind against the signature raises there and skips the fork's unregister and the run's exit
publication, so every later review on the session is refused as already active.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent.background_review as bg
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _usage_rows(db, session_id):
    with db._lock:
        rows = db._conn.execute(
            "SELECT * FROM session_model_usage WHERE session_id = ? ORDER BY task",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _usage(**overrides):
    base = {
        "model": "test-model",
        "provider": "test-provider",
        "base_url": None,
        "input_tokens": 1200,
        "output_tokens": 40,
        "cache_read_tokens": 1000,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "api_calls": 1,
        "estimated_cost_usd": None,
    }
    base.update(overrides)
    return base


def test_task_and_session_id_keywords_land_the_row_under_them(db):
    """A fork billed to another session under its own task never touches the parent's rows."""
    db.create_session("sess-parent", source="cli")
    db.create_session("sess-other", source="cli")
    parent = SimpleNamespace(_session_db=db, session_id="sess-parent")

    bg._record_review_usage_to_parent(
        parent,
        _usage(),
        task="plugin_side_question:next-prompt",
        session_id="sess-other",
    )

    assert _usage_rows(db, "sess-parent") == []
    (row,) = _usage_rows(db, "sess-other")
    assert row["task"] == "plugin_side_question:next-prompt"
    assert row["api_call_count"] == 1
    assert row["input_tokens"] == 1200


class _StubDB:
    """Only what the review path reaches: no turn-lease protocol, so no durable lease is tried."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def record_auxiliary_usage(self, session_id, task, **kwargs):
        self.calls.append((session_id, task, kwargs))


def test_review_fork_finally_bills_the_review_session_under_background_review():
    """The fork phase attributes its usage to the session it reviewed (a cold /refine's own id)
    as ``background_review`` and still publishes the run's exit afterwards."""
    parent = SimpleNamespace(_session_db=_StubDB(), session_id="sess-parent")
    finished: list = []

    def fake_build(
        agent,
        task_cfg=None,
        *,
        max_iterations,
        write_origin="background_review",
        session_id=None,
        review_prompt=None,
    ):
        fork = SimpleNamespace(
            _memory_enabled=True,
            _user_profile_enabled=False,
            _memory_write_origin=write_origin,
            session_id=session_id,
            run_conversation=lambda **kw: None,
            _session_messages=[],
        )
        return fork, {}, False

    noop = lambda *a, **k: None
    with (
        patch.object(bg, "build_cache_parity_fork", fake_build),
        patch.object(bg, "_track_review_fork", noop),
        patch.object(bg, "_snapshot_review_usage", lambda a: _usage()),
        patch.object(
            bg, "finish_background_review_run", lambda agent, run: finished.append(run)
        ),
        patch.object(bg, "_release_fork_clients", noop),
    ):
        bg._run_review_fork(
            parent,
            [],
            "review prompt",
            None,
            None,
            bg._ReviewForkState(),
            True,
            False,
            review_session_id="sess-refine-cold",
        )

    ((session_id, task, kwargs),) = parent._session_db.calls
    assert (session_id, task) == ("sess-refine-cold", "background_review")
    assert kwargs["api_call_count"] == 1 and kwargs["model"] == "test-model"
    assert finished == [None]  # the exit was published after the recorder returned
