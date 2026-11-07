"""Validated rework handoffs through the ``active_pr`` respawn guard.

``check_respawn_guard`` holds a ``ready`` card for 24h when a recent comment
names a GitHub PR: re-spawning the implementer risks a duplicate PR (#111910).
After legitimate rework (CI red, new push, changes requested) the operator or
receiver re-queues the card with ``hermes kanban unblock --reason <marker>`` or
``hermes kanban promote --reason <marker>``. The event those calls emit is the
only authority the guard accepts from them:

* ``unblocked`` carries the ``reason`` in its payload (it used to be dropped, so
  the card sat in ``ready`` for 24h after every rework);
* ``promoted_manual`` already carried it — and now also counts as a deliberate
  re-queue for ``recent_success``.

The marker is ``decisao=<class> pr=<n> head=<sha> causa=<cause>
resume=<action>``. ``pr`` must name the very PR the newest comment armed the
guard with; ``decisao``/``causa`` are closed lists; ``resume`` must match the
event kind. Anything missing, duplicated or divergent keeps the guard, and the
``UNBLOCK: <reason>`` comment beside the event never lifts anything on its own —
prose any writer can post is not authority.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_URL = "https://github.com/example/repo/pull/1240"
HEAD = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _marker(**overrides: object) -> str:
    """The exact marker shape the emitter writes, one field overridable."""
    fields = {
        "decisao": "ci-vermelho",
        "pr": "1240",
        "head": HEAD,
        "causa": "failure",
        "resume": "unblock",
    }
    fields.update(overrides)
    return " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)


def _backdate_comments(conn, task_id: str, seconds: int = 60) -> None:
    """Move the comment back so a later event is strictly newer (1s granularity)."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = created_at - ? WHERE task_id = ?",
            (seconds, task_id),
        )


def _card_with_pr(conn, *, assignee: str = "dev") -> str:
    """A ready card whose newest comment names a PR — the guard is armed."""
    task_id = kb.create_task(conn, title="rework me", assignee=assignee)
    kb.add_comment(conn, task_id, author=assignee, body=f"Opened {PR_URL} for review.")
    _backdate_comments(conn, task_id)
    return task_id


def _events(conn, task_id: str, kind: str) -> list[dict | None]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall()
    return [json.loads(row["payload"]) if row["payload"] else None for row in rows]


# ---------------------------------------------------------------------------
# a) A validated marker on ``unblocked`` lifts the guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cause", ["failure", "timed_out", "action_required"])
def test_valid_marker_on_unblock_lifts_active_pr(kanban_home: Path, cause: str) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"

        assert kb.block_task(conn, task_id, reason="waiting on the red CI") is True
        assert kb.unblock_task(conn, task_id, reason=_marker(causa=cause)) is True

        assert kbd.check_respawn_guard(conn, task_id) is None


# ---------------------------------------------------------------------------
# b) No payload / no reason keeps the guard (fail closed)
# ---------------------------------------------------------------------------


def test_unblock_without_reason_keeps_active_pr(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="waiting on the red CI") is True
        assert kb.unblock_task(conn, task_id) is True

        # The payload is byte-identical to what this event always carried.
        assert _events(conn, task_id, "unblocked") == [None]
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_unblock_payload_without_reason_keeps_active_pr(kanban_home: Path) -> None:
    """A ``status``/``resume_status`` payload alone is not a marker."""
    with kbc.connect() as conn:
        parent_id = kb.create_task(conn, title="upstream", assignee="dev")
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="waiting on the parent") is True
        # Linked while blocked: the unblock lands on the parent-gated phase.
        kb.link_tasks(conn, parent_id, task_id)
        assert kb.unblock_task(conn, task_id) is True

        assert _events(conn, task_id, "unblocked") == [
            {"status": "todo", "resume_status": "ready"}
        ]
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


# ---------------------------------------------------------------------------
# c) The marker must name the PR the newest comment armed the guard with
# ---------------------------------------------------------------------------


def test_marker_naming_another_pr_keeps_active_pr(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="rework") is True
        assert kb.unblock_task(conn, task_id, reason=_marker(pr="999")) is True

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


# ---------------------------------------------------------------------------
# d) ``head`` must be a 7..40 character hex SHA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "head",
    [
        "-",                                        # emitter has no SHA: fail closed
        "abc123",                                   # 6 chars: too short
        "0" * 41,                                   # 41 chars: too long
        "zzzzzzz",                                  # not hex
        "0123456789abcdef0123456789abcdef01234567z",  # hex + junk
    ],
)
def test_invalid_head_keeps_active_pr(kanban_home: Path, head: str) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="rework") is True
        assert kb.unblock_task(conn, task_id, reason=_marker(head=head)) is True

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


# ---------------------------------------------------------------------------
# e) ``decisao`` and ``causa`` are closed lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"decisao": "reabrir-tudo"},   # not a decision the emitter can make
        {"causa": "porque-sim"},       # not an observation it can report
        {"decisao": None},             # missing entirely
        {"causa": None},
    ],
)
def test_decision_or_cause_outside_the_allowlist_keeps_active_pr(
    kanban_home: Path, fields: dict,
) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="rework") is True
        assert kb.unblock_task(conn, task_id, reason=_marker(**fields)) is True

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


# ---------------------------------------------------------------------------
# f) Strict ordering: a same-second event stays guarded
# ---------------------------------------------------------------------------


def test_marker_in_the_same_second_as_the_pr_comment_keeps_active_pr(
    kanban_home: Path,
) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="rework") is True
        assert kb.unblock_task(conn, task_id, reason=_marker()) is True

        # Timestamps are second-granular: force the tie the guard must refuse.
        comment_ts = conn.execute(
            "SELECT created_at FROM task_comments WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'unblocked'",
                (comment_ts, task_id),
            )

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


# ---------------------------------------------------------------------------
# g) Generic ``promoted`` (the automatic hand-back) never lifts
# ---------------------------------------------------------------------------


def test_generic_promoted_event_never_lifts_active_pr(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "promoted", {"from": "todo", "to": "ready"})
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


# ---------------------------------------------------------------------------
# h) ``promoted_manual`` needs the full marker, and only for the same PR
# ---------------------------------------------------------------------------


def test_manual_promotion_without_marker_keeps_active_pr(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="rework") is True
        ok, err = kb.promote_task(
            conn, task_id, actor="operator",
            reason=_marker(pr=None, head=None, resume="promote"),
        )
        assert (ok, err) == (True, None)

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


@pytest.mark.parametrize("cause", ["failure", "timed_out", "action_required"])
def test_manual_promotion_with_valid_marker_lifts_active_pr(
    kanban_home: Path, cause: str,
) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="rework") is True
        ok, err = kb.promote_task(
            conn, task_id, actor="operator", reason=_marker(causa=cause, resume="promote"),
        )
        assert (ok, err) == (True, None)

        assert kbd.check_respawn_guard(conn, task_id) is None


def test_unblock_marker_cannot_authorise_a_promotion(kanban_home: Path) -> None:
    """``resume`` must match the kind: an unblock marker on a promotion is inert."""
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="rework") is True
        ok, _ = kb.promote_task(
            conn, task_id, actor="operator", reason=_marker(resume="unblock"),
        )
        assert ok is True

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


# ---------------------------------------------------------------------------
# i) ``promoted_manual`` counts as a deliberate re-queue for ``recent_success``
# ---------------------------------------------------------------------------


def test_manual_promotion_lifts_recent_success(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="run it again", assignee="dev")
        assert kb.claim_task(conn, task_id) is not None
        assert kb.complete_task(conn, task_id, summary="done") is True

        # Backdate the completed run so the promotion is unambiguously newer.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET ended_at = ended_at - 60 WHERE task_id = ?",
                (task_id,),
            )
            conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (task_id,))
        assert kbd.check_respawn_guard(conn, task_id) == "recent_success"

        ok, err = kb.promote_task(conn, task_id, actor="operator", reason="reviver")
        assert (ok, err) == (True, None)
        assert kbd.check_respawn_guard(conn, task_id) is None


# ---------------------------------------------------------------------------
# j) Existing handoffs keep working
# ---------------------------------------------------------------------------


def test_review_lane_early_return_is_unchanged(kanban_home: Path) -> None:
    """The review lane never runs the ``recent_success``/``active_pr`` guards."""
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="rework") is True
        assert kb.unblock_task(conn, task_id, reason="prose only, no marker") is True

        assert kbd.check_respawn_guard(conn, task_id, lane="review") is None
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_changes_requested_still_lifts_active_pr(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="changes requested", assignee="dev")
        claimed = kb.claim_task(conn, task_id)
        kb.add_comment(conn, task_id, author="dev", body=f"Opened {PR_URL} for review.")
        _backdate_comments(conn, task_id)
        assert kb.request_review(
            conn, task_id, summary="PR ready", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        review_claim = kb.claim_review_task(conn, task_id)
        ok, implementer = kb.request_changes(
            conn, task_id, reason="fix tests", expected_run_id=review_claim.current_run_id,
        )
        assert (ok, implementer) == (True, "dev")
        assert kbd.check_respawn_guard(conn, task_id) is None


def test_review_reopened_still_lifts_active_pr(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="review reopened", assignee="dev")
        claimed = kb.claim_task(conn, task_id)
        kb.add_comment(conn, task_id, author="dev", body=f"Opened {PR_URL} for review.")
        _backdate_comments(conn, task_id)
        assert kb.request_review(
            conn, task_id, summary="PR ready", reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert kb.reopen_review_task(conn, task_id) is True
        assert kbd.check_respawn_guard(conn, task_id) is None


def test_profile_handoff_is_still_the_only_assignment_that_lifts(
    kanban_home: Path,
) -> None:
    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.assign_task(conn, task_id, "dev") is True
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"

        assert kb.assign_task(conn, task_id, "closer") is True
        assert kbd.check_respawn_guard(conn, task_id) is None


# ---------------------------------------------------------------------------
# k) Kernel compatibility: old call sites keep the old payload
# ---------------------------------------------------------------------------


def test_unblock_without_reason_payload_is_unchanged(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="plain unblock", assignee="dev")
        assert kb.block_task(conn, task_id, reason="blocked") is True
        assert kb.unblock_task(conn, task_id) is True

        payload = _events(conn, task_id, "unblocked")
        assert payload == [None]
        assert kb.get_task(conn, task_id).status == "ready"


def test_unblock_reason_is_added_next_to_the_existing_keys(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        parent_id = kb.create_task(conn, title="upstream", assignee="dev")
        task_id = kb.create_task(conn, title="gated child", assignee="dev")
        assert kb.block_task(conn, task_id, reason="blocked") is True
        kb.link_tasks(conn, parent_id, task_id)
        assert kb.unblock_task(conn, task_id, reason=_marker()) is True

        assert _events(conn, task_id, "unblocked") == [
            {"status": "todo", "resume_status": "ready", "reason": _marker()}
        ]
        # The status semantics are untouched: still gated on the open parent.
        assert kb.get_task(conn, task_id).status == "todo"


# ---------------------------------------------------------------------------
# l) A lifted guard spawns the card exactly once
# ---------------------------------------------------------------------------


def test_validated_handoff_claims_the_card_once(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})

    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 424242

    with kbc.connect() as conn:
        task_id = _card_with_pr(conn)
        assert kb.block_task(conn, task_id, reason="rework") is True
        assert kb.unblock_task(conn, task_id, reason=_marker()) is True

        first = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert [task for task, _assignee, _ws in first.spawned].count(task_id) == 1
        assert spawns.count(task_id) == 1
        # One claim, one open run: ``block_task`` closes its own run row, so the
        # count that matters is the OPEN one (a second spawn would add another).
        claims = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'claimed'",
            (task_id,),
        ).fetchone()[0]
        assert claims == 1
        open_runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
            (task_id,),
        ).fetchone()[0]
        assert open_runs == 1
        assert kb.get_task(conn, task_id).status == "running"

        # The card is claimed now: a second tick finds nothing to spawn.
        second = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert task_id not in [task for task, _assignee, _ws in second.spawned]
        assert spawns.count(task_id) == 1
