"""Regression: the promote pass must honour the breaker's own verdict.

``_record_task_failure`` resolves the failure threshold itself. The
systemic-crash path in ``detect_crashed_workers`` passes ``failure_limit=1``
once the same error fingerprint has hit three tasks, so a single crash parks
the card — that is the whole point of the fast trip during a fleet-wide
outage.

``recompute_ready`` then re-resolved the threshold from *its own* caller, and
every call site but ``dispatch_once`` passes nothing, i.e.
``DEFAULT_FAILURE_LIMIT`` = 2. A card blocked at 1 was promoted straight back
on ``1 < 2`` — inside the same dispatcher tick.

Measured on the live board over 40 h (2026-09-04 → 06)::

    71  gave_up events followed by `promoted` within 2 s
        every one: effective_limit=1, trigger_outcome=crashed
    15  cards cycled 3–4 times each

and on ``t_7a81e147`` the resurrected run then burned its full 3600 s ceiling
(6.5 M weighted units) before dying at the wall.

Two things the fix has to get right, and the first cut got wrong:

* read ``failures`` (the count the breaker blocked at), not
  ``effective_limit``. On the ``force_trip`` path those disagree by design —
  the protocol-violation streak reports a limit of 3 while the unified
  counter sits at 1–2 — and trusting the reported limit *promotes* a card the
  breaker had just parked;
* the verdict may only TIGHTEN the caller's threshold. Anything else lets a
  payload loosen a block the caller would have held.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

try:  # the dispatch half was carved out of kanban_db upstream
    from hermes_cli import kanban_db_dispatch as _kbd
except ImportError:  # pragma: no cover - older single-module layout
    _kbd = kb

_record_task_failure = _kbd._record_task_failure


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _trip(conn, task_id: str, *, limit: int | None, force: bool = False) -> bool:
    """Trip the breaker exactly the way the crash reclaim path does."""
    return _record_task_failure(
        conn,
        task_id,
        error="Copilot ACP process exited early: no subscription token",
        outcome="crashed",
        failure_limit=limit,
        force_trip=force,
        release_claim=False,
        end_run=False,
    )


def _make(conn, **kw) -> str:
    return kb.create_task(
        conn,
        title=kw.pop("title", "worker crashes on a dead token"),
        assignee=kw.pop("assignee", "default"),
        **kw,
    )


def _gave_up_payload(conn, task_id: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'gave_up' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return json.loads(row["payload"])


def _set_payload(conn, task_id: str, payload) -> None:
    conn.execute(
        "UPDATE task_events SET payload = ? WHERE task_id = ? AND kind = 'gave_up'",
        (payload if isinstance(payload, str) else json.dumps(payload), task_id),
    )
    conn.commit()


# --------------------------------------------------------------------------
# The reproduction
# --------------------------------------------------------------------------

def test_systemic_trip_survives_a_default_limit_sweep(kanban_home: Path) -> None:
    """Blocked at 1, swept by a caller that assumes 2."""
    with kb.connect() as conn:
        tid = _make(conn)
        assert _trip(conn, tid, limit=1) is True
        assert kb.get_task(conn, tid).status == "blocked"

        # Every call site except the dispatcher sweep looks like this.
        kb.recompute_ready(conn)

        assert kb.get_task(conn, tid).status == "blocked"


def test_systemic_trip_survives_an_explicit_higher_limit(kanban_home: Path) -> None:
    """``failure_limit=2`` passed explicitly must not override the verdict
    either — the dispatcher's config value is a default for cards that have
    not yet been judged, not an appeal court for ones that have."""
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=1)
        kb.recompute_ready(conn, failure_limit=2)
        assert kb.get_task(conn, tid).status == "blocked"


# --------------------------------------------------------------------------
# force_trip: reported limit ≠ applied threshold
# --------------------------------------------------------------------------

def test_force_trip_verdict_does_not_loosen_the_block(kanban_home: Path) -> None:
    """The regression the first cut of this fix introduced.

    ``detect_crashed_workers`` trips the protocol-violation streak with
    ``force_trip=True, failure_limit=3`` while ``consecutive_failures`` is 2,
    because below-budget violations deliberately do not consume the unified
    counter. Reading ``effective_limit`` (3) instead of ``failures`` (2) makes
    ``2 >= 3`` false and promotes the card — where the unpatched code held it
    on ``2 >= 2``.
    """
    with kb.connect() as conn:
        tid = _make(conn, title="worker keeps violating the exit protocol")
        _trip(conn, tid, limit=3)                    # 1 of 3 — no trip yet
        assert _trip(conn, tid, limit=3, force=True) is True

        payload = _gave_up_payload(conn, tid)
        assert payload["failures"] == 2 and payload["effective_limit"] == 3, payload
        assert kb.get_task(conn, tid).status == "blocked"

        kb.recompute_ready(conn)                     # DEFAULT_FAILURE_LIMIT = 2

        assert kb.get_task(conn, tid).status == "blocked"


def test_force_trip_verdict_cannot_beat_a_stricter_caller(kanban_home: Path) -> None:
    """A dispatcher configured stricter than the recorded verdict keeps its
    own answer — the lookup tightens, it never loosens."""
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=3)
        _trip(conn, tid, limit=3, force=True)
        kb.recompute_ready(conn, failure_limit=1)
        assert kb.get_task(conn, tid).status == "blocked"


def test_verdict_prefers_failures_over_reported_limit(kanban_home: Path) -> None:
    """Pin the field choice itself, not just its effect."""
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=1)
        _set_payload(conn, tid, {"failures": 2, "effective_limit": 9})
        assert kb._breaker_trip_threshold(conn, tid) == 2


def test_verdict_may_only_tighten_never_loosen(kanban_home: Path) -> None:
    """The ``min`` is load-bearing, not decoration.

    A card tripped at 5 (it carried ``max_retries=5`` then), the override was
    later dropped, and the counter now stands at 3 under a dispatcher
    configured at 2. Taking the recorded verdict straight would read
    ``3 >= 5`` — false — and release a card the caller's own limit holds.
    """
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=1)
        _set_payload(conn, tid, {"failures": 5, "effective_limit": 5})
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 3, max_retries = NULL, "
            "status = 'blocked' WHERE id = ?",
            (tid,),
        )
        conn.commit()

        kb.recompute_ready(conn, failure_limit=2)

        assert kb.get_task(conn, tid).status == "blocked"


def test_recency_is_event_order_not_wall_clock(kanban_home: Path) -> None:
    """``ORDER BY id`` is causal order; ``created_at`` is a second-resolution
    timestamp that ties, and a clock step or a back-dated row makes it lie.

    Here the unblock genuinely happened last (higher id) while carrying an
    older ``created_at``. Ordering by the clock would resurrect the cleared
    verdict.
    """
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=1)
        kb.unblock_task(conn, tid)
        conn.execute(
            "UPDATE task_events SET created_at = ("
            "  SELECT MIN(created_at) - 3600 FROM task_events WHERE task_id = ?"
            ") WHERE task_id = ? AND kind = 'unblocked'",
            (tid, tid),
        )
        conn.commit()

        assert kb._breaker_trip_threshold(conn, tid) is None


def test_verdict_falls_back_to_reported_limit(kanban_home: Path) -> None:
    """Payloads written before ``failures`` existed still carry a verdict."""
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=1)
        _set_payload(conn, tid, {"effective_limit": 1})
        assert kb._breaker_trip_threshold(conn, tid) == 1


# --------------------------------------------------------------------------
# Guards against over-reach
# --------------------------------------------------------------------------

def test_card_without_a_verdict_keeps_the_callers_limit(kanban_home: Path) -> None:
    """A blocked card with no ``gave_up`` event behaves exactly as before the
    fix. This is the pre-#28712 auto-recover path that ``_has_sticky_block``
    deliberately leaves open."""
    with kb.connect() as conn:
        tid = _make(conn)
        conn.execute(
            "UPDATE tasks SET status = 'blocked', consecutive_failures = 1 "
            "WHERE id = ?",
            (tid,),
        )
        conn.commit()
        assert kb._breaker_trip_threshold(conn, tid) is None

        kb.recompute_ready(conn)  # DEFAULT_FAILURE_LIMIT = 2, failures = 1

        assert kb.get_task(conn, tid).status == "ready"


def test_per_task_max_retries_still_outranks_the_verdict(kanban_home: Path) -> None:
    """A human who set ``max_retries`` has looked at the card; that stays the
    top of the resolution order."""
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=1)
        conn.execute("UPDATE tasks SET max_retries = 3 WHERE id = ?", (tid,))
        conn.commit()

        kb.recompute_ready(conn)

        assert kb.get_task(conn, tid).status == "ready"


def test_operator_unblock_clears_the_verdict(kanban_home: Path) -> None:
    """Otherwise the fix trades a runaway worker for a card nobody can start:
    an ``unblocked`` event is a deliberate reset of the breaker's judgement."""
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=1)
        assert kb._breaker_trip_threshold(conn, tid) == 1

        kb.unblock_task(conn, tid)

        assert kb._breaker_trip_threshold(conn, tid) is None
        assert kb.get_task(conn, tid).status in ("ready", "todo")


def test_verdict_rearms_after_an_unblock(kanban_home: Path) -> None:
    """Recency is decided by event id. Swapping the ``ORDER BY`` for
    ``created_at`` — same second, ties broken arbitrarily — would pass the
    test above and fail this one."""
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=1)
        kb.unblock_task(conn, tid)
        assert _trip(conn, tid, limit=1) is True

        assert kb._breaker_trip_threshold(conn, tid) == 1
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "blocked"


# --------------------------------------------------------------------------
# Payload hygiene — the reader sits inside an open write transaction
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        {"failures": None},
        {"effective_limit": "nonsense"},
        {"failures": 0, "effective_limit": 0},
        {"failures": -3},
        {"failures": True},
    ],
    ids=["broken-json", "null", "non-numeric", "zero", "negative", "bool"],
)
def test_verdict_reader_degrades_to_no_verdict(kanban_home: Path, payload) -> None:
    """Every unusable shape must read as "no verdict" (caller's limit), never
    raise: a raise here stalls the promote pass for the whole board.

    ``0`` and negatives would pin the card forever (``failures >= 0`` is
    always true) and ``True`` would silently mean 1 — a number nobody wrote.
    """
    with kb.connect() as conn:
        tid = _make(conn)
        _trip(conn, tid, limit=1)
        _set_payload(conn, tid, payload)
        assert kb._breaker_trip_threshold(conn, tid) is None
