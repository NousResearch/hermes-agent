"""A ``timed_out`` notice must name the cause its event payload records.

One defect, three placed facts (card t_1080ad83):

1. ``agent.turn_finalizer._record_kanban_budget_exhausted`` records the cause of an
   iteration-budget exhaustion as ``event_payload_extra={"budget_used": …, "budget_max": …}``
   and no ``limit_seconds`` — only ``enforce_max_runtime`` (the runtime cap) sets that key.
2. ``_record_task_failure``'s non-trip path dropped the extra, so the ``timed_out`` event it
   appended carried only the free-text ``error``/``failures``/``retry_status``. Measured on the
   ops board: ``t_82490443`` event 96801, and all 19 ``timed_out`` events in the 3 days to
   2026-09-27 carried no ``limit_seconds``.
3. Both notice formatters then read the absent key as ``0`` and printed a runtime cap the worker
   never hit — ``max_runtime=0s`` on the board, "its time limit" on Telegram — so the recorded
   cause was invisible exactly when the operator needed to act on it.

The recorder and its two formatters are one invariant, so they are pinned in one file: the
production caller is driven into a real (temp) board and the payload it actually recorded is
rendered by both formatters.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

# A runtime-cap stop: the ONLY cause that records ``limit_seconds``.
CAP_PAYLOAD = {"limit_seconds": 900, "failures": 1, "retry_status": "ready"}
# A timeout whose caller recorded no cause fields at all.
CAUSELESS_PAYLOAD = {"failures": 1, "retry_status": "ready"}


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated board the production caller reaches through ``kbc.connect()``.

    ``HERMES_KANBAN_DB`` pins the DB ahead of ``HERMES_HOME`` in ``kanban_db_path()``, so a
    worker shell that exports it (every dispatched worker does) would otherwise point these
    writes at the live board; the conftest ``kanban_write_guard`` turns that into a test error,
    but the pin is dropped here so the tests exercise the temp board they mean to.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _exhaust_budget(conn, *, used: int = 200, maximum: int = 200):
    """Exhaust a claimed card's iteration budget through the production caller.

    Returns ``(task_id, event_row)`` for the ``timed_out`` event it recorded.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    tid = kb.create_task(conn, title="out of budget", assignee="platform-worker")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    _record_kanban_budget_exhausted(tid, used, maximum, logging.getLogger("test-budget"))
    event = conn.execute(
        "SELECT id, kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    return tid, event


def _run_metadata(conn, task_id: str) -> dict:
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,)
    ).fetchone()
    return json.loads(row["metadata"]) if row and row["metadata"] else {}


def _board_text(task_id: str, payload: dict) -> str:
    from tui_gateway.session_notifications import _format_kanban_event_text

    task = SimpleNamespace(title="out of budget", assignee="platform-worker")
    return _format_kanban_event_text(
        {"task_id": task_id}, task, SimpleNamespace(kind="timed_out", payload=payload), "ops"
    )


def _telegram_text(payload: dict) -> str:
    from gateway.kanban_watchers_notifier import _EVENT_FORMATTERS

    msg, _, _ = _EVENT_FORMATTERS["timed_out"](
        SimpleNamespace(payload=payload), SimpleNamespace(head="t_abc123")
    )
    return msg


def test_budget_exhaustion_records_the_cause_on_the_event(kanban_home):
    """The extra the caller passes reaches the event payload, not the floor."""
    with kbc.connect() as conn:
        _tid, event = _exhaust_budget(conn)
    assert event["kind"] == "timed_out"
    payload = json.loads(event["payload"])
    assert payload["budget_used"] == 200
    assert payload["budget_max"] == 200
    # The runtime cap is a DIFFERENT cause; a budget exhaustion never has one.
    assert "limit_seconds" not in payload
    assert payload["retry_status"] == "ready"


def test_budget_exhaustion_keeps_the_cause_on_the_run_row(kanban_home):
    """``detail`` is the run row's metadata too, so the cause survives in run history."""
    with kbc.connect() as conn:
        tid, _event = _exhaust_budget(conn)
        metadata = _run_metadata(conn, tid)
    assert metadata["budget_used"] == 200
    assert metadata["budget_max"] == 200
    assert metadata["failures"] == 1


def test_board_notice_names_the_budget_it_recorded(kanban_home):
    """``max_runtime=0s`` was a cap that never existed; the notice names the real cause."""
    with kbc.connect() as conn:
        tid, event = _exhaust_budget(conn)
    text = _board_text(tid, json.loads(event["payload"]))
    assert "timed out (exhausted its turn budget (200/200)); will retry" in text
    assert "max_runtime=0s" not in text


def test_board_notice_still_names_a_recorded_runtime_cap():
    text = _board_text("t_abc123", dict(CAP_PAYLOAD))
    assert "timed out (max_runtime=900s); will retry" in text


def test_board_notice_says_so_when_no_cause_was_recorded():
    text = _board_text("t_abc123", dict(CAUSELESS_PAYLOAD))
    assert "timed out (cause not recorded); will retry" in text
    assert "max_runtime=" not in text


def test_telegram_notice_names_the_budget_it_recorded(kanban_home):
    with kbc.connect() as conn:
        _tid, event = _exhaust_budget(conn)
    msg = _telegram_text(json.loads(event["payload"]))
    assert "exhausted its turn budget (200/200) and was stopped" in msg
    assert "time limit" not in msg


def test_telegram_notice_still_names_the_minute_limit():
    msg = _telegram_text(dict(CAP_PAYLOAD))
    assert "ran past its 15-minute limit and was stopped" in msg


def test_telegram_notice_says_so_when_no_cause_was_recorded():
    msg = _telegram_text(dict(CAUSELESS_PAYLOAD))
    assert "was stopped (cause not recorded)" in msg
    assert "limit" not in msg


# ---------------------------------------------------------------------------
# Review rework: ONE cause decision, read by both surfaces (PR #124629)
#
# The review found the two notifiers documenting themselves as mirrors of one cause
# formatter while their guards differed, so a single payload rendered a cap on one
# surface and "cause not recorded" on the other. The guards are no longer kept in step
# by hand: ``gateway.kanban_watchers_common.timed_out_cause`` decides, and both
# surfaces only render that decision. ``test_both_surfaces_agree_on_every_cause_shape``
# is the gate over the whole payload space; the named tests below are the four shapes
# the review (and this rework) found broken, kept because they read as the defect.
# ---------------------------------------------------------------------------

CAUSE_MATRIX = [
    ("recorded cap", {"limit_seconds": 900}),
    ("cap of one whole minute", {"limit_seconds": 60}),
    ("sub-minute cap", {"limit_seconds": 10}),
    ("cap from `--max-runtime -5`", {"limit_seconds": -5}),
    ("cap from `--max-runtime -5m`", {"limit_seconds": -300}),
    ("negative cap beside a budget", {"limit_seconds": -5, "budget_used": 5, "budget_max": 10}),
    ("cap of exactly zero", {"limit_seconds": 0}),
    ("recorded budget", {"budget_used": 200, "budget_max": 200}),
    ("budget with zero used", {"budget_used": 0, "budget_max": 200}),
    ("budget of zero against zero", {"budget_used": 0, "budget_max": 0}),
    ("non-numeric cap", {"limit_seconds": "not-a-number"}),
    ("non-numeric budget", {"budget_used": "x", "budget_max": 200}),
    ("no cause fields", {"failures": 1}),
    ("empty payload", {}),
]


def _cause(payload: dict) -> tuple:
    from gateway.kanban_watchers_common import timed_out_cause

    return timed_out_cause(payload)


@pytest.mark.parametrize("payload", [m[1] for m in CAUSE_MATRIX], ids=[m[0] for m in CAUSE_MATRIX])
def test_both_surfaces_agree_on_every_cause_shape(payload):
    """Whatever shape the payload has, each surface renders the ONE recorded cause.

    Asserts the renderings, not the guards: a surface may phrase the cause its own way,
    but it must name the recorded cap/budget exactly, and must name NO number the payload
    does not carry. A surface that invents a cause (or a cap) fails here.
    """
    from gateway.kanban_watchers_common import TIMED_OUT_BUDGET, TIMED_OUT_CAP

    kind, first, second = _cause(payload)
    board = _board_text("t_abc123", dict(payload))
    telegram = _telegram_text(dict(payload))

    assert "timed out" in board
    if kind == TIMED_OUT_CAP:
        assert f"max_runtime={first}s" in board
        minute = re.search(r"ran past its (\d+)-minute limit", telegram)
        if minute:  # a minute cap may be named only when it IS the recorded cap
            assert first % 60 == 0 and int(minute.group(1)) == first // 60
        else:
            assert "its time limit" in telegram
        assert "exhausted its turn budget" not in telegram
    elif kind == TIMED_OUT_BUDGET:
        named = f"exhausted its turn budget ({first}/{second})"
        assert named in board
        assert named in telegram
    else:
        assert "cause not recorded" in board
        assert "cause not recorded" in telegram
        assert "exhausted its turn budget" not in telegram
        # No cap is named for a payload that records none - that was the whole defect.
        assert "limit" not in telegram


def test_a_negative_cap_is_never_read_as_a_limit():
    """Finding 1: ``--max-runtime -5`` was "its 1-minute limit" on Telegram alone."""
    for payload in ({"limit_seconds": -5}, {"limit_seconds": -300}):
        assert _cause(payload)[0] == "unrecorded"
        assert "max_runtime=" not in _board_text("t_abc123", payload)
        assert "limit" not in _telegram_text(payload)


def test_a_zero_iteration_budget_is_the_cause_both_surfaces_name():
    """Finding 2: ``{budget_used: 0}`` was a budget on the board and nothing on Telegram."""
    for used, cap in ((0, 200), (0, 0)):
        payload = {"budget_used": used, "budget_max": cap}
        named = f"exhausted its turn budget ({used}/{cap})"
        assert named in _board_text("t_abc123", payload)
        assert named in _telegram_text(payload)


def test_a_non_numeric_cause_field_does_not_break_either_notice():
    """Finding 3, found in this rework: ``int(_payload(...))`` raised OUT of the notifier.

    A non-numeric ``limit_seconds`` is the shape the board's own suite already pinned
    (``test_kanban_notify_poller``); the Telegram path had no guard, so one poisoned field
    aborted the delivery of the whole notice.
    """
    for payload in ({"limit_seconds": "not-a-number"}, {"budget_used": "x", "budget_max": 200}):
        assert "timed out" in _board_text("t_abc123", payload)
        assert "cause not recorded" in _telegram_text(payload)


def test_a_sub_minute_cap_never_names_a_minute_it_did_not_record():
    """The review's minor: a 10 s cap rendered as "its 1-minute limit"."""
    assert "max_runtime=10s" in _board_text("t_abc123", {"limit_seconds": 10})
    msg = _telegram_text({"limit_seconds": 10})
    assert "minute limit" not in msg
    assert "its time limit" in msg


def test_the_board_formatter_survives_the_server_rebind():
    """The production wiring, pinned: ``method_ctx.bind_module`` re-creates every function in
    ``session_notifications`` against server.py's globals, so a name the body resolves from
    module globals must survive a namespace that holds none of this module's imports.

    The module-level ``contextlib`` this function used to lean on was such a name. Anything
    imported at module level would be dropped here — ``bind_module`` skips a plain import —
    which is why the shared decision is imported inside the body of the renderer.
    """
    from tui_gateway import session_notifications as sn
    from tui_gateway.method_ctx import bind_module

    server = SimpleNamespace()
    # A copy, so this test cannot re-point the module's own dispatch tables.
    bind_module(dict(vars(sn)), server, skip=("_",))

    assert server._kb_timed_out_cause({"limit_seconds": 900}) == "max_runtime=900s"
    assert server._kb_timed_out_cause({"budget_used": 0, "budget_max": 200}) == (
        "exhausted its turn budget (0/200)"
    )
    assert server._kb_timed_out_cause({"limit_seconds": -5}) == "cause not recorded"


@pytest.mark.parametrize(
    "val,seconds",
    [("30s", 30), ("5m", 300), ("2h", 7200), ("1d", 86400), ("90", 90), (30, 30), (None, None), ("", None)],
)
def test_parse_duration_still_accepts_every_real_cap(val, seconds):
    from hermes_cli.kanban import _parse_duration

    assert _parse_duration(val) == seconds


@pytest.mark.parametrize("val", ["0", "-5", "-5m", "0s", "0m", "0h"])
def test_parse_duration_refuses_a_non_positive_cap(val):
    """The write-time half of finding 1: ``enforce_max_runtime`` measures ``elapsed < limit``,
    so a zero or negative cap SIGTERMs the worker on its first tick and records a
    ``limit_seconds`` no notice can name honestly. It never reaches the store."""
    from hermes_cli.kanban import _parse_duration

    with pytest.raises(ValueError, match="at least 1 second"):
        _parse_duration(val)


def test_parse_duration_still_rejects_a_malformed_cap():
    from hermes_cli.kanban import _parse_duration

    with pytest.raises(ValueError, match="malformed duration"):
        _parse_duration("soon")
