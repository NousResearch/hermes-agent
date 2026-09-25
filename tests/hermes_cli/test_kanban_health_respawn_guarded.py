"""Health telemetry must not call a deliberate guard deferral "stuck".

During the t_6a6ac2d3 incident the dispatcher logged

    kanban dispatcher stuck: ready queue non-empty for N consecutive
    ticks but 0 workers spawned.

while it was in fact deferring the card on purpose via the respawn guard
(``respawn_guarded {"reason":"active_pr"}``, ~170 times). ``skipped_nonspawnable``
was already excluded from the bad-tick count on exactly those grounds;
``respawn_guarded`` must be too.

TWO implementations carry the counter and both are covered here:

* ``gateway/kanban_watchers.py`` — ``bad_ticks``, the live path;
* ``hermes_cli/kanban_ops.py`` — ``health_state``, the ``--force`` daemon.

Both tests drive the REAL loop body against a REAL temp board and assert
on the warning the operator would actually see. The negative control in
each (genuinely spawnable work + zero spawns) forbids the lazy fix of
blanket-disabling the telemetry.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
from unittest.mock import patch

import pytest


# The gateway and CLI both warn after this many consecutive bad ticks.
HEALTH_WINDOW = 6


@pytest.fixture()
def kb(monkeypatch):
    """Fresh HERMES_HOME + kanban DB with an 'a' profile that can spawn."""
    test_home = tempfile.mkdtemp(prefix="kanban_health_test_")
    for prof in ("a", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_LOGS_ROOT",
                "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK"):
        monkeypatch.delenv(var, raising=False)
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from types import SimpleNamespace
    from hermes_cli import kanban_db, kanban_db_connect, kanban_db_dispatch
    kanban_db.init_db()
    # profile_exists resolves from HOME, not HERMES_HOME: treat 'a' as a real profile.
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    # One handle over the decomposed modules: ``kb.create_task`` lives in
    # kanban_db, connect/dispatch helpers in their split modules.
    yield SimpleNamespace(
        db=kanban_db, dispatch=kanban_db_dispatch,
        connect_closing=kanban_db_connect.connect_closing,
        create_task=kanban_db.create_task,
        DispatchResult=kanban_db_dispatch.DispatchResult,
        guard_deferred_ids=kanban_db_dispatch.guard_deferred_ids,
        has_spawnable_ready=kanban_db_dispatch.has_spawnable_ready,
    )


def _ready_card(kb, title="waiting"):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title=title, assignee="a")
        conn.commit()
    return tid


def _result(kb, *, guarded=()):
    res = kb.DispatchResult()
    res.respawn_guarded = list(guarded)
    return res


# ---------------------------------------------------------------------------
# The gateway implementation (the live path)
# ---------------------------------------------------------------------------


def _run_gateway_ticks(kb, monkeypatch, caplog, tick_result_fn, ticks):
    """Drive ``_kanban_dispatcher_watcher`` for ``ticks`` iterations.

    Only the spawn side is stubbed (``dispatch_once``). Board listing, the
    DB, ``guard_deferred_ids`` and ``has_spawnable_ready`` are all real, so
    the telemetry decision under test runs for real.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {}
    runner._kanban_sub_fail_counts = {}

    monkeypatch.setattr(kb.dispatch, "dispatch_once", tick_result_fn)

    sleeps = {"n": 0}

    async def fake_sleep(delay):
        sleeps["n"] += 1
        # sleep(5) boot delay, then one 1s slice per tick.
        if sleeps["n"] > ticks:
            runner._running = False

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    cfg = {"kanban": {"dispatch_in_gateway": True,
                      "dispatch_interval_seconds": 1,
                      "auto_decompose": False}}

    with caplog.at_level(logging.WARNING, logger="gateway.kanban_watchers"):
        with patch("hermes_cli.config.load_config", return_value=cfg):
            with patch("asyncio.sleep", side_effect=fake_sleep):
                with patch("asyncio.to_thread", side_effect=fake_to_thread):
                    asyncio.run(runner._kanban_dispatcher_watcher())
    return [r.getMessage() for r in caplog.records]


def test_gateway_guard_deferred_tick_is_not_stuck(kb, monkeypatch, caplog):
    """Only-guarded ready work + 0 spawns must NOT accumulate bad ticks."""
    tid = _ready_card(kb)

    def fake_dispatch(conn, **kwargs):
        return _result(kb, guarded=[(tid, "active_pr")])

    msgs = _run_gateway_ticks(
        kb, monkeypatch, caplog, fake_dispatch, ticks=HEALTH_WINDOW + 3,
    )
    assert not [m for m in msgs if "dispatcher stuck" in m], (
        f"a guard deferral is a healthy dispatcher deciding, not a stuck "
        f"one; got {msgs}"
    )


def test_gateway_spawnable_work_with_zero_spawns_still_warns(
    kb, monkeypatch, caplog,
):
    """Negative control: the telemetry must not be blanket-disabled."""
    _ready_card(kb)

    def fake_dispatch(conn, **kwargs):
        return _result(kb)  # nothing spawned, nothing guarded

    msgs = _run_gateway_ticks(
        kb, monkeypatch, caplog, fake_dispatch, ticks=HEALTH_WINDOW + 3,
    )
    assert [m for m in msgs if "dispatcher stuck" in m], (
        f"genuinely spawnable ready work with 0 spawns must still warn; "
        f"got {msgs}"
    )


def test_gateway_other_spawnable_card_beside_a_guarded_one_still_warns(
    kb, monkeypatch, caplog,
):
    """Exclusion is per-card, not per-tick: one guarded card must not
    mask a second, genuinely spawnable one."""
    guarded = _ready_card(kb, title="guarded")
    _ready_card(kb, title="genuinely stuck")

    def fake_dispatch(conn, **kwargs):
        return _result(kb, guarded=[(guarded, "active_pr")])

    msgs = _run_gateway_ticks(
        kb, monkeypatch, caplog, fake_dispatch, ticks=HEALTH_WINDOW + 3,
    )
    assert [m for m in msgs if "dispatcher stuck" in m], (
        f"a guarded card must not suppress the warning for OTHER spawnable "
        f"work; got {msgs}"
    )


# ---------------------------------------------------------------------------
# The CLI `--force` daemon implementation
# ---------------------------------------------------------------------------


def _run_cli_ticks(kb, monkeypatch, capsys, results, ticks):
    """Drive ``_cmd_daemon``'s real ``_on_tick`` closure ``ticks`` times."""
    import hermes_cli.kanban as kanban_cli

    captured = {}

    def fake_run_daemon(**kwargs):
        captured["on_tick"] = kwargs["on_tick"]

    monkeypatch.setattr(kb.dispatch, "run_daemon", fake_run_daemon)

    args = argparse.Namespace(
        force=True, interval=1, max=None, pidfile=None, verbose=False,
        failure_limit=2,
    )
    assert kanban_cli._cmd_daemon(args) == 0
    on_tick = captured["on_tick"]
    for _ in range(ticks):
        on_tick(results())
    return capsys.readouterr().err


def test_cli_daemon_guard_deferred_tick_is_not_stuck(kb, monkeypatch, capsys):
    """Only-guarded ready work + 0 spawns must NOT accumulate bad ticks."""
    tid = _ready_card(kb)
    err = _run_cli_ticks(
        kb, monkeypatch, capsys,
        lambda: _result(kb, guarded=[(tid, "active_pr")]),
        ticks=HEALTH_WINDOW + 3,
    )
    assert "dispatcher stuck" not in err, err


def test_cli_daemon_spawnable_work_with_zero_spawns_still_warns(
    kb, monkeypatch, capsys,
):
    """Negative control for the CLI counter."""
    _ready_card(kb)
    err = _run_cli_ticks(
        kb, monkeypatch, capsys, lambda: _result(kb),
        ticks=HEALTH_WINDOW + 3,
    )
    assert "dispatcher stuck" in err, err


def test_cli_daemon_other_spawnable_card_beside_a_guarded_one_still_warns(
    kb, monkeypatch, capsys,
):
    """Per-card exclusion for the CLI counter too."""
    guarded = _ready_card(kb, title="guarded")
    _ready_card(kb, title="genuinely stuck")
    err = _run_cli_ticks(
        kb, monkeypatch, capsys,
        lambda: _result(kb, guarded=[(guarded, "active_pr")]),
        ticks=HEALTH_WINDOW + 3,
    )
    assert "dispatcher stuck" in err, err


# ---------------------------------------------------------------------------
# The shared derivation both implementations use
# ---------------------------------------------------------------------------


def test_guard_deferred_ids_accepts_every_tick_shape(kb):
    """One helper, so the two implementations cannot drift."""
    one = _result(kb, guarded=[("t_a", "active_pr")])
    two = _result(kb, guarded=[("t_b", "recent_success")])
    assert kb.guard_deferred_ids(one) == {"t_a"}
    assert kb.guard_deferred_ids([one, two]) == {"t_a", "t_b"}
    # The gateway's multi-board shape: (slug, result) pairs, Nones included.
    assert kb.guard_deferred_ids(
        [("main", one), ("other", None), ("third", two)]
    ) == {"t_a", "t_b"}
    assert kb.guard_deferred_ids(None) == set()
    assert kb.guard_deferred_ids([]) == set()


def test_has_spawnable_ready_honours_exclude_ids(kb, all_assignees_spawnable):
    """The probe narrows per card and never turns itself off wholesale."""
    tid = _ready_card(kb)
    other = _ready_card(kb, title="second")
    with kb.connect_closing() as conn:
        assert kb.has_spawnable_ready(conn) is True
        assert kb.has_spawnable_ready(conn, {tid}) is True  # `other` remains
        assert kb.has_spawnable_ready(conn, {tid, other}) is False
