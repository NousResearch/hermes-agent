"""A malformed ``kanban.max_spawn`` must fail CLOSED, not kill the dispatcher.

BUI-938 makes the watcher re-raise ``TypeError`` out of a tick, on purpose: a
bad call signature or type misuse inside a tick is a *programming* bug and must
surface rather than be masked as an "unexpected watcher error". That is the
right call — and it raises the stakes on every value the gateway feeds into a
tick.

``kanban.max_spawn`` was the one live-concurrency ceiling read straight out of
config with no conversion, while its siblings (``max_in_progress``,
``max_in_progress_per_profile``, ``failure_limit``) were all coerced. A quoted
``max_spawn: "5"`` — routine YAML/env plumbing — therefore reached
``dispatch_once`` as a ``str`` and blew up on the first comparison against a
running-task count::

    TypeError: '>=' not supported between instances of 'int' and 'str'

With BUI-938 in place that ``TypeError`` is classified as a programming error,
so it tears down the heartbeat, releases the singleton dispatcher lock, and
re-raises: an operator's config typo takes the whole fleet dispatcher offline.

These tests pin the fixed contract:

  * ``"5"`` is normalised to ``5`` and never reaches arithmetic as a string;
  * a value that cannot yield a usable ceiling (non-numeric string, negative,
    zero, bool, container) fails CLOSED — the dispatcher stays up and spawns
    *nothing* rather than falling back to unlimited;
  * an unset value still means "no ceiling";
  * quoting is not load-bearing — ``5.9`` and ``"5.9"`` are the same operator
    intent and must resolve the same way (see below);
  * and, critically, the BUI-938 surfacing is NOT weakened — a genuine
    ``TypeError`` out of a tick still propagates out of the watcher.

The quoting-parity follow-up
----------------------------

The first cut of the boundary converted with ``int()``, which reads a float and
a string differently:

    int(5.9)    -> 5              # silently truncated; ceiling nobody wrote
    int("5.9")  -> ValueError     # fail closed

So ``max_spawn: 5.9`` granted a live ceiling of 5 while ``max_spawn: "5.9"``
withheld every spawn — the same intent resolving two ways depending on whether
the YAML loader handed us a ``float`` or a ``str``. That is the exact ambiguity
this boundary was added to eliminate, reintroduced one layer down.

``int()`` also could not express the non-finite values YAML can produce:
``max_spawn: .inf`` raised ``OverflowError``, which is **not** a ``ValueError``
and therefore escaped the ``except`` entirely — out of ``_coerce_live_
concurrency_cap``, out of ``_kanban_dispatcher_watcher`` (the call site sits
above the per-tick guard), killing the dispatcher over a config value. A
fail-closed boundary that fails *fatal* on one of its inputs is not fail-closed.

Both spellings now take one path — ``float()``, then a finiteness screen, then
a whole-number requirement — so the answer depends on the value, never on how
it was typed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import asyncio
import logging

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.kanban_watchers import _coerce_live_concurrency_cap

from hermes_cli import kanban_db as kb


# --------------------------------------------------------------------------
# helpers (mirrors tests/gateway/test_kanban_watcher_programming_errors.py)
# --------------------------------------------------------------------------


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = None
    runner._kanban_notifier_profile = "default"
    return runner


def _dispatcher_config(max_spawn, **extra):
    """Dispatch-owning config with the aux paths off, so the tick under test is
    purely ``dispatch_once``. ``max_spawn`` is injected verbatim — including the
    malformed shapes an operator can actually write."""
    kanban = {
        "dispatch_in_gateway": True,
        "dispatch_interval_seconds": 1,
        "auto_decompose": False,
        "linear_bridge": {"enabled": False},
        "max_spawn": max_spawn,
        **extra,
    }
    return {"kanban": kanban}


async def _passthru_to_thread(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _stop_after(runner, n, calls=None):
    """Fake ``asyncio.sleep`` that stops the loop after N sleeps, so a watcher
    that (correctly) survives the tick terminates instead of spinning.

    Pass ``calls`` to observe how far the loop actually got: a watcher that
    dies on tick 1 never reaches N.
    """
    calls = [] if calls is None else calls

    async def fake_sleep(delay):
        calls.append(delay)
        if len(calls) >= n:
            runner._running = False

    return fake_sleep


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """A real, isolated board on disk so the dispatcher tick runs the real
    ``dispatch_once`` — the arithmetic that a raw string breaks."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def spawnable(monkeypatch):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


def _seed_ready(n):
    with kb.connect() as conn:
        for i in range(n):
            kb.create_task(conn, title=f"t{i}", assignee=f"prof{i}")


def _run_dispatcher(runner, config, spawns, ticks=None):
    """Drive real dispatcher ticks against the real board.

    Only the process-level side effects are stubbed: the singleton lock (so the
    test never contends with a live gateway), the heartbeat/zombie reaper, and
    ``_default_spawn`` — which would otherwise launch actual worker processes.
    ``dispatch_once`` itself is REAL; that is the whole point.
    """
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return None

    with patch("hermes_cli.config.load_config", return_value=config):
        with patch(
            "gateway.kanban_watchers._acquire_singleton_lock",
            return_value=(None, "unavailable"),
        ):
            with patch.object(kb, "list_boards", return_value=[{"slug": "default"}]):
                with patch.object(kb, "_default_spawn", side_effect=fake_spawn):
                    with patch.object(kb, "write_dispatcher_heartbeat"):
                        with patch.object(kb, "reap_worker_zombies", return_value=[]):
                            with patch(
                                "asyncio.sleep",
                                side_effect=_stop_after(runner, 3, ticks),
                            ):
                                with patch("asyncio.to_thread", side_effect=_passthru_to_thread):
                                    asyncio.run(runner._kanban_dispatcher_watcher())


# --------------------------------------------------------------------------
# the conversion boundary itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (5, 5),
        ("5", 5),          # quoted numeric — YAML/env plumbing, operator meant 5
        (" 5 ", 5),        # whitespace from an env var
        (None, None),      # unset — no ceiling, the documented default
    ],
)
def test_usable_values_resolve(raw, expected):
    assert _coerce_live_concurrency_cap(raw, "kanban.max_spawn") == expected


@pytest.mark.parametrize(
    "raw",
    [
        "abc",       # non-numeric string
        "",          # empty string from an unset env var
        "5.5",       # not a whole number
        -1,          # negative
        0,           # zero written explicitly
        True,        # bool is an int subclass; `max_spawn: true` is a mistake
        [],          # container
        {},
        object(),
    ],
)
def test_unusable_values_fail_closed(raw):
    """A value that cannot yield a usable ceiling resolves to 0 — "spawn
    nothing" — never to ``None``, which would mean *unlimited*. A typo in a
    safety ceiling must not remove the ceiling."""
    assert _coerce_live_concurrency_cap(raw, "kanban.max_spawn") == 0


def test_fail_closed_logs_an_operator_facing_error(caplog):
    """The operator has to be able to find this: name the setting, echo the
    bad value, and say what the dispatcher is doing about it."""
    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        _coerce_live_concurrency_cap("abc", "kanban.max_spawn")
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "kanban.max_spawn" in msg
    assert "'abc'" in msg
    assert "refusing to spawn" in msg


def test_usable_value_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        _coerce_live_concurrency_cap("5", "kanban.max_spawn")
    assert caplog.records == []


# --------------------------------------------------------------------------
# end-to-end through the real dispatcher watcher + real dispatch_once
# --------------------------------------------------------------------------


def test_quoted_numeric_max_spawn_does_not_kill_the_dispatcher(
    kanban_home, spawnable
):
    """THE regression. ``max_spawn: "5"`` reached ``dispatch_once`` raw and the
    first ``running_count + spawned >= max_spawn`` comparison raised
    ``TypeError`` — which BUI-938 then surfaced as a programming error, taking
    the dispatcher down. It must instead be read as 5 and simply cap spawning.
    """
    _seed_ready(8)
    runner = _make_runner()
    spawns = []

    # Must NOT raise: a config typo is operational input, not a watcher bug.
    _run_dispatcher(runner, _dispatcher_config("5"), spawns)

    assert len(spawns) == 5, "quoted \"5\" must behave exactly like 5"


def test_non_numeric_max_spawn_fails_closed_without_killing_dispatcher(
    kanban_home, spawnable, caplog
):
    """``max_spawn: "unlimited"`` must not crash the dispatcher — and must not
    quietly become unlimited either. Nothing spawns until it is fixed."""
    _seed_ready(4)
    runner = _make_runner()
    spawns = []

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        _run_dispatcher(runner, _dispatcher_config("unlimited"), spawns)

    assert spawns == [], "a malformed ceiling must fail closed, not fail open"
    assert any(
        "kanban.max_spawn" in r.getMessage() for r in caplog.records
    ), "operator needs a named, actionable error"
    with kb.connect() as conn:
        running = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()[0]
    assert running == 0


def test_negative_max_spawn_fails_closed_without_killing_dispatcher(
    kanban_home, spawnable, caplog
):
    """``max_spawn: -1`` is nonsense as a ceiling. It happened to stall the
    board already (the spawn loop breaks immediately on a negative cap) — but
    silently, with no diagnostic at all, which is its own operator trap. Fail
    closed *and say so*, and never fall back to unlimited."""
    _seed_ready(4)
    runner = _make_runner()
    spawns = []

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        _run_dispatcher(runner, _dispatcher_config(-1), spawns)

    assert spawns == []
    assert any(
        "kanban.max_spawn" in r.getMessage() for r in caplog.records
    ), "a negative ceiling stalled the board with no operator-facing error"


def test_unset_max_spawn_still_means_no_ceiling(kanban_home, spawnable):
    """``None`` is the *unset* case, not a malformed one: it must keep meaning
    "no ceiling" so the fail-closed path can't strand a correct board."""
    _seed_ready(3)
    runner = _make_runner()
    spawns = []

    _run_dispatcher(runner, _dispatcher_config(None), spawns)

    assert len(spawns) == 3


def test_malformed_max_spawn_keeps_the_loop_ticking(kanban_home, spawnable):
    """Fail closed, not fail *stopped*. The programming-error path tears the
    dispatcher down after a single tick; a config typo must leave the loop
    running — reclaim, promotion and reconciliation still have to happen while
    the operator fixes the value."""
    _seed_ready(1)
    runner = _make_runner()
    spawns = []
    ticks = []

    _run_dispatcher(runner, _dispatcher_config("nope"), spawns, ticks=ticks)

    assert spawns == []
    assert len(ticks) >= 3, (
        "dispatcher aborted instead of continuing to tick with a bad ceiling"
    )


# --------------------------------------------------------------------------
# the BUI-938 contract this fix must NOT weaken
# --------------------------------------------------------------------------


def test_genuine_typeerror_in_tick_still_surfaces(tmp_path):
    """Narrowing the *config* boundary must not blunt BUI-938. A real
    ``TypeError`` raised inside a tick — a wrong call signature, a type misuse
    in watcher code — still tears down and propagates out of the watcher."""
    runner = _make_runner()

    class _DummyConn:
        def close(self):
            pass

    def boom(*args, **kwargs):
        raise TypeError("dispatch_once() got an unexpected keyword argument")

    with patch(
        "hermes_cli.config.load_config", return_value=_dispatcher_config(2)
    ):
        with patch(
            "gateway.kanban_watchers._acquire_singleton_lock",
            return_value=(None, "unavailable"),
        ):
            with patch.object(kb, "kanban_db_path", return_value=tmp_path / "k.db"):
                with patch.object(kb, "list_boards", return_value=[{"slug": "default"}]):
                    with patch.object(kb, "connect", return_value=_DummyConn()):
                        with patch.object(kb, "dispatch_once", side_effect=boom):
                            with patch.object(kb, "write_dispatcher_heartbeat"):
                                with patch.object(kb, "reap_worker_zombies", return_value=[]):
                                    with patch("asyncio.sleep", side_effect=_stop_after(runner, 3)):
                                        with patch(
                                            "asyncio.to_thread",
                                            side_effect=_passthru_to_thread,
                                        ):
                                            with pytest.raises(TypeError):
                                                asyncio.run(
                                                    runner._kanban_dispatcher_watcher()
                                                )


def test_dispatch_once_receives_an_int_or_none_never_a_string(tmp_path):
    """Belt-and-braces on the boundary contract: whatever the operator wrote,
    ``dispatch_once`` is only ever handed ``int`` or ``None``."""
    runner = _make_runner()
    seen = []

    class _DummyConn:
        def close(self):
            pass

    def record(conn, **kwargs):
        seen.append(kwargs.get("max_spawn"))
        runner._running = False
        return None

    with patch(
        "hermes_cli.config.load_config", return_value=_dispatcher_config("5")
    ):
        with patch(
            "gateway.kanban_watchers._acquire_singleton_lock",
            return_value=(None, "unavailable"),
        ):
            with patch.object(kb, "kanban_db_path", return_value=tmp_path / "k.db"):
                with patch.object(kb, "list_boards", return_value=[{"slug": "default"}]):
                    with patch.object(kb, "connect", return_value=_DummyConn()):
                        with patch.object(kb, "dispatch_once", side_effect=record):
                            with patch.object(kb, "has_spawnable_ready", return_value=False):
                                with patch.object(kb, "has_spawnable_review", return_value=False):
                                    with patch.object(kb, "write_dispatcher_heartbeat"):
                                        with patch.object(kb, "reap_worker_zombies", return_value=[]):
                                            with patch(
                                                "asyncio.sleep",
                                                side_effect=_stop_after(runner, 3),
                                            ):
                                                with patch(
                                                    "asyncio.to_thread",
                                                    side_effect=_passthru_to_thread,
                                                ):
                                                    asyncio.run(
                                                        runner._kanban_dispatcher_watcher()
                                                    )

    assert seen, "dispatch tick should have run"
    assert seen[0] == 5
    assert isinstance(seen[0], int) and not isinstance(seen[0], bool)


# --------------------------------------------------------------------------
# quoting parity: the value decides, not the spelling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, quoted",
    [
        (5.9, "5.9"),            # THE case: int() truncated one, rejected the other
        (0.5, "0.5"),            # rounds toward a below-1 cap under int()
        (5.0, "5.0"),            # whole float — usable in both spellings
        (float("inf"), "inf"),   # int() raised OverflowError on the float
        (float("nan"), "nan"),
    ],
)
def test_quoting_does_not_change_the_answer(number, quoted):
    """A YAML loader decides whether a number arrives as ``float`` or ``str``.
    That is not a decision about concurrency, so it must not move the ceiling."""
    assert _coerce_live_concurrency_cap(
        number, "kanban.max_spawn"
    ) == _coerce_live_concurrency_cap(quoted, "kanban.max_spawn")


@pytest.mark.parametrize(
    "raw",
    [
        5.9,             # THE regression: int() made this a live ceiling of 5
        "5.9",
        0.5,
        "0.5",
        2.000001,        # a hair off whole is still not whole
        float("inf"),    # int() raised OverflowError, escaping the boundary
        float("-inf"),
        float("nan"),
        "inf",
        "1e400",         # overflows to inf on parse
    ],
)
def test_fractional_and_non_finite_values_fail_closed(raw):
    """Neither a fraction nor a non-finite value can be a live-worker ceiling.
    Fail closed instead of truncating to whatever ``int()`` happened to yield —
    a truncated cap is a ceiling the operator never wrote, granted silently."""
    assert _coerce_live_concurrency_cap(raw, "kanban.max_spawn") == 0


def test_non_finite_value_does_not_raise_out_of_the_boundary():
    """``int(float("inf"))`` raises ``OverflowError`` — not a ``ValueError``, so
    it sailed past the except clause and out of the watcher. The boundary must
    absorb it like any other unusable value."""
    for raw in (float("inf"), float("-inf"), float("nan"), "inf"):
        assert _coerce_live_concurrency_cap(raw, "kanban.max_spawn") == 0


def test_whole_floats_are_still_usable():
    """Rejecting fractions must not turn into rejecting every float:
    ``max_spawn: 5.0`` is an unambiguous 5 and stays one."""
    assert _coerce_live_concurrency_cap(5.0, "kanban.max_spawn") == 5
    assert _coerce_live_concurrency_cap("5.0", "kanban.max_spawn") == 5
    assert _coerce_live_concurrency_cap(1.0, "kanban.max_spawn") == 1


def test_integer_values_are_not_round_tripped_through_float():
    """An ``int`` is already exact; parsing it as a float to reach the same
    answer would introduce precision loss above 2**53 while fixing a precision
    bug. Absurd as a ceiling, but it must not be silently *changed*."""
    big = 2 ** 53 + 1
    assert _coerce_live_concurrency_cap(big, "kanban.max_spawn") == big


def test_fractional_fail_closed_logs_an_operator_facing_error(caplog):
    """The operator wrote ``5.9`` and got no workers — that has to be findable
    in the log, named and echoed, not inferred from an idle board."""
    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        _coerce_live_concurrency_cap(5.9, "kanban.max_spawn")
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "kanban.max_spawn" in msg
    assert "5.9" in msg
    assert "refusing to spawn" in msg


def test_fractional_max_spawn_fails_closed_without_killing_dispatcher(
    kanban_home, spawnable, caplog
):
    """End-to-end on the real dispatcher: ``max_spawn: 5.9`` used to truncate to
    a working ceiling of 5 and quietly spawn five workers. Same value quoted
    spawned none. Now it fails closed either way, and the loop keeps ticking."""
    _seed_ready(8)
    runner = _make_runner()
    spawns = []
    ticks = []

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        _run_dispatcher(runner, _dispatcher_config(5.9), spawns, ticks=ticks)

    assert spawns == [], "an unquoted fractional ceiling must not truncate to 5"
    assert any("kanban.max_spawn" in r.getMessage() for r in caplog.records)
    assert len(ticks) >= 3, "dispatcher aborted instead of continuing to tick"


def test_infinite_max_spawn_does_not_kill_the_dispatcher(
    kanban_home, spawnable, caplog
):
    """``max_spawn: .inf`` is legal YAML. It reached ``int()``, raised
    ``OverflowError``, and — because the coercion happens above the per-tick
    guard — took the dispatcher watcher down before the first tick."""
    _seed_ready(4)
    runner = _make_runner()
    spawns = []
    ticks = []

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        # Must NOT raise OverflowError out of the watcher.
        _run_dispatcher(runner, _dispatcher_config(float("inf")), spawns, ticks=ticks)

    assert spawns == [], "an unusable ceiling must not fall back to unlimited"
    assert any("kanban.max_spawn" in r.getMessage() for r in caplog.records)
    assert len(ticks) >= 3


def test_unquoted_and_quoted_agree_through_the_real_dispatcher(
    kanban_home, spawnable
):
    """The parity contract where it actually matters: same config value, two
    spellings, same number of workers on the board."""
    _seed_ready(8)
    unquoted_spawns = []
    _run_dispatcher(_make_runner(), _dispatcher_config(5.9), unquoted_spawns)

    quoted_spawns = []
    _run_dispatcher(_make_runner(), _dispatcher_config("5.9"), quoted_spawns)

    assert unquoted_spawns == quoted_spawns == []
