"""Instrumentation for the event-loop stalls that exit the gateway with code 75.

`shutdown_watchdog` probes the loop every 30s and kills the process after 3
consecutive misses -- so ~90s of unresponsiveness -- then dumps every thread
stack. One sample, taken 90 seconds after the stall began, is not enough to
name the callback that caused it: six stalls across 2026-08-31..09-02 produced
six different main-thread stacks.

asyncio can name the culprit itself, in real time, via `loop.set_debug(True)`
and `loop.slow_callback_duration`. It is off by default because debug mode adds
coroutine-origin tracking overhead to every callback.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from hermes_cli.gateway import _apply_loop_stall_diagnostics


class _FakeLoop:
    def __init__(self):
        self.debug = False
        self.slow_callback_duration = 0.1

    def set_debug(self, value):
        self.debug = bool(value)


def test_unset_leaves_the_loop_alone():
    loop = _FakeLoop()
    assert _apply_loop_stall_diagnostics(loop, None) is False
    assert loop.debug is False
    assert loop.slow_callback_duration == 0.1


def test_zero_is_off():
    loop = _FakeLoop()
    assert _apply_loop_stall_diagnostics(loop, 0) is False
    assert loop.debug is False


def test_negative_is_off():
    loop = _FakeLoop()
    assert _apply_loop_stall_diagnostics(loop, -1) is False
    assert loop.debug is False


def test_a_positive_threshold_enables_debug_and_sets_the_duration():
    loop = _FakeLoop()
    assert _apply_loop_stall_diagnostics(loop, 5) is True
    assert loop.debug is True
    assert loop.slow_callback_duration == 5.0


def test_a_string_from_yaml_is_accepted():
    """config.yaml round-trips scalars as strings often enough to matter."""
    loop = _FakeLoop()
    assert _apply_loop_stall_diagnostics(loop, "2.5") is True
    assert loop.slow_callback_duration == 2.5


def test_a_bad_value_warns_and_stays_off(caplog):
    loop = _FakeLoop()
    with caplog.at_level(logging.WARNING):
        assert _apply_loop_stall_diagnostics(loop, "soon") is False
    assert loop.debug is False
    assert any("slow_callback_seconds" in r.message for r in caplog.records)


def test_a_loop_that_rejects_set_debug_does_not_raise():
    """Diagnostics must never be the reason a gateway fails to boot."""

    class _Hostile:
        slow_callback_duration = 0.1

        def set_debug(self, value):
            raise RuntimeError("nope")

    assert _apply_loop_stall_diagnostics(_Hostile(), 5) is False


def test_the_e2ee_dep_check_is_not_awaited_on_the_loop():
    """`import mautrix.crypto` measures 2.21s (`python -X importtime`), and
    `connect()` ran it inline on the event loop on every Matrix connect. Two of
    the six observed stalls were sitting in exactly that import. It has to stay
    on a worker thread.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "plugins/platforms/matrix/adapter.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Name) and func.id == "_check_e2ee_deps":
                # Allowed only as the argument to to_thread, i.e. never called
                # here -- passed as a reference. A direct call has no parent
                # to_thread, so flag any Call node at all.
                offenders.append((node.name, getattr(call, "lineno", "?")))
    assert not offenders, (
        "_check_e2ee_deps() is called directly inside an async function "
        f"(blocking the event loop) at: {offenders}"
    )
