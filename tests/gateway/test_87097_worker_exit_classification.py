"""Test #87097: reaper log must classify each exited child by kind/code,
not label every reaped worker as a generic failure.

Issue: Worker watchers need to accept legacy callback shapes while
classifying exits by kind (clean exit, signal, non-zero, rate-limited,
unknown) instead of labeling every reaped child as a failure.

Drives the real ``_kanban_dispatcher_watcher`` loop for one tick after
stubbing its boot + dispatcher dependencies, then asserts the reaper
log line carries the per-pid classification.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from gateway import kanban_watchers as _kw


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _FakeRunner(_kw.GatewayKanbanWatchersMixin):
    """Stand-in for GatewayRunner carrying only the state the watcher
    loop reads between ticks."""

    def __init__(self) -> None:
        self._running = True
        self._kanban_dispatcher_lock_handle = None

    def _active_profile_name(self) -> str:
        return "default"

    _kanban_dispatcher_boot = lambda self: (lambda: {}, None, {"dispatch_interval_seconds": 0})


def _install_stubs(monkeypatch, fake_pids, classify_map):
    """Stub boot + every loop dependency. The reaper runs first in the
    tick (line 261), so as long as we flip ``_running`` after one tick,
    the rest of the body never executes."""
    from hermes_cli import kanban_db_dispatch as _kbd

    class _StubSettings:
        interval = 0.0

    class _StubDispatcher:
        def __init__(self, *_a, **_kw):
            pass

        def auto_decompose_tick(self, *_a, **_kw):
            return 0

        def tick_once(self):
            return []

        def ready_nonempty(self):
            return False

    async def _fake_sleep_initial(_seconds):
        return None

    async def _fake_to_thread(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            await result
        return result

    def _fake_resolve_settings(kanban_cfg, kb):
        return _StubSettings()

    def _fake_reap():
        return list(fake_pids)

    def _fake_classify(pid):
        return classify_map[pid]

    def _fake_dispatch_allowed():
        return True

    def _fake_auto_decompose(_cfg):
        return (False, 0)

    monkeypatch.setattr(_kw, "_to_thread_process_service", _fake_to_thread)
    monkeypatch.setattr(
        _kw, "_resolve_dispatcher_settings", _fake_resolve_settings
    )
    monkeypatch.setattr(_kw, "_KanbanDispatcher", _StubDispatcher)
    monkeypatch.setattr(_kw, "_kanban_dispatch_allowed", _fake_dispatch_allowed)
    monkeypatch.setattr(
        _kw, "_resolve_auto_decompose_settings", _fake_auto_decompose
    )
    monkeypatch.setattr(_kbd, "reap_worker_zombies", _fake_reap)
    monkeypatch.setattr(_kbd, "_classify_worker_exit", _fake_classify)

    async def _fake_sleep(self, _interval):
        # Flip _running so the loop exits after one iteration.
        self._running = False

    monkeypatch.setattr(
        _kw.GatewayKanbanWatchersMixin, "_sleep_between_ticks", _fake_sleep
    )
    # Also stub the module-level asyncio.sleep that the watcher uses for the
    # 5-second initial delay so tests run instantly.
    import asyncio as _asyncio
    monkeypatch.setattr(_kw.asyncio, "sleep", _fake_sleep_initial)


def _capture_logger():
    handler = _RecordingHandler()
    logger = _kw.logger
    logger.addHandler(handler)
    prior_level = logger.level
    logger.setLevel(logging.DEBUG)
    return handler, logger, prior_level


def _reaper_line(handler):
    msgs = [
        r.getMessage()
        for r in handler.records
        if "zombie" in r.getMessage() or "worker child" in r.getMessage()
    ]
    assert msgs, (
        "no reaper log record emitted by the dispatcher watcher; "
        "loop ran without ever logging a reaped batch"
    )
    return msgs[-1]


def test_reaper_log_classifies_clean_exit(monkeypatch):
    _install_stubs(
        monkeypatch,
        fake_pids=[101],
        classify_map={101: ("clean_exit", 0)},
    )
    handler, logger, prior_level = _capture_logger()
    try:
        runner = _FakeRunner()
        _run_one_tick(runner)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
    line = _reaper_line(handler)
    assert "101" in line, f"reaper log must name pid 101; got: {line}"
    assert "clean_exit" in line, (
        f"reaper log must classify rc=0 as 'clean_exit'; got: {line}"
    )


def _run_one_tick(runner):
    async def _go():
        await _kw.GatewayKanbanWatchersMixin._kanban_dispatcher_watcher(runner)

    return asyncio.run(_go())


def test_reaper_log_classifies_signal(monkeypatch):
    _install_stubs(
        monkeypatch,
        fake_pids=[202],
        classify_map={202: ("signaled", 9)},
    )
    handler, logger, prior_level = _capture_logger()
    try:
        runner = _FakeRunner()
        _run_one_tick(runner)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
    line = _reaper_line(handler)
    assert "202" in line, f"reaper log must name pid 202; got: {line}"
    assert "signaled" in line, (
        f"reaper log must classify signal-killed as 'signaled'; got: {line}"
    )
    assert "9" in line, f"reaper log must carry signal number 9; got: {line}"


def test_reaper_log_classifies_nonzero_exit(monkeypatch):
    _install_stubs(
        monkeypatch,
        fake_pids=[303],
        classify_map={303: ("nonzero_exit", 17)},
    )
    handler, logger, prior_level = _capture_logger()
    try:
        runner = _FakeRunner()
        _run_one_tick(runner)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
    line = _reaper_line(handler)
    assert "303" in line
    assert "nonzero_exit" in line, (
        f"reaper log must classify non-zero as 'nonzero_exit'; got: {line}"
    )
    assert "17" in line


def test_reaper_log_distinguishes_rate_limited(monkeypatch):
    _install_stubs(
        monkeypatch,
        fake_pids=[404],
        classify_map={404: ("rate_limited", 75)},
    )
    handler, logger, prior_level = _capture_logger()
    try:
        runner = _FakeRunner()
        _run_one_tick(runner)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
    line = _reaper_line(handler)
    assert "404" in line
    assert "rate_limited" in line, (
        f"reaper log must classify quota-wall as 'rate_limited'; got: {line}"
    )


def test_reaper_log_does_not_label_every_exit_as_failure(monkeypatch):
    """A clean exit and a signal must not be uniformly labelled as
    failures; the log must distinguish them and not call a clean rc=0 a
    'failure' or 'crashed' in the reaper line."""
    _install_stubs(
        monkeypatch,
        fake_pids=[11, 22],
        classify_map={
            11: ("clean_exit", 0),
            22: ("signaled", 15),
        },
    )
    handler, logger, prior_level = _capture_logger()
    try:
        runner = _FakeRunner()
        _run_one_tick(runner)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
    line = _reaper_line(handler)
    # Both pids must appear with their respective classifications.
    assert "11" in line and "clean_exit" in line
    assert "22" in line and "signaled" in line
    # The reaper line must NOT flatten them into a generic framing.
    assert "zombie worker(s)" not in line, (
        f"reaper log still uses the old uniform 'zombie worker(s)' framing; "
        f"it must classify each exit kind: {line}"
    )