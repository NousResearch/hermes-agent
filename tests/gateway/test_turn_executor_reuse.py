"""The turn executor reuses parked workers, and a reused thread carries nothing from its last turn.

``submit`` runs on the gateway loop. Starting a thread there waits until the new thread gets the
GIL, which froze the loop on every submit while many turns ran (#142061); a parked worker takes
the item with a queue put instead. Reuse is only safe if turn 1's thread-keyed state (interrupt
bit, per-turn thread-locals, the thread-scoped sudo cache, context variables) is gone in turn 2.
"""

from __future__ import annotations

import contextvars
import threading
import time

from gateway.turn_executor import _UnboundedThreadExecutor


def _wait_parked(ex, n=1):
    deadline = time.monotonic() + 5
    while len(getattr(ex, "_idle", ())) < n:  # a thread-per-item executor never parks one
        assert time.monotonic() < deadline, "the worker never parked"
        time.sleep(0.01)


def test_a_reused_worker_sees_none_of_the_previous_turns_thread_state():
    from hermes_constants import get_hermes_home, set_hermes_home_override
    from tools import interrupt, terminal_tool
    from tools import terminal_tool_sudo as sudo

    fresh_home = contextvars.Context().run(get_hermes_home)
    ex = _UnboundedThreadExecutor("reuse-probe")
    try:
        def turn_one():
            interrupt.set_interrupt(True, reason="user pressed stop")
            terminal_tool._callback_tls.approval = "session-a approval callback"
            set_hermes_home_override("/profiles/session-a")  # left set: the turn never resets it
            sudo._sudo_password_cache[f"thread:{threading.get_ident()}"] = "session-a sudo password"
            return threading.get_ident()

        first = ex.submit(turn_one).result(5)
        _wait_parked(ex)

        def turn_two():
            return dict(
                tid=threading.get_ident(),
                interrupted=interrupt.is_interrupted(),
                reason=interrupt.get_interrupt_reason(),
                approval=getattr(terminal_tool._callback_tls, "approval", None),
                home=get_hermes_home(),
                sudo=sudo._sudo_password_cache.get(f"thread:{threading.get_ident()}"),
            )

        seen = ex.submit(turn_two).result(5)
        assert seen.pop("tid") == first, "the parked worker was not reused"
        assert seen == dict(interrupted=False, reason=None, approval=None, home=fresh_home, sudo=None)
    finally:
        ex.shutdown(wait=True)


def test_submit_to_a_parked_worker_starts_no_thread_and_shutdown_releases_it(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._executor_lock = threading.Lock()
    runner._executor = None
    runner._housekeeping_executor = None
    runner._executor_closing = False
    ex = runner._get_executor()
    assert isinstance(ex, _UnboundedThreadExecutor)
    ex.submit(lambda: None).result(5)
    _wait_parked(ex)
    parked = ex._idle[0].thread
    assert parked is not None

    def no_thread_start(self):
        raise AssertionError("submit started a thread on the event loop while a worker was parked")

    with monkeypatch.context() as m:
        m.setattr(threading.Thread, "start", no_thread_start)
        assert ex.submit(lambda: "ran").result(5) == "ran"
    _wait_parked(ex)
    # A parked worker holds no work: shutdown counts nothing live and the worker exits.
    assert runner._shutdown_executor(drain_timeout=5) == 0
    parked.join(5)
    assert not parked.is_alive()


def test_a_reused_worker_does_not_log_as_the_previous_turns_session():
    import logging

    import hermes_logging

    ex = _UnboundedThreadExecutor("reuse-probe")
    try:
        def turn_one():
            hermes_logging.set_session_context("session-a")  # build_turn_context does this, never clears it
            return threading.get_ident()

        first = ex.submit(turn_one).result(5)
        _wait_parked(ex)

        def next_item():  # another session's pre-turn work, an idle gate, another profile's turn
            record = logging.getLogRecordFactory()("gateway.run", logging.INFO, __file__, 1, "msg", (), None)
            return threading.get_ident(), record.session_tag

        tid, tag = ex.submit(next_item).result(5)
        assert tid == first, "the parked worker was not reused"
        assert tag == "", f"the reused thread logs as {tag!r}"
    finally:
        ex.shutdown(wait=True)


def _hold_mid_import(monkeypatch, module_name):
    """Re-import the real ``module_name`` on a helper thread and park it before its body runs.

    Python lists a module in ``sys.modules`` when its import starts, so until ``release`` is set
    the name resolves to a module with none of its attributes: what a reused worker's reset saw
    while another turn thread was importing it for the first time (gateway-06efb1e8d9).
    """
    import importlib.util
    import sys

    package, _, child = module_name.rpartition(".")
    monkeypatch.delitem(sys.modules, module_name)  # restored (the original object) after the test
    monkeypatch.delattr(sys.modules[package], child)
    spec = importlib.util.find_spec(module_name)
    real_exec = spec.loader.exec_module
    started, release = threading.Event(), threading.Event()

    class _GatedLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            started.set()
            assert release.wait(10), "the test never released the import"
            real_exec(module)

    class _Finder:
        def find_spec(self, name, path=None, target=None):
            if name != module_name:
                return None
            gated = importlib.util.spec_from_file_location(name, spec.origin, loader=_GatedLoader())
            return gated

    finder = _Finder()
    sys.meta_path.insert(0, finder)
    importer = threading.Thread(target=importlib.import_module, args=(module_name,), daemon=True)
    importer.start()
    assert started.wait(5), "the import never started"
    sys.meta_path.remove(finder)

    def finish():
        release.set()
        importer.join(10)
        assert not importer.is_alive()
        return sys.modules[module_name]

    return finish


def test_reset_skips_a_sudo_module_another_thread_is_still_importing(monkeypatch):
    import sys

    from gateway.turn_thread_reset import reset_reused_thread_state

    finish = _hold_mid_import(monkeypatch, "tools.terminal_tool_sudo")
    try:
        assert not hasattr(sys.modules["tools.terminal_tool_sudo"], "_sudo_password_cache_lock")
        reset_reused_thread_state()  # was: AttributeError: partially initialized module ...
    finally:
        sudo = finish()
    key = f"thread:{threading.get_ident()}"
    sudo._sudo_password_cache[key] = "cached by this thread's turn"
    reset_reused_thread_state()
    assert key not in sudo._sudo_password_cache, "a fully loaded module's thread entry must still clear"


def test_reset_skips_an_interrupt_module_another_thread_is_still_importing(monkeypatch):
    import sys

    from gateway.turn_thread_reset import reset_reused_thread_state

    finish = _hold_mid_import(monkeypatch, "tools.interrupt")
    try:
        assert not hasattr(sys.modules["tools.interrupt"], "set_interrupt")
        reset_reused_thread_state()  # was: AttributeError: partially initialized module ...
    finally:
        interrupt = finish()
    interrupt.set_interrupt(True, reason="turn one's stop")
    reset_reused_thread_state()
    assert not interrupt.is_interrupted(), "a fully loaded module's interrupt bit must still clear"
    assert interrupt.get_interrupt_reason() is None
