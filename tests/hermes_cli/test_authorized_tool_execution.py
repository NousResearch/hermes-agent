"""A post-approval continuation is fail-closed, synchronous and single-use."""
from types import SimpleNamespace
import threading

import pytest

from hermes_cli.authorized_tool_execution import (
    authorized_tool_execution_active, hold_foreground_execution,
    run_authorized_tool_execution_middleware as run,
)
from tools.interrupt import set_interrupt


@pytest.fixture
def callbacks(monkeypatch, tmp_path):
    from hermes_cli import plugins
    values = []
    manager = SimpleNamespace(_middleware={"authorized_tool_execution": values},
                              _discovery_lock=threading.RLock(), scope_key=str(tmp_path))
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    return values


def test_default_no_callbacks_preserves_path(callbacks):
    assert run("terminal", {}, lambda: (42, authorized_tool_execution_active())) == (42, False)


@pytest.mark.parametrize("registered", [False, True])
def test_registration_worker_never_waits_for_its_own_discovery_lock(callbacks, monkeypatch, registered):
    monkeypatch.setattr("hermes_cli.plugins_loader.in_plugin_load_worker", lambda: True)
    if registered:
        callbacks.append(lambda next_call, **kwargs: next_call())
        with pytest.raises(RuntimeError, match="during plugin registration"):
            run("terminal", {}, lambda: pytest.fail("dispatched during discovery"))
    else:
        assert run("terminal", {}, lambda: "ordinary path") == "ordinary path"


def test_denial_never_falls_through(callbacks):
    def deny(**kwargs):
        raise RuntimeError("capacity unavailable")
    callbacks.append(deny)
    dispatched = []
    with pytest.raises(RuntimeError, match="capacity unavailable"):
        run("terminal", {}, lambda: dispatched.append(True))
    assert not dispatched and not authorized_tool_execution_active()


def test_args_cannot_rewrite_continuation_and_after_error_preserves_result(callbacks):
    calls, later = [], []
    args = {"command": "approved", "nested": [1]}
    def wrap(next_call, args, **kwargs):
        args["command"] = "unapproved"
        args["nested"].append(2)
        later.append(next_call)
        with pytest.raises(TypeError):
            next_call(args)
        with hold_foreground_execution():
            assert next_call() == "real-result"
        with pytest.raises(RuntimeError, match="single-use"):
            next_call()
        raise RuntimeError("cleanup detail must not replace the actual result")
    callbacks.append(wrap)
    def execute():
        calls.append(dict(args))
        assert authorized_tool_execution_active()
        return "real-result"
    assert run("terminal", args, execute) == "real-result"
    assert calls == [{"command": "approved", "nested": [1]}]
    with pytest.raises(RuntimeError, match="single-use"):
        later[0]()


def test_cancel_during_admission_cannot_start_tool(callbacks):
    calls, releases = [], []
    def wait(next_call, **kwargs):
        try:
            set_interrupt(True)
            return next_call()
        finally:
            releases.append(True)
    callbacks.append(wait)
    try:
        with pytest.raises(InterruptedError):
            run("terminal", {}, lambda: calls.append(True), clear_interrupt=True)
    finally:
        set_interrupt(False)
    assert not calls and releases == [True]


def test_continuation_cannot_escape_to_another_thread(callbacks):
    errors, calls = [], []
    def wrap(next_call, **kwargs):
        def worker():
            try:
                next_call()
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        return next_call()
    callbacks.append(wrap)
    run("terminal", {}, lambda: calls.append(True))
    assert len(errors) == 1 and calls == [True]


def test_registered_passthrough_does_not_change_foreground_lifetime(callbacks):
    callbacks.append(lambda next_call, **kwargs: next_call())
    assert run("terminal", {}, authorized_tool_execution_active) is False


def test_completed_result_is_authoritative(callbacks):
    def wrap(next_call, **kwargs):
        assert kwargs["middleware_schema_version"] == "hermes.middleware.v1"
        next_call()
        return "synthetic failure could invite duplicate execution"
    callbacks.append(wrap)
    assert run("terminal", {}, lambda: "completed") == "completed"


def test_opt_in_protection_restores_nested_scope(callbacks):
    with hold_foreground_execution():
        with hold_foreground_execution():
            assert authorized_tool_execution_active()
        assert authorized_tool_execution_active()
    assert not authorized_tool_execution_active()


def test_nested_callbacks_release_in_reverse_order_after_tool_error(callbacks):
    events = []
    def make(name):
        def wrap(next_call, **kwargs):
            events.append(name + ":acquire")
            try:
                return next_call()
            finally:
                events.append(name + ":release")
        return wrap
    callbacks.extend([make("a"), make("b")])
    def broken():
        raise ValueError("execution failed")
    with pytest.raises(ValueError, match="execution failed"):
        run("terminal", {}, broken)
    assert events == ["a:acquire", "b:acquire", "b:release", "a:release"]
