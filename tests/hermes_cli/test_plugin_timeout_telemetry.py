"""Contract tests for bounded, attributable plugin callback timeout telemetry."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def _events(manager: PluginManager) -> list[dict]:
    return [dict(event) for event in manager.get_hook_callback_telemetry()]


def test_timed_out_then_abandoned_and_suppression_outcomes_are_distinct(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.02)
    hold = threading.Event()

    def blocker(**_kwargs):
        hold.wait(timeout=2.0)

    manager = PluginManager()
    manager._hooks["post_tool_call"] = [blocker]

    assert manager.invoke_hook("post_tool_call", tool_call_id="call-timeout") == []
    assert manager.invoke_hook("post_tool_call", tool_call_id="call-abandoned") == []
    assert [event["outcome"] for event in _events(manager)] == [
        "timed_out",
        "callback_abandoned",
    ]

    hold.set()
    _wait_for(lambda: not manager._hook_abandoned)
    assert manager.invoke_hook("post_tool_call", tool_call_id="call-suppressed") == []
    assert _events(manager)[-1]["outcome"] == "suppression_window"


def test_same_identity_running_is_not_callback_global(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    hold = threading.Event()
    entered = threading.Event()
    starts: list[str] = []

    def blocker(**kwargs):
        starts.append(kwargs["session_id"])
        entered.set()
        hold.wait(timeout=2.0)

    manager = PluginManager()
    manager._hooks["post_tool_call"] = [blocker]

    first = threading.Thread(
        target=lambda: manager.invoke_hook("post_tool_call", session_id="session-a"),
        daemon=True,
    )
    first.start()
    assert entered.wait(timeout=1.0)

    assert manager.invoke_hook("post_tool_call", session_id="session-a") == []
    second = threading.Thread(
        target=lambda: manager.invoke_hook("post_tool_call", session_id="session-b"),
        daemon=True,
    )
    second.start()
    _wait_for(lambda: len(starts) == 2)

    assert _events(manager)[-1]["outcome"] == "same_identity_running"
    assert starts == ["session-a", "session-b"]
    hold.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)


def test_worker_start_failure_is_recorded_and_gate_is_released(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    real_start = threading.Thread.start
    attempts = 0

    def fail_once(thread):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cannot start sk-test_abcdefghijklmnop")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_once)
    calls: list[int] = []

    def callback(**_kwargs):
        calls.append(1)

    manager = PluginManager()
    manager._hooks["post_tool_call"] = [callback]

    assert manager.invoke_hook("post_tool_call", turn_id="turn-1") == []
    assert manager.invoke_hook("post_tool_call", turn_id="turn-1") == []
    event = _events(manager)[0]
    assert event["outcome"] == "worker_start_failed"
    assert "exception_type" not in event
    assert "exception_message" not in event
    assert "sk-test_abcdefghijklmnop" not in repr(event)
    assert calls == [1]


def test_callback_exception_is_redacted_bounded_and_does_not_stop_later_callbacks(
    monkeypatch, caplog
):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    secret = "sk-test_abcdefghijklmnop"

    def boom(**_kwargs):
        raise RuntimeError(f"authorization={secret}\n" + "x" * 2_000)

    survived: list[bool] = []
    manager = PluginManager()
    manager._hooks["post_tool_call"] = [boom, lambda **_kwargs: survived.append(True)]

    with caplog.at_level(logging.DEBUG, logger="hermes_cli.plugins"):
        assert manager.invoke_hook(
            "post_tool_call",
            tool_call_id="raw-correlation-id",
            args={"password": secret},
        ) == []

    event = _events(manager)[0]
    assert event["outcome"] == "callback_exception"
    assert event["exception_type"] == "RuntimeError"
    assert secret not in event["exception_message"]
    assert len(event["exception_message"]) <= 512
    assert "raw-correlation-id" not in repr(event)
    assert secret not in caplog.text
    assert survived == [True]


def test_callback_failure_dedupe_cardinality_is_bounded(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    manager = PluginManager()

    def boom(**kwargs):
        raise RuntimeError(f"failure-{kwargs['session_id']}")

    manager._hooks["post_tool_call"] = [boom]
    for index in range(300):
        manager.invoke_hook("post_tool_call", session_id=str(index))

    assert len(manager._hook_failures_reported) == 256
    assert len(manager.get_hook_callback_telemetry()) == 256


def test_identity_priority_digest_stability_and_manager_locality(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)

    def boom(**_kwargs):
        raise RuntimeError("boom")

    manager = PluginManager()
    manager._hooks["post_tool_call"] = [boom]
    payloads = [
        {"tool_call_id": "tool", "turn_id": "turn", "session_id": "session"},
        {"tool_call_id": "tool", "turn_id": "other", "session_id": "other"},
        {"turn_id": "turn", "session_id": "session"},
        {"session_id": "session"},
        {},
    ]
    for payload in payloads:
        manager.invoke_hook("post_tool_call", **payload)

    events = _events(manager)
    assert [event["identity_field"] for event in events] == [
        "tool_call_id",
        "tool_call_id",
        "turn_id",
        "session_id",
        "none",
    ]
    assert events[0]["correlation_digest"] == events[1]["correlation_digest"]
    assert events[0]["correlation_digest"] != events[2]["correlation_digest"]
    assert len(events[0]["correlation_digest"]) == 32
    assert events[-1]["correlation_digest"] == "none"

    other = PluginManager()
    other._hooks["post_tool_call"] = [boom]
    other.invoke_hook("post_tool_call", tool_call_id="tool")
    assert _events(other)[0]["correlation_digest"] != events[0]["correlation_digest"]


def test_ownership_comes_from_registration_metadata_and_disposal_clears_it(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)

    def boom(**_kwargs):
        raise RuntimeError("boom")

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="display-name", key="trusted/plugin-key"), manager
    )
    registration = context.register_hook("post_tool_call", boom)
    assert manager._hook_callback_owners[id(boom)][-1][1] == "trusted/plugin-key"
    manager.invoke_hook("post_tool_call", plugin="payload-forgery")
    assert _events(manager)[0]["plugin"] == "trusted/plugin-key"

    registration.dispose()
    assert id(boom) not in manager._hook_callback_owners

    manager._hooks["post_tool_call"] = [boom]
    manager.invoke_hook("post_tool_call")
    assert _events(manager)[-1]["plugin"] == "core/unowned"


def test_disposing_older_duplicate_registration_preserves_newer_owner(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)

    def boom(**_kwargs):
        raise RuntimeError("boom")

    manager = PluginManager()
    first = PluginContext(PluginManifest(name="first", key="first"), manager)
    second = PluginContext(PluginManifest(name="second", key="second"), manager)
    first_registration = first.register_hook("post_tool_call", boom)
    second_registration = second.register_hook("post_tool_call", boom)

    first_registration.dispose()
    manager.invoke_hook("post_tool_call", session_id="session")
    assert _events(manager)[0]["plugin"] == "second"

    second_registration.dispose()
    assert id(boom) not in manager._hook_callback_owners


def test_disposing_newer_duplicate_registration_restores_older_owner(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)

    def boom(**_kwargs):
        raise RuntimeError("boom")

    manager = PluginManager()
    first = PluginContext(PluginManifest(name="first", key="first"), manager)
    second = PluginContext(PluginManifest(name="second", key="second"), manager)
    first_registration = first.register_hook("post_tool_call", boom)
    second_registration = second.register_hook("post_tool_call", boom)

    second_registration.dispose()
    manager.invoke_hook("post_tool_call", session_id="session")
    assert _events(manager)[0]["plugin"] == "first"

    first_registration.dispose()
    assert id(boom) not in manager._hook_callback_owners


def test_duplicate_callback_owner_restore_ignores_other_callback_on_same_hook(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)

    def shared(**_kwargs):
        raise RuntimeError("boom")

    def other(**_kwargs):
        return None

    manager = PluginManager()
    first = PluginContext(PluginManifest(name="first", key="first"), manager)
    unrelated = PluginContext(PluginManifest(name="unrelated", key="unrelated"), manager)
    newest = PluginContext(PluginManifest(name="newest", key="newest"), manager)
    first.register_hook("post_tool_call", shared)
    unrelated.register_hook("post_tool_call", other)
    newest_registration = newest.register_hook("post_tool_call", shared)

    newest_registration.dispose()
    manager.invoke_hook("post_tool_call", session_id="session")
    assert _events(manager)[0]["plugin"] == "first"


def test_unload_all_clears_callback_ownership(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="owned", key="owned"), manager)
    context.register_hook("post_tool_call", lambda **_kwargs: None)
    assert manager._hook_callback_owners

    manager.unload()
    assert manager._hook_callback_owners == {}


def test_manager_reset_clears_telemetry_dedupe_and_owners_but_keeps_sequence_monotonic():
    manager = PluginManager()

    def callback(**_kwargs):
        return None

    manager._hook_callback_owners[id(callback)] = [(callback, "owner", object())]
    manager._hook_failures_reported[("hook",)] = None
    manager._record_hook_callback_telemetry(
        outcome="same_identity_running",
        hook_name="post_tool_call",
        callback=callback,
        kwargs={"session_id": "before-reset"},
    )
    assert manager.get_hook_callback_telemetry()[-1]["sequence"] == 1

    manager._reset_after_unload_all([])

    assert manager._hook_callback_owners == {}
    assert manager._hook_failures_reported == {}
    manager._record_hook_callback_telemetry(
        outcome="same_identity_running",
        hook_name="post_tool_call",
        callback=callback,
        kwargs={"session_id": "after-reset"},
    )
    assert manager.get_hook_callback_telemetry()[-1]["sequence"] == 2


def test_retention_cardinality_and_snapshot_are_bounded_and_immutable():
    manager = PluginManager()

    def callback(**_kwargs):
        return None

    for index in range(300):
        manager._record_hook_callback_telemetry(
            outcome="same_identity_running",
            hook_name=f"hook-{index}",
            callback=callback,
            kwargs={"session_id": f"session-{index}"},
            timeout=1.0,
        )

    snapshot = manager.get_hook_callback_telemetry(limit=1_000)
    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 256
    assert snapshot[0]["sequence"] == 45
    assert snapshot[-1]["sequence"] == 300
    assert len({event["outcome"] for event in snapshot}) == 1
    with pytest.raises(TypeError):
        snapshot[-1]["outcome"] = "tampered"
    assert manager.get_hook_callback_telemetry(limit=1)[0] is not snapshot[-1]
    assert len(manager.get_hook_callback_telemetry(limit=3)) == 3
    assert manager.get_hook_callback_telemetry(limit=0) == ()


def test_labels_are_canonically_redacted_printable_and_bounded(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    import hermes_cli.plugins_dispatch as dispatch

    secret = "sk-test_abcdefghijklmnop"

    def callback(**_kwargs):
        raise RuntimeError(secret)

    callback.__module__ = f"  module\n{secret}  "
    callback.__qualname__ = "q\x00" + "z" * 400
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="name", key=f" plugin\t{secret} "), manager
    )
    context.register_hook("post_tool_call", callback)
    manager.invoke_hook("post_tool_call", session_id="session")

    event = _events(manager)[0]
    rendered = repr(event)
    assert secret not in rendered
    for field in ("hook", "plugin", "callback", "exception_type", "exception_message"):
        assert event[field].isprintable()
    assert len(event["hook"]) <= 128
    assert len(event["plugin"]) <= 128
    assert len(event["callback"]) <= 192
    assert dispatch._canonical_telemetry_text(
        "Authorization: Bearer top-secret-token",
        max_chars=128,
        fallback="redacted",
    ) == "Authorization: Bearer ***"


def test_redaction_failure_falls_back_without_changing_callback_behavior(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    from agent import redact as redact_mod

    real_redactor = redact_mod.redact_sensitive_text
    caller_thread = threading.current_thread()

    def fail_in_caller(*args, **kwargs):
        if threading.current_thread() is caller_thread:
            raise RuntimeError("redactor failed")
        return real_redactor(*args, **kwargs)

    monkeypatch.setattr(redact_mod, "redact_sensitive_text", fail_in_caller)
    reached: list[bool] = []

    def boom(**_kwargs):
        raise RuntimeError("sk-test_abcdefghijklmnop")

    manager = PluginManager()
    manager._hooks["post_tool_call"] = [boom, lambda **_kwargs: reached.append(True)]

    assert manager.invoke_hook("post_tool_call", session_id="session") == []
    event = _events(manager)[0]
    assert event["callback"] == "unknown"
    assert event["exception_message"] == "redacted"
    assert reached == [True]


def test_store_failure_isolated_from_pre_tool_fail_closed(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)
    hold = threading.Event()

    def blocker(**_kwargs):
        hold.wait(timeout=1.0)

    class BrokenStore:
        def append(self, _event):
            raise RuntimeError("store failed")

        def __iter__(self):
            return iter(())

    manager = PluginManager()
    manager._hook_callback_telemetry_events = BrokenStore()
    manager._hooks["pre_tool_call"] = [blocker]

    assert manager.invoke_hook("pre_tool_call", session_id="session") == [
        {
            "action": "block",
            "message": "pre_tool_call plugin callback timed out or is still running",
        }
    ]
    hold.set()



def test_telemetry_failure_reporting_failure_is_also_isolated(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)
    hold = threading.Event()

    def blocker(**_kwargs):
        hold.wait(timeout=1.0)

    manager = PluginManager()
    manager._hooks["pre_tool_call"] = [blocker]
    monkeypatch.setattr(
        manager,
        "_record_hook_callback_telemetry",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )
    monkeypatch.setattr(
        "hermes_cli.plugins_dispatch.logger.debug",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("debug failed")),
    )

    assert manager.invoke_hook("pre_tool_call", session_id="session") == [
        {
            "action": "block",
            "message": "pre_tool_call plugin callback timed out or is still running",
        }
    ]
    hold.set()


@pytest.mark.parametrize(
    ("hook_name", "expected"),
    [
        (
            "pre_tool_call",
            [{"action": "block", "message": "pre_tool_call plugin callback timed out or is still running"}],
        ),
        ("pre_verify", []),
        ("on_session_start", []),
        ("post_tool_call", []),
        ("transform_tool_result", []),
    ],
)
def test_telemetry_failure_never_changes_timeout_fail_mode(
    monkeypatch, hook_name, expected
):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)
    hold = threading.Event()

    def blocker(**_kwargs):
        hold.wait(timeout=1.0)

    manager = PluginManager()
    manager._hooks[hook_name] = [blocker]
    monkeypatch.setattr(
        manager,
        "_record_hook_callback_telemetry",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )

    assert manager.invoke_hook(hook_name, session_id="session") == expected
    hold.set()


def test_telemetry_failure_does_not_break_exception_isolation(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    reached: list[bool] = []

    def boom(**_kwargs):
        raise RuntimeError("callback failed")

    manager = PluginManager()
    manager._hooks["post_tool_call"] = [boom, lambda **_kwargs: reached.append(True)]
    monkeypatch.setattr(
        manager,
        "_record_hook_callback_telemetry",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )

    assert manager.invoke_hook("post_tool_call", session_id="session") == []
    assert reached == [True]


@pytest.mark.parametrize(
    ("hook_name", "expected"),
    [
        (
            "pre_tool_call",
            [
                {
                    "action": "block",
                    "message": "pre_tool_call plugin callback timed out or is still running",
                }
            ],
        ),
        ("post_tool_call", []),
    ],
    ids=["fail-closed", "fail-open"],
)
@pytest.mark.parametrize(
    "exception_type",
    [SystemExit, KeyboardInterrupt],
    ids=["system-exit", "keyboard-interrupt"],
)
@pytest.mark.parametrize(
    "telemetry_boundary",
    ["event-construction", "storage", "diagnostic-logger"],
)
def test_telemetry_baseexception_preserves_timeout_cleanup_suppression_and_later_callbacks(
    monkeypatch, hook_name, expected, exception_type, telemetry_boundary
):
    """Side-channel BaseException must not change timeout-path control flow."""
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)
    hold = threading.Event()
    later_calls: list[str] = []

    def blocker(**_kwargs):
        hold.wait(timeout=1.0)

    manager = PluginManager()
    manager._hooks[hook_name] = [
        blocker,
        lambda **_kwargs: later_calls.append("later"),
    ]

    def raise_baseexception(message):
        raise exception_type(message)

    if telemetry_boundary == "event-construction":
        monkeypatch.setattr(
            manager,
            "_record_hook_callback_telemetry",
            lambda **_kwargs: raise_baseexception("telemetry-event-failure"),
        )
    elif telemetry_boundary == "storage":
        class BrokenStore:
            def append(self, _event):
                raise_baseexception("telemetry-storage-failure")

            def __iter__(self):
                return iter(())

        manager._hook_callback_telemetry_events = BrokenStore()
    else:
        monkeypatch.setattr(
            manager,
            "_record_hook_callback_telemetry",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("telemetry-event-failure")
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins_dispatch.logger.debug",
            lambda *_args, **_kwargs: raise_baseexception(
                "telemetry-diagnostic-failure"
            ),
        )

    try:
        assert manager.invoke_hook(hook_name, session_id="session") == expected
        assert later_calls == ["later"]
    finally:
        hold.set()

    _wait_for(
        lambda: not manager._hook_abandoned
        and not manager._hook_running_callbacks
    )

    assert manager.invoke_hook(hook_name, session_id="session") == expected
    assert later_calls == ["later", "later"]


def test_callback_systemexit_remains_isolated_when_telemetry_raises_keyboardinterrupt(
    monkeypatch,
):
    """Callback SystemExit and telemetry KeyboardInterrupt use distinct boundaries."""
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    later_calls: list[bool] = []

    def callback_exits(**_kwargs):
        raise SystemExit("callback-system-exit")

    manager = PluginManager()
    manager._hooks["post_tool_call"] = [
        callback_exits,
        lambda **_kwargs: later_calls.append(True),
    ]
    monkeypatch.setattr(
        manager,
        "_record_hook_callback_telemetry",
        lambda **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("telemetry-keyboard-interrupt")
        ),
    )

    assert manager.invoke_hook("post_tool_call", session_id="session") == []
    assert later_calls == [True]


def test_callback_keyboardinterrupt_still_propagates_without_entering_telemetry(
    monkeypatch,
):
    """A real callback KeyboardInterrupt is not a telemetry side-channel failure."""
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 1.0)
    telemetry_calls: list[bool] = []
    later_calls: list[bool] = []

    def callback_interrupts(**_kwargs):
        raise KeyboardInterrupt("callback-keyboard-interrupt")

    def telemetry_failure(**_kwargs):
        telemetry_calls.append(True)
        raise SystemExit("telemetry-system-exit")

    manager = PluginManager()
    manager._hooks["post_tool_call"] = [
        callback_interrupts,
        lambda **_kwargs: later_calls.append(True),
    ]
    monkeypatch.setattr(manager, "_record_hook_callback_telemetry", telemetry_failure)

    with pytest.raises(KeyboardInterrupt, match="callback-keyboard-interrupt"):
        manager.invoke_hook("post_tool_call", session_id="session")
    assert telemetry_calls == []
    assert later_calls == []
