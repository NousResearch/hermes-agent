"""Regression tests for #134831: one-shot settlement must arm the exit watchdog.

``_finalize_single_query`` settles the session (durable flush, finalize hook,
memory-provider shutdown) BEFORE ``_run_cleanup`` arms its own watchdog, and a
normal one-shot exit delivers no signal — so a memory-provider shutdown wedged on
a plugin's thread join (external providers bound the join budget to their HTTP
timeout) had no backstop at all: the answer was already on stdout and the process
lingered until the user killed it. Settlement must arm the signal-style leash
first (idempotent, 2x the cleanup timeout, yields once cleanup's own watchdog
is live so a slow-but-progressing settlement is never cut short).
"""

from __future__ import annotations

from types import SimpleNamespace


def _run_finalize(monkeypatch, events: list) -> None:
    """Drive ``_finalize_single_query`` with every settlement seam recorded."""
    import cli as cli_mod
    from hermes_cli import cli_shutdown

    fake_cli = SimpleNamespace(
        agent=None,
        _release_active_session=lambda: events.append("release_lease"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_arm_exit_watchdog_on_shutdown_signal",
        lambda: events.append("arm_watchdog"),
    )
    monkeypatch.setattr(
        cli_mod, "_flush_one_shot_session_store", lambda c: events.append("flush")
    )
    monkeypatch.setattr(
        cli_mod,
        "_notify_single_query_session_finalize",
        lambda c: events.append("finalize_hook"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_shutdown_agent_memory_provider",
        lambda a: events.append("memory_shutdown"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_wait_for_oneshot_background_completions",
        lambda c: events.append("linger"),
    )
    monkeypatch.setattr(
        cli_mod, "_run_cleanup", lambda **k: events.append("run_cleanup")
    )

    cli_shutdown._finalize_single_query(fake_cli)


def test_finalize_arms_watchdog_before_settlement(monkeypatch):
    events: list = []
    _run_finalize(monkeypatch, events)
    assert events[0] == "arm_watchdog", (
        "the exit watchdog must be armed before any settlement step can wedge "
        "(memory shutdown joins plugin threads; nothing else backstops this phase)"
    )
    assert "memory_shutdown" in events


def test_finalize_keeps_settlement_release_linger_order(monkeypatch):
    events: list = []
    _run_finalize(monkeypatch, events)
    assert events == [
        "arm_watchdog",
        "flush",
        "finalize_hook",
        "memory_shutdown",
        "release_lease",
        "linger",
        "run_cleanup",
    ]


def test_finalize_arms_watchdog_even_when_flush_raises(monkeypatch):
    """Settlement is best-effort; the backstop must not depend on its success."""
    import cli as cli_mod
    from hermes_cli import cli_shutdown

    events: list = []
    fake_cli = SimpleNamespace(
        agent=None,
        _release_active_session=lambda: events.append("release_lease"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_arm_exit_watchdog_on_shutdown_signal",
        lambda: events.append("arm_watchdog"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_flush_one_shot_session_store",
        lambda c: (_ for _ in ()).throw(RuntimeError("flush wedge")),
    )
    monkeypatch.setattr(
        cli_mod,
        "_notify_single_query_session_finalize",
        lambda c: events.append("finalize_hook"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_shutdown_agent_memory_provider",
        lambda a: events.append("memory_shutdown"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_wait_for_oneshot_background_completions",
        lambda c: events.append("linger"),
    )
    monkeypatch.setattr(
        cli_mod, "_run_cleanup", lambda **k: events.append("run_cleanup")
    )

    cli_shutdown._finalize_single_query(fake_cli)
    assert events[0] == "arm_watchdog"
