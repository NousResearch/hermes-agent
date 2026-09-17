"""Invariant tests for the stay-awake inhibitor (agent/stay_awake.py).

Contract under test:
- disabled → zero side effects (no process spawned, no OS call);
- turn_scope refcounts a single shared inhibitor across nested/concurrent turns
  (Windows SetThreadExecutionState is not nestable, so exactly one enter/exit pair
  must wrap the whole overlap);
- closed-display mode uses only the exact non-interactive pmset commands and
  restores the value observed before the scope;
- config default keeps the feature off.

Host-specific spawn behaviour (caffeinate/systemd-inhibit argv) is intentionally
not tested by faking ``platform.system()`` — repo policy tests OS behaviour on
that OS only.
"""
import subprocess
from unittest.mock import patch

import agent.stay_awake as stay_awake
from agent.stay_awake import StayAwake, turn_scope


def test_disabled_is_complete_noop():
    with patch("agent.stay_awake.subprocess.Popen") as popen:
        with StayAwake(enabled=False) as sa:
            assert sa._process is None
        popen.assert_not_called()


def test_macos_caffeinate_is_bound_to_the_parent_process(monkeypatch):
    monkeypatch.setattr(stay_awake.os, "getpid", lambda: 12345)
    with patch("agent.stay_awake.subprocess.Popen") as popen:
        StayAwake(enabled=True)._start_macos()
    assert popen.call_args.args[0] == ["/usr/bin/caffeinate", "-i", "-w", "12345"]


def test_turn_scope_refcounts_one_shared_inhibitor(monkeypatch):
    events = []

    class Fake:
        def __init__(self, enabled=False, *, mode="idle"):
            pass

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *a):
            events.append("exit")

    monkeypatch.setattr(stay_awake, "StayAwake", Fake)
    with turn_scope(enabled=True):
        with turn_scope(enabled=True):
            assert events == ["enter"]  # nested turn reuses the live inhibitor
        assert events == ["enter"]  # inner exit must NOT release it
    assert events == ["enter", "exit"]  # last turn out releases exactly once


def test_turn_scope_defaults_off_from_config(monkeypatch):
    monkeypatch.setattr(stay_awake, "_config_enabled", lambda: False)

    class Explode:
        def __init__(self, enabled=False, *, mode="idle"):
            raise AssertionError("inhibitor must not start when disabled")

    monkeypatch.setattr(stay_awake, "StayAwake", Explode)
    with turn_scope():
        pass


def test_closed_display_pmset_restores_the_previous_value(monkeypatch):
    calls = []
    reads = iter((0, 1))

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["/usr/bin/pmset", "-g"]:
            value = next(reads)
            return subprocess.CompletedProcess(command, 0, stdout=f" SleepDisabled\t{value}\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stay_awake.subprocess, "run", fake_run)
    inhibitor = StayAwake(enabled=True, mode="closed-display")
    inhibitor._start_macos_closed_display()
    inhibitor._stop_macos_closed_display()

    assert [call[0] for call in calls] == [
        ["/usr/bin/pmset", "-g"],
        ["/usr/bin/sudo", "-n", "/usr/bin/pmset", "-a", "disablesleep", "1"],
        ["/usr/bin/pmset", "-g"],
        ["/usr/bin/sudo", "-n", "/usr/bin/pmset", "-a", "disablesleep", "0"],
    ]
    assert all(call[1].get("check") is True for call in calls)


def test_closed_display_never_prompts_and_falls_back_to_idle(monkeypatch):
    monkeypatch.setattr(stay_awake, "_read_sleep_disabled", lambda: 0)
    monkeypatch.setattr(
        stay_awake,
        "_set_sleep_disabled",
        lambda value: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["/usr/bin/sudo", "-n"])
        ),
    )
    with patch.object(StayAwake, "_start_macos") as fallback:
        StayAwake(enabled=True, mode="closed-display")._start_macos_closed_display()
    fallback.assert_called_once_with()


def test_invalid_mode_normalizes_to_idle():
    assert StayAwake(enabled=True, mode="unknown")._mode == "idle"


def test_stale_recovery_is_mac_only(monkeypatch):
    monkeypatch.setattr(stay_awake.platform, "system", lambda: "Linux")
    assert stay_awake.recover_stale_power_protect_if_needed() == (False, None)


def test_stale_recovery_returns_a_user_notice(monkeypatch):
    monkeypatch.setattr(stay_awake.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(stay_awake, "recover_stale_power_protect", lambda *_args: True)
    recovered, message = stay_awake.recover_stale_power_protect_if_needed()
    assert recovered is True
    assert message is not None
    assert "stale Power Protect" in message
