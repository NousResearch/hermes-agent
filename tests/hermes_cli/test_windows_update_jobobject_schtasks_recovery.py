"""#107002: post-update Windows gateway liveness empty + registered
Scheduled Task → ``schtasks /Run`` recovery, then re-poll.

A parent Job Object can kill the respawned gateway during updater
teardown (#48820). When the first ``_wait_for_gateway_ready`` poll is
empty and a Hermes Scheduled Task is registered, the updater must try
``schtasks /Run`` so Task Scheduler starts the gateway outside that Job
Object, then poll again.

Fail-open stays the pre-fix path: no task / query error / non-zero
``/Run`` / still-empty second poll → original RuntimeError, never a
fake ✓. Ordinary ``hermes gateway start()`` is unchanged (Scheduled
Task remains login persistence only).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.main as hm
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli import update_cmd
from hermes_cli.update_cmd import _resume_windows_gateways_after_update
from hermes_cli.update_cmd_windows import _verify_relaunched_gateways_alive


_TASK = "Hermes_Gateway"


def _token(profiles: dict) -> dict:
    return {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": [],
        "unmapped": [],
    }


def _install_windows_resume_stubs(monkeypatch) -> None:
    monkeypatch.setattr(hm, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(
        gateway, "launch_detached_profile_gateway_restart", lambda *_a: True
    )
    monkeypatch.setattr(
        gateway, "launch_detached_gateway_restart_by_cmdline", lambda *_a: True
    )
    monkeypatch.setattr(
        gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: _TASK)


def _install_ready_then_recover(monkeypatch, schtasks_calls: list, *, after_run_pids):
    """First liveness poll is empty; a later poll returns *after_run_pids* only
    after ``schtasks`` was invoked with ``/Run``."""

    def fake_wait(**_kw):
        if any(call and call[0] == "/Run" for call in schtasks_calls):
            return list(after_run_pids)
        return []

    def fake_schtasks(args):
        schtasks_calls.append(list(args))
        return (0, "", "")

    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", fake_wait)
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)


class TestRelaunchSchtasksRecovery:
    def test_empty_liveness_plus_registered_task_recovers_via_run(
        self, monkeypatch
    ):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[4242])

        token = _token({"default": 1111})
        with patch("builtins.print"):
            _resume_windows_gateways_after_update(token)

        assert any(call and call[0] == "/Run" for call in schtasks_calls)
        assert ["/Run", "/TN", _TASK] in schtasks_calls
        assert token["resume_needed"] is False

    def test_verify_path_recovers_without_raising(self, monkeypatch):
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        monkeypatch.setattr(gateway_windows, "get_task_name", lambda: _TASK)
        monkeypatch.setattr(
            gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None
        )
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[4242])

        token = _token({"default": 1111})
        _verify_relaunched_gateways_alive(token, {"default": 1111}, [])

        assert ["/Run", "/TN", _TASK] in schtasks_calls

    def test_unregistered_task_raises_and_does_not_run(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
        schtasks_calls: list[list[str]] = []

        def fake_wait(**_kw):
            return []

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", fake_wait)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        token = _token({"default": 1111})
        with patch("builtins.print"):
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)

        assert not any(call and call[0] == "/Run" for call in schtasks_calls)
        assert token["resume_needed"] is True

    def test_first_liveness_ready_never_touches_schtasks(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        schtasks_calls: list[list[str]] = []
        monkeypatch.setattr(
            gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [777]
        )

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        token = _token({"default": 1111})
        with patch("builtins.print"):
            _resume_windows_gateways_after_update(token)

        assert schtasks_calls == []
        assert token["resume_needed"] is False

    def test_schtasks_run_nonzero_stays_on_original_failure(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        schtasks_calls: list[list[str]] = []

        def fake_wait(**_kw):
            return []

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (1, "", "access denied")

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", fake_wait)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        token = _token({"default": 1111})
        with patch("builtins.print"):
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)

        assert ["/Run", "/TN", _TASK] in schtasks_calls
        assert token["resume_needed"] is True

    def test_second_liveness_still_empty_raises(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[])

        token = _token({"default": 1111})
        with patch("builtins.print"):
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)

        assert ["/Run", "/TN", _TASK] in schtasks_calls
        assert token["resume_needed"] is True

    def test_task_query_error_does_not_run(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)

        def boom():
            raise OSError("schtasks query failed")

        monkeypatch.setattr(gateway_windows, "is_task_registered", boom)
        schtasks_calls: list[list[str]] = []

        def fake_wait(**_kw):
            return []

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", fake_wait)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        token = _token({"default": 1111})
        with patch("builtins.print"):
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)

        assert not any(call and call[0] == "/Run" for call in schtasks_calls)


class TestColdStartSchtasksRecovery:
    def _install_cold_start_stubs(self, monkeypatch) -> None:
        monkeypatch.setattr(hm, "_is_windows", lambda: True)
        monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **_k: [])
        monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: False)
        monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda: 9001)
        monkeypatch.setattr(
            gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None
        )
        monkeypatch.setattr(gateway_windows, "get_task_name", lambda: _TASK)

    def test_empty_liveness_plus_registered_task_recovers_via_run(
        self, monkeypatch
    ):
        self._install_cold_start_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[4242])

        with patch("builtins.print"):
            assert update_cmd._cold_start_windows_gateway_after_update() is True

        assert ["/Run", "/TN", _TASK] in schtasks_calls

    def test_unregistered_task_raises_and_does_not_run(self, monkeypatch):
        self._install_cold_start_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
        schtasks_calls: list[list[str]] = []
        monkeypatch.setattr(
            gateway_windows, "_wait_for_gateway_ready", lambda **_kw: []
        )

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        with pytest.raises(RuntimeError, match="did not become ready"):
            update_cmd._cold_start_windows_gateway_after_update()

        assert not any(call and call[0] == "/Run" for call in schtasks_calls)
