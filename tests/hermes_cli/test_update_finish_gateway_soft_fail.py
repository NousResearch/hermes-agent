"""#106692: already-up-to-date / handoff tail must soft-fail Windows gateway resume.

After a successful ``_repair_current_checkout``, ``_finish_already_up_to_date``
called ``_resume_windows_gateways_after_update`` directly. When Job Object
teardown (#48820) left ``_wait_for_gateway_ready`` empty, that raise aborted
an otherwise successful update (``HERMES_UPDATE_REEXEC=1`` handoff child or
normal already-up-to-date). The pull path already soft-fails via
``_resume_windows_gateways_and_merge_outcome``.

The atexit handler registered in ``_cmd_update_impl`` then retried resume
with ``resume_needed`` still True, producing "Exception ignored in atexit
callback".
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.main as hm
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli import update_cmd
from hermes_cli.update_cmd_windows import _resume_windows_gateways_after_update


def _token(profiles: dict) -> dict:
    return {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": [],
        "unmapped": [],
    }


def _plan():
    return update_cmd._CheckoutPlan(
        auto_stash_ref=None,
        commit_count=0,
        in_place_update=False,
        parked_branch_switched=False,
        prompt_for_restore=False,
        switch_block_reason=None,
        upstream_checked=True,
    )


def _call_finish(*, checkout_complete: bool, token):
    update_cmd._finish_already_up_to_date(
        ["git"],
        "main",
        "main",
        _plan(),
        assume_yes=True,
        gateway_mode=False,
        gw_input_fn=None,
        pre_update_snapshot_id=None,
        desktop_dir=None,
        had_desktop_app_before_update=False,
        active_lazy_features=[],
        active_tool_dependencies=[],
        _windows_gateway_resume=token,
    )


class TestFinishAlreadyUpToDateGatewaySoftFail:
    """Handoff / already-up-to-date tail: liveness miss must not abort a good checkout."""

    @pytest.fixture(autouse=True)
    def _windows_job_object_dead_respawn(self, monkeypatch):
        """Same #48820 liveness hole: launch returns True, ready poll is empty."""
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
            gateway_windows, "_wait_for_gateway_ready", lambda **_kw: []
        )
        monkeypatch.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
        monkeypatch.setattr(
            update_cmd, "_apply_pending_fleet_restart_catchup", lambda: None
        )
        monkeypatch.setattr(
            update_cmd, "_repair_current_checkout", lambda **_kw: True
        )

    def test_unverified_alive_does_not_abort_when_checkout_complete(
        self, monkeypatch
    ):
        """Successful venv repair + dead Job Object respawn → warning, no raise."""
        monkeypatch.setenv("HERMES_UPDATE_REEXEC", "1")
        token = _token({"default": 1111})
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **k: printed.append(a)):
            _call_finish(checkout_complete=True, token=token)

        text = " ".join(str(part) for row in printed for part in row)
        assert "Windows gateway service restart incomplete" in text
        assert "not verified alive" in text
        # Recovery guidance from the liveness gate must still surface.
        assert "could not be verified" in text or "hermes gateway restart" in text

    def test_atexit_does_not_double_fire_resume_after_explicit_finish(
        self, monkeypatch
    ):
        """After the explicit resume in ``_finish_already_up_to_date``, the
        ``_cmd_update_impl`` atexit callback must be a no-op (resume_needed
        cleared, or the handler unregistered)."""
        token = _token({"default": 1111})
        resume_attempts = []

        def tracking_resume(state):
            resume_attempts.append(bool(state and state.get("resume_needed")))
            return _resume_windows_gateways_after_update(state)

        monkeypatch.setattr(hm, "_resume_windows_gateways_after_update", tracking_resume)

        printed = []
        with patch("builtins.print", side_effect=lambda *a, **k: printed.append(a)):
            _call_finish(checkout_complete=True, token=token)
            # Simulate the atexit handler registered at update_cmd.py:1270-1271.
            tracking_resume(token)

        assert token.get("resume_needed") is False
        assert resume_attempts == [True, False]
        text = " ".join(str(part) for row in printed for part in row)
        assert text.count("Windows gateway service restart incomplete") == 1

    def test_checkout_incomplete_still_exits_after_soft_fail_resume(
        self, monkeypatch
    ):
        """Venv repair failed → still ``sys.exit(1)``; resume miss must not
        replace that fail-closed gate with an uncaught RuntimeError."""
        monkeypatch.setattr(
            update_cmd, "_repair_current_checkout", lambda **_kw: False
        )
        monkeypatch.setattr(update_cmd, "_finalize_receipt", lambda *_a, **_k: None)
        token = _token({"default": 1111})
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **k: printed.append(a)):
            with pytest.raises(SystemExit) as excinfo:
                _call_finish(checkout_complete=False, token=token)
        assert excinfo.value.code == 1
        text = " ".join(str(part) for row in printed for part in row)
        assert "Windows gateway service restart incomplete" in text
        assert token.get("resume_needed") is False
