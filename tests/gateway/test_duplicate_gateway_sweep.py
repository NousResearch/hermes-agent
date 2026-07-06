"""Tests for the orphan-gateway sweep in start_gateway (#DAN-2057 incident).

_handle_duplicate_gateway_runtimes closes the gap where `gateway run
--replace` killed only the pid-file-tracked instance while untracked
orphan gateways survived and stacked (three concurrent gateways, frozen
cron scheduler, 2026-07-04).
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

import gateway.run as gw_run


def _patch_scan(pids):
    """Patch the lazily-imported hermes_cli.gateway._scan_gateway_pids."""
    mock_mod = MagicMock()
    if isinstance(pids, Exception):
        mock_mod._scan_gateway_pids.side_effect = pids
    else:
        mock_mod._scan_gateway_pids.return_value = list(pids)
    return patch.dict(sys.modules, {"hermes_cli.gateway": mock_mod}), mock_mod


class TestReplaceSweep:
    def test_sweeps_orphans_term_then_kill_for_survivor(self):
        """TERM every orphan; SIGKILL only the one that survives the wait."""
        scan_patch, _ = _patch_scan([111, 222])
        term_calls = []

        def fake_terminate(pid, *, force=False):
            term_calls.append((pid, force))

        # 111 dies after TERM; 222 survives until force-killed.
        def fake_pid_exists(pid):
            return pid == 222 and (222, True) not in term_calls

        with scan_patch, \
                patch("gateway.status.terminate_pid", side_effect=fake_terminate), \
                patch("gateway.status._pid_exists", side_effect=fake_pid_exists), \
                patch("gateway.status.write_planned_stop_marker") as marker, \
                patch.object(gw_run.time, "sleep"), \
                patch.object(gw_run.time, "monotonic", side_effect=[0.0, 1.0, 2.0, 11.0, 12.0]):
            assert gw_run._handle_duplicate_gateway_runtimes(replace=True) is True

        assert (111, False) in term_calls
        assert (222, False) in term_calls
        assert (222, True) in term_calls          # survivor force-killed
        assert (111, True) not in term_calls      # dead orphan not force-killed
        assert marker.call_count == 2

    def test_no_orphans_no_kills(self):
        scan_patch, _ = _patch_scan([])
        with scan_patch, patch("gateway.status.terminate_pid") as term:
            assert gw_run._handle_duplicate_gateway_runtimes(replace=True) is True
        term.assert_not_called()

    def test_scan_failure_fails_open(self):
        """A broken scan must never keep the gateway down."""
        scan_patch, _ = _patch_scan(RuntimeError("ps exploded"))
        with scan_patch, patch("gateway.status.terminate_pid") as term:
            assert gw_run._handle_duplicate_gateway_runtimes(replace=True) is True
        term.assert_not_called()

    def test_term_failure_on_one_orphan_continues_to_next(self):
        scan_patch, _ = _patch_scan([111, 222])
        term_calls = []

        def fake_terminate(pid, *, force=False):
            if pid == 111 and not force:
                raise PermissionError("nope")
            term_calls.append((pid, force))

        with scan_patch, \
                patch("gateway.status.terminate_pid", side_effect=fake_terminate), \
                patch("gateway.status._pid_exists", return_value=False), \
                patch("gateway.status.write_planned_stop_marker"), \
                patch.object(gw_run.time, "sleep"):
            assert gw_run._handle_duplicate_gateway_runtimes(replace=True) is True

        assert (222, False) in term_calls


class TestNonReplaceRefusal:
    def test_orphans_present_refuses_startup_without_killing(self, capsys):
        scan_patch, _ = _patch_scan([333])
        with scan_patch, patch("gateway.status.terminate_pid") as term:
            assert gw_run._handle_duplicate_gateway_runtimes(replace=False) is False
        term.assert_not_called()
        out = capsys.readouterr().out
        assert "333" in out
        assert "--replace" in out

    def test_clean_scan_allows_startup(self):
        scan_patch, _ = _patch_scan([])
        with scan_patch:
            assert gw_run._handle_duplicate_gateway_runtimes(replace=False) is True


class TestScanArguments:
    def test_scan_scoped_to_current_profile_and_strict_runtimes(self):
        """Sweep must not match other profiles or restart-manager CLIs."""
        scan_patch, mock_mod = _patch_scan([])
        with scan_patch:
            gw_run._handle_duplicate_gateway_runtimes(replace=True)
        _, kwargs = mock_mod._scan_gateway_pids.call_args
        assert kwargs["all_profiles"] is False
        assert kwargs["include_restart_managers"] is False
