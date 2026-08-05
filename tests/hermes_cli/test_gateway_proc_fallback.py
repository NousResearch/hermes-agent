"""Tests for /proc-based gateway PID detection in Docker environments.

Verifies that _scan_gateway_pids() uses /proc/*/cmdline when available
(Docker without procps) and falls back to ps only when /proc is absent.

See: NousResearch/hermes-agent#7622
"""

import os
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.gateway as gateway_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GATEWAY_CMD = "python -m hermes_cli.main gateway run"
_OTHER_CMD = "python -m some_other_thing"


def _fake_proc_dir(entries: dict):
    """Return side_effects that simulate /proc: isdir → True, listdir → pids,
    open(cmdline) → null-delimited command bytes."""
    def _isdir(path):
        return str(path) == "/proc"

    def _listdir(path):
        if str(path) == "/proc":
            return [str(pid) for pid in entries] + ["self", "version"]
        raise FileNotFoundError(path)

    def _open(path, mode="r", **kwargs):
        path_str = str(path)
        if "/cmdline" in path_str:
            pid = int(path_str.split("/proc/")[1].split("/")[0])
            raw = entries.get(pid, "").encode("utf-8").replace(b" ", b"\x00")
            m = MagicMock()
            m.read.return_value = raw
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            return m
        raise FileNotFoundError(path)

    return _isdir, _listdir, _open


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.linux_only
class TestProcFallback:
    """_scan_gateway_pids reads /proc when available, skips ps.

    Linux-only: ``/proc/<pid>/cmdline`` is the subject. The non-Windows arm of
    ``_scan_gateway_pids`` is selected by the real host here, so the previous
    ``is_windows`` stub is gone — only the /proc filesystem itself is faked so
    the scan sees deterministic PIDs.
    """

    def test_detects_gateway_pid_via_proc(self):
        my_pid = os.getpid()
        entries = {
            my_pid: "python -m hermes_cli.main",   # own process — excluded
            12345: _GATEWAY_CMD,
            99999: _OTHER_CMD,
        }
        _isdir, _listdir, _open = _fake_proc_dir(entries)

        with (
            patch("os.path.isdir", side_effect=_isdir),
            patch("os.listdir", side_effect=_listdir),
            patch("builtins.open", side_effect=_open),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run") as mock_ps,
        ):
            pids = gateway_mod._scan_gateway_pids(set(), all_profiles=True)

        assert 12345 in pids
        assert 99999 not in pids
        mock_ps.assert_not_called()  # ps must NOT be called when /proc worked




    def test_proc_permission_error_skips_pid(self):
        def _isdir(path):
            return str(path) == "/proc"

        def _listdir(path):
            if str(path) == "/proc":
                return ["12345", "self"]
            raise FileNotFoundError

        def _open(path, mode="r", **kwargs):
            raise PermissionError("no access")

        with (
            patch("os.path.isdir", side_effect=_isdir),
            patch("os.listdir", side_effect=_listdir),
            patch("builtins.open", side_effect=_open),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run") as mock_ps,
        ):
            pids = gateway_mod._scan_gateway_pids(set(), all_profiles=True)

        # PermissionError swallowed — empty result, no crash
        assert 12345 not in pids
        mock_ps.assert_not_called()  # /proc dir existed, so ps not called


class TestPsFallbackBsdCompat:
    """Verify the ps fallback command uses BSD/macOS-compatible flags.

    The old ``ps -A eww`` command fails on macOS/BSD because the BSD ``e``
    flag (show environment) is not valid there.  The fix replaces it with
    ``ps -Aww`` which works on both BSD and procps ps.
    """

    def test_ps_flags_are_bsd_compatible(self):
        """ps is called with ``-Aww``, not ``-A eww``."""
        with (
            patch("hermes_cli.gateway.is_windows", return_value=False),
            patch("os.path.isdir", side_effect=lambda p: p != "/proc"),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run") as mock_run,
        ):
            # Return a failing rc=1 so the scan finishes quickly.
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr=""
            )
            pids = gateway_mod._scan_gateway_pids(set())

        assert not pids  # rc=1 → empty result
        # Find the ps call among all subprocess.run invocations
        ps_call = None
        for call_args in mock_run.call_args_list:
            args = call_args[0][0] if call_args[0] else []
            if args and args[0] == "ps":
                ps_call = args
                break
        assert ps_call is not None, "ps was not invoked at all"
        assert "-Aww" in ps_call, (
            f"Expected ps -Aww, got: {ps_call}"
        )
        assert "eww" not in " ".join(ps_call), (
            f"ps eww flag still present: {ps_call}"
        )

    def test_ps_command_includes_pid_and_command_columns(self):
        """The ps command requests ``pid=,command=`` output columns."""
        with (
            patch("hermes_cli.gateway.is_windows", return_value=False),
            patch("os.path.isdir", side_effect=lambda p: p != "/proc"),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="  123 gateway run\n", stderr=""
            )
            gateway_mod._scan_gateway_pids(set())

        ps_call = None
        for call_args in mock_run.call_args_list:
            args = call_args[0][0] if call_args[0] else []
            if args and args[0] == "ps":
                ps_call = args
                break
        assert ps_call is not None
        assert "-o" in ps_call and "pid=,command=" in ps_call, (
            f"Missing -o pid=,command= in: {ps_call}"
        )


class TestGetServicePidsAllProfiles:
    """_get_service_pids(all_profiles=...) discovery across profiles."""

    def test_default_scope_uses_current_profile_label(self):
        """Without all_profiles, only the current profile's launchd agent is
        queried (``launchctl list <label>``)."""
        with (
            patch("hermes_cli.gateway.is_macos", return_value=True),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=False),
            patch(
                "hermes_cli.gateway.get_launchd_label",
                return_value="ai.hermes.gateway.myprofile",
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="123\t0\tai.hermes.gateway.myprofile", stderr=""
            )
            pids = gateway_mod._get_service_pids()

        assert pids == {123}
        # Must have used ``launchctl list <label>``, not bare ``launchctl list``
        launchctl_calls = [
            c[0][0]
            for c in mock_run.call_args_list
            if c[0] and c[0][0] and c[0][0][0] == "launchctl"
        ]
        assert len(launchctl_calls) == 1
        assert launchctl_calls[0][1] == "list"
        assert launchctl_calls[0][2] == "ai.hermes.gateway.myprofile"

    def test_all_profiles_enumerates_all_gateway_labels(self):
        """With all_profiles=True, ``launchctl list`` is called without a label
        filter, and every row whose last column starts with ``ai.hermes.gateway``
        is collected."""
        with (
            patch("hermes_cli.gateway.is_macos", return_value=True),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=False),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=(
                    "123\t0\tai.hermes.gateway.profile-a\n"
                    "456\t0\tai.hermes.gateway.profile-b\n"
                    "789\t0\tcom.apple.some.other.agent\n"
                ),
                stderr="",
            )
            pids = gateway_mod._get_service_pids(all_profiles=True)

        assert pids == {123, 456}
        # The non-gateway label (789) must NOT be included.
        assert 789 not in pids
        # Must have used bare ``launchctl list``
        launchctl_calls = [
            c[0][0]
            for c in mock_run.call_args_list
            if c[0] and c[0][0] and c[0][0][0] == "launchctl"
        ]
        assert len(launchctl_calls) == 1
        assert launchctl_calls[0] == ["launchctl", "list"]

    def test_all_profiles_empty_when_no_gateway_labels(self):
        """When no ai.hermes.gateway* labels exist, all_profiles returns empty."""
        with (
            patch("hermes_cli.gateway.is_macos", return_value=True),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=False),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="789\t0\tcom.apple.some.other.agent\n",
                stderr="",
            )
            pids = gateway_mod._get_service_pids(all_profiles=True)

        assert pids == set()

    def test_all_profiles_handles_broken_pid_column(self):
        """Non-integer PID entries are silently skipped."""
        with (
            patch("hermes_cli.gateway.is_macos", return_value=True),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=False),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=(
                    "-\t0\tai.hermes.gateway.broken\n"
                    "  123  \t0\tai.hermes.gateway.ok\n"
                ),
                stderr="",
            )
            pids = gateway_mod._get_service_pids(all_profiles=True)

        assert pids == {123}

    def test_all_profiles_preserves_systemd_behavior(self):
        """systemd scope is unaffected by the all_profiles switch — it already
        lists every hermes-gateway* unit unconditionally."""
        with (
            patch("hermes_cli.gateway.is_macos", return_value=False),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            def _run_side_effect(args, **kwargs):
                args_list = list(args) if args else []
                cmd_str = " ".join(str(a) for a in args_list[:4])
                if "list-units" in cmd_str:
                    return MagicMock(
                        returncode=0,
                        stdout="hermes-gateway-jarvis.service loaded active running\n",
                        stderr="",
                    )
                if "show" in cmd_str and "MainPID" in cmd_str:
                    return MagicMock(returncode=0, stdout="123\n", stderr="")
                return MagicMock(returncode=0, stdout="", stderr="")

            mock_run.side_effect = _run_side_effect
            pids = gateway_mod._get_service_pids(all_profiles=True)

        assert pids == {123}
