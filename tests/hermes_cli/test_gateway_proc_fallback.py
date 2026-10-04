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


@pytest.mark.platforms("linux")
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


@pytest.mark.platforms("linux")
class TestPsFallbackBsdCompat:
    """The ps fallback must use flags BSD/macOS ps accepts (#73626, #74075).

    ``ps -A eww`` fails on macOS (BSD ``e`` is not the procps flag), which
    made gateway discovery silently return nothing whenever /proc is absent.
    Linux-only like ``TestProcFallback``: the real host selects the POSIX arm
    and only /proc's absence is faked, to force the ps rung.
    """

    def test_ps_fallback_uses_bsd_compatible_flags_and_columns(self):
        with (
            patch("os.path.isdir", side_effect=lambda p: p != "/proc"),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            assert not gateway_mod._scan_gateway_pids(set())

        ps_calls = [
            c[0][0] for c in mock_run.call_args_list if c[0] and c[0][0] and c[0][0][0] == "ps"
        ]
        assert ps_calls, "ps was not invoked at all"
        ps_call = ps_calls[0]
        assert "-Aww" in ps_call and "eww" not in " ".join(ps_call), ps_call
        assert "-o" in ps_call and "pid=,command=" in ps_call, ps_call


class TestGetServicePidsAllProfiles:
    """_get_service_pids(all_profiles=...) discovery across profiles."""

    @pytest.mark.platforms("macos")
    def test_default_scope_uses_current_profile_label(self):
        """Without all_profiles, only the current profile's launchd agent is
        located (per-label domain-explicit probe, #73627)."""
        located = []

        def _fake_locate(label):
            located.append(label)
            return ("gui/501", 123)

        with (
            patch("hermes_cli.gateway.supports_systemd_services", return_value=False),
            patch(
                "hermes_cli.gateway.get_launchd_label",
                return_value="ai.hermes.gateway.myprofile",
            ),
            patch(
                "hermes_cli.gateway._locate_launchd_gateway_service",
                side_effect=_fake_locate,
            ),
            patch("subprocess.run") as mock_run,
        ):
            pids = gateway_mod._get_service_pids()

        assert pids == {123}
        # Default scope: exactly the current profile's label, no fleet
        # enumeration and no bare `launchctl list` scan.
        assert located == ["ai.hermes.gateway.myprofile"]
        launchctl_calls = [
            c[0][0]
            for c in mock_run.call_args_list
            if c[0] and c[0][0] and c[0][0][0] == "launchctl"
        ]
        assert launchctl_calls == []

    @pytest.mark.platforms("macos")
    def test_all_profiles_enumerates_all_gateway_labels(self):
        """With all_profiles=True, every install-derived gateway label is
        located (#73627), and the bare ``launchctl list`` prefix scan still
        widens the EXCLUDE set with unmapped ai.hermes.gateway* agents
        (#74075 belt-and-suspenders)."""
        located = []
        label_pids = {
            "ai.hermes.gateway": 123,
            "ai.hermes.gateway-profile-b": 456,
        }

        def _fake_locate(label):
            located.append(label)
            pid = label_pids.get(label)
            return ("gui/501", pid) if pid else (None, None)

        with (
            patch("hermes_cli.gateway.supports_systemd_services", return_value=False),
            patch(
                "hermes_cli.gateway.get_launchd_label",
                return_value="ai.hermes.gateway",
            ),
            patch(
                "hermes_cli.gateway.launchd_gateway_labels_for_install",
                return_value=["ai.hermes.gateway", "ai.hermes.gateway-profile-b"],
            ),
            patch(
                "hermes_cli.gateway._locate_launchd_gateway_service",
                side_effect=_fake_locate,
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=(
                    "999\t0\tai.hermes.gateway-unmapped\n"
                    "789\t0\tcom.apple.some.other.agent\n"
                ),
                stderr="",
            )
            pids = gateway_mod._get_service_pids(all_profiles=True)

        # Label-derived fleet + prefix-scan stragglers; non-gateway excluded.
        assert pids == {123, 456, 999}
        assert 789 not in pids
        assert sorted(located) == [
            "ai.hermes.gateway",
            "ai.hermes.gateway-profile-b",
        ]
        launchctl_calls = [
            c[0][0]
            for c in mock_run.call_args_list
            if c[0] and c[0][0] and c[0][0][0] == "launchctl"
        ]
        assert launchctl_calls == [["launchctl", "list"]]

    @pytest.mark.platforms("linux")
    def test_all_profiles_preserves_systemd_behavior(self):
        """systemd scope is unaffected by the all_profiles switch — it already
        lists every hermes-gateway* unit unconditionally."""
        with (
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
