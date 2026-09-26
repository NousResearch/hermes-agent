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


# ---------------------------------------------------------------------------
# Shared-host user scoping (upstream #105719): the scan must not count other
# users' gateways. See ``_scan_gateway_pids`` / ``_get_service_pids``.
# ---------------------------------------------------------------------------

_FOREIGN_UID = 19999


def _stat_with_proc_owners(owners: dict[int, int]):
    """os.stat side_effect: faked ``/proc/<pid>`` dir owners, real stat otherwise."""

    def _stat(path, *args, **kwargs):
        path_str = str(path)
        marker = "/proc/"
        if path_str.startswith(marker) and path_str[len(marker):].isdigit():
            pid = int(path_str[len(marker):])
            if pid in owners:
                result = os.stat_result(
                    (0o40755, 123, 456, 1, owners[pid], owners[pid], 0, 0, 0, 0)
                )
                return result
        return _real_stat(path, *args, **kwargs)

    return _stat


_real_stat = os.stat


@pytest.mark.linux_only
class TestScanUserScoping:
    """_scan_gateway_pids only reports gateways owned by the invoking uid (#105719)."""

    def test_proc_scan_filters_other_users_gateway(self):
        my_uid = os.geteuid()
        my_pid = 12345
        foreign_pid = 22222
        _isdir, _listdir, _open = _fake_proc_dir({
            my_pid: _GATEWAY_CMD,
            foreign_pid: _GATEWAY_CMD,  # same argv, different user
        })

        with (
            patch("os.path.isdir", side_effect=_isdir),
            patch("os.listdir", side_effect=_listdir),
            patch("builtins.open", side_effect=_open),
            patch("os.stat", side_effect=_stat_with_proc_owners({
                my_pid: my_uid, foreign_pid: _FOREIGN_UID,
            })),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run") as mock_ps,
        ):
            pids = gateway_mod._scan_gateway_pids(set(), all_profiles=True)

        assert pids == [my_pid]
        mock_ps.assert_not_called()

    def test_ps_fallback_filters_other_users_gateway(self):
        my_uid = os.geteuid()
        rows = (
            f"  {12345}  {my_uid} {_GATEWAY_CMD}\n"
            f"  {22222}  {_FOREIGN_UID} {_GATEWAY_CMD}\n"
        )

        def _run_side_effect(args, **kwargs):
            assert args[:2] == ["ps", "-Aww"]
            return MagicMock(returncode=0, stdout=rows, stderr="")

        with (
            patch("os.path.isdir", return_value=False),
            patch("subprocess.run", side_effect=_run_side_effect),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
        ):
            pids = gateway_mod._scan_gateway_pids(set(), all_profiles=True)

        assert pids == [12345]

    def test_ps_fallback_first_call_uses_uid_column(self):
        """The invoking-uid scope needs the owner even where /proc does not exist (macOS/BSD)."""
        calls: list[list[str]] = []

        def _run_side_effect(args, **kwargs):
            calls.append(list(args))
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("os.path.isdir", return_value=False),
            patch("subprocess.run", side_effect=_run_side_effect),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
        ):
            assert gateway_mod._scan_gateway_pids(set()) == []

        assert calls[0][:4] == ["ps", "-Aww", "-o", "pid=,uid=,command="]

    def test_ps_fallback_legacy_format_keeps_legacy_include(self):
        """A ps without a uid column falls back to the legacy format; unstatable
        pids keep the legacy include (documented fail-open, see docstring)."""
        rows = f"  {12345} {_GATEWAY_CMD}\n"
        outputs = iter([
            MagicMock(returncode=1, stdout="", stderr="ps: bad format\n"),
            MagicMock(returncode=0, stdout=rows, stderr=""),
        ])

        def _stat_no_proc(path, *args, **kwargs):
            path_str = str(path)
            if path_str.startswith("/proc/") and path_str[len("/proc/"):].isdigit():
                raise FileNotFoundError(path_str)
            return _real_stat(path, *args, **kwargs)

        with (
            patch("os.path.isdir", return_value=False),
            patch("subprocess.run", side_effect=lambda args, **k: next(outputs)),
            patch("os.stat", side_effect=_stat_no_proc),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
        ):
            pids = gateway_mod._scan_gateway_pids(set(), all_profiles=True)

        assert pids == [12345]


class TestServicePidsUserScoping:
    """_get_service_pids default scope only adopts system-manager units owned by
    the invoking uid; the all_profiles protection sweep stays over-inclusive."""

    def test_system_scope_foreign_uid_unit_excluded(self):
        my_uid = os.geteuid()

        def _run_side_effect(args, **kwargs):
            args_list = list(args)
            if "--user" in args_list:
                return MagicMock(returncode=0, stdout="", stderr="")
            cmd_str = " ".join(str(a) for a in args_list[:4])
            if "list-units" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout="hermes-gateway.service loaded active running\n",
                    stderr="",
                )
            if "show" in cmd_str and "MainPID" in cmd_str:
                return MagicMock(returncode=0, stdout="456\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("hermes_cli.gateway.is_macos", return_value=False),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=True),
            patch("subprocess.run", side_effect=_run_side_effect),
            patch("os.stat", side_effect=_stat_with_proc_owners({456: _FOREIGN_UID})),
        ):
            pids = gateway_mod._get_service_pids(all_profiles=False)

        assert pids == set()

    def test_system_scope_own_uid_unit_kept(self):
        my_uid = os.geteuid()

        def _run_side_effect(args, **kwargs):
            args_list = list(args)
            if "--user" in args_list:
                return MagicMock(returncode=0, stdout="", stderr="")
            cmd_str = " ".join(str(a) for a in args_list[:4])
            if "list-units" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout="hermes-gateway.service loaded active running\n",
                    stderr="",
                )
            if "show" in cmd_str and "MainPID" in cmd_str:
                return MagicMock(returncode=0, stdout="456\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("hermes_cli.gateway.is_macos", return_value=False),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=True),
            patch("subprocess.run", side_effect=_run_side_effect),
            patch("os.stat", side_effect=_stat_with_proc_owners({456: my_uid})),
        ):
            pids = gateway_mod._get_service_pids(all_profiles=False)

        assert pids == {456}

    def test_user_scope_unit_needs_no_uid_proof(self):
        """``systemctl --user`` is per-user by construction; its MainPID is adopted
        even when /proc cannot confirm the owner (race, hidepid)."""

        def _run_side_effect(args, **kwargs):
            args_list = list(args)
            if "--user" not in args_list:
                return MagicMock(returncode=0, stdout="", stderr="")
            cmd_str = " ".join(str(a) for a in args_list)
            if "list-units" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout="hermes-gateway.service loaded active running\n",
                    stderr="",
                )
            if "show" in cmd_str and "MainPID" in cmd_str:
                return MagicMock(returncode=0, stdout="789\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        def _stat_never_finds(path, *args, **kwargs):
            path_str = str(path)
            if path_str.startswith("/proc/") and path_str[len("/proc/"):].isdigit():
                raise FileNotFoundError(path_str)
            return _real_stat(path, *args, **kwargs)

        with (
            patch("hermes_cli.gateway.is_macos", return_value=False),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=True),
            patch("subprocess.run", side_effect=_run_side_effect),
            patch("os.stat", side_effect=_stat_never_finds),
        ):
            pids = gateway_mod._get_service_pids(all_profiles=False)

        assert pids == {789}

    def test_all_profiles_fleet_sweep_stays_overinclusive(self):
        """all_profiles=True is a protection-only set (#41403/#73626): foreign-uid
        system units stay included so the kill sweep never reclassifies them."""

        def _run_side_effect(args, **kwargs):
            args_list = list(args)
            if "--user" in args_list:
                return MagicMock(returncode=0, stdout="", stderr="")
            cmd_str = " ".join(str(a) for a in args_list[:4])
            if "list-units" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout="hermes-gateway-alice.service loaded active running\n",
                    stderr="",
                )
            if "show" in cmd_str and "MainPID" in cmd_str:
                return MagicMock(returncode=0, stdout="456\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("hermes_cli.gateway.is_macos", return_value=False),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=True),
            patch("subprocess.run", side_effect=_run_side_effect),
            patch("os.stat", side_effect=_stat_with_proc_owners({456: _FOREIGN_UID})),
        ):
            pids = gateway_mod._get_service_pids(all_profiles=True)

        assert pids == {456}
