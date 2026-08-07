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


@pytest.fixture(autouse=True)
def _same_pid_namespace_by_default(monkeypatch, request):
    """Keep synthetic PIDs in scanner tests in the caller's PID namespace."""
    if request.cls is TestPidNamespacePredicate:
        return
    monkeypatch.setattr(
        gateway_mod,
        "_pid_in_current_namespace",
        lambda _pid: True,
        raising=False,
    )


class TestPidNamespacePredicate:
    def test_accepts_same_namespace(self):
        with patch(
            "hermes_cli.gateway.os.stat",
            side_effect=[MagicMock(st_ino=100), MagicMock(st_ino=100)],
        ):
            assert gateway_mod._pid_in_current_namespace(12345) is True

    def test_rejects_foreign_namespace(self):
        with patch(
            "hermes_cli.gateway.os.stat",
            side_effect=[MagicMock(st_ino=100), MagicMock(st_ino=200)],
        ):
            assert gateway_mod._pid_in_current_namespace(12345) is False

    def test_rejects_uninspectable_target(self):
        with patch(
            "hermes_cli.gateway.os.stat",
            side_effect=[MagicMock(st_ino=100), FileNotFoundError()],
        ):
            assert gateway_mod._pid_in_current_namespace(12345) is False

    def test_preserves_legacy_behavior_if_caller_namespace_is_unavailable(self):
        with patch("hermes_cli.gateway.os.stat", side_effect=OSError()):
            assert gateway_mod._pid_in_current_namespace(12345) is True


class TestProcFallback:
    """_scan_gateway_pids reads /proc when available, skips ps."""

    def test_detects_gateway_pid_via_proc(self):
        my_pid = os.getpid()
        entries = {
            my_pid: "python -m hermes_cli.main",   # own process — excluded
            12345: _GATEWAY_CMD,
            99999: _OTHER_CMD,
        }
        _isdir, _listdir, _open = _fake_proc_dir(entries)

        with (
            patch("hermes_cli.gateway.is_windows", return_value=False),
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

    def test_ignores_gateway_pid_from_foreign_pid_namespace(self, monkeypatch):
        """A host update must not treat an LXC gateway as a local gateway."""
        entries = {
            12345: _GATEWAY_CMD,
            23456: _GATEWAY_CMD,
        }
        _isdir, _listdir, _open = _fake_proc_dir(entries)
        monkeypatch.setattr(
            gateway_mod,
            "_pid_in_current_namespace",
            lambda pid: pid == 12345,
        )

        with (
            patch("hermes_cli.gateway.is_windows", return_value=False),
            patch("os.path.isdir", side_effect=_isdir),
            patch("os.listdir", side_effect=_listdir),
            patch("builtins.open", side_effect=_open),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run") as mock_ps,
        ):
            pids = gateway_mod._scan_gateway_pids(set(), all_profiles=True)

        assert pids == [12345]
        mock_ps.assert_not_called()

    def test_ps_fallback_ignores_gateway_pid_from_foreign_namespace(
        self, monkeypatch
    ):
        """The ps fallback enforces the same PID-namespace boundary as /proc."""
        monkeypatch.setattr(
            gateway_mod,
            "_pid_in_current_namespace",
            lambda pid: pid == 12345,
        )
        ps_output = "\n".join(
            [
                f"12345 {_GATEWAY_CMD}",
                f"23456 {_GATEWAY_CMD}",
            ]
        ) + "\n"

        with (
            patch("hermes_cli.gateway.is_windows", return_value=False),
            patch("os.path.isdir", return_value=False),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run") as mock_ps,
        ):
            mock_ps.return_value = MagicMock(returncode=0, stdout=ps_output)
            pids = gateway_mod._scan_gateway_pids(set(), all_profiles=True)

        assert pids == [12345]
        mock_ps.assert_called_once_with(
            ["ps", "-A", "eww", "-o", "pid=,command="],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )


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
            patch("hermes_cli.gateway.is_windows", return_value=False),
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
