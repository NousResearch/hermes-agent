"""Tests for /proc-based gateway PID detection in Docker environments.

Verifies that _scan_gateway_pids() uses /proc/*/cmdline when available
(Docker without procps) and falls back to ps only when /proc is absent.

See: NousResearch/hermes-agent#7622
"""

import os
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# ps fallback branch (/proc absent) — pins the merged `-Aeww` spelling that
# macOS 26 requires (`ps -A eww` exits 1 there), plus the parser's pid/command
# extraction, exit-code handling, and BSD `ps aux` column fallback.
# ---------------------------------------------------------------------------

_PS_AEWW_OUTPUT = """\
  PID COMMAND
    1 /sbin/launchd
12345 python -m hermes_cli.main gateway run
23456 python -m hermes_cli.main gateway status
34567 python -m some_other_thing
45678 grep gateway
"""


def _ps_path_mocks(ps_returncode=0, ps_stdout=_PS_AEWW_OUTPUT):
    """Patches that force _scan_gateway_pids down the ps fallback branch:
    POSIX (not Windows), no /proc, and a canned ps subprocess result."""
    result = MagicMock()
    result.returncode = ps_returncode
    result.stdout = ps_stdout
    return result


class TestPsFallback:
    """_scan_gateway_pids uses `ps -Aeww` when /proc is absent."""

    def _run_scan(self, ps_result):
        with (
            patch("hermes_cli.gateway.is_windows", return_value=False),
            patch("os.path.isdir", return_value=False),  # no /proc
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run", return_value=ps_result) as mock_ps,
        ):
            pids = gateway_mod._scan_gateway_pids(set(), all_profiles=True)
        return pids, mock_ps

    def test_invokes_ps_with_merged_aeww_spelling(self):
        """Pin the exact argv: `ps -Aeww`, never the separated `ps -A eww`
        that macOS 26 rejects with exit 1 (the bug this fixes)."""
        _, mock_ps = self._run_scan(_ps_path_mocks())
        argv = mock_ps.call_args[0][0]
        assert argv[:2] == ["ps", "-Aeww"], (
            f"ps must use the merged -Aeww form (macOS 26 rejects '-A eww'); got {argv}"
        )

    def test_extracts_gateway_pid_from_ps_output(self):
        """A realistic `ps -Aeww -o pid=,command=` listing yields only the
        gateway `run` process — status/management, unrelated, and grep lines
        are all excluded."""
        pids, _ = self._run_scan(_ps_path_mocks())
        assert 12345 in pids
        assert 23456 not in pids  # `gateway status` is not a runtime process
        assert 34567 not in pids  # unrelated process
        assert 45678 not in pids  # grep line

    def test_nonzero_ps_exit_returns_empty(self):
        """macOS 26's `ps -A eww` failure mode: a non-zero exit means no
        scan results (and no crash), not a partial parse."""
        pids, _ = self._run_scan(_ps_path_mocks(ps_returncode=1, ps_stdout=""))
        assert pids == []

    def test_bsd_ps_aux_column_fallback(self):
        """When the `pid= command=` header parse misses, the parser falls back
        to BSD `ps aux` column positions (USER PID ... COMMAND)."""
        aux_output = (
            "USER              PID  %CPU %MEM      VSZ    RSS   TT  STAT STARTED      TIME COMMAND\n"
            "wmbt7052        12345   0.0  0.2 435534576 136768   ??  S     8:01AM   0:00.94 "
            "python -m hermes_cli.main gateway run\n"
            "wmbt7052        23456   0.0  0.1 435534576  55056   ??  S     8:01AM   0:00.12 "
            "python -m some_other_thing\n"
        )
        pids, _ = self._run_scan(_ps_path_mocks(ps_stdout=aux_output))
        assert 12345 in pids
        assert 23456 not in pids

