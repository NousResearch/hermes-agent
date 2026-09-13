"""Regression tests for #104076: the --no-overlay probe silently dropped the
flag when handed a bare ``cua-driver`` name under a thin PATH.

The probe (``_cua_driver_supports_no_overlay``) used to spawn the literal
command name it was given. Both real entry points pass a resolved path on
current main, but the helper's own default is the bare ``cua-driver``, and
when the probing process's PATH lacks the install dir the subprocess raises
FileNotFoundError — which the bare ``except`` swallowed, returning False
with no log line. On a headless X11 worker the dropped flag means the
full-screen overlay keeps painting over the real screen, and once its X11
channel breaks every capture returns the same stale frame forever.

Contract under test:
- a bare name is RESOLVED before probing (the flag survives thin PATHs);
- a probe that cannot find the binary returns False AND logs a warning
  (``FileNotFoundError`` is diagnosable, not silent);
- any other probe failure also logs instead of vanishing;
- a present binary that lacks the flag stays a silent False (old driver,
  not an error).
"""

import logging
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use import cua_backend, cua_backend_driver


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """The probe is lru_cache'd per command; tests must not see each other's."""
    cua_backend_driver._cua_driver_supports_no_overlay.cache_clear()
    yield
    cua_backend_driver._cua_driver_supports_no_overlay.cache_clear()


class TestBareNameResolution:
    def test_bare_name_probes_resolved_path(self):
        """The literal 'cua-driver' name must be resolved before probing, so
        the flag survives a probing process whose PATH lacks the install dir
        (#104076's exact failure)."""
        with patch.object(
            cua_backend_driver, "resolve_cua_driver_cmd",
            return_value="/home/user/.local/bin/cua-driver",
        ) as resolve, patch.object(
            cua_backend, "_run_driver",
            return_value=CompletedProcess(args=[], returncode=0, stdout="--no-overlay", stderr=""),
        ) as run:
            assert cua_backend_driver._cua_driver_supports_no_overlay("cua-driver") is True
        resolve.assert_called_once()
        # The probe must have run the RESOLVED path, not the bare name.
        assert run.call_args.args[0] == "/home/user/.local/bin/cua-driver"

    def test_bare_name_unresolvable_falls_back_to_literal(self):
        """No path separator and nothing resolvable: probe the literal name
        anyway (old behaviour) — a real binary on a thin PATH may still spawn."""
        with patch.object(cua_backend_driver, "resolve_cua_driver_cmd", return_value=None), \
             patch.object(
                 cua_backend, "_run_driver",
                 return_value=CompletedProcess(args=[], returncode=0, stdout="--no-overlay", stderr=""),
             ) as run:
            assert cua_backend_driver._cua_driver_supports_no_overlay("cua-driver") is True
        assert run.call_args.args[0] == "cua-driver"

    def test_path_bearing_command_skips_resolution(self):
        """An explicit path is probed as-is; resolve_cua_driver_cmd is never
        consulted (the caller already resolved it)."""
        with patch.object(
            cua_backend_driver, "resolve_cua_driver_cmd",
        ) as resolve, patch.object(
            cua_backend, "_run_driver",
            return_value=CompletedProcess(args=[], returncode=0, stdout="--no-overlay", stderr=""),
        ):
            assert cua_backend_driver._cua_driver_supports_no_overlay(
                "/opt/cua/cua-driver") is True
        resolve.assert_not_called()


class TestProbeFailureLogging:
    def test_file_not_found_logs_warning(self, caplog):
        """Binary not found is the #104076 signature: must return False WITH a
        warning naming the command, not vanish silently."""
        with patch.object(cua_backend_driver, "resolve_cua_driver_cmd", return_value=None), \
             patch.object(cua_backend, "_run_driver", side_effect=FileNotFoundError("cua-driver")):
            with caplog.at_level(logging.WARNING, logger="tools.computer_use.cua_backend"):
                verdict = cua_backend_driver._cua_driver_supports_no_overlay("cua-driver")
        assert verdict is False
        assert "not found while probing" in caplog.text
        assert "overlay policy will NOT be applied" in caplog.text

    def test_other_probe_failure_logs_warning(self, caplog):
        """Timeouts and other spawn failures also log — every silent False is
        a debugging dead end."""
        with patch.object(
            cua_backend, "_run_driver",
            side_effect=TimeoutError("probe timed out"),
        ):
            with caplog.at_level(logging.WARNING, logger="tools.computer_use.cua_backend"):
                verdict = cua_backend_driver._cua_driver_supports_no_overlay(
                    "/opt/cua/cua-driver")
        assert verdict is False
        assert "probe failed" in caplog.text

    def test_old_driver_without_flag_stays_silent(self, caplog):
        """A present binary whose --help lacks the flag is a supported-old
        verdict, not an error: silent False, no warning."""
        with patch.object(
            cua_backend, "_run_driver",
            return_value=CompletedProcess(args=[], returncode=0, stdout="usage: cua-driver", stderr=""),
        ):
            with caplog.at_level(logging.WARNING, logger="tools.computer_use.cua_backend"):
                verdict = cua_backend_driver._cua_driver_supports_no_overlay(
                    "/opt/cua/cua-driver")
        assert verdict is False
        assert caplog.text == ""


class TestFlagSurvivesEndToEnd:
    def test_default_bare_cmd_arg_still_gets_flag(self):
        """_mcp_args_with_overlay_flag's DEFAULT is the bare 'cua-driver'
        (bound at import). With the policy on and a resolvable install, the
        flag must now be appended where it previously fell off."""
        with patch.object(cua_backend, "_cua_no_overlay", return_value=True), \
             patch.object(
                 cua_backend_driver, "resolve_cua_driver_cmd",
                 return_value="/home/user/.local/bin/cua-driver",
             ), \
             patch.object(
                 cua_backend, "_run_driver",
                 return_value=CompletedProcess(args=[], returncode=0, stdout="--no-overlay", stderr=""),
             ):
            args = cua_backend_driver._mcp_args_with_overlay_flag(["mcp"])
        assert args == ["mcp", "--no-overlay"]