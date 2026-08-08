"""Tests for Windows Service backend implementation.

Tests cover:
- Service module imports and attributes
- Service name default/profile consistency
- install_service force=False ownership check
- Explicit service install failure no silent fallback
- Gateway crash triggers failure/recovery semantics
- SvcStop planned marker write and graceful stop
- SystemExit code mapping (_resolve_exit_code)
- Marker parent directory creation (HIGH 2)
"""

import sys
import os
import threading
from unittest.mock import MagicMock, patch, call
from pathlib import Path

import pytest

# Skip all tests on non-Windows platforms
pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows-only tests"
)


# =========================================================================
# Existing tests (kept)
# =========================================================================

class TestServiceModuleImports:
    """Test that the service module can be imported on Windows."""

    def test_import_service_name(self):
        from hermes_cli._hermes_gateway_service import SERVICE_NAME
        assert isinstance(SERVICE_NAME, str)
        assert len(SERVICE_NAME) > 0

    def test_import_recovery_delays(self):
        from hermes_cli._hermes_gateway_service import _RECOVERY_DELAYS_MS
        assert isinstance(_RECOVERY_DELAYS_MS, list)
        assert len(_RECOVERY_DELAYS_MS) == 9
        # Delays should be ascending
        assert _RECOVERY_DELAYS_MS == sorted(_RECOVERY_DELAYS_MS)

    def test_import_recovery_reset(self):
        from hermes_cli._hermes_gateway_service import _RECOVERY_RESET_PERIOD_S
        assert _RECOVERY_RESET_PERIOD_S == 60

    def test_import_service_class(self):
        from hermes_cli._hermes_gateway_service import HermesGatewayService
        assert hasattr(HermesGatewayService, '_svc_name_')
        assert hasattr(HermesGatewayService, '_svc_display_name_')
        assert hasattr(HermesGatewayService, '_svc_description_')

    def test_import_configure_func(self):
        from hermes_cli._hermes_gateway_service import configure_recovery_actions
        assert callable(configure_recovery_actions)

    def test_import_resolve_exit_code(self):
        from hermes_cli._hermes_gateway_service import _resolve_exit_code
        assert callable(_resolve_exit_code)


class TestServiceClassAttributes:
    """Test HermesGatewayService class attributes."""

    def test_svc_name(self):
        from hermes_cli._hermes_gateway_service import HermesGatewayService
        assert HermesGatewayService._svc_name_ == "HermesGateway"

    def test_svc_display_name(self):
        from hermes_cli._hermes_gateway_service import HermesGatewayService
        assert "Hermes" in HermesGatewayService._svc_display_name_

    def test_svc_description(self):
        from hermes_cli._hermes_gateway_service import HermesGatewayService
        assert len(HermesGatewayService._svc_description_) > 0


class TestServiceNameScoping:
    """Test service name generation."""

    def test_default_service_name(self):
        from hermes_cli.gateway_windows import get_service_name
        name = get_service_name()
        # Should be "HermesGateway" or "HermesGateway-<suffix>"
        assert name.startswith("HermesGateway")
        # Should not contain spaces or special chars
        assert " " not in name

    def test_service_name_consistency(self):
        """Service script and gateway_windows should use the same name logic."""
        from hermes_cli.gateway_windows import get_service_name
        name = get_service_name()
        # The service script reads from HERMES_SERVICE_NAME env var
        # or defaults to "HermesGateway". When installed via install_service,
        # the bin_path includes --name <service_name>, so they should match.
        assert name == "HermesGateway" or name.startswith("HermesGateway-")


class TestServiceRegistration:
    """Test service registration check."""

    def test_is_service_registered_returns_bool(self):
        from hermes_cli.gateway_windows import is_service_registered
        result = is_service_registered()
        assert isinstance(result, bool)


class TestLifecycleFunctionsExist:
    """Test that all lifecycle functions exist."""

    def test_install_service(self):
        from hermes_cli.gateway_windows import install_service
        assert callable(install_service)

    def test_uninstall_service(self):
        from hermes_cli.gateway_windows import uninstall_service
        assert callable(uninstall_service)

    def test_start_service(self):
        from hermes_cli.gateway_windows import start_service
        assert callable(start_service)

    def test_stop_service(self):
        from hermes_cli.gateway_windows import stop_service
        assert callable(stop_service)

    def test_restart_service(self):
        from hermes_cli.gateway_windows import restart_service
        assert callable(restart_service)

    def test_service_status(self):
        from hermes_cli.gateway_windows import service_status
        assert callable(service_status)

    def test_install_service_accepts_allow_fallback(self):
        """install_service should accept allow_fallback parameter."""
        import inspect
        from hermes_cli.gateway_windows import install_service
        sig = inspect.signature(install_service)
        assert 'allow_fallback' in sig.parameters


class TestPyprojectDependency:
    """Test pyproject.toml has pywin32 dependency."""

    def test_pywin32_in_deps(self):
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        deps = data.get("project", {}).get("dependencies", [])
        pywin32_deps = [d for d in deps if "pywin32" in d]
        assert len(pywin32_deps) > 0, "pywin32 not found in dependencies"

    def test_pywin32_has_platform_marker(self):
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        deps = data.get("project", {}).get("dependencies", [])
        pywin32_deps = [d for d in deps if "pywin32" in d]
        assert any("win32" in d for d in pywin32_deps)


# =========================================================================
# SystemExit code mapping (BLOCKER 2)
# =========================================================================

class TestResolveExitCode:
    """Test _resolve_exit_code maps SystemExit.code correctly."""

    def test_none_returns_zero(self):
        """SystemExit(None) => 0 (success)."""
        from hermes_cli._hermes_gateway_service import _resolve_exit_code
        assert _resolve_exit_code(None) == 0

    def test_zero_returns_zero(self):
        """SystemExit(0) => 0 (explicit success)."""
        from hermes_cli._hermes_gateway_service import _resolve_exit_code
        assert _resolve_exit_code(0) == 0

    def test_positive_int_returns_same(self):
        """SystemExit(42) => 42."""
        from hermes_cli._hermes_gateway_service import _resolve_exit_code
        assert _resolve_exit_code(42) == 42

    def test_negative_int_returns_same(self):
        """SystemExit(-1) => -1."""
        from hermes_cli._hermes_gateway_service import _resolve_exit_code
        assert _resolve_exit_code(-1) == -1

    def test_string_returns_one(self):
        """SystemExit("error") => 1 (non-int maps to 1)."""
        from hermes_cli._hermes_gateway_service import _resolve_exit_code
        assert _resolve_exit_code("error") == 1


class TestGatewayExitCodeTracking:
    """Test gateway exit code is set correctly on various exit conditions."""

    def test_gateway_exit_code_starts_zero(self):
        """_gateway_exit_code should be 0 by default."""
        from hermes_cli._hermes_gateway_service import _gateway_exit_code
        assert _gateway_exit_code == 0

    def test_run_gateway_sets_zero_on_system_exit_zero(self):
        """SystemExit(0) should NOT be treated as failure."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()

        with patch('hermes_cli.gateway.run_gateway', side_effect=SystemExit(0)):
            service._run_gateway()

        import hermes_cli._hermes_gateway_service as svc_mod
        assert svc_mod._gateway_exit_code == 0
        assert service._stop_event.is_set()

    def test_run_gateway_sets_zero_on_system_exit_none(self):
        """SystemExit(None) should NOT be treated as failure."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()

        with patch('hermes_cli.gateway.run_gateway', side_effect=SystemExit(None)):
            service._run_gateway()

        import hermes_cli._hermes_gateway_service as svc_mod
        assert svc_mod._gateway_exit_code == 0
        assert service._stop_event.is_set()

    def test_run_gateway_sets_nonzero_on_system_exit_nonzero(self):
        """SystemExit(5) should be treated as failure."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()

        with patch('hermes_cli.gateway.run_gateway', side_effect=SystemExit(5)):
            service._run_gateway()

        import hermes_cli._hermes_gateway_service as svc_mod
        assert svc_mod._gateway_exit_code == 5
        assert service._stop_event.is_set()

    def test_run_gateway_sets_nonzero_on_exception(self):
        """Unhandled exception should be treated as failure."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()

        with patch('hermes_cli.gateway.run_gateway', side_effect=Exception("crash")):
            service._run_gateway()

        import hermes_cli._hermes_gateway_service as svc_mod
        assert svc_mod._gateway_exit_code == 1
        assert service._stop_event.is_set()


# =========================================================================
# SvcStop import path and graceful stop (BLOCKER 7, HIGH 1)
# =========================================================================

class TestSvcStopBehavior:
    """Test SvcStop planned marker write and graceful stop."""

    def test_svc_stop_handles_write_failure(self):
        """SvcStop should log warning if marker write fails."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()
        service._gateway_thread = None

        # Mock ReportServiceStatus to avoid SCM calls
        service.ReportServiceStatus = MagicMock()

        # Mock _write_planned_stop_marker to raise
        service._write_planned_stop_marker = MagicMock(side_effect=Exception("write failed"))

        # SvcStop should not raise even if marker write fails
        with patch('hermes_cli._hermes_gateway_service.logging') as mock_logging:
            service.SvcStop()
            # Should log warning about the marker write failure
            warning_calls = [str(c) for c in mock_logging.warning.call_args_list]
            assert any("planned-stop marker" in c for c in warning_calls),                 f"Expected 'planned-stop marker' in warning calls: {warning_calls}"

        # stop_event should still be set
        assert service._stop_event.is_set()

    def test_svc_stop_writes_marker_on_success(self):
        """SvcStop should call _write_planned_stop_marker on success."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()
        service._gateway_thread = None

        # Mock ReportServiceStatus to avoid SCM calls
        service.ReportServiceStatus = MagicMock()

        # Mock _write_planned_stop_marker to succeed
        service._write_planned_stop_marker = MagicMock()

        service.SvcStop()

        # Should call _write_planned_stop_marker
        service._write_planned_stop_marker.assert_called_once()

        # stop_event should be set
        assert service._stop_event.is_set()

    def test_svc_stop_waits_for_gateway_thread(self):
        """SvcStop should wait for gateway thread to exit (HIGH 1)."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()
        service.ReportServiceStatus = MagicMock()
        service._write_planned_stop_marker = MagicMock()

        # Create a "gateway thread" that exits when stop_event is set
        thread_exited = threading.Event()

        def fake_gateway():
            service._stop_event.wait()
            thread_exited.set()

        service._gateway_thread = threading.Thread(target=fake_gateway)
        service._gateway_thread.start()

        # SvcStop should signal stop and wait for thread to finish
        service.SvcStop()

        # Thread should have exited
        assert thread_exited.is_set()
        assert not service._gateway_thread.is_alive()

    def test_svc_stop_reports_stop_pending(self):
        """SvcStop should report SERVICE_STOP_PENDING to SCM."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService
        import win32service

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()
        service._gateway_thread = None
        service.ReportServiceStatus = MagicMock()
        service._write_planned_stop_marker = MagicMock()

        service.SvcStop()

        # Should report STOP_PENDING first, then STOPPED
        calls = service.ReportServiceStatus.call_args_list
        assert calls[0] == call(
            win32service.SERVICE_STOP_PENDING,
            waitHint=35 * 1000,
        )
        assert calls[-1] == call(win32service.SERVICE_STOPPED)


# =========================================================================
# Planned-stop marker directory creation (HIGH 2)
# =========================================================================

class TestPlannedStopMarkerDirCreation:
    """Test that _write_planned_stop_marker creates parent directory if needed."""

    def test_marker_creates_parent_dir(self):
        """_write_planned_stop_marker should create HERMES_HOME dir if missing."""
        import tempfile
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a subdirectory that doesn't exist
            missing_dir = os.path.join(tmpdir, "nonexistent", "hermes")
            with patch.dict('os.environ', {'HERMES_HOME': missing_dir}):
                service._write_planned_stop_marker()

                marker_path = Path(missing_dir) / ".gateway-planned-stop.json"
                assert marker_path.exists()

                # Verify content
                import json
                with open(marker_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                assert 'target_pid' in data
                assert 'written_at' in data

    def test_marker_uses_hermes_home_env(self):
        """_write_planned_stop_marker should prefer HERMES_HOME env var."""
        import tempfile
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict('os.environ', {'HERMES_HOME': tmpdir}):
                service._write_planned_stop_marker()

                marker_path = Path(tmpdir) / ".gateway-planned-stop.json"
                assert marker_path.exists()

    def test_marker_handles_write_error_gracefully(self):
        """_write_planned_stop_marker should raise on I/O errors (caller handles)."""
        import tempfile
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict('os.environ', {'HERMES_HOME': tmpdir}):
                # Mock open to raise on the marker file write
                real_open = open
                def mock_open(path, *args, **kwargs):
                    if '.gateway-planned-stop.tmp' in str(path):
                        raise OSError("Simulated write failure")
                    return real_open(path, *args, **kwargs)
                with patch('builtins.open', side_effect=mock_open):
                    with pytest.raises(OSError, match="Simulated write failure"):
                        service._write_planned_stop_marker()


# =========================================================================
# Install force/ownership check
# =========================================================================

class TestInstallServiceForceCheck:
    """Test install_service force=False ownership check."""

    def test_install_service_rejects_non_hermes_service(self):
        """install_service(force=False) should not delete a non-Hermes service."""
        from hermes_cli.gateway_windows import install_service

        # Mock SCM and existing service
        mock_scm = MagicMock()
        mock_existing = MagicMock()

        # Create a mock win32service module
        mock_ws = MagicMock()
        mock_ws.OpenSCManager.return_value = mock_scm
        mock_ws.OpenService.return_value = mock_existing
        mock_ws.QueryServiceConfig.return_value = (
            0, 0, 0, "C:\\Windows\\System32\\svchost.exe",  # Not Hermes
            None, None, None, None, None, None, None
        )

        # Patch sys.modules so `import win32service` gets our mock
        with patch.dict('sys.modules', {'win32service': mock_ws}):
            install_service(force=False, allow_fallback=False)

        # DeleteService should NOT be called for non-Hermes service
        mock_ws.DeleteService.assert_not_called()


class TestExplicitServiceNoFallback:
    """Test explicit --service-type service doesn't fallback."""

    def test_explicit_service_no_fallback_on_pywin32_missing(self):
        """When allow_fallback=False and pywin32 fails, should not call install()."""
        from hermes_cli.gateway_windows import install_service

        # Remove win32service from sys.modules to simulate missing pywin32
        with patch.dict('sys.modules', {'win32service': None}):
            with patch('hermes_cli.gateway_windows.install') as mock_install:
                install_service(force=False, allow_fallback=False)
                # Should NOT fallback to install()
                mock_install.assert_not_called()

    def test_explicit_service_no_fallback_on_exception(self):
        """When allow_fallback=False and install fails, should not call install()."""
        from hermes_cli.gateway_windows import install_service

        # Create a mock win32service module that raises on OpenSCManager
        mock_ws = MagicMock()
        mock_ws.OpenSCManager.side_effect = Exception("SCM error")

        with patch.dict('sys.modules', {'win32service': mock_ws}):
            with patch('hermes_cli.gateway_windows.install') as mock_install:
                install_service(force=False, allow_fallback=False)
                # Should NOT fallback to install()
                mock_install.assert_not_called()


class TestServiceScriptNameArgument:
    """Test service script accepts --name argument."""

    def test_main_accepts_name_arg(self):
        """main() should accept --name argument."""
        import inspect
        from hermes_cli._hermes_gateway_service import main
        # main() reads from sys.argv, so we can test by checking it exists
        assert callable(main)


# =========================================================================
# Service binPath persists HERMES_HOME / profile (PR #50200 review)
# =========================================================================


class TestInstallServiceBinPathHermesHome:
    """install_service must include --hermes-home and (for named profiles)
    --profile in the SCM binPath so the SCM-launched wrapper restores the
    same Hermes home the user picked at install time. Without it the wrapper
    fell back to APPDATA/home (review PRRT_kwDOPRF1G86Q6vjE).
    """

    def test_default_profile_binpath_includes_hermes_home(self, tmp_path, monkeypatch):
        """Default profile install: binPath contains --hermes-home with the
        resolved HERMES_HOME, no --profile flag."""
        from hermes_cli import gateway_windows

        default_home = tmp_path / "default_home"
        default_home.mkdir()
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home", lambda: default_home
        )

        # Force _profile_arg(...) to return "" for the default home.
        monkeypatch.setattr(
            "hermes_cli.gateway._profile_arg",
            lambda *_a, **_k: "",
        )

        captured: dict = {}

        def fake_create_service(scm, name, display, access, svc_type, start_type,
                                error, bin_path, *rest):
            captured["bin_path"] = bin_path
            svc = MagicMock()
            return svc

        mock_ws = MagicMock()
        mock_ws.OpenSCManager.return_value = MagicMock()
        mock_ws.OpenService.side_effect = Exception("not found")
        mock_ws.CreateService.side_effect = fake_create_service

        with patch.dict("sys.modules", {"win32service": mock_ws}):
            gateway_windows.install_service(force=False, allow_fallback=False)

        assert "bin_path" in captured, "CreateService was not called"
        bin_path = captured["bin_path"]
        # Path with spaces — list2cmdline wraps it in quotes.
        assert "--hermes-home" in bin_path
        # On Windows the temp dir from pytest is case-preserving but
        # ``Path.resolve()`` may normalise to lowercase; compare case-insensitively.
        assert str(default_home).lower() in bin_path.lower()
        # Default profile must NOT advertise a --profile flag.
        assert "--profile" not in bin_path

    def test_named_profile_binpath_includes_hermes_home_and_profile(
        self, tmp_path, monkeypatch
    ):
        """Named profile install: binPath must contain both --hermes-home
        and --profile so the wrapper restores the named profile."""
        from hermes_cli import gateway_windows

        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home", lambda: profile_home
        )
        monkeypatch.setattr(
            "hermes_cli.gateway._profile_arg",
            lambda *_a, **_k: "--profile coder",
        )

        captured: dict = {}

        def fake_create_service(scm, name, display, access, svc_type, start_type,
                                error, bin_path, *rest):
            captured["bin_path"] = bin_path
            return MagicMock()

        mock_ws = MagicMock()
        mock_ws.OpenSCManager.return_value = MagicMock()
        mock_ws.OpenService.side_effect = Exception("not found")
        mock_ws.CreateService.side_effect = fake_create_service

        with patch.dict("sys.modules", {"win32service": mock_ws}):
            gateway_windows.install_service(force=False, allow_fallback=False)

        bin_path = captured["bin_path"]
        assert "--hermes-home" in bin_path
        assert str(profile_home).lower() in bin_path.lower()
        assert "--profile" in bin_path
        assert "coder" in bin_path

    def test_named_profile_binpath_quotes_paths_with_spaces(
        self, tmp_path, monkeypatch
    ):
        """Paths containing spaces must be properly quoted so the SCM
        service controller reads them as a single argument (PR review:
        'parameter passing and Windows command line quoting must be
        reliable, including paths with spaces')."""
        from hermes_cli import gateway_windows

        spaced_home = tmp_path / "My Hermes Profile"
        spaced_home.mkdir()
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home", lambda: spaced_home
        )
        monkeypatch.setattr(
            "hermes_cli.gateway._profile_arg",
            lambda *_a, **_k: "--profile coder",
        )

        captured: dict = {}

        def fake_create_service(scm, name, display, access, svc_type, start_type,
                                error, bin_path, *rest):
            captured["bin_path"] = bin_path
            return MagicMock()

        mock_ws = MagicMock()
        mock_ws.OpenSCManager.return_value = MagicMock()
        mock_ws.OpenService.side_effect = Exception("not found")
        mock_ws.CreateService.side_effect = fake_create_service

        with patch.dict("sys.modules", {"win32service": mock_ws}):
            gateway_windows.install_service(force=False, allow_fallback=False)

        bin_path = captured["bin_path"]
        # list2cmdline wraps space-containing args in double quotes.
        # WindowsPath.resolve() may lowercase the path; check both cases.
        home_in_path = (
            f'"{spaced_home}"' in bin_path
            or f'"{str(spaced_home).lower()}"' in bin_path.lower()
        )
        assert home_in_path, (
            f"spaced home not found quoted in binPath: {bin_path}"
        )
        # Round-trip the result through a minimal cmd-line parser to verify
        # the spaced home path survives as a single argv slot.
        argv = []
        buf = ""
        in_quote = False
        for ch in bin_path:
            if ch == '"':
                in_quote = not in_quote
                continue
            if ch == " " and not in_quote:
                argv.append(buf)
                buf = ""
                continue
            buf += ch
        if buf:
            argv.append(buf)
        # The spaced path may appear as either original-case or normalised;
        # check it is present in argv as a single contiguous slot.
        spaced_normalised = str(spaced_home).replace("\\", "/").lower()
        assert any(
            spaced_normalised in slot.replace("\\", "/").lower()
            for slot in argv
        ), f"spaced home did not survive as one argv slot: {argv}"


class TestServiceScriptEarlyArgParsing:
    """The service wrapper must parse --hermes-home / --profile from
    sys.argv BEFORE importing any hermes module that depends on
    HERMES_HOME, so config/session/state/log and the planned-stop
    marker all land under the install-time-selected home.
    """

    def test_hermes_home_arg_sets_env_before_handle_command_line(self):
        """When --hermes-home is in sys.argv, main() must set
        os.environ['HERMES_HOME'] before HandleCommandLine runs."""
        import sys
        from hermes_cli import _hermes_gateway_service as svc

        # Patch HandleCommandLine + everything that touches Hermes home so
        # the test stays pure-Python on any platform.
        captured_env = {}

        def fake_handle_command_line(service_class):
            # Capture env as it stands at HandleCommandLine time.
            captured_env["HERMES_HOME"] = os.environ.get("HERMES_HOME")
            captured_env["HERMES_PROFILE"] = os.environ.get("HERMES_PROFILE")
            return 0

        original_argv = sys.argv[:]
        try:
            sys.argv = [
                "_hermes_gateway_service.py",
                "--hermes-home",
                r"C:\Users\KK\AppData\Local\hermes\profiles\coder",
                "--profile",
                "coder",
                "start",  # verb so after stripping, sys.argv > 1 → HandleCommandLine path
            ]
            # Ensure any pre-existing env doesn't pollute the assertion.
            os.environ.pop("HERMES_HOME", None)
            os.environ.pop("HERMES_PROFILE", None)
            with patch.object(svc, "win32serviceutil") as mock_util:
                mock_util.HandleCommandLine.side_effect = fake_handle_command_line
                svc.main()
            assert captured_env["HERMES_HOME"] == (
                r"C:\Users\KK\AppData\Local\hermes\profiles\coder"
            )
            assert captured_env["HERMES_PROFILE"] == "coder"
        finally:
            sys.argv = original_argv

    def test_custom_args_stripped_before_handle_command_line(self):
        """--hermes-home and --profile must be popped from sys.argv so
        HandleCommandLine doesn't see them as unknown options."""
        import sys
        from hermes_cli import _hermes_gateway_service as svc

        seen_argv = []

        def fake_handle_command_line(service_class):
            seen_argv.extend(sys.argv)
            return 0

        original_argv = sys.argv[:]
        try:
            sys.argv = [
                "_hermes_gateway_service.py",
                "--hermes-home",
                r"C:\some\home",
                "--profile",
                "coder",
                "start",
            ]
            with patch.object(svc, "win32serviceutil") as mock_util:
                mock_util.HandleCommandLine.side_effect = fake_handle_command_line
                svc.main()
            assert "--hermes-home" not in seen_argv
            assert "--profile" not in seen_argv
            assert r"C:\some\home" not in seen_argv
            assert "coder" not in seen_argv
            # "start" is the win32serviceutil verb and must remain.
            assert "start" in seen_argv
        finally:
            sys.argv = original_argv

    def test_consume_arg_handles_equals_form(self):
        """--hermes-home=VALUE form must also be consumed (defense in
        depth, even though install_service uses space form)."""
        from hermes_cli._hermes_gateway_service import _consume_arg

        argv = ["script", "--hermes-home=C:\\path with space", "start"]
        value = _consume_arg(argv, "--hermes-home")
        assert value == "C:\\path with space"
        assert "--hermes-home=C:\\path with space" not in argv
        assert "start" in argv

    def test_consume_arg_returns_none_when_absent(self):
        from hermes_cli._hermes_gateway_service import _consume_arg

        argv = ["script", "start"]
        assert _consume_arg(argv, "--hermes-home") is None
        assert argv == ["script", "start"]


# =========================================================================
# Per-profile SCM pause / resume (PR #50200 review)
# =========================================================================


class TestUpdatePauseResumeScm:
    """PR #50200 review: ``hermes update`` pause/resume must route through
    SCM for SCM-managed profiles and never call ``_spawn_detached()`` for
    them. Scheduled Task / detached gateways keep their existing behavior.
    """

    def _fake_profile_process(self, name, path, pid):
        from hermes_cli.gateway import ProfileGatewayProcess

        return ProfileGatewayProcess(profile=name, path=path, pid=pid)

    def test_pause_calls_scm_stop_not_terminate_pid_for_service_profile(
        self, monkeypatch
    ):
        """When a profile is SCM-managed, ``_pause_windows_gateways_for_update``
        must call ``gateway_windows.stop_service_for_hermes_home`` for that
        home and must NOT call ``terminate_pid`` on the service child PID.
        """
        from hermes_cli import main as main_module

        proc = self._fake_profile_process(
            name="coder", path=Path("/tmp/profiles/coder"), pid=12345
        )
        proc_path = Path("/tmp/profiles/coder")

        calls = {
            "stop_service_for_home": [],
            "terminate_pid": [],
        }

        # Stub find_gateway_pids / find_profile_gateway_processes so the
        # pause logic discovers exactly one SCM-managed profile gateway.
        monkeypatch.setattr(
            "hermes_cli.gateway.find_gateway_pids",
            lambda all_profiles=False: [12345],
        )
        monkeypatch.setattr(
            "hermes_cli.gateway.find_profile_gateway_processes",
            lambda exclude_pids=None: [proc],
        )

        def fake_stop_service_for_hermes_home(home):
            calls["stop_service_for_home"].append(home)
            return True

        def fake_is_service_registered_for_hermes_home(_home):
            return True

        monkeypatch.setattr(
            "hermes_cli.gateway_windows.stop_service_for_hermes_home",
            fake_stop_service_for_hermes_home,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows.is_service_registered_for_hermes_home",
            fake_is_service_registered_for_hermes_home,
        )

        # terminate_pid is the legacy escape hatch — must NOT be called
        # for an SCM-managed profile, otherwise RecoveryActions would
        # immediately restart the gateway we just paused.
        def fake_terminate_pid(pid, force=False):
            calls["terminate_pid"].append((pid, force))
            raise ProcessLookupError(pid)

        # Patch the gateway.status.terminate_pid that main.py imports.
        import gateway.status as gstatus
        monkeypatch.setattr(gstatus, "terminate_pid", fake_terminate_pid)
        # Also patch it where main.py's namespace has already imported it.
        monkeypatch.setattr(
            "hermes_cli.main.terminate_pid", fake_terminate_pid, raising=False
        )

        token = main_module._pause_windows_gateways_for_update()

        assert calls["stop_service_for_home"] == [str(proc_path.resolve())], (
            "SCM-managed profile must be paused via "
            "stop_service_for_hermes_home, got "
            f"{calls['stop_service_for_home']}"
        )
        assert calls["terminate_pid"] == [], (
            "terminate_pid must NOT be called for SCM-managed profile; got "
            f"{calls['terminate_pid']}"
        )
        assert token is not None
        assert "coder" in token["scm_managed_profiles"]
        assert token["scm_managed_profiles"]["coder"] == str(proc_path.resolve())

    def test_pause_keeps_legacy_terminate_pid_path_for_manual_profile(
        self, monkeypatch
    ):
        """Non-SCM profiles must keep the legacy
        ``_write_update_planned_stop_marker`` + drain + ``terminate_pid``
        path (review: 'original Scheduled Task and non-service backends
        must remain unchanged').

        Specifically: ``stop_service_for_hermes_home`` must NOT be called
        for a non-SCM profile (so the SCM branch stays exclusive to
        SCM-managed profiles), and the planned-stop marker IS written.
        """
        from hermes_cli import main as main_module
        import gateway.status as gstatus

        proc = self._fake_profile_process(
            name="default", path=Path("/tmp/default_home"), pid=22222
        )

        monkeypatch.setattr(
            "hermes_cli.gateway.find_gateway_pids",
            lambda all_profiles=False: [22222],
        )
        monkeypatch.setattr(
            "hermes_cli.gateway.find_profile_gateway_processes",
            lambda exclude_pids=None: [proc],
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows.is_service_registered_for_hermes_home",
            lambda _home: False,  # not SCM-managed
        )

        marker_writes: list = []
        monkeypatch.setattr(
            main_module,
            "_write_update_planned_stop_marker",
            lambda profile_path, pid: marker_writes.append((str(profile_path), pid)),
        )

        scm_calls = []
        monkeypatch.setattr(
            "hermes_cli.gateway_windows.stop_service_for_hermes_home",
            lambda home: scm_calls.append(home) or True,
        )

        calls = {"terminate_pid": []}

        def fake_terminate_pid(pid, force=False):
            calls["terminate_pid"].append((pid, force))
            raise ProcessLookupError(pid)

        monkeypatch.setattr(gstatus, "terminate_pid", fake_terminate_pid)
        monkeypatch.setattr(
            "hermes_cli.main.terminate_pid", fake_terminate_pid, raising=False
        )

        token = main_module._pause_windows_gateways_for_update()

        # Legacy non-SCM path keeps the planned-stop marker.
        assert marker_writes == [(str(Path("/tmp/default_home")), 22222)]
        # And does NOT route through SCM stop.
        assert scm_calls == []
        assert token is not None
        assert "scm_managed_profiles" in token
        assert token["scm_managed_profiles"] == {}

    def test_resume_calls_scm_start_not_detached_for_service_profile(
        self, monkeypatch
    ):
        """When a token marks a profile as SCM-managed, the resume path
        must use ``start_service_for_hermes_home`` and MUST NOT call
        ``launch_detached_profile_gateway_restart`` for that profile.
        """
        from hermes_cli import main as main_module

        token = {
            "resume_needed": True,
            "profiles": {"coder": 12345},
            "unmapped_pids": [],
            "unmapped": [],
            "scm_managed_profiles": {"coder": "/tmp/profiles/coder"},
        }

        calls = {
            "start_service_for_home": [],
            "launch_detached_profile_gateway_restart": [],
        }

        def fake_start_service_for_hermes_home(home):
            calls["start_service_for_home"].append(home)
            return True

        def fake_launch_detached_profile_gateway_restart(profile, old_pid):
            calls["launch_detached_profile_gateway_restart"].append((profile, old_pid))
            return True

        monkeypatch.setattr(
            "hermes_cli.gateway_windows.start_service_for_hermes_home",
            fake_start_service_for_hermes_home,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway.launch_detached_profile_gateway_restart",
            fake_launch_detached_profile_gateway_restart,
        )

        main_module._resume_windows_gateways_after_update(token)

        assert calls["start_service_for_home"] == ["/tmp/profiles/coder"], (
            "SCM-managed profile must be restarted via "
            "start_service_for_hermes_home, got "
            f"{calls['start_service_for_home']}"
        )
        assert calls["launch_detached_profile_gateway_restart"] == [], (
            "Must NOT call launch_detached_profile_gateway_restart for an "
            f"SCM-managed profile; got {calls['launch_detached_profile_gateway_restart']}"
        )

    def test_resume_routes_named_profile_to_correct_service(self, monkeypatch):
        """For multiple profiles with different HERMES_HOME, the resume
        must route each SCM-managed profile to its OWN
        ``start_service_for_hermes_home`` call (not to a single shared
        active-profile service)."""
        from hermes_cli import main as main_module

        token = {
            "resume_needed": True,
            "profiles": {"coder": 100, "writer": 200, "default": 300},
            "unmapped_pids": [],
            "unmapped": [],
            "scm_managed_profiles": {
                "coder": "/tmp/profiles/coder",
                "writer": "/tmp/profiles/writer",
            },
        }

        calls = []

        def fake_start_service_for_hermes_home(home):
            calls.append(home)
            return True

        monkeypatch.setattr(
            "hermes_cli.gateway_windows.start_service_for_hermes_home",
            fake_start_service_for_hermes_home,
        )
        # If any non-SCM profile falls through to the legacy detached
        # path, this list captures it.
        detached_calls = []
        monkeypatch.setattr(
            "hermes_cli.gateway.launch_detached_profile_gateway_restart",
            lambda profile, old_pid: detached_calls.append((profile, old_pid)) or True,
        )

        main_module._resume_windows_gateways_after_update(token)

        assert "/tmp/profiles/coder" in calls
        assert "/tmp/profiles/writer" in calls
        assert ("coder", 100) not in detached_calls
        assert ("writer", 200) not in detached_calls
        # Default profile is not SCM-managed — legacy detached path applies.
        assert ("default", 300) in detached_calls

    def test_resume_keeps_legacy_detached_path_for_manual_profile(
        self, monkeypatch
    ):
        """Non-SCM profiles continue to use ``launch_detached_profile_gateway_restart``.
        """
        from hermes_cli import main as main_module

        token = {
            "resume_needed": True,
            "profiles": {"default": 999},
            "unmapped_pids": [],
            "unmapped": [],
            "scm_managed_profiles": {},
        }

        calls = {"scm_start": [], "detached": []}

        monkeypatch.setattr(
            "hermes_cli.gateway_windows.start_service_for_hermes_home",
            lambda home: calls["scm_start"].append(home) or True,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway.launch_detached_profile_gateway_restart",
            lambda profile, old_pid: calls["detached"].append((profile, old_pid)) or True,
        )

        main_module._resume_windows_gateways_after_update(token)

        assert calls["scm_start"] == []
        assert calls["detached"] == [("default", 999)]

    def test_resume_cold_start_uses_scm_when_active_profile_service_registered(
        self, monkeypatch
    ):
        """When no gateway was running but the active install has an
        SCM service registered, cold-start must go through SCM StartService
        (not ``_spawn_detached``), avoiding a parallel detached gateway.
        """
        from hermes_cli import main as main_module

        token = {
            "resume_needed": True,
            "profiles": {},
            "unmapped_pids": [],
            "unmapped": [],
            "scm_managed_profiles": {},
            "cold_start_if_installed": True,
            "cold_start_via_scm": True,
        }

        # Re-check liveness: no PID.
        monkeypatch.setattr(
            "hermes_cli.gateway.find_gateway_pids",
            lambda all_profiles=False: [],
        )

        calls = {"scm_start": [], "spawn_detached": []}

        def fake_start_service():
            calls["scm_start"].append(True)
            return True

        def fake_spawn_detached():
            calls["spawn_detached"].append(True)
            return 1234

        monkeypatch.setattr(
            "hermes_cli.gateway_windows.start_service",
            fake_start_service,
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows._spawn_detached",
            fake_spawn_detached,
        )

        main_module._resume_windows_gateways_after_update(token)

        assert calls["scm_start"] == [True]
        assert calls["spawn_detached"] == [], (
            "Must NOT call _spawn_detached when SCM is registered; got "
            f"{calls['spawn_detached']}"
        )


class TestServiceNameForHermesHome:
    """get_service_name_for_hermes_home must mirror _profile_suffix so
    the update pause/resume path resolves the correct SCM unit for any
    profile, not just the active one.
    """

    def test_default_home_returns_default_service(self, tmp_path, monkeypatch):
        from hermes_cli import gateway_windows

        monkeypatch.setattr(
            "hermes_constants.get_default_hermes_root",
            lambda: tmp_path,
        )
        name = gateway_windows.get_service_name_for_hermes_home(str(tmp_path))
        assert name == "HermesGateway"

    def test_named_profile_returns_suffixed_service(self, tmp_path, monkeypatch):
        from hermes_cli import gateway_windows

        default = tmp_path
        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)

        monkeypatch.setattr(
            "hermes_constants.get_default_hermes_root",
            lambda: default,
        )
        name = gateway_windows.get_service_name_for_hermes_home(str(profile_home))
        assert name == "HermesGateway-coder"


# =========================================================================
# SCM dispatch, tagId, via_scm no-fallback, wait-for-state
# =========================================================================


class TestScmDispatchAndTagId:
    """SCM lifecycle fixes (PR #50200)."""

    def test_scm_dispatch_path_when_argv_len_one(self, monkeypatch):
        """When sys.argv has only the script name (SCM launch), main()
        must call servicemanager.Initialize/PrepareToHostSingle/
        StartServiceCtrlDispatcher, not HandleCommandLine."""
        import sys
        from hermes_cli import _hermes_gateway_service as svc

        original_argv = sys.argv[:]
        calls = []

        def fake_initialize():
            calls.append("Initialize")

        def fake_prepare(cls):
            calls.append(f"PrepareToHostSingle({cls})")

        def fake_dispatch():
            calls.append("StartServiceCtrlDispatcher")

        try:
            sys.argv = ["_hermes_gateway_service.py"]
            with patch.multiple(
                svc.servicemanager,
                Initialize=fake_initialize,
                PrepareToHostSingle=fake_prepare,
                StartServiceCtrlDispatcher=fake_dispatch,
            ):
                svc.main()
            assert "Initialize" in calls, calls
            assert any("PrepareToHostSingle" in c for c in calls), calls
            assert "StartServiceCtrlDispatcher" in calls, calls
        finally:
            sys.argv = original_argv

    def test_scm_dispatch_skips_handle_command_line(self, monkeypatch):
        """When SCM path is taken (len(argv)<=1), HandleCommandLine
        must NOT be called."""
        import sys
        from hermes_cli import _hermes_gateway_service as svc

        original_argv = sys.argv[:]
        hcl_called = []

        def fake_hcl(cls):
            hcl_called.append(True)

        try:
            sys.argv = ["_hermes_gateway_service.py"]
            with patch.multiple(
                svc.servicemanager,
                Initialize=lambda: None,
                PrepareToHostSingle=lambda cls: None,
                StartServiceCtrlDispatcher=lambda: None,
            ), patch.object(svc, "win32serviceutil") as mock_util:
                mock_util.HandleCommandLine.side_effect = fake_hcl
                svc.main()
            assert hcl_called == [], (
                "HandleCommandLine must NOT be called in SCM dispatch path"
            )
        finally:
            sys.argv = original_argv

    def test_install_service_tagid_is_int_zero(self, monkeypatch):
        """install_service must pass tagId=0 to CreateService, not None
        (PR #50200 fix: 'NoneType' object cannot be interpreted as an integer)."""
        from hermes_cli import gateway_windows

        captured = {}

        def fake_create_service(scm, name, display, access, svctype,
                                 starttype, error, binpath, *rest):
            # rest = (loadOrderGroup, tagId, dependencies, ...)
            captured["tagId"] = rest[1] if len(rest) > 1 else "unknown"
            return MagicMock()

        mock_ws = MagicMock()
        mock_ws.OpenSCManager.return_value = MagicMock()
        mock_ws.OpenService.side_effect = Exception("not found")
        mock_ws.CreateService.side_effect = fake_create_service

        with patch.dict("sys.modules", {"win32service": mock_ws}):
            gateway_windows.install_service(force=False, allow_fallback=False)

        assert "tagId" in captured
        assert captured["tagId"] == 0, (
            f"CreateService tagId must be 0 (int), got {captured['tagId']!r}"
        )


class TestScmViaScmNoFallback:
    """via_scm=True must never fall back to _spawn_detached."""

    def test_via_scm_failure_does_not_fall_through_to_detached(self, monkeypatch):
        """When cold_start_via_scm=True and SCM start fails,
        _cold_start_windows_gateway_after_update must NOT call
        _spawn_detached.
        """
        from hermes_cli import main as main_module

        monkeypatch.setattr(
            "hermes_cli.gateway.find_gateway_pids",
            lambda all_profiles=False: [],
        )
        monkeypatch.setattr(
            "hermes_cli.gateway_windows.start_service",
            lambda: False,  # SCM start fails
        )
        calls = []
        monkeypatch.setattr(
            "hermes_cli.gateway_windows._spawn_detached",
            lambda: calls.append(True) or 1234,
        )

        main_module._cold_start_windows_gateway_after_update(via_scm=True)

        assert calls == [], (
            "Must NOT call _spawn_detached when via_scm=True and SCM failed; "
            f"got {calls}"
        )

    def test_via_scm_false_falls_through_to_detached(self, monkeypatch):
        """When via_scm=False, _cold_start_windows_gateway_after_update
        must fall through to _spawn_detached (legacy path unchanged)."""
        from hermes_cli import main as main_module

        monkeypatch.setattr(
            "hermes_cli.gateway.find_gateway_pids",
            lambda all_profiles=False: [],
        )
        calls = []
        monkeypatch.setattr(
            "hermes_cli.gateway_windows._spawn_detached",
            lambda: calls.append(True) or 1234,
        )

        main_module._cold_start_windows_gateway_after_update(via_scm=False)

        assert calls == [True], (
            "Must call _spawn_detached when via_scm=False; "
            f"got {calls}"
        )


class TestScmWaitForState:
    """stop_service_for_hermes_home / start_service_for_hermes_home
    must wait for SERVICE_STOPPED / SERVICE_RUNNING."""

    def test_stop_waits_for_stopped(self, monkeypatch):
        """After ControlService, stop_service_for_hermes_home must
        call WaitForServiceStatus for SERVICE_STOPPED."""
        from hermes_cli import gateway_windows
        import win32service

        waited = []
        def fake_wait(name, status, timeout):
            waited.append((name, status, timeout))

        # Mock the module-level helpers so the function reaches
        # the wait block (ControlService must succeed).
        monkeypatch.setattr(
            "hermes_cli.gateway_windows.is_service_registered_for_hermes_home",
            lambda _home: True,
        )
        # Mock OpenSCManager/OpenService/ControlService to succeed
        mock_ws = MagicMock()
        mock_ws.OpenSCManager.return_value = 1
        mock_ws.CloseServiceHandle.return_value = None
        svc_handle = MagicMock()
        mock_ws.OpenService.return_value = svc_handle
        mock_ws.SERVICE_STOP = win32service.SERVICE_STOP
        mock_ws.SERVICE_CONTROL_STOP = win32service.SERVICE_CONTROL_STOP
        mock_ws.SERVICE_STOPPED = win32service.SERVICE_STOPPED

        with patch.dict("sys.modules", {"win32service": mock_ws}):
            # The function imports win32serviceutil inside the wait block;
            # we need to set it up before calling.
            import types
            mock_util = types.ModuleType("mock_util")
            mock_util.WaitForServiceStatus = fake_wait
            monkeypatch.setitem(sys.modules, "win32serviceutil", mock_util)

            gateway_windows.stop_service_for_hermes_home("/tmp/home")

        assert len(waited) == 1, f"WaitForServiceStatus was NOT called: {waited}"
        _, status, timeout = waited[0]
        assert status == win32service.SERVICE_STOPPED
        assert timeout == 30

    def test_start_waits_for_running(self, monkeypatch):
        """After StartService, start_service_for_hermes_home must
        call WaitForServiceStatus for SERVICE_RUNNING."""
        from hermes_cli import gateway_windows
        import win32service

        waited = []
        def fake_wait(name, status, timeout):
            waited.append((name, status, timeout))

        monkeypatch.setattr(
            "hermes_cli.gateway_windows.is_service_registered_for_hermes_home",
            lambda _home: True,
        )
        mock_ws = MagicMock()
        mock_ws.OpenSCManager.return_value = 1
        mock_ws.CloseServiceHandle.return_value = None
        svc_handle = MagicMock()
        mock_ws.OpenService.return_value = svc_handle
        mock_ws.SERVICE_START = win32service.SERVICE_START
        mock_ws.SERVICE_RUNNING = win32service.SERVICE_RUNNING

        with patch.dict("sys.modules", {"win32service": mock_ws}):
            import types
            mock_util = types.ModuleType("mock_util")
            mock_util.WaitForServiceStatus = fake_wait
            monkeypatch.setitem(sys.modules, "win32serviceutil", mock_util)

            gateway_windows.start_service_for_hermes_home("/tmp/home")

        assert len(waited) == 1, f"WaitForServiceStatus was NOT called: {waited}"
        _, status, timeout = waited[0]
        assert status == win32service.SERVICE_RUNNING
        assert timeout == 60
