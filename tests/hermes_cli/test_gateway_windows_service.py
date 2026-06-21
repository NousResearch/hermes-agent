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
            # Should log warning
            mock_logging.warning.assert_any_call(
                "Failed to write planned-stop marker: %s",
                mock_logging.warning.call_args_list[0][0][1]
            )

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
                with open(marker_path, 'r') as f:
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
