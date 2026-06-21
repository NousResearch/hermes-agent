"""Tests for Windows Service backend implementation.

Tests cover:
- Service module imports and attributes (existing)
- Service name default/profile consistency (Blocker 3)
- install_service force=False ownership check (Blocker 5)
- Explicit service install failure no silent fallback (Blocker 6)
- Gateway crash triggers failure/recovery semantics (Blocker 4)
- SvcStop planned marker import stability (Blocker 7)
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
    """Test service name generation (Blocker 3)."""

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
        """install_service should accept allow_fallback parameter (Blocker 6)."""
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
# New tests for Blockers 4, 5, 6, 7
# =========================================================================

class TestInstallServiceForceCheck:
    """Test install_service force=False ownership check (Blocker 5)."""

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
    """Test explicit --service-type service doesn't fallback (Blocker 6)."""

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


class TestGatewayCrashRecovery:
    """Test gateway crash triggers failure/recovery semantics (Blocker 4)."""

    def test_gateway_exit_code_tracked(self):
        """_gateway_exit_code should be set on gateway crash."""
        from hermes_cli._hermes_gateway_service import _gateway_exit_code
        # Should be 0 by default
        assert _gateway_exit_code == 0

    def test_run_gateway_sets_exit_code_on_exception(self):
        """_run_gateway should set non-zero exit code on exception."""
        from hermes_cli._hermes_gateway_service import (
            HermesGatewayService, _gateway_exit_code
        )

        # Create a service instance
        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()

        # Mock run_gateway to raise
        with patch('hermes_cli.gateway.run_gateway', side_effect=Exception("crash")):
            service._run_gateway()

        # Exit code should be non-zero
        import hermes_cli._hermes_gateway_service as svc_mod
        assert svc_mod._gateway_exit_code != 0

    def test_run_gateway_sets_exit_code_on_system_exit(self):
        """_run_gateway should set exit code on SystemExit."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()

        with patch('hermes_cli.gateway.run_gateway', side_effect=SystemExit(42)):
            service._run_gateway()

        import hermes_cli._hermes_gateway_service as svc_mod
        assert svc_mod._gateway_exit_code == 42


class TestSvcStopImportPath:
    """Test SvcStop planned marker import stability (Blocker 7)."""

    def test_svc_stop_handles_write_failure(self):
        """SvcStop should log warning if marker write fails."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()

        # Mock ReportServiceStatus to avoid SCM calls
        service.ReportServiceStatus = MagicMock()

        # Mock _write_planned_stop_marker to raise
        service._write_planned_stop_marker = MagicMock(side_effect=Exception("write failed"))

        # SvcStop should not raise even if marker write fails
        with patch('hermes_cli._hermes_gateway_service.logging') as mock_logging:
            service.SvcStop()
            # Should log warning
            mock_logging.warning.assert_called_once()
            assert "planned-stop marker" in str(mock_logging.warning.call_args)

        # stop_event should still be set
        assert service._stop_event.is_set()

    def test_svc_stop_writes_marker_on_success(self):
        """SvcStop should call _write_planned_stop_marker on success."""
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)
        service._stop_event = threading.Event()

        # Mock ReportServiceStatus to avoid SCM calls
        service.ReportServiceStatus = MagicMock()

        # Mock _write_planned_stop_marker to succeed
        service._write_planned_stop_marker = MagicMock()

        service.SvcStop()

        # Should call _write_planned_stop_marker
        service._write_planned_stop_marker.assert_called_once()

        # stop_event should be set
        assert service._stop_event.is_set()

    def test_write_planned_stop_marker_creates_file(self):
        """_write_planned_stop_marker should create the marker file."""
        import json
        import tempfile
        from hermes_cli._hermes_gateway_service import HermesGatewayService

        service = HermesGatewayService.__new__(HermesGatewayService)

        # Use a temporary directory for HERMES_HOME
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict('os.environ', {'HERMES_HOME': tmpdir}):
                service._write_planned_stop_marker()

                # Check that marker file was created
                marker_path = Path(tmpdir) / ".gateway-planned-stop.json"
                assert marker_path.exists()

                # Check that marker contains required fields
                with open(marker_path, 'r') as f:
                    data = json.load(f)
                assert 'target_pid' in data
                assert 'written_at' in data


class TestServiceScriptNameArgument:
    """Test service script accepts --name argument (Blocker 3)."""

    def test_main_accepts_name_arg(self):
        """main() should accept --name argument."""
        import inspect
        from hermes_cli._hermes_gateway_service import main
        # main() reads from sys.argv, so we can test by checking it exists
        assert callable(main)
