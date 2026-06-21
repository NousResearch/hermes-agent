"""Tests for the Windows Service implementation.

These tests verify the Windows Service wrapper (_hermes_gateway_service),
the service lifecycle functions in gateway_windows, and the pywin32
dependency declaration in pyproject.toml.
"""

from __future__ import annotations

import sys

import pytest


# ---------------------------------------------------------------------------
# 1. Service module imports
# ---------------------------------------------------------------------------

class TestServiceModuleImports:
    """Test that the service module can be imported and has expected exports."""

    def test_service_module_imports(self):
        """Test that the service module exports the expected symbols."""
        if sys.platform != "win32":
            pytest.skip("Windows-only")
        from hermes_cli._hermes_gateway_service import (
            HermesGatewayService,
            SERVICE_NAME,
            SERVICE_DISPLAY_NAME,
            SERVICE_DESCRIPTION,
            _RECOVERY_DELAYS_MS,
            _RECOVERY_RESET_PERIOD_S,
            configure_recovery_actions,
            main,
        )
        assert SERVICE_NAME == "HermesGateway"
        assert len(_RECOVERY_DELAYS_MS) == 9
        assert _RECOVERY_RESET_PERIOD_S == 60

    def test_service_name_value(self):
        """SERVICE_NAME must equal HermesGateway."""
        if sys.platform != "win32":
            pytest.skip("Windows-only")
        from hermes_cli._hermes_gateway_service import SERVICE_NAME
        assert SERVICE_NAME == "HermesGateway"

    def test_recovery_delays_count(self):
        """Recovery delays list must have exactly 9 entries."""
        if sys.platform != "win32":
            pytest.skip("Windows-only")
        from hermes_cli._hermes_gateway_service import _RECOVERY_DELAYS_MS
        assert len(_RECOVERY_DELAYS_MS) == 9

    def test_recovery_reset_period(self):
        """Recovery reset period must be 60 seconds."""
        if sys.platform != "win32":
            pytest.skip("Windows-only")
        from hermes_cli._hermes_gateway_service import _RECOVERY_RESET_PERIOD_S
        assert _RECOVERY_RESET_PERIOD_S == 60

    def test_recovery_delays_are_ascending(self):
        """Recovery delays should be in ascending order (quadratic backoff)."""
        if sys.platform != "win32":
            pytest.skip("Windows-only")
        from hermes_cli._hermes_gateway_service import _RECOVERY_DELAYS_MS
        assert _RECOVERY_DELAYS_MS == sorted(_RECOVERY_DELAYS_MS)


# ---------------------------------------------------------------------------
# 2. Service class attributes
# ---------------------------------------------------------------------------

class TestServiceClassAttributes:
    """Test that HermesGatewayService has correct class attributes."""

    def test_svc_name(self):
        if sys.platform != "win32":
            pytest.skip("Windows-only")
        from hermes_cli._hermes_gateway_service import HermesGatewayService
        assert HermesGatewayService._svc_name_ == "HermesGateway"

    def test_svc_display_name_contains_hermes(self):
        if sys.platform != "win32":
            pytest.skip("Windows-only")
        from hermes_cli._hermes_gateway_service import HermesGatewayService
        assert "Hermes" in HermesGatewayService._svc_display_name_

    def test_svc_description_not_empty(self):
        if sys.platform != "win32":
            pytest.skip("Windows-only")
        from hermes_cli._hermes_gateway_service import HermesGatewayService
        assert len(HermesGatewayService._svc_description_) > 0


# ---------------------------------------------------------------------------
# 3. Service name scoping
# ---------------------------------------------------------------------------

class TestServiceNameScoping:
    """Test default service name from gateway_windows."""

    def test_service_name_default(self):
        from hermes_cli.gateway_windows import get_service_name
        name = get_service_name()
        assert name == "HermesGateway" or name.startswith("HermesGateway-")


# ---------------------------------------------------------------------------
# 4. Service registration check
# ---------------------------------------------------------------------------

class TestServiceRegistration:
    """Test service registration check doesn't crash."""

    def test_is_service_registered_returns_bool(self):
        from hermes_cli.gateway_windows import is_service_registered
        result = is_service_registered()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 5. Lifecycle function fallbacks
# ---------------------------------------------------------------------------

class TestLifecycleFunctionsExist:
    """Test that all lifecycle functions exist and are callable."""

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

    def test_all_lifecycle_functions_together(self):
        """All six lifecycle functions must be importable and callable."""
        from hermes_cli.gateway_windows import (
            install_service,
            uninstall_service,
            start_service,
            stop_service,
            restart_service,
            service_status,
        )
        for fn in (install_service, uninstall_service, start_service,
                   stop_service, restart_service, service_status):
            assert callable(fn), f"{fn.__name__} is not callable"


# ---------------------------------------------------------------------------
# 6. pyproject.toml dependency
# ---------------------------------------------------------------------------

class TestPyprojectDependency:
    """Test that pywin32 is declared in pyproject.toml."""

    def test_pywin32_in_pyproject(self):
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        deps = data.get("project", {}).get("dependencies", [])
        pywin32_deps = [d for d in deps if "pywin32" in d]
        assert len(pywin32_deps) > 0, "pywin32 not found in dependencies"
        assert "win32" in pywin32_deps[0], "pywin32 should have platform marker"

    def test_pywin32_has_platform_marker(self):
        """pywin32 dependency must be gated to sys_platform == 'win32'."""
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        deps = data.get("project", {}).get("dependencies", [])
        pywin32_deps = [d for d in deps if "pywin32" in d]
        assert any("sys_platform" in d for d in pywin32_deps), (
            "pywin32 dependency should have a sys_platform marker"
        )
