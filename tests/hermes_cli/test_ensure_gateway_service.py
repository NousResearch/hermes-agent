"""Service setup failure reporting; consent integration lives in test_gateway_service_choice."""

import hermes_cli.gateway as gateway_mod
from hermes_cli.gateway_setup_service import ensure_gateway_service


def _patch_host(monkeypatch, *, container=False, systemd=True, macos=False, windows=False):
    monkeypatch.setattr("hermes_constants.is_container", lambda: container)
    monkeypatch.setattr(gateway_mod, "supports_systemd_services", lambda: systemd)
    monkeypatch.setattr(gateway_mod, "is_macos", lambda: macos)
    monkeypatch.setattr(gateway_mod, "is_windows", lambda: windows)


class TestEnsureGatewayService:
    def test_container_is_noop(self, monkeypatch, capsys):
        _patch_host(monkeypatch, container=True)
        called = []
        monkeypatch.setattr(gateway_mod, "systemd_install", lambda **kw: called.append("install"))

        assert ensure_gateway_service(install=True) is False
        assert not called
        out = capsys.readouterr().out
        assert "restart policy" in out

    def test_no_service_manager_is_noop(self, monkeypatch, capsys):
        _patch_host(monkeypatch, systemd=False)
        assert ensure_gateway_service(install=True) is False
        out = capsys.readouterr().out
        assert "hermes gateway" in out

    def test_already_running_short_circuits(self, monkeypatch):
        _patch_host(monkeypatch)
        monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: True)
        called = []
        monkeypatch.setattr(gateway_mod, "systemd_install", lambda **kw: called.append("install"))
        monkeypatch.setattr(gateway_mod, "systemd_start", lambda **kw: called.append("start"))

        assert ensure_gateway_service(install=True) is True
        assert not called



    def test_installed_but_stopped_starts_without_reinstall(self, monkeypatch):
        _patch_host(monkeypatch)
        monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)
        monkeypatch.setattr(gateway_mod, "_is_service_installed", lambda: True)
        calls = []
        monkeypatch.setattr(gateway_mod, "systemd_install", lambda **kw: calls.append("install"))
        monkeypatch.setattr(gateway_mod, "systemd_start", lambda **kw: calls.append("start"))

        assert ensure_gateway_service(install=True) is True
        assert calls == ["start"]

    def test_conflicting_units_block_install(self, monkeypatch):
        _patch_host(monkeypatch)
        monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)
        monkeypatch.setattr(gateway_mod, "_is_service_installed", lambda: False)
        monkeypatch.setattr(gateway_mod, "has_conflicting_systemd_units", lambda: True)
        warned = []
        monkeypatch.setattr(
            gateway_mod, "print_systemd_scope_conflict_warning", lambda: warned.append(True)
        )
        calls = []
        monkeypatch.setattr(gateway_mod, "systemd_install", lambda **kw: calls.append("install"))

        assert ensure_gateway_service(install=True) is False
        assert warned and not calls



    def test_never_raises_on_install_failure(self, monkeypatch, capsys):
        _patch_host(monkeypatch)
        monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)
        monkeypatch.setattr(gateway_mod, "_is_service_installed", lambda: False)
        monkeypatch.setattr(gateway_mod, "has_conflicting_systemd_units", lambda: False)

        def boom(**kw):
            raise RuntimeError("dbus fell over")

        monkeypatch.setattr(gateway_mod, "systemd_install", boom)

        assert ensure_gateway_service(install=True) is False
        out = capsys.readouterr().out
        assert "hermes gateway install" in out

    def test_never_raises_on_sys_exit(self, monkeypatch, capsys):
        """Install paths that sys.exit() must not abort setup/import."""
        _patch_host(monkeypatch)
        monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)
        monkeypatch.setattr(gateway_mod, "_is_service_installed", lambda: False)
        monkeypatch.setattr(gateway_mod, "has_conflicting_systemd_units", lambda: False)

        def bail(**kw):
            raise SystemExit(1)

        monkeypatch.setattr(gateway_mod, "systemd_install", bail)

        assert ensure_gateway_service(install=True) is False
        out = capsys.readouterr().out
        assert "hermes gateway install" in out

    def test_user_systemd_unreachable_reports_remediation(self, monkeypatch, capsys):
        _patch_host(monkeypatch)
        monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)
        monkeypatch.setattr(gateway_mod, "_is_service_installed", lambda: True)

        def unreachable(**kw):
            raise gateway_mod.UserSystemdUnavailableError("no D-Bus session\nenable linger")

        monkeypatch.setattr(gateway_mod, "systemd_start", unreachable)

        assert ensure_gateway_service(install=True) is False
        out = capsys.readouterr().out
        assert "enable linger" in out
