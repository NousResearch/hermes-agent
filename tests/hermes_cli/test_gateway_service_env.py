from pathlib import Path

from hermes_cli import gateway as gateway_cli


def test_user_unit_carries_sealed_install_resource_roots(monkeypatch, tmp_path):
    """ExecStart is the store interpreter, not the packaging wrapper that exports these, and systemd
    starts the unit from a clean environment. Without them a Nix/Homebrew gateway finds no locale
    catalog and renders raw i18n keys (``gateway.reset.header_default``)."""
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path",
                        lambda system=False: tmp_path / "hermes-gateway.service")
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", "/nix/store/abc-hermes/share/hermes-agent/locales")
    monkeypatch.setenv("HERMES_INSTALL_ROOT", "/nix/store/abc-hermes/share/hermes-agent")
    monkeypatch.delenv("HERMES_OPTIONAL_MCPS", raising=False)

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert 'Environment="HERMES_BUNDLED_LOCALES=/nix/store/abc-hermes/share/hermes-agent/locales"' in unit
    assert 'Environment="HERMES_INSTALL_ROOT=/nix/store/abc-hermes/share/hermes-agent"' in unit
    assert "HERMES_OPTIONAL_MCPS" not in unit


def test_installed_unit_keeps_resource_roots_when_the_shell_lacks_them(monkeypatch, tmp_path):
    """The unit is regenerated and compared on every start/restart/status, so a later shell that
    never sourced the packaging wrapper must not "repair" the resource roots away."""
    unit_path = tmp_path / "hermes-gateway.service"
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", "/nix/store/abc-hermes/share/hermes-agent/locales")
    unit_path.write_text(gateway_cli.generate_systemd_unit(system=False), encoding="utf-8")

    monkeypatch.delenv("HERMES_BUNDLED_LOCALES")

    assert gateway_cli.systemd_unit_is_current(system=False)
    assert "HERMES_BUNDLED_LOCALES=/nix/store/abc-hermes/share/hermes-agent/locales" in \
        gateway_cli.generate_systemd_unit(system=False)


def test_system_unit_remaps_caller_home_resource_roots(monkeypatch, tmp_path):
    """Under sudo the caller's ``/root/...`` resource roots are unreadable to the target user, so a
    source install's roots are remapped the way the PATH and LD_LIBRARY_PATH entries are."""
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path",
                        lambda system=False: tmp_path / "hermes-gateway.service")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(gateway_cli, "_system_service_identity",
                        lambda run_as_user=None: ("alice", "alice", "/home/alice", 1001))
    monkeypatch.setattr(gateway_cli, "_build_service_path_dirs", list)
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", "/root/.hermes/locales")

    unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

    assert 'Environment="HERMES_BUNDLED_LOCALES=/home/alice/.hermes/locales"' in unit
