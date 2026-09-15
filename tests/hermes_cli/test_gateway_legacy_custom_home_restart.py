"""Legacy systemd restart routing for custom HERMES_HOME installs."""

from pathlib import Path
from types import SimpleNamespace

from hermes_cli import gateway


def _restart_fixture(monkeypatch, tmp_path, *, pinned_home: Path):
    machine_home = tmp_path / "machine-home"
    current_home = tmp_path / "custom-home"
    machine_home.mkdir()
    current_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(current_home))

    legacy_unit = machine_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        "[Service]\n" f'Environment="HERMES_HOME={pinned_home}"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway, "_refuse_from_inside_gateway", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway, "_guard_named_profile_under_multiplexer", lambda **kwargs: None)
    monkeypatch.setattr(gateway, "_dispatch_via_service_manager_if_s6", lambda *args, **kwargs: False)
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "_preflight_user_systemd", lambda: None)
    monkeypatch.setattr(gateway, "stop_profile_gateway", lambda: False)
    monkeypatch.setattr(gateway, "_wait_for_gateway_exit", lambda **kwargs: None)

    systemctl_calls = []
    monkeypatch.setattr(
        gateway,
        "_run_systemctl",
        lambda args, **kwargs: systemctl_calls.append((list(args), kwargs)),
    )
    foreground_calls = []
    monkeypatch.setattr(gateway, "run_gateway", lambda **kwargs: foreground_calls.append(kwargs))
    return current_home, legacy_unit, systemctl_calls, foreground_calls


def test_restart_owned_bare_legacy_unit_through_systemd(monkeypatch, tmp_path):
    current_home = tmp_path / "custom-home"
    _, legacy_unit, systemctl_calls, foreground_calls = _restart_fixture(
        monkeypatch, tmp_path, pinned_home=current_home
    )

    assert gateway.get_service_name().startswith("hermes-gateway-")
    assert not gateway.get_systemd_unit_path(system=False).exists()
    gateway._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

    assert [call[0] for call in systemctl_calls] == [["restart", "hermes-gateway"]]
    assert foreground_calls == []
    assert legacy_unit.exists()


def test_restart_foreign_bare_legacy_unit_keeps_foreground_control(monkeypatch, tmp_path):
    foreign_home = tmp_path / "foreign-home"
    foreign_home.mkdir()
    _, legacy_unit, systemctl_calls, foreground_calls = _restart_fixture(
        monkeypatch, tmp_path, pinned_home=foreign_home
    )
    before = legacy_unit.read_bytes()

    gateway._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

    assert systemctl_calls == []
    assert foreground_calls == [{"verbose": 0, "force": False}]
    assert legacy_unit.read_bytes() == before
