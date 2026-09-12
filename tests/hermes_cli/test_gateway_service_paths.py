from unittest.mock import patch

import hermes_cli.gateway as gateway_cli


def test_service_path_skips_nonexistent_node_modules(tmp_path):
    """Service PATH should not include node_modules/.bin if it doesn't exist."""
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    node_modules_bin = str(tmp_path / "node_modules" / ".bin")
    assert node_modules_bin not in dirs


def test_service_path_includes_node_modules_when_present(tmp_path):
    """Service PATH should include node_modules/.bin when it exists."""
    nm_bin = tmp_path / "node_modules" / ".bin"
    nm_bin.mkdir(parents=True)
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    assert str(nm_bin) in dirs


def _stub_systemd_path_builders(monkeypatch):
    monkeypatch.setattr(gateway_cli, "_build_service_path_dirs", lambda: ["/opt/hermes/venv/bin"])
    monkeypatch.setattr(gateway_cli, "_build_user_local_paths", lambda home, existing: [])
    monkeypatch.setattr(gateway_cli, "_build_wsl_interop_paths", lambda existing: [])
    monkeypatch.setattr(gateway_cli, "_append_node_dir_for_service", lambda *a, **k: None)


def _unit_path_dirs(unit: str) -> list[str]:
    path_line = next(l for l in unit.splitlines() if l.startswith('Environment="PATH='))
    baked = path_line.split("PATH=", 1)[1].strip('"')
    return baked.split(":")


def test_systemd_unit_path_includes_invoking_shell_path(monkeypatch):
    # #107296: NixOS coreutils live in /run/current-system/sw/bin, which is
    # on the installer shell PATH but was omitted from the systemd unit.
    nix_sw = "/run/current-system/sw/bin"
    monkeypatch.setenv("PATH", f"/nix/store/fake-env/bin:{nix_sw}:/usr/bin")
    _stub_systemd_path_builders(monkeypatch)
    unit = gateway_cli.generate_systemd_unit(system=False)
    baked = _unit_path_dirs(unit)
    assert nix_sw in baked
    assert "/nix/store/fake-env/bin" in baked
    assert "/usr/bin" in baked
    # venv still first
    assert baked[0] == "/opt/hermes/venv/bin"
    # FHS tail still present
    assert "/bin" in baked


def test_systemd_unit_path_empty_env_keeps_fhs_tail(monkeypatch):
    monkeypatch.setenv("PATH", "")
    _stub_systemd_path_builders(monkeypatch)
    unit = gateway_cli.generate_systemd_unit(system=False)
    baked = _unit_path_dirs(unit)
    assert "/usr/bin" in baked
    assert baked  # non-empty
    assert baked[0] == "/opt/hermes/venv/bin"
    assert "" not in baked


def test_systemd_unit_path_skips_empty_and_whitespace_segments(monkeypatch):
    nix_sw = "/run/current-system/sw/bin"
    monkeypatch.setenv("PATH", f"/nix/store/fake-env/bin::{nix_sw}:   :/usr/bin")
    _stub_systemd_path_builders(monkeypatch)
    unit = gateway_cli.generate_systemd_unit(system=False)
    baked = _unit_path_dirs(unit)
    assert nix_sw in baked
    assert "/nix/store/fake-env/bin" in baked
    assert "" not in baked
    assert "   " not in baked


def test_systemd_system_unit_also_keeps_shell_path(monkeypatch, tmp_path):
    nix_sw = "/run/current-system/sw/bin"
    monkeypatch.setenv("PATH", f"/nix/store/fake-env/bin:{nix_sw}:/usr/bin")
    _stub_systemd_path_builders(monkeypatch)
    monkeypatch.setattr(
        gateway_cli,
        "_system_service_identity",
        lambda run_as_user=None: ("alice", "alice", str(tmp_path / "alice"), 1001),
    )
    unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")
    baked = _unit_path_dirs(unit)
    assert nix_sw in baked
    assert "/nix/store/fake-env/bin" in baked
    assert baked[0] == "/opt/hermes/venv/bin"
    assert "/bin" in baked

