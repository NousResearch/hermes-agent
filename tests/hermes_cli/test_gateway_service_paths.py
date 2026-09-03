from pathlib import Path
from unittest.mock import patch


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True)
    return path


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


def test_service_path_includes_runtime_checkout_node_bin_for_python_overlay(
    tmp_path, monkeypatch
):
    from hermes_cli import gateway as gateway_cli

    overlay = _mkdir(tmp_path / "overlay")
    runtime = _mkdir(tmp_path / "runtime")
    active_venv = _mkdir(runtime / ".venv")
    _mkdir(active_venv / "bin")
    runtime_node_bin = _mkdir(runtime / "node_modules" / ".bin")
    hermes_home = _mkdir(tmp_path / "hermes-home")

    monkeypatch.setattr(gateway_cli.sys, "prefix", str(active_venv))
    monkeypatch.setattr(gateway_cli.sys, "base_prefix", str(tmp_path / "base-python"))
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: hermes_home)

    result = gateway_cli._build_service_path_dirs(project_root=overlay)

    assert result == [str(active_venv / "bin"), str(runtime_node_bin)]


def test_service_path_does_not_infer_project_root_from_custom_named_venv(
    tmp_path, monkeypatch
):
    from hermes_cli import gateway as gateway_cli

    overlay = _mkdir(tmp_path / "overlay")
    venv_parent = _mkdir(tmp_path / "venvs")
    active_venv = _mkdir(venv_parent / "hermes-runtime")
    _mkdir(active_venv / "bin")
    unrelated_node_bin = _mkdir(venv_parent / "node_modules" / ".bin")
    hermes_home = _mkdir(tmp_path / "hermes-home")

    monkeypatch.setattr(gateway_cli.sys, "prefix", str(active_venv))
    monkeypatch.setattr(gateway_cli.sys, "base_prefix", str(tmp_path / "base-python"))
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: hermes_home)

    result = gateway_cli._build_service_path_dirs(project_root=overlay)

    assert result == [str(active_venv / "bin")]
    assert str(unrelated_node_bin) not in result


def test_service_path_does_not_duplicate_node_bin_without_overlay(
    tmp_path, monkeypatch
):
    from hermes_cli import gateway as gateway_cli

    runtime = _mkdir(tmp_path / "runtime")
    active_venv = _mkdir(runtime / "venv")
    venv_bin = _mkdir(active_venv / "bin")
    node_bin = _mkdir(runtime / "node_modules" / ".bin")
    hermes_home = _mkdir(tmp_path / "hermes-home")

    monkeypatch.setattr(gateway_cli.sys, "prefix", str(active_venv))
    monkeypatch.setattr(gateway_cli.sys, "base_prefix", str(tmp_path / "base-python"))
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: hermes_home)

    result = gateway_cli._build_service_path_dirs(project_root=runtime)

    assert result == [str(venv_bin), str(node_bin)]


def test_generated_unit_remains_current_under_python_source_overlay(
    tmp_path, monkeypatch
):
    from hermes_cli import gateway as gateway_cli

    runtime = _mkdir(tmp_path / "runtime")
    overlay = _mkdir(tmp_path / "overlay")
    active_venv = _mkdir(runtime / ".venv")
    _mkdir(active_venv / "bin")
    runtime_node_bin = _mkdir(runtime / "node_modules" / ".bin")
    hermes_home = _mkdir(tmp_path / "hermes-home")
    unit_path = tmp_path / "hermes-gateway.service"

    monkeypatch.setattr(gateway_cli.sys, "prefix", str(active_venv))
    monkeypatch.setattr(gateway_cli.sys, "base_prefix", str(tmp_path / "base-python"))
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(
        gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path
    )
    monkeypatch.setattr(
        gateway_cli,
        "_sync_hermes_home_from_systemd_unit",
        lambda system=False: None,
    )
    monkeypatch.setattr(
        gateway_cli, "_append_node_dir_for_service", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        gateway_cli, "_build_user_local_paths", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        gateway_cli, "_build_wsl_interop_paths", lambda *args, **kwargs: []
    )

    monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", runtime)
    installed = gateway_cli.generate_systemd_unit(system=False)
    assert str(runtime_node_bin) in installed
    unit_path.write_text(installed, encoding="utf-8")

    monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", overlay)

    assert gateway_cli.systemd_unit_is_current(system=False) is True
