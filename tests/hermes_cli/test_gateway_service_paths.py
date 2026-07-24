from pathlib import Path
from unittest.mock import patch


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


def test_service_path_includes_hermes_home_node_modules(tmp_path):
    """Service PATH should include ~/.hermes/node_modules/.bin when it exists."""
    hermes_nm = tmp_path / ".hermes" / "node_modules" / ".bin"
    hermes_nm.mkdir(parents=True)
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    assert str(hermes_nm) in dirs


def test_service_path_explicit_hermes_home_parameter(tmp_path):
    """hermes_home parameter overrides get_hermes_home() without patching."""
    target_hermes = tmp_path / "target_user" / ".hermes"
    node_bin = target_hermes / "node" / "bin"
    node_bin.mkdir(parents=True)

    from hermes_cli.gateway import _build_service_path_dirs
    # Pass hermes_home explicitly — get_hermes_home() must not be called.
    dirs = _build_service_path_dirs(project_root=tmp_path, hermes_home=target_hermes)
    assert str(node_bin) in dirs


def test_service_path_system_unit_uses_target_user_hermes_home(tmp_path):
    """generate_systemd_unit(system=True) must include target user's hermes-local node paths.

    Regression test for the sudo path bug: when run via sudo, get_hermes_home()
    returns /root/.hermes, whose node/bin doesn't exist.  The generated unit
    must use the target user's hermes_home instead so node/bin is included when
    it exists under that user's home.
    """
    root_home = tmp_path / "root"
    target_home = tmp_path / "ubuntu"
    root_hermes = root_home / ".hermes"
    target_hermes = target_home / ".hermes"

    # Only the target user has ~/.hermes/node/bin — root's does not exist.
    target_node_bin = target_hermes / "node" / "bin"
    target_node_bin.mkdir(parents=True)
    root_hermes.mkdir(parents=True)

    from hermes_cli.gateway import generate_systemd_unit

    with (
        patch("hermes_cli.gateway.get_hermes_home", return_value=root_hermes),
        patch("hermes_cli.gateway.Path.home", return_value=root_home),
        patch("hermes_cli.gateway._system_service_identity",
              return_value=("ubuntu", "ubuntu", str(target_home))),
        patch("hermes_cli.gateway._get_restart_drain_timeout", return_value=0),
        patch("hermes_cli.gateway.get_python_path", return_value="/usr/bin/python3"),
        patch("hermes_cli.gateway._detect_venv_dir", return_value=None),
        patch("hermes_cli.gateway.shutil.which", return_value=None),
        patch("hermes_cli.gateway._build_user_local_paths", return_value=[]),
        patch("hermes_cli.gateway._build_wsl_interop_paths", return_value=[]),
    ):
        unit = generate_systemd_unit(system=True)

    assert str(target_node_bin) in unit, (
        f"Expected {target_node_bin} in PATH but got:\n{unit}"
    )
