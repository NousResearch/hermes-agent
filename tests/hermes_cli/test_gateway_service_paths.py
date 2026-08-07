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


def test_user_local_paths_includes_home_bin_when_present(tmp_path):
    """~/bin should be on the service PATH when it exists.

    ~/bin is on the default Debian/Ubuntu login PATH and is where CLIs like
    the 1Password `op` binary commonly live. Omitting it from the generated
    systemd unit means `op://` secret references (e.g. TELEGRAM_BOT_TOKEN)
    fail to resolve under the service even though they work in a login shell.
    """
    home_bin = tmp_path / "bin"
    home_bin.mkdir()
    from hermes_cli.gateway import _build_user_local_paths
    result = _build_user_local_paths(tmp_path, [])
    assert str(home_bin) in result


def test_user_local_paths_skips_home_bin_when_absent(tmp_path):
    """~/bin should not be added when it doesn't exist."""
    from hermes_cli.gateway import _build_user_local_paths
    result = _build_user_local_paths(tmp_path, [])
    assert str(tmp_path / "bin") not in result
