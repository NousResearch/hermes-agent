from unittest.mock import patch


def test_launchd_user_home_uses_explicit_override(monkeypatch, tmp_path):
    from hermes_cli.gateway import _launchd_user_home

    monkeypatch.setenv("HERMES_LAUNCHD_HOME", str(tmp_path))

    assert _launchd_user_home() == tmp_path


def test_launchd_user_home_falls_back_to_home_when_pwd_lookup_fails(monkeypatch):
    from hermes_cli.gateway import _launchd_user_home

    monkeypatch.delenv("HERMES_LAUNCHD_HOME", raising=False)
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    monkeypatch.setenv("HOME", "/Users/example")

    with patch("pwd.getpwuid", side_effect=KeyError("uid not found")):
        assert _launchd_user_home().as_posix() == "/Users/example"


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
