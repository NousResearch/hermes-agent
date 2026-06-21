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


def test_service_path_uses_explicit_hermes_home_over_get_hermes_home(tmp_path):
    """When hermes_home is passed, it is used instead of get_hermes_home()."""
    target_home = tmp_path / "target_user" / ".hermes"
    target_node = target_home / "node" / "bin"
    target_node.mkdir(parents=True)

    default_home = tmp_path / "calling_user" / ".hermes"
    (default_home / "node").mkdir(parents=True)
    # No node/bin under default_home — only target_home has it

    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=default_home):
        dirs = _build_service_path_dirs(
            project_root=tmp_path, hermes_home=str(target_home)
        )
    assert str(target_node) in dirs, (
        "Should find node/bin under the explicit hermes_home, "
        "not the default get_hermes_home() path"
    )


def test_generate_systemd_unit_includes_target_user_node_bin(tmp_path, monkeypatch):
    """System-scoped unit should include node/bin from the target user's home."""
    target_home = tmp_path / "home" / "hermes"
    target_hermes = target_home / ".hermes"
    target_node_bin = target_hermes / "node" / "bin"
    target_node_bin.mkdir(parents=True)

    from hermes_cli.gateway import generate_systemd_unit

    with patch("hermes_cli.gateway._system_service_identity") as mock_identity:
        mock_identity.return_value = ("hermes", "hermes", str(target_home))
        with patch(
            "hermes_cli.gateway._hermes_home_for_target_user",
            return_value=str(target_hermes),
        ):
            with patch(
                "hermes_cli.gateway._profile_arg_for_target_user",
                return_value="",
            ):
                unit = generate_systemd_unit(system=True)

    assert "PATH=" in unit, "System unit should have a PATH environment variable"
    assert str(target_node_bin) in unit, (
        f"System unit PATH should include {target_node_bin}"
    )

