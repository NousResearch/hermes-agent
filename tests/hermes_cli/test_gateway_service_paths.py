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


def test_systemd_unit_roundtrip_sudo_path_matches(tmp_path):
    """PATH in system unit generated under sudo must match user expectation.

    When ``sudo hermes gateway install --system`` generates a unit, every
    PATH entry must be identical to what the real user would generate —
    otherwise the "outdated service definition" warning is perpetual
    because the self-heal also runs under sudo.
    """
    import re

    root_home = tmp_path / "root"
    user_home = tmp_path / "home" / "hermes"
    user_hermes = user_home / ".hermes"
    user_node_bin = user_hermes / "node" / "bin"
    user_node_bin.mkdir(parents=True)

    from hermes_cli.gateway import generate_systemd_unit

    def _extract_path(unit_text: str) -> list[str]:
        m = re.search(r'Environment="PATH=([^"]+)"', unit_text)
        return m.group(1).split(":") if m else []

    # Shared mocks (same venv, python, working dir for both)
    common = [
        patch("hermes_cli.gateway._detect_venv_dir", return_value=None),
        patch("hermes_cli.gateway.get_python_path", return_value="/usr/bin/python3"),
        patch("hermes_cli.gateway._stable_service_working_dir", return_value=str(user_hermes)),
        patch("hermes_cli.gateway._system_service_identity",
              return_value=("hermes", "hermes", str(user_home))),
        patch("hermes_cli.gateway._hermes_home_for_target_user",
              return_value=str(user_hermes)),
        patch("hermes_cli.gateway._profile_arg_for_target_user", return_value=""),
        patch("hermes_cli.gateway.shutil.which", return_value=None),
    ]

    # Generate as "sudo" (Path.home() = /root)
    with patch("pathlib.Path.home", return_value=root_home), \
         patch("hermes_cli.gateway.get_hermes_home", return_value=root_home / ".hermes"):
        for p in common:
            p.start()
        sudo_unit = generate_systemd_unit(system=True)
        for p in common:
            p.stop()

    # Generate as real user (Path.home() = /home/hermes)
    with patch("pathlib.Path.home", return_value=user_home), \
         patch("hermes_cli.gateway.get_hermes_home", return_value=user_hermes):
        for p in common:
            p.start()
        user_unit = generate_systemd_unit(system=True)
        for p in common:
            p.stop()

    sudo_path = _extract_path(sudo_unit)
    user_path = _extract_path(user_unit)

    assert sudo_path == user_path, (
        f"PATH entries must match regardless of calling user:\n"
        f"  sudo:  {sudo_path}\n"
        f"  user:  {user_path}"
    )
    assert str(user_node_bin) in sudo_path, (
        f"Target user's {user_node_bin} must be in PATH"
    )



def test_system_unit_no_caller_node_leakage(tmp_path):
    """Root's node path must not leak into the target user's systemd unit.

    When the calling user (root/sudo) has a node binary under its own home
    (e.g. nvm install), the remapped path under the target user's home may
    not exist.  The resolved node path must be remapped to the target user's
    home but should not be blindly appended without remapping.
    """
    root_home = tmp_path / "root"
    user_home = tmp_path / "home" / "hermes"
    user_hermes = user_home / ".hermes"
    # Target user has NO node/bin
    (user_hermes / "node").mkdir(parents=True)

    # Root has a node binary under its own home (e.g. nvm)
    root_node = root_home / ".nvm" / "versions" / "node" / "v22" / "bin"
    root_node.mkdir(parents=True)

    from hermes_cli.gateway import generate_systemd_unit

    with patch("pathlib.Path.home", return_value=root_home), \
         patch("hermes_cli.gateway.get_hermes_home", return_value=root_home / ".hermes"), \
         patch("hermes_cli.gateway._system_service_identity",
               return_value=("hermes", "hermes", str(user_home))), \
         patch("hermes_cli.gateway._hermes_home_for_target_user",
               return_value=str(user_hermes)), \
         patch("hermes_cli.gateway._profile_arg_for_target_user",
               return_value=""), \
         patch("hermes_cli.gateway.shutil.which",
               return_value=str(root_node / "node")):
        unit = generate_systemd_unit(system=True)

    # The remapped path may exist or not — but the raw root path must
    # never appear in the generated unit.
    raw_root_path = str(root_node)
    assert raw_root_path not in unit, (
        f"Root's node path ({raw_root_path}) must not leak into the "
        "target user's systemd unit"
    )


