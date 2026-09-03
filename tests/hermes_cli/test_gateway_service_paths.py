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


def test_service_path_includes_existing_package_manager_dirs(tmp_path):
    """Homebrew/Linuxbrew/usr-local bin dirs must be on the service PATH when present.

    launchd's default PATH (/usr/bin:/bin:/usr/sbin:/sbin) and systemd's
    minimal unit PATH both omit these, so CLI tools installed via Homebrew
    (gh, jq, ffmpeg, ...) are invisible to cron job scripts and any
    subprocess launched from the running gateway service, even though the
    same tools resolve fine in an interactive shell. Regression coverage
    for the gap that made a cron job's `gh` call fail with
    FileNotFoundError under the gateway service.
    """
    from pathlib import Path
    import hermes_cli.gateway as gateway_module

    real_is_dir = Path.is_dir
    existing = {"/opt/homebrew/bin", "/usr/local/bin"}

    def fake_is_dir(self):
        if str(self) in existing:
            return True
        if str(self) in {
            "/opt/homebrew/sbin",
            "/usr/local/sbin",
            "/home/linuxbrew/.linuxbrew/bin",
            "/home/linuxbrew/.linuxbrew/sbin",
        }:
            return False
        return real_is_dir(self)

    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"), \
         patch.object(Path, "is_dir", fake_is_dir):
        dirs = gateway_module._build_service_path_dirs(project_root=tmp_path)

    assert "/opt/homebrew/bin" in dirs
    assert "/usr/local/bin" in dirs
    assert "/opt/homebrew/sbin" not in dirs
    assert "/home/linuxbrew/.linuxbrew/bin" not in dirs


def test_service_path_omits_absent_package_manager_dirs(tmp_path):
    """No package-manager dir should be added when none exist on disk."""
    from pathlib import Path
    import hermes_cli.gateway as gateway_module

    pm_dirs = {
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/local/sbin",
        "/home/linuxbrew/.linuxbrew/bin",
        "/home/linuxbrew/.linuxbrew/sbin",
    }
    real_is_dir = Path.is_dir

    def fake_is_dir(self):
        if str(self) in pm_dirs:
            return False
        return real_is_dir(self)

    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"), \
         patch.object(Path, "is_dir", fake_is_dir):
        dirs = gateway_module._build_service_path_dirs(project_root=tmp_path)

    assert not (set(dirs) & pm_dirs)
