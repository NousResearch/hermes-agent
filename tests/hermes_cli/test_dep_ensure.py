from unittest.mock import patch


def test_ensure_dependency_skips_when_present():
    """ensure_dependency is a no-op when the dep is already available."""
    from hermes_cli.dep_ensure import ensure_dependency
    with patch("hermes_cli.dep_ensure.shutil") as mock_shutil:
        mock_shutil.which.return_value = "/usr/bin/node"
        result = ensure_dependency("node", interactive=False)
        assert result is True


def test_ensure_dependency_returns_false_when_missing_noninteractive():
    """ensure_dependency returns False for missing dep in non-interactive mode."""
    from hermes_cli.dep_ensure import ensure_dependency
    with patch("hermes_cli.dep_ensure.shutil") as mock_shutil:
        mock_shutil.which.return_value = None
        with patch("hermes_cli.dep_ensure._find_install_script", return_value=(None, None)):
            result = ensure_dependency("node", interactive=False)
            assert result is False


def test_find_install_script_from_checkout(tmp_path):
    """_find_install_script finds scripts/install.sh in a git checkout."""
    from hermes_cli.dep_ensure import _find_install_script
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "install.sh").write_text("#!/bin/bash", encoding="utf-8")
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", False):
        path, shell = _find_install_script(package_dir=tmp_path / "hermes_cli", repo_root=tmp_path)
    assert path is not None
    assert path.name == "install.sh"
    assert shell == "bash"


def test_find_install_script_from_wheel(tmp_path):
    """_find_install_script finds bundled install.sh in a wheel."""
    from hermes_cli.dep_ensure import _find_install_script
    bundled = tmp_path / "hermes_cli" / "scripts"
    bundled.mkdir(parents=True)
    (bundled / "install.sh").write_text("#!/bin/bash", encoding="utf-8")
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", False):
        path, shell = _find_install_script(package_dir=tmp_path / "hermes_cli", repo_root=tmp_path)
    assert path is not None
    assert path.name == "install.sh"
    assert shell == "bash"


def test_find_install_script_prefers_ps1_on_windows(tmp_path):
    """On Windows, _find_install_script should find install.ps1."""
    scripts_dir = tmp_path / "hermes_cli" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "install.ps1").write_text("# fake")
    (scripts_dir / "install.sh").write_text("# fake")
    from hermes_cli.dep_ensure import _find_install_script
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", True):
        path, shell = _find_install_script(package_dir=tmp_path / "hermes_cli")
        assert path == scripts_dir / "install.ps1"
        assert shell == "powershell"


def test_find_install_script_returns_sh_on_posix(tmp_path):
    """On POSIX, _find_install_script should find install.sh."""
    scripts_dir = tmp_path / "hermes_cli" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "install.ps1").write_text("# fake")
    (scripts_dir / "install.sh").write_text("# fake")
    from hermes_cli.dep_ensure import _find_install_script
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", False):
        path, shell = _find_install_script(package_dir=tmp_path / "hermes_cli")
        assert path == scripts_dir / "install.sh"
        assert shell == "bash"


def test_find_install_script_falls_back_to_repo_root(tmp_path):
    """When no bundled script, check repo root."""
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "scripts" / "install.sh").write_text("# fake")
    from hermes_cli.dep_ensure import _find_install_script
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", False):
        path, shell = _find_install_script(package_dir=tmp_path / "hermes_cli", repo_root=repo_root)
        assert path == repo_root / "scripts" / "install.sh"
        assert shell == "bash"


def test_find_install_script_returns_none_when_missing(tmp_path):
    from hermes_cli.dep_ensure import _find_install_script
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", False):
        result = _find_install_script(package_dir=tmp_path / "x", repo_root=tmp_path / "y")
        assert result == (None, None)


def test_has_system_browser_checks_windows_names():
    from hermes_cli.dep_ensure import _has_system_browser
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", True), \
         patch("hermes_cli.dep_ensure.shutil") as mock_shutil:
        mock_shutil.which.side_effect = lambda name: "/fake/msedge.exe" if name == "msedge" else None
        assert _has_system_browser() is True


def test_has_system_browser_checks_posix_names():
    from hermes_cli.dep_ensure import _has_system_browser
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", False), \
         patch("hermes_cli.dep_ensure.shutil") as mock_shutil:
        mock_shutil.which.return_value = None
        assert _has_system_browser() is False


def test_has_hermes_agent_browser_windows_path(tmp_path):
    node_dir = tmp_path / "node"
    node_dir.mkdir(parents=True)
    candidate = node_dir / "agent-browser.cmd"
    candidate.write_text("#!/bin/sh\nexit 0\n")
    candidate.chmod(0o755)
    from hermes_cli.dep_ensure import _has_hermes_agent_browser
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", True), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        assert _has_hermes_agent_browser() is True


def test_has_hermes_agent_browser_posix_path(tmp_path):
    bin_dir = tmp_path / "node" / "bin"
    bin_dir.mkdir(parents=True)
    candidate = bin_dir / "agent-browser"
    candidate.write_text("#!/bin/sh\nexit 0\n")
    candidate.chmod(0o755)
    from hermes_cli.dep_ensure import _has_hermes_agent_browser
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", False), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        assert _has_hermes_agent_browser() is True


def test_has_hermes_agent_browser_legacy_node_modules_path(tmp_path):
    """Legacy git-clone installs put agent-browser in $HERMES_HOME/node_modules/.bin/."""
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    candidate = bin_dir / "agent-browser"
    candidate.write_text("#!/bin/sh\nexit 0\n")
    candidate.chmod(0o755)
    from hermes_cli.dep_ensure import _has_hermes_agent_browser
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", False), \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        assert _has_hermes_agent_browser() is True


def test_ensure_dependency_uses_powershell_on_windows(tmp_path):
    from hermes_cli.dep_ensure import ensure_dependency
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "install.ps1").write_text("# fake")
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", True), \
         patch("hermes_cli.dep_ensure._DEP_CHECKS", {"node": lambda: False}), \
         patch("hermes_cli.dep_ensure._find_install_script", return_value=(scripts_dir / "install.ps1", "powershell")), \
         patch("hermes_cli.dep_ensure.shutil") as mock_shutil, \
         patch("hermes_constants.get_hermes_home", return_value=tmp_path / "fakehome"), \
         patch("subprocess.run") as mock_run, \
         patch("sys.stdin") as mock_stdin:
        mock_shutil.which.side_effect = lambda name: "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" if name == "powershell" else None
        mock_stdin.isatty.return_value = False
        mock_run.return_value = type("R", (), {"returncode": 0})()
        ensure_dependency("node", interactive=False)
        cmd = mock_run.call_args[0][0]
        assert "powershell" in cmd[0].lower()
        assert "-Ensure" in cmd
        assert cmd[cmd.index("-Ensure") + 1] == "node"
        assert "-HermesHome" in cmd
        assert str(tmp_path / "fakehome") in cmd
