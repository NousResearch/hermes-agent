from unittest.mock import Mock, patch




def test_acp_browser_setup_ignores_disabled_cli_browser_toolset(tmp_path):
    """An explicit ACP setup request must install browser support."""
    from acp_adapter.entry import _run_setup_browser

    script = tmp_path / "install.sh"
    script.touch()
    config = {"platform_toolsets": {"cli": ["file", "terminal", "web"]}}
    browser_check = Mock(side_effect=[False, True])
    with patch("hermes_cli.dep_ensure._DEP_CHECKS", {"node": lambda: True, "browser": browser_check}), \
         patch("hermes_cli.config.load_config", return_value=config), \
         patch("hermes_cli.dep_ensure._find_install_script", return_value=(script, "bash")), \
         patch("hermes_cli.dep_ensure.subprocess.run", return_value=Mock(returncode=0)) as run:
        assert _run_setup_browser(assume_yes=True) == 0

    assert run.call_args.args[0] == ["bash", str(script), "--ensure", "browser"]
    assert browser_check.call_count == 2


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








def test_ensure_dependency_uses_powershell_on_windows(tmp_path):
    from hermes_cli.dep_ensure import ensure_dependency
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "install.ps1").write_text("# fake", encoding="utf-8")
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
