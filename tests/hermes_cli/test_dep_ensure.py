from unittest.mock import patch




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


def test_ensure_dependency_honors_sealed_env(monkeypatch):
    """HERMES_DISABLE_LAZY_INSTALLS=1 (no durable target) must block the
    install.sh shell-out — same policy gate as pip lazy installs. Regression:
    before this gate, every hermetic test / sealed container that hit a
    missing dep spawned a real ~2-7s bash install.sh run (issue #79771
    follow-up: test_browser_homebrew_paths.py spent 8s of its 36s here)."""
    from hermes_cli.dep_ensure import ensure_dependency
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.delenv("HERMES_LAZY_INSTALL_TARGET", raising=False)
    with patch("hermes_cli.dep_ensure._DEP_CHECKS", {"node": lambda: False}), \
         patch("subprocess.run") as mock_run:
        assert ensure_dependency("node", interactive=False) is False
    mock_run.assert_not_called()


def test_ensure_dependency_honors_config_kill_switch(monkeypatch):
    """security.allow_lazy_installs: false must block the install.sh path."""
    from hermes_cli.dep_ensure import ensure_dependency
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    with patch("hermes_cli.dep_ensure._DEP_CHECKS", {"node": lambda: False}), \
         patch("tools.lazy_deps._allow_lazy_installs", return_value=False), \
         patch("subprocess.run") as mock_run:
        assert ensure_dependency("node", interactive=False) is False
    mock_run.assert_not_called()








def test_ensure_dependency_uses_powershell_on_windows(tmp_path):
    from hermes_cli.dep_ensure import ensure_dependency
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "install.ps1").write_text("# fake")
    with patch("hermes_cli.dep_ensure._IS_WINDOWS", True), \
         patch("hermes_cli.dep_ensure._DEP_CHECKS", {"node": lambda: False}), \
         patch("tools.lazy_deps.lazy_installs_allowed", return_value=True), \
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
