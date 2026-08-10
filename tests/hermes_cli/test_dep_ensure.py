from unittest.mock import patch

import pytest


@pytest.mark.linux_only
def test_find_install_script_from_checkout(tmp_path):
    """_find_install_script finds scripts/install.sh in a git checkout.

    ``linux_only``: the POSIX arm picks ``install.sh`` + ``bash``, which is
    already what ``_IS_WINDOWS`` reports here — nothing needs faking.
    """
    from hermes_cli.dep_ensure import _find_install_script
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "install.sh").write_text("#!/bin/bash", encoding="utf-8")
    path, shell = _find_install_script(package_dir=tmp_path / "hermes_cli", repo_root=tmp_path)
    assert path is not None
    assert path.name == "install.sh"
    assert shell == "bash"








def test_has_npx_agent_browser_true_when_npx_resolves():
    """agent-browser resolves lazily via npx on the default install (#43564)
    — _has_npx_agent_browser mirrors the runtime cascade so the "browser" dep
    check doesn't wrongly report it missing."""
    from hermes_cli.dep_ensure import _has_npx_agent_browser
    import tools.browser_tool as bt

    with patch.object(bt, "_find_agent_browser", return_value="npx agent-browser"), \
         patch.object(bt, "_requires_real_termux_browser_install", return_value=False):
        assert _has_npx_agent_browser() is True


def test_has_npx_agent_browser_false_on_termux_local_bare_npx():
    from hermes_cli.dep_ensure import _has_npx_agent_browser
    import tools.browser_tool as bt

    with patch.object(bt, "_find_agent_browser", return_value="npx agent-browser"), \
         patch.object(bt, "_requires_real_termux_browser_install", return_value=True):
        assert _has_npx_agent_browser() is False


def test_has_npx_agent_browser_false_when_nothing_resolves():
    from hermes_cli.dep_ensure import _has_npx_agent_browser
    import tools.browser_tool as bt

    def _raise(**_kw):
        raise FileNotFoundError("agent-browser CLI not found")

    with patch.object(bt, "_find_agent_browser", _raise):
        assert _has_npx_agent_browser() is False


@pytest.mark.windows_only
def test_ensure_dependency_uses_powershell_on_windows(tmp_path):
    """``windows_only``: the assertion is that we shell out to a real
    PowerShell. Faking ``_IS_WINDOWS`` on Linux also required faking
    ``shutil.which`` into inventing a powershell.exe that isn't there."""
    from hermes_cli.dep_ensure import ensure_dependency
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "install.ps1").write_text("# fake")
    with patch("hermes_cli.dep_ensure._DEP_CHECKS", {"node": lambda: False}), \
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
