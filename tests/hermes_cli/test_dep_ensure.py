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


def test_find_agent_browser_lazy_install_cycle_terminates(monkeypatch):
    """tools.browser_tool._find_agent_browser's "nothing found" branch calls
    ensure_dependency("browser"), whose "browser" check now includes
    _has_npx_agent_browser() -> _find_agent_browser(validate=False) again.
    That nested call must NOT be able to trigger another ensure_dependency
    call (only validate=True does that) — verifying the cycle is bounded to
    one extra rescan, not unbounded recursion, using the real functions on
    both sides rather than mocking the cycle away."""
    import shutil
    import tools.browser_tool as bt
    from hermes_cli import dep_ensure

    monkeypatch.setattr(bt, "_cached_agent_browser", None)
    monkeypatch.setattr(bt, "_agent_browser_resolved", False)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(bt, "_resolve_npx_bin", lambda: None)
    monkeypatch.setattr(dep_ensure, "_has_system_browser", lambda: False)
    monkeypatch.setattr(dep_ensure, "_has_hermes_agent_browser", lambda: False)
    monkeypatch.setattr(dep_ensure, "_find_install_script", lambda *a, **k: (None, None))

    real_find_agent_browser = bt._find_agent_browser
    validate_calls = []

    def counting_find_agent_browser(*, validate=True):
        validate_calls.append(validate)
        return real_find_agent_browser(validate=validate)

    monkeypatch.setattr(bt, "_find_agent_browser", counting_find_agent_browser)

    with pytest.raises(FileNotFoundError):
        bt._find_agent_browser(validate=True)

    # One outer validate=True call, plus exactly one bounded nested
    # validate=False rescan from _has_npx_agent_browser inside
    # ensure_dependency's "browser" check — not unbounded recursion, and not
    # a second ensure_dependency("browser") call (which would show up as a
    # second `True` in this list).
    assert validate_calls == [True, False]


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
