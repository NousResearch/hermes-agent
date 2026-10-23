"""`hermes --tui` reports PM's node failure like the dashboard PTY route does, not as a traceback."""
import sys

import pytest
from hermes_cli import main_tui_launch


def test_tui_launch_reports_pm_node_failure_without_traceback(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    from pm import InstallError

    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.chdir(tmp_path)

    def refuse(tui_dir, tui_dev):
        raise InstallError("node", "not installed and lazy installs are disabled: node",
                           "enable security.allow_lazy_installs or run `hermes pm install`")

    monkeypatch.setattr(main_tui_launch, "_make_tui_argv", refuse)
    monkeypatch.setattr(main_mod.subprocess, "call",
                        lambda *args, **kwargs: pytest.fail("the TUI must not start"))

    with pytest.raises(SystemExit) as exited:
        main_mod._launch_tui()

    assert exited.value.code == 1
    err = capsys.readouterr().err
    assert "lazy installs are disabled" in err
    assert "hermes pm install" in err
    assert "hermes --cli" in err
