from pathlib import Path
from types import SimpleNamespace

from hermes_cli.uninstall import run_uninstall


def test_full_uninstall_removes_named_profiles(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    hermes_home = home / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model: test\n")
    profiles_root = hermes_home / "profiles"
    profiles_root.mkdir()
    (profiles_root / "coder").mkdir()

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "placeholder.txt").write_text("x")

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.uninstall.get_project_root", lambda: project_root)
    monkeypatch.setattr("hermes_cli.uninstall.uninstall_gateway_service", lambda: False)
    monkeypatch.setattr("hermes_cli.uninstall.remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr("hermes_cli.uninstall.remove_wrapper_script", lambda: [])
    monkeypatch.setattr("builtins.input", lambda _prompt='': "2")

    prompts = iter(["2", "yes"])
    monkeypatch.setattr("builtins.input", lambda _prompt='': next(prompts))

    run_uninstall(SimpleNamespace())

    assert not hermes_home.exists()
    assert not profiles_root.exists()
    assert not project_root.exists()

    out = capsys.readouterr().out
    assert "Removed" in out
