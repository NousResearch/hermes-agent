import builtins
from pathlib import Path


def test_move_cwd_outside_rehomes_when_inside_target(tmp_path, monkeypatch):
    from hermes_cli import uninstall

    target = tmp_path / "hermes"
    nested = target / "hermes-agent" / "venv"
    fallback = tmp_path / "fallback"
    nested.mkdir(parents=True)
    fallback.mkdir()

    original_cwd = Path.cwd()
    monkeypatch.chdir(nested)
    monkeypatch.setattr(uninstall.tempfile, "gettempdir", lambda: str(fallback))

    uninstall._move_cwd_outside(target)

    assert Path.cwd() == fallback
    monkeypatch.chdir(original_cwd)


def test_schedule_windows_tree_removal_spawns_detached_helper(tmp_path, monkeypatch):
    from hermes_cli import uninstall

    target = tmp_path / "hermes"
    target.mkdir()
    popen_calls = []

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(uninstall.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(uninstall, "_is_windows", lambda: True)
    monkeypatch.setattr(uninstall.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(uninstall.subprocess, "DETACHED_PROCESS", 0x8, raising=False)
    monkeypatch.setattr(uninstall.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    assert uninstall._schedule_windows_tree_removal(target) is True

    helper = next(tmp_path.glob("hermes-uninstall-*.cmd"))
    contents = helper.read_text(encoding="utf-8")
    assert f'set "TARGET={target.resolve()}"' in contents
    assert 'rmdir /s /q "%TARGET%"' in contents
    assert popen_calls[0][0] == ["cmd.exe", "/d", "/c", str(helper)]
    assert popen_calls[0][1]["creationflags"] == 0x08000008


def test_run_uninstall_full_windows_schedules_deferred_cleanup_for_live_home(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli import uninstall

    hermes_home = tmp_path / "hermes"
    project_root = hermes_home / "hermes-agent"
    project_root.mkdir(parents=True)

    answers = iter(["2", "yes"])
    scheduled = []

    def fake_rmtree(path, *args, **kwargs):
        raise PermissionError(f"locked: {path}")

    monkeypatch.setattr(uninstall, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall, "_is_windows", lambda: True)
    monkeypatch.setattr(uninstall, "_is_default_hermes_home", lambda _path: False)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_path_from_windows_registry", lambda _path: [])
    monkeypatch.setattr(uninstall, "remove_hermes_env_vars_windows", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_portable_tooling_windows", lambda _path: [])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    monkeypatch.setattr(uninstall.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(uninstall, "_schedule_windows_tree_removal", lambda path: scheduled.append(path) or True)

    uninstall.run_uninstall(args=None)

    out = capsys.readouterr().out
    assert scheduled == [hermes_home]
    assert f"Scheduled deferred removal of {hermes_home} after Hermes exits" in out
