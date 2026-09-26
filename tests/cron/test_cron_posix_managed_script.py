from pathlib import Path
from cron import scheduler_script


def test_cron_python_invocation_posix_managed_venv(monkeypatch, tmp_path):
    """On POSIX PM-managed installs, run scripts with the selected venv's python (#123440)."""
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()

    fake_venv = tmp_path / "venv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)
    fake_python = fake_bin / "python"
    fake_python.touch()

    fake_store_py = tmp_path / "store" / "python3"
    fake_store_py.parent.mkdir(parents=True)
    fake_store_py.touch()

    monkeypatch.setattr(scheduler_script.sys, "platform", "darwin")
    monkeypatch.setattr(scheduler_script, "Path", lambda *args, **kwargs: fake_repo if (args and "__file__" in str(args[0])) else Path(*args, **kwargs))
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda r: fake_store_py)
    monkeypatch.setattr("pm.environments.selected_venv", lambda r: fake_venv)
    monkeypatch.setattr("pm.environments.venv_python", lambda v: fake_python)

    exe, overlay = scheduler_script._cron_python_invocation("/usr/bin/python3")

    assert exe == str(fake_python)
    assert overlay == {"HERMES_DISABLE_LAZY_INSTALLS": "1"}
    assert "PYTHONPATH" not in overlay


def test_cron_python_invocation_posix_unmanaged_fallback(monkeypatch, tmp_path):
    """On POSIX non-PM installs, keep the calling interpreter without overlay."""
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()

    monkeypatch.setattr(scheduler_script.sys, "platform", "darwin")
    monkeypatch.setattr(scheduler_script, "Path", lambda *args, **kwargs: fake_repo if (args and "__file__" in str(args[0])) else Path(*args, **kwargs))
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda r: None)

    exe, overlay = scheduler_script._cron_python_invocation("/custom/venv/bin/python3")

    assert exe == "/custom/venv/bin/python3"
    assert overlay == {}


def test_script_argv_posix_managed_returns_direct_argv(monkeypatch, tmp_path):
    """_script_argv returns direct [venv_python, script_path] on POSIX managed install."""
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()

    fake_venv = tmp_path / "venv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)
    fake_python = fake_bin / "python"
    fake_python.touch()

    fake_store_py = tmp_path / "store" / "python3"
    fake_store_py.parent.mkdir(parents=True)
    fake_store_py.touch()

    script = tmp_path / "task.py"
    script.touch()

    monkeypatch.setattr(scheduler_script.sys, "platform", "darwin")
    monkeypatch.setattr(scheduler_script, "Path", lambda *args, **kwargs: fake_repo if (args and "__file__" in str(args[0])) else Path(*args, **kwargs))
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda r: fake_store_py)
    monkeypatch.setattr("pm.environments.selected_venv", lambda r: fake_venv)
    monkeypatch.setattr("pm.environments.venv_python", lambda v: fake_python)

    argv, overlay, err = scheduler_script._script_argv(script)

    assert err is None
    assert argv == [str(fake_python), str(script)]
    assert overlay == {"HERMES_DISABLE_LAZY_INSTALLS": "1"}
