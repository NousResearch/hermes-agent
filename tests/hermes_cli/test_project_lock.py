import os
import shutil
import uuid
from pathlib import Path

from hermes_cli import project_lock


def test_process_role_distinguishes_cli_and_electron_helpers():
    roots = {10, 20, 30}
    assert project_lock._process_role({"pid": 10, "exe": r"C:\repo\venv\Scripts\hermes.exe", "cmdline": []}, roots) == "cli-launcher"
    assert project_lock._process_role({"pid": 20, "exe": r"C:\Hermes\Hermes.exe", "cmdline": ["Hermes.exe"]}, roots) == "desktop-main"
    assert project_lock._process_role({"pid": 30, "exe": r"C:\Hermes\Hermes.exe", "cmdline": ["Hermes.exe", "--type=renderer"]}, roots) == "desktop-renderer"
    assert project_lock._process_role({"pid": 40, "exe": "pwsh.exe", "cmdline": []}, roots) == "child"


def test_same_or_child_is_boundary_safe(tmp_path):
    project = tmp_path / "project"
    child = project / "nested"
    sibling = tmp_path / "project-old"
    assert project_lock._same_or_child(str(project), project)
    assert project_lock._same_or_child(str(child), project)
    assert not project_lock._same_or_child(str(sibling), project)


def test_release_path_passes_only_after_probe_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(project_lock, "_process_snapshot", lambda _path: ([], [], None))
    monkeypatch.setattr(project_lock, "rename_probe", lambda _path: {"ok": True, "winerror": None, "message": "PASS"})
    result = project_lock.release_path(str(tmp_path))
    assert result["released"] is True
    assert result["action"] == "NO_ACTIVE_LOCK"


def test_release_path_moves_its_own_cwd_before_probing(monkeypatch):
    original_cwd = Path.cwd()
    runtime = original_cwd / ".hermes" / "task-runtime" / f"project-lock-{uuid.uuid4().hex}"
    project = runtime / "project"
    hermes_home = runtime / "hermes-home"
    project.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(project_lock, "_process_snapshot", lambda _path: ([], [], None))
    monkeypatch.setattr(
        project_lock,
        "rename_probe",
        lambda path: {
            "ok": not project_lock._same_or_child(str(Path.cwd()), path),
            "winerror": None,
            "message": "PASS",
        },
    )

    try:
        monkeypatch.chdir(project)
        result = project_lock.release_path(str(project))

        assert result["released"] is True
        assert result["action"] == "CURRENT_PROCESS_CWD_RELEASED"
        assert Path.cwd() == hermes_home
    finally:
        monkeypatch.chdir(original_cwd)
        shutil.rmtree(runtime)


def test_diagnose_moves_its_own_cwd_only_for_probe(monkeypatch):
    original_cwd = Path.cwd()
    runtime = original_cwd / ".hermes" / "task-runtime" / f"project-lock-{uuid.uuid4().hex}"
    project = runtime / "project"
    hermes_home = runtime / "hermes-home"
    project.mkdir(parents=True)
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(project_lock, "_process_snapshot", lambda _path: ([], [], None))

    def probe(_path):
        assert Path.cwd() == hermes_home
        return {"ok": True, "winerror": None, "message": "PASS"}

    monkeypatch.setattr(project_lock, "rename_probe", probe)
    try:
        monkeypatch.chdir(project)
        result = project_lock.diagnose(str(project))

        assert result["is_current_process_cwd"] is True
        assert result["rename_probe"]["ok"] is True
        assert Path.cwd() == project
    finally:
        monkeypatch.chdir(original_cwd)
        shutil.rmtree(runtime)


def test_diagnose_recommends_chdir_for_non_hermes_holder(monkeypatch):
    runtime = Path.cwd() / ".hermes" / "task-runtime" / f"project-lock-{uuid.uuid4().hex}"
    runtime.mkdir(parents=True)
    holder = {"pid": 42, "ppid": 1, "name": "pwsh.exe", "cwd": str(runtime)}
    monkeypatch.setattr(project_lock, "_process_snapshot", lambda _path: ([], [holder], None))
    monkeypatch.setattr(
        project_lock,
        "_rename_probe_without_current_cwd",
        lambda _path: {"ok": False, "winerror": 32, "message": "busy"},
    )
    try:
        result = project_lock.diagnose(str(runtime))
        assert result["recommended_action"] == "CLOSE_OR_CHDIR_LISTED_PROCESS"
    finally:
        shutil.rmtree(runtime)


def test_diagnose_excludes_its_own_process_from_lock_owners(monkeypatch):
    runtime = Path.cwd() / ".hermes" / "task-runtime" / f"project-lock-{uuid.uuid4().hex}"
    runtime.mkdir(parents=True)
    current = {"pid": os.getpid(), "ppid": 1, "name": "python.exe", "cwd": str(runtime)}
    caller = {"pid": 42, "ppid": 1, "name": "pwsh.exe", "cwd": str(runtime)}
    monkeypatch.setattr(
        project_lock,
        "_process_snapshot",
        lambda _path: ([], [current, caller], None),
    )
    monkeypatch.setattr(
        project_lock,
        "_rename_probe_without_current_cwd",
        lambda _path: {"ok": False, "winerror": 32, "message": "busy"},
    )
    try:
        result = project_lock.diagnose(str(runtime))
        assert result["known_open_handles"] == [caller]
    finally:
        shutil.rmtree(runtime)


def test_diagnose_recommends_release_only_for_hermes_holder(monkeypatch):
    runtime = Path.cwd() / ".hermes" / "task-runtime" / f"project-lock-{uuid.uuid4().hex}"
    runtime.mkdir(parents=True)
    holder = {"pid": 42, "ppid": 1, "name": "Hermes.exe", "cwd": str(runtime)}
    hermes = [dict(holder, role="desktop-main")]
    monkeypatch.setattr(project_lock, "_process_snapshot", lambda _path: (hermes, [holder], None))
    monkeypatch.setattr(
        project_lock,
        "_rename_probe_without_current_cwd",
        lambda _path: {"ok": False, "winerror": 32, "message": "busy"},
    )
    try:
        result = project_lock.diagnose(str(runtime))
        assert result["recommended_action"] == "RUN_RELEASE_PATH_OR_CLOSE_MATCHING_HERMES_SESSION"
    finally:
        shutil.rmtree(runtime)


def test_release_path_fails_closed_for_unknown_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(project_lock, "_process_snapshot", lambda _path: ([], [], None))
    monkeypatch.setattr(project_lock, "rename_probe", lambda _path: {"ok": False, "winerror": 32, "message": "busy"})
    monkeypatch.setattr(project_lock, "_request_desktop_release", lambda _path: None)
    result = project_lock.release_path(str(tmp_path))
    assert result["released"] is False
    assert result["action"] == "GRACEFUL_DESKTOP_RELEASE_FAILED_RESTART_REQUIRED"


def test_release_path_fails_closed_when_desktop_control_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(project_lock, "_process_snapshot", lambda _path: ([], [], "HANDLE_ENUMERATION_UNAVAILABLE_REQUIRES_ADMIN"))
    monkeypatch.setattr(project_lock, "rename_probe", lambda _path: {"ok": False, "winerror": 32, "message": "busy"})
    monkeypatch.setattr(project_lock, "_request_desktop_release", lambda _path: None)
    result = project_lock.release_path(str(tmp_path))
    assert result["released"] is False
    assert result["action"] == "GRACEFUL_DESKTOP_RELEASE_FAILED_RESTART_REQUIRED"


def test_release_path_reprobes_after_desktop_release(tmp_path, monkeypatch):
    probes = iter([
        {"ok": False, "winerror": 32, "message": "busy"},
        {"ok": True, "winerror": None, "message": "pass"},
    ])
    monkeypatch.setattr(project_lock, "_process_snapshot", lambda _path: ([], [], "HANDLE_ENUMERATION_UNAVAILABLE_REQUIRES_ADMIN"))
    monkeypatch.setattr(project_lock, "rename_probe", lambda _path: next(probes))
    monkeypatch.setattr(project_lock, "_request_desktop_release", lambda _path: {"released": True})
    result = project_lock.release_path(str(tmp_path))
    assert result["released"] is True
    assert result["action"] == "GRACEFUL_DESKTOP_RELEASE"
