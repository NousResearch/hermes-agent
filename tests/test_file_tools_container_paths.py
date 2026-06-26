from pathlib import Path

from tools import file_tools


WORKSPACE_PROJECTS = Path("/workspace") / "projects"


def test_preserve_workspace_absolute_path_for_docker(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    path = str(WORKSPACE_PROJECTS / "example" / "proof.txt")

    assert file_tools._preserve_container_absolute_path(path) is True
    assert file_tools._resolve_path_for_task(path) == Path(path)


def test_preserve_outputs_absolute_path_for_docker(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    path = "/outputs/example/proof.txt"

    assert file_tools._preserve_container_absolute_path(path) is True
    assert file_tools._resolve_path_for_task(path) == Path(path)


def test_do_not_preserve_workspace_absolute_path_for_local(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")

    assert file_tools._preserve_container_absolute_path(str(WORKSPACE_PROJECTS / "example" / "proof.txt")) is False


def test_do_not_preserve_host_absolute_path_for_docker(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    assert file_tools._preserve_container_absolute_path("/host/workspaces/example/proof.txt") is False


def test_relative_paths_still_resolve_against_base(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    assert file_tools._resolve_path_for_task("relative/proof.txt") == tmp_path / "relative" / "proof.txt"
