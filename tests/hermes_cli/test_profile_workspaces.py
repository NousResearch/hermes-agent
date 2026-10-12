"""New profile workspaces must be persistent, destination-owned directories."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.config import DEFAULT_CONFIG, atomic_config_write, read_user_config_raw
from hermes_cli.config_effective import load_user_config_effective


@pytest.fixture
def profile_root(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda _name: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda _name: None)
    atomic_config_write(root / "config.yaml", {
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "model": {"default": "test-model", "provider": "openrouter"},
    })
    return root


def test_fresh_profile_persists_its_published_workspace(profile_root):
    home = profiles.create_profile("researcher", no_alias=True)

    workspace = home / "workspace"
    assert workspace.is_dir()
    assert read_user_config_raw(home / "config.yaml").get("terminal", {}).get("cwd") == str(workspace)
    assert load_user_config_effective(home / "config.yaml")["terminal"]["cwd"] == str(workspace)


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
def test_clone_rebases_source_owned_workspace_without_changing_source(profile_root, clone_all):
    source_workspace = profile_root / "workspace"
    source_workspace.mkdir()
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": "local", "cwd": str(source_workspace), "timeout": 77}
    atomic_config_write(profile_root / "config.yaml", config)
    original = (profile_root / "config.yaml").read_bytes()

    home = profiles.create_profile("cloned", clone_from="default", clone_all=clone_all,
                                   clone_config=not clone_all, no_alias=True)

    terminal = load_user_config_effective(home / "config.yaml")["terminal"]
    assert terminal["cwd"] == str(home / "workspace")
    assert terminal["timeout"] == 77
    assert (profile_root / "config.yaml").read_bytes() == original


def test_clone_rebases_lexically_equivalent_source_workspace(profile_root):
    source_workspace = profile_root / "workspace"
    source_workspace.mkdir()
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": "local", "cwd": str(source_workspace / ".." / "workspace")}
    atomic_config_write(profile_root / "config.yaml", config)

    home = profiles.create_profile("cloned", clone_from="default", clone_config=True, no_alias=True)

    assert (home / "workspace").is_dir()
    assert read_user_config_raw(home / "config.yaml")["terminal"]["cwd"] == str(home / "workspace")


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("source_cwd", [None, "", ".", "auto", "cwd", "profile-home", "null-terminal",
                                       "tilde-home", "tilde-workspace", "lexical-home"])
def test_clone_without_a_project_gets_its_own_workspace(profile_root, source_cwd, clone_all):
    config = read_user_config_raw(profile_root / "config.yaml")
    if source_cwd == "null-terminal":
        config["terminal"] = None
    elif source_cwd is not None:
        aliases = {"profile-home": str(profile_root), "tilde-home": "~/.hermes",
                   "tilde-workspace": "~/.hermes/workspace", "lexical-home": str(profile_root / "workspace" / "..")}
        config["terminal"] = {"cwd": aliases.get(source_cwd, source_cwd)}
    atomic_config_write(profile_root / "config.yaml", config)

    home = profiles.create_profile("cloned", clone_from="default", clone_config=not clone_all,
                                   clone_all=clone_all, no_alias=True)

    assert load_user_config_effective(home / "config.yaml")["terminal"]["cwd"] == str(home / "workspace")


@pytest.mark.parametrize("backend", ["ssh", "docker", "singularity", "modal", "daytona", "vercel"])
def test_clone_keeps_nonlocal_backend_working_directory(profile_root, backend):
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": backend, "cwd": "."}
    atomic_config_write(profile_root / "config.yaml", config)

    home = profiles.create_profile("remote", clone_from="default", clone_config=True, no_alias=True)

    assert load_user_config_effective(home / "config.yaml")["terminal"]["cwd"] == "."


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("env_type", ["ssh", "docker"])
@pytest.mark.parametrize("raw_cwd", [".", "  ~/remote/../project  "], ids=["placeholder", "remote-path"])
def test_clone_keeps_legacy_nonlocal_backend_raw_working_directory(profile_root, clone_all, env_type, raw_cwd):
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"env_type": env_type, "cwd": raw_cwd, "timeout": 77}
    atomic_config_write(profile_root / "config.yaml", config)
    original = (profile_root / "config.yaml").read_bytes()

    home = profiles.create_profile("remote", clone_from="default", clone_all=clone_all,
                                   clone_config=not clone_all, no_alias=True)

    assert read_user_config_raw(home / "config.yaml")["terminal"] == config["terminal"]
    assert load_user_config_effective(home / "config.yaml")["terminal"]["cwd"] == raw_cwd
    assert (profile_root / "config.yaml").read_bytes() == original


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("backend,env_type", [("local", "ssh"), ("ssh", "local")])
def test_clone_workspace_backend_wins_over_legacy_env_type(profile_root, clone_all, backend, env_type):
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": backend, "env_type": env_type, "cwd": ".", "timeout": 77}
    atomic_config_write(profile_root / "config.yaml", config)
    original = (profile_root / "config.yaml").read_bytes()

    home = profiles.create_profile("cloned", clone_from="default", clone_all=clone_all,
                                   clone_config=not clone_all, no_alias=True)

    expected_cwd = str(home / "workspace") if backend == "local" else "."
    assert read_user_config_raw(home / "config.yaml")["terminal"] == {
        "backend": backend, "env_type": env_type, "cwd": expected_cwd, "timeout": 77,
    }
    assert load_user_config_effective(home / "config.yaml")["terminal"]["cwd"] == expected_cwd
    assert (profile_root / "config.yaml").read_bytes() == original


def test_clone_keeps_an_explicit_external_project(profile_root, tmp_path):
    project = tmp_path / "application"
    project.mkdir()
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": "local", "cwd": str(project)}
    atomic_config_write(profile_root / "config.yaml", config)

    home = profiles.create_profile("project-bot", clone_from="default", clone_config=True, no_alias=True)

    assert load_user_config_effective(home / "config.yaml")["terminal"]["cwd"] == str(project)


def test_clone_keeps_raw_external_project_when_source_workspace_is_symlinked(profile_root, tmp_path):
    project = tmp_path / "application"
    project.mkdir()
    (profile_root / "workspace").symlink_to(project, target_is_directory=True)
    raw_cwd = f"  {project / '..' / 'application'}  "
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": "local", "cwd": raw_cwd}
    atomic_config_write(profile_root / "config.yaml", config)

    home = profiles.create_profile("project-bot", clone_from="default", clone_config=True, no_alias=True)

    assert read_user_config_raw(home / "config.yaml")["terminal"]["cwd"] == raw_cwd


def test_seeded_workspace_keeps_local_cli_launch_directory(profile_root, tmp_path, monkeypatch):
    from hermes_cli.cli_config_load import _mirror_config_to_env

    home = profiles.create_profile("cli-compat", no_alias=True, no_skills=True)
    config = read_user_config_raw(home / "config.yaml")
    assert config["terminal"]["cwd"] == str(home / "workspace")
    project = tmp_path / "launch-project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.setenv("TERMINAL_CWD", "previous-workspace")

    _mirror_config_to_env(config, True)

    assert config["terminal"]["cwd"] == str(project)
    assert os.environ["TERMINAL_CWD"] == str(project)
    assert read_user_config_raw(home / "config.yaml")["terminal"]["cwd"] == str(home / "workspace")


def test_cli_new_profile_persists_workspace_in_a_fresh_process(profile_root):
    env = {**os.environ, "HOME": str(profile_root.parent), "HERMES_HOME": str(profile_root)}
    env.pop("HERMES_PROFILE", None)
    result = subprocess.run([
        sys.executable, "-m", "hermes_cli.main", "profile", "create", "cli-new", "--no-alias", "--no-skills",
    ], env=env, capture_output=True, text=True, timeout=60, check=False)

    assert result.returncode == 0, result.stderr
    home = profile_root / "profiles" / "cli-new"
    assert (home / "workspace").is_dir()
    assert load_user_config_effective(home / "config.yaml")["terminal"]["cwd"] == str(home / "workspace")
    reopened = subprocess.run([
        sys.executable, "-m", "hermes_cli.main", "-p", "cli-new", "config", "get", "terminal.cwd",
    ], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert reopened.returncode == 0, reopened.stderr
    assert reopened.stdout.strip() == str(home / "workspace")


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("backend", ["ssh", "docker"])
@pytest.mark.parametrize("raw_cwd", [".", "~/.remote", "/workspace"])
def test_clone_uses_copied_env_backend(profile_root, clone_all, backend, raw_cwd):
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"cwd": raw_cwd, "timeout": 77}
    atomic_config_write(profile_root / "config.yaml", config)
    env_path = profile_root / ".env"
    env_path.write_text(f"TERMINAL_ENV={backend}\nUNRELATED_WORKSPACE_TEST=value\n", encoding="utf-8")
    original_yaml = (profile_root / "config.yaml").read_bytes()
    original_env = env_path.read_bytes()

    home = profiles.create_profile("remote-env", clone_from="default", clone_all=clone_all,
                                   clone_config=not clone_all, no_alias=True)

    assert read_user_config_raw(home / "config.yaml")["terminal"] == config["terminal"]
    assert (home / ".env").read_bytes() == original_env
    assert (profile_root / "config.yaml").read_bytes() == original_yaml
    assert env_path.read_bytes() == original_env


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("env_backend", [None, "ssh", "docker"])
def test_canonical_null_inherits_destination_policy_not_alias(profile_root, clone_all, env_backend):
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": None, "env_type": "ssh" if env_backend is None else "local",
                          "cwd": ".", "timeout": 77}
    atomic_config_write(profile_root / "config.yaml", config)
    env_path = profile_root / ".env"
    env_path.write_text(f"TERMINAL_ENV={env_backend}\n" if env_backend else "# no backend override\n",
                        encoding="utf-8")
    original_yaml = (profile_root / "config.yaml").read_bytes()
    original_env = env_path.read_bytes()

    home = profiles.create_profile("null-backend", clone_from="default", clone_all=clone_all,
                                   clone_config=not clone_all, no_alias=True)

    terminal = read_user_config_raw(home / "config.yaml")["terminal"]
    expected_cwd = "." if env_backend else str(home / "workspace")
    assert terminal == {**config["terminal"], "cwd": expected_cwd}
    assert (profile_root / "config.yaml").read_bytes() == original_yaml
    assert env_path.read_bytes() == original_env
    assert (home / ".env").read_bytes() == original_env


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("env_backend", [None, "ssh", "docker"])
def test_null_legacy_alias_inherits_destination_policy(profile_root, clone_all, env_backend):
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"env_type": None, "cwd": "."}
    atomic_config_write(profile_root / "config.yaml", config)
    env_path = profile_root / ".env"
    env_path.write_text(f"TERMINAL_ENV={env_backend}\n" if env_backend else "# use defaults\n",
                        encoding="utf-8")

    home = profiles.create_profile("null-alias", clone_from="default", clone_all=clone_all,
                                   clone_config=not clone_all, no_alias=True)

    expected_cwd = "." if env_backend else str(home / "workspace")
    assert read_user_config_raw(home / "config.yaml")["terminal"] == {**config["terminal"], "cwd": expected_cwd}


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("selector", ["backend", "env_type"])
@pytest.mark.parametrize("yaml_backend,env_backend", [("local", "ssh"), ("ssh", "local"), ("docker", "local")])
def test_yaml_backend_selector_wins_over_copied_env(profile_root, clone_all, selector, yaml_backend, env_backend):
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {selector: yaml_backend, "cwd": ".", "timeout": 77}
    if selector == "backend":
        config["terminal"]["env_type"] = env_backend
    atomic_config_write(profile_root / "config.yaml", config)
    env_path = profile_root / ".env"
    env_path.write_text(f"TERMINAL_ENV={env_backend}\n", encoding="utf-8")
    original_yaml = (profile_root / "config.yaml").read_bytes()
    original_env = env_path.read_bytes()

    home = profiles.create_profile("yaml-wins", clone_from="default", clone_all=clone_all,
                                   clone_config=not clone_all, no_alias=True)

    expected_cwd = str(home / "workspace") if yaml_backend == "local" else "."
    assert read_user_config_raw(home / "config.yaml")["terminal"] == {**config["terminal"], "cwd": expected_cwd}
    assert (home / ".env").read_bytes() == original_env
    assert env_path.read_bytes() == original_env
    assert (profile_root / "config.yaml").read_bytes() == original_yaml


@pytest.mark.parametrize("mode", ["fresh", "config", "full"])
@pytest.mark.parametrize("file_backend,ambient_backend", [(None, "ssh"), ("ssh", "local"), ("docker", "local")])
def test_creation_ignores_ambient_and_uncopied_source_backend(profile_root, monkeypatch, mode, file_backend, ambient_backend):
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"cwd": "."}
    atomic_config_write(profile_root / "config.yaml", config)
    env_path = profile_root / ".env"
    env_path.write_text(f"TERMINAL_ENV={file_backend}\n" if file_backend else "# use defaults\n",
                        encoding="utf-8")
    original_yaml = (profile_root / "config.yaml").read_bytes()
    original_env = env_path.read_bytes()
    monkeypatch.setenv("TERMINAL_ENV", ambient_backend)

    home = profiles.create_profile("ambient", clone_from=None if mode == "fresh" else "default",
                                   clone_all=mode == "full", clone_config=mode == "config", no_alias=True)

    expected_cwd = "." if mode != "fresh" and file_backend else str(home / "workspace")
    assert read_user_config_raw(home / "config.yaml")["terminal"]["cwd"] == expected_cwd
    assert os.environ["TERMINAL_ENV"] == ambient_backend
    assert (profile_root / "config.yaml").read_bytes() == original_yaml
    assert env_path.read_bytes() == original_env


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("backend", ["local", "ssh"])
def test_clone_materializes_config_and_env_without_mutating_source(profile_root, tmp_path, clone_all, backend):
    yaml_target = tmp_path / "source-config.yaml"
    env_target = tmp_path / "source.env"
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": None, "cwd": ".", "timeout": 77}
    atomic_config_write(yaml_target, config)
    env_target.write_text(f"TERMINAL_ENV={backend}\nUNRELATED_WORKSPACE_TEST=value\n", encoding="utf-8")
    (profile_root / "config.yaml").unlink()
    (profile_root / "config.yaml").symlink_to(yaml_target)
    (profile_root / ".env").symlink_to(env_target)
    original_yaml, original_env = yaml_target.read_bytes(), env_target.read_bytes()

    home = profiles.create_profile("symlink-source", clone_from="default", clone_all=clone_all,
                                   clone_config=not clone_all, no_alias=True)

    expected_cwd = str(home / "workspace") if backend == "local" else "."
    assert read_user_config_raw(home / "config.yaml")["terminal"] == {**config["terminal"], "cwd": expected_cwd}
    assert not (home / "config.yaml").is_symlink()
    assert not (home / ".env").is_symlink()
    assert (home / ".env").read_bytes() == original_env
    assert yaml_target.read_bytes() == original_yaml
    assert env_target.read_bytes() == original_env
    assert (profile_root / "config.yaml").readlink() == yaml_target
    assert (profile_root / ".env").readlink() == env_target


@pytest.mark.parametrize("clone_all", [False, True], ids=["config-clone", "full-clone"])
@pytest.mark.parametrize("backend", ["ssh", "docker"])
def test_cli_remote_clone_reopens_saved_policy(profile_root, clone_all, backend):
    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": None, "env_type": "local", "cwd": ".", "timeout": 77}
    atomic_config_write(profile_root / "config.yaml", config)
    env_path = profile_root / ".env"
    env_path.write_text(f"TERMINAL_ENV={backend}\n", encoding="utf-8")
    original_yaml, original_env = (profile_root / "config.yaml").read_bytes(), env_path.read_bytes()
    env = {**os.environ, "HOME": str(profile_root.parent), "HERMES_HOME": str(profile_root), "TERMINAL_ENV": "local"}
    env.pop("HERMES_PROFILE", None)
    command = [sys.executable, "-m", "hermes_cli.main", "profile", "create", "cli-remote",
               "--clone-from", "default", "--no-alias"]
    if clone_all:
        command.append("--clone-all")

    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60, check=False)

    assert result.returncode == 0, result.stderr
    home = profile_root / "profiles" / "cli-remote"
    assert read_user_config_raw(home / "config.yaml")["terminal"] == config["terminal"]
    assert (home / ".env").read_bytes() == original_env
    reopened = subprocess.run([
        sys.executable, "-m", "hermes_cli.main", "-p", "cli-remote", "config", "get", "terminal.cwd",
    ], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert reopened.returncode == 0, reopened.stderr
    assert reopened.stdout.strip() == "."
    assert (profile_root / "config.yaml").read_bytes() == original_yaml
    assert env_path.read_bytes() == original_env


def test_published_workspace_is_saved_before_atomic_rename(profile_root, monkeypatch):
    rename = os.rename
    published = []

    def observe_publish(source, destination, **kwargs):
        source, destination = Path(source), Path(destination)
        if source.name.startswith(".before-publish.staging-"):
            assert not destination.exists()
            assert (source / "workspace").is_dir()
            assert read_user_config_raw(source / "config.yaml")["terminal"]["cwd"] == str(destination / "workspace")
            published.append(destination)
        return rename(source, destination, **kwargs)

    monkeypatch.setattr(os, "rename", observe_publish)

    home = profiles.create_profile("before-publish", no_alias=True, no_skills=True)

    assert published == [home]


@pytest.mark.parametrize("backend", ["local", "ssh", "docker"])
def test_dashboard_http_creation_shares_workspace_policy(profile_root, backend):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_cli.web_routers.profiles import router

    config = read_user_config_raw(profile_root / "config.yaml")
    config["terminal"] = {"backend": None, "cwd": ".", "timeout": 77}
    atomic_config_write(profile_root / "config.yaml", config)
    (profile_root / ".env").write_text(f"TERMINAL_ENV={backend}\n", encoding="utf-8")
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.post("/api/profiles", json={"name": "dashboard-clone", "clone_from": "default"})

    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    home = profile_root / "profiles" / "dashboard-clone"
    expected_cwd = str(home / "workspace") if backend == "local" else "."
    assert read_user_config_raw(home / "config.yaml")["terminal"] == {**config["terminal"], "cwd": expected_cwd}
