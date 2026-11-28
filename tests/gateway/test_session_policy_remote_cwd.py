"""A remote or container terminal backend never runs in the client's host launch folder.

Ink sends its launch folder as every session's ``cwd``; that path exists on this host only. ssh and
the sandboxes keep their own configured cwd (or their default) unless the user picked the folder, and
docker with ``docker_mount_cwd_to_workspace`` still mounts it at ``/workspace``.
"""
import json
from dataclasses import asdict

import pytest

LAUNCH = object()  # the client's host launch folder


@pytest.mark.parametrize("terminal,params,expected", [
    ({"backend": "ssh", "cwd": "/srv/app"}, {}, ("/srv/app", None)),
    ({"backend": "ssh"}, {}, ("~", None)),
    ({"backend": "docker", "cwd": "/workspace/app"}, {}, ("/workspace/app", None)),
    ({"backend": "docker", "docker_mount_cwd_to_workspace": True}, {}, ("/workspace", LAUNCH)),
    ({"backend": "local"}, {}, (LAUNCH, None)),
    ({"backend": "ssh", "cwd": "/srv/app"}, {"cwd_explicit": True}, (LAUNCH, None)),
])
def test_terminal_cwd_follows_the_backend(tmp_path, monkeypatch, terminal, params, expected):
    from gateway.session_policy import build_policy, policy_scope, restore_policy
    from tools.terminal_tool import _get_env_config

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({"terminal": terminal}), encoding="utf-8")  # JSON is YAML
    monkeypatch.setenv("HERMES_HOME", str(home))
    launch = tmp_path / "project"
    launch.mkdir()

    policy = build_policy({"source": "tui", "cwd": str(launch), "model": "m", **params}, {})
    with policy_scope(restore_policy(asdict(policy))):
        config = _get_env_config()
    assert (config["cwd"], config.get("host_cwd")) == tuple(str(launch) if v is LAUNCH else v for v in expected)


def _remote_home(tmp_path, monkeypatch, backend):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({"terminal": {"backend": backend}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))


def test_ssh_backend_accepts_a_cwd_that_only_exists_remotely(tmp_path, monkeypatch):
    """An ACP editor on an ssh backend names a folder on the remote host: this host cannot probe it."""
    from pathlib import Path

    from gateway.session_policy import build_policy, workspace_ready

    _remote_home(tmp_path, monkeypatch, "ssh")
    remote = "/srv/remote/project"
    assert not Path(remote).exists()
    policy = build_policy({"source": "acp", "cwd": remote, "cwd_explicit": True, "model": "m"}, {})
    assert policy.cwd == remote
    assert json.loads(policy.terminal_json)["TERMINAL_CWD"] == remote
    assert workspace_ready(policy)


@pytest.mark.parametrize("backend,cwd", [("ssh", "relative/dir"), ("local", "/srv/remote/project")])
def test_cwd_still_refused_when_relative_or_missing_on_a_local_backend(tmp_path, monkeypatch, backend, cwd):
    from gateway.session_policy import build_policy
    from hermes_state_runtime import RuntimeStoreError

    _remote_home(tmp_path, monkeypatch, backend)
    with pytest.raises(RuntimeStoreError):
        build_policy({"source": "acp", "cwd": cwd, "model": "m"}, {})


async def _move(tmp_path, monkeypatch, backend, cwd):
    """Run Move to project's prepare step against a policy frozen under *backend*."""
    from dataclasses import asdict

    from gateway import session_mutation_policy as mutation
    from gateway.session_policy import build_policy

    _remote_home(tmp_path, monkeypatch, backend)
    start = tmp_path / "start"
    start.mkdir()
    policy = build_policy({"source": "gui", "cwd": str(start), "model": "m"}, {})
    prepared = {"snapshot": {"receipt": {"policy": asdict(policy)}}}
    monkeypatch.setattr(mutation, "_rebind", lambda authority, prepared, old, new, **extra: (new, extra))
    return await mutation.prepare_workspace(None, None, {"cwd": cwd}, prepared)


@pytest.mark.asyncio
async def test_move_to_project_accepts_a_remote_only_folder_on_ssh(tmp_path, monkeypatch):
    remote = "/srv/remote/project"
    policy, extra = await _move(tmp_path, monkeypatch, "ssh", remote)
    assert policy.cwd == remote
    assert json.loads(policy.terminal_json)["TERMINAL_CWD"] == remote
    assert extra == {"git": {"branch": "", "root": ""}}  # no host git probe for a remote folder


@pytest.mark.asyncio
@pytest.mark.parametrize("cwd", ["relative/project", "~user/project"])
async def test_move_to_project_on_ssh_still_requires_an_absolute_path(tmp_path, monkeypatch, cwd):
    from hermes_state_runtime import RuntimeStoreError

    with pytest.raises(RuntimeStoreError):
        await _move(tmp_path, monkeypatch, "ssh", cwd)


@pytest.mark.asyncio
async def test_move_to_project_on_local_backend_still_requires_an_existing_folder(tmp_path, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError

    with pytest.raises(RuntimeStoreError):
        await _move(tmp_path, monkeypatch, "local", "/srv/remote/project")


@pytest.mark.asyncio
@pytest.mark.parametrize("cwd", ["~", "~/project"])
async def test_ssh_remote_home_cwd_kept_for_acp_create_and_move(tmp_path, monkeypatch, cwd):
    """SSH's ~ names the remote user's home, never the gateway host's home."""
    from dataclasses import asdict

    from gateway import session_mutation_policy as mutation
    from gateway.session_policy import build_policy, restore_policy, workspace_ready

    _remote_home(tmp_path, monkeypatch, "ssh")
    policy = build_policy({"source": "acp", "cwd": cwd, "cwd_explicit": True, "model": "m"}, {})
    assert policy.cwd == cwd
    assert json.loads(policy.terminal_json)["TERMINAL_CWD"] == cwd
    assert workspace_ready(policy)

    prepared = {"snapshot": {"receipt": {"policy": asdict(policy)}}}
    monkeypatch.setattr(mutation, "_rebind", lambda authority, prepared, old, new, **extra: (new, extra))
    moved, extra = await mutation.prepare_workspace(None, None, {"cwd": cwd}, prepared)
    assert moved.cwd == cwd
    assert restore_policy(asdict(moved)).cwd == cwd
    assert json.loads(moved.terminal_json)["TERMINAL_CWD"] == cwd
    assert extra == {"git": {"branch": "", "root": ""}}


@pytest.mark.parametrize("backend,cwd", [("ssh", "relative/dir"), ("ssh", "   "), ("local", "/srv/remote/project"),
                                         ("modal", "~/project")])
def test_restore_refuses_a_stored_cwd_the_backend_would_not_accept(tmp_path, monkeypatch, backend, cwd):
    """restore_policy applies the same backend-aware cwd rule as creation; ~ is an ssh-only spelling."""
    from dataclasses import asdict, replace

    from gateway.session_policy import build_policy, restore_policy

    _remote_home(tmp_path, monkeypatch, backend)
    start = tmp_path / "start"
    start.mkdir()
    stored = asdict(replace(build_policy({"source": "acp", "cwd": str(start), "model": "m"}, {}), cwd=cwd))
    with pytest.raises(ValueError):
        restore_policy(stored)
