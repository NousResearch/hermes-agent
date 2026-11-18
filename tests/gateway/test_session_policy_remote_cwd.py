"""A remote or container terminal backend never runs in the client's host launch folder.

Ink sends its launch folder as every session's ``cwd``; that path exists on this host only. ssh and
the sandboxes keep their own configured cwd (or their default) unless the user picked the folder, and
docker with ``docker_mount_cwd_to_workspace`` still mounts it at ``/workspace``.
"""
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("cwd", ["/work/hermes-test", "/home/hermes-test/work", "/Users/hermes-test/work"])
def test_container_namespace_survives_create_restore_and_turn(tmp_path, monkeypatch, cwd):
    """A declared container mount is not required to exist on the gateway host (#140527)."""
    from gateway import session_managed_worker
    from gateway.session_policy import build_policy, policy_scope, restore_policy
    from tools.terminal_tool import _get_env_config, _resolve_command_cwd, record_session_cwd, clear_session_cwd

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    terminal = {"backend": "docker", "cwd": cwd, "docker_mount_cwd_to_workspace": False,
                "docker_volumes": [f"{tmp_path}:{cwd}:ro"]}
    (home / "config.yaml").write_text(json.dumps({"terminal": terminal}), encoding="utf-8")
    assert not Path(cwd).exists()
    policy = build_policy({"source": "tui", "cwd": cwd, "model": "m"}, {})
    policy = restore_policy(asdict(policy))

    async def execute(authority, ref, row, accepted_policy):
        assert accepted_policy == policy
        return "worker started"

    monkeypatch.setattr(session_managed_worker, "execute_managed", execute)
    authority = SimpleNamespace(pending_results={})
    assert asyncio.run(session_managed_worker.admit_managed(
        authority, None, {"admission_id": "container-turn"}, policy)) == "worker started"
    assert authority.pending_results == {}
    with policy_scope(policy):
        config = _get_env_config()
        assert (config["cwd"], config["host_cwd"]) == (cwd, None)
        assert config["docker_volumes"] == terminal["docker_volumes"]
        record_session_cwd("container-namespace", cwd)
        try:
            assert _resolve_command_cwd(workdir=None, default_cwd="/root", env_type="docker",
                                        session_key="container-namespace") == cwd
        finally:
            clear_session_cwd("container-namespace")


@pytest.mark.parametrize("mount", [False, True])
def test_host_namespace_still_requires_existing_directory(tmp_path, monkeypatch, mount):
    from gateway.session_policy import build_policy, policy_scope, workspace_ready
    from hermes_state_runtime import RuntimeStoreError

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    terminal = {"backend": "docker" if mount else "local", "docker_mount_cwd_to_workspace": mount}
    (home / "config.yaml").write_text(json.dumps({"terminal": terminal}), encoding="utf-8")
    cwd = tmp_path / "workspace"
    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        build_policy({"cwd": str(cwd), "model": "m"}, {})
    cwd.mkdir()
    policy = build_policy({"cwd": str(cwd), "model": "m"}, {})
    assert workspace_ready(policy)
    cwd.rmdir()
    assert not workspace_ready(policy)
    with pytest.raises(RuntimeStoreError, match="workspace_unavailable"):
        with policy_scope(policy):
            pass
