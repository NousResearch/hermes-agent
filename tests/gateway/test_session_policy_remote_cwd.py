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
