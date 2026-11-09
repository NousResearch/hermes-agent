"""MEDIA files a live root Docker sandbox restricted (``chmod 600``) reach the gateway: the delivery
gate hands that one file back to the host user through the container, which keeps its hardening."""

import json
import os
import subprocess

import pytest

from gateway.platforms.base import validate_media_delivery_path
from tools.environments import docker as docker_env
from tools.environments import docker_ownership

pytestmark = pytest.mark.platforms("linux")


def _fake_docker(monkeypatch, sandbox, *, user: str = ""):
    """A live root sandbox container mounting ``sandbox``; ``exec ... chown`` really chowns the host
    file (as the container's root would), so the gateway's later read is the real check."""
    calls = []
    info = {"Config": {"User": user}, "HostConfig": {"CapAdd": ["CHOWN"]}, "Image": "sha256:img",
            "Mounts": [{"Type": "bind", "Source": str(sandbox / "workspace"), "Destination": "/workspace"},
                       {"Type": "bind", "Source": str(sandbox / "home"), "Destination": "/root"}]}

    def _capture(cmd, **_kw):
        calls.append(cmd)
        out = {"info": "[name=seccomp,profile=builtin]", "ps": "live-cid\n", "inspect": json.dumps([info])}[cmd[1]]
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    def _run(cmd, **_kw):
        calls.append(cmd)
        target = cmd[cmd.index("-P") + 1]
        host = sandbox / "workspace" / os.path.relpath(target, "/workspace")
        host.chmod(0o600)  # the handed-back owner can read its 0600 file
        return subprocess.CompletedProcess(cmd, 0, stdout=target.encode() + b"\0", stderr=b"")

    docker_ownership._remapping_cache.clear()
    monkeypatch.setattr(docker_ownership, "run_capture", _capture)
    monkeypatch.setattr(docker_ownership.subprocess, "run", _run)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    return calls


@pytest.mark.parametrize("user, delivered", [("", True), ("1000:1000", False)])
def test_unreadable_sandbox_media_is_handed_back_before_delivery(monkeypatch, tmp_path, user, delivered):
    sandbox = tmp_path / ".hermes" / "sandboxes" / "docker" / "default"
    (sandbox / "workspace").mkdir(parents=True)
    (sandbox / "home").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    target = sandbox / "workspace" / "test.txt"
    target.write_text("top-secret")
    target.chmod(0)  # what root's 0600 file is to the gateway user: unreadable
    calls = _fake_docker(monkeypatch, sandbox, user=user)

    safe = validate_media_delivery_path("/workspace/test.txt")

    assert safe == str(target.resolve())
    execs = [c for c in calls if c[1] == "exec"]
    if delivered:
        assert open(safe).read() == "top-secret"
        assert execs and execs[0][execs[0].index("-P") + 1] == "/workspace/test.txt"
        assert b"/workspace/test.txt\0" in (sandbox / docker_ownership.HANDBACK_RECORD).read_bytes()
    else:  # a non-root sandbox's files are already the host user's: nothing is chowned
        assert not execs and not os.access(safe, os.R_OK)
