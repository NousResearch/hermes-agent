"""MEDIA files a live root Docker sandbox restricted (``chmod 600``) reach the gateway: reply
extraction hands that one file back to the host user through the container, which keeps its
hardening, off the event loop and without ever losing the reply text."""

import json
import os
import subprocess
import threading

import pytest

from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.gateway.platforms.base import BasePlatformAdapter, SendResult
from hermes_agent.gateway.platforms.event import MessageEvent
from hermes_agent.gateway.session import SessionSource
from hermes_agent.tools.environments import docker as docker_env
from hermes_agent.tools.environments import docker_ownership

# Root reads a mode-0 file, so the permission premise only holds for an unprivileged gateway.
pytestmark = [pytest.mark.platforms("linux"),
              pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads mode-0 files")]


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.DISCORD)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _fake_docker(monkeypatch, sandbox, *, user: str = "", ps_timeout: bool = False):
    """A live root sandbox container mounting ``sandbox``; ``exec ... chown`` really makes the host
    file readable (as the container's root would), so the gateway's later read is the real check."""
    calls = []
    info = {"Config": {"User": user}, "HostConfig": {"CapAdd": ["CHOWN"]}, "Image": "sha256:img",
            "Mounts": [{"Type": "bind", "Source": str(sandbox / "workspace"), "Destination": "/workspace"},
                       {"Type": "bind", "Source": str(sandbox / "home"), "Destination": "/root"}]}

    def _capture(cmd, **_kw):
        calls.append((cmd, threading.current_thread()))
        if ps_timeout and cmd[1] == "ps":
            raise subprocess.TimeoutExpired(cmd, 10)
        out = {"info": "[name=seccomp,profile=builtin]", "ps": "live-cid\n", "inspect": json.dumps([info])}[cmd[1]]
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    def _run(cmd, **_kw):
        calls.append((cmd, threading.current_thread()))
        target = cmd[cmd.index("-P") + 1]
        (sandbox / "workspace" / os.path.relpath(target, "/workspace")).chmod(0o600)
        return subprocess.CompletedProcess(cmd, 0, stdout=target.encode() + b"\0", stderr=b"")

    docker_ownership._remapping_cache.clear()
    monkeypatch.setattr(docker_ownership, "run_capture", _capture)
    monkeypatch.setattr(docker_ownership.subprocess, "run", _run)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    return calls


def _sandbox_file(monkeypatch, tmp_path):
    sandbox = tmp_path / ".hermes" / "sandboxes" / "docker" / "default"
    (sandbox / "workspace").mkdir(parents=True)
    (sandbox / "home").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    target = sandbox / "workspace" / "test.txt"
    target.write_text("top-secret")
    target.chmod(0)  # what root's 0600 file is to the gateway user: unreadable
    return sandbox, target


async def _extract(text: str):
    event = MessageEvent(text="send it", source=SessionSource(platform=Platform.DISCORD, chat_id="c"))
    return await _Adapter()._extract_response_content(text, event, "", is_ephemeral_response=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("user, delivered", [("", True), ("1000:1000", False)])
async def test_unreadable_sandbox_media_is_handed_back_before_delivery(monkeypatch, tmp_path, user, delivered):
    sandbox, target = _sandbox_file(monkeypatch, tmp_path)
    calls = _fake_docker(monkeypatch, sandbox, user=user)

    out = await _extract("here MEDIA:/workspace/test.txt")

    assert out.media_files == [(str(target.resolve()), False)]
    assert calls and all(t is not threading.main_thread() for _, t in calls), "Docker CLI ran on the event loop"
    execs = [c for c, _ in calls if c[1] == "exec"]
    if delivered:
        assert target.read_text() == "top-secret"
        assert execs and execs[0][execs[0].index("-P") + 1] == "/workspace/test.txt"
        assert b"/workspace/test.txt\0" in (sandbox / docker_ownership.HANDBACK_RECORD).read_bytes()
    else:  # a non-root sandbox's files are already the host user's: nothing is chowned
        assert not execs and not os.access(target, os.R_OK)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["docker_ps_timeout", "record_unwritable"])
async def test_failed_handback_keeps_the_reply(monkeypatch, tmp_path, failure):
    sandbox, target = _sandbox_file(monkeypatch, tmp_path)
    _fake_docker(monkeypatch, sandbox, ps_timeout=failure == "docker_ps_timeout")
    if failure == "record_unwritable":
        (sandbox / docker_ownership.HANDBACK_RECORD).mkdir()  # open("ab") raises IsADirectoryError

    out = await _extract("the report is attached MEDIA:/workspace/test.txt")

    assert out.text_content == "the report is attached"
    assert out.media_files == [(str(target.resolve()), False)]
