"""The display bridge must exec the RFB relay in the ROUTED profile's sandbox.

A multiplexed ``hermes serve`` launches with the default profile's terminal (local) bridged into
the process env; a docker-persistent secondary profile's Bot Screen is registered as
``profile:<name>``. ``_open_rfb`` set the home override but never bound the routed profile's
terminal scope, so ``_resolve_container_task_id`` read the LAUNCH profile's TERMINAL_* and keyed
the sandbox as ``home:<path>`` — the registry lookup missed and the relay died with
``LocalEnvironment cannot exec argv`` instead of streaming (#140892).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from hermes_cli.web_routers.display import _open_rfb


class _FakeDocker:
    """Only what the RFB relay path reads off a registered DockerEnvironment."""

    _docker_exe = "docker"
    _container_id = "c0ffee"

    def get_temp_dir(self):
        return "/tmp"  # no-tmp: ok — sandbox-side path


class _PipeRelay:
    """Stand-in Popen with REAL pipe fds so ``_open_rfb`` can wire its asyncio pumps."""

    def __init__(self):
        read_out, self._write_out = os.pipe()
        self._read_in, write_in = os.pipe()
        self.stdout = os.fdopen(read_out, "rb", buffering=0)
        self.stdin = os.fdopen(write_in, "wb", buffering=0)

    def close_all(self):
        for stream in (self.stdout, self.stdin):
            try:
                stream.close()
            except OSError:
                pass
        for fd in (self._write_out, self._read_in):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.fixture
def multiplexed_serve(tmp_path, monkeypatch):
    """A routed docker-persistent profile served by a local-terminal launch profile."""
    launch_home = tmp_path / "launch-home"
    profile_home = tmp_path / "serve-root" / "profiles" / "assistant"
    (profile_home / "bot-desktop").mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "terminal:\n  backend: docker\n  container_persistent: true\n", encoding="utf-8")
    (profile_home / "bot-desktop" / "sandbox.json").write_text(
        json.dumps({"backend": "DockerEnvironment"}), encoding="utf-8")
    # The launch (default) profile runs local: no TERMINAL_* is bridged into the process env.
    for name in ("TERMINAL_ENV", "TERMINAL_CONTAINER_PERSISTENT"):
        monkeypatch.delenv(name, raising=False)
    import hermes_constants as hc

    monkeypatch.setattr(hc, "get_routing_process_hermes_home", lambda: launch_home)
    return profile_home


def test_open_rfb_execs_in_the_registered_profile_sandbox(tmp_path, monkeypatch, multiplexed_serve):
    profile_home = multiplexed_serve
    from tools import terminal_tool as tt
    from tools.bot_desktop import sandbox_host
    from tools.environments import streams

    env = _FakeDocker()
    monkeypatch.setitem(tt._active_environments, "profile:assistant", env)
    monkeypatch.setattr(sandbox_host, "_user_for", lambda e: "pn")
    spawned = {}
    relay = _PipeRelay()

    def _fake_open_stream(env_arg, argv, *, child_env=None, user=None, stderr=None):
        spawned["env"] = env_arg
        spawned["argv"] = list(argv)
        spawned["user"] = user
        return relay

    monkeypatch.setattr(streams, "open_stream", _fake_open_stream)
    try:
        _reader, _writer, returned_relay = asyncio.run(_open_rfb(profile_home))
    finally:
        relay.close_all()
    assert returned_relay is relay
    # The relay execs the environment the gateway registered for THIS profile — not whatever the
    # launch profile's ambient TERMINAL_* would key (a LocalEnvironment or nothing at all).
    assert spawned["env"] is env
    assert spawned["user"] == "pn"
    assert any("rfb.sock" in part for part in spawned["argv"])


def test_open_rfb_names_the_unresolvable_policy_as_an_os_error(tmp_path, monkeypatch, multiplexed_serve):
    profile_home = multiplexed_serve
    (profile_home / "config.yaml").write_text("terminal: [unbalanced", encoding="utf-8")
    with pytest.raises(OSError, match="terminal policy unavailable"):
        asyncio.run(_open_rfb(profile_home))
