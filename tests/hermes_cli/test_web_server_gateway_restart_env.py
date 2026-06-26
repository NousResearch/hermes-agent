"""Regression test for the desktop gateway-restart button (issue #224).

``gateway/run.py`` sets ``_HERMES_GATEWAY=1`` as a module-level import side
effect. On the desktop's managed runtime the dashboard is served from inside the
gateway process, so a ``hermes gateway restart`` spawned by the dashboard via
``_spawn_hermes_action`` used to inherit that marker and trip the gateway CLI's
"Refusing to restart the gateway from inside the gateway process" guard — the
restart button did nothing. The spawn helper must strip ``_HERMES_GATEWAY`` so
the detached child is the "external shell" the guard expects.
"""

import subprocess

from hermes_cli import web_server


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 4321

    def poll(self):
        return None


def _spawn_capture(monkeypatch, tmp_path, *, extra_env=None):
    captured = {}

    def _fake_popen(cmd, **kwargs):
        proc = _FakePopen(cmd, **kwargs)
        captured["env"] = kwargs.get("env")
        return proc

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(web_server, "_ACTION_LOG_DIR", tmp_path)
    # Mark this (test) process as if it were running inside the gateway.
    monkeypatch.setenv("_HERMES_GATEWAY", "1")

    web_server._spawn_hermes_action(
        ["gateway", "restart"], "gateway-restart", extra_env=extra_env
    )
    return captured["env"]


def test_spawned_action_strips_gateway_marker(monkeypatch, tmp_path):
    env = _spawn_capture(monkeypatch, tmp_path)
    # The child must NOT look like the running gateway, or the restart guard fires.
    assert "_HERMES_GATEWAY" not in env
    # Other expected env is still present.
    assert env["HERMES_NONINTERACTIVE"] == "1"


def test_extra_env_still_applied_without_marker(monkeypatch, tmp_path):
    env = _spawn_capture(
        monkeypatch, tmp_path, extra_env={"HERMES_GATEWAY_FORCE_TAKEOVER": "1"}
    )
    assert "_HERMES_GATEWAY" not in env
    assert env["HERMES_GATEWAY_FORCE_TAKEOVER"] == "1"
    assert env["HERMES_NONINTERACTIVE"] == "1"
