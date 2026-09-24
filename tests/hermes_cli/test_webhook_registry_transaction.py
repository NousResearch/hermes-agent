"""Registry writers share the plugin's persistent flock, not merely an atomic rename."""

import asyncio
import json
import os
import stat
import subprocess
import sys
import threading
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli import webhook as wh
from hermes_cli.web_routers import ops
from hermes_cli.web_models import WebhookCreate, WebhookEnabledToggle


def _cli(action, name, **kw):
    args = dict(name=name, webhook_action=action, secret="", events="", description="",
                prompt="", skills="", deliver="log", deliver_chat_id="", route_profile=None,
                script="")
    args.update(kw)
    wh.webhook_command(Namespace(**args))


@pytest.mark.linux_only
def test_plugin_lock_serializes_cli_and_dashboard_mutations(tmp_path, monkeypatch):
    """A plugin holding flock may safely replace the registry before each native writer reads it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    path.write_text(json.dumps({"base": {"secret": "original", "profile": "default"}}))
    actions = [
        lambda: _cli("subscribe", "base"),
        lambda: asyncio.run(ops.create_webhook(WebhookCreate(name="dashboard"))),
        lambda: asyncio.run(ops.set_webhook_enabled("dashboard", WebhookEnabledToggle(enabled=False))),
        lambda: asyncio.run(ops.delete_webhook("dashboard")),
        lambda: _cli("remove", "base"),
    ]
    subs = {}
    for index, action in enumerate(actions):
        # This subprocess emulates the plugin's existing sibling-lock protocol.
        child = subprocess.Popen([sys.executable, "-u", "-c", '''
import fcntl, json, os, sys
from pathlib import Path
p = Path(sys.argv[1]); lock = p.with_name(p.name + ".lock")
fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("LOCKED", flush=True)
sys.stdin.readline()
subs = json.loads(p.read_text())
subs["plugin_%s" % sys.argv[2]] = {"secret": "plugin"}
t = p.with_name("plugin.tmp"); t.write_text(json.dumps(subs)); os.replace(t, p)
fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
''' , str(path), str(index)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 text=True, env={**os.environ, "HERMES_HOME": str(tmp_path)})
        try:
            assert child.stdout.readline().strip() == "LOCKED"
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(action)
                # Release before asserting: executor shutdown must not strand a waiter.
                threading.Event().wait(0.15)
                completed_while_locked = future.done()
                child.stdin.write("go\n"); child.stdin.flush()
                future.result(timeout=10)
            assert not completed_while_locked, "native mutation ignored plugin's lock"
            assert child.wait(timeout=10) == 0
            subs = json.loads(path.read_text())
            assert all(f"plugin_{n}" in subs for n in range(index + 1))
            if index == 0:
                assert subs["base"]["secret"] == "original"
                assert subs["base"]["profile"] == "default"
        finally:
            if child.poll() is None:
                child.stdin.write("go\n"); child.stdin.flush()
                child.wait(timeout=10)
    assert "base" not in subs
    assert "dashboard" not in subs
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.with_name(path.name + ".lock").stat().st_mode) == 0o600


@pytest.mark.parametrize("bad", ["{broken", "[]", "null"])
def test_mutations_fail_closed_on_corrupt_registry(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    path.write_text(bad)
    actions = [lambda: _cli("subscribe", "new"), lambda: _cli("remove", "old"),
               lambda: asyncio.run(ops.create_webhook(WebhookCreate(name="new"))),
               lambda: asyncio.run(ops.delete_webhook("old")),
               lambda: asyncio.run(ops.set_webhook_enabled("old", WebhookEnabledToggle(enabled=False)))]
    for action in actions:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            action()
        assert path.read_text() == bad


def test_dashboard_profile_a_b_a_registry_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    for name in ("a", "b"):
        profile = tmp_path / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n")
    for profile in ("a", "b", "a"):
        name = "route_a" if profile == "a" else "route_b"
        asyncio.run(ops.create_webhook(WebhookCreate(name=name), profile=profile))
    a = json.loads((tmp_path / "profiles/a/webhook_subscriptions.json").read_text())
    b = json.loads((tmp_path / "profiles/b/webhook_subscriptions.json").read_text())
    assert set(a) == {"route_a"} and set(b) == {"route_b"}
    assert not (tmp_path / "webhook_subscriptions.json").exists()
    asyncio.run(ops.set_webhook_enabled("route_a", WebhookEnabledToggle(enabled=False), profile="a"))
    assert json.loads((tmp_path / "profiles/b/webhook_subscriptions.json").read_text()) == b
    assert json.loads((tmp_path / "profiles/a/webhook_subscriptions.json").read_text())["route_a"]["enabled"] is False
