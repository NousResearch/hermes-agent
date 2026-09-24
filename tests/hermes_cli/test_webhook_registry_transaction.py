"""Registry writers share the plugin's persistent flock, not merely an atomic rename."""

import asyncio
import errno
import json
import os
import stat
import subprocess
import sys
import threading
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException

from hermes_cli import webhook as wh
from hermes_cli.web_routers import ops
from hermes_cli.web_models import WebhookCreate, WebhookEnabledToggle


def _cli(action, name, **kw):
    args = dict(name=name, webhook_action=action, secret="", events="", description="",
                prompt="", skills="", deliver="log", deliver_chat_id="", route_profile=None,
                script="")
    args.update(kw)
    return wh.webhook_command(Namespace(**args))


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
fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
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
            started = threading.Event()
            def run_action():
                started.set()
                action()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_action)
                assert started.wait(timeout=10)
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
def test_mutations_fail_closed_on_corrupt_registry(tmp_path, monkeypatch, bad, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    path.write_text(bad)
    for action in (lambda: _cli("subscribe", "new"), lambda: _cli("remove", "old")):
        action()
        assert "Error: Could not update webhook subscriptions" in capsys.readouterr().out
        assert path.read_text() == bad
    actions = [lambda: asyncio.run(ops.create_webhook(WebhookCreate(name="new"))),
               lambda: asyncio.run(ops.delete_webhook("old")),
               lambda: asyncio.run(ops.set_webhook_enabled("old", WebhookEnabledToggle(enabled=False)))]
    for action in actions:
        with pytest.raises(HTTPException) as error:
            action()
        assert error.value.status_code == 409
        assert "registry" in error.value.detail
        assert path.read_text() == bad

@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW unavailable")
def test_planted_lock_symlink_does_not_chmod_target_or_write_registry(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    path.write_text('{"keep":{"secret":"safe"}}')
    target = tmp_path / "unrelated"
    target.write_text("untouched")
    target.chmod(0o644)
    path.with_name(path.name + ".lock").symlink_to(target)
    _cli("subscribe", "new")
    assert "Error: Could not update webhook subscriptions" in capsys.readouterr().out
    with pytest.raises(HTTPException) as error:
        asyncio.run(ops.create_webhook(WebhookCreate(name="new")))
    assert error.value.status_code == 500
    assert target.read_text() == "untouched"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert json.loads(path.read_text()) == {"keep": {"secret": "safe"}}


@pytest.mark.skipif(os.name == "nt", reason="POSIX hardlink and fchmod regression")
def test_planted_lock_hardlink_does_not_chmod_target_or_write_registry(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    path.write_text('{"keep":{"secret":"safe"}}')
    unrelated = tmp_path / "unrelated"
    unrelated.write_text("untouched")
    unrelated.chmod(0o644)
    lock = path.with_name(path.name + ".lock")
    os.link(unrelated, lock)
    before = unrelated.stat()
    assert before.st_ino == lock.stat().st_ino and before.st_nlink == 2
    _cli("subscribe", "new")
    assert "singly-linked regular file" in capsys.readouterr().out
    with pytest.raises(HTTPException) as error:
        asyncio.run(ops.create_webhook(WebhookCreate(name="new")))
    assert error.value.status_code == 409
    after = unrelated.stat()
    assert (after.st_dev, after.st_ino, after.st_nlink) == (before.st_dev, before.st_ino, 2)
    assert (lock.stat().st_ino, lock.stat().st_nlink) == (before.st_ino, 2)
    assert stat.S_IMODE(after.st_mode) == 0o644
    assert unrelated.read_text() == "untouched"
    assert json.loads(path.read_text()) == {"keep": {"secret": "safe"}}


def test_dashboard_overwrite_reports_persisted_enabled_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    path.write_text(json.dumps({"route": {"enabled": False, "secret": "old", "custom": 1}}))
    result = asyncio.run(ops.create_webhook(WebhookCreate(name="route", secret="new")))
    stored = json.loads(path.read_text())["route"]
    assert stored["enabled"] is False and result["enabled"] is False
    assert stored["secret"] == result["secret"] == "new"
    assert stored["custom"] == 1
    asyncio.run(ops.set_webhook_enabled("route", WebhookEnabledToggle(enabled=True)))
    result = asyncio.run(ops.create_webhook(WebhookCreate(name="route")))
    assert result["enabled"] is True
    assert json.loads(path.read_text())["route"]["enabled"] is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync")
def test_transaction_surfaces_directory_fsync_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    real_fsync = os.fsync
    def fail_directory_sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "directory fsync failed")
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", fail_directory_sync)
    result = asyncio.run(ops.create_webhook(WebhookCreate(name="route")))
    assert result["durable"] is False and "warning" in result
    assert result["secret"] == json.loads(wh._subscriptions_path().read_text())["route"]["secret"]
    assert asyncio.run(ops.list_webhooks())["subscriptions"][0]["secret_set"] is True
    updated = asyncio.run(ops.create_webhook(WebhookCreate(name="route", secret="rotated")))
    assert updated["durable"] is False and updated["secret"] == "rotated"
    assert json.loads(wh._subscriptions_path().read_text())["route"]["secret"] == "rotated"
    assert _cli("subscribe", "cli", secret="cli-key") is None
    output = capsys.readouterr().out
    assert "Secret: cli-key" in output and "WARNING" in output
    assert json.loads(wh._subscriptions_path().read_text())["cli"]["secret"] == "cli-key"
    toggled = asyncio.run(ops.set_webhook_enabled("route", WebhookEnabledToggle(enabled=False)))
    assert toggled["durable"] is False and toggled["enabled"] is False
    assert json.loads(wh._subscriptions_path().read_text())["route"]["enabled"] is False
    deleted = asyncio.run(ops.delete_webhook("route"))
    assert deleted["durable"] is False and "route" not in json.loads(wh._subscriptions_path().read_text())
    _cli("remove", "cli")
    assert "WARNING" in capsys.readouterr().out
    assert "cli" not in json.loads(wh._subscriptions_path().read_text())


def test_webhook_main_forwards_mutation_exit_status(monkeypatch):
    from hermes_cli import main
    monkeypatch.setattr(wh, "webhook_command", lambda _args: 1)
    assert main.cmd_webhook(Namespace()) == 1
    monkeypatch.setattr(wh, "webhook_command", lambda _args: 0)
    assert main.cmd_webhook(Namespace()) == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX file fsync")
def test_busy_registry_replace_never_rewrites_target_or_claims_durability(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    path.write_text('{"kept": {"secret": "old"}}')
    path.chmod(0o600)
    before = path.read_bytes()
    inode = path.stat().st_ino
    real_replace, real_fsync = os.replace, os.fsync
    target_sync_attempts = []

    def busy_registry_replace(src, dst):
        if os.fspath(dst) == os.fspath(path):
            raise OSError(errno.EBUSY, "registry bind mount is busy")
        return real_replace(src, dst)

    def fail_target_sync(fd):
        if os.fstat(fd).st_ino == inode:
            target_sync_attempts.append(fd)
            raise OSError(errno.EIO, "target file fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(os, "replace", busy_registry_replace)
    monkeypatch.setattr(os, "fsync", fail_target_sync)
    with pytest.raises(HTTPException) as error:
        asyncio.run(ops.create_webhook(WebhookCreate(name="new", secret="not-published")))
    assert error.value.status_code == 500
    assert "not-published" not in str(error.value.detail)
    assert wh.webhook_command(Namespace(name="cli", webhook_action="subscribe", secret="not-published",
                                        events="", description="", prompt="", skills="", deliver="log",
                                        deliver_chat_id="", route_profile=None, script="")) == 1
    output = capsys.readouterr().out
    assert "not-published" not in output and "WARNING" not in output
    assert path.read_bytes() == before and path.stat().st_ino == inode
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert target_sync_attempts == []  # no in-place copy or suppressed target EIO
    assert not list(tmp_path.glob(".webhook_subscriptions_*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX file fsync")
def test_prepublication_fsync_eio_does_not_publish_or_disclose_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    path.write_text('{"kept": {"secret": "old"}}')
    before = path.read_bytes()
    real_fsync = os.fsync
    def fail_file_sync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "file fsync failed")
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", fail_file_sync)
    with pytest.raises(HTTPException) as error:
        asyncio.run(ops.create_webhook(WebhookCreate(name="new", secret="not-written")))
    assert error.value.status_code == 500
    assert "not-written" not in str(error.value.detail)
    assert wh.webhook_command(Namespace(name="cli", webhook_action="subscribe", secret="not-written",
                                        events="", description="", prompt="", skills="", deliver="log",
                                        deliver_chat_id="", route_profile=None, script="")) == 1
    assert "not-written" not in capsys.readouterr().out
    assert path.read_bytes() == before
    assert "new" not in json.loads(path.read_text())


@pytest.mark.parametrize("bad", ["{broken", "[]", '{"bad": "not-a-route"}',
                                      '{"bad": {"events": 23}}',
                                      '{"bad": {"skills": "code"}}',
                                      '{"bad": {"events": [23]}}'])
def test_administrative_reads_report_malformed_registry(tmp_path, monkeypatch, capsys, bad):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    path.write_text(bad)
    with pytest.raises(HTTPException) as error:
        asyncio.run(ops.list_webhooks())
    assert error.value.status_code == 409
    _cli("list", "")
    assert "Error: Could not read webhook subscriptions" in capsys.readouterr().out
    assert path.read_text() == bad

def test_malformed_target_refused_without_clobbering_other_routes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    initial = {"bad": "bad-route", "good": {"secret": "old", "custom": {"x": 1},
                                            "script": "/old/script", "deliver_only": True}}
    path.write_text(json.dumps(initial))
    for action in (lambda: _cli("subscribe", "bad"), lambda: _cli("remove", "bad")):
        action()
        assert "not an object" in capsys.readouterr().out
        assert json.loads(path.read_text()) == initial
    for action in (lambda: ops.create_webhook(WebhookCreate(name="bad")),
                   lambda: ops.delete_webhook("bad"),
                   lambda: ops.set_webhook_enabled("bad", WebhookEnabledToggle(enabled=False))):
        with pytest.raises(HTTPException) as error:
            asyncio.run(action())
        assert error.value.status_code == 409
        assert "not an object" in error.value.detail
        assert json.loads(path.read_text()) == initial
    _cli("subscribe", "good")
    updated = json.loads(path.read_text())["good"]
    assert updated["custom"] == {"x": 1}
    assert "script" not in updated and "deliver_only" not in updated
    asyncio.run(ops.create_webhook(WebhookCreate(name="good")))
    result = json.loads(path.read_text())
    assert result["good"]["custom"] == {"x": 1}
    assert result["bad"] == "bad-route"

def test_absent_cli_remove_does_not_replace_registry_inode(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wh, "_is_webhook_enabled", lambda: True)
    path = wh._subscriptions_path()
    contents = '{"kept": {"secret": "old"}}\n'
    path.write_text(contents)
    inode = path.stat().st_ino
    _cli("remove", "missing")
    assert "No subscription named 'missing'" in capsys.readouterr().out
    assert path.stat().st_ino == inode
    assert path.read_text() == contents
    with pytest.raises(HTTPException) as error:
        asyncio.run(ops.delete_webhook("missing"))
    assert error.value.status_code == 404
    assert path.stat().st_ino == inode
    assert path.read_text() == contents


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
