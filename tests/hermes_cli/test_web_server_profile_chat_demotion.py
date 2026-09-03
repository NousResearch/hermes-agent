"""Root-run dashboard chats must drop to the profile dir's owner (#94847).

A machine-level dashboard running as root, opening Chat for a profile that
belongs to a different OS user, spawned the chat child as root with
HERMES_HOME=<that profile dir> — ordinary UI writes then created root-owned
state (projects.db, notices) that the profile's own gateway could no longer
read. The child must run as the profile dir's owner; a drop that cannot be
computed must refuse the chat instead of silently spawning root.
"""
from __future__ import annotations

import os
import types

import pytest

import hermes_cli.web_server as ws_mod


class _Stat:
    def __init__(self, uid, gid):
        self.st_uid = uid
        self.st_gid = gid


@pytest.fixture
def profile_dir(tmp_path, monkeypatch):
    d = tmp_path / "echo-profile"
    d.mkdir()
    monkeypatch.setattr(ws_mod, "_resolve_profile_dir", lambda name: d)
    return d


def _as_root(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)


class TestProfileChatDemotion:
    def test_non_root_server_never_demotes(self, profile_dir, monkeypatch):
        monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
        preexec, overrides, err = ws_mod._profile_chat_demotion("echo")
        assert preexec is None and overrides is None and err is None

    def test_no_profile_or_current_profile_no_demotion(self, profile_dir, monkeypatch):
        _as_root(monkeypatch)
        for name in ("", None, "current", "CURRENT"):
            preexec, overrides, err = ws_mod._profile_chat_demotion(name)
            assert preexec is None and err is None, name

    def test_root_owned_profile_dir_no_demotion(self, profile_dir, monkeypatch):
        _as_root(monkeypatch)
        monkeypatch.setattr(os, "stat", lambda p: _Stat(0, 0))
        preexec, overrides, err = ws_mod._profile_chat_demotion("echo")
        assert preexec is None and overrides is None and err is None

    def test_non_root_owned_dir_demotes_to_owner(self, profile_dir, monkeypatch):
        _as_root(monkeypatch)
        monkeypatch.setattr(os, "stat", lambda p: _Stat(985, 300))
        monkeypatch.setattr(
            ws_mod.pwd if hasattr(ws_mod, "pwd") else __import__("pwd"),
            "getpwuid",
            lambda uid: types.SimpleNamespace(
                pw_gid=300, pw_dir="/home/hermes-echo", pw_name="hermes-echo"
            ),
            raising=False,
        )
        preexec, overrides, err = ws_mod._profile_chat_demotion("echo")
        assert err is None
        assert preexec is not None
        assert overrides == {
            "HOME": "/home/hermes-echo",
            "USER": "hermes-echo",
            "LOGNAME": "hermes-echo",
            # Root's session handles are marked for stripping (None), not
            # inherited — the demoted child must not reach root's
            # ssh-agent or runtime bus.
            "SSH_AUTH_SOCK": None,
            "SSH_AGENT_PID": None,
            "XDG_RUNTIME_DIR": None,
            "DBUS_SESSION_BUS_ADDRESS": None,
        }
        # The preexec performs the ordered drop: initgroups → gid → uid.
        import hermes_cli.web_server as fresh_ws

        calls = []
        monkeypatch.setattr(os, "initgroups", lambda u, g: calls.append(("initgroups", u, g)), raising=False)
        monkeypatch.setattr(os, "setgroups", lambda g: calls.append(("groups", g)))
        monkeypatch.setattr(os, "setgid", lambda g: calls.append(("gid", g)))
        monkeypatch.setattr(os, "setuid", lambda u: calls.append(("uid", u)))
        # Re-resolve so the closure binds the patched os functions.
        preexec, _, _ = fresh_ws._profile_chat_demotion("echo")
        preexec()
        assert calls == [("initgroups", "hermes-echo", 300), ("gid", 300), ("uid", 985)]

    def test_failed_supplementary_group_drop_kills_child(
        self, profile_dir, monkeypatch
    ):
        # A swallowed setgroups failure would leave root's supplementary
        # groups attached to the demoted child — the OSError must propagate
        # so ptyprocess terminates the forked child before the target
        # program starts (#94847 fail-closed).
        _as_root(monkeypatch)
        monkeypatch.setattr(os, "stat", lambda p: _Stat(985, 300))
        import pwd

        monkeypatch.setattr(pwd, "getpwuid", lambda uid: (_ for _ in ()).throw(KeyError(uid)))

        def _boom(groups):
            raise PermissionError("setgroups denied")

        import hermes_cli.web_server as fresh_ws

        monkeypatch.setattr(os, "setgroups", _boom)
        monkeypatch.setattr(os, "setgid", lambda g: None)
        monkeypatch.setattr(os, "setuid", lambda u: None)
        preexec, _, _ = fresh_ws._profile_chat_demotion("echo")
        with pytest.raises(OSError):
            preexec()

    def test_owner_missing_from_passwd_still_drops_by_dir_ids(
        self, profile_dir, monkeypatch
    ):
        _as_root(monkeypatch)
        monkeypatch.setattr(os, "stat", lambda p: _Stat(985, 300))
        import pwd

        monkeypatch.setattr(pwd, "getpwuid", lambda uid: (_ for _ in ()).throw(KeyError(uid)))
        preexec, overrides, err = ws_mod._profile_chat_demotion("echo")
        assert err is None
        assert preexec is not None
        # no HOME/USER data — ids alone still drop, and root's session
        # handles are still stripped.
        assert overrides == {
            "SSH_AUTH_SOCK": None,
            "SSH_AGENT_PID": None,
            "XDG_RUNTIME_DIR": None,
            "DBUS_SESSION_BUS_ADDRESS": None,
        }

    def test_stat_failure_fails_closed_with_error(self, profile_dir, monkeypatch):
        _as_root(monkeypatch)

        def _boom(p):
            raise PermissionError("denied")

        monkeypatch.setattr(os, "stat", _boom)
        preexec, overrides, err = ws_mod._profile_chat_demotion("echo")
        assert preexec is None and overrides is None
        assert err is not None and "cannot inspect profile dir" in err

    def test_pty_spawn_forwards_preexec_fn(self, monkeypatch):
        from hermes_cli import pty_bridge

        assert pty_bridge.ptyprocess is not None, "ptyprocess must be installed"
        captured = {}

        class _FakeProc:
            pid = 4242
            fd = 99

        def fake_spawn(argv, cwd=None, env=None, preexec_fn=None, dimensions=None, **kw):
            captured["preexec"] = preexec_fn
            return _FakeProc()

        monkeypatch.setattr(
            pty_bridge.ptyprocess.PtyProcess, "spawn", staticmethod(fake_spawn)
        )
        marker = lambda: None
        bridge = pty_bridge.PtyBridge.spawn(["sh"], preexec_fn=marker)
        assert captured["preexec"] is marker
        assert bridge.pid == 4242
