"""The Windows SSH backend spawn must carry the Desktop marker (#135784).

``spawn_backend`` is the Windows half of the Desktop SSH spawn contract from
#96490: the child binds loopback, carries per-spawn credentials on argv, and
must also set ``HERMES_DESKTOP=1`` so ``_desktop_loopback_auth_exempt`` can let
the Desktop's session-token calls through when a public ``dashboard.public_url``
engages the auth gate. The POSIX spawn (``remote-lifecycle.ts``) injects the
same marker via ``exec env HERMES_DESKTOP=1``; without it the Windows child is
"ready" but every authenticated call 401s with ``no_cookie``.
"""

import os
import sys
import types
from types import SimpleNamespace

import pytest

from hermes_cli import windows_ssh_runtime as wsr


def test_child_env_sets_the_desktop_marker(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    assert wsr._desktop_child_env()["HERMES_DESKTOP"] == "1"


def test_child_env_marker_overrides_any_inherited_value(monkeypatch):
    # Same semantics as the POSIX `exec env HERMES_DESKTOP=1`: unconditional.
    monkeypatch.setenv("HERMES_DESKTOP", "0")
    assert wsr._desktop_child_env()["HERMES_DESKTOP"] == "1"


def test_child_env_still_strips_interpreter_scoping(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/venv")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/lib")
    env = wsr._desktop_child_env()
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONPATH" not in env


def test_child_env_passes_through_the_remaining_environment(monkeypatch):
    monkeypatch.setenv("HERMES_SPAWN_PROBE", "carried")
    env = wsr._desktop_child_env()
    assert env["HERMES_SPAWN_PROBE"] == "carried"


def test_child_env_satisfies_the_loopback_exemption_gate(monkeypatch):
    """The spawned env must be exactly what the gate needs — and no more.

    Fail-closed pairing: the marker plus per-spawn credentials exempts the
    loopback bind, while a non-loopback bind stays gated even with both.
    """
    from hermes_cli import web_server

    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_DESKTOP", wsr._desktop_child_env()["HERMES_DESKTOP"])
    assert web_server._desktop_loopback_auth_exempt("127.0.0.1", ssh_owner_nonce="a" * 16)
    assert not web_server._desktop_loopback_auth_exempt("0.0.0.0", ssh_owner_nonce="a" * 16)


def test_spawn_backend_hands_the_desktop_child_env_to_the_backend(monkeypatch, tmp_path):
    """End to end: the detached Windows child's env carries the marker."""
    read_fd, write_fd = os.pipe()
    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.open_osfhandle = lambda handle, flags: write_fd
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(wsr, "_root", lambda: tmp_path)
    monkeypatch.setattr(wsr, "_ensure_scope", lambda ownership_id: tmp_path)
    monkeypatch.setattr(wsr, "_resolve_direct_command", lambda hermes_path: [str(tmp_path / "hermes.cmd")])
    fake_win32con = SimpleNamespace(GENERIC_WRITE=1, READ_CONTROL=2, CREATE_NEW=1,
                                    FILE_ATTRIBUTE_NORMAL=0, FILE_SHARE_READ=1, FILE_SHARE_WRITE=2)
    monkeypatch.setattr(wsr, "_win32", lambda: SimpleNamespace(win32con=fake_win32con))
    monkeypatch.setattr(wsr, "_open", lambda *args, **kwargs: 0)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(wsr.subprocess, "Popen", fake_popen)
    import psutil

    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 1234.5))
    try:
        result = wsr.spawn_backend({
            "ownershipId": "a" * 32,
            "spawnNonce": "0123456789abcdef",
            "hermesPath": str(tmp_path / "hermes.cmd"),
        })
    finally:
        os.close(read_fd)

    assert result["pid"] == 4321
    assert captured["env"]["HERMES_DESKTOP"] == "1"
    assert "--ssh-session-token-file" in captured["args"]
    assert "--ssh-owner-nonce" in captured["args"]
