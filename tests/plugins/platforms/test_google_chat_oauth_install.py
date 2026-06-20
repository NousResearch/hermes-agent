from __future__ import annotations

import builtins
import subprocess
import sys


def test_install_deps_falls_back_to_uv_when_pip_is_missing(monkeypatch):
    from plugins.platforms.google_chat import oauth

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"googleapiclient", "google_auth_oauthlib"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    calls = []

    def fake_check_call(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == [sys.executable, "-m", "pip"]:
            raise subprocess.CalledProcessError(1, cmd)
        return 0

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(oauth.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(oauth.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    assert oauth.install_deps() is True
    assert calls[0][:3] == [sys.executable, "-m", "pip"]
    assert calls[1][:5] == ["/usr/bin/uv", "pip", "install", "--python", sys.executable]
