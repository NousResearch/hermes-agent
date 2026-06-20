from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace


def test_ensure_sdk_installed_falls_back_to_uv_when_pip_is_missing(monkeypatch):
    from plugins.memory.honcho import cli

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "honcho":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == [sys.executable, "-m", "pip"]:
            return SimpleNamespace(returncode=1, stderr="No module named pip")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli, "_prompt", lambda *args, **kwargs: "y")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    assert cli._ensure_sdk_installed() is True
    assert calls[0][:3] == [sys.executable, "-m", "pip"]
    assert calls[1][:5] == ["/usr/bin/uv", "pip", "install", "--python", sys.executable]
