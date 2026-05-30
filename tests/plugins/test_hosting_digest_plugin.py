from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _load_plugin():
    path = Path(__file__).resolve().parents[2] / "plugins" / "hosting-digest" / "__init__.py"
    spec = importlib.util.spec_from_file_location("hosting_digest_plugin", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Ctx:
    def __init__(self):
        self.commands = {}

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = {
            "handler": handler,
            "description": description,
            "args_hint": args_hint,
        }


def test_registers_hosting_digest_command():
    plugin = _load_plugin()
    ctx = _Ctx()

    plugin.register(ctx)

    assert "hosting-digest" in ctx.commands
    assert "hosting infrastructure cost digest" in ctx.commands["hosting-digest"]["description"]


def test_handler_runs_configured_digest_script(tmp_path, monkeypatch):
    plugin = _load_plugin()
    script = tmp_path / "digest.py"
    script.write_text('print("digest ok")\n', encoding="utf-8")
    monkeypatch.setenv("HOSTING_DIGEST_SCRIPT", str(script))

    assert plugin._handle_slash("") == "digest ok"


def test_handler_reports_nonzero_exit(tmp_path, monkeypatch):
    plugin = _load_plugin()
    script = tmp_path / "digest.py"
    script.write_text('import sys\nprint("boom")\nsys.exit(3)\n', encoding="utf-8")
    monkeypatch.setenv("HOSTING_DIGEST_SCRIPT", str(script))

    assert plugin._handle_slash("") == "hosting-digest failed: boom"


def test_handler_reports_timeout(tmp_path, monkeypatch):
    plugin = _load_plugin()
    script = tmp_path / "digest.py"
    script.write_text('print("slow")\n', encoding="utf-8")
    monkeypatch.setenv("HOSTING_DIGEST_SCRIPT", str(script))

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="digest", timeout=1)

    monkeypatch.setattr(plugin.subprocess, "run", _raise_timeout)

    assert plugin._handle_slash("") == "hosting-digest timed out after 60s."
