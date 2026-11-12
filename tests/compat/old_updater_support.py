"""Shared guards for historical exports; real child execution lives in test_old_updater_takeover."""

import builtins
from contextlib import contextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import urllib.request

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def no_external_work(monkeypatch):
    """Reject old-parent installs, process control, downloads and module reloads."""
    import pm

    def forbidden(*args, **kwargs):
        pytest.fail(f"old updater attempted work outside the fresh child: {args!r}")

    for name in ("ensure", "sync_venv", "ensure_environment", "build_environment",
                 "ensure_python_tool", "ensure_import"):
        monkeypatch.setattr(pm, name, forbidden)
    for name in ("Popen", "run"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("system", "kill"):
        monkeypatch.setattr(os, name, forbidden)
    monkeypatch.setattr(importlib, "reload", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    return forbidden


class FreshChild:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        # Nonzero by default: an unconditional SystemExit(0) is not a handoff.
        self.returncode = 19
        self.result = {}
        self.requests = []
        self.paths = []

    def run(self, command, *, cwd, env):
        """Intercept ONLY the fresh-child seam, leaving the real JSON bridge intact."""
        assert command[:7] == [
            sys.executable, "-I", "-S", "-B", "-X", "utf8", str(ROOT / "hermes_cli/_update_takeover.py"),
        ], f"old installer/fallback ran instead of takeover: {command!r}"
        assert len(command) == 9
        assert Path(command[0]).is_absolute()
        assert Path(cwd) == ROOT
        assert not any(key.startswith(("PYTHON", "UV_")) or key == "VIRTUAL_ENV" for key in env)
        context, result = map(Path, command[7:])
        assert context.is_absolute() and result.is_absolute()
        assert context.parent == result.parent and context != result
        request = json.loads(context.read_text(encoding="utf-8"))
        assert request["root"] == str(ROOT)
        assert request["home"] == os.environ["HERMES_HOME"]
        assert request["argv"] == sys.argv
        self.requests.append(request)
        self.paths.append((context, result))
        result.write_text(json.dumps(self.result), encoding="utf-8")
        return subprocess.CompletedProcess(command, self.returncode)

    @contextmanager
    def exits(self):
        """Each independent probe must actually hand off, not reuse another test's result."""
        before_calls = len(self.requests)
        before_env = dict(os.environ)
        before_modules = dict(sys.modules)
        home = Path(os.environ["HERMES_HOME"])
        before_files = {p: p.read_bytes() for p in home.rglob("*") if p.is_file()}
        real_import = builtins.__import__
        real_import_module = importlib.import_module

        def check_import(name):
            if name == "pm" or name.startswith("pm.") or name in {
                "hermes_cli._update_takeover", "hermes_cli.update_finish",
            }:
                pytest.fail(f"fresh updater imported in the old parent: {name}")

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            resolved = importlib.util.resolve_name("." * level + name, globals["__package__"]) if level else name
            check_import(resolved)
            for attr in fromlist or ():
                check_import(f"{resolved}.{attr}")
            return real_import(name, globals, locals, fromlist, level)

        def guarded_import_module(name, package=None):
            check_import(importlib.util.resolve_name(name, package) if name.startswith(".") else name)
            return real_import_module(name, package)

        with self.monkeypatch.context() as guard:
            guard.setattr(builtins, "__import__", guarded_import)
            guard.setattr(importlib, "import_module", guarded_import_module)
            with pytest.raises(SystemExit) as stopped:
                yield
        assert stopped.value.code == self.returncode
        assert len(self.requests) == before_calls + 1
        assert all(not path.exists() for path in self.paths[-1])
        assert dict(os.environ) == before_env
        assert all(sys.modules.get(name) is module for name, module in before_modules.items())
        assert {p: p.read_bytes() for p in home.rglob("*") if p.is_file()} == before_files


@pytest.fixture
def fresh_child(monkeypatch, no_external_work):
    from hermes_cli import _old_updater

    # Production caches across finally/atexit reentry. Test probes model separate
    # historical processes, so they must not inherit the previous probe's status.
    monkeypatch.setattr(_old_updater, "_result", None)
    child = FreshChild(monkeypatch)
    monkeypatch.setattr(subprocess, "run", child.run)
    return child
