"""Canary: importing a shipped module must never install, update or restart anything.

2026-10-01: an import smoke over the whole tree, run under a sandboxed HOME/HERMES_HOME,
imported ``hermes_cli.psutil_android``. That shim called ``stop_for_relaunch()`` at module
level, so the import ran a full update takeover (dependency sync, launcher publication,
``finish_update`` -> ``_restart_gateway_fleet_after_update``). The launchd branch rewrote the
REAL account's ``~/Library/LaunchAgents/ai.hermes.gateway.plist`` (the plist path is resolved
from ``pwd``, not ``HOME``) to point at the scratch home and restarted the live gateway on it.

This test imports every module of every shipped package (the ``packages.find`` list in
pyproject.toml) in ONE fresh interpreter, with HOME and HERMES_HOME in a temp dir and the
service managers shadowed on PATH. An audit hook in the child records and REFUSES every
service-manager exec, every child Python interpreter (it would escape the hook) and every
write outside the temp root, so the probe itself cannot repeat the incident. Any refusal, any service-manager call, and any import that exits the process fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.platforms("posix")  # POSIX service managers and paths

_SERVICE_MANAGERS = ("launchctl", "systemctl", "loginctl", "schtasks")

_CHILD = textwrap.dedent(r'''
    import importlib, json, os, sys

    root, report = sys.argv[1], sys.argv[2]
    safe = tuple(os.path.realpath(p) for p in sys.argv[3].split(os.pathsep) if p)
    managers = set(sys.argv[4].split(","))
    modules = json.loads(sys.argv[5])
    python_names = {os.path.basename(sys.executable), os.path.basename(os.path.realpath(sys.executable))}
    violations = []
    current = ["<startup>"]

    def _outside(path):
        try:
            real = os.path.realpath(os.fsdecode(path))
        except Exception:
            return None
        if real == "/dev/null" or real.startswith(safe):
            return None
        return real

    def _hook(event, args):
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn"):
            argv = args[1] if len(args) > 1 and args[1] else [args[0]]
            try:
                exe = os.path.basename(os.fsdecode(list(argv)[0] if not isinstance(argv, (str, bytes)) else argv))
            except Exception:
                exe = ""
            if exe in managers:
                violations.append({"module": current[0], "kind": "service-manager", "detail": repr(argv)[:300]})
                raise PermissionError(f"import canary: refused {exe}")
            if exe.startswith(("python", "pythonw")) or exe in python_names:
                # A child interpreter escapes this hook (``-I`` drops PYTHONPATH), and the
                # 2026-10-01 incident ran the whole updater in exactly such a child.
                violations.append({"module": current[0], "kind": "python-child", "detail": repr(argv)[:300]})
                raise PermissionError("import canary: refused child interpreter")
        elif event == "os.system":
            cmd = os.fsdecode(args[0]) if args and isinstance(args[0], (str, bytes)) else ""
            if any(m in cmd.split() for m in managers):
                violations.append({"module": current[0], "kind": "service-manager", "detail": cmd[:300]})
                raise PermissionError("import canary: refused os.system service-manager call")
        elif event == "open":
            path, mode = args[0], args[1]
            if isinstance(path, (str, bytes, os.PathLike)) and mode and any(c in str(mode) for c in "wax+"):
                if (real := _outside(path)) is not None:
                    violations.append({"module": current[0], "kind": "write", "detail": real})
                    raise PermissionError(f"import canary: refused write to {real}")
        elif event in ("os.rename", "os.replace", "os.symlink", "os.link", "shutil.copyfile"):
            target = args[1] if len(args) > 1 else None
            if isinstance(target, (str, bytes, os.PathLike)) and (real := _outside(target)) is not None:
                violations.append({"module": current[0], "kind": event, "detail": real})
                raise PermissionError(f"import canary: refused {event} to {real}")
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate"):
            target = args[0] if args else None
            if isinstance(target, (str, bytes, os.PathLike)) and (real := _outside(target)) is not None:
                violations.append({"module": current[0], "kind": event, "detail": real})
                raise PermissionError(f"import canary: refused {event} on {real}")

    sys.path.insert(0, root)
    sys.argv = ["import-canary"]
    sys.addaudithook(_hook)
    exited = []
    for name in modules:
        current[0] = name
        try:
            importlib.import_module(name)
        except SystemExit as exc:
            exited.append({"module": name, "code": repr(exc.code)})
        except BaseException:
            pass  # optional third-party deps; this canary is about side effects, not importability
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({"violations": violations, "exited": exited, "imported": len(modules)}, fh)
''')


def _shipped_modules() -> list[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    packages = sorted({p for p in includes if not p.endswith(".*")})
    names: set[str] = set()
    for pkg in packages:
        base = ROOT / pkg.replace(".", "/")
        if not (base / "__init__.py").is_file():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(ROOT).with_suffix("")
            parts = rel.parts
            if any(p in {"tests", "__pycache__", "node_modules"} or p.startswith(".") for p in parts):
                continue
            if parts[-1] == "__main__" or not all(p.isidentifier() for p in parts):
                continue
            names.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return sorted(names)


def test_importing_every_shipped_module_has_no_service_or_out_of_home_side_effects(tmp_path):
    modules = _shipped_modules()
    assert "hermes_cli.psutil_android" in modules  # the module that caused the 2026-10-01 incident
    assert len(modules) > 100

    home = tmp_path / "home"
    hermes_home = home / ".hermes"  # the native-default shape: owns the BARE service label
    hermes_home.mkdir(parents=True)
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    stubs = tmp_path / "stubbin"
    stubs.mkdir()
    calls = tmp_path / "service-manager-calls.log"
    for name in _SERVICE_MANAGERS:
        stub = stubs / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{calls}"\nexit 1\n', encoding="utf-8")
        stub.chmod(0o755)

    report = tmp_path / "report.json"
    env = {
        "PATH": f"{stubs}{os.pathsep}/usr/bin{os.pathsep}/bin{os.pathsep}/usr/sbin{os.pathsep}/sbin",
        "HOME": str(home),
        "HERMES_HOME": str(hermes_home),
        "TMPDIR": str(tmpdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }
    safe = os.pathsep.join([str(tmp_path), str(Path("/dev"))])
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(ROOT), str(report), safe, ",".join(_SERVICE_MANAGERS),
         json.dumps(modules)],
        cwd=str(tmp_path), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900,
    )
    assert report.is_file(), f"canary child died (rc={proc.returncode}):\n{proc.stderr[-4000:]}"
    result = json.loads(report.read_text(encoding="utf-8"))
    stub_calls = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []

    problems = [f"  {e['module']}: exited the process at import (code {e['code']})" for e in result["exited"]]
    problems += [f"  {v['module']}: {v['kind']} {v['detail']}" for v in result["violations"]]
    problems += [f"  service-manager stub called: {c}" for c in stub_calls]
    assert not problems, "import-time side effects:\n" + "\n".join(problems)
