"""Gateway /restart & /update argv on the PM store Python (#122620).

``gateway.run._resolve_hermes_bin`` feeds both the POSIX detached watcher
shell and the Windows watcher child. On a store-Python gateway the bare
``sys.executable -m hermes_cli.main`` form re-execs into ``ModuleNotFoundError``
(the store interpreter has no application packages; this process resolves
``hermes_cli`` only via the launcher prelude), so the resolver must return the
launcher prelude argv there. On a venv interpreter the module form keeps
winning over a PATH-planted ``hermes`` (#111569).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix", "windows")


def test_store_python_reexec_uses_launcher_prelude(tmp_path, monkeypatch):
    runtime = tmp_path / "pm-store"
    entry = runtime / "python-3.14.7+fake"
    exe = entry / ("python.exe" if os.name == "nt" else "bin/python3")
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.symlink_to(Path(sys.executable))
    (runtime / "facts.json").write_text(
        json.dumps({"packages": {"python": {"entry": "python-3.14.7+fake"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))

    from hermes_cli import _launchers

    assert _launchers.running_on_store_python() is True

    from gateway import run as gateway_run

    argv = gateway_run._resolve_hermes_bin()
    assert argv is not None
    assert argv[0] == str(exe)
    assert argv[1:3] == ["-I", "-c"]
    assert "import hermes_bootstrap" in argv[3]
    assert "runpy.run_module('hermes_cli.main'" in argv[3]

    # Without the store record (venv/dev interpreter) the module form wins
    # over a PATH-planted hermes — the #111569 property must survive.
    (Path(os.environ["HERMES_RUNTIME_DIR"]) / "facts.json").unlink()
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/tmp/planted/hermes")
    assert gateway_run._resolve_hermes_bin() == [sys.executable, "-m", "hermes_cli.main"]
