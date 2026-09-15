"""PM publication plus the real backend, using prepared dependencies offline.

Only acquisition is substituted: real uv/Python come from the test host.
A local wheel exposes its prepared dependency closure through a load-bearing
.pth. PM's worker creates and selects the generation and the real publisher
writes the launcher. No production dependency install or fake server is used.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main() -> None:
    repository = Path(__file__).resolve().parents[4]
    temp = Path(sys.argv[1])
    root = Path(os.environ["HERMES_HOME"]) / "hermes-agent"
    root.mkdir(parents=True)
    for name in ("hermes_cli", "pm", "agent", "tools", "gateway", "tui_gateway", "cron", "plugins"):
        shutil.copytree(repository / name, root / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for source in repository.glob("*.py"):
        shutil.copy2(source, root / source.name)
    sys.path.insert(0, str(root))
    uv = shutil.which("uv")
    assert uv, "real uv must be prepared on PATH"
    sites = [p for p in sys.path if p.endswith("site-packages")]
    assert sites, "run with the prepared Hermes Python dependency environment"
    wheels = temp / "wheels"
    wheels.mkdir()
    dist = "desktop_backend_deps-1.dist-info"
    entries = {
        "desktop_backend_deps.pth": "\n".join(sites) + "\n",
        "desktop_backend_proof.py": "VALUE = 'selected by PM'\n",
        f"{dist}/METADATA": "Metadata-Version: 2.1\nName: desktop-backend-deps\nVersion: 1\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    entries[f"{dist}/RECORD"] = "".join(f"{name},,\n" for name in entries)
    with zipfile.ZipFile(wheels / "desktop_backend_deps-1-py3-none-any.whl", "w") as wheel:
        for name, content in entries.items():
            wheel.writestr(name, content)
    (root / "pyproject.toml").write_text(
        '[project]\nname="desktop-startup-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["desktop-backend-deps==1"]\n'
        '[project.optional-dependencies]\nall=[]\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8")
    subprocess.run([uv, "lock", "--project", str(root), "--python", sys.executable],
                   check=True, stdout=sys.stderr, timeout=30)
    import pm
    from pm import paths
    from pm.lock import Facts
    from pm.store import current_target, tree_digest
    from hermes_cli.runtime_paths import selected_venv
    from hermes_cli._launchers import ensure_install_launchers

    worker = root / "pm" / "worker.py"
    worker_code = (
        "import runpy, sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(root)!r}); import pm._uv; "
        f"pm._uv._toolchain = lambda **kwargs: (Path({uv!r}), Path({sys.executable!r})); "
        f"runpy.run_path({str(worker)!r}, run_name='__main__')"
    )
    client = importlib.import_module("pm.client")
    setattr(client, "runtime_command", lambda *args, **kwargs: [sys.executable, "-I", "-c", worker_code])
    pm.sync_venv(explicit=True, project_root=root)
    selected = selected_venv(root)
    assert selected and selected.is_relative_to(Path(os.environ["HERMES_HOME"]))
    store = paths.store_root()
    entry = store / "python-fixture"
    python = entry / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.symlink_to(getattr(sys, "_base_executable", sys.executable))
    Facts(paths.facts_path()).record("python", "fixture", entry.name, {"PATH": [str(python.parent)]},
                                   store, target=current_target(), digest=tree_digest(entry))
    launchers = ensure_install_launchers(root, root / ".hermes" / "bin")
    assert launchers
    (root / "desktop_launch_probe.py").write_text(
        "import desktop_backend_proof, json, sys\n"
        "print(json.dumps({'python': sys.executable, 'module': desktop_backend_proof.__file__, "
        "'value': desktop_backend_proof.VALUE}))\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "launcher": str(root / ".hermes/bin/hermes"),
                      "python": str(python), "selected": str(selected)}))


if __name__ == "__main__":
    main()