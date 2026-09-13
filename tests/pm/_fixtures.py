"""Real artifacts and isolated PM workers shared by lifecycle tests."""
from __future__ import annotations

from functools import partial
import hashlib
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import importlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import zipfile

import pytest


def _wheel(directory: Path, name: str, version: str = "1.0", requirements=()) -> Path:
    metadata = f"{name}-{version}.dist-info"
    entries = {
        f"{name}/__init__.py": f"__version__ = {version!r}\n",
        f"{metadata}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements),
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    entries[f"{metadata}/RECORD"] = "".join(f"{path},,\n" for path in entries)
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, body in entries.items():
            archive.writestr(path, body)
    return wheel


def _run(command, *, cwd: Path, env: dict) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def make_tar(docroot: Path, name: str, files: dict[str, str]) -> tuple[str, str]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    payload = buf.getvalue()
    (docroot / name).write_bytes(payload)
    return name, hashlib.sha256(payload).hexdigest()


@pytest.fixture
def served(tmp_path):
    docroot = tmp_path / "www"
    docroot.mkdir()
    handler = partial(SimpleHTTPRequestHandler, directory=str(docroot))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield docroot, f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.fixture(scope="module")
def isolated_python(tmp_path_factory):
    from pm.runtime_stage import stage_runtime

    root = tmp_path_factory.mktemp("pm-python")
    uv = shutil.which("uv")
    assert uv, "the worker contract requires real uv"
    python = stage_runtime(Path(uv), Path(sys.executable), root)
    _run([str(python), "-I", "-c", "import importlib.util; assert importlib.util.find_spec('yaml') is None"],
         cwd=root, env=dict(os.environ))
    return python


@pytest.fixture
def client(tmp_path, monkeypatch, isolated_python):
    from pm import paths

    client = importlib.import_module("pm.client")
    monkeypatch.setattr("pm.runtime.runtime_python", lambda **kwargs: isolated_python)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    return client


@pytest.fixture
def build_worker(client, isolated_python, monkeypatch):
    """Inject prepared real tools only inside the independent worker process."""
    uv = shutil.which("uv")
    assert uv, "the worker contract requires real uv"
    worker = Path(client.__file__).with_name("worker.py")
    script = (
        "import runpy, sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(worker.parent.parent)!r}); "
        "import pm._uv; "
        f"pm._uv._toolchain = lambda **kwargs: (Path({uv!r}), Path({sys.executable!r})); "
        f"runpy.run_path({str(worker)!r}, run_name='__main__')"
    )
    monkeypatch.setattr(client, "runtime_command", lambda path, **kwargs: [str(isolated_python), "-I", "-B", "-c", script])
    monkeypatch.setattr(client, "is_runtime", lambda: False)

    def caller_engine(*args, **kwargs):
        pytest.fail("dependency engine ran in the caller, not the PM worker")

    for module, names in {
        "pm.operations": ("build_environment", "lock_project"),
        "pm.build_operations": ("build_requirements_environment", "export_requirements", "check_project_lock"),
    }.items():
        for name in names:
            monkeypatch.setattr(importlib.import_module(module), name, caller_engine)
    return client