"""The native bundle pipeline publishes only after a real staged sync succeeds."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from scripts.bundles import native


def test_bundle_stages_git_tree_and_runs_native_children_before_manifest(tmp_path, monkeypatch):
    from hermes_cli.runtime_paths import site_packages

    output = tmp_path / "payload"
    target_python = output / "staged-python" / ("python.exe" if os.name == "nt" else "bin/python")
    target_python.parent.mkdir(parents=True)
    # PM seals a payload-owned base interpreter, not an external venv launcher.
    # The POSIX host supplies its stdlib; Windows needs it beside the executable.
    if os.name == "nt":
        shutil.copytree(Path(sys.base_prefix), target_python.parent, dirs_exist_ok=True)
    else:
        shutil.copy2(Path(getattr(sys, "_base_executable")).resolve(), target_python)
    repo = tmp_path / "repo"
    repo.mkdir()
    pm_project = Path(__file__).resolve().parents[2] / "pm"
    (repo / "pm").mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(pm_project / name, repo / "pm" / name)
    (repo / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="1.0.0"\nrequires-python=">=3.11"\n[project.optional-dependencies]\npayloadtest=[]\n[tool.uv]\npackage=false\n', encoding="utf-8")
    uv = shutil.which("uv")
    assert uv, "native bundle test requires uv"
    env = {**os.environ, "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never", "UV_CACHE_DIR": str(tmp_path / "cache")}
    subprocess.run([uv, "lock", "--python", sys.executable], cwd=repo, env=env, check=True, capture_output=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    monkeypatch.setattr("pm.paths.repo_root", lambda: repo)
    monkeypatch.setattr(native, "_bundle_package_names", lambda: [])
    monkeypatch.setattr(native, "_install_names", lambda names: 0)
    monkeypatch.setattr(native, "_store", lambda: SimpleNamespace(root=output / "tools", entry=lambda _: target_python.parent))
    monkeypatch.setattr(native, "_facts", lambda: SimpleNamespace(get=lambda _: {"entry": "python"}, entries_in_use=lambda: []))
    monkeypatch.setattr(native, "get_package", lambda _: SimpleNamespace(binary=lambda *args: target_python))
    monkeypatch.setattr(native, "pm_uv", lambda: (uv, dict(env)))
    monkeypatch.setattr(native, "_arch_guard", lambda store: [])
    monkeypatch.setattr("scripts.bundles.payload.relativize_links", lambda root: 0)
    monkeypatch.setattr("pm.extras.ANCHORS", {"payloadtest": "bundle_probe.present"})
    monkeypatch.setattr("pm.packages.uv_cache_dir", lambda: tmp_path / "cache")
    real_run = native._run_live
    calls = []
    witness = tmp_path / "inventory-python.json"
    fail_inventory = False

    def child(argv, *, cwd, env):
        calls.append(argv[1])
        assert not (output / "manifest.json").exists()
        marker = json.loads((output / "pm-runtime/pm-runtime.json").read_text())
        assert (output / "pm-runtime" / marker["python"]).resolve() == target_python
        assert (output / "pm-runtime" / marker["sitePackages"]).is_dir()
        result = real_run(argv, cwd=cwd, env=env)
        if argv[1] == "sync" and result[0] == 0:
            site = site_packages(output / "venv")
            if fail_inventory:
                shutil.rmtree(site)
            else:
                package = site / "bundle_probe"
                package.mkdir()
                (package / "__init__.py").write_text(
                    "import json, pathlib, sys\n"
                    f"pathlib.Path({str(witness)!r}).write_text(json.dumps(sys.executable), encoding='utf-8')\n",
                    encoding="utf-8",
                )
                (package / "present.py").write_text("", encoding="utf-8")
        return result

    monkeypatch.setattr(native, "_run_live", child)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "original"))
    assert native.stage_native(SimpleNamespace(out=str(output), ref="HEAD")) == 0
    assert calls == ["venv", "sync"]
    assert (output / "hermes-agent/pyproject.toml").is_file()
    assert json.loads((output / "manifest.json").read_text())["repo"] == "hermes-agent"
    feature_file = output / "enabled-features.json"
    assert json.loads(feature_file.read_text(encoding="utf-8"))["extras"] == ["payloadtest"]
    assert Path(json.loads(witness.read_text(encoding="utf-8"))) == target_python
    assert os.environ["HERMES_RUNTIME_DIR"] == str(tmp_path / "original")

    before = feature_file.read_bytes()
    fail_inventory = True
    assert native.stage_native(SimpleNamespace(out=str(output), ref="HEAD")) == 1
    assert not (output / "manifest.json").exists()
    assert feature_file.read_bytes() == before
    assert os.environ["HERMES_RUNTIME_DIR"] == str(tmp_path / "original")

    monkeypatch.setattr(native, "_run_live", lambda *a, **kw: (1, "injected failure"))
    assert native.stage_native(SimpleNamespace(out=str(output), ref="HEAD")) == 1
    assert not (output / "manifest.json").exists()
    assert os.environ["HERMES_RUNTIME_DIR"] == str(tmp_path / "original")


def test_staged_cache_installs_built_wheel_without_unsigned_zip(tmp_path):
    uv = shutil.which("uv")
    assert uv, "native bundle test requires uv"
    package = tmp_path / "package"
    package.mkdir()
    (package / "pyproject.toml").write_text(
        '[project]\nname="cache-proof"\nversion="1.0.0"\n'
        '[build-system]\nrequires=["setuptools"]\nbuild-backend="setuptools.build_meta"\n',
        encoding="utf-8",
    )
    (package / "cache_proof.py").write_text("VALUE = 'installed from cached wheel'\n", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    archive = dist / "cache_proof-1.0.0.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        source.add(package, arcname="cache_proof-1.0.0")
    cache = tmp_path / "build-cache"
    env = {**os.environ, "UV_CACHE_DIR": str(cache), "UV_NO_CONFIG": "1", "UV_PYTHON_DOWNLOADS": "never"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(dist)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/{archive.name}"
    try:
        subprocess.run(
            [uv, "pip", "install", "--python", sys.executable, "--target", str(tmp_path / "first"),
             "--no-build-isolation", "--no-deps", url],
            env=env, cwd=tmp_path, capture_output=True, text=True, check=True, timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert list(cache.rglob("*.whl")), "the actual uv build must create the redundant ZIP"
    shipped = tmp_path / "payload/uv-cache"
    native.stage_uv_cache(cache, shipped)
    assert not list(shipped.rglob("*.whl"))
    assert list(cache.rglob("*.whl")), "the build machine's cache must not change"

    # The stopped server and absent source trees make a fallback build impossible.
    shutil.rmtree(dist)
    shutil.rmtree(package)
    shutil.rmtree(cache)
    for source in (shipped / "sdists-v9").rglob("src"):
        if source.is_dir():
            shutil.rmtree(source)
    installed = tmp_path / "installed"
    result = subprocess.run(
        [uv, "pip", "install", "--python", sys.executable, "--target", str(installed),
         "--no-deps", "--offline", url],
        env={**env, "UV_CACHE_DIR": str(shipped)}, cwd=tmp_path,
        capture_output=True, text=True, check=True, timeout=60,
    )
    assert "Building" not in result.stderr
    probe = subprocess.run(
        [sys.executable, "-c", "import cache_proof; print(cache_proof.VALUE)"],
        cwd=tmp_path, env={**env, "PYTHONPATH": str(installed)},
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert probe.stdout.strip() == "installed from cached wheel"
