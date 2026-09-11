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

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
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
    for name in ("pyproject.toml", "uv.lock", "lock.json"):
        shutil.copy2(pm_project / name, repo / "pm" / name)
    (repo / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="1.0.0"\nrequires-python=">=3.11"\n[project.scripts]\nprobe="entry:main"\n[project.optional-dependencies]\npayloadtest=[]\n[tool.uv]\npackage=false\n', encoding="utf-8")
    from scripts.build.inputs import RESOURCE_ENV
    for name in RESOURCE_ENV:
        (repo / name).mkdir()
        (repo / name / "asset").write_text("required", encoding="utf-8")
    (repo / "entry.py").write_text("def main(): return 0\n", encoding="utf-8")
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
    monkeypatch.setattr(native, "_facts", lambda: SimpleNamespace(get=lambda _: {"entry": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}, entries_in_use=lambda: []))
    monkeypatch.setattr(native, "get_package", lambda _: SimpleNamespace(binary=lambda *args: target_python))
    monkeypatch.setattr("pm.client.is_runtime", lambda: True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    monkeypatch.setattr(native, "_arch_guard", lambda store: [])
    monkeypatch.setattr("scripts.bundles.payload.relativize_links", lambda root: 0)
    monkeypatch.setattr("pm.extras.ANCHORS", {"payloadtest": "bundle_probe.present"})
    import pm
    real_build = pm.build_environment
    calls = []
    witness = tmp_path / "inventory-python.json"
    fail_inventory = False

    def build(**kwargs):
        assert "uv" not in kwargs
        assert kwargs["sealed"] is True
        calls.append(kwargs)
        assert not (output / "manifest.json").exists()
        marker = json.loads((output / "pm-runtime/pm-runtime.json").read_text())
        assert (output / "pm-runtime" / marker["python"]).resolve() == target_python
        assert (output / "pm-runtime" / marker["sitePackages"]).is_dir()
        result = real_build(**kwargs)
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

    monkeypatch.setattr(pm, "build_environment", build)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "original"))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    assert native._stage_native(SimpleNamespace(out=str(output), ref="HEAD")) == 0
    assert calls[0]["all_extras"] is True
    assert calls[0]["cache"] == tmp_path / "cache"
    assert (output / "hermes-agent/pyproject.toml").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["repo"] == "hermes-agent"
    command = "bin/probe.exe" if os.name == "nt" else "bin/probe"
    assert manifest["runtime"]["commands"] == {"probe": command}
    feature_file = output / "enabled-features.json"
    assert json.loads(feature_file.read_text(encoding="utf-8"))["extras"] == ["payloadtest"]
    assert Path(json.loads(witness.read_text(encoding="utf-8"))) == target_python
    assert os.environ["HERMES_RUNTIME_DIR"] == str(tmp_path / "original")

    before = feature_file.read_bytes()
    fail_inventory = True
    assert native._stage_native(SimpleNamespace(out=str(output), ref="HEAD")) == 1
    assert not (output / "manifest.json").exists()
    assert feature_file.read_bytes() == before
    assert os.environ["HERMES_RUNTIME_DIR"] == str(tmp_path / "original")

    from pm.package import InstallError
    def fail_build(**kwargs):
        raise InstallError("venv", "injected failure")
    monkeypatch.setattr(pm, "build_environment", fail_build)
    assert native._stage_native(SimpleNamespace(out=str(output), ref="HEAD")) == 1
    assert not (output / "manifest.json").exists()
    assert os.environ["HERMES_RUNTIME_DIR"] == str(tmp_path / "original")

    import pytest
    (repo / "pm/lock.json").write_text("{}", encoding="utf-8")
    subprocess.run(["git", "add", "pm/lock.json"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=f@example.test", "commit", "-m", "different pins"], cwd=repo, check=True, capture_output=True)
    monkeypatch.setattr(native, "_install_names", lambda names: pytest.fail("mismatched pins reached provisioning"))
    with pytest.raises(ValueError, match="PM lock differs"):
        native._stage_native(SimpleNamespace(out=str(output), ref="HEAD"))


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


def test_native_dispatch_isolates_process_state_on_real_child_failure(tmp_path, monkeypatch):
    # Compiler provisioning has its own native test; this probe must stop
    # at the invalid revision without installing tools on a developer host.
    monkeypatch.setattr("scripts.build.windows_deps.prepare_windows_environment", lambda **kwargs: dict(kwargs["env"]))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "user-home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "user-tools"))
    before = dict(os.environ)
    out = tmp_path / "output"
    assert native.stage_native(SimpleNamespace(out=str(out), ref="missing-build-test-ref")) != 0
    assert dict(os.environ) == before
    assert not (tmp_path / "user-home").exists()
    assert not (tmp_path / "user-tools").exists()
    assert not (out / "manifest.json").exists()
    assert not list(out.glob(".build-*"))


def test_native_dispatch_keeps_cache_across_failed_children(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.build.windows_deps.prepare_windows_environment", lambda **kwargs: dict(kwargs["env"]))
    out = tmp_path / "payload"
    cache = tmp_path / "persistent-cache"
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "ambient-cache"))
    original = dict(os.environ)
    run = subprocess.run
    attempts = []

    def child(command, *, cwd, env):
        assert command[command.index("-m") + 1] == "scripts.bundles.native"
        assert Path(env["UV_CACHE_DIR"]) == cache
        attempts.append(Path(env["HOME"]))
        # Run a real child using the actual dispatch environment. Its cache
        # write must survive the failing process and temporary-HOME cleanup.
        return run([sys.executable, "-c",
                    "import os,sys; from pathlib import Path; "
                    "p=Path(os.environ['UV_CACHE_DIR']); p.mkdir(exist_ok=True); "
                    "f=p/'reused'; f.write_text(f.read_text()+'x' if f.exists() else 'x'); sys.exit(17)"],
                   cwd=cwd, env=env)

    monkeypatch.setattr(native.subprocess, "run", child)
    for _ in range(2):
        assert native.stage_native(SimpleNamespace(out=out, ref="HEAD", cache=cache)) == 17
    assert (cache / "reused").read_text() == "xx"
    assert all(not home.exists() for home in attempts)
    assert not (tmp_path / "ambient-cache").exists()
    assert dict(os.environ) == original
