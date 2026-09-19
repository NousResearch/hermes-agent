"""The native bundle pipeline publishes only after a real staged sync succeeds."""
from __future__ import annotations

import hashlib
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

import pytest

from scripts.bundles import native


def test_bundle_stages_git_tree_and_runs_native_children_before_manifest(tmp_path, monkeypatch):
    import importlib
    import inspect

    from pm.environments import site_packages
    from pm.lock import Facts
    from pm.registry import get_package
    from pm.store import tree_digest
    from tests.pm._fixtures import _wheel

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    output = tmp_path / "payload"
    canonical = tmp_path / "canonical"
    lock, target = native._lockfile(), native.current_target()
    python_package = get_package("python")
    python_entry = python_package.store_entry(lock.version("python"), target)
    source_python = python_package.binary(canonical / python_entry, target)
    target_python = output / "tools" / source_python.relative_to(canonical)
    source_python.parent.mkdir(parents=True)
    # PM seals a payload-owned base interpreter, not an external venv launcher.
    # Windows carries its stdlib beside the executable; POSIX PM pythons are
    # RELOCATABLE builds that resolve sys.prefix relative to their own
    # tree — a bare ELF copy falls back to the compile-time /install prefix
    # and cannot even create its venv. Stage the full toolchain, mirroring
    # the store layout the payload's tools/ directory promises.
    if os.name == "nt":
        shutil.copytree(Path(sys.base_prefix), source_python.parent, dirs_exist_ok=True)
    else:
        shutil.copytree(Path(getattr(sys, "_base_executable")).resolve().parents[1],
                        source_python.parents[1], dirs_exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    source = Path(__file__).resolve().parents[2]
    shutil.copytree(source / "pm", repo / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "hermes_cli").mkdir()
    for name in ("__init__.py", "runtime_state.py"):
        shutil.copy2(source / "hermes_cli" / name, repo / "hermes_cli" / name)
    shutil.copy2(source / "hermes_constants.py", repo / "hermes_constants.py")
    wheels = repo / "wheels"
    wheels.mkdir()
    witness = tmp_path / "inventory-python.json"
    wheel = _wheel(wheels, "bundle_probe")
    import zipfile
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("bundle_probe/witness/__init__.py", "import json,pathlib,sys\n"
                         f"pathlib.Path({str(witness)!r}).write_text(json.dumps(sys.executable), encoding='utf-8')\n")
        archive.writestr("bundle_probe/witness/present.py", "")
    (repo / "pyproject.toml").write_text(
        '[project]\nname="fixture"\nversion="1.0.0"\nrequires-python=">=3.11"\n'
        '[project.scripts]\nprobe="entry:main"\n[project.optional-dependencies]\npayloadtest=["bundle-probe==1.0"]\n'
        '[tool.uv]\npackage=false\nno-index=true\nfind-links=["wheels"]\n', encoding="utf-8")
    selected = {"agent-browser", "chromium", "uv", "python"}
    stale = {"chromium-headless-shell", "retired-tool"}
    user_store = tmp_path / "user-tools"
    for store in (output / "tools", user_store):
        store.mkdir(parents=True, exist_ok=True)
        facts = Facts(store / "facts.json")
        for name in selected | stale:
            entry = store / f"cached-{name}"
            entry.mkdir(exist_ok=True)
            (entry / "payload").write_text(name, encoding="utf-8")
            facts.record(name, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", entry.name, {}, store)
        (store / "orphaned-version").mkdir()
        (store / ".partials").mkdir()
    user_before = {path.relative_to(user_store): path.read_bytes() if path.is_file() else None for path in user_store.rglob("*")}

    from scripts.build.inputs import RESOURCE_ENV
    for name in RESOURCE_ENV:
        (repo / name).mkdir()
        (repo / name / "asset").write_text("required", encoding="utf-8")
    (repo / "entry.py").write_text("import bundle_probe\ndef main(): print(bundle_probe.__version__); return 7\n", encoding="utf-8")
    uv = shutil.which("uv")
    assert uv, "native bundle test requires uv"
    uv_package = get_package("uv")
    uv_entry = canonical / uv_package.store_entry(lock.version("uv"), target)
    uv_entry.mkdir(parents=True)
    shutil.copy2(uv, uv_package.binary(uv_entry, target))
    # Browser payloads are inert: this test exercises their real dependency
    # closure, admission and pruning, not browser execution or downloads.
    for name in ("agent-browser", "chromium"):
        package = get_package(name)
        entry = canonical / package.store_entry(lock.version(name), target)
        binary = (entry / ("chrome.exe" if os.name == "nt" else "chrome")
                  if name == "chromium" else package.binary(entry, target))
        binary.parent.mkdir(parents=True)
        binary.write_text("inert browser fixture", encoding="utf-8")
        if name == "chromium":
            (entry / "INSTALLATION_COMPLETE").touch()
    facts = Facts(canonical / "facts.json")
    for name in selected:
        package = get_package(name)
        entry = canonical / package.store_entry(lock.version(name), target)
        (entry / "payload").write_text(name, encoding="utf-8")
        facts.record(name, lock.version(name), entry.name, package.env(entry, target), canonical,
                     target=target, artifacts=[row["sha256"] for row in lock.artifacts(name, target)],
                     digest=tree_digest(entry))
    canonical_before = {name: tree_digest(canonical / facts.get(name)["entry"]) for name in selected}
    canonical_facts = (canonical / "facts.json").read_bytes()
    env = {**os.environ, "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never", "UV_CACHE_DIR": str(tmp_path / "cache")}
    subprocess.run([uv, "lock", "--python", sys.executable], cwd=repo, env=env, check=True, capture_output=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    monkeypatch.setattr("pm.paths.repo_root", lambda: repo)
    (repo / "untracked").write_text("must not ship", encoding="utf-8")
    monkeypatch.setattr(native, "_bundle_package_names", lambda: ["agent-browser", "uv"])
    monkeypatch.setattr(importlib.import_module("pm.install"), "_prepare_artifacts",
                        lambda *args, **kwargs: pytest.fail("prepared tools reached acquisition"))
    monkeypatch.setattr("pm.client.is_runtime", lambda: True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    monkeypatch.setattr("pm.extras.ANCHORS", {"payloadtest": "bundle_probe.witness.present"})
    import pm
    real_stage = pm.stage_manager_runtime

    def stage(**kwargs):
        assert kwargs["cache"] == tmp_path / "cache"
        assert kwargs["python"] == target_python
        assert kwargs["project"] == output / "hermes-agent/pm"
        return real_stage(**kwargs)

    monkeypatch.setattr(pm, "stage_manager_runtime", stage)
    real_build = pm.build_environment
    install_timeout = inspect.signature(real_build).parameters["timeout"].default
    calls = []
    fail_inventory = False

    def build(**kwargs):
        assert "uv" not in kwargs
        assert kwargs["sealed"] is True
        assert kwargs["python"] == target_python
        assert kwargs["source"] == output / "hermes-agent"
        assert kwargs.get("timeout", install_timeout) > install_timeout
        calls.append(kwargs)
        assert not (output / "manifest.json").exists()
        marker_path = output / "pm-runtime/pm-runtime.json"
        assert marker_path.is_file(), "PM must be staged before the application environment"
        marker = json.loads(marker_path.read_text())
        assert (output / "pm-runtime" / marker["python"]).resolve() == target_python
        assert (output / "pm-runtime" / marker["sitePackages"]).is_dir()
        result = real_build(**kwargs)
        site = site_packages(output / "venv")
        if fail_inventory:
            shutil.rmtree(site)
        return result

    monkeypatch.setattr(pm, "build_environment", build)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(user_store))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    prepared = native.prepare_native(out=output, ref="HEAD", source=repo, cache=tmp_path / "cache", tools=canonical, env=env)
    assert {name: tree_digest(canonical / facts.get(name)["entry"]) for name in selected} == canonical_before
    assert (canonical / "facts.json").read_bytes() == canonical_facts
    assert not target_python.samefile(source_python)
    assert prepared == output.with_name(output.name + ".prepared.json")
    assert not prepared.is_relative_to(output)
    assert prepared.is_file()
    assert not (output / "manifest.json").exists()
    from scripts.bundles.native_prepared import load_prepared, preparation_lock
    with preparation_lock(output):
        with pytest.raises(ValueError, match="already in use"):
            native.finish_native(prepared, {})
    assert prepared.is_file()
    # Dependencies are already installed. Reject changed bytes rather than
    # healing them, including a substituted directory with identical bytes.
    inventory = json.loads(prepared.read_text())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert inventory["revision"] == revision
    assert inventory["inputs"]["ref"] == "HEAD"
    changed_inputs = json.loads(prepared.read_text())
    changed_inputs["inputs"]["project"] = str(repo / "pyproject.toml")
    prepared.write_text(json.dumps(changed_inputs), encoding="utf-8")
    with pytest.raises(ValueError, match="run preparation again"):
        load_prepared(prepared)
    prepared.write_text(json.dumps(inventory), encoding="utf-8")
    for relative in ("hermes-agent/entry.py", "hermes-agent/uv.lock",
                     "venv/pyvenv.cfg", "pm-runtime/pm-runtime.json",
                     "enabled-features.json"):
        changed = output / relative
        original = changed.read_bytes()
        changed.write_bytes(original + b"\nchanged")
        with pytest.raises(ValueError, match="run preparation again"):
            load_prepared(prepared)
        changed.write_bytes(original)
    moved = output / "hermes-agent"
    outside = tmp_path / "substituted-source"
    moved.rename(outside)
    moved.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="run preparation again"):
        load_prepared(prepared)
    moved.unlink()
    outside.rename(moved)
    # A source entry linked outside the payload must never be admitted, even
    # when its link text itself is freshly covered by the stored digest.
    linked = moved / "escape"
    linked.symlink_to(witness)
    from scripts.bundles.native_prepared import _source_digest
    inventory["digests"]["hermes-agent"] = _source_digest(moved)
    prepared.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="run preparation again"):
        load_prepared(prepared)
    linked.unlink()
    inventory["digests"]["hermes-agent"] = _source_digest(moved)
    prepared.write_text(json.dumps(inventory), encoding="utf-8")
    frontend = tmp_path / "web-product"
    frontend.mkdir()
    (frontend / "index.html").write_text("built web", encoding="utf-8")
    with monkeypatch.context() as strict:
        def forbidden(*args, **kwargs):
            pytest.fail("strict finish entered dependency acquisition or a child process")
        strict.setattr(pm, "build_environment", forbidden)
        strict.setattr(pm, "stage_manager_runtime", forbidden)
        strict.setattr(pm, "prepare_tools", forbidden)
        strict.setattr(pm, "stage_tools", forbidden)
        strict.setattr(subprocess, "run", forbidden)
        assert native.finish_native(prepared, {"web": frontend}) == 0
        (output / "hermes-agent/install-stamp.json").write_text('{"variant":"bundled"}', encoding="utf-8")
        (output / "manifest.json").write_text('{"variant":"bundled"}', encoding="utf-8")
        launcher = output / ("bin/probe.exe" if os.name == "nt" else "bin/probe")
        launcher.rename(launcher.with_name("renamed-launcher"))
        (frontend / "index.html").write_text("store web", encoding="utf-8")
        assert native.finish_native(prepared, {"web": frontend}) == 0
    assert (output / "hermes-agent/hermes_cli/web_dist/index.html").read_text() == "store web"
    assert calls[0]["all_extras"] is True
    assert calls[0]["cache"] == tmp_path / "cache"
    assert (output / "hermes-agent/pyproject.toml").is_file()
    assert not (output / "hermes-agent/untracked").exists()
    assert not (output / "hermes-agent/.git").exists()
    facts = Facts(output / "tools/facts.json")
    for name in stale:
        assert facts.get(name) is None
        assert not (output / "tools" / f"cached-{name}").exists()
    for name in selected:
        entry = get_package(name).store_entry(lock.version(name), target)
        assert facts.get(name)["entry"] == entry
        assert facts.get(name)["version"] == lock.version(name)
        assert facts.get(name)["target"] == target
        assert facts.get(name)["artifacts"] == [row["sha256"] for row in lock.artifacts(name, target)]
        assert (output / "tools" / entry / "payload").read_text(encoding="utf-8-sig") == name
        assert not (output / "tools" / f"cached-{name}").exists()
    assert not (output / "tools/orphaned-version").exists()
    assert (output / "tools/.partials").is_dir()
    assert user_before == {path.relative_to(user_store): path.read_bytes() if path.is_file() else None for path in user_store.rglob("*")}

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["repo"] == "hermes-agent"
    command = "bin/probe.exe" if os.name == "nt" else "bin/probe"
    assert manifest["runtime"]["commands"] == {"probe": command}
    feature_file = output / "enabled-features.json"
    assert os.environ["HERMES_RUNTIME_DIR"] == str(user_store)

    moved = tmp_path / "installed elsewhere"
    output.rename(moved)
    try:
        run = subprocess.run([str(moved / command)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert run.returncode == 7, run.stderr
        assert run.stdout.strip() == "1.0"
        from pm import runtime as runtime_api, paths
        with monkeypatch.context() as patch:
            patch.setattr(paths, "repo_root", lambda: moved / "hermes-agent")
            run = subprocess.run(runtime_api.runtime_command(moved / "hermes-agent/pm/launch.py", ["status"]),
                                 cwd=tmp_path, env=runtime_api.runtime_environment(), capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stderr
        assert "no pm sync receipt" in run.stdout
    finally:
        moved.rename(output)
    assert json.loads(feature_file.read_text(encoding="utf-8"))["extras"] == ["payloadtest"]
    assert Path(json.loads(witness.read_text(encoding="utf-8-sig"))) == target_python
    before = feature_file.read_bytes()
    fail_inventory = True
    args = SimpleNamespace(out=str(output), ref="HEAD", source=repo, cache=tmp_path / "cache", tools=canonical)
    assert native._stage_native(args) == 1
    assert not (output / "manifest.json").exists()
    assert feature_file.read_bytes() == before
    assert os.environ["HERMES_RUNTIME_DIR"] == str(user_store)

    from pm.package import InstallError
    def fail_build(**kwargs):
        raise InstallError("venv", "injected failure")
    monkeypatch.setattr(pm, "build_environment", fail_build)
    assert native._stage_native(args) == 1
    assert not (output / "manifest.json").exists()
    assert os.environ["HERMES_RUNTIME_DIR"] == str(user_store)

    (repo / "pm/lock.json").write_text("{}", encoding="utf-8")
    subprocess.run(["git", "add", "pm/lock.json"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=f@example.test", "commit", "-m", "different pins"], cwd=repo, check=True, capture_output=True)
    monkeypatch.setattr("pm.prepare_tools", lambda names, **kwargs: pytest.fail("mismatched pins reached provisioning"))
    with pytest.raises(ValueError, match="PM lock differs"):
        native._stage_native(args)


def test_staged_cache_prunes_entries_the_lock_cannot_resolve(tmp_path, monkeypatch):
    """Ship gate: a rolled CI cache snapshot carries wheels for superseded pins;
    the staged payload cache must contain exactly what the shipped lock resolves."""
    lock = tmp_path / "repo"
    lock.mkdir()
    (lock / "uv.lock").write_text(
        "[[package]]\n"
        'name = "keep-me"\n'
        'version = "1.0.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        "\n"
        "[[package]]\n"
        'name = "also-keep"\n'
        'version = "2.0.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n',
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    buckets = {
        "keep-me": ("keep_me-1.0.0.dist-info", True),
        "also-keep": ("also_keep-2.0.0.dist-info", True),
        "superseded-pin": ("superseded_pin-0.9.9.dist-info", False),
        "old-dependency": ("old_dependency-3.2.1.dist-info", False),
    }
    for bucket, (dist_info, _keep) in buckets.items():
        path = cache / "archive-v0" / bucket / dist_info
        path.parent.mkdir(parents=True)
        path.write_text("", encoding="utf-8")
    # an unversioned bucket (no dist-info) must survive — it is not identifiable
    unknown = cache / "archive-v0" / "unknownbucket"
    unknown.mkdir(parents=True)
    (unknown / "data.bin").write_bytes(b"x")
    pruned = native.prune_uv_cache_to_lock(cache, lock)
    assert pruned == 2
    assert (cache / "archive-v0" / "keep-me").is_dir()
    assert (cache / "archive-v0" / "also-keep").is_dir()
    assert not (cache / "archive-v0" / "superseded-pin").exists()
    assert not (cache / "archive-v0" / "old-dependency").exists()
    assert (unknown / "data.bin").exists()


def test_staged_cache_prunes_wheel_index_entries_outside_the_lock(tmp_path):
    lock = tmp_path / "repo"
    lock.mkdir()
    (lock / "uv.lock").write_text(
        "[[package]]\n"
        'name = "keep-me"\n'
        'version = "1.0.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n',
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    for entry in ("keep-me", "superseded-pin"):
        path = cache / "wheels-v6" / "pypi" / entry
        path.mkdir(parents=True)
        (path / "metadata.msgpack").write_bytes(b"")
    sdist = cache / "sdists-v9" / "pypi" / "old-sdist-only"
    sdist.mkdir(parents=True)
    (sdist / "metadata.msgpack").write_bytes(b"")
    native.prune_uv_cache_to_lock(cache, lock)
    assert (cache / "wheels-v6" / "pypi" / "keep-me").is_dir()
    assert not (cache / "wheels-v6" / "pypi" / "superseded-pin").exists()
    assert not (cache / "sdists-v9" / "pypi" / "old-sdist-only").exists()


@pytest.mark.parametrize("pointer, shard", [("revision.http", ""), ("revision.rev", "build-settings")])
def test_staged_cache_skips_build_inputs_before_copying(tmp_path, monkeypatch, pointer, shard):
    cache = tmp_path / "cache"
    revision = Path("sdists-v9/index/package/revision")
    wheels = revision / shard
    waste = {
        revision / "src/target/release/build.exe": b"build output",
    }
    kept = {
        wheels / "metadata.msgpack": b"wheel metadata",
        wheels / "cache_proof-1.0-py3-none-any.whl": b"built wheel ZIP ships for offline rebuilds",
        revision.parent / pointer: b"revision pointer",
        wheels / "cache_proof-1.0-py3-none-any/src/template.whl": b"package data",
        Path("archive-v0/entry/cache_proof/src/__init__.py"): b"archive data",
        Path("other-bucket/src/keep.whl"): b"unrelated data",
    }
    for relative, data in {**waste, **kept}.items():
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    copyfile = shutil.copyfile

    def record_copy(source, destination, **kwargs):
        assert Path(source).relative_to(cache) not in waste, "build inputs must never be copied"
        return copyfile(source, destination, **kwargs)

    shipped = tmp_path / "payload/uv-cache"
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", record_copy)
        native.stage_uv_cache(cache, shipped)
    assert all(not (shipped / relative).exists() for relative in waste)
    assert all((shipped / relative).read_bytes() == data for relative, data in kept.items())
    assert all((cache / relative).read_bytes() == data for relative, data in {**waste, **kept}.items())


SELF_BUILD_BACKEND = '''import os, zipfile


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = "cache_proof-1.0.0-py3-none-any.whl"
    path = os.path.join(wheel_directory, name)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("cache_proof/__init__.py", "VALUE = 'installed from cached wheel'\\n")
        info = "cache_proof-1.0.0.dist-info/"
        zf.writestr(info + "METADATA", "Metadata-Version: 2.1\\nName: cache-proof\\nVersion: 1.0.0\\n")
        zf.writestr(
            info + "WHEEL",
            "Wheel-Version: 1.0\\nGenerator: selfbuild\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
        )
        zf.writestr(info + "RECORD", "")
    return name


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    path = os.path.join(metadata_directory, "cache_proof-1.0.0.dist-info")
    os.mkdir(path)
    with open(os.path.join(path, "METADATA"), "w") as handle:
        handle.write("Metadata-Version: 2.1\\nName: cache-proof\\nVersion: 1.0.0\\n")
    return "cache_proof-1.0.0.dist-info"
'''


def test_staged_cache_ships_full_wheel_set_and_rebuilds_offline(tmp_path):
    from tests.pm._fixtures import _wheel

    uv = shutil.which("uv")
    assert uv, "native bundle test requires uv"
    package = tmp_path / "package"
    package.mkdir()
    # Self-contained PEP 517 backend (stdlib only, requires=[]): the offline
    # rebuild below must not depend on build tools absent from the fixture.
    (package / "pyproject.toml").write_text(
        '[build-system]\nrequires=[]\nbuild-backend="selfbuild"\nbackend-path=["."]\n',
        encoding="utf-8",
    )
    (package / "selfbuild.py").write_text(SELF_BUILD_BACKEND, encoding="utf-8")
    (package / "cache_proof.py").write_text("VALUE = 'installed from cached wheel'\n", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    archive = dist / "cache_proof-1.0.0.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        source.add(package, arcname="cache_proof-1.0.0")
    wheel = _wheel(dist, "wheel_proof")
    for name, artifact in (("cache-proof", archive), ("wheel-proof", wheel)):
        index = dist / "simple" / name
        index.mkdir(parents=True)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        (index / "index.html").write_text(
            f'<a href="../../{artifact.name}#sha256={digest}">{artifact.name}</a>', encoding="utf-8",
        )
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="offline-proof"\nversion="1.0"\nrequires-python=">=3.11"\n'
        'dependencies=["cache-proof==1.0.0", "wheel-proof==1.0"]\n[tool.uv]\npackage=false\n',
        encoding="utf-8",
    )
    cache = tmp_path / "build-cache"
    env = {**os.environ, "UV_CACHE_DIR": str(cache), "UV_NO_CONFIG": "1", "UV_PYTHON_DOWNLOADS": "never"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(dist)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    index_url = f"http://127.0.0.1:{server.server_port}/simple"
    try:
        warm = subprocess.run(
            [uv, "pip", "install", "--python", sys.executable, "--target", str(tmp_path / "first"),
             "--no-deps", "--index-url", index_url,
             "cache-proof==1.0.0", "wheel-proof==1.0"],
            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60,
        )
        assert warm.returncode == 0, warm.stderr + warm.stdout
        locked = subprocess.run(
            [uv, "lock", "--python", sys.executable, "--index-url", index_url],
            env=env, cwd=project, capture_output=True, text=True, timeout=60,
        )
        assert locked.returncode == 0, locked.stderr
        warmed = subprocess.run(
            [uv, "sync", "--python", sys.executable, "--frozen", "--index-url", index_url],
            env=env, cwd=project, capture_output=True, text=True, timeout=60,
        )
        assert warmed.returncode == 0, warmed.stderr

        built_zips = list(cache.rglob("*.whl"))
        assert built_zips, "the actual uv build must create the redundant ZIP"
        sources = [path for bucket in cache.glob("sdists-v*") for path in bucket.rglob("src") if path.is_dir()]
        assert sources, "the actual uv build must leave its source tree in the cache"
        shipped = tmp_path / "payload/uv-cache"
        native.stage_uv_cache(cache, shipped)
        assert all(not (shipped / path.relative_to(cache)).exists() for path in sources)
        assert all(path.exists() for path in [*built_zips, *sources]), "the build machine's cache must not change"
        # The full cache ships every wheel the build resolved, downloadable
        # or built: a per-install venv rebuild must not depend on network
        # reachability for any package the payload already shipped.
        assert any(shipped.rglob("wheel_proof*")), "downloadable wheels ship in the full cache"

        # A fresh mutable venv rebuilds OFFLINE from the shipped cache alone.
        shutil.rmtree(cache, ignore_errors=True)
        shutil.rmtree(tmp_path / "first", ignore_errors=True)
        shutil.rmtree(project / ".venv", ignore_errors=True)
        result = subprocess.run(
            [uv, "sync", "--python", sys.executable, "--frozen", "--offline"],
            env={**env, "UV_CACHE_DIR": str(shipped)}, cwd=project,
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "Building" not in result.stderr and "Built" not in result.stdout, result.stderr + result.stdout
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    probe = subprocess.run(
        [str(python), "-I", "-c",
         "import cache_proof, wheel_proof; print(cache_proof.VALUE); print(wheel_proof.__version__)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    assert probe.stdout.splitlines() == ["installed from cached wheel", "1.0"]


def test_native_dispatch_isolates_process_state_on_real_child_failure(tmp_path, monkeypatch):
    # Compiler provisioning has its own native test; this probe must stop
    # at the invalid revision without installing tools on a developer host.
    monkeypatch.setattr("scripts.build.windows_deps.prepare_windows_environment", lambda **kwargs: dict(kwargs["env"]))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "user-home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "user-tools"))
    before = dict(os.environ)
    out = tmp_path / "output"
    assert native.stage_native(SimpleNamespace(out=str(out), ref="missing-build-test-ref", cache=tmp_path / "cache")) != 0
    assert dict(os.environ) == before
    assert not (tmp_path / "user-home").exists()
    assert not (tmp_path / "user-tools").exists()
    assert not (out / "manifest.json").exists()
    assert not list(out.glob(".build-*"))


def test_native_preparation_refuses_symlinked_output_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr("pm.prepare_tools", lambda *args, **kwargs: pytest.fail("unsafe output reached acquisition"))
    output = tmp_path / "payload"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("untouched", encoding="utf-8")
    (output / "hermes-agent").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|escaped"):
        native.prepare_native(out=output, ref="HEAD", source=Path(__file__).resolve().parents[2],
                              cache=tmp_path / "cache")
    assert (outside / "keep").read_text() == "untouched"
    assert not output.with_name(output.name + ".prepared.json").exists()


@pytest.mark.parametrize("cache_source,explicit_compilers,status", [
    ("explicit", True, 17), ("ambient", False, 0), ("default", False, 17),
])
def test_native_dispatch_child_environment(tmp_path, monkeypatch, cache_source, explicit_compilers, status):
    from pm.packages import uv_cache_dir

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "host")
    monkeypatch.setattr("scripts.build.windows_deps.prepare_windows_environment", lambda **kwargs: dict(kwargs["env"]))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "user"))
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    ambient = tmp_path / "ambient"
    if cache_source != "default":
        monkeypatch.setenv("UV_CACHE_DIR", str(ambient))
    cache = {"explicit": tmp_path / "explicit", "ambient": ambient, "default": uv_cache_dir()}[cache_source]
    compilers = {}
    for key, name in (("CARGO_HOME", ".cargo"), ("RUSTUP_HOME", ".rustup")):
        home = tmp_path / ("custom" if explicit_compilers else "host") / name
        home.mkdir(parents=True)
        (home / "fixture-state").write_text(key, encoding="utf-8")
        compilers[key] = str(home)
        monkeypatch.delenv(key, raising=False)
        if explicit_compilers:
            monkeypatch.setenv(key, str(home))
    before, run, homes = dict(os.environ), subprocess.run, []
    observed = tmp_path / "child.json"

    def child(command, *, cwd, env):
        assert command[command.index("-m") + 1] == "scripts.bundles.native"
        return run([sys.executable, "-c",
                    "import os,json,sys; from pathlib import Path; "
                    "Path(sys.argv[1]).write_text(json.dumps(dict(os.environ))); "
                    "assert all((Path(os.environ[k])/'fixture-state').read_text() == k for k in ('CARGO_HOME','RUSTUP_HOME')); "
                    "p=Path(os.environ['UV_CACHE_DIR']); p.mkdir(parents=True,exist_ok=True); "
                    "f=p/'reused'; f.write_text(f.read_text()+'x' if f.exists() else 'x'); sys.exit(int(sys.argv[2]))",
                    str(observed), str(status)], cwd=cwd, env=env)

    monkeypatch.setattr(native.subprocess, "run", child)
    out = tmp_path / "payload"
    for _ in range(2):
        assert native.stage_native(SimpleNamespace(out=out, ref="HEAD", cache=cache if cache_source == "explicit" else None)) == status
        env = json.loads(observed.read_text(encoding="utf-8-sig"))
        homes.append(Path(env["HOME"]))
        assert env["HOME"] == env["USERPROFILE"] != str(tmp_path / "host")
        assert {key: env[key] for key in compilers} == compilers
        assert Path(env["UV_CACHE_DIR"]) == cache
        assert Path(env["HERMES_RUNTIME_DIR"]) == out / "tools"
        assert Path(env["HERMES_HOME"]) == homes[-1] / ".hermes"
    assert (cache / "reused").read_text(encoding="utf-8-sig") == "xx"
    assert all(not home.exists() for home in homes)
    assert dict(os.environ) == before
