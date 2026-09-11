"""PM-owned uv commands use the installed interpreter, never host discovery."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm.lock import Facts, Lockfile
from pm.package import InstallError
from pm.packages import Python, Uv
from pm.store import current_target

@pytest.fixture(autouse=True)
def isolated_machine_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))



@pytest.fixture
def installed_uv(tmp_path, monkeypatch):
    import pm.paths as paths
    import pm.registry as registry

    uv = shutil.which("uv")
    assert uv, "the interpreter selection contract requires real uv"
    store = tmp_path / "store"
    lock = Lockfile(tmp_path / "lock.json")
    target = current_target()
    digest = "1" * 64
    entry = store / "uv"
    entry.mkdir(parents=True)
    binary = entry / ("uv.exe" if os.name == "nt" else "uv")
    shutil.copy2(uv, binary)
    uvx = Path(uv).with_name("uvx" + binary.suffix)
    assert uvx.is_file(), "the real uv distribution must include uvx"
    shutil.copy2(uvx, entry / uvx.name)
    lock.set_pin("uv", "test", {target: {"url": "https://test.invalid/uv", "sha256": digest}})
    lock.set_pin("python", "test", {target: {"url": "https://test.invalid/python", "sha256": digest}})
    lock.save()
    facts = Facts(store / "facts.json")
    facts.record("uv", "test", entry.name, {}, store, target=target, artifacts=[digest])
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    monkeypatch.setattr(paths, "store_root", lambda: store)
    monkeypatch.setattr(paths, "writable_store_root", lambda: store)
    monkeypatch.setitem(registry._packages, "uv", Uv())
    return tmp_path, binary, facts, target, digest


def test_all_uv_commands_keep_the_pm_interpreter(installed_uv, monkeypatch):
    import pm.registry as registry

    root, uv, facts, target, digest = installed_uv
    selected = root / "store" / "selected-python"
    clean = {key: value for key, value in os.environ.items() if not key.startswith("UV_")}
    clean.update({"UV_NO_CONFIG": "1", "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never"})
    subprocess.run([str(uv), "venv", "--python", sys.executable, str(selected)],
                   cwd=root, env=clean, check=True, capture_output=True, timeout=60)
    python = selected / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    class FixturePython(Python):
        binary_rel = {"win32": "Scripts/python.exe", "posix": "bin/python"}

    monkeypatch.setitem(registry._packages, "python", FixturePython())
    facts.record("python", "test", selected.name, {}, selected.parent,
                 target=target, artifacts=[digest])
    monkeypatch.setenv("UV_PYTHON", str(root / "ambient-python"))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(root / "ambient-venv"))
    before = dict(os.environ)
    managed_uv, env = importlib.import_module("pm.ensure").uv(realize=False)
    assert managed_uv == str(uv)
    assert env.get("UV_PYTHON") == str(python)
    assert "UV_PROJECT_ENVIRONMENT" not in env
    assert dict(os.environ) == before
    uvx, uvx_env = importlib.import_module("pm.ensure").uv("uvx", realize=False)
    assert Path(uvx) == uv.with_name("uvx" + uv.suffix)
    assert uvx_env["UV_PYTHON"] == str(python)
    subprocess.run([uvx, "--version"], cwd=root, env=uvx_env, check=True, capture_output=True, timeout=30)

    project = root / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="pm-python-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    (project / ".python-version").write_text(str(root / "host-only-python"), encoding="utf-8")
    env.pop("UV_NO_CONFIG")
    env.update({"UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never"})
    environment = root / "candidate"
    env["VIRTUAL_ENV"] = str(environment)
    for args in (["venv", str(environment)], ["lock"], ["sync", "--frozen", "--active"],
                 ["run", "--active", "--no-sync", "python", "-c", "import sys; print(sys.prefix)"]):
        result = subprocess.run([managed_uv, *args], cwd=project, env=env,
                                capture_output=True, text=True, check=True, timeout=60)
    assert Path(result.stdout.strip()).resolve() == environment.resolve()

    incomplete = root / "store" / "uv-without-uvx"
    incomplete.mkdir()
    shutil.copy2(uv, incomplete / uv.name)
    facts.record("uv", "test", incomplete.name, {}, incomplete.parent,
                 target=target, artifacts=[digest])
    ensure = importlib.import_module("pm.ensure")
    assert ensure.uv("uvx", realize=False)[0] is None
    with pytest.raises(InstallError, match="binary is missing"):
        ensure.uv("uvx")


def test_uv_refuses_discovery_when_pm_python_is_missing(installed_uv, monkeypatch):
    root, _, facts, target, digest = installed_uv
    ensure = importlib.import_module("pm.ensure")
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: False)
    monkeypatch.setenv("UV_PYTHON", str(root / "ambient-python"))
    managed_uv, env = ensure.uv(realize=False)
    assert managed_uv is None
    assert "UV_PYTHON" not in env
    assert not (root / "store" / "python-test").exists()
    with pytest.raises(InstallError, match="python"):
        ensure.uv()

    entry = root / "store" / "missing-binary"
    entry.mkdir()
    facts.record("python", "test", entry.name, {}, entry.parent,
                 target=target, artifacts=[digest])
    recorded = facts.path.read_bytes()
    assert ensure.uv(realize=False)[0] is None
    with pytest.raises(InstallError, match="lazy installs are disabled: python"):
        ensure.uv()
    assert facts.path.read_bytes() == recorded
    assert not list(entry.iterdir())


@pytest.mark.platforms("windows")
def test_bundled_uv_uses_a_verified_writable_python_without_changing_runtime(installed_uv, monkeypatch):
    import pm.paths as paths
    import pm.registry as registry
    from pm.store import Store, tree_digest

    root, uv_binary, facts, target, digest = installed_uv
    ensure = importlib.import_module("pm.ensure")
    shipped = uv_binary.parent.parent
    writable = root / "writable-tools"
    monkeypatch.setattr(paths, "writable_store_root", lambda: writable)
    (shipped.parent / "manifest.json").write_text("{}", encoding="utf-8")

    class FixturePython(Python):
        def verify(self, entry, target):
            return "" if self.binary(entry, target).read_bytes() == b"pinned interpreter" else "damaged Python"

    class FixtureUv(Uv):
        emulated_arch_targets = {target}

    monkeypatch.setitem(registry._packages, "uv", FixtureUv())
    facts.record("uv", "test", uv_binary.parent.name, {}, shipped,
                 target=target, artifacts=[digest], digest=tree_digest(uv_binary.parent))
    python = FixturePython()
    monkeypatch.setitem(registry._packages, "python", python)
    entry = shipped / python.store_entry("test", target)
    entry.mkdir()
    binary = entry / "python.exe"
    binary.write_bytes(b"pinned interpreter")
    (entry / "python.dll").write_bytes(b"pinned runtime")
    facts.record("python", "test", entry.name, python.env(entry, target), shipped,
                 target=target, artifacts=[digest], digest=tree_digest(entry))
    before = facts.path.read_bytes()
    shipped_digest = tree_digest(entry)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: False)

    def no_download(*args, **kwargs):
        raise AssertionError("the verified Python must be copied without downloading")

    monkeypatch.setattr(Store, "fetch_many", no_download)
    assert ensure.uv(realize=False)[0] is None
    assert not writable.exists()
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        ensure.uv()
    assert not writable.exists()

    resolved_uv, env = ensure.uv(explicit=True)
    copied = writable / entry.name
    assert resolved_uv == str(uv_binary)
    assert env["UV_PYTHON"] == str(copied / "python.exe")
    assert tree_digest(copied) == shipped_digest
    copied_fact = Facts(writable / "facts.json").get("python")
    assert copied_fact["digest"] == shipped_digest
    assert copied_fact["artifacts"] == [digest]
    assert copied_fact["target"] == target
    assert ensure.installed_package("python").binary == binary
    assert str(copied) not in ensure.env_for("python")["PATH"]

    def no_copy(*args, **kwargs):
        raise AssertionError("a matching copy must be reused")

    with monkeypatch.context() as reuse:
        reuse.setattr(shutil, "copytree", no_copy)
        assert ensure.uv(realize=False)[1]["UV_PYTHON"] == env["UV_PYTHON"]
        assert ensure.uv()[1]["UV_PYTHON"] == env["UV_PYTHON"]

    assert facts.path.read_bytes() == before
    assert tree_digest(entry) == shipped_digest


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("damage", ["source", "copy", "publication"])
def test_copy_failure_preserves_previous_python(installed_uv, monkeypatch, damage):
    import pm.registry as registry
    from pm.store import Store, tree_digest

    root, _, facts, target, digest = installed_uv
    ensure = importlib.import_module("pm.ensure")
    shipped = facts.path.parent
    writable = root / "writable-tools"
    writable.mkdir()
    python = Python()
    monkeypatch.setattr(python, "verify", lambda *_: "")
    monkeypatch.setitem(registry._packages, "python", python)
    entry_name = python.store_entry("test", target)
    source = shipped / entry_name
    source.mkdir()
    (source / "python.exe").write_bytes(b"new interpreter")
    facts.record("python", "test", entry_name, {}, shipped,
                 target=target, artifacts=[digest], digest=tree_digest(source))
    previous = writable / entry_name
    previous.mkdir()
    (previous / "python.exe").write_bytes(b"previous interpreter")
    previous_facts = Facts(writable / "facts.json")
    previous_facts.record("python", "previous", entry_name, {}, writable,
                          target=target, artifacts=[digest], digest=tree_digest(previous))
    before = previous_facts.path.read_bytes()
    copytree = shutil.copytree

    if damage == "source":
        (source / "python.exe").write_bytes(b"damaged source")
    elif damage == "copy":
        def damaged_copy(src, dest, **kwargs):
            copytree(src, dest, **kwargs)
            (dest / "python.exe").write_bytes(b"damaged copy")
        monkeypatch.setattr(shutil, "copytree", damaged_copy)
    else:
        def failed_publication(*args, **kwargs):
            raise OSError("publication refused")
        monkeypatch.setattr(Store, "publish", failed_publication)

    with pytest.raises(InstallError, match="verification|copied bytes|publication refused"):
        ensure._install(python, ensure._lockfile(), previous_facts, Store(writable), target,
                        copy_from=(facts, Store(shipped)))
    assert (previous / "python.exe").read_bytes() == b"previous interpreter"
    assert previous_facts.path.read_bytes() == before
