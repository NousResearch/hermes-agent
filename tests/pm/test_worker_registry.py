"""Registered package declarations survive a fresh, isolated interpreter."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tarfile
import textwrap

import pytest

import pm
from pm import registry
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.fixture(autouse=True)
def restore_registry(monkeypatch):
    monkeypatch.setattr(registry, "_packages", dict(registry._packages))


@pytest.fixture(scope="module")
def worker_python(tmp_path_factory):
    from pm.runtime_stage import stage_runtime

    environment = tmp_path_factory.mktemp("registry-worker-python")
    uv = shutil.which("uv")
    assert uv, "the worker contract requires real uv"
    return stage_runtime(Path(uv), Path(sys.executable), environment)


@pytest.mark.parametrize("operation", ["ensure", "stage_only"])
def test_registered_package_installs_archive_in_real_worker(tmp_path, monkeypatch, worker_python, dl_server, operation):
    from pm import paths

    monkeypatch.setattr("pm.runtime.runtime_python", lambda: worker_python)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    source = tmp_path / "package.py"
    source.write_text(textwrap.dedent("""\
        import os
        from pm import Package, register

        @register
        class ArchivePackage(Package):
            name = "registry-worker-archive"

            def stage(self, store, staged, version, target):
                (staged / "worker-pid").write_text(str(os.getpid()))

            def verify(self, entry, target):
                return '' if (entry / 'payload.txt').read_text() == 'plugin archive' else 'bad payload'
        """))
    _load_file(source, monkeypatch)
    payload = b"plugin archive"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo("payload.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    data = stream.getvalue()
    RangeHandler.payloads["/plugin.tar.gz"] = data
    target = pm.current_target()
    lock = pm.Lockfile(paths.lockfile_path())
    lock.set_pin("registry-worker-archive", "1", {target: {
        "url": url(dl_server, "/plugin.tar.gz"), "sha256": hashlib.sha256(data).hexdigest(),
    }})
    lock.save()
    engine = importlib.import_module("pm.ensure")
    monkeypatch.setattr(engine, operation, lambda *a, **kw: pytest.fail("install ran in caller"))
    if operation == "ensure":
        pm.ensure("registry-worker-archive", explicit=True)
        entry = paths.store_root() / pm.Facts(paths.facts_path()).get("registry-worker-archive")["entry"]
    else:
        from pm.client import stage_only
        entry = stage_only("registry-worker-archive", target)
    assert (entry / "payload.txt").read_bytes() == payload
    assert int((entry / "worker-pid").read_text()) != os.getpid()


def _load_file(path, monkeypatch, name="worker_registry_package"):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _child(definitions, code, *, setup=""):
    root = Path(pm.__file__).resolve().parent.parent
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from pm.registry import load_package_definitions, get_package\n"
        "from pm.package import InstallError\n"
        + setup + "\n"
        "load_package_definitions(json.loads(sys.stdin.read()))\n"
        + textwrap.dedent(code)
    )
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        input=json.dumps(definitions), capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("failure", ["dependency", "collision", "name", "not-class"])
def test_definition_import_failures_are_package_errors(tmp_path, monkeypatch, failure):
    source = tmp_path / "package.py"
    source.write_text("from pm import Package\nclass ExternalPackage(Package):\n    name = 'broken-worker-test'\n")
    module = _load_file(source, monkeypatch)
    pm.register(module.ExternalPackage)
    definitions = registry.package_definitions([module.ExternalPackage.name])
    setup = ""
    if failure == "dependency":
        source.write_text(
            "from pathlib import Path\n"
            f"with Path({str(tmp_path / 'imports')!r}).open('a') as log: log.write('imported\\n')\n"
            "import no_such_plugin_dependency\n"
        )
        setup = f"sys.path.insert(0, {str(tmp_path)!r})"
        # Make the module importable, so an internal ModuleNotFoundError must
        # not trigger a second attempt through its source path.
        definitions[0]["module"] = "package"
    elif failure == "collision":
        definitions[0]["module"] = "pm.package"
    elif failure == "name":
        source.write_text("from pm import Package\nclass ExternalPackage(Package):\n    name = 'changed-name'\n")
    else:
        source.write_text("ExternalPackage = object()\n")
    result = _child(definitions, "", setup=setup)
    assert result.returncode != 0
    assert "InstallError: broken-worker-test:" in result.stderr
    assert "package definition" in result.stderr
    if failure == "dependency":
        assert (tmp_path / "imports").read_text().splitlines() == ["imported"]
        assert "no_such_plugin_dependency" in result.stderr
    if failure == "collision":
        assert "different source" in result.stderr


@pytest.mark.parametrize("kind", ["local", "unbound", "main"])
def test_non_importable_registration_fails_with_package_remedy(kind):
    class LocalPackage(pm.Package):
        name = "local-worker-test"

    if kind == "unbound":
        LocalPackage.__qualname__ = "NotAModuleAttribute"
    elif kind == "main":
        LocalPackage.__module__ = "__main__"
    pm.register(LocalPackage)
    with pytest.raises(pm.InstallError) as caught:
        registry.package_definitions([LocalPackage.name])
    assert caught.value.package == LocalPackage.name
    assert "definition" in caught.value.cause
    assert "module-level" in caught.value.remedy
    # An unrelated non-importable definition cannot break a built-in install.
    assert registry.package_definitions(["node"]) == []


def test_file_registered_package_runs_its_own_definition_in_child(tmp_path, monkeypatch):
    source = tmp_path / "package.py"
    source.write_text(textwrap.dedent("""\
        from pm import Package, InstallError, register

        @register
        class ExternalPackage(Package):
            name = "external-worker-test"

            def fetch_url(self, version, target):
                raise InstallError(self.name, "external definition reached", "plugin remedy")
        """))
    module = _load_file(source, monkeypatch)
    # Public registration also works without a decorator at import time.
    pm.register(module.ExternalPackage)
    definitions = registry.package_definitions()
    assert [item["name"] for item in definitions] == [module.ExternalPackage.name]
    result = _child(definitions, """\
        package = get_package('external-worker-test')
        try:
            package.fetch_url('1', 'linux-x64')
        except InstallError as exc:
            print(json.dumps([exc.package, exc.cause, exc.remedy]))
        else:
            raise AssertionError('custom package was not loaded')
        """)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "external-worker-test", "external definition reached", "plugin remedy",
    ]


def test_namespaced_definition_keeps_relative_sibling_imports(tmp_path, monkeypatch):
    import types

    root = types.ModuleType("_pm_test_plugins")
    root.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, root.__name__, root)
    plugin = tmp_path / "example"
    plugin.mkdir()
    (plugin / "__init__.py").write_text("PREFIX = 'namespaced-'\n")
    (plugin / "sibling.py").write_text("NAME = 'namespaced-worker-test'\n")
    package = types.ModuleType("_pm_test_plugins.example")
    package.__path__ = [str(plugin)]
    package.__file__ = str(plugin / "__init__.py")
    package.PREFIX = "namespaced-"
    monkeypatch.setitem(sys.modules, package.__name__, package)
    source = plugin / "packages.py"
    source.write_text("from .sibling import NAME\nfrom . import PREFIX\nassert NAME.startswith(PREFIX)\nfrom pm import Package, register\n"
                      "@register\nclass ExternalPackage(Package):\n    name = NAME\n")
    module = _load_file(source, monkeypatch, "_pm_test_plugins.example.packages")
    definitions = registry.package_definitions([module.ExternalPackage.name])
    result = _child(definitions, "assert get_package('namespaced-worker-test').name == 'namespaced-worker-test'")
    assert result.returncode == 0, result.stderr
