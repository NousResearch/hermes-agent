"""Real isolated workers: no parent imports or callables cross the wire."""
from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import shutil
import sys

import pytest

from pm import paths
from pm.package import InstallError
from pm.runtime import runtime_python
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401
from tests.pm.test_runtime_wheelhouse import locked_wheelhouse  # noqa: F401
from tests.pm._fixtures import (
    _wheel,
    build_worker as build_worker,
    client as client,
    isolated_python as isolated_python,
)

# Spawns children with a home it builds itself; the parent's must stay real.
pytestmark = pytest.mark.real_machine_home


def test_isolated_worker_preserves_install_error(client, monkeypatch):
    engine = importlib.import_module("pm.install")
    monkeypatch.setattr(engine, "ensure", lambda *a, **kw: pytest.fail("engine ran in caller"))
    with pytest.raises(InstallError) as caught:
        client.ensure("node", explicit=True)
    assert caught.value.package == "node"
    assert caught.value.cause == "not in the lockfile"
    assert caught.value.remedy == "add it with `hermes pm lock --bump`"
    assert not paths.facts_path().exists()



def test_refused_or_already_paused_install_does_not_acquire_runtime(client, monkeypatch):
    import threading
    from pm.downloader import DownloadPaused

    monkeypatch.setattr(client, "runtime_command", lambda path, **kwargs: pytest.fail("refusal acquired PM runtime"))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        client.ensure("node")
    paused = threading.Event()
    paused.set()
    with pytest.raises(DownloadPaused):
        client.ensure("node", explicit=True, pause_event=paused)


def _current_environment(tmp_path, monkeypatch, members):
    from pm.environments import install_state_dir
    from pm.lock import Facts
    from pm.packages import Venv

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    environment = install_state_dir(repo) / "environments" / "existing" / "venv"
    environment.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = test\n")
    Facts(paths.runtime_facts_path()).record_state(
        "venv", Venv().expected_stamp([], plugin_dirs=members), [], environment=environment,
    )
    return repo


@pytest.mark.parametrize("member_shape", ["paths", "sources"])
@pytest.mark.parametrize("route", ["worker", "direct", "foreign-runtime"])
def test_currency_probe_preserves_union_and_candidate_inputs(client, tmp_path, monkeypatch, isolated_python,
                                                            member_shape, route):
    import json
    from pm.environments import runtime_facts_path, selected_venv
    from pm.lock import Facts
    from pm.packages import Venv

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = candidate / "plugin.yaml"
    manifest.write_text('name: candidate\npython_dependencies: ["candidate-dep==1"]\n')
    members = [candidate] if member_shape == "paths" else {tmp_path / "installed": candidate}
    repo = _current_environment(tmp_path, monkeypatch, members)
    environment = selected_venv(repo)
    recorded = ["incumbent-extra", "provider-extra"]
    facts_path = runtime_facts_path(repo)
    Facts(facts_path).record_state(
        "venv", Venv(repo).expected_stamp(recorded, plugin_dirs=members), recorded,
        environment=environment,
    )
    monkeypatch.setattr(client, "is_runtime", lambda: route != "worker")
    if route == "foreign-runtime":
        monkeypatch.setattr(paths, "repo_root", lambda: tmp_path / "other-project")
    root_args = {"project_root": repo} if route != "worker" else {}
    acquisitions = []

    def ready_runtime(*, bootstrap, cache):
        assert bootstrap is False, "currency probe attempted to bootstrap PM"
        acquisitions.append(bootstrap)
        return isolated_python

    monkeypatch.setattr("pm.runtime.runtime_python", ready_runtime)
    if route != "direct":
        engine = importlib.import_module("pm.install")
        monkeypatch.setattr(engine, "venv_is_current", lambda **kw: pytest.fail("probe ran in caller"))
    def snapshot():
        return {path.relative_to(tmp_path): (path.read_bytes() if path.is_file() else None)
                for path in tmp_path.rglob("*")}

    from pm.lock import Lockfile
    pins = Lockfile(paths.lockfile_path())
    pins.set_pin("python", "test.1", {"any": {"url": "https://example.invalid/python", "sha256": "1" * 64}})
    pins.save()
    Facts(facts_path).record_state("venv", Venv(repo).expected_stamp(recorded, plugin_dirs=members), recorded,
                                   environment=environment)
    before = snapshot()
    assert client.venv_is_current(extras=["provider-extra"], plugin_dirs=members, **root_args)
    assert client.venv_is_current(extras=[], plugin_dirs=members, **root_args)
    assert not client.venv_is_current(extras=["new-extra"], plugin_dirs=members, **root_args)
    assert not client.venv_is_current(extras=recorded, plugin_dirs=[], **root_args)
    assert snapshot() == before, "currency queries changed dependency state"
    assert bool(acquisitions) is (route != "direct")
    pins.set_pin("uv", "unrelated", {"any": {"url": "https://example.invalid/uv", "sha256": "3" * 64}})
    pins.save()
    unchanged = snapshot()
    assert client.venv_is_current(extras=[], plugin_dirs=members, **root_args)
    assert snapshot() == unchanged
    for version, digest in [("test.1", "2" * 64), ("test.2", "1" * 64)]:
        pins.set_pin("python", version, {"any": {"url": "https://example.invalid/python", "sha256": digest}})
        pins.save()
        changed = snapshot()
        assert not client.venv_is_current(plugin_dirs=members, **root_args)
        assert snapshot() == changed
    pins.set_pin("python", "test.1", {"any": {"url": "https://example.invalid/python", "sha256": "1" * 64}})
    pins.save()
    marker = environment / "pyvenv.cfg"
    contents = marker.read_bytes()
    marker.unlink()
    changed = snapshot()
    assert not client.venv_is_current(plugin_dirs=members, **root_args)
    assert snapshot() == changed
    marker.write_bytes(contents)
    assert client.venv_is_current(plugin_dirs=members, **root_args)

    manifest.write_text('name: candidate\npython_dependencies: ["candidate-dep==2"]\n')
    changed = snapshot()
    assert not client.venv_is_current(extras=["provider-extra"], plugin_dirs=members, **root_args)
    assert snapshot() == changed
    assert selected_venv(repo) == environment
    # Corruption must not be mistaken for a missing or current environment.
    data = json.loads(facts_path.read_text())
    data["packages"]["venv"]["extras"] = "provider-extra"
    facts_path.write_text(json.dumps(data))
    malformed = snapshot()
    with pytest.raises(ValueError, match="invalid recorded dependency state"):
        client.venv_is_current(extras=[], plugin_dirs=members, **root_args)
    assert snapshot() == malformed


def _assert_worker_holds_lock(repo):
    from pm.environments import install_state_dir
    from hermes_cli.runtime_state import _lock

    with (install_state_dir(repo) / ".install.lock").open("a+b") as lock:
        assert not _lock(lock.fileno(), wait=False), "callback escaped the worker's runtime lock"


@pytest.mark.parametrize("explicit", [True, False])
def test_sync_discovers_profile_members_after_worker_acquires_lock(client, tmp_path, monkeypatch, isolated_python, explicit):
    from concurrent.futures import ThreadPoolExecutor
    import time
    from hermes_cli.runtime_state import runtime_lock
    from tests.pm._fixtures import worker_toolchain

    sibling = tmp_path / "home/profiles/sibling/plugins/dependency"
    sibling.mkdir(parents=True)
    (sibling / "plugin.yaml").write_text("name: dependency\npython_dependencies: [fixture-dep==1]\n")
    repo = _current_environment(tmp_path, monkeypatch, [sibling])
    ready = tmp_path / "waiting-for-lock"
    worker_toolchain(client, monkeypatch, isolated_python,
        "from contextlib import contextmanager\nimport hermes_cli.runtime_state as state\n"
        "original = state.runtime_lock\n@contextmanager\ndef lock(project):\n"
        f"    Path({str(ready)!r}).touch()\n"
        "    with original(project):\n        yield\nstate.runtime_lock = lock\n")
    with ThreadPoolExecutor() as executor:
        with runtime_lock(repo):
            future = executor.submit(client.sync_venv, explicit=explicit, selection={
                "home": str(tmp_path / "home"), "enabled": ["plain"], "disabled": [],
            })
            deadline = time.monotonic() + 15
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "worker never reached the install lock"
            assert not future.done(), "worker ignored the install lock"
            (sibling.parent.parent / "config.yaml").write_text("plugins:\n  enabled: [dependency]\n")
        future.result(timeout=30)
    assert client.venv_is_current()


@pytest.mark.parametrize("current", [True, False])
@pytest.mark.parametrize("tools_present", [False, True], ids=["cold-tools", "ready-tools"])
def test_lazy_disabled_sync_does_not_bootstrap_tools(client, tmp_path, monkeypatch, isolated_python, current, tools_present):
    import json
    from pm import receipt

    repo = _current_environment(tmp_path, monkeypatch, [])
    if not current:
        (repo / "uv.lock").write_text("version = 2\n")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    # Exercise runtime acquisition too: the other worker tests supply a ready
    # interpreter, which hides an explicit tool install before worker refusal.
    monkeypatch.setattr("pm.runtime.runtime_python", runtime_python)
    from pm import _uv
    original_toolchain = _uv._toolchain

    def toolchain(**kwargs):
        assert kwargs.get("realize") is False, "lazy-disabled sync bootstrapped tools"
        if tools_present:
            return tmp_path / "uv", isolated_python
        return original_toolchain(**kwargs)

    monkeypatch.setattr(_uv, "_toolchain", toolchain)
    monkeypatch.setattr("pm.runtime_stage.stage_runtime",
                        lambda *a, **kw: pytest.fail("lazy-disabled sync prepared PM runtime"))
    with receipt.worker_context("lazy-disabled-sync"):
        with pytest.raises(InstallError, match="lazy installs are disabled") as caught:
            client.sync_venv([], plugin_dirs=[])
        result = receipt.last_for_update("lazy-disabled-sync", consume=True)
    assert caught.value.package == "pm-runtime"
    assert result is not None
    assert result["outcome"] == "failed"
    receipts = list((tmp_path / "home" / "logs" / "update_receipts").glob("pm_*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text()) == result
    assert not paths.facts_path().exists()


def test_invalid_selection_waits_for_failed_receipt_and_lock_release(client, tmp_path, monkeypatch):
    import json
    from pm.environments import install_state_dir
    from hermes_cli.runtime_state import _lock

    repo = _current_environment(tmp_path, monkeypatch, [])
    home = tmp_path / "home"
    (home / "config.yaml").write_text("plugins: []\n")
    with pytest.raises(ValueError, match="plugins must be a mapping"):
        client.sync_venv([], explicit=True, selection={"home": str(home), "enabled": [], "disabled": []})
    receipts = list((home / "logs" / "update_receipts").glob("pm_*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["outcome"] == "failed"
    with (install_state_dir(repo) / ".install.lock").open("a+b") as lock:
        assert _lock(lock.fileno(), wait=False)


def _node_archive(server, body=b"#!/bin/sh\nexit 0\n"):
    import hashlib
    import io
    import zipfile
    from pm.lock import Lockfile
    from pm.registry import get_package
    from pm.store import current_target

    target = current_target()
    package = get_package("node")
    relative = package.binary(Path("."), target)
    assert relative is not None
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo((Path("node-package") / relative).as_posix())
        info.external_attr = 0o100755 << 16
        archive.writestr(info, body)
    payload = stream.getvalue()
    RangeHandler.payloads["/node.zip"] = payload
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin("node", "1", {target: {"url": url(server, "/node.zip"),
                                      "sha256": hashlib.sha256(payload).hexdigest()}})
    lock.save()
    return target, relative, body


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("operation", ["ensure", "stage_only"])
def test_worker_artifact_lifecycle_keeps_identity_and_relays_progress(client, dl_server, operation):
    import hashlib
    from pm.lock import Facts, Lockfile
    from pm.store import tree_digest

    target, relative, body = _node_archive(dl_server)
    stages, downloads = [], []

    def progress(*args):
        stages.append(args)
        return object()  # Notifications never serialize caller-owned return values.

    if operation == "stage_only":
        entry = client.stage_only("node", target, progress=progress)
        assert isinstance(entry, Path)
        assert not paths.facts_path().exists()
    else:
        base = {"PATH": "/caller/bin", "CALLER": "kept", "PYTHONPATH": "/caller/dependencies"}
        runner = client.ensure("node", explicit=True, base_env=base,
                               progress=lambda *args: stages.append(args),
                               download_progress=lambda *args: downloads.append(args))
        fact = Facts(paths.facts_path()).get("node")
        entry = paths.store_root() / fact["entry"]
        assert runner.env["CALLER"] == "kept"
        assert runner.env["PYTHONPATH"] == base["PYTHONPATH"]
        assert runner.env["PATH"].endswith(base["PATH"])
        assert downloads and downloads[-1][0] == downloads[-1][1]
        assert all(isinstance(row, tuple) for rows in downloads[-1][2].values() for row in rows)
    assert (entry / relative).read_bytes() == body
    assert stages and stages[0][0] == "download"

    def install():
        if operation == "stage_only":
            return client.stage_only("node", target)
        client.ensure("node", explicit=True)
        return entry

    first_tree = tree_digest(entry)
    RangeHandler.payloads.clear()
    assert install() == entry
    assert tree_digest(entry) == first_tree
    _, _, replacement = _node_archive(dl_server, b"#!/bin/sh\nexit 0\n# repinned\n")
    assert install() == entry
    assert (entry / relative).read_bytes() == replacement
    assert not list(paths.store_root().glob("fetch-*"))
    good_tree = tree_digest(entry)
    previous_facts = paths.facts_path().read_bytes() if paths.facts_path().exists() else None
    lock = Lockfile(paths.lockfile_path())
    RangeHandler.payloads["/node.zip"] = b"not an archive"
    lock.set_pin("node", "1", {target: {"url": url(dl_server, "/node.zip"),
                                      "sha256": hashlib.sha256(b"not an archive").hexdigest()}})
    lock.save()
    with pytest.raises(RuntimeError if operation == "stage_only" else InstallError, match="zip"):
        install()
    assert tree_digest(entry) == good_tree
    if operation == "stage_only":
        assert not paths.facts_path().exists(), "foreign staging cannot publish host identity"
    else:
        assert paths.facts_path().read_bytes() == previous_facts


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("from_progress", [False, True])
def test_pause_event_reaches_running_worker_without_hanging(client, dl_server, monkeypatch, from_progress):
    import threading
    from pm.downloader import DownloadPaused
    from pm.lock import Facts

    _node_archive(dl_server, b"#!/bin/sh\nexit 0\n#" + b"x" * (8 << 20))
    RangeHandler.slow_per_chunk = 0.03
    pause, transferring = threading.Event(), threading.Event()
    original = RangeHandler.do_GET

    def get(handler):
        if handler.headers.get("Range") not in (None, "bytes=0-0"):
            transferring.set()
        original(handler)

    monkeypatch.setattr(RangeHandler, "do_GET", get)

    def cancel():
        assert transferring.wait(10), "worker never began transferring"
        pause.set()

    thread = None
    progress = None
    if from_progress:
        def progress(stage, done, total, label):
            if stage == "download" and done:
                pause.set()
    else:
        thread = threading.Thread(target=cancel)
        thread.start()
    try:
        with pytest.raises(DownloadPaused):
            client.ensure("node", explicit=True, progress=progress, pause_event=pause)
    finally:
        if thread is not None:
            thread.join(timeout=15)
            assert not thread.is_alive()
    assert Facts(paths.facts_path()).get("node") is None


@pytest.mark.platforms("posix")
def test_installed_tool_needs_no_pm_runtime(client, dl_server, monkeypatch):
    _node_archive(dl_server)
    client.ensure("node", explicit=True)
    monkeypatch.setattr("pm.runtime.runtime_python", lambda: pytest.fail("hot path bootstrapped PM"))
    assert client.ensure("node", base_env={"PATH": "caller"}).env["PATH"].endswith("caller")


def test_environment_probe_needs_no_pm_runtime(client, monkeypatch, tmp_path):
    from pm import environment_python, python_tool
    monkeypatch.setattr("pm.runtime.runtime_python", lambda: pytest.fail("probe bootstrapped PM"))
    root = tmp_path / "not-created"
    assert environment_python("test", root=root) is None
    assert python_tool("test", "test", root=root) is None
    assert not root.exists()


def test_worker_receipt_is_exact_even_if_latest_is_replaced(client, tmp_path, monkeypatch):
    import json
    from pm import receipt

    _current_environment(tmp_path, monkeypatch, [])
    original_accept = receipt.accept_worker_receipt
    received = []

    def accept(data, update_id):
        # Simulate a different process publishing after this worker completes.
        point = tmp_path / "home" / "logs" / "update_receipts" / "latest.json"
        point.write_text(json.dumps({"update_id": "unrelated", "outcome": "failed"}))
        received.append(data)
        original_accept(data, update_id)

    monkeypatch.setattr(receipt, "accept_worker_receipt", accept)
    with receipt.worker_context("my-update"):
        client.sync_venv([], explicit=True, plugin_dirs=[])
        result = receipt.last_for_update("my-update", consume=True)
    assert received and result == received[0]
    assert result["update_id"] == "my-update" and result["outcome"] == "ok"


def _patch_worker_apply(client, monkeypatch, isolated_python, body):
    """Fault injection at the build boundary, not an alternate transport/engine."""
    import textwrap

    worker = Path(client.__file__).with_name("worker.py")
    script = (
        "import os, runpy, sys\n"
        f"sys.path.insert(0, {str(worker.parent.parent)!r})\n"
        "from pm.packages import Venv\n"
        "from pm.workspace import ResolutionConflict\n"
        "def apply(self, *args, **kwargs):\n"
        + textwrap.indent(body, "    ") + "\n"
        "Venv.apply = apply\n"
        f"runpy.run_path({str(worker)!r}, run_name='__main__')\n"
    )
    monkeypatch.setattr(client, "runtime_command", lambda path, **kwargs: [str(isolated_python), "-I", "-B", "-c", script])


def test_resolution_conflict_survives_worker_and_receipt(client, tmp_path, monkeypatch, isolated_python, capfd):
    from pm import receipt
    from pm.workspace import ResolutionConflict

    repo = _current_environment(tmp_path, monkeypatch, [])
    (repo / "uv.lock").write_text("version = 2\n")
    _patch_worker_apply(client, monkeypatch, isolated_python,
                        "print('engine stdout', flush=True)\n"
                        "os.write(1, b'native stdout\\n')\n"
                        "raise ResolutionConflict('venv', 'impossible union', 'change member')")
    with receipt.worker_context("conflict-update"):
        with pytest.raises(ResolutionConflict) as caught:
            client.sync_venv([], explicit=True, plugin_dirs=[])
        result = receipt.last_for_update("conflict-update", consume=True)
    assert (caught.value.package, caught.value.cause, caught.value.remedy) == (
        "venv", "impossible union", "change member")
    assert result["outcome"] == "failed"
    assert "engine stdout" in capfd.readouterr().err


def test_failed_facts_write_restores_exact_config_before_reporting(client, tmp_path, monkeypatch, isolated_python):
    from tests.pm._fixtures import worker_toolchain
    from pm.environments import install_state_dir

    repo = _current_environment(tmp_path, monkeypatch, [])
    home = tmp_path / "home"
    config = home / "config.yaml"
    config.write_text("# preserve me\nplugins: {enabled: [old]}\n")
    previous = config.read_bytes()
    facts = (install_state_dir(repo) / "facts.json").read_bytes()
    (repo / "uv.lock").write_text("version = 2\n")
    worker_toolchain(client, monkeypatch, isolated_python,
        "from pm.packages import Venv\nfrom pm.lock import Facts\n"
        "Venv.apply = lambda *args, **kwargs: {}\n"
        "def fail(*args, **kwargs):\n"
        f"    assert b'new' in Path({str(config)!r}).read_bytes()\n"
        "    raise OSError('facts disk full')\nFacts.record_state = fail\n")
    with pytest.raises(OSError, match="facts disk full"):
        client.sync_venv(explicit=True, selection={"home": str(home), "enabled": ["new"], "disabled": []})
    assert config.read_bytes() == previous
    assert (install_state_dir(repo) / "facts.json").read_bytes() == facts
    assert not (install_state_dir(repo) / "publication.json").exists()


def test_invalid_arguments_keep_the_engine_exception_type(client):
    with pytest.raises(ValueError, match="repair restores"):
        client.sync_venv([], repair=True)
    with pytest.raises(ValueError, match="environment name"):
        client.ensure_environment("../escape", ["example==1"], explicit=True)
    with pytest.raises(KeyError):
        client.ensure("no-such-package", explicit=True)


def test_worker_death_reports_transport_failure(client, monkeypatch, isolated_python):
    monkeypatch.setattr(client, "runtime_command", lambda path, **kwargs: [str(isolated_python), "-I", "-c", "import os; os._exit(7)"])
    with pytest.raises(InstallError, match="worker.*result"):
        client.ensure("node", explicit=True)


def test_foreign_checkout_sync_uses_its_own_pm_generation(client, tmp_path, monkeypatch, isolated_python):
    from pm import venv_is_current
    from pm.environments import selected_venv, runtime_facts_path

    from tests.pm._fixtures import worker_toolchain
    worker_toolchain(client, monkeypatch, isolated_python)
    foreign = tmp_path / "other checkout"
    foreign.mkdir()
    (foreign / "pyproject.toml").write_text(
        '[project]\nname="foreign-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\n', encoding="utf-8")
    client.lock_project(foreign, offline=True, explicit=True)
    assert not venv_is_current(project_root=foreign)
    original_root = paths.repo_root()
    client.sync_venv([], project_root=foreign, plugin_dirs=[], explicit=True)
    selected = selected_venv(foreign)
    assert selected != foreign / "venv"
    assert selected.is_relative_to(runtime_facts_path(foreign).parent)
    assert runtime_facts_path(foreign).is_file()
    assert venv_is_current(project_root=foreign)
    assert paths.repo_root() == original_root
    assert not runtime_facts_path(original_root).exists()


def test_cold_manager_build_does_not_bootstrap_a_worker(client, tmp_path, monkeypatch, locked_wheelhouse):
    import pm

    wheels, _ = locked_wheelhouse

    uv = shutil.which("uv")
    assert uv
    monkeypatch.setattr(client, "runtime_command", lambda *a, **kw: pytest.fail("manager build requested itself"))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable))
                        if kwargs == {"realize": False} else pytest.fail("bootstrap tried to install tools"))
    project = Path(__file__).resolve().parents[2] / "pm"
    python = pm.stage_manager_runtime(python=Path(sys.executable), destination=tmp_path / "manager",
                                     project=project, wheelhouse=wheels, offline=True)
    result = subprocess.run([str(python), "-I", "-c", "import packaging, tomli_w, truststore; from ruamel.yaml import YAML"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    with pytest.raises(FileExistsError):
        pm.stage_manager_runtime(python=Path(sys.executable), destination=python.parent.parent)


def test_worker_side_environment_reuses_and_keeps_selection_on_failed_tool(client, tmp_path, monkeypatch, isolated_python):
    import zipfile
    from pm import environment_python, python_tool

    from tests.pm._fixtures import worker_toolchain
    worker_toolchain(client, monkeypatch, isolated_python)
    wheel = _wheel(tmp_path, "side_dep")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("side_dep/cli.py", "def main():\n    print('real side dependency')\n")
        archive.writestr("side_dep-1.0.dist-info/entry_points.txt", "[console_scripts]\nside-proof = side_dep.cli:main\n")
    requirements = [f"side-dep @ {wheel.as_uri()}"]
    root = tmp_path / "side environment"
    assert environment_python("proof", root=root) is None
    executable = client.ensure_python_tool("proof", requirements, "side-proof", root=root, explicit=True)
    assert python_tool("proof", "side-proof", root=root) == executable
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "real side dependency"
    selected = environment_python("proof", root=root)
    selection = (root / "active.json").read_bytes()
    generations = set(root.glob("gen-*"))
    assert client.ensure_environment("proof", requirements, root=root) == selected
    assert set(root.glob("gen-*")) == generations
    with pytest.raises(InstallError, match="do not provide"):
        client.ensure_python_tool("proof", requirements, "missing-command", root=root, explicit=True)
    assert (root / "active.json").read_bytes() == selection
    assert set(root.glob("gen-*")) == generations
    assert python_tool("proof", "side-proof", root=root) == executable
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        client.ensure_environment("proof", ["absent-dependency==0"], root=root)
    assert (root / "active.json").read_bytes() == selection


@pytest.mark.parametrize("operation", ["activate", "stage_manager_runtime"])
@pytest.mark.parametrize("route", ["client", "wire"])
def test_unknown_worker_operation_is_not_dispatched(client, monkeypatch, tmp_path, isolated_python, operation, route):
    import json

    arguments = {"destination": str(tmp_path / "unused")}
    if route == "client":
        monkeypatch.setattr(client, "runtime_command", lambda *a, **kw: pytest.fail("unsupported operation acquired PM"))
        with pytest.raises(KeyError, match=operation):
            client._request(operation, arguments)
    else:
        request = {"id": "unsupported", "operation": operation, "arguments": arguments,
                   "callbacks": [], "packages": [],
                   "context": {"repo": str(paths.repo_root()), "lockfile": str(paths.lockfile_path())}}
        worker = Path(client.__file__).with_name("worker.py")
        result = subprocess.run(client.runtime_command(worker), input=json.dumps(request) + "\n",
                                capture_output=True, text=True, encoding="utf-8", timeout=30,
                                env=client.runtime_environment())
        assert result.returncode == 0, result.stderr
        response = json.loads(result.stdout)
        assert response["error"]["type"] == "KeyError", response
        assert operation in response["error"]["message"]
    assert not paths.facts_path().exists()
