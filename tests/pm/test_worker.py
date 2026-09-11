"""Real isolated workers: no parent imports or callables cross the wire."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess

import venv

import pytest

from pm import paths
from pm.package import InstallError
from pm.runtime import runtime_python
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.fixture(scope="module")
def isolated_python(tmp_path_factory):
    root = tmp_path_factory.mktemp("pm-python")
    venv.EnvBuilder(with_pip=False).create(root)
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    probe = subprocess.run(
        [str(python), "-I", "-c", "import importlib.util; assert importlib.util.find_spec('yaml') is None"],
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    return python


@pytest.fixture
def client(tmp_path, monkeypatch, isolated_python):
    client = importlib.import_module("pm.client")
    monkeypatch.setattr("pm.runtime.runtime_python", lambda **kwargs: isolated_python)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    return client


def test_isolated_worker_preserves_install_error(client, monkeypatch):
    engine = importlib.import_module("pm.ensure")
    monkeypatch.setattr(engine, "ensure", lambda *a, **kw: pytest.fail("engine ran in caller"))
    with pytest.raises(InstallError) as caught:
        client.ensure("node", explicit=True)
    assert caught.value.package == "node"
    assert caught.value.cause == "not in the lockfile"
    assert caught.value.remedy == "add it with `hermes pm lock --bump`"
    assert not paths.facts_path().exists()


def test_realized_uv_returns_worker_selected_binaries_and_caller_env(client, tmp_path):
    from pm.lock import Facts, Lockfile
    from pm.registry import get_package
    from pm.store import current_target

    target = current_target()
    lock = Lockfile(paths.lockfile_path())
    facts = Facts(paths.facts_path())
    binaries = {}
    for name in ("python", "uv"):
        package = get_package(name)
        entry = paths.store_root() / package.store_entry("1", target)
        binary = package.binary(entry, target)
        assert binary is not None
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"installed binary fixture")
        if name == "uv":
            binary = binary.with_name("uvx" + binary.suffix)
            binary.write_bytes(b"installed uvx fixture")
        binaries[name] = str(binary)
        digest = "a" * 64
        lock.set_pin(name, "1", {target: {"url": "https://unused.invalid/archive", "sha256": digest}})
        facts.record(name, "1", entry.name, {}, paths.store_root(), target=target, artifacts=[digest])
    lock.save()
    binary, env = client.uv("uvx", venv=tmp_path / "project-env", base_env={"KEEP": "caller", "PYTHONPATH": "bad"})
    assert binary == binaries["uv"]
    assert env["UV_PYTHON"] == binaries["python"] and env["KEEP"] == "caller"
    assert env["VIRTUAL_ENV"] == str(tmp_path / "project-env")
    assert "PYTHONPATH" not in env


def test_refused_or_already_paused_install_does_not_acquire_runtime(client, monkeypatch):
    import threading
    from pm.downloader import DownloadPaused

    monkeypatch.setattr(client, "runtime_command", lambda path: pytest.fail("refusal acquired PM runtime"))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        client.ensure("node")
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        client.uv()
    paused = threading.Event()
    paused.set()
    with pytest.raises(DownloadPaused):
        client.ensure("node", explicit=True, pause_event=paused)


def _current_environment(tmp_path, monkeypatch, members):
    from hermes_cli.runtime_paths import install_state_dir
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


def _assert_worker_holds_lock(repo):
    from hermes_cli.runtime_paths import install_state_dir
    from hermes_cli.runtime_state import _lock

    with (install_state_dir(repo) / ".install.lock").open("a+b") as lock:
        assert not _lock(lock.fileno(), wait=False), "callback escaped the worker's runtime lock"


@pytest.mark.parametrize("explicit", [True, False])
def test_sync_callbacks_preserve_member_mapping_and_lock(client, tmp_path, monkeypatch, explicit):
    identity, staged = tmp_path / "installed", tmp_path / "staged"
    staged.mkdir()
    (staged / "plugin.yaml").write_text("name: test\n")
    members = {identity: staged}
    repo = _current_environment(tmp_path, monkeypatch, members)
    events = []

    def select():
        _assert_worker_holds_lock(repo)
        events.append("members")
        return members

    class Publication:
        def __call__(self):
            pytest.fail("successful no-op publication was undone")

        def finish(self):
            _assert_worker_holds_lock(repo)
            events.append("finish")

    def before_publish():
        _assert_worker_holds_lock(repo)
        events.append("publish")
        return Publication()

    client.sync_venv([], explicit=explicit, plugin_dirs=select, before_publish=before_publish)
    assert events == ["members", "publish", "finish"]


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
    engine = importlib.import_module("pm.ensure")
    original_uv = engine.uv

    def uv(*args, **kwargs):
        assert kwargs.get("realize") is False, "lazy-disabled sync bootstrapped tools"
        if tools_present:
            return str(tmp_path / "uv"), {"UV_PYTHON": str(isolated_python)}
        return original_uv(*args, **kwargs)

    monkeypatch.setattr(engine, "uv", uv)
    monkeypatch.setattr("pm.runtime_stage.stage_runtime",
                        lambda *a, **kw: pytest.fail("lazy-disabled sync prepared PM runtime"))
    selections = []

    def select():
        _assert_worker_holds_lock(repo)
        selections.append("selected")
        return []

    with receipt.worker_context("lazy-disabled-sync"):
        with pytest.raises(InstallError, match="lazy installs are disabled") as caught:
            client.sync_venv([], plugin_dirs=select)
        result = receipt.last_for_update("lazy-disabled-sync", consume=True)
    assert caught.value.package == "pm-runtime"
    assert selections == []
    assert result is not None
    assert result["outcome"] == "failed"
    receipts = list((tmp_path / "home" / "logs" / "update_receipts").glob("pm_*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text()) == result
    assert not paths.facts_path().exists()


def test_callback_exception_waits_for_failed_receipt_and_lock_release(client, tmp_path, monkeypatch):
    import json
    from hermes_cli.runtime_paths import install_state_dir
    from hermes_cli.runtime_state import _lock

    repo = _current_environment(tmp_path, monkeypatch, [])
    error = LookupError("selection disappeared")

    def fail():
        _assert_worker_holds_lock(repo)
        raise error

    with pytest.raises(LookupError) as caught:
        client.sync_venv([], explicit=True, plugin_dirs=fail)
    assert caught.value is error
    receipts = list((tmp_path / "home" / "logs" / "update_receipts").glob("pm_*.json"))
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
def test_worker_installs_real_archive_and_relays_progress(client, dl_server, operation):
    from pm.lock import Facts

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


def test_uv_probe_needs_no_pm_runtime(client, monkeypatch):
    monkeypatch.setattr("pm.runtime.runtime_python", lambda: pytest.fail("probe bootstrapped PM"))
    binary, env = client.uv(realize=False, venv=Path("project-venv"), base_env={"KEPT": "yes"})
    assert binary is None
    assert env["KEPT"] == "yes" and env["VIRTUAL_ENV"] == "project-venv"


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
    monkeypatch.setattr(client, "runtime_command", lambda path: [str(isolated_python), "-I", "-B", "-c", script])


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


def test_failed_finish_runs_undo_before_propagating_callback_exception(client, tmp_path, monkeypatch, isolated_python):
    repo = _current_environment(tmp_path, monkeypatch, [])
    (repo / "uv.lock").write_text("version = 2\n")
    _patch_worker_apply(client, monkeypatch, isolated_python, "return {}")
    events = []
    error = LookupError("finish failed")

    class Publication:
        def __call__(self):
            _assert_worker_holds_lock(repo)
            events.append("undo")
            return object()  # Return values of effect-only callbacks are ignored.

        def finish(self):
            _assert_worker_holds_lock(repo)
            events.append("finish")
            raise error

    def publish():
        _assert_worker_holds_lock(repo)
        events.append("publish")
        return Publication()

    with pytest.raises(LookupError) as caught:
        client.sync_venv([], explicit=True, plugin_dirs=[], before_publish=publish)
    assert caught.value is error
    assert events == ["publish", "finish", "undo"]


def test_invalid_arguments_keep_the_engine_exception_type(client):
    with pytest.raises(ValueError, match="repair restores"):
        client.sync_venv([], repair=True)
    with pytest.raises(ValueError, match="unknown uv executable"):
        client.uv("not-uv")
    with pytest.raises(KeyError):
        client.ensure("no-such-package", explicit=True)


def test_worker_death_reports_transport_failure(client, monkeypatch, isolated_python):
    monkeypatch.setattr(client, "runtime_command", lambda path: [str(isolated_python), "-I", "-c", "import os; os._exit(7)"])
    with pytest.raises(InstallError, match="worker.*result"):
        client.ensure("node", explicit=True)


def test_unknown_worker_operation_is_not_dispatched(client):
    with pytest.raises((KeyError, RuntimeError), match="activate"):
        client._request("activate", {})
    assert not paths.facts_path().exists()
