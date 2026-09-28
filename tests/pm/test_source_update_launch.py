"""Source-update launch completion must publish a real PM generation.

Only tool acquisition is substituted: the isolated worker uses the test host's
uv/Python. Currency checks, resolution, installation, validation, facts and
selection all run through production PM. The shared worker fixture stages PM's
small locked dependency graph; the application project needs no downloads.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import pm
from hermes_cli import venv_sync
from pm.environments import install_state_dir, runtime_facts_path, selected_venv, site_packages
from pm import paths
from pm.lock import Facts
from pm.package import InstallError
from tests.pm._fixtures import isolated_python  # noqa: F401

# Spawns children with a home it builds itself; the parent's must stay real.
pytestmark = pytest.mark.real_machine_home


@pytest.fixture
def source_launch(tmp_path, monkeypatch, isolated_python):
    client = importlib.import_module("pm.client")
    engine = importlib.import_module("pm.install")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "tool-lock.json")

    uv = shutil.which("uv")
    assert uv, "source launch integration requires real uv"
    worker = Path(client.__file__).with_name("worker.py")
    worker_code = (
        "import runpy, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(worker.parent.parent)!r})\n"
        "import pm._uv\n"
        f"pm._uv._toolchain = lambda **kwargs: (Path({uv!r}), Path({sys.executable!r}))\n"
        f"runpy.run_path({str(worker)!r}, run_name='__main__')\n"
    )
    command = [str(isolated_python), "-I", "-B", "-c", worker_code]
    monkeypatch.setattr(client, "runtime_command", lambda *args, **kwargs: command)
    monkeypatch.setattr(engine, "sync_venv", lambda *args, **kwargs: pytest.fail("sync escaped worker isolation"))

    root = tmp_path / "source with spaces"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "install-stamp.json").write_text(
        json.dumps({"updateMechanism": "self"}), encoding="utf-8",
    )
    # The startup heal hands the shared completion tail (launchers, products,
    # maintenance) to the checkout's own hermes_cli/source_completion.py. This
    # source slice has no products; record the hand-off instead of running it.
    (root / "hermes_cli").mkdir()
    (root / "hermes_cli" / "source_completion.py").write_text(
        "import json, sys\n"
        f"open({str(tmp_path / 'completion-calls')!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "launch-proof"\nversion = "1"\nrequires-python = ">=3.11"\n'
        '[project.optional-dependencies]\nall = []\nlaunch-extra = []\n'
        '[tool.uv]\npackage = false\nno-index = true\noffline = true\n', encoding="utf-8",
    )
    pm.lock_project(root, offline=True, explicit=True)

    # Acquisition uses the real host interpreter; publish its installed fact
    # through PM too. Unrecorded store bytes are deliberately not launchable.
    from pm.store import current_target, tree_digest

    store = paths.store_root()
    entry = store / "python-test"
    store_python = entry / "bin" / "python3"
    store_python.parent.mkdir(parents=True)
    # A Nix wrapper resets sys.executable; use the underlying binary instead.
    store_python.symlink_to(sys._base_executable)
    Facts(paths.facts_path()).record(
        "python", "test", entry.name, {"PATH": [str(store_python.parent)]}, store,
        target=current_target(), digest=tree_digest(entry),
    )
    return root, store_python, command


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("update", ["launch", "sync", "pm-update"])
def test_source_python_pin_update_survives_real_gc(source_launch, tmp_path, monkeypatch, update):
    from hermes_cli import _launchers
    from pm.cli import cmd_gc
    from pm.lock import Lockfile
    from pm.store import current_target, tree_digest
    from tests.hermes_cli.test_source_launcher_publication import BOOT_FILES

    root, old_python, _ = source_launch
    repository = Path(__file__).resolve().parents[2]
    for relative in BOOT_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, destination)
    (root / "source_probe.py").write_text(
        "import json, sys, selected_probe\n"
        "print(json.dumps({'executable': sys.executable, 'value': selected_probe.VALUE}))\n",
        encoding="utf-8",
    )
    pin = Lockfile(paths.lockfile_path())
    pin.set_pin("python", "A", {})
    pin.save()
    pm.sync_venv(explicit=True, project_root=root)
    # Initial installation is allowed to publish. No explicit writer is called
    # after replacement: the source-update owner must refresh this same command.
    _launchers.ensure_install_launchers(root, root / ".hermes" / "bin")
    command = _launchers.installation_command(root, module="source_probe")
    (site_packages(selected_venv(root)) / "selected_probe.py").write_text("VALUE = 'A'\n")
    child_env = {**os.environ, "HERMES_DISABLE_LAZY_INSTALLS": "1"}
    before = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=30)
    assert before.returncode == 0, before.stderr
    assert json.loads(before.stdout)["executable"] == str(old_python)

    # Only acquisition is substituted. This is a new real interpreter entry,
    # a changed tool pin and a real Facts publication, not a launcher rewrite.
    store = paths.store_root()
    new_python = store / "python-B" / "bin" / "python3"
    new_python.parent.mkdir(parents=True)
    new_python.symlink_to(sys._base_executable)
    Facts(paths.facts_path()).record(
        "python", "B", "python-B", {"PATH": [str(new_python.parent)]}, store,
        target=current_target(), digest=tree_digest(new_python.parent.parent),
    )
    if update != "pm-update":
        pin.set_pin("python", "B", {})
        pin.save()
        assert not pm.venv_is_current(project_root=root)
    previous = selected_venv(root)
    if update == "launch":
        assert venv_sync.prepare_launch(root, []) == new_python
        # The heal finished the whole tail, with update wording, through the checkout's own completion.
        calls = [json.loads(line) for line in (tmp_path / "completion-calls").read_text().splitlines()]
        assert calls == [["--source", str(root), "--finish-update"]]
    elif update == "sync":
        # This is the PM sync -> launcher publication sequence now split across
        # update_completion._prepare and _complete_selected. The former checkout
        # case adds no distinct PM/GC coverage; the fresh-interpreter handoff is
        # exercised separately in test_update_completion_process.py.
        assert venv_sync.sync(root) == {"state": "synced", "ok": True}
    else:
        from types import SimpleNamespace
        from pm import cli
        from pm.update import Resolved

        monkeypatch.setattr(paths, "repo_root", lambda: root)
        monkeypatch.setattr(cli, "repo_root", lambda: root)
        # The latest release and artifact acquisition are fixture inputs; the
        # CLI must still pin, ensure, synchronize and publish through its owners.
        monkeypatch.setattr(cli, "resolve_package", lambda *a, **k: Resolved("python", "A", "semver", "B"))
        monkeypatch.setattr(cli, "_pin_artifacts", lambda *a: {})
        monkeypatch.setattr(importlib.import_module("pm.install"), "sync_venv", pm.sync_venv)
        assert cli.cmd_update(SimpleNamespace(names=["python"], target=None, check=False, uv=False, npm=False, termux=False)) == 0
        assert Lockfile(paths.lockfile_path()).version("python") == "B"
    assert pm.venv_is_current(project_root=root)
    assert selected_venv(root) != previous
    (site_packages(selected_venv(root)) / "selected_probe.py").write_text("VALUE = 'B'\n")
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    assert cmd_gc(None) == 0
    assert not old_python.exists(), "real PM GC did not remove the superseded Python"
    assert new_python.is_file()
    after = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=30)
    assert after.returncode == 0, after.stderr
    assert json.loads(after.stdout) == {"executable": str(new_python), "value": "B"}


@pytest.mark.platforms("posix")
def test_launcher_publication_failure_retries_without_rebuilding_dependencies(source_launch, monkeypatch):
    from hermes_cli import _launchers

    root, store_python, _ = source_launch
    with monkeypatch.context() as failed_publication:
        failed_publication.setattr(_launchers, "ensure_install_launchers", lambda *a: [])
        result = venv_sync.sync(root)
    assert not result["ok"] and "launcher publication failed" in result["detail"]
    assert pm.venv_is_current(project_root=root)
    committed = runtime_facts_path(root).read_bytes()
    assert venv_sync.prepare_launch(root, []) == store_python
    assert runtime_facts_path(root).read_bytes() == committed
    assert (root / ".hermes" / "bin" / "hermes").is_file()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("checkout", [False, True])
def test_source_publication_leaves_external_install_launchers_alone(source_launch, checkout):
    root, _, _ = source_launch
    (root / "install-stamp.json").write_text(
        json.dumps({"updateMechanism": "external", "distribution": "nix"}), encoding="utf-8",
    )
    if not checkout:
        (root / ".git").rmdir()
    launcher = root / ".hermes" / "bin" / "hermes"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("externally owned launcher\n", encoding="utf-8")
    assert venv_sync.sync(root)["ok"]
    assert launcher.read_text(encoding="utf-8") == "externally owned launcher\n"


def _fact(root):
    fact = Facts(runtime_facts_path(root), strict=True).get("venv")
    assert fact is not None
    selected = selected_venv(root)
    assert Path(fact["environment"]) == selected
    assert selected.is_relative_to(install_state_dir(root) / "environments")
    assert (selected / "pyvenv.cfg").is_file()
    assert Path(fact["resolved_lock"]).is_file()
    probe = subprocess.run(
        [str(selected / "bin" / "python"), "-I", "-c", "import json,sys; print(json.dumps(sys.prefix))"],
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    assert Path(json.loads(probe.stdout)) == selected
    return fact


def _receipts(tmp_path):
    return set((tmp_path / "home" / "logs" / "update_receipts").glob("pm_*.json"))


@pytest.mark.platforms("posix")
def test_launch_without_marker_publishes_then_skips_and_rebuilds_on_lock_change(source_launch, tmp_path):
    root, store_python, _ = source_launch
    assert not runtime_facts_path(root).exists()
    assert not (root / ".update-incomplete").exists()
    assert not pm.venv_is_current(project_root=root)

    assert venv_sync.prepare_launch(root, []) == store_python
    first = _fact(root)
    assert first["extras"] == ["all"]
    assert pm.venv_is_current(project_root=root)
    facts_bytes = runtime_facts_path(root).read_bytes()
    generations = set((install_state_dir(root) / "environments").iterdir())
    receipts = _receipts(tmp_path)
    assert receipts

    # A caller still in its old interpreter must re-exec, but must not sync again.
    assert venv_sync.prepare_launch(root, []) == store_python
    assert runtime_facts_path(root).read_bytes() == facts_bytes
    assert set((install_state_dir(root) / "environments").iterdir()) == generations
    assert _receipts(tmp_path) == receipts, "current launch performed another sync"

    lock = root / "uv.lock"
    lock.write_bytes(lock.read_bytes() + b"\n# source update changes the committed lock\n")
    assert not pm.venv_is_current(project_root=root)
    assert venv_sync.prepare_launch(root, []) == store_python
    rebuilt = _fact(root)
    assert rebuilt["stamp"] != first["stamp"]
    assert rebuilt["environment"] != first["environment"]
    assert rebuilt["extras"] == first["extras"]
    assert Path(first["environment"]).is_dir()
    assert pm.venv_is_current(project_root=root)
    assert not (root / ".update-incomplete").exists()


@pytest.mark.platforms("posix")
def test_process_spawned_by_the_update_commits_dependencies_but_not_the_tail(source_launch, tmp_path):
    """A process an update spawns before its dependencies are current (its restarted gateway)
    must not boot on a tree built for another interpreter; it syncs, but leaves the tail alone."""
    import time
    from hermes_cli.update_lock import update_marker_path
    from pm.environments import committed_venv

    root, store_python, _ = source_launch
    marker = update_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{os.getppid()}\n{int(time.time())}\n", encoding="utf-8")  # the updater is our ancestor
    assert committed_venv(root) is None

    assert venv_sync.prepare_launch(root, []) == store_python
    assert committed_venv(root) == Path(_fact(root)["environment"])
    assert not (tmp_path / "completion-calls").exists(), "the tail is the updater's, not its child's"
    assert not venv_sync.completion_pending_path(root).exists()

    facts_bytes, receipts = runtime_facts_path(root).read_bytes(), _receipts(tmp_path)
    venv_sync.prepare_launch(root, [])
    assert runtime_facts_path(root).read_bytes() == facts_bytes
    assert _receipts(tmp_path) == receipts, "a committed child synced again under the updater's claim"


@pytest.mark.platforms("posix")
def test_failed_real_sync_preserves_previous_selection_and_retries(source_launch, tmp_path):
    root, store_python, _ = source_launch
    # Established PM installs must retain their selection, not gain legacy [all].
    pm.sync_venv(["launch-extra"], explicit=True, project_root=root)
    previous = _fact(root)
    facts_bytes = runtime_facts_path(root).read_bytes()
    generations = set((install_state_dir(root) / "environments").iterdir())
    lock = root / "uv.lock"
    valid_lock = lock.read_bytes()
    lock.write_text("version = 1\nnot valid TOML\n", encoding="utf-8")
    markers = [root / name for name in (".update-incomplete", ".lazy-refresh-incomplete")]
    for marker in markers:
        marker.write_text("legacy pending install", encoding="utf-8")

    for _ in range(2):
        receipts = _receipts(tmp_path)
        with pytest.raises(InstallError, match="uv sync exited"):
            venv_sync.prepare_launch(root, [])
        failed, = _receipts(tmp_path) - receipts
        assert json.loads(failed.read_text(encoding="utf-8"))["outcome"] == "failed"
        assert runtime_facts_path(root).read_bytes() == facts_bytes
        assert _fact(root) == previous
        assert set((install_state_dir(root) / "environments").iterdir()) == generations
        assert not pm.venv_is_current(project_root=root)
        assert all(marker.read_text(encoding="utf-8") == "legacy pending install" for marker in markers)

    lock.write_bytes(valid_lock + b"\n# corrected source update\n")
    assert venv_sync.prepare_launch(root, []) == store_python
    rebuilt = _fact(root)
    assert rebuilt["environment"] != previous["environment"]
    assert rebuilt["extras"] == ["launch-extra"]
    assert pm.venv_is_current(project_root=root)
    assert not any(marker.exists() for marker in markers)


@pytest.mark.platforms("posix")
def test_source_update_that_removes_a_recorded_extra_still_syncs(source_launch):
    """A dropped extra (hindsight, 73c598e319) must not brick every later sync."""
    root, store_python, _ = source_launch
    pm.sync_venv(["all", "launch-extra"], explicit=True, project_root=root)
    assert _fact(root)["extras"] == ["all", "launch-extra"]

    pyproject = root / "pyproject.toml"
    pyproject.write_text(pyproject.read_text(encoding="utf-8").replace("launch-extra = []\n", ""),
                         encoding="utf-8")
    pm.lock_project(root, offline=True, explicit=True)
    assert not pm.venv_is_current(project_root=root)

    assert venv_sync.prepare_launch(root, []) == store_python
    assert _fact(root)["extras"] == ["all"]
    assert pm.venv_is_current(project_root=root)


@pytest.mark.platforms("posix")
def test_recorded_extra_spelled_differently_from_its_declaration_survives(source_launch):
    """uv matches extras by PEP 685 name; `launch_extra` is the declared `launch-extra`."""
    root, store_python, _ = source_launch
    pm.sync_venv(["all", "launch_extra"], explicit=True, project_root=root)
    assert _fact(root)["extras"] == ["all", "launch_extra"]

    lock = root / "uv.lock"
    lock.write_bytes(lock.read_bytes() + b"\n# source update changes the committed lock\n")
    assert venv_sync.prepare_launch(root, []) == store_python
    assert _fact(root)["extras"] == ["all", "launch_extra"]
    assert pm.venv_is_current(project_root=root)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("mode", ["script", "module", "command"])
def test_real_bootstrap_reexecs_before_app_imports(source_launch, tmp_path, isolated_python, mode):
    root, store_python, worker_command = source_launch
    repository = Path(__file__).resolve().parents[2]
    # Copy the real bootstrap so it owns this disposable source install. The
    # other modules remain real checkout imports; only acquisition is injected.
    shutil.copy2(repository / "hermes_bootstrap.py", root / "hermes_bootstrap.py")
    (root / "launch_test_tools.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(1, {str(repository)!r})\n"
        "import pm.client\n"
        "from pm import paths\n"
        f"paths.lockfile_path = lambda: Path({str(tmp_path / 'tool-lock.json')!r})\n"
        f"pm.client.runtime_command = lambda *args, **kwargs: {worker_command!r}\n",
        encoding="utf-8",
    )
    entry = root / "launch_probe.py"
    entry.write_text(
        "import launch_test_tools\n"
        "import hermes_bootstrap\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "from pm.environments import selected_venv, site_packages\n"
        "from hermes_cli.venv_sync import prepare_launch\n"
        "root = Path(__file__).parent\n"
        "selected = selected_venv(root)\n"
        "print(json.dumps({'executable': sys.executable, 'args': sys.argv[1:],\n"
        "    'site': str(site_packages(selected)), 'path': sys.path,\n"
        "    'module': getattr(__spec__, 'name', None),\n"
        "    'launch_current': prepare_launch(root, sys.argv[1:]) is None}))\n",
        encoding="utf-8",
    )
    args = ["--profile", "name with spaces", "-c", "session"]
    invocation = {
        "script": [str(entry)],
        "module": ["-m", "launch_probe"],
        "command": ["-c", "import launch_probe"],
    }[mode]
    assert not runtime_facts_path(root).exists()
    activated_old = tmp_path / "activated-old-dependencies"
    if mode == "script":
        pm.sync_venv(["all"], explicit=True, project_root=root)
        old_site = site_packages(selected_venv(root))
        # Executable .pth files are real activation hooks. If bootstrap selects
        # the old generation before completing the update, this leaves evidence.
        (old_site / "old_dependency.pth").write_text(
            f"import pathlib; pathlib.Path({str(activated_old)!r}).touch()\n", encoding="utf-8",
        )
        activation_probe = subprocess.run(
            [str(isolated_python), "-I", "-c",
             f"import sys; sys.path.insert(0, {str(repository)!r}); "
             "from pathlib import Path; from pm.environments import activate_dependencies; "
             f"activate_dependencies(Path({str(root)!r}))"],
            capture_output=True, text=True, timeout=30,
        )
        assert activation_probe.returncode == 0, activation_probe.stderr
        assert activated_old.is_file(), "positive control did not execute the old activation hook"
        activated_old.unlink()
        lock = root / "uv.lock"
        lock.write_bytes(lock.read_bytes() + b"\n# update before activation\n")
    if mode == "module":
        # These leftovers must not trigger an immediate second (repair) sync
        # after launch completion, which would validate the full app graph.
        for name in (".update-incomplete", ".lazy-refresh-incomplete"):
            (root / name).write_text("legacy pending install", encoding="utf-8")
    before_receipts = _receipts(tmp_path)
    result = subprocess.run(
        [str(isolated_python), *invocation, *args], cwd=root,
        env=dict(os.environ), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["executable"] == str(store_python)
    assert output["args"] == args
    assert output["launch_current"] is True
    assert output["module"] == (None if mode == "script" else "launch_probe")
    selected = selected_venv(root)
    assert Path(output["site"]).is_relative_to(selected)
    assert output["site"] in output["path"]
    assert not activated_old.exists(), "old dependencies activated before source-update completion"
    assert _fact(root)["extras"] == ["all"]
    receipt, = _receipts(tmp_path) - before_receipts
    assert json.loads(receipt.read_text(encoding="utf-8"))["outcome"] == "ok"
    assert not (root / ".update-incomplete").exists()
    assert not (root / ".lazy-refresh-incomplete").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("argv", [[], ["--version"]])
def test_failed_launch_completion_degrades_to_a_warning(source_launch, tmp_path, isolated_python, argv):
    """An update whose dependency sync cannot finish (offline, bad lock) must leave a usable
    CLI on the previous generation with a warning — and a metadata query must not even try."""
    root, store_python, worker_command = source_launch
    repository = Path(__file__).resolve().parents[2]
    shutil.copy2(repository / "hermes_bootstrap.py", root / "hermes_bootstrap.py")
    (root / "launch_test_tools.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(1, {str(repository)!r})\n"
        "import pm.client\n"
        "from pm import paths\n"
        f"paths.lockfile_path = lambda: Path({str(tmp_path / 'tool-lock.json')!r})\n"
        f"pm.client.runtime_command = lambda *args, **kwargs: {worker_command!r}\n",
        encoding="utf-8",
    )
    entry = root / "launch_probe.py"
    entry.write_text(
        "import launch_test_tools\n"
        "import hermes_bootstrap\n"
        "import json, sys\n"
        "print(json.dumps({'executable': sys.executable, 'args': sys.argv[1:]}))\n",
        encoding="utf-8",
    )
    pm.sync_venv(["all"], explicit=True, project_root=root)
    previous = _fact(root)
    lock = root / "uv.lock"
    lock.write_text("version = 1\nnot valid TOML\n", encoding="utf-8")
    assert not pm.venv_is_current(project_root=root)
    before_receipts = _receipts(tmp_path)

    result = subprocess.run(
        [str(isolated_python), str(entry), *argv], cwd=root,
        env=dict(os.environ), capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"executable": str(isolated_python), "args": argv}
    assert _fact(root) == previous
    if argv:
        assert "source-update completion failed" not in result.stderr
        assert _receipts(tmp_path) == before_receipts, "a metadata query attempted a dependency sync"
    else:
        assert "source-update completion failed" in result.stderr
        assert "hermes update" in result.stderr

