"""A fresh launch finishes a source update using PM's success record, not a marker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import venv_sync
from pm.environments import runtime_facts_path


@pytest.fixture(autouse=True)
def _no_tool_downloads(monkeypatch):
    """The launch sync publishes lockfile tools first; these tests cover the sync decision."""
    import pm.client

    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)


@pytest.fixture
def completion_tail(monkeypatch):
    """Record the source-completion child prepare_launch spawns after a sync instead of running it.

    The real child is ``hermes_cli/source_completion.py`` from the checkout under test — a
    scratch tree here — building products with the selected interpreter; the tests below
    cover the sync decision, not the build. ``interrupt_wait`` makes the next ``wait()``
    raise KeyboardInterrupt, as ^C does to the parent inside the real call.
    """
    class Spawned(list):
        exit_code = 0
        kwargs: dict = {}
        interrupt_wait = False
        terminated = False

    spawned = Spawned()

    class FakeTail:
        def __init__(self):
            self.args = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def wait(self, timeout=None):
            if spawned.interrupt_wait:
                spawned.interrupt_wait = False
                raise KeyboardInterrupt
            return spawned.exit_code

        def terminate(self):
            spawned.terminated = True

        def kill(self):
            pass

        def communicate(self, *args, **kwargs):
            return ("", "")

        def poll(self):
            return None

    def popen(command, **kwargs):
        spawned.append(command)
        spawned.kwargs = kwargs
        return FakeTail()

    monkeypatch.setattr(venv_sync.subprocess, "Popen", popen)
    return spawned


def _self_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n")
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self"}))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    return root


def _committed_checkout(tmp_path, monkeypatch):
    """A scratch source checkout whose completed stamp names its live HEAD.

    Mirrors a fresh official install: the installer ran the full tail and
    recorded the tree, so launching at the same commit owes no rebuild.
    """
    from hermes_cli import _launchers

    from hermes_cli.source_stamp import write_source_stamp

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n")
    (root / ".gitignore").write_text("install-stamp.json\n")
    for args in (["init"], ["config", "user.email", "test@example.com"],
                  ["config", "user.name", "test"], ["add", "-A"],
                  ["commit", "-m", "init"]):
        subprocess.run(["git", *args], cwd=root, check=True,
                         capture_output=True, timeout=30)
    assert write_source_stamp(root)["dirty"] is False
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))
    return root


@pytest.mark.parametrize("marker_age, owed", [(-60, False), (60, True)])
def test_completed_tree_discharges_only_a_marker_older_than_its_stamp(
        tmp_path, monkeypatch, completion_tail, marker_age, owed):
    """Fresh install under a preserved home (#123314): a marker armed before the stamp
    owes no rebuild of the same SHA. One armed after it (a same-commit `hermes update`
    that failed or was killed) is newer debt and the launch still finishes it."""
    import pm

    root = _committed_checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    pending = venv_sync.arm_completion(root)
    stamped = (root / "install-stamp.json").stat().st_mtime
    os.utime(pending, (stamped + marker_age, stamped + marker_age))
    assert venv_sync.prepare_launch(root, []) is None
    assert len(completion_tail) == (1 if owed else 0)
    assert not pending.exists()


@pytest.mark.parametrize("edited", [False, True])
def test_pristine_home_provisions_dependencies_without_rebuild(tmp_path, monkeypatch, completion_tail, edited):
    """Pristine HERMES_HOME on a freshly installed tree (#123314): with no facts the venv
    needs provisioning, but the stamp names a clean HEAD so no product rebuild is owed.
    Local edits since the stamp void that proof: the tail runs as before."""
    import pm

    root = _committed_checkout(tmp_path, monkeypatch)
    if edited:
        (root / "pyproject.toml").write_text("[project]\nname='edited'\n")
    assert not venv_sync.completion_pending_path(root).exists()
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)
    syncs = []
    monkeypatch.setattr(pm, "sync_venv", lambda *args, **kwargs: syncs.append((args, kwargs)))
    assert venv_sync.prepare_launch(root, []) == Path(sys.executable)
    assert len(syncs) == 1
    assert len(completion_tail) == (1 if edited else 0)
    assert not venv_sync.completion_pending_path(root).exists()


def test_adoption_stamp_never_discharges_an_owed_tail(tmp_path, monkeypatch, completion_tail):
    """Boot-time adoption stamps HEAD but builds nothing: an unstamped blessed root with a
    pending marker still owes its tail, on this launch and on every later one."""
    import pm

    root = _committed_checkout(tmp_path, monkeypatch)
    blessed = tmp_path / "home/hermes-agent"
    blessed.parent.mkdir(parents=True)
    root.rename(blessed)
    (blessed / "install-stamp.json").unlink()
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    pending = venv_sync.arm_completion(blessed)
    completion_tail.exit_code = 1
    for _ in range(2):
        with pytest.raises(RuntimeError, match="completion failed"):
            venv_sync.prepare_launch(blessed, [])
    assert "adoptedAt" in json.loads((blessed / "install-stamp.json").read_text())
    assert len(completion_tail) == 2
    assert pending.exists()


@pytest.mark.parametrize("argv", [["--version"], ["-V"], ["--help"], ["-p", "work", "-h"]])
def test_metadata_query_never_waits_on_source_completion(tmp_path, monkeypatch, argv):
    """`hermes --version` offline must answer from the tree, not run a network-bound sync."""
    import pm

    root = _self_checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: pytest.fail("metadata query reached PM"))
    assert venv_sync.prepare_launch(root, argv) is None


def test_failed_completion_tail_is_retried_without_rebuilding_dependencies(tmp_path, monkeypatch, completion_tail):
    """Dependencies committed, tail failed: the next launch owes the tail only."""
    import pm
    from hermes_cli import _launchers

    root = _self_checkout(tmp_path, monkeypatch)
    fact = runtime_facts_path(root)
    syncs = []
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: fact.is_file())
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))

    def sync(extras=None, **kwargs):
        syncs.append(extras)
        fact.parent.mkdir(parents=True, exist_ok=True)
        fact.write_text(json.dumps({"packages": {"venv": {"stamp": "complete", "extras": ["all"]}}}))

    monkeypatch.setattr(pm, "sync_venv", sync)
    completion_tail.exit_code = 1
    with pytest.raises(RuntimeError, match="run `hermes update`"):
        venv_sync.prepare_launch(root, [])
    assert len(syncs) == 1 and len(completion_tail) == 1
    assert pm.venv_is_current()

    with pytest.raises(RuntimeError, match="run `hermes update`"):
        venv_sync.prepare_launch(root, [])
    assert len(syncs) == 1, "current dependencies were rebuilt for a tail retry"
    assert len(completion_tail) == 2

    completion_tail.exit_code = 0
    # Age the recorded failure past the backoff window (#122206): a flaky
    # tail that a plain relaunch fixes is the historical case, but only once
    # the backoff window has elapsed — fresh failures wait it out.
    attempts = venv_sync._completion_attempts_path(root)
    old = 1.0
    os.utime(attempts, (old, old))
    # Dependencies are already this interpreter's: the tail alone owes no re-exec.
    assert venv_sync.prepare_launch(root, []) is None
    assert len(syncs) == 1 and len(completion_tail) == 3
    assert venv_sync.prepare_launch(root, []) is None
    assert len(completion_tail) == 3, "a finished tail was run again"


@pytest.mark.parametrize("script", ["source_completion.py", "update_completion.py"])
def test_prepared_completion_import_does_not_start_another_tail(tmp_path, monkeypatch, completion_tail, script):
    """Maintenance imports the CLI while its own completion marker is still present."""
    import pm
    from hermes_cli import _launchers

    root = _self_checkout(tmp_path, monkeypatch)
    worker = root / "hermes_cli" / script
    worker.parent.mkdir()
    worker.touch()
    venv_sync.arm_completion(root)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))
    monkeypatch.setattr(sys, "argv", [str(worker), "--source", str(root), "--finish-update", "--prepared"])

    assert venv_sync.prepare_launch(root, sys.argv[1:]) is None
    assert not completion_tail
    assert venv_sync.completion_pending_path(root).is_file()


def test_same_named_script_outside_the_checkout_still_repairs(tmp_path, monkeypatch, completion_tail):
    """Only this checkout's own tail is exempt: another tree's worker still owes this one's repair."""
    import pm
    from hermes_cli import _launchers

    root = _self_checkout(tmp_path, monkeypatch)
    foreign = tmp_path / "other" / "hermes_cli" / "source_completion.py"
    foreign.parent.mkdir(parents=True)
    foreign.touch()
    venv_sync.arm_completion(root)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))
    monkeypatch.setattr(sys, "argv", [str(foreign), "--prepared"])

    assert venv_sync.prepare_launch(root, sys.argv[1:]) is None
    assert len(completion_tail) == 1
    assert not venv_sync.completion_pending_path(root).is_file()


def test_completion_tail_output_stays_off_stdout(tmp_path, monkeypatch, completion_tail):
    """The automatic tail runs in front of the user's command, which may be piping JSON."""
    import pm
    from hermes_cli import _launchers

    root = _self_checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: None)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))
    venv_sync.prepare_launch(root, [])
    assert completion_tail.kwargs["stdout"] is sys.__stderr__


def test_interrupted_tail_is_terminated_and_the_marker_survives_for_the_next_launch(
        tmp_path, monkeypatch, completion_tail):
    """^C while the tail child runs must not leave that child prompting in the background.

    The interactive install the tail performs (a plugin dependency y/N) swallows SIGINT in
    its own input(), so the parent has to terminate it explicitly; the pending marker stays
    armed so the next launch finishes the tail instead of stranding a half-done migration.
    """
    import pm
    from hermes_cli import _launchers

    root = _self_checkout(tmp_path, monkeypatch)
    fact = runtime_facts_path(root)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: fact.is_file())
    fact.parent.mkdir(parents=True, exist_ok=True)
    fact.write_text(json.dumps({"packages": {"venv": {"stamp": "complete", "extras": ["all"]}}}))
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: None)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: Path(sys.executable))

    pending = venv_sync.completion_pending_path(root)
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text("source update tail not finished\n")

    completion_tail.interrupt_wait = True
    with pytest.raises(KeyboardInterrupt):
        venv_sync.prepare_launch(root, [])
    assert completion_tail.terminated, "the tail child was left alive after ^C"
    assert pending.is_file(), "an interrupted tail must stay armed for the next launch"

    completion_tail.exit_code = 0
    assert venv_sync.prepare_launch(root, []) is None
    assert not pending.exists(), "the retried tail did not clear the completion marker"


def test_first_launch_syncs_without_marker_then_uses_completion_fact(tmp_path, monkeypatch, completion_tail):
    import pm
    from hermes_cli import _launchers

    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n")
    (root / "uv.lock").write_text("lock\n")
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self"}))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    fact = runtime_facts_path(root)
    calls = []
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: fact.is_file())
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))

    def sync(extras=None, **kwargs):
        calls.append((extras, kwargs))
        fact.parent.mkdir(parents=True, exist_ok=True)
        fact.write_text(json.dumps({"packages": {"venv": {"stamp": "complete", "extras": ["all"]}}}))

    monkeypatch.setattr(pm, "sync_venv", sync)
    # A shipped updater may have written this before it reaches an inert shim.
    # It is obsolete after successful sync, not the trigger for that sync.
    (root / ".update-incomplete").write_text("pid=-1\n")
    assert venv_sync.prepare_launch(root, []) == Path(sys.executable)
    assert calls == [(["all"], {"explicit": True, "project_root": root, "evict_incompatible_plugins": True})]
    assert not (root / ".update-incomplete").exists()
    assert any("source_completion.py" in str(part) for cmd in completion_tail for part in cmd)
    assert venv_sync.prepare_launch(root, []) is None
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["script", "module", "command"])
def test_relaunch_keeps_invocation_and_checkout_imports(tmp_path, mode):
    root = tmp_path / "source"
    root.mkdir()
    (root / "checkout_only.py").write_text("value = 'from checkout'\n")
    script = root / "entry.py"
    script.write_text("import checkout_only, json, sys\nprint(json.dumps([checkout_only.value, sys.argv[1:]]))\n")
    argv = [str(script), "--profile", "name with spaces", "-c", "session"]
    orig = [sys.executable, *argv]
    module = None
    if mode == "module":
        module = "entry"
        orig = [sys.executable, "-m", module, *argv[1:]]
    elif mode == "command":
        argv[0] = "-c"
        orig = [sys.executable, "-c", "import entry", *argv[1:]]
    command = venv_sync.relaunch_command(Path(sys.executable), root, argv, orig, module)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["from checkout", argv[1:]]


@pytest.mark.parametrize("owner,argv", [(None, []), ("external", []), ("electron-updater", []), ("self", ["-p", "coder", "pm", "repair"])])
def test_non_self_or_pm_launch_cannot_trigger_update(tmp_path, monkeypatch, owner, argv):
    import pm
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\n")
    if owner:
        (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": owner}))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: pytest.fail("unowned launch reached PM"))
    assert venv_sync.prepare_launch(root, argv) is None


def test_failed_launch_keeps_previous_completion_and_retries(tmp_path, monkeypatch):
    import pm
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self"}))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    fact = runtime_facts_path(root)
    fact.parent.mkdir(parents=True)
    previous = '{"packages":{"venv":{"stamp":"previous","extras":["all","anthropic"]}}}'
    fact.write_text(previous)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)
    calls = []
    def fail(extras, **kwargs):
        calls.append(extras)
        raise RuntimeError("network unavailable")
    monkeypatch.setattr(pm, "sync_venv", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="network unavailable"):
            venv_sync.prepare_launch(root, [])
        assert fact.read_text() == previous
    assert calls == [None, None]
    assert not (root / ".update-incomplete").exists()


def test_blessed_legacy_install_is_adopted_before_sync(tmp_path, monkeypatch, completion_tail):
    import pm
    from hermes_cli import _launchers
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    root = home / "hermes-agent"
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\n")
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)
    calls = []
    monkeypatch.setattr(pm, "sync_venv", lambda *args, **kw: calls.append(args))
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))
    assert venv_sync.prepare_launch(root, []) == Path(sys.executable)
    assert json.loads((root / "install-stamp.json").read_text())["source"] == "adoption"
    assert calls == [(["all"],)]


def test_relaunch_runs_zip_launchers_and_preserves_interpreter_options(tmp_path):
    import zipfile
    launcher = tmp_path / "hermes.exe"
    with zipfile.ZipFile(launcher, "w") as archive:
        archive.writestr("__main__.py", "import json,sys; print(json.dumps([sys.argv[1:], sys.stdout.write_through, sys.flags.utf8_mode]))")
    original = [sys.executable, "-u", "-X", "utf8", str(launcher), "arg with spaces"]
    command = venv_sync.relaunch_command(Path(sys.executable), tmp_path, [str(launcher), "arg with spaces"], original, "__main__")
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [["arg with spaces"], True, 1]


def test_live_old_update_blocks_launch_sync(tmp_path, monkeypatch):
    import pm
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self"}))
    marker = root / ".update-incomplete"
    marker.write_text(f"pid={os.getpid()}\n")
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: pytest.fail("raced old updater"))
    with pytest.raises(RuntimeError, match="still running"):
        venv_sync.prepare_launch(root, [])
    assert marker.is_file()
    # Fresh post-sync verification children may boot under a live updater.
    from hermes_cli import _launchers
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))
    assert venv_sync.prepare_launch(root, []) is None
    assert marker.is_file()


def test_launch_under_the_owning_update_does_not_run_the_tail_again(tmp_path, monkeypatch, completion_tail):
    """The tail imports the application, whose entry point runs prepare_launch: inside the
    process tree of the update that owns the pending tail it must be a no-op, not recurse."""
    import time
    import pm
    from hermes_cli.update_lock import update_marker_path

    root = _self_checkout(tmp_path, monkeypatch)
    pending = venv_sync.completion_pending_path(root)
    pending.parent.mkdir(parents=True)
    pending.write_text("owed\n")
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    marker = update_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{os.getppid()}\n{int(time.time())}\n")  # an ancestor holds the update

    assert venv_sync.prepare_launch(root, []) is None
    assert completion_tail == []
    assert pending.is_file(), "the owning update's obligation was discharged by its own tail"


@pytest.mark.parametrize("marker,value,owed_by_cli", [
    ("HERMES_SUPERVISED_CHILD", "1", True),
    ("HERMES_S6_SUPERVISED_CHILD", "true", True),
    ("HERMES_SUPERVISED_CHILD", "0", False),
    ("HERMES_SUPERVISED_CHILD", "false", False),
])
def test_supervised_launch_leaves_a_pending_tail_to_the_cli(
    tmp_path, monkeypatch, capsys, completion_tail, marker, value, owed_by_cli
):
    """A supervised start leaves a pending tail to the CLI; its manager restarts it every boot (#123340).

    systemd/launchd/s6 restart the gateway under a policy; with a marker that never
    clears, each boot would rebuild the completion environment until the disk fills.
    The obligation stays with `hermes update`. An off value (`0`/`false`) is not a
    supervised launch and still repays the tail.
    """
    import pm
    from hermes_cli import _launchers

    root = _self_checkout(tmp_path, monkeypatch)
    pending = venv_sync.completion_pending_path(root)
    pending.parent.mkdir(parents=True)
    pending.write_text("owed\n")
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))
    monkeypatch.delenv("HERMES_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.setenv(marker, value)

    assert venv_sync.prepare_launch(root, ["gateway", "run"]) is None
    if owed_by_cli:
        assert completion_tail == []
        assert pending.is_file(), "a supervised child discharged an obligation the CLI still owes"
        out, err = capsys.readouterr()
        assert out == "" and err.count("run `hermes update`") == 1
    else:
        assert len(completion_tail) == 1
        assert not pending.is_file()


def test_supervised_launch_with_stale_dependencies_still_syncs(
    tmp_path, monkeypatch, completion_tail
):
    """Stale dependencies stay one-shot for a supervised child (#123340 review).

    A hand-run ``git pull`` leaves the tree's lockfile ahead of the installed
    tools with no pending marker; a manager restart must still sync and finish
    (pre-image behavior) instead of booting on the stale dependency graph — or
    crash-looping under ``Restart=always`` with no sync at all. The sticky-tail
    exemption above must not swallow this one-shot condition.
    """
    import pm
    from hermes_cli import _launchers

    root = _self_checkout(tmp_path, monkeypatch)
    syncs = []
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: False)
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: syncs.append(a))
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _, **__: Path(sys.executable))
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")

    venv_sync.prepare_launch(root, ["gateway", "run"])
    assert syncs, "a supervised child booted on a stale dependency graph without syncing"
    assert completion_tail, "the tail armed by that sync was never finished"
    assert not venv_sync.completion_pending_path(root).is_file()
