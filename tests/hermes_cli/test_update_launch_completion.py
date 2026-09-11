"""A fresh launch finishes a source update using PM's success record, not a marker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import venv_sync
from hermes_cli.runtime_paths import runtime_facts_path


def test_first_launch_syncs_without_marker_then_uses_completion_fact(tmp_path, monkeypatch):
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
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: Path(sys.executable))

    def sync(extras=None, **kwargs):
        calls.append((extras, kwargs))
        fact.parent.mkdir(parents=True)
        fact.write_text(json.dumps({"packages": {"venv": {"stamp": "complete", "extras": ["all"]}}}))

    monkeypatch.setattr(pm, "sync_venv", sync)
    # A shipped updater may have written this before it reaches an inert shim.
    # It is obsolete after successful sync, not the trigger for that sync.
    (root / ".update-incomplete").write_text("pid=-1\n")
    assert venv_sync.prepare_launch(root, []) == Path(sys.executable)
    assert calls == [(["all"], {"explicit": True, "project_root": root})]
    assert not (root / ".update-incomplete").exists()
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
    previous = '{"packages":{"venv":{"stamp":"previous","extras":["all","dev"]}}}'
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


def test_blessed_legacy_install_is_adopted_before_sync(tmp_path, monkeypatch):
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
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: Path(sys.executable))
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
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: Path(sys.executable))
    assert venv_sync.prepare_launch(root, []) is None
    assert marker.is_file()
