"""Doctor and launched commands share PM's read-only installed selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

import pm
from hermes_cli import doctor_tools
from pm import paths
from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import current_target, tree_digest


@pytest.fixture
def tool_store(tmp_path, monkeypatch):
    rg = shutil.which("rg")
    assert rg is not None, "this executable parity test requires ripgrep"
    version_output = subprocess.run(
        [rg, "--version"], capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    version = version_output.splitlines()[0].split()[1]
    home = tmp_path / "home"
    primary = tmp_path / "payload" / "tools"
    primary.mkdir(parents=True)
    (primary.parent / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(primary))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    target = current_target()
    package = get_package("ripgrep")
    digest = hashlib.sha256(Path(rg).read_bytes()).hexdigest()
    lock = Lockfile(lock_path)
    lock.set_pin("ripgrep", version, {target: {"url": Path(rg).as_uri(), "sha256": digest}})
    lock.save()

    def publish(root):
        entry = root / package.store_entry(version, target)
        binary = package.binary(entry, target)
        assert binary is not None
        binary.parent.mkdir(parents=True)
        binary.symlink_to(rg)
        Facts(root / "facts.json").record(
            "ripgrep", version, entry.name, package.env(entry, target), root,
            target=target, artifacts=[digest], digest=tree_digest(entry),
        )
        return binary

    hostile = tmp_path / "hostile"
    hostile.mkdir()
    hostile_rg = hostile / "rg"
    hostile_rg.write_text("#!/bin/sh\nprintf 'hostile PATH ripgrep\\n'\n", encoding="utf-8")
    hostile_rg.chmod(0o755)
    monkeypatch.setenv("PATH", str(hostile))
    return primary, publish, version_output, hostile_rg


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("primary_state", ["absent", "stale"])
def test_sealed_store_extension_matches_launched_tool(tool_store, primary_state, capsys):
    primary, publish, version_output, _hostile = tool_store
    if primary_state == "stale":
        publish(primary)
        facts_path = primary / "facts.json"
        data = json.loads(facts_path.read_text(encoding="utf-8"))
        data["packages"]["ripgrep"]["version"] = "0.0.0"
        facts_path.write_text(json.dumps(data), encoding="utf-8")
    binary = publish(paths.writable_store_root())
    before = {p: p.read_bytes() for root in (primary, paths.writable_store_root())
              for p in root.rglob("*") if p.is_file()}
    # The selected writable extension must not require repairing the payload.
    sealed_paths = [primary, *(p for p in primary.rglob("*") if not p.is_symlink())]
    modes = {p: stat.S_IMODE(p.stat().st_mode) for p in sealed_paths}
    for path, mode in modes.items():
        path.chmod(mode & ~0o222)
    try:
        selected = pm.installed_package("ripgrep")
        assert selected is not None and selected.binary == binary
        env = pm.env_for("ripgrep", base_env={"PATH": str(_hostile.parent)})
        assert shutil.which("rg", path=env["PATH"]) == str(binary)
        child = subprocess.run(["rg", "--version"], env=env, capture_output=True,
                               text=True, check=True, timeout=10)
        assert child.stdout == version_output
        assert doctor_tools._doctor_tool("rg") == (str(binary), "(pm store)")
        doctor_tools._check_git_and_rg(False)
        assert "✓ ripgrep (rg) (pm store)" in capsys.readouterr().out
        assert {p: p.read_bytes() for root in (primary, paths.writable_store_root())
                for p in root.rglob("*") if p.is_file()} == before
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("damage", ["version", "target", "artifacts", "missing-binary", "sparse", "malformed"])
def test_rejected_store_is_not_reported_as_runtime_available(tool_store, damage, capsys):
    primary, publish, _version_output, hostile = tool_store
    binary = publish(primary)
    facts_path = primary / "facts.json"
    data = json.loads(facts_path.read_text(encoding="utf-8"))
    fact = data["packages"]["ripgrep"]
    if damage == "missing-binary":
        binary.unlink()
    elif damage == "sparse":
        data["packages"]["ripgrep"] = {"entry": fact["entry"]}
    elif damage != "malformed":
        fact[damage] = ["wrong-artifact"] if damage == "artifacts" else "outdated"
    facts_path.write_text("not JSON" if damage == "malformed" else json.dumps(data), encoding="utf-8")
    before = facts_path.read_bytes()

    assert pm.installed_package("ripgrep") is None
    assert doctor_tools._pm_tool_path("ripgrep") is None
    assert doctor_tools._doctor_tool("rg") == (str(hostile), "")
    child = subprocess.run(["rg", "--version"], env=pm.env_for("ripgrep"),
                           capture_output=True, text=True, check=True, timeout=10)
    assert child.stdout.strip() == "hostile PATH ripgrep"
    doctor_tools._check_git_and_rg(False)
    out = capsys.readouterr().out
    assert "✓ ripgrep (rg)" in out and "(pm store)" not in out
    assert facts_path.read_bytes() == before