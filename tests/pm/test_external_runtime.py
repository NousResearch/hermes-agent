"""Externally packaged tools do not owe PM receipts, but must remain executable."""

import json
import os
from pathlib import Path

import pytest

import pm
from pm import paths
from pm.lock import Facts, Lockfile
from pm.store import current_target


@pytest.fixture
def external_runtime(tmp_path, monkeypatch):
    # Nix keeps the wheel and the install stamp in separate derivations.
    code = tmp_path / "wheel"
    code.mkdir()
    root = tmp_path / "agent" / "share" / "hermes-agent"
    root.mkdir(parents=True)
    monkeypatch.setattr(paths, "repo_root", lambda: code)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    tools = {}
    for name, command in (("node", "node"), ("npm", "npm"),
                          ("ripgrep", "rg"), ("ffmpeg", "ffmpeg")):
        binary = tmp_path / "nix-store" / name / "bin" / command
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        binary.chmod(0o755)
        tools[name] = str(binary)
    stamp = root / "install-stamp.json"
    stamp.write_text(json.dumps({"distribution": "nix", "updateMechanism": "external",
                                 "externalPackages": tools}), encoding="utf-8")
    monkeypatch.setenv("PATH", os.pathsep.join(str(Path(tool).parent) for tool in tools.values()))
    return stamp, tools


@pytest.mark.parametrize("state", ["fresh", "tool-store", "runtime-selection"])
def test_external_tools_need_no_pm_receipts(external_runtime, state):
    if state != "fresh":
        facts = paths.facts_path() if state == "tool-store" else paths.runtime_facts_path()
        Facts(facts).record_state("venv", "selection", [])
        if state == "tool-store":
            old_node = paths.store_root() / "old-node" / "bin" / "node"
            old_node.parent.mkdir(parents=True)
            old_node.write_text("old PM tool", encoding="utf-8")
            old_node.chmod(0o755)
            lock = Lockfile(paths.lockfile_path())
            target = current_target()
            Facts(facts).record("node", lock.version("node"), "old-node",
                                {"PATH": [str(old_node.parent)]}, paths.store_root(),
                                target=target,
                                artifacts=[a["sha256"] for a in lock.artifacts("node", target)])
        before = facts.read_bytes()
    original_path = os.environ["PATH"]

    assert pm.activate(allow_incomplete=True) == []
    assert os.environ["PATH"] == original_path
    if state != "fresh":
        assert facts.read_bytes() == before
    else:
        assert not paths.facts_path().exists()
        assert not paths.runtime_facts_path().exists()


@pytest.mark.parametrize("state", ["fresh", "runtime-selection"])
@pytest.mark.parametrize("damage", ["missing", "directory", "relative",
                                    pytest.param("not-executable", marks=pytest.mark.platforms("posix"))])
def test_external_tool_damage_is_reported_without_installing(external_runtime, tmp_path, monkeypatch, state, damage):
    stamp, tools = external_runtime
    if state != "fresh":
        Facts(paths.runtime_facts_path()).record_state("venv", "selection", [])
    Path(tools["node"]).unlink()
    if damage == "directory":
        Path(tools["node"]).mkdir()
    elif damage == "not-executable":
        Path(tools["node"]).write_text("unusable tool", encoding="utf-8")
        Path(tools["node"]).chmod(0o644)
    elif damage == "relative":
        tools["node"] = "bin/node"
        data = json.loads(stamp.read_text())
        data["externalPackages"] = tools
        stamp.write_text(json.dumps(data), encoding="utf-8")
    replacement = tmp_path / "shell-bin" / "node"
    replacement.parent.mkdir()
    replacement.write_text("shell replacement", encoding="utf-8")
    replacement.chmod(0o755)
    monkeypatch.setenv("PATH", str(replacement.parent) + os.pathsep + os.environ["PATH"])

    assert pm.drift(include_venv=False) == {
        "node": "external executable missing or not executable",
    }
    assert not paths.facts_path().exists()


@pytest.mark.parametrize("mechanism", ["self", "external", "checkout"])
def test_external_declarations_do_not_hide_pm_owned_drift(external_runtime, mechanism):
    stamp, tools = external_runtime
    # A tool omitted from the declaration is still PM's responsibility;
    # a self-managed install cannot use the declaration to bypass PM at all.
    del tools["ripgrep"]
    if mechanism == "checkout":
        (paths.repo_root() / ".git").mkdir()
    stamp.write_text(json.dumps({"updateMechanism": "external" if mechanism == "checkout" else mechanism,
                                 "externalPackages": tools}),
                     encoding="utf-8")
    Facts(paths.facts_path()).record_state("venv", "selection", [])

    drift = pm.drift(include_venv=False)
    expected = {"ripgrep"} if mechanism == "external" else {"node", "npm", "ripgrep", "ffmpeg"}
    assert set(drift) == expected
    assert set(drift.values()) == {"not installed or outdated"}


def test_external_tools_do_not_hide_python_selection_drift(external_runtime, monkeypatch):
    import importlib

    Facts(paths.runtime_facts_path()).record_state("venv", "stale-selection", [])
    monkeypatch.setattr(importlib.import_module("pm.install"), "venv_is_current", lambda: False)

    assert pm.drift() == {"venv": "out of sync with uv.lock"}


@pytest.mark.parametrize("with_facts", [False, True])
def test_doctor_reports_external_tools_and_detects_damage(external_runtime, tmp_path, capsys, with_facts):
    from pm.cli import cmd_doctor

    stamp, tools = external_runtime
    # A declared tool must be checked even when PM normally delegates it to
    # the host (Git on POSIX), before missing_reason can classify it as n/a.
    git = tmp_path / "external-git"
    git.write_text("external tool", encoding="utf-8")
    git.chmod(0o755)
    tools["git"] = str(git)
    data = json.loads(stamp.read_text())
    data["externalPackages"] = tools
    stamp.write_text(json.dumps(data), encoding="utf-8")
    if with_facts:
        Facts(paths.facts_path()).record_state("venv", "selection", [])
        before = paths.facts_path().read_bytes()

    assert cmd_doctor(None) == 0
    output = capsys.readouterr().out
    for name, executable in tools.items():
        assert f"✓ {name}: externally managed ({executable})" in output

    Path(tools["node"]).unlink()
    assert cmd_doctor(None) == 1
    assert "✗ node: external executable missing or not executable" in capsys.readouterr().out
    if with_facts:
        assert paths.facts_path().read_bytes() == before
    else:
        assert not paths.facts_path().exists()


@pytest.mark.parametrize("mechanism", ["external", "self"])
def test_doctor_does_not_hide_pm_owned_dependencies(external_runtime, capsys, mechanism):
    from pm.cli import cmd_doctor

    stamp, tools = external_runtime
    del tools["ripgrep"]
    stamp.write_text(json.dumps({"updateMechanism": mechanism, "externalPackages": tools}),
                     encoding="utf-8")

    assert cmd_doctor(None) == 1
    output = capsys.readouterr().out
    expected = {"ripgrep"} if mechanism == "external" else {"node", "npm", "ripgrep", "ffmpeg"}
    for name in expected:
        assert f"✗ {name}: not installed" in output
    if mechanism == "external":
        assert "✗ node:" not in output
