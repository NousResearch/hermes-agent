"""Unreadable managed candidates remain repairable and never poison child PATH.

Regression contracts from #75419 (ChiChi), #97212 (sanjaynandanj), and
#83589 (nezuchills). Filesystem faults are injected at the failed OS boundary;
the resolver, managed-presence decision, and PATH composition run unchanged.
"""

import os
from pathlib import Path

import pytest

import hermes_constants as hc


@pytest.mark.parametrize("error", [PermissionError(13, "denied"), OSError(22, "unreachable")])
@pytest.mark.parametrize("healed", [False, True])
def test_unreadable_managed_candidate_reaches_heal(tmp_path, monkeypatch, error, healed):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    candidate = hc.iter_hermes_node_dirs()[0] / hc._candidate_node_command_names("npm")[0]
    candidate.parent.mkdir(parents=True)
    candidate.write_text("fixture", encoding="utf-8")
    candidate.chmod(0o755)
    real_is_file = Path.is_file
    repaired = False
    attempts = []

    def stat_file(path):
        if path == candidate and not repaired:
            raise error
        return real_is_file(path)

    def heal():
        nonlocal repaired
        attempts.append(True)
        repaired = healed
        return healed

    monkeypatch.setattr(Path, "is_file", stat_file)
    monkeypatch.setattr(hc, "heal_hermes_managed_node", heal)
    monkeypatch.setattr(hc, "_version_probe_ok", lambda path: repaired and path == str(candidate))
    monkeypatch.setattr(hc, "_managed_node_tree_outdated", lambda: False)
    monkeypatch.setattr(hc, "find_node_executable_on_path", lambda _: pytest.fail("broken managed tree fell back to PATH"))

    assert hc.hermes_managed_node_tree_present()
    assert hc.find_node_executable("npm") == (str(candidate) if healed else None)
    assert attempts == [True]


@pytest.mark.windows_only
@pytest.mark.parametrize("explicit", [False, True])
def test_windows_path_probe_skips_denied_candidate(tmp_path, monkeypatch, explicit):
    early, late = tmp_path / "early", tmp_path / "late"
    early.mkdir()
    late.mkdir()
    denied, good = early / "npm.cmd", late / "npm.cmd"
    good.write_text("@echo off\n", encoding="utf-8")
    real_is_file = Path.is_file

    def stat_file(path):
        if path == denied:
            raise PermissionError(13, "denied")
        return real_is_file(path)

    monkeypatch.setattr(Path, "is_file", stat_file)
    monkeypatch.setenv("PATH", os.pathsep.join(map(str, (early, late))))
    assert hc.find_node_executable_on_path(str(denied) if explicit else "npm") == (
        None if explicit else str(good))
    assert hc.node_tool_runnable(str(denied)) is False


@pytest.mark.parametrize("already_on_path", [False, True])
def test_unreadable_managed_directory_never_enters_child_path(tmp_path, monkeypatch, already_on_path):
    home = tmp_path / "home"
    denied, usable = home / "node", home / "node" / "bin"
    usable.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    real_scandir = os.scandir

    def scandir(path):
        if Path(path) == denied:
            raise PermissionError(13, "denied")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    original = {"PATH": str(denied) if already_on_path else str(tmp_path / "external")}
    merged = hc.with_hermes_node_path(original)
    assert str(denied) not in merged["PATH"].split(os.pathsep)
    assert str(usable) in merged["PATH"].split(os.pathsep)
    assert original["PATH"] == (str(denied) if already_on_path else str(tmp_path / "external"))

    from tools.environments.local import _managed_runtime_path_entries
    assert str(denied) not in _managed_runtime_path_entries()
    assert str(usable) in _managed_runtime_path_entries()
