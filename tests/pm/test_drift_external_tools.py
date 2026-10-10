"""Drift in a sealed lane (#134249): the artifact's own packaging may provide a
tool pm's store can never satisfy (Nix: a read-only ``/nix/store`` and a loader
that refuses the generic ELFs a PM download stages there). A binary that
resolves on ambient PATH outside pm's stores counts as satisfied; git
checkouts, unstamped trees, absent binaries and pm-owned stores keep the
strict verdict."""
import sys
from pathlib import Path

import pytest

import pm.install as install
from pm import paths
from pm.registry import get_package
from pm.store import current_target

_BINARY = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"


def _fake_bin(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / name
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return binary


@pytest.fixture()
def sealed_tree(tmp_path, monkeypatch):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "install-stamp.json").write_text('{"updateMechanism": "self"}')
    store = tmp_path / "store"
    store.mkdir()
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(paths, "install_stamp_path", lambda _project_root: root / "install-stamp.json")
    monkeypatch.setattr(paths, "store_root", lambda: store)
    monkeypatch.setattr(paths, "writable_store_root", lambda: shadow)
    return root


def test_steward_provided_binary_on_path_is_satisfied(sealed_tree, monkeypatch):
    fake = _fake_bin(sealed_tree.parent / "bin", _BINARY)
    monkeypatch.setenv("PATH", str(fake.parent))
    assert install._externally_provided(get_package("ffmpeg"), current_target()) is True


def test_git_checkout_is_never_satisfied_by_path(sealed_tree, monkeypatch):
    (sealed_tree / ".git").mkdir()
    fake = _fake_bin(sealed_tree.parent / "bin", _BINARY)
    monkeypatch.setenv("PATH", str(fake.parent))
    assert install._externally_provided(get_package("ffmpeg"), current_target()) is False


def test_unstamped_tree_is_not_satisfied(sealed_tree, monkeypatch):
    (sealed_tree / "install-stamp.json").unlink()
    fake = _fake_bin(sealed_tree.parent / "bin", _BINARY)
    monkeypatch.setenv("PATH", str(fake.parent))
    assert install._externally_provided(get_package("ffmpeg"), current_target()) is False


def test_pm_store_binary_still_reports_drift(sealed_tree, monkeypatch):
    fake = _fake_bin(paths.store_root(), _BINARY)
    monkeypatch.setenv("PATH", str(fake.parent))
    assert install._externally_provided(get_package("ffmpeg"), current_target()) is False


def test_absent_binary_is_not_provided(sealed_tree, monkeypatch):
    monkeypatch.setenv("PATH", str(sealed_tree.parent / "nothing-here"))
    assert install._externally_provided(get_package("ffmpeg"), current_target()) is False


def test_drift_skips_steward_provided_tools_but_reports_the_rest(sealed_tree, monkeypatch):
    facts = sealed_tree.parent / "facts.json"
    facts.write_text("{}")
    monkeypatch.setattr(paths, "facts_path", lambda: facts)
    monkeypatch.setattr(paths, "runtime_facts_path", lambda: sealed_tree.parent / "absent.json")
    monkeypatch.setattr(install, "_installed_location", lambda *args, **kwargs: None)
    monkeypatch.setattr(install, "_facts", lambda: {})
    monkeypatch.setattr(install, "_store", lambda: None)
    fake = _fake_bin(sealed_tree.parent / "bin", "node.exe" if sys.platform == "win32" else "node")
    monkeypatch.setenv("PATH", str(fake.parent))

    problems = install.drift()

    assert "node" not in problems
    assert problems  # tools nothing provides still report honestly
