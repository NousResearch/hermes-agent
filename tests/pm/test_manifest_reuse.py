"""The pinned native-plugin admission shortcut must never bypass PM policy."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import pytest

from pm.manifest_reuse import _installed_versions_match, reusable_manifest_selection
from pm.publication import PluginSelection


def test_installed_distribution_metadata_without_running_pth(tmp_path):
    env = tmp_path / "venv"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    site = env / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    dist = site / "cryptography-50.0.2.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: cryptography\nVersion: 50.0.2\n")
    (site / "bad.pth").write_text("import sys; raise AssertionError('pth was executed')\n")
    assert _installed_versions_match(env, {"cryptography": "50.0.2"})
    assert not _installed_versions_match(env, {"cryptography": "49.0.0"})
    assert not _installed_versions_match(env, {"missing": "50.0.2"})


def _scenario(tmp_path, monkeypatch, *, requirements=("cryptography>=41,<51",)):
    import pm.environments
    import pm.install
    import pm.workspace

    state = tmp_path / "state"
    generation = state / "environments" / "generation-1"
    environment = generation / "venv"
    environment.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = /trusted/python\n")
    lock = generation / "workspace" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        '[[package]]\nname = "cryptography"\nversion = "50.0.2"\n'
        'source = {registry = "https://pypi.org/simple"}\n'
    )
    manifest = tmp_path / "candidate" / "plugin.yaml"
    manifest.parent.mkdir()
    manifest.write_text(
        "name: candidate\nkind: standalone\n" +
        "python_dependencies:\n" +
        "".join(f"  - {json.dumps(spec)}\n" for spec in requirements)
    )
    proposal = object.__new__(PluginSelection)
    proposal.members = [manifest.parent]
    package = SimpleNamespace(
        project_root=lambda: tmp_path,
        expected_stamp=lambda enabled, **kwargs: "baseline",
    )
    monkeypatch.setattr(pm.environments, "install_state_dir", lambda project: state)
    monkeypatch.setattr(pm.workspace, "build_member_dirs", lambda enabled: [])
    monkeypatch.setattr(pm.install, "_runtime_state_matches", lambda fact, stamp: fact.get("stamp") == stamp == "baseline")
    monkeypatch.setattr("pm.manifest_reuse._installed_versions_match",
                        lambda env, versions: versions == {"cryptography": "50.0.2"})
    fact = {"stamp": "baseline", "extras": [],
            "environment": str(environment), "resolved_lock": str(lock)}
    return package, proposal, fact


def test_native_candidate_reuses_exact_pinned_dependency(tmp_path, monkeypatch):
    package, proposal, fact = _scenario(tmp_path, monkeypatch)
    assert reusable_manifest_selection(package, proposal, fact, [], {"plugin_dirs": proposal.members})


@pytest.mark.parametrize("requirement", [
    "cryptography>=51", "cryptography[ssh]>=41", "cryptography>=41; python_version>='3'",
    "cryptography @ https://example.invalid/wheel.whl",
    "unlocked-package>=1",
])
def test_incompatible_or_unqualified_requirement_fails_closed(tmp_path, monkeypatch, requirement):
    package, proposal, fact = _scenario(tmp_path, monkeypatch, requirements=(requirement,))
    assert not reusable_manifest_selection(package, proposal, fact, [], {"plugin_dirs": proposal.members})


def test_source_requirements_cannot_disappear_from_the_admission_check(tmp_path, monkeypatch):
    package, proposal, fact = _scenario(
        tmp_path, monkeypatch, requirements=(
            "cryptography>=41,<51",
            "unreviewed @ https://example.invalid/unreviewed.whl",
        ))
    # read_python_declaration deliberately filters untrusted direct URLs,
    # but admission must never interpret the surviving subset as complete.
    assert not reusable_manifest_selection(
        package, proposal, fact, [], {"plugin_dirs": proposal.members})


def test_stale_state_and_multiple_members_cannot_reuse(tmp_path, monkeypatch):
    package, proposal, fact = _scenario(tmp_path, monkeypatch)
    bad = dict(fact, stamp="stale")
    assert not reusable_manifest_selection(package, proposal, bad, [], {"plugin_dirs": proposal.members})
    for key in ("environment", "resolved_lock"):
        bad = {**fact, key: None}
        assert not reusable_manifest_selection(package, proposal, bad, [], {"plugin_dirs": proposal.members})
    assert not reusable_manifest_selection(package, proposal, fact, ["new-extra"], {"plugin_dirs": proposal.members})
    other = tmp_path / "other"
    other.mkdir()
    assert not reusable_manifest_selection(package, proposal, fact, [],
                                           {"plugin_dirs": [*proposal.members, other]})


def test_no_build_publication_keeps_original_environment_and_lock(tmp_path, monkeypatch):
    import pm.install as install
    from pm import receipt
    from hermes_cli import runtime_state

    package, proposal, fact = _scenario(tmp_path, monkeypatch)
    proposal.publish = lambda project: None
    package.expected_stamp = lambda enabled, **kwargs: "new-stamp"
    package.apply = lambda *args, **kwargs: pytest.fail("package environment rebuild is forbidden")
    facts = SimpleNamespace(get=lambda name: fact, record_state=lambda *args, **kwargs: recorded.append((args, kwargs)))
    recorded = []
    monkeypatch.setattr("pm.manifest_reuse.reusable_manifest_selection",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(runtime_state, "finish_publication", lambda root: None)
    monkeypatch.setattr(receipt, "record_venv_rebuild", lambda *args: results.append(args))
    results = []
    install._commit_selection(package, facts, proposal, enabled=[], stamp="new-stamp",
                              inputs={"plugin_dirs": proposal.members}, current=False,
                              repair=False, explicit=True)
    assert recorded[0][0] == ("venv", "new-stamp", [])
    assert recorded[0][1]["environment"] == Path(fact["environment"])
    assert recorded[0][1]["resolved_lock"] == Path(fact["resolved_lock"])
    assert results == [(False, "pinned native requirements already installed")]
