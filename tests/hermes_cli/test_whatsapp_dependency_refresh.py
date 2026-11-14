"""Updating the bridge must install its new dependency pin before pairing."""

import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("installed_version,installed_hash,mirror,refresh", [
    ("old", "new-hash", False, True),
    ("new", "new-hash", False, False),
    ("new", "new-hash", True, False),
    ("new", "different-hash", True, True),
])
def test_setup_refreshes_stale_dependencies_but_keeps_matching_install(
        tmp_path, monkeypatch, installed_version, installed_hash, mirror, refresh):
    from hermes_cli import main_platform_setup as setup
    import hermes_constants

    dependency = "node_modules/@whiskeysockets/baileys"
    package_dir = tmp_path / dependency
    package_dir.mkdir(parents=True)
    (package_dir / "package.json").write_text("{}", encoding="utf-8")
    desired = {
        "packages": {
            "": {"dependencies": {"@whiskeysockets/baileys": "new"}},
            dependency: {"version": "new", "resolved": "https://example.test/new.tgz", "integrity": "new-hash"},
        }
    }
    present = dict(desired["packages"][dependency], version=installed_version, integrity=installed_hash)
    if mirror:
        present["resolved"] = "https://mirror.example.test/new.tgz"
    actual = {"packages": {dependency: present}}
    (tmp_path / "package-lock.json").write_text(json.dumps(desired), encoding="utf-8")
    (tmp_path / "node_modules" / ".package-lock.json").write_text(json.dumps(actual), encoding="utf-8")
    calls = []

    def install(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == str(tmp_path)
        assert kwargs["timeout"] == 450
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("WHATSAPP_NPM_INSTALL_TIMEOUT", "450")
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda name: "isolated-npm")
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", dict)
    monkeypatch.setattr(setup.subprocess, "run", install)
    assert setup._whatsapp_install_bridge(tmp_path)
    assert bool(calls) is refresh
