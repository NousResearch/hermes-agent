"""Payload stores follow the payload, not the launching shell's home."""

import json
import os

import pytest

from pm.environments import site_packages, store_root, venv_python_version
from hermes_constants import get_default_hermes_root


def test_store_resolution_follows_relocated_payload(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    payload = tmp_path / "agent-payload"
    repo = payload / "hermes-agent"
    repo.mkdir(parents=True)
    stamp = repo / "install-stamp.json"
    stamp.write_text(json.dumps({"payload": "bundled", "runtime": {
        "repoDir": "hermes-agent", "toolsDir": "tools",
    }}))
    manifest = payload / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "repo": "hermes-agent",
                                    "venv": "venv", "store": "tools",
                                    "runtime": {"toolsDir": "tools"}}))
    tools = payload / "tools"
    tools.mkdir()
    facts = {"schema": 1, "packages": {"node": {"entry": "node-test"}}}
    (tools / "facts.json").write_text(json.dumps(facts))
    (tools / "node-test").mkdir()

    for destination in (payload, tmp_path / "relocated payload"):
        if destination != payload:
            payload.rename(destination)
        repo = destination / "hermes-agent"
        resolved = store_root(repo)
        assert resolved == destination / "tools"
        installed = json.loads((resolved / "facts.json").read_text())
        assert (resolved / installed["packages"]["node"]["entry"]).is_dir()

    override = tmp_path / "stage-tools"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(override))
    assert store_root(repo) == override
    monkeypatch.delenv("HERMES_RUNTIME_DIR")
    (repo / "install-stamp.json").write_text(json.dumps({"runtimeDir": str(override)}))
    (destination / "manifest.json").write_text(json.dumps({"repo": "other-repo"}))
    assert store_root(repo) == override
    (repo / "install-stamp.json").unlink()
    assert store_root(repo) == get_default_hermes_root() / "tools"


@pytest.mark.parametrize("escape", ["relative", "absolute", "symlink"])
def test_payload_store_cannot_escape_payload(tmp_path, monkeypatch, escape):
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    payload = tmp_path / "agent-payload"
    repo = payload / "hermes-agent"
    repo.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    values = {"relative": "../outside", "absolute": str(outside), "symlink": "tools"}
    if escape == "symlink":
        (payload / "tools").symlink_to(outside, target_is_directory=True)
    (payload / "manifest.json").write_text(json.dumps({
        "repo": "hermes-agent", "store": values[escape],
    }))
    with pytest.raises(RuntimeError, match="payload store escapes its root"):
        store_root(repo)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(outside))
    assert store_root(repo) == outside

def test_site_packages_follows_the_venv_python_not_the_caller(tmp_path):
    """Regression: the tree must be dated from the venv, not from this process.

    An app-driven upgrade rebuilt the dependency environment with CPython 3.14
    while the PATH shim ran 3.11, so site_packages() pointed at
    lib/python3.11/site-packages inside a 3.14 venv. activate_dependencies()
    then found no tree and failed the shim *after* a successful update -- it
    only tolerates a missing tree when no generation was published.
    """
    venv = tmp_path / "venv"
    (venv / "lib" / "python3.9" / "site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.9.20\n", encoding="utf-8")

    assert venv_python_version(venv) == (3, 9)
    if os.name != "nt":
        # The contract: the composed path is the venv's real tree. Dated from the
        # caller instead, it names python<this-interpreter> and misses this dir.
        assert site_packages(venv) == venv / "lib" / "python3.9" / "site-packages"
        assert site_packages(venv).is_dir()


def test_venv_python_version_falls_back_to_the_lib_directory(tmp_path):
    venv = tmp_path / "venv"
    (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)

    assert venv_python_version(venv) == (3, 12)


def test_venv_python_version_is_none_without_evidence(tmp_path):
    assert venv_python_version(tmp_path / "absent") is None
