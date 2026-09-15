"""Payload stores follow the payload, not the launching shell's home."""

import json

import pytest

from hermes_cli.runtime_paths import store_root
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