"""Collective Wisdom plugin: wire-contract conformance and consent invariants.

The Gateway recomputes every hash we send; ``wisdom_hash_vectors.json`` is its published
vector set, so a drift here is a live 409/422 on every share.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from plugins.wisdom import package as pkg

VECTORS = json.loads((Path(__file__).parent / "wisdom_hash_vectors.json").read_text(encoding="utf-8"))


def test_content_and_description_hashes_match_gateway_vectors(tmp_path):
    files = [(f["path"], f["mode"], base64.b64decode(f["content_base64"])) for f in VECTORS["files"]]
    assert pkg.verify_files(files) == VECTORS["content_hash"]
    for f in VECTORS["files"]:
        assert pkg.sha256_address(base64.b64decode(f["content_base64"])) == f["hash"]
    manifest_raw = next(b for p, _, b in files if p == "skill.manifest.json")
    assert pkg.sha256_address(manifest_raw) == VECTORS["package_manifest_hash"]
    # Re-serialising the parsed manifest must reproduce the server's canonical bytes exactly.
    assert pkg.manifest_bytes(pkg.parse_manifest(manifest_raw)) == manifest_raw
    desc = pkg.sanitize_description(VECTORS["author_description_input"])
    assert desc == VECTORS["canonical_author_description"]
    assert pkg.sha256_address(desc.encode("utf-8")) == VECTORS["author_description_hash"]
    for case in VECTORS["content_hash_cases"]:
        pairs = [(f["path"], f["hash"]) for f in case["files"]]
        assert pkg.content_hash(pairs) == case["content_hash"], case["name"]


@pytest.mark.parametrize("bad", [
    [("SKILL.md", "file", b"# x\n"), ("scripts/run.sh", "file", b"echo hi\n")],       # active content dir
    [("SKILL.md", "file", b"# x\nsee scripts/run.sh\n")],                                # reference to active content
    [("SKILL.md", "file", b"#!/bin/sh\n")],                                              # shebang
    [("SKILL.md", "file", b"# x\n"), ("refs/a.md", "file", b"a"), ("refs/A.md", "file", b"b")],  # case collision
    [("SKILL.md", "exec", b"# x\n")],                                                    # exec mode
    [("SKILL.md", "file", b"# x\n"), ("refs/../SKILL.md", "file", b"# y\n")],            # traversal
])
def test_instruction_only_contract_refuses_active_or_unsafe_content(bad):
    with pytest.raises(pkg.PackageError):
        pkg.verify_files(bad, require_manifest=False)


class _FakeClient:
    """Enough Gateway to drive install(): the fixture package is the published version."""
    org_id = "org-test"

    def __init__(self):
        self.files = [(f["path"], f["mode"], base64.b64decode(f["content_base64"])) for f in VECTORS["files"]]
        self.recorded = []
        self.identities = []

    def register_identity(self, ident):
        self.identities.append(ident)

    def skill(self, skill_id):
        return {"skill": {"id": skill_id, "slug": "canonical", "state": "active", "takedown_generation": 0},
                "versions": [{"version": 1}]}

    def version(self, skill_id, version):
        return {"version": {"version": version, "content_hash": VECTORS["content_hash"],
                            "security_check": {"status": "pass", "summary": "ok"}}}

    def content(self, skill_id, version, *, installation_id, takedown_generation):
        return VECTORS["content_hash"], self.files

    def record_install(self, **kw):
        self.recorded.append(kw)
        return {"installed_version": kw["version"], "effective_update_mode": "MANUAL"}


class _State(dict):
    def get(self, k, default=None):
        return super().get(k, default)

    def set(self, k, v):
        self[k] = v


def test_install_writes_only_after_native_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.wisdom.service import NotConfirmed, Wisdom
    client, state = _FakeClient(), _State()
    svc = Wisdom(state, client=client)
    seen = []

    with pytest.raises(NotConfirmed):
        svc.install("sk1", version=None, confirm=lambda t, d: seen.append((t, d)) or False)
    assert "v1" in seen[0][0] and VECTORS["content_hash"] in seen[0][1]
    assert client.recorded == [] and not (tmp_path / "skills").exists()

    result = svc.install("sk1", version=None, confirm=lambda t, d: True)
    dest = Path(result["path"])
    assert dest == tmp_path / "skills" / "_wisdom" / "org-test" / "canonical"
    assert (dest / "SKILL.md").read_bytes() == base64.b64decode(VECTORS["files"][0]["content_base64"])
    assert client.recorded[0]["installation_id"] == state["installation_id"] == client.identities[0]
    assert state["installed"]["sk1"]["version"] == 1


def test_bundled_plugin_loads_and_gates_tools_on_entitlement(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import plugins as pmod
    from tools import registry as reg
    mgr = pmod.PluginManager()
    mgr.discover_and_load()
    loaded = mgr._plugins["wisdom"]
    assert loaded.enabled, loaded.error
    entry = reg.registry.get_entry("wisdom_install")
    assert entry is not None and entry.toolset == "wisdom"
    assert "wisdom" in pmod.get_plugin_commands()
    assert entry.check_fn() is False  # no Nous token in a temp HERMES_HOME
    monkeypatch.setattr("plugins.wisdom.client.entitlement", lambda: {"org_id": "o", "scopes": ("wisdom:read",)})
    assert entry.check_fn() is True
