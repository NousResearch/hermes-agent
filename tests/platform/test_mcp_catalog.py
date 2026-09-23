"""NOVA's own MCP catalogue entries: what they may and may not do.

They sit beside the upstream catalogue in the image and are compiled into agents exactly
like upstream entries, so the rules that keep the upstream catalogue safe apply here too —
pinned versions, no secret in the manifest, no quiet replacement of an approved entry.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
NOVA_CATALOGUE = REPO / "nova" / "mcp-catalog"
UPSTREAM_CATALOGUE = REPO / "optional-mcps"
ENTRIES = sorted(p for p in NOVA_CATALOGUE.iterdir() if (p / "manifest.yaml").is_file())


def _manifest(entry: Path) -> dict:
    return yaml.safe_load((entry / "manifest.yaml").read_text(encoding="utf-8"))


def test_there_are_entries_to_check():
    assert ENTRIES, "nova/mcp-catalog/ holds no manifests"


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_an_entry_never_shadows_an_upstream_one(entry):
    """The image build fails on a collision too; this says so before anyone builds."""
    assert not (UPSTREAM_CATALOGUE / entry.name).exists()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_every_package_is_pinned_to_an_exact_version(entry):
    """A floating package is code that changes under a deployed agent. The Google Sheets
    server broke on an unpinned dependency the day it was added — pins are not pedantry."""
    transport = _manifest(entry)["transport"]
    if transport["type"] != "stdio":
        return
    packages = [a for a in transport.get("args", []) if not a.startswith("-")]
    assert packages
    for package in packages:
        assert re.search(r"(@|==)\d+\.\d+\.\d+$", package), f"{package} is not pinned"


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_secrets_are_referenced_never_written(entry):
    manifest = _manifest(entry)
    declared = {e["name"] for e in manifest.get("auth", {}).get("env", [])}
    for key, value in (manifest["transport"].get("env") or {}).items():
        references = set(re.findall(r"\$\{([^}]+)\}", str(value)))
        assert references, f"{key} holds a literal value; secrets belong in the agent's .env"
        assert references <= declared, f"{key} references a variable the entry does not declare"


def test_the_runtime_parses_the_composed_catalogue(tmp_path, monkeypatch):
    """Built the way the image builds it, read by the runtime's own parser, compiled by NOVA."""
    composed = tmp_path / "catalogue"
    shutil.copytree(UPSTREAM_CATALOGUE, composed)
    for entry in ENTRIES:
        shutil.copytree(entry, composed / entry.name)
    monkeypatch.setenv("HERMES_OPTIONAL_MCPS", str(composed))

    from hermes_cli.mcp_catalog import catalog_diagnostics, list_catalog
    from nova.runtime.hermes.extensions import credential_env_for, server_config

    names = {e.name for e in list_catalog()}
    assert {e.name for e in ENTRIES} <= names
    assert not catalog_diagnostics()
    for entry in ENTRIES:
        config, reason = server_config(entry.name)
        assert not reason and config
        slots, problem = credential_env_for(entry.name)
        assert not problem
        declared = {e["name"] for e in _manifest(entry).get("auth", {}).get("env", [])}
        assert {s["name"] for s in slots} == declared
