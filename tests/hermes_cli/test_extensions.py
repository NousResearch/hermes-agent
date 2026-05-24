from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli.extensions import (
    ExtensionManifestError,
    extensions_command,
    load_extension_manifest,
    load_registry_index,
    render_inspect_receipt,
    search_registry,
)


VALID_MANIFEST = """
manifest_version: 1
kind: extension-pack
id: personal-ops/google-workspace-pack
name: Google Workspace Ops Pack
version: 0.1.0
description: Manage Gmail, Calendar, Drive, Docs, and Sheets from Hermes.
publisher:
  name: Example Publisher
  verified: false
compatibility:
  hermes: ">=0.9.0"
  platforms: [macos, linux]
contents:
  skills:
    - path: skills/productivity/google-workspace/SKILL.md
  plugins:
    - path: plugins/google-workspace/plugin.yaml
  mcp_servers:
    - name: google-workspace
      command: npx
  cron_recipes:
    - path: recipes/daily-agenda.yaml
permissions:
  env:
    - name: GOOGLE_CLIENT_ID
      required: true
      secret: false
    - name: GOOGLE_CLIENT_SECRET
      required: true
      secret: true
  network:
    hosts:
      - accounts.google.com
      - gmail.googleapis.com
  filesystem:
    reads: []
    writes:
      - "$HERMES_HOME/plugins/google-workspace/**"
  toolsets:
    - terminal
    - web
  shell:
    allowed_commands:
      - gws
      - python
    arbitrary_shell: false
  outbound_messages: false
risk:
  declared_level: medium
  rationale: OAuth integration with access to Google Workspace data.
""".strip()


REGISTRY_INDEX = """
{
  "schema_version": 1,
  "generated_at": "2026-05-24T00:00:00Z",
  "extensions": [
    {
      "id": "personal-ops/google-workspace-pack",
      "name": "Google Workspace Ops Pack",
      "kind": "extension-pack",
      "version": "0.1.0",
      "description": "Manage Gmail, Calendar, Drive, Docs, and Sheets from Hermes.",
      "publisher": "Example Publisher",
      "trust_level": "community",
      "risk_level": "medium",
      "tags": ["google", "workspace", "gmail"]
    }
  ]
}
""".strip()


def _write_manifest(tmp_path: Path, text: str = VALID_MANIFEST) -> Path:
    path = tmp_path / "extension.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _write_index(tmp_path: Path, text: str = REGISTRY_INDEX) -> Path:
    path = tmp_path / "index.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_extension_manifest_validates_and_normalizes(tmp_path: Path) -> None:
    manifest = load_extension_manifest(_write_manifest(tmp_path))

    assert manifest["id"] == "personal-ops/google-workspace-pack"
    assert manifest["kind"] == "extension-pack"
    assert manifest["permissions"]["env"][1]["name"] == "GOOGLE_CLIENT_SECRET"


def test_load_extension_manifest_rejects_missing_required_field(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, VALID_MANIFEST.replace("id: personal-ops/google-workspace-pack\n", ""))

    with pytest.raises(ExtensionManifestError, match="id"):
        load_extension_manifest(path)


def test_load_extension_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        VALID_MANIFEST.replace(
            "skills/productivity/google-workspace/SKILL.md",
            "../secrets/SKILL.md",
        ),
    )

    with pytest.raises(ExtensionManifestError, match="path traversal"):
        load_extension_manifest(path)


def test_registry_search_matches_tag_name_and_publisher(tmp_path: Path) -> None:
    index = load_registry_index(_write_index(tmp_path))

    assert [entry["id"] for entry in search_registry(index, "workspace")] == [
        "personal-ops/google-workspace-pack"
    ]
    assert [entry["id"] for entry in search_registry(index, "gmail")] == [
        "personal-ops/google-workspace-pack"
    ]
    assert [entry["id"] for entry in search_registry(index, "example")] == [
        "personal-ops/google-workspace-pack"
    ]


def test_render_inspect_receipt_includes_permission_categories(tmp_path: Path) -> None:
    manifest = load_extension_manifest(_write_manifest(tmp_path))
    output = render_inspect_receipt(manifest, source="fixture")

    assert "Permission Preview" in output
    assert "Env Vars" in output
    assert "Network Hosts" in output
    assert "Filesystem Writes" in output
    assert "Toolsets" in output
    assert "Shell Commands" in output
    assert "MCP Servers" in output
    assert "Cron Recipes" in output
    assert "Outbound Messages" in output
    assert "Risk: medium" in output
    assert "inspect-only" in output


def test_extensions_search_command_prints_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = extensions_command(
        Namespace(extensions_action="search", query="workspace", index=_write_index(tmp_path))
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "personal-ops/google-workspace-pack" in output
    assert "medium" in output
    assert "community" in output


def test_extensions_search_command_no_match_returns_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = extensions_command(
        Namespace(extensions_action="search", query="kubernetes", index=_write_index(tmp_path))
    )

    assert result == 1
    assert "No matching extensions" in capsys.readouterr().out


def test_extensions_inspect_command_prints_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = extensions_command(
        Namespace(extensions_action="inspect", manifest=_write_manifest(tmp_path))
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Google Workspace Ops Pack" in output
    assert "GOOGLE_CLIENT_SECRET" in output
    assert "Phase 1 is inspect-only" in output


def test_extensions_namespace_is_wired_into_main_cli(tmp_path: Path) -> None:
    index_path = _write_index(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "extensions",
            "search",
            "workspace",
            "--index",
            str(index_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "personal-ops/google-workspace-pack" in result.stdout
