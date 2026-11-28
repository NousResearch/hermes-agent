"""Tests: repo provenance collection across the four surfaces.

Fixture trees stand in for ``~/.hermes``; ``_config`` is stubbed so the tests
never read the developer's real config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import repos_provenance as prov


@pytest.fixture
def home(tmp_path):
    return tmp_path / "hermes-home"


@pytest.fixture
def no_config(monkeypatch):
    monkeypatch.setattr(prov, "_config", dict)


def _plugin(
    home: Path, name: str, *, sidecar: dict | None = None, origin: str | None = None
) -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    if origin is not None:
        git = plugin_dir / ".git"
        git.mkdir()
        (git / "config").write_text(
            f'[remote "origin"]\n\turl = {origin}\n', encoding="utf-8"
        )
    if sidecar is not None:
        plugins = home / "plugins"
        rows = (
            json.loads((plugins / ".install-metadata.json").read_text(encoding="utf-8"))
            if (plugins / ".install-metadata.json").exists()
            else {}
        )
        rows[name] = sidecar
        (plugins / ".install-metadata.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )


def _skill_manifest(home: Path, skills: dict) -> Path:
    path = home.parent / ".agents" / "skills" / ".manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"skills": skills}), encoding="utf-8")
    return path


def test_sidecar_row_wins_over_git_remote(home, no_config):
    _plugin(
        home,
        "plug",
        sidecar={
            "catalog": {"repo": "https://github.com/o/r", "sha": "a" * 40},
            "pinned": True,
            "revision": "a" * 40,
        },
        origin="https://github.com/other/repo",
    )
    rows = prov.collect_installs(home)
    assert len(rows) == 1
    assert rows[0].surface == prov.SURFACE_PLUGIN
    assert rows[0].repo == "https://github.com/o/r"
    assert rows[0].sha == "a" * 40
    assert rows[0].provenance == prov.PROV_SIDECAR


def test_git_only_plugin_is_self_cloned(home, no_config):
    _plugin(home, "selfcloned", origin="https://github.com/o/r")
    rows = prov.collect_installs(home)
    assert rows[0].provenance == prov.PROV_GIT_REMOTE
    assert rows[0].repo == "https://github.com/o/r"


def test_manual_drop_without_repo_is_not_a_repo_install(home, no_config):
    (home / "plugins" / "handmade").mkdir(parents=True)
    (home / "plugins" / "handmade" / "plugin.yaml").write_text(
        "name: x\n", encoding="utf-8"
    )
    assert prov.collect_installs(home) == []


def test_plugin_status_from_config(home, monkeypatch):
    _plugin(home, "on", sidecar={"revision": "a" * 40})
    _plugin(home, "off", sidecar={"revision": "b" * 40})
    monkeypatch.setattr(
        prov, "_config", lambda: {"plugins": {"enabled": ["on"], "disabled": ["off"]}}
    )
    rows = {row.name: row for row in prov.collect_installs(home)}
    assert rows["on"].status == prov.STATUS_ENABLED
    assert rows["off"].status == prov.STATUS_DISABLED


def test_skills_manifest_parses_owner_repo_ref(home, no_config):
    _skill_manifest(
        home,
        {
            "agent-browser": {
                "source": "vercel-labs/agent-browser@main:skills/agent-browser",
                "version": "1.2.3",
                "installedAt": "2026-01-01T00:00:00Z",
            }
        },
    )
    rows = prov.collect_installs(home)
    assert len(rows) == 1
    row = rows[0]
    assert row.surface == prov.SURFACE_SKILL
    assert row.repo == "https://github.com/vercel-labs/agent-browser"
    assert row.ref == "main"
    assert row.sha == "1.2.3"
    assert row.provenance == prov.PROV_MANIFEST
    assert row.manage["uninstall"] == "hermes skills uninstall agent-browser"


def test_mcp_endpoint_row(home, monkeypatch):
    monkeypatch.setattr(
        prov,
        "_config",
        lambda: {
            "mcp_servers": {
                "deepwiki": {"url": "https://mcp.deepwiki.com/mcp", "enabled": True}
            }
        },
    )
    rows = prov.collect_installs(home)
    assert len(rows) == 1
    assert rows[0].surface == prov.SURFACE_MCP
    assert rows[0].status == prov.STATUS_ENABLED
    assert rows[0].provenance == prov.PROV_ENDPOINT


def test_mcp_disabled_row(home, monkeypatch):
    monkeypatch.setattr(
        prov,
        "_config",
        lambda: {
            "mcp_servers": {
                "n8n": {"url": "http://localhost:5678/mcp", "enabled": False}
            }
        },
    )
    rows = prov.collect_installs(home)
    assert rows[0].status == prov.STATUS_DISABLED


def test_unclaimed_git_dir_is_a_clone(home, no_config):
    repo = home / "dev" / "some-tool"
    repo.mkdir(parents=True)
    git = repo / ".git"
    git.mkdir()
    (git / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/o/tool\n', encoding="utf-8"
    )
    rows = prov.collect_installs(home)
    assert [row.surface for row in rows] == [prov.SURFACE_CLONE]
    assert rows[0].repo == "https://github.com/o/tool"
    assert rows[0].status == prov.STATUS_PRESENT


def test_claimed_paths_are_not_double_counted(home, no_config):
    _plugin(home, "plug", origin="https://github.com/o/r")
    rows = prov.collect_installs(home)
    assert [row.surface for row in rows] == [prov.SURFACE_PLUGIN]


def test_results_are_sorted_by_id(home, no_config):
    _plugin(home, "bbb", origin="https://github.com/o/b")
    _plugin(home, "aaa", origin="https://github.com/o/a")
    ids = [row.id for row in prov.collect_installs(home)]
    assert ids == sorted(ids)


def test_to_row_omits_empty_manage():
    row = prov.RepoInstall(surface=prov.SURFACE_CLONE, name="x").to_row()
    assert "manage" not in row
    assert row["repo"] is None
