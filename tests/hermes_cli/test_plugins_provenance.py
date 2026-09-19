"""Tests: plugin provenance reconciliation — the 2x2.

Row-presence says "hermes installed this"; `.git`-presence cross-checks
it. The reconciliation IS the disambiguation (settled 2026-09-03,
plugin-auto-update plan Task 1):
  row+git   = git install         (updatable via the recorded source)
  neither   = manual drop         (not auto-updatable, listed as such)
  git only  = self-cloned         (adoptable: origin URL ready)
  row only  = provenance drift    (flagged with a reinstall remedy)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.plugins_provenance import (
    Provenance,
    ProvenanceClass,
    plugins_provenance,
)


def _write_sidecar(home: Path, rows: dict) -> None:
    plugins = home / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    (plugins / ".install-metadata.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def _fake_git_dir(plugin_dir: Path, origin_url: str = "https://example/o/r") -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    git = plugin_dir / ".git"
    git.mkdir()
    # a real-enough git dir: remote config the adopt path reads
    (git / "config").write_text(
        f'[remote "origin"]\n\turl = {origin_url}\n', encoding="utf-8"
    )


def _manual_dir(plugin_dir: Path) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text("name: x\n", encoding="utf-8")


def test_git_install_row_and_git(tmp_path):
    _write_sidecar(tmp_path, {"plug": {"pinned": False, "revision": "a" * 40, "source": "https://example/o/r"}})
    _fake_git_dir(tmp_path / "plugins" / "plug")
    prov = plugins_provenance(tmp_path / "plugins")
    assert len(prov) == 1
    assert prov[0].klass is ProvenanceClass.GIT
    assert prov[0].row is not None and prov[0].row["revision"] == "a" * 40


def test_manual_drop_neither(tmp_path):
    _write_sidecar(tmp_path, {})
    _manual_dir(tmp_path / "plugins" / "handmade")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.MANUAL


def test_self_cloned_git_without_row(tmp_path):
    _write_sidecar(tmp_path, {})
    _fake_git_dir(tmp_path / "plugins" / "selfcloned", "https://example/m/r")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.SELF_CLONED
    assert prov[0].origin_url == "https://example/m/r"


def test_drift_row_without_git(tmp_path):
    _write_sidecar(tmp_path, {"ghost": {"pinned": False, "revision": "b" * 40, "source": "https://example/g/r"}})
    _manual_dir(tmp_path / "plugins" / "ghost")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.DRIFT
    assert prov[0].row["source"] == "https://example/g/r"


def test_missing_sidecar_is_all_manual_or_selfcloned(tmp_path):
    # no sidecar at all: everything classifies from the dir alone
    _fake_git_dir(tmp_path / "plugins" / "cloned")
    _manual_dir(tmp_path / "plugins" / "dropped")
    prov = plugins_provenance(tmp_path / "plugins")
    classes = {p.name: p.klass for p in prov}
    assert classes["cloned"] is ProvenanceClass.SELF_CLONED
    assert classes["dropped"] is ProvenanceClass.MANUAL


def test_update_url_tag_round_trips(tmp_path):
    # a saved update_url tag rides the row; the manifest is claims
    _write_sidecar(
        tmp_path,
        {"plug": {"pinned": False, "revision": "c" * 40, "source": "https://example/o/r",
                  "update_url": "https://example/feed.yml"}},
    )
    _fake_git_dir(tmp_path / "plugins" / "plug")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].saved_update_url == "https://example/feed.yml"