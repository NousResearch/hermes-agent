"""Tests: plugin provenance reconciliation — the 2x2 plus the artifact cell.

Row-presence says "hermes installed this"; `.git`-presence cross-checks
it; a shipped install.json (the release-archive marker) proves the
artifact half. The reconciliation IS the disambiguation (settled
2026-09-03, plugin-auto-update plan; artifact cell: UPD-5, est-qrazy):
  row+git              = git install         (updatable via the recorded source)
  neither              = manual drop         (not auto-updatable, listed as such)
  git only             = self-cloned         (adoptable: origin URL ready)
  row+install.json     = artifact-pinned     (class proven; catalog is the feed)
  row only, no marker  = provenance drift    (flagged with a reinstall remedy)
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


def _write_install_json(plugin_dir: Path, data) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "install.json").write_text(
        json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8"
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


# ── the artifact cell (UPD-5: archive installs are proven, not drift) ──


def test_artifact_row_plus_install_json_no_git_is_proven(tmp_path):
    """Sidecar row + a release-archive marker, no .git by design: the
    class is artifact-pinned (proven), not drift — and the remedy text
    for drift ('reinstall from the recorded source' → a git tree) must
    not fire for this cell."""
    _write_sidecar(tmp_path, {"packed": {"pinned": False, "revision": "9a6f8ac",
                                         "source": "https://github.com/o/r"}})
    _manual_dir(tmp_path / "plugins" / "packed")
    _write_install_json(
        tmp_path / "plugins" / "packed",
        {"version": "1.4.3", "source_commit": "9a6f8ac", "packaged_at": "2026-10-01T00:00:00Z"},
    )
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.ARTIFACT
    marker = prov[0].install_json
    assert marker is not None
    assert marker["version"] == "1.4.3"
    assert marker["source_commit"] == "9a6f8ac"


def test_artifact_backward_compat_marker_without_source_repo(tmp_path):
    """Archives packed before the one-schema PR carry no source_repo —
    the marker's commit alone still proves the class."""
    _write_sidecar(tmp_path, {"packed": {"revision": "a" * 40, "source": "https://example/o/r"}})
    _write_install_json(tmp_path / "plugins" / "packed", {"version": "1.0.0", "source_commit": "a" * 40})
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.ARTIFACT


def test_artifact_source_repo_must_agree_with_the_row(tmp_path):
    """A marker naming a different repo than the row paid for is drift,
    not proof (host/path spelling is normalized — .git/branch suffixes
    are spelling too)."""
    _write_sidecar(tmp_path, {"packed": {"revision": "a" * 40, "source": "https://github.com/o/r"}})
    _write_install_json(
        tmp_path / "plugins" / "packed",
        {"version": "1.0.0", "source_commit": "a" * 40, "source_repo": "https://github.com/evil/clone.git"},
    )
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.DRIFT


def test_artifact_source_repo_agreement_is_spelling_normalized(tmp_path):
    _write_sidecar(tmp_path, {"packed": {"revision": "a" * 40, "source": "https://github.com/o/r#sub/dir"}})
    _write_install_json(
        tmp_path / "plugins" / "packed",
        {"version": "1.0.0", "source_commit": "a" * 40, "source_repo": "git@github.com:O/R.git"},
    )
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.ARTIFACT


def test_unreadable_install_json_is_not_proof(tmp_path):
    _write_sidecar(tmp_path, {"packed": {"revision": "a" * 40, "source": "https://example/o/r"}})
    _write_install_json(tmp_path / "plugins" / "packed", "{not json")
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.DRIFT


def test_install_json_without_row_is_manual(tmp_path):
    """The marker proves the archive, only the sidecar row proves
    hermes installed it — a dropped bundle is still a manual drop."""
    _write_sidecar(tmp_path, {})
    _write_install_json(tmp_path / "plugins" / "dropped", {"version": "1.0.0", "source_commit": "d" * 40})
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.MANUAL


def test_git_wins_over_the_marker(tmp_path):
    """row + .git + install.json = a git install (a pulled tree may keep
    the marker); never reclassified as artifact."""
    _write_sidecar(tmp_path, {"plug": {"pinned": False, "revision": "a" * 40, "source": "https://example/o/r"}})
    _fake_git_dir(tmp_path / "plugins" / "plug")
    _write_install_json(tmp_path / "plugins" / "plug", {"version": "1.0.0", "source_commit": "a" * 40})
    prov = plugins_provenance(tmp_path / "plugins")
    assert prov[0].klass is ProvenanceClass.GIT
    assert prov[0].install_json is None


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

def test_origin_url_uses_the_resolved_git_and_survives_no_git_at_all(tmp_path, monkeypatch):
    """Provenance runs from the gateway service, where PATH can be minimal: the probe must use the
    same resolved git as install/update, and with no git anywhere the .git/config parse still
    yields the origin (never a bare ``git`` spawn that raises or picks up a stray binary)."""
    import subprocess
    from hermes_cli import plugins_cmd, plugins_provenance

    plugin = tmp_path / "plugins" / "p"
    plugin.mkdir(parents=True)
    _fake_git_dir(plugin, "https://example/from-config")
    spawned = []
    real_run = subprocess.run

    def record(argv, *a, **k):
        spawned.append(argv[0])
        return real_run(argv, *a, **k)

    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr(plugins_cmd, "_resolve_git_executable", lambda: None)
    assert plugins_provenance._git_origin_url(plugin) == "https://example/from-config"
    assert spawned == []

    monkeypatch.setattr(plugins_cmd, "_resolve_git_executable", lambda: "/nonexistent/resolved-git")
    assert plugins_provenance._git_origin_url(plugin) == "https://example/from-config"
    assert spawned == ["/nonexistent/resolved-git"]
