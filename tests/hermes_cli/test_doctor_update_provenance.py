"""Doctor's plugin update-provenance check (SPEC-05 warning surface).

Bounded and READ-ONLY: the pure 2x2 plugin provenance reconciliation
(hermes_cli.plugins_provenance — the actual authority) plus the
manifest/sidecar update-url cross-check, surfaced as doctor rows. NO
network (no fetch, no ls-remote — the cadence owns online checks), no
mkdir — the plugins dir is computed, never created.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch

import hermes_cli.doctor_state as ds


# --- plugin provenance rows ---------------------------------------------
def _make_plugin(plugins: Path, name, *, sidecar=None, git=False, manifest_update_url=None):
    pdir = plugins / name
    pdir.mkdir(parents=True, exist_ok=True)
    if sidecar is not None:
        side = plugins / ".install-metadata.json"
        rows = {}
        if side.is_file():
            rows = json.loads(side.read_text(encoding="utf-8-sig"))
        rows[name] = sidecar
        side.write_text(json.dumps(rows), encoding="utf-8")
    if git:
        (pdir / ".git").mkdir()
        (pdir / ".git" / "config").write_text(
            '[remote "origin"]\n\turl = https://example.com/x.git\n', encoding="utf-8"
        )
    if manifest_update_url is not None:
        (pdir / "plugin.yaml").write_text(
            f"name: {name}\nupdate_url: {manifest_update_url}\n", encoding="utf-8"
        )
    return pdir


def test_no_plugins_dir_is_info(tmp_path):
    rows = ds._plugin_provenance_rows(tmp_path / "missing")
    assert rows == [("info", "No plugins directory yet (nothing to check provenance for)", "")]


def test_drift_plugin_warns_with_reinstall_remedy(tmp_path):
    _make_plugin(tmp_path, "drifty", sidecar={"source": "git", "update_url": "https://example.com/x"})
    rows = ds._plugin_provenance_rows(tmp_path)
    warns = [r for r in rows if r[0] == "warn"]
    assert len(warns) == 1
    assert "'drifty'" in warns[0][1] and "drift" in warns[0][1]
    assert "reinstall" in str(warns[0])


def test_manual_and_self_cloned_are_info(tmp_path):
    _make_plugin(tmp_path, "dropped")
    _make_plugin(tmp_path, "cloned", git=True)
    rows = ds._plugin_provenance_rows(tmp_path)
    assert any("manually" in text and kind == "info" for kind, text, _ in rows)
    assert any("'cloned'" in text and "adopt" in detail for kind, text, detail in rows if kind == "info")
    assert not any(kind == "warn" for kind, _, _ in rows)


def test_git_plugin_in_good_standing_is_ok(tmp_path):
    _make_plugin(
        tmp_path, "fine",
        sidecar={"source": "git", "update_url": "https://example.com/x"},
        git=True, manifest_update_url="https://example.com/x",
    )
    rows = ds._plugin_provenance_rows(tmp_path)
    assert not any(kind == "warn" for kind, _, _ in rows)
    assert any(kind == "ok" and "good standing" in text for kind, text, _ in rows)


def test_manifest_url_without_saved_url_warns(tmp_path):
    _make_plugin(tmp_path, "sneaky", sidecar={"source": "git"}, git=True,
                 manifest_update_url="https://evil.example/x")
    rows = ds._plugin_provenance_rows(tmp_path)
    warns = [r for r in rows if r[0] == "warn"]
    assert len(warns) == 1
    assert "'sneaky'" in warns[0][1] and "no url was saved" in warns[0][1]


@pytest.mark.parametrize("claimed", [None, "https://evil.example/x"])
def test_manifest_url_mismatch_warns(tmp_path, claimed):
    _make_plugin(tmp_path, "swapped", sidecar={"source": "git", "update_url": "https://example.com/x"},
                 git=True, manifest_update_url=claimed)
    rows = ds._plugin_provenance_rows(tmp_path)
    warns = [r for r in rows if r[0] == "warn"]
    assert len(warns) == 1
    assert "update_url mismatch" in str(warns[0])


# --- doctor-side wiring -------------------------------------------------


def test_check_is_read_only(tmp_path, monkeypatch, capsys):
    """The check must not create or modify anything under HERMES_HOME —
    no plugins/ mkdir; a missing dir is informational, not a warning."""
    import hermes_constants as config_mod

    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    ds._check_update_provenance(False)
    out = capsys.readouterr().out
    assert "No plugins directory yet" in out
    assert not (tmp_path / "plugins").exists()
    assert "⚠" not in out


def test_check_swallows_provenance_read_failure(tmp_path, monkeypatch, capsys):
    import hermes_constants as config_mod

    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    (tmp_path / "plugins").mkdir()
    with patch(
        "hermes_cli.plugins_provenance.plugins_provenance",
        side_effect=RuntimeError("disk gone"),
    ):
        ds._check_update_provenance(False)
    out = capsys.readouterr().out
    assert "could not be read" in out
    assert "disk gone" in out  # unreadable provenance must not look healthy
