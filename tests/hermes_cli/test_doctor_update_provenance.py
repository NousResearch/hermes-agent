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
def _make_plugin(plugins: Path, name, *, sidecar=None, git=False, manifest_update_url=None,
                 install_json=None):
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
    if install_json is not None:
        (pdir / "install.json").write_text(json.dumps(install_json), encoding="utf-8")
    if manifest_update_url is not None:
        (pdir / "plugin.yaml").write_text(
            f"name: {name}\nupdate_url: {manifest_update_url}\n", encoding="utf-8"
        )
    return pdir


def test_no_plugins_dir_is_info(tmp_path):
    rows = ds._plugin_provenance_rows(tmp_path / "missing")
    assert rows == [("info", "No plugins directory yet (nothing to check provenance for)", "")]


def test_mixed_provenance_diagnostic_is_read_only(tmp_path, monkeypatch, capsys):
    plugins = tmp_path / 'plugins'
    url = 'https://example.com/x'
    cases = [
        ('drifty', {'source': 'git', 'update_url': url}, False, None, None, 'warn', 'reinstall'),
        ('dropped', None, False, None, None, 'info', 'manually'),
        ('cloned', None, True, None, None, 'info', 'adopt'),
        ('fine', {'source': 'git', 'update_url': url}, True, url, None, 'ok', 'good standing'),
        ('sneaky', {'source': 'git'}, True, 'https://evil.example/x', None, 'warn', 'no url was saved'),
        ('swapped', {'source': 'git', 'update_url': url}, True, 'https://evil.example/x', None, 'warn', 'update_url mismatch'),
        ('removed', {'source': 'git', 'update_url': url}, True, None, None, 'warn', 'update_url mismatch'),
        # UPD-5 (est-qrazy): archive install — row + install.json, no .git —
        # is class proven, never the drift warning with a git-clone remedy.
        ('packed', {'source': 'archive', 'revision': 'c' * 40}, False, None,
         {'version': '1.4.3', 'source_commit': 'c' * 40, 'packaged_at': '2026-10-01T00:00:00Z'},
         'ok', 'artifact-pinned'),
    ]
    for name, sidecar, git, manifest, marker, *_ in cases:
        _make_plugin(plugins, name, sidecar=sidecar, git=git, manifest_update_url=manifest,
                     install_json=marker)
    before = {p: p.read_bytes() for p in plugins.rglob('*') if p.is_file()}
    rows = ds._plugin_provenance_rows(plugins)
    for name, *_, severity, remedy in cases:
        assert any(kind == severity and name in text and remedy in text + detail
                   for kind, text, detail in rows)
    monkeypatch.setattr('hermes_constants.get_hermes_home', lambda: tmp_path)
    ds._check_update_provenance(False)
    output = capsys.readouterr().out
    assert all(name in output for name, *_ in cases)
    assert {p: p.read_bytes() for p in plugins.rglob('*') if p.is_file()} == before


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
