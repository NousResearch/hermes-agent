"""pm.features: the frozen bundle feature set (enabled-features.json).

Lazy installs OFF = the bundle's feature list is FROZEN to the file the
bundle wrote (the EXACT extras that installed on that target); pm sync
never deviates and never installs a plugin member.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pm.features as feats


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    """Point features_path at a temp runtime dir (store_root().parent)."""
    store = tmp_path / "tools"
    store.mkdir()
    monkeypatch.setattr(
        feats,
        "features_path",
        lambda base=None: (base if base is not None else store.parent)
        / feats.FEATURES_FILENAME,
    )
    return tmp_path


def test_write_then_read_roundtrip(rooted):
    path = feats.write_features(["web", "acp", "web"])
    assert path.is_file()
    got = feats.read_features()
    assert got == ["acp", "web"]  # sorted, deduped


def test_read_features_none_when_absent(rooted):
    assert feats.read_features() is None


def test_read_features_none_on_garbage(rooted):
    feats.features_path().write_text("{ not json", encoding="utf-8")
    assert feats.read_features() is None


def test_features_path_in_bundle_uses_payload_root(rooted):
    payload = rooted / "payload"
    payload.mkdir()
    assert feats.features_path(payload) == payload / "enabled-features.json"


def test_installed_extras_reports_only_anchor_resolved(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[project]\n"
        'name = "hermes-agent"\n'
        "[project.optional-dependencies]\n"
        'present = ["x"]\n'
        'absent = ["y"]\n',
        encoding="utf-8",
    )
    venv = tmp_path / "venv"
    site = venv / "Lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "somepkg.py").write_text("x = 1\n", encoding="utf-8")

    import pm.extras as extras_mod

    monkeypatch.setattr(
        extras_mod,
        "ANCHORS",
        {**extras_mod.ANCHORS, "present": "somepkg", "absent": "missingmod"},
    )
    got = feats.installed_extras(repo, venv)
    assert "present" in got
    assert "absent" not in got


def test_sync_venv_refuses_outside_frozen_extras(rooted, monkeypatch):
    feats.write_features(["web", "acp"])

    import sys

    ensure_mod = sys.modules["pm.ensure"]
    from pm.package import InstallError

    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: False)
    with pytest.raises(InstallError) as exc:
        ensure_mod.sync_venv(["slack"], explicit=True)
    assert "frozen" in str(exc.value) or "outside" in str(exc.value)


def test_sync_venv_allows_frozen_extras_when_lazy_off(rooted, monkeypatch):
    feats.write_features(["web"])

    import sys

    ensure_mod = sys.modules["pm.ensure"]

    # lazy off, request within the frozen set: passes the gate (may still
    # no-op on the stamp — we only assert no refusal here, by making the
    # stamp match so sync_venv returns early)
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: False)
    venv_pkg = ensure_mod.get_package("venv")
    monkeypatch.setattr(
        venv_pkg, "expected_stamp", lambda extras: "stamp"
    )
    monkeypatch.setattr(ensure_mod, "_facts", lambda: {"venv": {"stamp": "stamp", "extras": ["web"]}})
    ensure_mod.sync_venv(["web"])  # no raise
