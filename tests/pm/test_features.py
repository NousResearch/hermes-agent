"""pm.features: the frozen bundle feature set (enabled-features.json).

Lazy installs OFF = the bundle's feature list is FROZEN to the file the
bundle wrote (the EXACT extras that installed on that target); pm sync
never deviates and never installs a plugin member.
"""

from __future__ import annotations

import pytest

import pm.features as feats


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    """Point features_path at a temp runtime dir (store_root().parent)."""
    store = tmp_path / "tools"
    store.mkdir()
    monkeypatch.setattr("pm.paths.store_root", lambda: store)
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
    from pm import paths
    from pm.lock import Facts
    from hermes_cli.runtime_paths import install_state_dir, runtime_facts_path

    ensure_mod = sys.modules["pm.ensure"]

    # Matching stamp alone cannot certify a vanished environment. Reuse only
    # the recorded selection while retaining the disabled acquisition policy.
    repo = rooted / "repo"
    repo.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    environment = install_state_dir(repo) / "environments" / "frozen" / "venv"
    environment.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = fixture\n")
    Facts(runtime_facts_path(repo)).record_state("venv", "stamp", ["web"], environment=environment)
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: False)
    venv_pkg = ensure_mod.get_package("venv")
    monkeypatch.setattr(
        venv_pkg, "expected_stamp", lambda extras: "stamp"
    )
    monkeypatch.setattr(venv_pkg, "apply", lambda *args, **kwargs: pytest.fail("current frozen environment rebuilt"))
    ensure_mod.sync_venv(["web"])
    (environment / "pyvenv.cfg").unlink()
    from pm.package import InstallError
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        ensure_mod.sync_venv(["web"])
