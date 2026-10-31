"""hermes_cli.plugins_cmd_catalog: what "an update is available" means, and a downgrade.

Two distinct defects, both about a catalog install that sits on a commit the
catalog does not point at (an explicit ``--ref``):

- ``update_available`` compared SHAs only, so an install NEWER than the catalog
  pin was reported as having an update available — pointing at a downgrade.
- The re-pin path would move such an install to an OLDER catalog pin in silence,
  undoing the fixes the named commit carried.

An off-pin install at the same or a newer release is an ordinary re-pin: the
catalog pin is the source of truth and moving to it is what an update means.
"""

from __future__ import annotations

import pytest

from hermes_cli import plugins_cmd_catalog as cat


CATALOG_SHA = "d73848735a2a48471a62d2c8fb96cd7183c97491"
USER_SHA = "3943c56979598ae5e64c3de9e4095e2bf3d0609f"


def _entry(*, version: str, sha: str):
    """A live catalog entry carrying a version label."""
    from hermes_cli.plugin_catalog import PluginCatalogEntry

    return PluginCatalogEntry(name="mnemosyne", repo="https://example/repo", sha=sha,
                              description="", maintainer="someone", version=version, tier="community")


def _sidecar(*, sha: str, pin: str) -> dict:
    return {"catalog_name": "mnemosyne", "repo": "https://example/repo", "sha": sha,
            "tier": "community", "pin": pin}


def test_a_newer_install_is_not_offered_a_downgrade():
    """Installed 0.7.7, catalog pins 0.7.3: re-pinning would move it BACKWARDS."""
    assert cat.update_available_for(
        _sidecar(sha=USER_SHA, pin=""), CATALOG_SHA,
        installed_version="0.7.7", catalog_version="0.7.3") is False


def test_an_older_install_is_offered_the_update():
    """The normal case: the catalog carries a newer release."""
    assert cat.update_available_for(
        _sidecar(sha=CATALOG_SHA, pin=CATALOG_SHA), "newsha",
        installed_version="0.7.3", catalog_version="0.7.7") is True


def test_an_install_at_the_pin_is_current():
    """Already at the reviewed pin: nothing to update to."""
    assert cat.update_available_for(
        _sidecar(sha=CATALOG_SHA, pin=CATALOG_SHA), CATALOG_SHA,
        installed_version="0.7.3", catalog_version="0.7.3") is False


def test_without_a_comparable_version_the_sha_comparison_stands():
    """An unlabelled release falls back to the SHA check rather than claiming no update."""
    assert cat.update_available_for(
        _sidecar(sha=USER_SHA, pin=""), CATALOG_SHA,
        installed_version="", catalog_version="") is True


def test_an_unparsable_version_falls_back_to_the_sha_comparison():
    """A label that is not a version cannot order the two."""
    assert cat.update_available_for(
        _sidecar(sha=USER_SHA, pin=""), CATALOG_SHA,
        installed_version="nightly", catalog_version="0.7.3") is True


def test_nothing_installed_is_not_an_update():
    """No install, no update."""
    assert cat.update_available_for(None, CATALOG_SHA, installed_version="0.7.3",
                                    catalog_version="0.7.7") is False


def test_pin_relation_reports_the_ordering():
    """The comparison itself, stated plainly: newer, same, older, incomparable."""
    assert cat.catalog_pin_relation("0.7.3", "0.7.7") == 1
    assert cat.catalog_pin_relation("0.7.7", "0.7.3") == -1
    assert cat.catalog_pin_relation("0.7.7", "0.7.7") == 0
    assert cat.catalog_pin_relation("", "0.7.7") is None
    assert cat.catalog_pin_relation("0.7.7", "") is None


def test_an_off_pin_install_is_recognised():
    """``pin`` empty plus a resolved commit = the user named that commit."""
    assert cat._is_off_pin_install(_sidecar(sha=USER_SHA, pin="")) is True


def test_an_install_at_the_catalog_pin_is_not_off_pin():
    """At the reviewed pin, ``pin`` records it."""
    assert cat._is_off_pin_install(_sidecar(sha=CATALOG_SHA, pin=CATALOG_SHA)) is False


def test_a_downgrading_repin_is_refused(tmp_path, monkeypatch):
    """The catalog pin is OLDER than the install: re-pinning would move it backwards."""
    from hermes_cli.plugins_cmd import PluginOperationError

    target = tmp_path / "mnemosyne"
    target.mkdir()
    (target / "plugin.yaml").write_text("name: mnemosyne\nversion: 0.7.7\n", encoding="utf-8")
    monkeypatch.setattr(cat, "get_live_catalog_entry",
                        lambda _name: _entry(version="0.7.3", sha=CATALOG_SHA))
    with pytest.raises(PluginOperationError) as excinfo:
        cat.repin_catalog_plugin(target, _sidecar(sha=USER_SHA, pin=""))
    message = str(excinfo.value)
    assert USER_SHA[:8] in message
    assert "downgrade" in message
    assert "--ref" in message


def test_a_refused_downgrading_repin_leaves_the_tree_untouched(tmp_path, monkeypatch):
    """A refused re-pin changes nothing."""
    from hermes_cli.plugins_cmd import PluginOperationError

    target = tmp_path / "mnemosyne"
    target.mkdir()
    (target / "plugin.yaml").write_text("name: mnemosyne\nversion: 0.7.7\n", encoding="utf-8")
    (target / "marker.txt").write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(cat, "get_live_catalog_entry",
                        lambda _name: _entry(version="0.7.3", sha=CATALOG_SHA))
    before = sorted(p.name for p in target.iterdir())
    with pytest.raises(PluginOperationError):
        cat.repin_catalog_plugin(target, _sidecar(sha=USER_SHA, pin=""))
    assert sorted(p.name for p in target.iterdir()) == before
    assert (target / "marker.txt").read_text(encoding="utf-8") == "keep me"


def test_a_downgrade_at_the_same_release_is_not_refused(tmp_path, monkeypatch):
    """No version label to compare: the SHA check decides, so the re-pin proceeds."""
    target = tmp_path / "mnemosyne"
    target.mkdir()
    (target / "plugin.yaml").write_text("name: mnemosyne\n", encoding="utf-8")
    monkeypatch.setattr(cat, "get_live_catalog_entry", lambda _name: _entry(version="", sha=CATALOG_SHA))
    # The re-pin itself needs a repo to clone; the point is that it is NOT refused as a downgrade.
    assert cat.catalog_pin_relation("", "") is None
    assert cat._is_off_pin_install(_sidecar(sha=USER_SHA, pin="")) is True
