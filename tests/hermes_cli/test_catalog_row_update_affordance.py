"""hermes_cli.plugins_cmd_catalog: the plugin row's update affordance.

``update_available`` and the version label next to the update button have to agree. The label is the
pin's own release and is shown as the TARGET of the update, so a row offering "Update to v1.0.0"
for an install that already IS v1.0.0 reads as broken — and a row offering an update that would move
the install BACKWARDS to an older release is worse.

Three states, all distinguishable from the installed tree alone:
- the catalog pin carries a NEWER release: an update, labelled with that release;
- the catalog pin carries the SAME release at another commit: a sync, labelled by commit (no version);
- the catalog pin carries an OLDER release: no update at all.
"""

from __future__ import annotations

import pytest

from hermes_cli import plugins_cmd_catalog as cat
from hermes_cli import plugins_cmd as pc


CATALOG_SHA = "d73848735a2a48471a62d2c8fb96cd7183c97491"
OTHER_SHA = "3943c56979598ae5e64c3de9e4095e2bf3d0609f"


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A plugins dir with one installed catalog plugin, and a way to set its release."""
    home = tmp_path / "home"
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)

    def _install(version: str, *, sha: str, pin: str):
        target = plugins_dir / "mnemosyne"
        target.mkdir(exist_ok=True)
        (target / "plugin.yaml").write_text(
            f"name: mnemosyne\nversion: {version}\ndescription: d\nmaintainer: m\n", encoding="utf-8")
        record = {
            "catalog": {"name": "mnemosyne", "repo": "https://example/repo", "sha": sha,
                        "tier": "community", "pin": pin},
            "pinned": True, "revision": sha, "source": "https://example/repo"}
        pc._update_install_record("mnemosyne", lambda _current: record)
        return target

    return _install


def _row(target, catalog_version: str):
    return cat.catalog_row_fields(target, {"mnemosyne": CATALOG_SHA}, {"mnemosyne": catalog_version})


def test_a_newer_catalog_release_is_an_update(installed):
    """The ordinary case: the catalog carries a newer release, and the row names it."""
    target = installed("0.7.3", sha=OTHER_SHA, pin="")
    row = _row(target, "0.7.7")
    assert row["update_available"] is True
    assert row["catalog_version"] == "0.7.7"


def test_the_same_release_at_another_commit_is_a_sync_not_a_version_update(installed):
    """The install already carries the pin's version: the target is a commit, so no version label."""
    target = installed("1.0.0", sha=OTHER_SHA, pin="")
    row = _row(target, "1.0.0")
    assert row["update_available"] is True          # the commit moved, the sync is real
    assert row["catalog_version"] is None            # but it is not "an update to v1.0.0"


def test_an_older_catalog_release_is_not_offered(installed):
    """The catalog pin is OLDER than the install: offering an update would move it backwards."""
    target = installed("0.7.7", sha=OTHER_SHA, pin="")
    row = _row(target, "0.7.3")
    assert row["update_available"] is False


def test_the_label_does_not_decide_the_comparison(installed):
    """The comparison reads the catalog's release, not what the row ended up showing.

    A row whose label was dropped (same release) must still compare against that release, or an
    older pin would look like a plain SHA difference and get offered again.
    """
    target = installed("0.7.7", sha=OTHER_SHA, pin="")
    row = _row(target, "0.7.7")
    assert row["catalog_version"] is None
    assert row["update_available"] is True   # same release: a sync, still worth offering


def test_an_install_at_the_pin_is_current(installed):
    """Already at the reviewed pin: nothing to update to."""
    target = installed("0.7.3", sha=CATALOG_SHA, pin=CATALOG_SHA)
    row = _row(target, "0.7.3")
    assert row["update_available"] is False


def test_an_unknown_installed_version_keeps_the_sha_signal(installed):
    """A manifest with no version cannot be ordered against the catalog: the SHA check stands."""
    target = installed("0.0.0", sha=OTHER_SHA, pin="")
    (target / "plugin.yaml").write_text("name: mnemosyne\ndescription: d\n", encoding="utf-8")
    row = _row(target, "0.7.7")
    assert row["update_available"] is True
    assert row["catalog_version"] == "0.7.7"
