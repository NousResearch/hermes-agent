"""hermes_cli.plugins_updates: ``check-updates`` must not offer a downgrade.

``check-updates`` is the read-only surface a user reads to decide whether to update. It compares
the installed commit against the catalog pin, and a SHA difference alone is not an update: the
catalog's release can be OLDER than the installed one (an install sitting on a commit the catalog
does not point at — an explicit ``--ref``, a local pin), and re-pinning then moves the install
BACKWARDS.

The version labels decide when both are known. A same-version pin advance is still an update (the
commit moved), and an entry with no comparable version keeps the SHA comparison.
"""

from __future__ import annotations

import pytest

from hermes_cli import plugins_updates as pu


def test_the_relation_helper_states_the_ordering():
    """The comparison itself: newer, same, older, incomparable."""
    assert pu._version_relation("0.7.3", "0.7.7") == 1
    assert pu._version_relation("0.7.7", "0.7.3") == -1
    assert pu._version_relation("0.7.7", "0.7.7") == 0
    assert pu._version_relation("", "0.7.7") is None
    assert pu._version_relation("0.7.7", "") is None
    assert pu._version_relation("nightly", "0.7.7") is None


def test_the_installed_version_comes_from_the_plugin_manifest(tmp_path):
    """The plugin's own manifest states its release; the install record does not."""
    plugin = tmp_path / "mnemosyne"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        "name: mnemosyne\nversion: 0.7.7\ndescription: d\nmaintainer: m\n", encoding="utf-8")
    assert pu._installed_manifest_version(plugin) == "0.7.7"


def test_an_unreadable_manifest_yields_no_version(tmp_path):
    """A missing manifest is not an error: the comparison simply cannot order the two."""
    assert pu._installed_manifest_version(tmp_path / "absent") == ""
