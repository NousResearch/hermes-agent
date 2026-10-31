"""A catalog install dialog's ``plugins.manage install`` is held to the commit its ``inspect`` read.

The backend reloads the live catalog entry at Install, so a re-pin between the dialog's inspection and the
click would otherwise install (and, through the one checkout, publish the desktop half of) a tree the
dialog never showed. Reusing a disabled tree already on disk is held to the same commit. The catalog
install itself stays what it always was: pinned to the catalog's sha, annotated-tag pins included.
Real RPC dispatch, real clone of a local repo; only the https transport is redirected.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import hermes_yaml as yaml
from tests.hermes_cli.plugin_worker_support import (
    isolated_python as isolated_python,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
    plugin_world as plugin_world,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
)
from tui_gateway import server

URL = "https://example.invalid/acme/demo.git"


@pytest.fixture(autouse=True)
def _offline_pm(plugin_world):
    (plugin_world.home / "config.yaml").unlink()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([
        "git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args,
    ], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, marker: str) -> str:
    (repo / "marker.txt").write_text(marker, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", marker)
    return _git(repo, "rev-parse", "HEAD")


def _rpc(action: str, **params):
    return server.handle_request({"id": "1", "method": "plugins.manage",
                                  "params": {"action": action, "identifier": URL, "catalog_name": "demo", **params}})


@pytest.fixture
def world(tmp_path, monkeypatch):
    """``(home, repo, pin)``: a repo published through catalog entry ``demo``; ``pin(sha)`` re-pins it."""
    from hermes_cli import plugins_cmd, plugins_cmd_catalog
    from hermes_cli.plugin_catalog import PluginCatalogEntry
    repo = tmp_path / "demo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "plugin.yaml").write_text(yaml.safe_dump({"name": "demo", "version": "1.0.0"}), encoding="utf-8")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    real = plugins_cmd._clone_plugin_repo
    monkeypatch.setattr(plugins_cmd, "_clone_plugin_repo",
                        lambda dest, _url, revision, subdir=None: real(dest, repo.as_uri(), revision, subdir))
    current: dict[str, PluginCatalogEntry] = {}

    def pin(sha: str) -> None:
        current["demo"] = PluginCatalogEntry(name="demo", repo=URL, sha=sha, description="d", maintainer="m")
    monkeypatch.setattr(plugins_cmd_catalog, "get_live_catalog_entry", lambda name: current.get(name))
    from hermes_constants import get_hermes_home
    return get_hermes_home(), repo, pin


def _record(home: Path) -> dict:
    return json.loads((home / "plugins" / ".install-metadata.json").read_text())["demo"]


def test_a_catalog_entry_repinned_after_the_dialog_inspected_it_is_refused(world):
    home, repo, pin = world
    reviewed = _commit(repo, "reviewed")
    pin(reviewed)
    assert _rpc("inspect")["result"]["sha"] == reviewed
    pin(_commit(repo, "moved"))  # the catalog refreshes while the dialog is open

    refused = _rpc("install", inspected_sha=reviewed, enable=False)["error"]

    assert "source changed since you reviewed it" in refused["message"]
    assert refused["data"] == {"reason": "source_moved"}  # the dialog re-inspects and shows the new pin
    assert not (home / "plugins" / "demo").exists()


def test_a_disabled_tree_at_another_commit_is_not_enabled_for_the_reviewed_one(world):
    home, repo, pin = world
    old = _commit(repo, "old")
    pin(old)
    assert _rpc("install", enable=False)["result"]["ok"] is True  # an earlier install left it disabled
    reviewed = _commit(repo, "reviewed")
    pin(reviewed)

    result = _rpc("install", inspected_sha=reviewed, enable=True)

    assert "already exists" in result["error"]["message"]
    assert _record(home)["revision"] == old
    from hermes_cli.plugins_cmd import _get_enabled_set
    assert "demo" not in _get_enabled_set()


def test_a_disabled_tree_at_the_reviewed_commit_is_still_reused(world):
    home, repo, pin = world
    reviewed = _commit(repo, "reviewed")
    pin(reviewed)
    assert _rpc("install", enable=False)["result"]["ok"] is True

    result = _rpc("install", inspected_sha=reviewed, enable=False)["result"]

    assert result["ok"] is True and _record(home)["revision"] == reviewed


def test_an_annotated_tag_pin_installs_at_its_commit_and_stays_a_catalog_pin(world):
    home, repo, pin = world
    reviewed = _commit(repo, "reviewed")
    _git(repo, "tag", "-a", "v1.0", reviewed, "-m", "v1.0")
    _commit(repo, "later")
    tag_object = _git(repo, "rev-parse", "v1.0")
    assert tag_object != reviewed
    pin(tag_object)
    assert _rpc("inspect")["result"]["sha"] == reviewed  # inspect reports the commit the tag peels to

    result = _rpc("install", inspected_sha=reviewed, enable=False)["result"]

    assert result["ok"] is True
    record = _record(home)
    assert (record["revision"], record["pinned"]) == (reviewed, True)
    assert record["catalog"]["pin"] == tag_object  # still at the catalog's pin: `update` follows re-pins
    assert (home / "plugins" / "demo" / "marker.txt").read_text() == "reviewed"


def test_a_disabled_tree_at_an_annotated_tag_pin_is_reused_for_that_pin(world):
    """Native Desktop's dialog names the catalog's pin as published (the tag object), not the commit."""
    home, repo, pin = world
    reviewed = _commit(repo, "reviewed")
    _git(repo, "tag", "-a", "v1.0", reviewed, "-m", "v1.0")
    tag_object = _git(repo, "rev-parse", "v1.0")
    pin(tag_object)
    assert _rpc("install", enable=False)["result"]["ok"] is True

    for shown in (tag_object, reviewed):
        assert _rpc("install", inspected_sha=shown, enable=False)["result"]["ok"] is True
    assert _record(home)["revision"] == reviewed
