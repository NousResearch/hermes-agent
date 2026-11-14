"""The catalog entry's ``requires_hermes`` is enforced, not just displayed: an install by catalog name
refuses on a Hermes the spec rejects (CLI, dashboard/Desktop, TUI RPC and the left-core auto-install all
go through one seam), and ``update`` keeps the installed pin rather than moving to one that needs a newer
Hermes. The running version comes from the same source the plugin.yaml gate reads. Real git, file://
repos, a seeded catalog, temp HERMES_HOME."""

from __future__ import annotations

import json
import shutil
import subprocess as sp

import pytest

from hermes_cli import left_core_migration as lcm
from hermes_cli import plugin_catalog as pc_cat
from hermes_cli import plugins_cmd as pc
from hermes_cli import plugins_manifest
from tests.hermes_cli.test_plugins_cmd_catalog import _commit, _head, world  # world: the seeded catalog + plugins dir
from tests.pm._fixtures import client, isolated_python  # fixtures world needs

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

REFUSAL = "cat-plugin needs Hermes >=0.31; this is 0.30.0."


@pytest.fixture
def gated(request, monkeypatch):
    """*world* whose catalog entry carries ``state["requires"]`` as its requires_hermes, on Hermes 0.30.0.
    Console output is captured in ``printed``."""
    base = request.getfixturevalue("world")
    state = base["state"]
    state["requires"] = ""

    def _entries():
        return [pc_cat.PluginCatalogEntry(name="cat-plugin", repo=state.get("repo_uri") or base["repo"].as_uri(),
                                          sha=state["pin"],
                                          description="d", maintainer="t", requires_hermes=state["requires"])]

    monkeypatch.setattr(pc_cat, "load_catalog", lambda catalog_dir=None: _entries())
    state["running"] = "0.30.0"
    monkeypatch.setattr(plugins_manifest, "running_hermes_version", lambda: state["running"])
    printed: list[str] = []
    console = type("C", (), {"print": lambda _self, *a, **_k: printed.append(" ".join(map(str, a)))})()
    monkeypatch.setattr(pc, "_console", lambda: console)
    return {**base, "printed": printed}


def _rpc(params: dict) -> dict:
    from tui_gateway import server
    return server.handle_request({"id": "1", "method": "plugins.manage", "params": params})


def test_install_by_catalog_name_refused_on_every_surface_when_hermes_is_too_old(gated):
    gated["state"]["requires"] = ">=0.31"

    with pytest.raises(SystemExit):  # `hermes plugins install <name>` (and `hermes setup`'s importer step)
        pc.cmd_install("cat-plugin", enable=False)
    assert any(REFUSAL in line and "and install again" in line for line in gated["printed"])

    dashboard = pc.dashboard_install_plugin("", force=False, enable=True, catalog_name="cat-plugin")
    assert dashboard["ok"] is False and REFUSAL in dashboard["error"]

    rpc = _rpc({"action": "install", "catalog_name": "cat-plugin"})  # Desktop / TUI
    assert REFUSAL in rpc["error"]["message"]

    said: list[str] = []
    feature = lcm.LeftCoreFeature(plugin="cat-plugin", label="Cat", in_use=lambda *_a, **_k: True, unchanged="x")
    assert lcm._install_one(gated["plugins_dir"].parent, feature,
                            install=lcm._install_into(gated["plugins_dir"].parent), say=said.append) is False
    assert REFUSAL.rstrip(".") in said[0]

    assert not (gated["plugins_dir"] / "cat-plugin").exists()


@pytest.mark.parametrize(("running", "spec"), [("0.30.0", ">=0.30"), ("unknown", ">=99.0")])
def test_install_proceeds_when_hermes_satisfies_the_spec_or_its_version_is_unknown(gated, running, spec):
    # "unknown" is a tagless checkout: the plugin.yaml gate lets it through, so the catalog gate must too.
    gated["state"].update(running=running, requires=spec)
    assert plugins_manifest.requires_hermes_error({"requires_hermes": spec}) is None

    result = pc.dashboard_install_plugin("", force=False, enable=False, catalog_name="cat-plugin")

    assert result["ok"] is True, result
    assert _head(gated["plugins_dir"] / "cat-plugin") == gated["sha1"]


def test_update_keeps_the_installed_pin_when_the_new_pin_needs_a_newer_hermes(gated):
    from hermes_cli.plugins_cmd_update_all import plan_update_all
    from hermes_cli.plugins_updates import run_checks

    assert pc.dashboard_install_plugin("", force=False, enable=False, catalog_name="cat-plugin")["ok"] is True
    target = gated["plugins_dir"] / "cat-plugin"
    gated["state"].update(pin=gated["sha2"], requires=">=0.31")

    with pytest.raises(SystemExit):  # `hermes plugins update cat-plugin`
        pc.cmd_update("cat-plugin", interactive=False)
    assert any(REFUSAL in line and "after you update Hermes" in line for line in gated["printed"])
    dashboard = pc.dashboard_update_user_plugin("cat-plugin")
    assert dashboard["ok"] is False and REFUSAL in dashboard["error"]
    assert REFUSAL in _rpc({"action": "update", "name": "cat-plugin"})["error"]["message"]
    # `hermes plugins update` (all) lists it as held, not as an update it will apply.
    plan = plan_update_all(run_checks(gated["plugins_dir"], include_pip=False))
    assert not plan.movable
    assert [(row.name, "after updating Hermes" in why) for row, why in plan.skipped] == [("cat-plugin", True)]
    assert _head(target) == gated["sha1"]

    gated["state"]["running"] = "0.31.0"
    assert pc.dashboard_update_user_plugin("cat-plugin")["unchanged"] is False
    assert _head(target) == gated["sha2"]


def test_portable_catalog_entry_gets_its_hermes_floor_from_the_catalog(gated, tmp_path):
    # A portable package (plugin.json) has no requires_hermes of its own: the catalog entry is its only floor.
    from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1
    repo = tmp_path / "portable-repo"
    repo.mkdir()
    sp.run(["git", "init", "-q"], cwd=repo, check=True)
    shas = []
    for version in ("1.0.0", "2.0.0"):
        (repo / "plugin.json").write_text(json.dumps(
            {"$schema": PLUGIN_SCHEMA_V1, "name": "cat-plugin", "version": version}), encoding="utf-8")
        shas.append(_commit(repo, version))
    gated["state"].update(repo_uri=repo.as_uri(), pin=shas[0], requires=">=0.31")

    refused = pc.dashboard_install_plugin("", force=False, enable=False, catalog_name="cat-plugin")
    assert refused["ok"] is False and REFUSAL in refused["error"]
    assert not (gated["plugins_dir"] / "cat-plugin").exists()

    gated["state"]["requires"] = ""
    assert pc.dashboard_install_plugin("", force=False, enable=False, catalog_name="cat-plugin")["ok"] is True
    gated["state"].update(pin=shas[1], requires=">=0.31")
    held = pc.dashboard_update_user_plugin("cat-plugin")
    assert held["ok"] is False and REFUSAL in held["error"]
    assert _head(gated["plugins_dir"] / "cat-plugin") == shas[0]
