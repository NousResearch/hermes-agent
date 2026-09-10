"""Catalog-aware ``hermes plugins`` surface (hermes_cli/plugins_cmd_catalog.py): a bare catalog name installs
the PINNED sha and records provenance; the kill list blocks every install path (CLI needs an explicit
bypass, dashboard/TUI have none); ``update`` re-pins instead of pulling. Real git, file:// repos."""

from __future__ import annotations

import json
import os
import shutil
import subprocess as sp
from pathlib import Path

import pytest

from hermes_cli import plugin_catalog as pc_cat
from hermes_cli import plugins_cmd as pc
from hermes_cli import plugins_cmd_catalog as cat

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

_GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def _commit(repo: Path, msg: str) -> str:
    sp.run(["git", "add", "-A"], cwd=repo, check=True, env=_GIT_ENV)
    sp.run(["git", "commit", "-q", "-m", msg], cwd=repo, check=True, env=_GIT_ENV)
    return sp.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A file:// plugin repo with two commits, a catalog pinned to the FIRST, an isolated plugins dir."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "plugin.yaml").write_text("name: cat-plugin\nversion: 1.0.0\ndescription: d\n")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n")
    sp.run(["git", "init", "-q"], cwd=repo, check=True, env=_GIT_ENV)
    sha1 = _commit(repo, "v1")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass  # v2\n")
    sha2 = _commit(repo, "v2")

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)
    monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: False)
    monkeypatch.setattr(pc, "_console", lambda: type("C", (), {"print": lambda *a, **k: None})())

    # Catalog: one entry pinned to sha1, mutable via state["pin"]; kill list via state["removed"]. The
    # real loader is https-only, so the fixture entry is built directly (file:// repo).
    state = {"pin": sha1, "removed": []}

    def _entries():
        return [pc_cat.PluginCatalogEntry(name="cat-plugin", repo=repo.as_uri(), sha=state["pin"],
                                          description="d", maintainer="t")]

    monkeypatch.setattr(pc_cat, "load_catalog", lambda catalog_dir=None: _entries())
    monkeypatch.setattr(pc_cat, "fetch_live_catalog", lambda **_: None)  # in-tree only, no network
    monkeypatch.setattr(pc_cat, "load_removed_list", lambda catalog_dir=None: list(state["removed"]))
    return {"repo": repo, "sha1": sha1, "sha2": sha2, "plugins_dir": plugins_dir, "state": state}


def _head(path: Path) -> str:
    return sp.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()


def test_catalog_name_installs_pinned_sha_with_sidecar_then_update_repins(world, monkeypatch):
    entry = pc_cat.get_live_catalog_entry("cat-plugin")
    assert entry is not None
    target, _m, name = cat.install_catalog_entry(entry, force=False)
    assert name == "cat-plugin"
    assert _head(target) == world["sha1"] != world["sha2"]  # pinned, not HEAD
    sidecar = json.loads((target / cat.CATALOG_SIDECAR).read_text())
    assert (sidecar["catalog_name"], sidecar["sha"]) == ("cat-plugin", world["sha1"])
    assert cat.catalog_annotation(target) == f"catalog:community@{world['sha1'][:8]}"

    # Dashboard update on a catalog install = re-pin. Pin unchanged → no-op.
    assert pc.dashboard_update_user_plugin("cat-plugin") == {
        "ok": True, "name": "cat-plugin", "sha": world["sha1"], "unchanged": True}
    # Bump the catalog pin → the checkout moves to exactly that sha.
    world["state"]["pin"] = world["sha2"]
    assert pc.dashboard_update_user_plugin("cat-plugin")["unchanged"] is False
    assert _head(world["plugins_dir"] / "cat-plugin") == world["sha2"]


def test_kill_list_blocks_cli_dashboard_and_tui_paths(world, monkeypatch):
    world["state"]["removed"].append(
        pc_cat.RemovedEntry(name="cat-plugin", repo=world["repo"].as_uri(), reason="malware"))
    # Dashboard/TUI: catalog name AND raw repo URL both refused, no bypass parameter exists.
    assert "malware" in pc.dashboard_install_plugin("", force=False, enable=False, catalog_name="cat-plugin")["error"]
    assert "malware" in pc.dashboard_install_plugin(world["repo"].as_uri(), force=False, enable=False)["error"]
    assert not (world["plugins_dir"] / "cat-plugin").exists()
    # CLI: refused by default, `--allow-removed` installs anyway.
    with pytest.raises(SystemExit):
        pc.cmd_install("cat-plugin", enable=False)
    pc.cmd_install("cat-plugin", enable=False, allow_removed=True)
    assert (world["plugins_dir"] / "cat-plugin" / cat.CATALOG_SIDECAR).exists()
    assert cat.removed_annotation("cat-plugin", world["plugins_dir"] / "cat-plugin") == "malware"
