"""An entry recorded with an env the package code no longer produces is re-recorded on the next
prepare, without re-fetching, so build admission accepts stores built before the change (a Desktop
bundle restoring a pm-store cache from before Chromium stopped exporting AGENT_BROWSER_EXECUTABLE_PATH)."""
from __future__ import annotations

import importlib.util
import sys

from pm.lock import Facts, Lockfile


def test_prepare_rerecords_an_outdated_env_without_refetching(tmp_path, monkeypatch):
    from pm import build_operations
    from pm.registry import _packages
    from pm.store import current_target, tree_digest

    target, store = current_target(), tmp_path / "tools"
    definition = tmp_path / "env_fixture.py"
    definition.write_text(
        "from pm.package import Package\n"
        "class EnvTool(Package):\n"
        "    name = 'env-tool'\n"
        "    def env(self, entry, target): return {'TOOL_ROOT': str(entry)}\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("env_fixture", definition)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "env_fixture", module)
    spec.loader.exec_module(module)
    package = module.EnvTool()
    monkeypatch.setitem(_packages, package.name, package)

    lock = Lockfile(tmp_path / "lock.json")
    # An unreachable URL: any attempt to re-fetch fails the test.
    lock.set_pin(package.name, "1.0", {target: {"url": "https://invalid.test/tool.tgz", "sha256": "a" * 64}})
    lock.save()
    monkeypatch.setattr("pm.install._lockfile", lambda: lock)
    entry = store / package.store_entry("1.0", target)
    entry.mkdir(parents=True)
    (entry / "data").write_text("tool", encoding="utf-8")
    stale = {"TOOL_ROOT": str(entry), "RETIRED_EXECUTABLE_PATH": str(entry / "data")}
    Facts(store / "facts.json").record(package.name, "1.0", entry.name, stale, store, target=target,
                                       artifacts=["a" * 64], digest=tree_digest(entry))

    build_operations.prepare_tools([package.name], out=store, target=target)

    assert Facts(store / "facts.json").env_for(package.name, store) == package.env(entry, target)
    selected = build_operations.verified_tools([package.name], source_store=store, target=target, lock=lock)
    assert selected.entries[package.name].path == entry
    assert (entry / "data").read_text(encoding="utf-8") == "tool"
