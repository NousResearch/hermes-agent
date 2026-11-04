"""A fix to the bytes a package stages reaches entries an older ``stage`` already published.

Installed entries are skipped by every install path once current, so a ``stage`` change only ever
reached new installs and the next pin bump. ``Package.restage_edits`` names the files ``stage`` now
writes differently; ``hermes update`` (``restage_recorded``) and desktop build preparation apply them
in place and re-record the digest. Regression for #148552, where the bundled git's ``helper-selector``
default has to reach installs that already exist.
"""

from __future__ import annotations

import pytest

from pm import paths, registry
from pm.build_operations import prepare_tools, verified_tools
from pm.install import _installed_location, ensure
from pm.lock import Facts, Lockfile
from pm.store import Store, current_target
from tests.pm._fixtures import served as served  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
from tests.pm.test_pm_core import FakeTool, pm_env as pm_env  # noqa: PLC0414 -- the self-alias re-exports the pytest fixture


class RestagedTool(FakeTool):
    """The same pin after its package code learned to rewrite one staged file."""

    def stage(self, store, staged, version, target):
        super().stage(store, staged, version, target)
        for rel, data in self.restage_edits(staged, target).items():
            (staged / rel).write_bytes(data)

    def restage_edits(self, root, target):
        script = root / "bin" / "faketool"
        return {} if script.read_bytes() == b"#!fixed" else {"bin/faketool": b"#!fixed"}


def _through_update(build_store) -> Store:
    """What ``hermes update``'s tool step (``ensure_tools_for_sync``) runs: publish, then carry
    stage fixes. The entry must still verify against its recorded digest afterwards."""
    from pm.install import restage_recorded

    ensure("faketool", explicit=True)
    restage_recorded()
    location = _installed_location(registry.get_package("faketool"), Lockfile(paths.lockfile_path()),
                                   current_target(), verify=True)
    assert location is not None, "the entry no longer verifies against its recorded digest"
    return location[1]


def _through_build_preparation(build_store) -> Store:
    """Desktop bundle preparation from a cached store; admission re-checks the recorded digest."""
    prepare_tools(["faketool"], out=build_store, target=current_target())
    verified_tools(["faketool"], source_store=build_store, target=current_target())
    return Store(build_store)


def _entry(store: Store):
    fact = Facts(store.root / "facts.json").get("faketool")
    assert fact is not None
    return store.entry(fact["entry"])


@pytest.mark.parametrize("realize", [_through_update, _through_build_preparation])
def test_an_installed_entry_takes_a_staging_fix_in_place_without_refetching(pm_env, tmp_path, realize):
    _lock, _runtime, docroot, _base_url = pm_env
    entry = _entry(realize(tmp_path / "build-tools"))
    assert (entry / "bin" / "faketool").read_bytes() == b"#!x"

    registry._packages["faketool"] = RestagedTool()
    (docroot / "faketool-1.0.tar.gz").unlink()  # a re-fetch would now fail the install
    assert _entry(realize(tmp_path / "build-tools")) == entry

    assert (entry / "bin" / "faketool").read_bytes() == b"#!fixed"


def test_an_entry_that_no_longer_matches_its_record_is_not_edited_or_revouched(pm_env):
    from pm.install import restage_recorded

    ensure("faketool", explicit=True)
    store = Store(paths.writable_store_root())
    entry = _entry(store)
    (entry / "stray").write_text("changed after publication", encoding="utf-8")
    recorded = Facts(store.root / "facts.json").get("faketool")

    registry._packages["faketool"] = RestagedTool()
    restage_recorded()

    assert (entry / "bin" / "faketool").read_bytes() == b"#!x"
    assert Facts(store.root / "facts.json").get("faketool") == recorded
