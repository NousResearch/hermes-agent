"""A live process holding a tools entry that already matches the update's bytes (#139088).

Windows refuses to rename a directory while any process keeps a file inside it
open without delete sharing, which is what a running gateway or Desktop does to
``tools\\git-*`` and ``tools\\python-*``. When an explicit install re-realizes the
SAME version (the recorded digest no longer matches the tree), the rename of the
live entry to ``.previous-*`` used to fail the whole run with WinError 5/32, so
every update ended "Desktop app build owed". The OS produces the hold here.
"""

from __future__ import annotations

import pytest

from pm.lock import Facts
from pm.store import tree_digest
from tests.pm.test_pm_core import pm_env as pm_env  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
from tests.pm._fixtures import served as served  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


def _installed_with_drifted_digest(pm_env):
    """Install faketool, then make its recorded digest disagree with the realized tree."""
    from pm.install import ensure

    _, runtime, *_ = pm_env
    ensure("faketool", base_env={})
    facts = Facts(runtime / "facts.json")
    fact = facts.get("faketool")
    entry = runtime / fact["entry"]
    facts.record("faketool", fact["version"], fact["entry"], {}, runtime, target=fact.get("target"),
                 artifacts=fact.get("artifacts"), digest="0" * 64)
    return runtime, entry


@pytest.mark.platforms("windows")
def test_held_entry_matching_the_pin_keeps_the_explicit_install_working(pm_env):
    from pm.install import ensure

    runtime, entry = _installed_with_drifted_digest(pm_env)
    with open(entry / "bin" / "faketool", "rb"):
        runner = ensure("faketool", base_env={}, explicit=True)
    assert entry.name in runner.env["PATH"]
    assert (entry / "bin" / "faketool").read_bytes() == b"#!x"
    assert not (runtime / f".previous-{entry.name}").exists()
    assert Facts(runtime / "facts.json").get("faketool")["digest"] == tree_digest(entry)


@pytest.mark.platforms("windows")
def test_held_entry_with_different_bytes_still_fails_the_install(pm_env):
    from pm.install import ensure
    from pm.package import InstallError

    runtime, entry = _installed_with_drifted_digest(pm_env)
    (entry / "bin" / "extra").write_bytes(b"not in the pin")
    with open(entry / "bin" / "faketool", "rb"), pytest.raises(InstallError):
        ensure("faketool", base_env={}, explicit=True)
    assert (entry / "bin" / "extra").exists()
    assert not (runtime / f".previous-{entry.name}").exists()
