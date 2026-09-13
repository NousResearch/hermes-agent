"""Installed runtime lookup uses the same shipped/addition precedence as ensure."""

from __future__ import annotations

import pm
from pm import paths
from pm.ensure import ensure
from pm.lock import Lockfile
from tests.pm.test_pm_authority import pm_env, served  # noqa: F401


def test_installed_lookup_prefers_current_pin_then_recorded_fallback(pm_env, tmp_path, monkeypatch):
    ensure("faketool", explicit=True)
    original = pm.installed_package("faketool")
    assert original is not None and original.binary.is_file()

    lock = Lockfile(pm_env["lockfile_path"])
    lock.set_pin("faketool", "2.0", {"any": {
        "url": pm_env["base_url"] + "/faketool-2.0.tar.gz", "sha256": "b" * 64,
    }})
    lock.save()
    assert pm.installed_package("faketool") is None
    assert pm.installed_package("faketool", allow_outdated=True) == original

    shipped = tmp_path / "sealed" / "tools"
    shipped.mkdir(parents=True)
    monkeypatch.setattr(paths, "store_root", lambda: shipped)
    monkeypatch.setattr(paths, "writable_store_root", lambda: pm_env["runtime"])
    assert pm.installed_package("faketool", allow_outdated=True) == original
    original.binary.unlink()
    assert pm.installed_package("faketool", allow_outdated=True) is None
