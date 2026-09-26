"""``plugin_compat.disable_reason`` never scans a bundled plugin.

Bundled plugins ship in this tree, which CI keeps off compat paths, so the
post-removal skip has nothing to find there — and the check costs a config
read on every bundled load (discovery included).

This pins the call count (plugin_hits is never called for a bundled
manifest), not walked work: _scan_root already kept bundled plugins from
being walked before this guard existed.
"""

from __future__ import annotations

import datetime as _dt
from types import SimpleNamespace

from hermes_cli import plugin_compat


def test_bundled_plugins_are_never_scanned_or_disabled(monkeypatch):
    scanned = []
    monkeypatch.setattr(plugin_compat, "allow_deprecated_imports", lambda config=None: False)
    monkeypatch.setattr(plugin_compat, "plugin_hits", lambda m: scanned.append(m.source) or ["hit"])
    after = plugin_compat.COMPAT_REMOVAL_DATE + _dt.timedelta(days=1)

    assert plugin_compat.disable_reason(SimpleNamespace(source="bundled"), today=after) is None
    assert scanned == []

    # Positive control: the same hit under a user plugin IS scanned and disables it.
    assert plugin_compat.disable_reason(SimpleNamespace(source="user"), today=after)
    assert scanned == ["user"]
