"""Cross-tenant served-profile guard: a host gateway from ANOTHER HERMES_HOME never serves this
one (#121352).

Two Hermes tenants on one host (separate ``HERMES_HOME``s, each a multiplexing gateway with its
own ``default`` profile) — the second gateway's start guard sees the FIRST tenant's host record
serving a profile named ``default``, treats it as "someone already serves YOU", prints
``The host gateway already serves profile 'default'`` and exits 78 (EX_CONFIG), so the unit parks
and the second tenant stays down.

``_served_by_another_host_gateway`` accepted any host-record gateway whose ``home`` merely
*differs* from this process's home — which is exactly the satellite-profile case it was built
for, but also matches a foreign tenant: two homes share the profile NAME, not the installation
root. A gateway serves this process's profile only when the two homes belong to the same
installation (equal, or one an ancestor of the other: ``<root>`` vs ``<root>/profiles/<name>``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_constants


def _fake_gateway(home: str):
    return SimpleNamespace(
        home=home, profile_label="default",
        describe=lambda: f"PID 1 (home={home}); serves: default",)


@pytest.fixture
def cross_tenant(tmp_path, monkeypatch):
    root_a = tmp_path / "hermes-a" / "home"
    home_b = tmp_path / "hermes-b" / "home"
    root_a.mkdir(parents=True)
    home_b.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    from gateway import status as gw_status
    from hermes_cli import gateway as gw
    monkeypatch.setattr(gw_status, "_get_process_hermes_home", lambda: str(home_b))
    monkeypatch.setattr(gw, "host_multiplexer_serving",
                        lambda name=None: _fake_gateway(str(root_a)))
    return gw, root_a, home_b


def test_other_tenants_default_gateway_is_not_the_owner(cross_tenant):
    gw, _root_a, _home_b = cross_tenant

    # RED pre-fix: returned the foreign gateway, so `_named_profile_refused_under_multiplexer`
    # refused the second tenant with exit 78 instead of letting it start.
    assert gw._served_by_another_host_gateway("default") is None


def test_satellite_profile_is_still_served_by_its_own_roots_gateway(tmp_path, monkeypatch):
    """No regression: the multiplexer-vs-satellite case the home check was built for (#97120)."""
    root = tmp_path / "hermes"
    home_x = root / "profiles" / "coder"
    home_x.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home_x))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    from gateway import status as gw_status
    from hermes_cli import gateway as gw
    monkeypatch.setattr(gw_status, "_get_process_hermes_home", lambda: str(home_x))
    monkeypatch.setattr(gw, "host_multiplexer_serving",
                        lambda name=None: _fake_gateway(str(root)))

    owner = gw._served_by_another_host_gateway("coder")
    assert owner is not None and owner.home == str(root)
