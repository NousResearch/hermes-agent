from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import profiles
from hermes_cli.plugin_host_child import RemotePluginContext
from plugin_runtime.context import PluginContext
from plugin_runtime.host_bindings import clear_plugin_host_bindings


def _context(home):
    manifest = SimpleNamespace(name="demo")
    manager = SimpleNamespace(home_path=home)
    return PluginContext(manifest, manager)


def test_plugin_context_exposes_stable_profile_contract(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    named = root / "profiles" / "alice"
    named.mkdir(parents=True)

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: root / "profiles")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    clear_plugin_host_bindings()
    profiles._bind_plugin_profile_contract()
    ctx = _context(named)

    assert ctx.profile_name == "alice"
    assert ctx.profile_home == str(named.resolve())
    assert ctx.resolve_profile_home("default") == str(root.resolve())
    assert ctx.resolve_profile_home("alice") == str(named.resolve())
    ctx.validate_profile_name("alice")
    with pytest.raises(ValueError):
        ctx.validate_profile_name("../escape")


def test_settled_served_profiles_uses_live_gateway_record_not_desired_roster(
    tmp_path, monkeypatch
):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr(
        profiles,
        "profiles_to_serve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("desired profile roster must not back the plugin contract")
        ),
    )
    monkeypatch.setattr(
        "gateway.host_rendezvous.read_record",
        lambda role: SimpleNamespace(profiles=("default", "alice")),
    )

    clear_plugin_host_bindings()
    profiles._bind_plugin_profile_contract()

    assert _context(root).settled_served_profiles() == ("default", "alice")


def test_remote_plugin_context_carries_profile_identity_and_home():
    remote = RemotePluginContext(
        SimpleNamespace(),
        "demo",
        {
            "plugin_id": "demo",
            "profile_name": "alice",
            "profile_home": "/profiles/alice",
            "manifest": {"name": "demo"},
            "ctx_methods": [],
        },
    )

    assert remote.profile_name == "alice"
    assert remote.profile_home == "/profiles/alice"
