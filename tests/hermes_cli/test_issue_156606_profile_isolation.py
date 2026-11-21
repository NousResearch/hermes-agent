"""Dispatch 168 C1 — profile-home and capability isolation for #156606.

Consent AND credential-pool reads must use the owning plugin manager's
immutable profile home, never the ambient home of a multiplexed gateway.
An opaque scope id alone never authorizes another profile's token.
"""
from __future__ import annotations

import json

import pytest

from agent.quota_fetch import OFFICIAL_KEY_LIMITS

PLUGIN = "quota-plugin"
CAP = "llm.quota_observation"
SCOPE_A = "a" * 32
SCOPE_B = "b" * 32
TOKEN_A = "synthetic-token-A-not-real"
TOKEN_B = "synthetic-token-B-not-real"


def _home(tmp_path, name, *, granted, scope, token):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({
        "plugins": {"entries": {PLUGIN: {"granted_capabilities":
            [CAP, "llm.credential_scope"] if granted else []}}}
    }), encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({
        "credential_pool": {"openrouter": [{
            "id": scope[:8], "label": "synthetic", "auth_type": "api_key", "priority": 0,
            "access_token": token, "source": "manual", "scope_id": scope,
        }]}
    }), encoding="utf-8")
    return home


def _ctx(home):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager(scope_key=str(home))
    return PluginContext(PluginManifest(name=PLUGIN, source="user", key=PLUGIN), manager), manager


def _install_http_capture(monkeypatch, seen, *, body=None):
    payload = body or {"data": {"free_model_daily_requests": {"limit": 10, "remaining": 7, "used": 3}}}
    monkeypatch.setattr(
        "agent.quota_fetch._http_get_json",
        lambda url, token, timeout: (seen.update(token=token, timeout=timeout), (200, payload))[1],
    )


def _ambient(monkeypatch, home):
    monkeypatch.setenv("HERMES_HOME", str(home))


def test_a_granted_b_denied_uses_own_home_while_b_ambient(tmp_path, monkeypatch):
    home_a = _home(tmp_path, "A", granted=True, scope=SCOPE_A, token=TOKEN_A)
    _home(tmp_path, "B", granted=False, scope=SCOPE_B, token=TOKEN_B)
    ctx_a, _ = _ctx(home_a)
    seen = {}
    _install_http_capture(monkeypatch, seen)
    _ambient(monkeypatch, tmp_path / "B")

    result = ctx_a.quota_fetch(provider_id="openrouter", credential_scope_id=SCOPE_A,
                               quota_source=OFFICIAL_KEY_LIMITS)
    assert result["status"] == "ok"
    assert seen.get("token") == TOKEN_A, seen
    assert TOKEN_A not in json.dumps(result)


def test_a_denied_b_granted_same_plugin_id_refuses_without_network(tmp_path, monkeypatch):
    _home(tmp_path, "A", granted=False, scope=SCOPE_A, token=TOKEN_A)
    home_b = _home(tmp_path, "B", granted=True, scope=SCOPE_B, token=TOKEN_B)
    ctx_a, _ = _ctx(tmp_path / "A")
    ctx_b, _ = _ctx(home_b)
    seen = {}
    _install_http_capture(monkeypatch, seen)
    _ambient(monkeypatch, home_b)  # a granted profile is ambient — must NOT grant A

    denied = ctx_a.quota_fetch(provider_id="openrouter", credential_scope_id=SCOPE_A,
                               quota_source=OFFICIAL_KEY_LIMITS)
    assert denied["reason"] == "capability_not_granted"
    assert seen == {}

    _ambient(monkeypatch, tmp_path / "A")
    allowed = ctx_b.quota_fetch(provider_id="openrouter", credential_scope_id=SCOPE_B,
                                quota_source=OFFICIAL_KEY_LIMITS)
    assert allowed["status"] == "ok"
    assert seen.get("token") == TOKEN_B, seen


def test_cross_profile_scope_lookup_never_selects_other_token(tmp_path, monkeypatch):
    home_a = _home(tmp_path, "A", granted=True, scope=SCOPE_A, token=TOKEN_A)
    _home(tmp_path, "B", granted=True, scope=SCOPE_B, token=TOKEN_B)
    ctx_a, _ = _ctx(home_a)
    seen = {}
    _install_http_capture(monkeypatch, seen)
    _ambient(monkeypatch, tmp_path / "B")  # B is ambient and owns SCOPE_B

    result = ctx_a.quota_fetch(provider_id="openrouter", credential_scope_id=SCOPE_B,
                               quota_source=OFFICIAL_KEY_LIMITS)
    assert result["reason"] == "scope_not_found"
    assert seen == {}


def test_interleaved_profiles_each_use_their_own_credential(tmp_path, monkeypatch):
    home_a = _home(tmp_path, "A", granted=True, scope=SCOPE_A, token=TOKEN_A)
    home_b = _home(tmp_path, "B", granted=True, scope=SCOPE_B, token=TOKEN_B)
    ctx_a, _ = _ctx(home_a)
    ctx_b, _ = _ctx(home_b)
    seen = {}
    _install_http_capture(monkeypatch, seen)

    _ambient(monkeypatch, home_b)
    assert ctx_a.quota_fetch(provider_id="openrouter", credential_scope_id=SCOPE_A,
                             quota_source=OFFICIAL_KEY_LIMITS)["status"] == "ok"
    assert seen["token"] == TOKEN_A
    _ambient(monkeypatch, home_a)
    assert ctx_b.quota_fetch(provider_id="openrouter", credential_scope_id=SCOPE_B,
                             quota_source=OFFICIAL_KEY_LIMITS)["status"] == "ok"
    assert seen["token"] == TOKEN_B


def test_hook_gated_fields_follow_manager_home_not_ambient(tmp_path, monkeypatch):
    from hermes_cli.plugin_capabilities import CAPABILITY_REGISTRY  # noqa: F401  (registry presence)

    home_a = _home(tmp_path, "A", granted=True, scope=SCOPE_A, token=TOKEN_A)
    _home(tmp_path, "B", granted=False, scope=SCOPE_B, token=TOKEN_B)
    _, manager_a = _ctx(home_a)
    _, manager_b = _ctx(tmp_path / "B")

    callback = lambda **kwargs: None
    for manager in (manager_a, manager_b):
        manager._hook_callback_owners = {"post_api_request": {id(callback): (PLUGIN, callback)}}
    payload = {"credential_scope": {"credential_scope_id": SCOPE_A}, "quota_observation": {"x": 1},
               "text": "safe"}

    _ambient(monkeypatch, tmp_path / "B")  # denied profile ambient
    assert manager_a._kwargs_for_hook_callback("post_api_request", callback, payload) == payload

    _ambient(monkeypatch, home_a)  # granted profile ambient
    sliced = manager_b._kwargs_for_hook_callback("post_api_request", callback, payload)
    assert sliced == {"text": "safe"}
    assert TOKEN_A not in json.dumps(sliced)
