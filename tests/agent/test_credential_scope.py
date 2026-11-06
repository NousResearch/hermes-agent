# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""Request-bound opaque credential-scope identity for plugin hooks.

Covers: scope-id lifecycle (mint / stability / persistence / replacement),
fail-closed resolution, attempt-level attribution across retry rotation,
concurrency isolation between agents, secret-freeness, and the
``llm.credential_scope`` capability gate on hook delivery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from agent.credential_scope import (
    attempt_credential_scope,
    capture_attempt_credential_scope,
    resolve_credential_scope,
)

_PROVIDER = "nvidia"
_TOKEN_A = "synthetic-token-AAAA-not-a-real-key"
_TOKEN_B = "synthetic-token-BBBB-not-a-real-key"


def _entry(entry_id: str, token: str, *, priority: int = 0, scope_id=None) -> PooledCredential:
    raw = {
        "id": entry_id, "label": entry_id, "auth_type": "api_key",
        "priority": priority, "access_token": token, "source": "manual",
    }
    if scope_id is not None:
        raw["scope_id"] = scope_id
    return PooledCredential.from_dict(_PROVIDER, raw)


def _pool() -> CredentialPool:
    return CredentialPool(provider=_PROVIDER, entries=[
        _entry("entry-a", _TOKEN_A, priority=0),
        _entry("entry-b", _TOKEN_B, priority=1),
    ])


class _Agent:
    def __init__(self, pool, entry_id, api_key):
        self.provider = _PROVIDER
        self._credential_pool = pool
        self._credential_pool_entry_id = entry_id
        self.api_key = api_key

    def rotate_to(self, entry_id, api_key):
        self._credential_pool_entry_id = entry_id
        self.api_key = api_key


# ── scope id lifecycle ───────────────────────────────────────────────────────


class TestScopeIdLifecycle:
    def test_distinct_entries_get_distinct_scope_ids(self):
        pool = _pool()
        a, b = pool.entries()
        assert a.scope_id and b.scope_id
        assert a.scope_id != b.scope_id

    def test_scope_id_is_not_derived_from_token(self):
        (entry,) = CredentialPool(provider=_PROVIDER, entries=[_entry("e1", _TOKEN_A)]).entries()
        assert entry.scope_id
        assert entry.scope_id != _TOKEN_A
        assert _TOKEN_A not in entry.scope_id
        assert entry.scope_id != hashlib.sha256(_TOKEN_A.encode()).hexdigest()
        assert entry.scope_id != hashlib.sha256(_TOKEN_A.encode()).hexdigest()[:32]
        assert entry.scope_id != hashlib.md5(_TOKEN_A.encode()).hexdigest()

    def test_scope_survives_token_refresh_replace(self):
        """A refreshed token (dataclasses.replace, as the refresh path does)
        keeps the entry's scope id."""
        entry = _entry("entry-a", _TOKEN_A)
        refreshed = replace(entry, access_token="synthetic-token-AAAA-rotated")
        assert refreshed.scope_id == entry.scope_id
        assert refreshed.id == entry.id

    def test_scope_persists_in_row_and_roundtrips(self):
        entry = _entry("entry-a", _TOKEN_A)
        row = entry.to_dict()
        assert row["scope_id"] == entry.scope_id
        assert row["access_token"] == _TOKEN_A  # row itself is the store; scope adds no secret
        reloaded = PooledCredential.from_dict(_PROVIDER, row)
        assert reloaded.scope_id == entry.scope_id

    def test_legacy_row_without_scope_id_gets_one(self):
        raw = {"id": "legacy", "label": "legacy", "auth_type": "api_key",
               "priority": 0, "access_token": _TOKEN_A, "source": "manual"}
        entry = PooledCredential.from_dict(_PROVIDER, raw)
        assert entry.scope_id

    def test_replacement_entry_gets_fresh_scope(self):
        """Removing and re-adding the same credential mints a new scope."""
        first = _entry("entry-a", _TOKEN_A)
        second = _entry("entry-a", _TOKEN_A)
        assert first.scope_id != second.scope_id


# ── resolution ───────────────────────────────────────────────────────────────


class TestResolution:
    def test_known_scope_for_bound_pool_entry(self):
        pool = _pool()
        agent = _Agent(pool, "entry-b", _TOKEN_B)
        scope = resolve_credential_scope(agent)
        entry_b = next(e for e in pool.entries() if e.id == "entry-b")
        assert scope.provider_id == _PROVIDER
        assert scope.identity_status == "known"
        assert scope.credential_scope_id == entry_b.scope_id

    def test_unknown_when_keyed_but_no_pool(self):
        agent = _Agent(None, None, _TOKEN_A)
        scope = resolve_credential_scope(agent)
        assert scope.identity_status == "unknown"
        assert scope.credential_scope_id is None

    def test_unknown_when_entry_id_not_in_pool(self):
        agent = _Agent(_pool(), "missing-entry", _TOKEN_A)
        scope = resolve_credential_scope(agent)
        assert scope.identity_status == "unknown"
        assert scope.credential_scope_id is None

    def test_unavailable_when_keyless(self):
        agent = _Agent(None, None, "")
        scope = resolve_credential_scope(agent)
        assert scope.identity_status == "unavailable"
        assert scope.credential_scope_id is None

    def test_unavailable_when_no_provider(self):
        agent = _Agent(None, None, _TOKEN_A)
        agent.provider = ""
        scope = resolve_credential_scope(agent)
        assert scope.identity_status == "unavailable"


# ── attempt attribution ──────────────────────────────────────────────────────


class TestAttemptAttribution:
    def test_snapshot_survives_retry_rotation(self):
        """Attempt 1 on A fails, retry rotates to B: attempt 1's outcome must
        still be attributed to A."""
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        scope_a = capture_attempt_credential_scope(agent, "req-1").credential_scope_id

        agent.rotate_to("entry-b", _TOKEN_B)
        # The failed attempt's error hook reads the snapshot for req-1.
        assert attempt_credential_scope(agent, "req-1").credential_scope_id == scope_a

        # The next attempt on B captures B.
        scope_b = capture_attempt_credential_scope(agent, "req-2").credential_scope_id
        assert scope_b != scope_a
        assert attempt_credential_scope(agent, "req-2").credential_scope_id == scope_b

    def test_concurrent_agents_do_not_exchange_scopes(self):
        pool = _pool()
        agent_a = _Agent(pool, "entry-a", _TOKEN_A)
        agent_b = _Agent(pool, "entry-b", _TOKEN_B)
        capture_attempt_credential_scope(agent_a, "req-a")
        capture_attempt_credential_scope(agent_b, "req-b")
        assert attempt_credential_scope(agent_a, "req-a").credential_scope_id != (
            attempt_credential_scope(agent_b, "req-b").credential_scope_id
        )

    def test_scope_payload_carries_no_secret(self):
        pool = _pool()
        agent = _Agent(pool, "entry-a", _TOKEN_A)
        payload = capture_attempt_credential_scope(agent, "req-1").to_dict()
        encoded = json.dumps(payload)
        assert _TOKEN_A not in encoded
        assert _TOKEN_B not in encoded
        assert set(payload) == {"provider_id", "credential_scope_id", "identity_status"}


# ── capability gate ──────────────────────────────────────────────────────────


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _register(manager, plugin_id, sink):
    from hermes_cli.plugins import PluginContext, PluginManifest

    manifest = PluginManifest(name=plugin_id, source="user", key=plugin_id)
    ctx = PluginContext(manifest, manager)

    def _cb(**kwargs):
        sink.append(kwargs)

    ctx.register_hook("pre_api_request", _cb)
    return _cb


def _register_with_handle(manager, plugin_id, sink):
    from hermes_cli.plugins import PluginContext, PluginManifest

    manifest = PluginManifest(name=plugin_id, source="user", key=plugin_id)
    ctx = PluginContext(manifest, manager)

    def _cb(**kwargs):
        sink.append(kwargs)

    handle = ctx.register_hook("pre_api_request", _cb)
    return _cb, handle


class TestCapabilityGate:
    def test_capability_registered_consent_only(self):
        from hermes_cli.plugin_capabilities import CAPABILITY_REGISTRY

        spec = CAPABILITY_REGISTRY["llm.credential_scope"]
        assert spec.legacy_path == ()
        assert spec.description

    def test_granted_plugin_receives_scope_ungranted_does_not(self, hermes_home):
        from hermes_cli.plugin_capabilities import record_consent
        from hermes_cli.plugins import PluginManager

        record_consent("quota-plugin", ["llm.credential_scope"], ["llm.credential_scope"])
        manager = PluginManager()
        granted, denied = [], []
        _register(manager, "quota-plugin", granted)
        _register(manager, "other-plugin", denied)

        manager.invoke_hook(
            "pre_api_request", api_request_id="req-1",
            credential_scope={"provider_id": _PROVIDER,
                              "credential_scope_id": "scope-xyz",
                              "identity_status": "known"},
            provider=_PROVIDER,
        )
        assert granted and granted[0]["credential_scope"]["credential_scope_id"] == "scope-xyz"
        assert denied and "credential_scope" not in denied[0]
        # Other payload fields are unaffected for the ungranted plugin.
        assert denied[0]["provider"] == _PROVIDER

    def test_ownerless_callback_is_fail_closed(self, hermes_home):
        """Callbacks registered outside track_hook() never receive the scope."""
        from hermes_cli.plugins import PluginManager

        manager = PluginManager()
        seen = []
        manager._hooks.setdefault("pre_api_request", []).append(lambda **kw: seen.append(kw))
        manager.invoke_hook(
            "pre_api_request", api_request_id="req-1",
            credential_scope={"provider_id": _PROVIDER,
                              "credential_scope_id": "scope-xyz",
                              "identity_status": "known"},
        )
        assert seen and "credential_scope" not in seen[0]

    def test_stale_owner_entry_never_regrants_disposed_callback(self, hermes_home):
        """After a registration is disposed (targeted unload), re-appending the
        same callable outside track_hook() must not restore its grant."""
        from hermes_cli.plugin_capabilities import record_consent
        from hermes_cli.plugins import PluginManager

        record_consent("quota-plugin", ["llm.credential_scope"], ["llm.credential_scope"])
        manager = PluginManager()
        seen = []
        cb, handle = _register_with_handle(manager, "quota-plugin", seen)
        handle.dispose()
        assert cb not in manager._hooks.get("pre_api_request", [])
        assert not any(
            pid == "quota-plugin"
            for owners in manager._hook_callback_owners.values()
            for pid, _ref in owners.values()
        )
        manager._hooks.setdefault("pre_api_request", []).append(cb)  # ownerless re-append
        manager.invoke_hook(
            "pre_api_request", api_request_id="req-1",
            credential_scope={"provider_id": _PROVIDER,
                              "credential_scope_id": "scope-xyz",
                              "identity_status": "known"},
        )
        assert seen and "credential_scope" not in seen[0]

    def test_unload_all_clears_hook_owners(self, hermes_home):
        from hermes_cli.plugin_capabilities import record_consent
        from hermes_cli.plugins import PluginManager

        record_consent("quota-plugin", ["llm.credential_scope"], ["llm.credential_scope"])
        manager = PluginManager()
        _register(manager, "quota-plugin", [])
        assert any(manager._hook_callback_owners.values())
        manager.unload()
        assert not any(manager._hook_callback_owners.values())

    def test_scope_field_survives_hook_payload_sanitization(self):
        """The sanitizer must not redact the (non-secret) scope dict."""
        from agent.api_request_hooks import ApiRequestHooksMixin

        scope = {"provider_id": _PROVIDER, "credential_scope_id": "abc123",
                 "identity_status": "known"}
        sanitized = ApiRequestHooksMixin._sanitize_hook_payload({"credential_scope": scope})
        assert sanitized["credential_scope"] == scope
