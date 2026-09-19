"""Named custom pools sharing an endpoint must retain their identity (#109015)."""

import json
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture(params=[("alpha", "beta"), ("beta", "alpha")])
def shared_endpoint_pools(request, tmp_path, monkeypatch):
    from agent.credential_pool import load_pool
    from hermes_cli.runtime_provider import resolve_runtime_provider

    other, own = request.param
    endpoint = "http://127.0.0.1:12345/v1"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Put the other identity first to expose URL-only lookup in both directions.
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "providers": {
            name: {"name": f"{name} relay", "base_url": endpoint}
            for name in (other, own)
        },
    }, sort_keys=False), encoding="utf-8")
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {
            pool_id: [{
                "id": pool_id, "label": pool_id, "auth_type": "api_key",
                "priority": 0, "source": "manual", "access_token": f"key-{name}",
            }]
            for name in (other, own)
            for pool_id in (name, f"custom:{name}-relay")
        },
    }), encoding="utf-8")
    # Real resolution and auth.json loading, not patched identity helpers.
    resolved = resolve_runtime_provider(requested=f"custom:{own}")
    assert resolved["base_url"] == endpoint
    assert resolved["api_key"] == f"key-{own}"
    return own, other, endpoint, resolved, load_pool


def test_shared_endpoint_pool_aliases_are_identity_bound(shared_endpoint_pools):
    from agent.credential_pool import credential_pool_matches_provider, resolve_runtime_pool_key

    own, other, endpoint, resolved, load_pool = shared_endpoint_pools
    for identity in (f"custom:{own}", f"custom:{own}-relay", own):
        assert resolve_runtime_pool_key(identity, endpoint) == own
        for pool_id in (other, f"custom:{other}-relay"):
            assert not credential_pool_matches_provider(load_pool(pool_id), identity, base_url=endpoint)
        for pool_id in (own, f"custom:{own}-relay"):
            assert credential_pool_matches_provider(load_pool(pool_id), identity, base_url=endpoint)
        for pool_id in (own, f"custom:{own}-relay"):
            # Explicit custom identities cannot override the configured endpoint.
            assert not credential_pool_matches_provider(
                load_pool(pool_id), f"custom:{own}", base_url="https://foreign.invalid/v1",
            )


def test_shared_endpoint_foreign_pool_is_rejected_by_consumers(shared_endpoint_pools):
    from agent.agent_init import _finalize_routing
    from agent.agent_runtime_helpers import recover_with_credential_pool

    own, other, endpoint, resolved, load_pool = shared_endpoint_pools
    for pool_id in (other, f"custom:{other}-relay", own, f"custom:{own}-relay"):
        pool = load_pool(pool_id)
        agent = SimpleNamespace(
            provider=f"custom:{own}", base_url=endpoint, api_key=resolved["api_key"],
            model="test-model", api_mode="chat_completions", _credential_pool=pool,
            _is_openrouter_url=lambda: False,
        )
        _finalize_routing(agent, "chat_completions", pool)
        if pool_id in (own, f"custom:{own}-relay"):
            assert agent._credential_pool is pool
            continue
        assert agent._credential_pool is None
        # Restored/fallback state can reintroduce a stale pool after construction.
        agent._credential_pool = pool
        before = [entry.to_dict() for entry in pool.entries()]
        recovered, _ = recover_with_credential_pool(
            agent, status_code=429, has_retried_429=True, error_context={"message": "Rate limit"},
        )
        assert recovered is False
        assert [entry.to_dict() for entry in pool.entries()] == before
        assert agent.api_key == resolved["api_key"]
