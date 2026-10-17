"""Session routes cannot turn an excluded pooled account into a standalone key."""

import json

import pytest

from tests.gateway import test_codex_effective_model_routing as routing_fixtures
from tests.gateway.test_codex_effective_model_routing import BASE, OTHER, TARGET, _runner, _source

home = routing_fixtures.home

EXCLUDED = [
    {"model_cooldowns": {TARGET: 4_000_000_000}},
    {"last_status": "dead"},
    {"last_status": "exhausted", "last_error_code": 429, "last_error_reset_at": 4_000_000_000_000},
]
FOREIGN = "https://session-endpoint.example/backend-api/codex"


def _row(token="synthetic-blocked", **status):
    return {"id": token, "label": token, "source": "manual", "priority": 0,
            "auth_type": "oauth", "access_token": token, "refresh_token": "synthetic-refresh",
            "expires_at_ms": 4_000_000_000_000, "base_url": BASE, **status}


def _route(home, path, rows, *, key="synthetic-blocked", base=BASE, provider="openai-codex", pool=None):
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": rows}}))
    runner = _runner(home)
    source = _source()
    session = runner.session_store.get_or_create_session(source).session_key
    runner._session_state(session).conversation.model_override = {
        "model": TARGET, "provider": provider, "base_url": base, "api_key": key,
        "api_mode": "codex_responses", "credential_pool": pool}
    if path == "cached":
        return runner._apply_session_model_override(session, OTHER, {})
    return runner._resolve_session_agent_runtime(source=source, session_key=session)


@pytest.fixture
def quota_still_exhausted(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth._probe_codex_quota_restored", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.auth_codex._probe_codex_quota_restored", lambda *a, **kw: False)


@pytest.mark.parametrize("path", ["fresh", "cached"])
@pytest.mark.parametrize("status", EXCLUDED)
@pytest.mark.parametrize("sibling", ["none", "same-endpoint", "foreign-endpoint"])
def test_excluded_session_key_is_replaced_or_refused(home, quota_still_exhausted, path, status, sibling):
    rows = [_row(**status)]
    if sibling != "none":
        rows.append(_row("synthetic-healthy", base_url=BASE if sibling == "same-endpoint" else FOREIGN))
    if sibling == "same-endpoint":
        model, runtime = _route(home, path, rows)
        assert model == TARGET
        assert runtime["provider"] == "openai-codex"
        assert runtime["base_url"] == BASE
        assert runtime["api_key"] == "synthetic-healthy"
        assert runtime["credential_pool"].select(model=model).runtime_api_key == "synthetic-healthy"
    else:
        with pytest.raises(RuntimeError):
            _route(home, path, rows)


@pytest.mark.parametrize("path", ["fresh", "cached"])
def test_attached_pool_does_not_hide_a_new_bench(home, path):
    from agent.credential_pool import load_pool
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [_row()]}}))
    stale_pool = load_pool("openai-codex")
    assert stale_pool.select(model=TARGET).runtime_api_key == "synthetic-blocked"
    _, runtime = _route(home, path, [_row(**EXCLUDED[0]), _row("synthetic-healthy")], pool=stale_pool)
    assert runtime["api_key"] == "synthetic-healthy"
    assert runtime["credential_pool"].select(model=TARGET).runtime_api_key == "synthetic-healthy"


@pytest.mark.parametrize("path", ["fresh", "cached"])
@pytest.mark.parametrize("status", [
    {"model_cooldowns": {OTHER: 4_000_000_000}},
    {"model_cooldowns": {TARGET: 1}},
    {"last_status": "exhausted", "last_error_reset_at": 1_000},
])
def test_eligible_session_key_remains_available(home, path, status):
    model, runtime = _route(home, path, [_row(**status)])
    assert model == TARGET
    assert runtime["api_key"] == "synthetic-blocked"
    assert runtime["base_url"] == BASE


@pytest.mark.parametrize("path", ["fresh", "cached"])
@pytest.mark.parametrize("kind", ["independent-key", "different-endpoint", "different-provider"])
def test_session_key_identity_prevents_pool_credential_borrowing(home, path, kind):
    key = "synthetic-independent" if kind == "independent-key" else "synthetic-blocked"
    base = FOREIGN if kind == "different-endpoint" else BASE
    provider = "custom:session" if kind == "different-provider" else "openai-codex"
    model, runtime = _route(home, path, [_row(**EXCLUDED[0]), _row("synthetic-healthy")],
                            key=key, base=base, provider=provider)
    assert model == TARGET
    assert runtime["api_key"] == key
    assert runtime["base_url"] == base
    assert runtime["provider"] == provider
    assert runtime.get("credential_pool") is None
