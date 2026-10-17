"""Independent review overlay: real resolvers, synthetic disk stores, no inference."""

import json
import time
import pytest
from tests.gateway import test_codex_effective_model_routing as routing_fixtures
from tests import credential_routing_fixtures as quota_fixtures

home = routing_fixtures.home
codex_quota_not_restored = quota_fixtures.codex_quota_not_restored
pytestmark = pytest.mark.usefixtures("codex_quota_not_restored")
from tests.gateway.test_codex_effective_model_routing import (
    _runner,
    _source,
    TARGET,
    OTHER,
    BASE,
)

FOREIGN = "https://independent.example/backend-api/codex"


def row(token="synthetic-blocked", source="manual", **extra):
    return dict(
        id=token,
        label=token,
        auth_type="oauth",
        source=source,
        priority=0,
        access_token=token,
        refresh_token="synthetic-refresh",
        expires_at_ms=4_000_000_000_000,
        base_url=BASE,
        **extra,
    )


def store(home, rows, singleton=None):
    data = {"credential_pool": {"openai-codex": rows}}
    if singleton:
        data["providers"] = {
            "openai-codex": {
                "tokens": {
                    "access_token": singleton,
                    "refresh_token": "synthetic-refresh",
                }
            }
        }
    (home / "auth.json").write_text(json.dumps(data))


def session(
    home,
    path,
    *,
    token="synthetic-blocked",
    base=BASE,
    provider="openai-codex",
    model=TARGET,
):
    runner = _runner(home)
    source = _source()
    key = runner.session_store.get_or_create_session(source).session_key
    runner._session_state(key).conversation.model_override = {
        "model": model,
        "provider": provider,
        "api_key": token,
        "base_url": base,
        "api_mode": "codex_responses",
    }
    if path == "cached":
        model, runtime = runner._apply_session_model_override(key, OTHER, {})
    else:
        model, runtime = runner._resolve_session_agent_runtime(
            source=source, session_key=key
        )
    # Actual credential carried forward to agent config, not merely pool availability.
    config = runner._resolve_turn_agent_config("hello", model, runtime)
    return model, runtime, config["runtime"]


@pytest.mark.parametrize("source", ["manual", "manual:device_code", "device_code"])
@pytest.mark.parametrize("state", ["dead", "model", "exhausted"])
def test_singleton_alias_cannot_bypass_via_newly_seeded_duplicate(home, source, state):
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback

    status = {
        "dead": {"last_status": "dead"},
        "model": {"model_cooldowns": {TARGET: 4_000_000_000}},
        "exhausted": {
            "last_status": "exhausted",
            "last_error_code": 429,
            "last_error_reset_at": 4_000_000_000,
        },
    }[state]
    store(home, [row(source=source, **status)], singleton="synthetic-blocked")
    config = json.loads((home / "config.yaml").read_text())
    config["fallback_providers"] = [
        {
            "provider": "openrouter",
            "model": "fallback-model",
            "api_key": "synthetic-fallback",
        }
    ]
    runtime, fallback = resolve_runtime_with_fallback(
        config, requested="openai-codex", target_model=TARGET
    )
    assert runtime["api_key"] != "synthetic-blocked", (runtime, fallback)
    assert runtime["api_key"] == "synthetic-fallback"
    assert fallback is not None


@pytest.mark.parametrize("path", ["fresh", "cached"])
@pytest.mark.parametrize(
    "status",
    [
        {"model_cooldowns": {TARGET: 4_000_000_000}},
        {"last_status": "dead"},
        {
            "last_status": "exhausted",
            "last_error_code": 429,
            "last_error_reset_at": 4_000_000_000,
        },
        {
            "last_status": "exhausted",
            "last_error_code": 429,
            "last_status_at": time.time(),
        },
    ],
)
def test_same_endpoint_sibling_wire_key(home, path, status):
    store(home, [row(**status), row("synthetic-healthy")])
    model, runtime, wire = session(home, path)
    assert model == TARGET
    assert wire["api_key"] == "synthetic-healthy"
    assert wire["base_url"] == BASE
    assert (
        runtime["credential_pool"].select(model=model).runtime_api_key
        == wire["api_key"]
    )


@pytest.mark.parametrize("path", ["fresh", "cached"])
def test_mixed_endpoint_siblings_do_not_hide_healthy_same_endpoint(home, path):
    foreign = row("synthetic-foreign")
    foreign["base_url"] = FOREIGN
    store(
        home,
        [
            row(model_cooldowns={TARGET: 4_000_000_000}),
            foreign,
            row("synthetic-healthy"),
        ],
    )
    _model, _runtime, wire = session(home, path)
    assert wire["api_key"] == "synthetic-healthy"
    assert wire["base_url"] == BASE


@pytest.mark.parametrize("path", ["fresh", "cached"])
def test_healthy_pinned_endpoint_not_refused_due_to_foreign_priority(home, path):
    foreign = row("synthetic-foreign")
    foreign["base_url"] = FOREIGN
    own = row()
    own["priority"] = 1
    store(home, [foreign, own])
    _model, _runtime, wire = session(home, path)
    assert wire["api_key"] == "synthetic-blocked"
    assert wire["base_url"] == BASE


@pytest.mark.parametrize("path", ["fresh", "cached"])
@pytest.mark.parametrize("kind", ["key", "endpoint", "provider"])
def test_independent_route_identity(home, path, kind):
    store(
        home, [row(model_cooldowns={TARGET: 4_000_000_000}), row("synthetic-healthy")]
    )
    key = "synthetic-independent" if kind == "key" else "synthetic-blocked"
    base = FOREIGN if kind == "endpoint" else BASE
    provider = "custom:independent" if kind == "provider" else "openai-codex"
    _, runtime, wire = session(home, path, token=key, base=base, provider=provider)
    assert wire["api_key"] == key
    assert wire["provider"] == provider
    assert wire["base_url"] == base
    assert runtime.get("credential_pool") is None


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize(
    "reset", [4_000_000_000, 4_000_000_000_000, "2096-10-02T07:06:40Z"]
)
def test_diagnostic_future_exhaustion_is_read_only(home, alias, reset):
    from hermes_cli.auth import resolve_codex_runtime_credentials, AuthError

    store(
        home,
        [row(last_status="exhausted", last_error_code=429, last_error_reset_at=reset)],
        singleton="synthetic-blocked" if alias else None,
    )
    before = (home / "auth.json").read_bytes()
    with pytest.raises(AuthError):
        resolve_codex_runtime_credentials(read_only=True, target_model=TARGET)
    assert (home / "auth.json").read_bytes() == before
    assert not (home / "auth.lock").exists()


@pytest.mark.parametrize(
    "status",
    [
        {"last_status": "exhausted", "last_error_reset_at": 1000},
        {"model_cooldowns": {OTHER: 4_000_000_000}},
    ],
)
def test_eligible_alias_is_read_only(home, status):
    from hermes_cli.auth import resolve_codex_runtime_credentials

    store(home, [row(**status)], singleton="synthetic-blocked")
    before = (home / "auth.json").read_bytes()
    assert (
        resolve_codex_runtime_credentials(read_only=True, target_model=TARGET)[
            "api_key"
        ]
        == "synthetic-blocked"
    )
    assert (home / "auth.json").read_bytes() == before


def test_unknown_model_does_not_assume_config_default(home):
    from hermes_cli.auth import resolve_codex_runtime_credentials, AuthError

    store(
        home,
        [row(model_cooldowns={OTHER: 4_000_000_000})],
        singleton="synthetic-blocked",
    )
    before = (home / "auth.json").read_bytes()
    with pytest.raises(AuthError):
        resolve_codex_runtime_credentials(read_only=True)
    assert (home / "auth.json").read_bytes() == before


def test_explicit_key_precedence_even_if_alias(home):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    store(home, [row(model_cooldowns={TARGET: 4_000_000_000})])
    runtime = resolve_runtime_provider(
        requested="openai-codex",
        target_model=TARGET,
        explicit_api_key="synthetic-blocked",
        explicit_base_url=FOREIGN,
    )
    assert runtime["api_key"] == "synthetic-blocked"
    assert runtime["base_url"] == FOREIGN
    assert runtime.get("credential_pool") is None
