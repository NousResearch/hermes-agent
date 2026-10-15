"""Independent acceptance probes; synthetic homes, no inference."""

import json
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
    BASE,
)


def _write(home, rows, singleton=None):
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


def _row(**kwargs):
    return dict(
        id="blocked",
        label="blocked",
        auth_type="oauth",
        source="manual",
        priority=0,
        access_token="synthetic-blocked",
        refresh_token="synthetic-refresh",
        expires_at_ms=4_000_000_000_000,
        base_url=BASE,
        **kwargs,
    )


def test_exhausted_singleton_alias_is_not_usable(home):
    from hermes_cli.auth import resolve_codex_runtime_credentials, AuthError

    row = _row(
        last_status="exhausted", last_error_code=429, last_error_reset_at=4_000_000_000
    )
    _write(home, [row], singleton="synthetic-blocked")
    before = (home / "auth.json").read_bytes()
    try:
        import inspect

        kwargs = (
            {"target_model": TARGET}
            if "target_model"
            in inspect.signature(resolve_codex_runtime_credentials).parameters
            else {}
        )
        result = resolve_codex_runtime_credentials(read_only=True, **kwargs)
    except AuthError:
        return
    assert (home / "auth.json").read_bytes() == before
    assert result["api_key"] != "synthetic-blocked", result


def test_exhausted_singleton_alias_runtime_walks_fallback(home):
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback

    _write(
        home,
        [
            _row(
                last_status="exhausted",
                last_error_code=429,
                last_error_reset_at=4_000_000_000,
            )
        ],
        singleton="synthetic-blocked",
    )
    config = json.loads((home / "config.yaml").read_text())
    config["fallback_providers"] = [
        {
            "provider": "openrouter",
            "model": "fallback-model",
            "api_key": "synthetic-fallback",
        }
    ]
    (home / "config.yaml").write_text(json.dumps(config))
    runtime, fallback = resolve_runtime_with_fallback(
        config, requested="openai-codex", target_model=TARGET
    )
    assert fallback is not None, runtime
    assert runtime["api_key"] == "synthetic-fallback"


@pytest.mark.parametrize(
    "status", [{"model_cooldowns": {TARGET: 4_000_000_000}}, {"last_status": "dead"}]
)
@pytest.mark.parametrize("path", ["fresh", "cached"])
def test_cached_session_key_is_not_authoritative_when_pool_excludes_it(
    home, status, path
):
    _write(home, [_row(**status)])
    runner = _runner(home)
    source = _source()
    key = runner.session_store.get_or_create_session(source).session_key
    runner._session_state(key).conversation.model_override = {
        "model": TARGET,
        "provider": "openai-codex",
        "base_url": BASE,
        "api_key": "synthetic-blocked",
        "api_mode": "codex_responses",
    }
    try:
        if path == "fresh":
            _model, runtime = runner._resolve_session_agent_runtime(
                source=source, session_key=key
            )
        else:
            _model, runtime = runner._apply_session_model_override(key, TARGET, {})
    except RuntimeError:
        return
    assert (
        runtime.get("api_key") != "synthetic-blocked"
        or runtime.get("credential_pool") is not None
    ), runtime
