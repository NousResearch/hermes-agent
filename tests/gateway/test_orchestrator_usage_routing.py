from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

from agent.delegation_routing import ProviderUsage
from gateway.run import GatewayRunner

PRIMARY_PROVIDER = "openai-codex"
PRIMARY_MODEL = "gpt-5.6-sol"
FALLBACK_PROVIDER = "google-antigravity"
FALLBACK_MODEL = "gemini-3.1-pro-high"

CONFIG = {
    "agent": {
        "orchestrator_usage_routing": {
            "enabled": True,
            "primary_provider": PRIMARY_PROVIDER,
            "primary_model": PRIMARY_MODEL,
            "fallback_provider": FALLBACK_PROVIDER,
            "fallback_model": FALLBACK_MODEL,
            "switch_at_remaining_percent": 10,
            "restore_above_remaining_percent": 10,
            "usage_ttl_seconds": 900,
            "usage_stale_seconds": 7200,
            "primary_usage_window_prefixes": ["Session", "Weekly"],
        }
    }
}


def _runner():
    runner = object.__new__(GatewayRunner)
    runner._service_tier = None
    return runner


def _runtime(provider=PRIMARY_PROVIDER):
    fallback = provider == FALLBACK_PROVIDER
    return {
        "api_key": "agy-key" if fallback else "codex-key",
        "base_url": "https://agy.invalid" if fallback else "https://codex.invalid",
        "provider": provider,
        "requested_provider": provider,
        "api_mode": "antigravity" if fallback else "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
        "max_tokens": 1234,
        "provider_project_id": "agy-project" if fallback else None,
    }


def _resolved(provider):
    value = _runtime(provider)
    value["project_id"] = value.pop("provider_project_id")
    value["model"] = FALLBACK_MODEL if provider == FALLBACK_PROVIDER else PRIMARY_MODEL
    return value


def _route(
    *,
    current_provider,
    current_model,
    remaining,
    freshness,
    resolve=None,
    config=CONFIG,
    current_agent=None,
):
    usage = ProviderUsage(
        provider=PRIMARY_PROVIDER,
        remaining_percent=remaining,
        freshness=freshness,
        age_seconds=1,
    )
    resolver = resolve if resolve is not None else _resolved(FALLBACK_PROVIDER)
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch(
            "agent.delegation_usage_cache.build_usage_view",
            return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
        ),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=resolver) as runtime_resolve,
    ):
        route = GatewayRunner._resolve_turn_agent_config(
            _runner(),
            "hi",
            current_model,
            _runtime(current_provider),
            current_agent=current_agent,
        )
    return route, runtime_resolve


def test_low_codex_switches_to_agy_at_turn_boundary():
    route, resolver = _route(
        current_provider=PRIMARY_PROVIDER,
        current_model=PRIMARY_MODEL,
        remaining=10,
        freshness="fresh",
    )
    assert route["model"] == FALLBACK_MODEL
    assert route["runtime"]["provider"] == FALLBACK_PROVIDER
    assert route["runtime"]["provider_project_id"] == "agy-project"
    resolver.assert_called_once_with(
        requested=FALLBACK_PROVIDER, target_model=FALLBACK_MODEL
    )


def test_explicit_non_policy_model_is_not_overridden():
    explicit_model = "gpt-5.4"
    route, resolver = _route(
        current_provider=PRIMARY_PROVIDER,
        current_model=explicit_model,
        remaining=3,
        freshness="fresh",
        current_agent=SimpleNamespace(
            model=PRIMARY_MODEL,
            provider=PRIMARY_PROVIDER,
            requested_provider=PRIMARY_PROVIDER,
            api_key="codex-key",
            base_url="https://codex.invalid",
            api_mode="codex_responses",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            provider_project_id=None,
            max_tokens=1234,
        ),
    )
    assert route["model"] == explicit_model
    assert route["runtime"]["provider"] == PRIMARY_PROVIDER
    resolver.assert_not_called()


def test_fresh_reset_above_ten_restores_codex():
    route, resolver = _route(
        current_provider=FALLBACK_PROVIDER,
        current_model=FALLBACK_MODEL,
        remaining=80,
        freshness="fresh",
        resolve=_resolved(PRIMARY_PROVIDER),
    )
    assert route["model"] == PRIMARY_MODEL
    assert route["runtime"]["provider"] == PRIMARY_PROVIDER
    resolver.assert_called_once_with(
        requested=PRIMARY_PROVIDER, target_model=PRIMARY_MODEL
    )


def test_stale_good_reading_preserves_current_agy_without_resolution():
    route, resolver = _route(
        current_provider=FALLBACK_PROVIDER,
        current_model=FALLBACK_MODEL,
        remaining=80,
        freshness="stale",
    )
    assert route["model"] == FALLBACK_MODEL
    assert route["runtime"]["provider"] == FALLBACK_PROVIDER
    resolver.assert_not_called()


def test_unknown_preserves_current_agy_without_resolution():
    route, resolver = _route(
        current_provider=FALLBACK_PROVIDER,
        current_model=FALLBACK_MODEL,
        remaining=None,
        freshness="unknown",
    )
    assert route["runtime"]["provider"] == FALLBACK_PROVIDER
    resolver.assert_not_called()


def test_unknown_preserves_cached_session_agy_when_base_is_codex():
    route, resolver = _route(
        current_provider=PRIMARY_PROVIDER,
        current_model=PRIMARY_MODEL,
        remaining=None,
        freshness="unknown",
        current_agent=SimpleNamespace(
            model=FALLBACK_MODEL,
            provider=FALLBACK_PROVIDER,
            requested_provider=FALLBACK_PROVIDER,
            api_key="agy-key",
            base_url="https://agy.invalid",
            api_mode="antigravity",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
        ),
    )
    assert route["model"] == FALLBACK_MODEL
    assert route["runtime"]["provider"] == FALLBACK_PROVIDER
    resolver.assert_not_called()


def test_cached_agent_route_state_is_session_scoped():
    runner = _runner()
    agent = SimpleNamespace(provider=FALLBACK_PROVIDER, model=FALLBACK_MODEL)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {"chat:one": (agent, ("sig",), 3, "session-1")}

    assert (
        runner._cached_agent_for_orchestrator_route("chat:one", "session-1")
        is agent
    )
    assert (
        runner._cached_agent_for_orchestrator_route("chat:one", "session-2")
        is None
    )
    assert runner._cached_agent_for_orchestrator_route("chat:two", "session-1") is None


def test_cached_agent_session_rotation_preserves_same_agent_route_state():
    runner = _runner()
    agent = SimpleNamespace(provider=FALLBACK_PROVIDER, model=FALLBACK_MODEL)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {"chat:one": (agent, "sig", 3, "session-1")}

    runner._rotate_cached_agent_session_id(
        "chat:one", "session-1", "session-rotated", agent
    )

    assert (
        runner._cached_agent_for_orchestrator_route("chat:one", "session-rotated")
        is agent
    )
    assert runner._cached_agent_for_orchestrator_route("chat:one", "session-1") is None
    assert runner._agent_cache["chat:one"] == (
        agent,
        "sig",
        3,
        "session-rotated",
    )


def test_cached_agent_session_rotation_rejects_stale_or_different_agent():
    runner = _runner()
    agent = SimpleNamespace(provider=FALLBACK_PROVIDER, model=FALLBACK_MODEL)
    other = SimpleNamespace(provider=PRIMARY_PROVIDER, model=PRIMARY_MODEL)
    original = (agent, "sig", 3, "session-1")
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {"chat:one": original}

    runner._rotate_cached_agent_session_id(
        "chat:one", "different-old", "session-rotated", agent
    )
    runner._rotate_cached_agent_session_id(
        "chat:one", "session-1", "session-rotated", other
    )

    assert runner._agent_cache["chat:one"] == original


def test_rotated_session_unknown_usage_keeps_cached_agy():
    runner = _runner()
    agent = SimpleNamespace(
        model=FALLBACK_MODEL,
        provider=FALLBACK_PROVIDER,
        requested_provider=FALLBACK_PROVIDER,
        api_key="agy-key",
        base_url="https://agy.invalid",
        api_mode="antigravity",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        provider_project_id="agy-project",
        max_tokens=1234,
    )
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {"chat:one": (agent, "sig", 3, "session-1")}
    assert runner._rotate_cached_agent_session_id(
        "chat:one", "session-1", "session-rotated", agent
    )
    current_agent = runner._cached_agent_for_orchestrator_route(
        "chat:one", "session-rotated"
    )

    route, resolver = _route(
        current_provider=PRIMARY_PROVIDER,
        current_model=PRIMARY_MODEL,
        remaining=None,
        freshness="unknown",
        current_agent=current_agent,
    )

    assert route["model"] == FALLBACK_MODEL
    assert route["runtime"]["provider"] == FALLBACK_PROVIDER
    resolver.assert_not_called()


def test_gateway_session_cache_drives_unknown_hold_then_fresh_recovery():
    runner = _runner()
    agent = SimpleNamespace(
        model=FALLBACK_MODEL,
        provider=FALLBACK_PROVIDER,
        requested_provider=FALLBACK_PROVIDER,
        api_key="agy-key",
        base_url="https://agy.invalid",
        api_mode="antigravity",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
    )
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {"chat:one": (agent, ("sig",), 3, "session-1")}
    current_agent = runner._cached_agent_for_orchestrator_route(
        "chat:one", "session-1"
    )

    for remaining, freshness, expected_provider, expected_model in (
        (None, "unknown", FALLBACK_PROVIDER, FALLBACK_MODEL),
        (80, "fresh", PRIMARY_PROVIDER, PRIMARY_MODEL),
    ):
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER,
            remaining_percent=remaining,
            freshness=freshness,
            age_seconds=1,
        )
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider"
            ) as runtime_resolve,
        ):
            route = runner._resolve_turn_agent_config(
                "hi",
                PRIMARY_MODEL,
                _runtime(PRIMARY_PROVIDER),
                current_agent=current_agent,
            )
        assert route["model"] == expected_model
        assert route["runtime"]["provider"] == expected_provider
        runtime_resolve.assert_not_called()


def test_disabled_is_inert():
    route, resolver = _route(
        current_provider=PRIMARY_PROVIDER,
        current_model=PRIMARY_MODEL,
        remaining=0,
        freshness="fresh",
        config={"agent": {"orchestrator_usage_routing": {"enabled": False}}},
    )
    assert route["runtime"]["provider"] == PRIMARY_PROVIDER
    resolver.assert_not_called()


def test_target_runtime_resolution_failure_stays_current_without_secret_leak():
    usage = ProviderUsage(
        provider=PRIMARY_PROVIDER,
        remaining_percent=1,
        freshness="fresh",
        age_seconds=1,
    )
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=CONFIG),
        patch(
            "agent.delegation_usage_cache.build_usage_view",
            return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
        ),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=RuntimeError("account@example.com secret-token"),
        ),
    ):
        route = GatewayRunner._resolve_turn_agent_config(
            _runner(), "hi", PRIMARY_MODEL, _runtime(PRIMARY_PROVIDER)
        )
    assert route["model"] == PRIMARY_MODEL
    assert route["runtime"]["provider"] == PRIMARY_PROVIDER
    assert "secret-token" not in repr(route)
