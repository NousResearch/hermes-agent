"""Regression tests for #134587: an API-server per-request ``provider`` must swap the
default provider's ``request_overrides`` (e.g. a custom provider's ``extra_body``) and
``capabilities`` for the requested provider's, not keep them on the agent.

``_resolve_request_runtime_agent_kwargs()`` mirrors ``gateway.run``'s provider resolver,
and ``_apply_runtime_agent_overrides()`` is the merge the per-request provider path runs
after the agent kwargs were resolved from the gateway default — both ends must carry the
two fields for the swap to happen (an absent/None value is skipped, leaving the default
provider's body riding along to the other provider).
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

_DEFAULT_EXTRA_BODY = {"max_tokens": 4096, "include_reasoning": True}


def _runtime(provider: str, **extra) -> dict:
    base = {
        "api_key": "sk-" + uuid.uuid4().hex,
        "base_url": f"https://{provider}.test/v1",
        "provider": provider,
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "request_overrides": {"extra_body": dict(_DEFAULT_EXTRA_BODY)},
        "capabilities": {"supports_vision": False},
    }
    base.update(extra)
    return base


def _resolve(provider_runtime: dict) -> dict:
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kw: dict(provider_runtime),
    ):
        from gateway.platforms.api_server import _resolve_request_runtime_agent_kwargs

        return _resolve_request_runtime_agent_kwargs(
            "requested", target_model="model-x"
        )


def test_resolve_returns_requested_provider_request_overrides_and_capabilities():
    resolved = _resolve(_runtime("openai-codex"))
    assert resolved["request_overrides"] == {"extra_body": dict(_DEFAULT_EXTRA_BODY)}
    assert resolved["capabilities"] == {"supports_vision": False}


def test_resolve_defaults_absent_overrides_to_empty_dicts_not_none():
    resolved = _resolve(_runtime("openai", request_overrides=None, capabilities=None))
    # Empty dicts (not None) so _apply_runtime_agent_overrides() replaces, not skips.
    assert resolved["request_overrides"] == {}
    assert resolved["capabilities"] == {}


def test_apply_replaces_default_provider_overrides_with_requested_ones():
    from gateway.platforms.api_server import _apply_runtime_agent_overrides

    runtime_kwargs = {  # gateway default: custom provider with an extra_body
        "api_key": "sk-" + uuid.uuid4().hex,
        "provider": "custom:default",
        "request_overrides": {"extra_body": dict(_DEFAULT_EXTRA_BODY)},
        "capabilities": {"supports_vision": False},
    }
    requested = _resolve(
        _runtime("openai-codex", request_overrides={}, capabilities={})
    )
    _apply_runtime_agent_overrides(runtime_kwargs, requested)
    assert runtime_kwargs["provider"] == "openai-codex"
    assert runtime_kwargs["request_overrides"] == {}
    assert runtime_kwargs["capabilities"] == {}


def test_apply_carries_requested_provider_own_overrides():
    from gateway.platforms.api_server import _apply_runtime_agent_overrides

    runtime_kwargs = {
        "provider": "custom:default",
        "request_overrides": {},
        "capabilities": {},
    }
    own = {"extra_body": {"text": {"verbosity": "low"}}}
    requested = _resolve(
        _runtime(
            "openai-codex",
            request_overrides=own,
            capabilities={"supports_vision": True},
        )
    )
    _apply_runtime_agent_overrides(runtime_kwargs, requested)
    assert runtime_kwargs["request_overrides"] == own
    assert runtime_kwargs["capabilities"] == {"supports_vision": True}


def test_apply_skips_absent_keys_like_session_model_overrides():
    from gateway.platforms.api_server import _apply_runtime_agent_overrides

    runtime_kwargs = {
        "provider": "custom:default",
        "request_overrides": {"extra_body": {}},
    }
    # A session /model override carries none of the runtime fields — nothing may change.
    _apply_runtime_agent_overrides(runtime_kwargs, {"model": "gpt-6.1-sol"})
    assert runtime_kwargs["request_overrides"] == {"extra_body": {}}
