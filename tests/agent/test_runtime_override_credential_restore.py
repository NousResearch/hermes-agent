"""Regression contract for PR #99053 finding G3: a model-only override scope must not
roll back client credential state it never changed.

A 401/quota retry inside the scope rotates credentials (`_swap_credential` updates
`api_key`, `base_url` and `_client_kwargs[...]`). On scope exit the pre-override
`_client_kwargs` was restored while the live `api_key` stayed rotated, so the next
per-request client was rebuilt from the stale dict and reused the failed credential.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from agent.runtime_override import apply_runtime_override
from run_agent import AIAgent
from tests.agent.test_runtime_override import (
    _OVERRIDE_MODEL,
    _SESSION_MODEL,
    _build_model_owned_agent,
    _patch_model_owned_resolution,
)


def test_credential_rotation_inside_the_scope_is_not_rolled_back(monkeypatch):
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    agent.api_key = "old-key"
    agent.base_url = "https://api.openai.com/v1"
    agent._client_kwargs = {"api_key": "old-key", "base_url": agent.base_url}

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        # What the retry loop's credential-pool rotation does on a 401/quota error
        # (`_swap_credential` -> `_apply_primary_runtime_fields`).
        agent.api_key = "rotated-key"
        agent.base_url = "https://rotated.example.com/v1"
        agent._client_kwargs["api_key"] = agent.api_key
        agent._client_kwargs["base_url"] = agent.base_url

    assert agent.api_key == "rotated-key"
    assert agent._client_kwargs.get("api_key") == "rotated-key", (
        "the scope rolled the wire credentials back to the failed key"
    )
    assert agent._client_kwargs.get("base_url") == agent.base_url, (
        "the scope rolled the wire base_url back to the failed route"
    )


def test_model_only_scope_leaves_client_kwargs_untouched(monkeypatch):
    """The scope owns neither dict, so a plain override must not rewrite them."""
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    agent._client_kwargs = {"service_tier": "standard", "api_key": "old-key"}
    agent._transport_cache = {"chat_completions": "warm"}
    before = (dict(agent._client_kwargs), dict(agent._transport_cache))

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        pass

    assert (dict(agent._client_kwargs), dict(agent._transport_cache)) == before


def test_transport_cache_cleared_by_rotation_stays_cleared(monkeypatch):
    """`_swap_credential` clears the transport cache on a route change; the scope
    must not re-populate it with transports bound to the failed route."""
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    agent._transport_cache = {"chat_completions": "transport-for-failed-route"}

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        agent._transport_cache.clear()  # what the rotation does on a route change

    assert agent._transport_cache == {}, (
        "the scope resurrected transports bound to the failed route"
    )


def test_real_swap_credential_inside_the_scope_persists(monkeypatch):
    """Drive the real credential-swap path (`_swap_credential`) instead of assigning
    attributes, so the lifecycle bookkeeping it performs is covered too."""
    monkeypatch.setattr("hermes_cli.anon_auth.route_can_serve_model", lambda *a, **k: True)
    agent = AIAgent(
        api_key="old-key",
        base_url="https://api.openai.com/v1",
        provider="openai",
        api_mode="chat_completions",
        model="session-model",
        session_id="s-swap",
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
    )
    try:
        agent._client_kwargs = {"api_key": "old-key", "base_url": agent.base_url}

        with apply_runtime_override(agent, {"model": "override-model"}):
            entry = SimpleNamespace(
                id="cred-2",
                runtime_api_key="rotated-key",
                runtime_base_url="https://api.openai.com/v1",
            )
            assert agent._swap_credential(entry) is True
            assert agent.api_key == "rotated-key"

        assert agent.model == "session-model"
        assert agent.api_key == "rotated-key"
        assert agent._client_kwargs.get("api_key") == "rotated-key", (
            "the scope rolled the wire credential back to the failed key"
        )
    finally:
        agent.close()


def test_rotation_survives_an_exception_inside_the_scope(monkeypatch):
    """A turn that raises after rotating credentials must not roll the rotation back
    (third-party review: opencode/nemotron-3-super negative cases)."""
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    agent.api_key = "old-key"
    agent._client_kwargs = {"api_key": "old-key"}

    with pytest.raises(RuntimeError):
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            agent.api_key = "rotated-key"
            agent._client_kwargs["api_key"] = "rotated-key"
            raise RuntimeError("turn blew up after the rotation")

    assert agent.api_key == "rotated-key"
    assert agent._client_kwargs["api_key"] == "rotated-key"


def test_refused_swap_changes_nothing(monkeypatch):
    """`_swap_credential` refuses an entry whose route cannot serve the model; a refused
    swap must leave the live credentials exactly as they were."""
    monkeypatch.setattr("hermes_cli.anon_auth.route_can_serve_model", lambda *a, **k: False)
    agent = AIAgent(
        api_key="old-key",
        base_url="https://api.openai.com/v1",
        provider="openai",
        api_mode="chat_completions",
        model="session-model",
        session_id="s-refused",
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
    )
    try:
        agent._client_kwargs = {"api_key": "old-key"}
        entry = SimpleNamespace(id="cred-x", runtime_api_key="rotated-key",
                                runtime_base_url="https://api.openai.com/v1")
        with apply_runtime_override(agent, {"model": "override-model"}):
            assert agent._swap_credential(entry) is False
        assert agent.api_key == "old-key"
        assert agent._client_kwargs.get("api_key") == "old-key"
    finally:
        agent.close()
