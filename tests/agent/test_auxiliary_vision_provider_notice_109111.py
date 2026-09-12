"""Regression tests for issue #109111.

Before the fix:
  - ``_try_anthropic()`` ignored the caller's model entirely — the aux client bound the
    curated ``default_aux_model`` for the Anthropic profile, which happens to equal
    ``claude-haiku-4-5-20251001`` (what most users configure), so the swap was invisible.
    ``_STRICT_VISION_BACKENDS["anthropic"]`` and ``_resolve_api_key_branch`` both dropped
    the ``model`` argument on the floor.
  - When an explicitly configured ``auxiliary.vision.provider`` failed to resolve (e.g.
    pointed at an unauthenticated backend), ``_resolve_call_client`` silently fell back to
    the auto vision chain, which happily served the call on the main model. A negative
    test (sabotage the provider, repeat the analysis) changed nothing and the detour was
    only visible in debug logs, never to the user.
"""

from __future__ import annotations

import pytest

import agent.auxiliary_client as aux
from tools.vision_tools import _with_route_notice


# ---------------------------------------------------------------------------
# 1. _try_anthropic honors an explicit model
# ---------------------------------------------------------------------------

@pytest.fixture
def anthropic_env(monkeypatch):
    """Hermetic Anthropic aux resolution: no pool, standalone token, fake SDK client."""
    monkeypatch.setattr(aux, "_select_pool_entry", lambda provider: (False, None))
    monkeypatch.setattr(
        "agent.anthropic_credentials.resolve_anthropic_token", lambda: "sk-test-token")
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client",
                        lambda token, base_url: object())
    monkeypatch.setattr(aux, "_get_aux_model_for_provider",
                        lambda provider_id, **kw: "claude-haiku-4-5-20251001")


def test_try_anthropic_explicit_model_wins(anthropic_env):
    _, model = aux._try_anthropic(model="claude-sonnet-4-5-20261001")
    assert model == "claude-sonnet-4-5-20261001"


def test_try_anthropic_defaults_to_curated_aux_model(anthropic_env):
    _, model = aux._try_anthropic()
    assert model == "claude-haiku-4-5-20251001"


def test_strict_vision_backend_forwards_model(anthropic_env, monkeypatch):
    captured = {}

    def fake_try_anthropic(explicit_api_key=None, model=None):
        captured["model"] = model
        return None, None

    monkeypatch.setattr(aux, "_try_anthropic", fake_try_anthropic)
    aux._resolve_strict_vision_backend("anthropic", "claude-sonnet-4-5-20261001")
    assert captured["model"] == "claude-sonnet-4-5-20261001"


# ---------------------------------------------------------------------------
# 2. Vision provider fallback surfaces a notice via route_info
# ---------------------------------------------------------------------------

def _fake_vision_resolver(unavailable_provider, fallback_provider, fallback_client):
    def resolve(provider=None, model=None, *, base_url=None, api_key=None,
                async_mode=False, main_runtime=None):
        if provider == unavailable_provider:
            return unavailable_provider, None, None
        return fallback_provider, fallback_client, "fallback-model"
    return resolve


def test_vision_fallback_records_notice_in_route_info(monkeypatch):
    sentinel_client = object()
    monkeypatch.setattr(
        aux, "resolve_vision_provider_client",
        _fake_vision_resolver("openrouter", "anthropic", sentinel_client))

    route_info: dict = {}
    route = aux._resolve_call_client(
        "vision", provider=None, model=None, base_url=None, api_key=None,
        resolved_provider="openrouter", resolved_model=None,
        resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
        main_runtime=None, async_mode=False, route_info=route_info)

    assert route.client is sentinel_client
    assert "openrouter" in route_info.get("fallback_notice", "")
    assert "anthropic" in route_info.get("fallback_notice", "")


def test_vision_no_fallback_leaves_route_info_clean(monkeypatch):
    sentinel_client = object()
    monkeypatch.setattr(
        aux, "resolve_vision_provider_client",
        _fake_vision_resolver("__never__", "anthropic", sentinel_client))

    route_info: dict = {}
    aux._resolve_call_client(
        "vision", provider=None, model=None, base_url=None, api_key=None,
        resolved_provider="anthropic", resolved_model="claude-haiku-4-5-20251001",
        resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
        main_runtime=None, async_mode=False, route_info=route_info)

    assert "fallback_notice" not in route_info


def test_prepare_aux_request_forwards_route_info(monkeypatch, tmp_path):
    """``route_info`` flows from ``async_call_llm``/``call_llm`` down to the resolver."""
    captured = {}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("")

    def fake_resolve_call_client(task, **kw):
        captured.update(kw)
        return aux._ResolvedAuxRoute(object(), "m", "anthropic", "anthropic")

    monkeypatch.setattr(aux, "_resolve_call_client", fake_resolve_call_client)

    route_info: dict = {}
    aux._prepare_aux_request(
        "vision", provider=None, model=None, base_url=None, api_key=None,
        main_runtime={}, messages=[{"role": "user", "content": "hi"}],
        temperature=None, max_tokens=None, tools=None, timeout=None,
        extra_body=None, reasoning_config=None, extra_headers=None,
        api_mode=None, route_info=route_info, async_mode=True)

    assert captured.get("route_info") is route_info


# ---------------------------------------------------------------------------
# 3. The notice reaches the tool result
# ---------------------------------------------------------------------------

def test_with_route_notice_prefixes_analysis():
    debug_call_data: dict = {}
    out = _with_route_notice(
        "the analysis", {"fallback_notice": "Configured vision provider 'openrouter' is unavailable"},
        debug_call_data)
    assert out.startswith("[Configured vision provider 'openrouter' is unavailable]")
    assert out.endswith("the analysis")
    assert "openrouter" in debug_call_data["vision_fallback_notice"]


def test_with_route_notice_passthrough_without_notice():
    debug_call_data: dict = {}
    out = _with_route_notice("the analysis", {}, debug_call_data)
    assert out == "the analysis"
    assert "vision_fallback_notice" not in debug_call_data
