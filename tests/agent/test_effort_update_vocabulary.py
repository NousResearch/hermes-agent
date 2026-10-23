"""Regression coverage for declared effort vocabularies in #139537's marker path."""
import copy
import sys

import pytest

from agent.effort_updates import make_effort_update_message
from agent.transports.codex import ResponsesApiTransport


@pytest.fixture
def declared_catalog(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("router")
    module = sys.modules[type(profile).__module__]
    catalog = {}
    monkeypatch.setattr(module, "_efforts_cache", catalog)
    monkeypatch.setattr(module, "_disk_checked", True)
    return catalog


def _request(previous, effort, **route):
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        make_effort_update_message(effort, previous),
        {"role": "user", "content": "next"},
    ]
    original = copy.deepcopy(messages)
    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        messages=messages, reasoning_config={"effort": effort},
        capabilities={"reasoning_effort_updates": True}, **route,
    )
    assert messages == original
    return transport.preflight_kwargs(kwargs)


@pytest.mark.parametrize("previous,effort", [("low", "max"), ("max", "low"), ("low", "minimal")])
def test_marker_and_baseline_use_declared_catalog_ladder(declared_catalog, previous, effort):
    declared_catalog["effort-fixture"] = ["minimal", "low", "medium", "high", "xhigh", "max"]
    kwargs = _request(
        previous, effort, model="effort-fixture", provider="custom:relay",
        base_url="https://api.router.com/v1",
    )
    assert kwargs["reasoning"]["effort"] == previous
    assert [item for item in kwargs["input"] if item.get("type") == "configuration_update"] == [
        {"type": "configuration_update", "reasoning": {"effort": effort}},
    ]


def test_empty_declared_ladder_strips_markers_and_reasoning(declared_catalog):
    declared_catalog["effort-fixture"] = []
    kwargs = _request(
        "low", "high", model="effort-fixture", provider="router",
        base_url="https://api.router.com/v1",
    )
    assert "reasoning" not in kwargs
    assert not any(item.get("type") == "configuration_update" for item in kwargs["input"])


@pytest.mark.parametrize("effort", ["ultra", "MINIMAL", " minimal", "", None, True, 1, [], {}])
def test_preflight_rejects_malformed_marker_effort(effort):
    transport = ResponsesApiTransport()
    with pytest.raises(ValueError, match="supported reasoning.effort"):
        transport.preflight_kwargs({
            "model": "effort-fixture",
            "instructions": "Test request validation.",
            "input": [{"type": "configuration_update", "reasoning": {"effort": effort}}],
        })


@pytest.mark.parametrize("codex_backend", [False, True])
def test_catalog_cannot_lower_official_astra_floor(declared_catalog, codex_backend):
    declared_catalog["gpt-6-astra"] = ["none", "minimal", "low", "high", "max"]
    kwargs = _request(
        "none", "high", model="gpt-6-astra", provider="router",
        base_url=("https://chatgpt.com/backend-api/codex" if codex_backend
                  else "https://api.openai.com/v1"),
        is_codex_backend=codex_backend,
    )
    assert kwargs["reasoning"]["effort"] == "low"
    assert [item for item in kwargs["input"] if item.get("type") == "configuration_update"] == [
        {"type": "configuration_update", "reasoning": {"effort": "high"}},
    ]
