"""A model the capability catalog reports as text-only must never serve the vision route.

Reproduction of the upstream defect: `resolve_vision_provider_client` resolved any model a
config or fallback entry named, so a text-only model was handed an image payload and answered
with a hard 400 ("This API supports text input only") — a doomed request per image turn. Both
runtime fallback-chain walkers had the same gap: a text-only candidate kept its place in the
vision chain, unlike the context-window screen that already skips unusable candidates.

The contract after the fix:
- `resolve_vision_provider_client` refuses a resolved model the catalog reports as
  `supports_vision=False`: provider is returned, client/model are `None`, so the caller falls
  through instead of spending the request.
- Both fallback-chain walkers skip a text-only candidate when `task == "vision"`; chat and
  compression chains are untouched (a text-only model is a perfectly good chat fallback).
- Only a definitive `False` blocks. Unknown models (`None` verdict) and lookup failures keep
  their attempt — the screen may never invent a verdict.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import agent.auxiliary_client as aux
from agent import image_routing
import hermes_cli.config as hermes_config

PROVIDER = "zai"
FAILED_PROVIDER = "anthropic"  # a different provider: same-provider entries are skipped by design
TEXT_ONLY_MODEL = "text-only-model"
VISION_MODEL = "vision-capable-model"


@pytest.fixture
def capability(monkeypatch):
    """Capability lookup seam: only TEXT_ONLY_MODEL has a definitive verdict."""
    verdicts: dict[str, bool | None] = {
        TEXT_ONLY_MODEL: False,
        VISION_MODEL: True,
    }

    def _lookup(provider, model, cfg=None, *, requested_provider=""):
        return verdicts.get(str(model))

    monkeypatch.setattr(image_routing, "_lookup_supports_vision", _lookup)
    monkeypatch.setattr(hermes_config, "load_config_readonly", dict)
    # Sibling tests may have marked backends unhealthy; chain walks must judge fresh state.
    monkeypatch.setattr(aux, "_aux_unhealthy_until", {})
    monkeypatch.setattr(aux, "_aux_unhealthy_logged_at", {})
    monkeypatch.setattr(aux, "_aux_unhealthy_reason", {})
    return verdicts


def _patch_resolution(monkeypatch):
    """Stub entry resolution so a walk exercises ONLY the screening logic."""
    clients: dict[str, MagicMock] = {}

    def _resolve(entry):
        model = str(entry.get("model") or "")
        clients.setdefault(model, MagicMock(name=f"client-{model}"))
        return clients[model], model or None

    monkeypatch.setattr(aux, "_resolve_fallback_entry", _resolve)
    return clients


# ---------------------------------------------------------------------------
# fallback-chain walkers
# ---------------------------------------------------------------------------


def test_configured_chain_skips_a_text_only_candidate(capability, monkeypatch):
    _patch_resolution(monkeypatch)
    chain = [
        {"provider": PROVIDER, "model": TEXT_ONLY_MODEL},
        {"provider": PROVIDER, "model": VISION_MODEL},
    ]
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {"fallback_chain": chain})

    _client, model, label = aux._try_configured_fallback_chain("vision", FAILED_PROVIDER, reason="error")

    assert model == VISION_MODEL
    assert label == f"fallback_chain[1]({PROVIDER})"


def test_configured_chain_of_only_text_only_candidates_yields_nothing(capability, monkeypatch):
    _patch_resolution(monkeypatch)
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {"fallback_chain": [
        {"provider": PROVIDER, "model": TEXT_ONLY_MODEL},
    ]})

    assert aux._try_configured_fallback_chain("vision", FAILED_PROVIDER, reason="error") == (None, None, "")


def test_main_chain_skips_a_text_only_candidate(capability, monkeypatch):
    _patch_resolution(monkeypatch)
    from hermes_cli import fallback_config

    monkeypatch.setattr(fallback_config, "get_fallback_chain", lambda cfg: [
        {"provider": PROVIDER, "model": TEXT_ONLY_MODEL},
        {"provider": PROVIDER, "model": VISION_MODEL},
    ])

    _client, model, provider_label = aux._try_main_fallback_chain("vision", FAILED_PROVIDER)

    assert model == VISION_MODEL
    assert provider_label == PROVIDER  # third element is the provider, not the chain label


def test_non_vision_task_keeps_a_text_only_candidate(capability, monkeypatch):
    """Chat-style tasks are untouched: a text-only model is a valid fallback there."""
    clients = _patch_resolution(monkeypatch)
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {"fallback_chain": [
        {"provider": PROVIDER, "model": TEXT_ONLY_MODEL},
        {"provider": PROVIDER, "model": VISION_MODEL},
    ]})

    client, model, _label = aux._try_configured_fallback_chain("compression", FAILED_PROVIDER, reason="error")

    assert model == TEXT_ONLY_MODEL
    assert client is clients[TEXT_ONLY_MODEL]


def test_unknown_capability_still_gets_its_attempt(capability, monkeypatch):
    """Only a definitive False blocks; an unknown model keeps its place in the chain."""
    _patch_resolution(monkeypatch)
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {"fallback_chain": [
        {"provider": PROVIDER, "model": "model-with-no-catalog-verdict"},
    ]})

    _client, model, _label = aux._try_configured_fallback_chain("vision", FAILED_PROVIDER, reason="error")

    assert model == "model-with-no-catalog-verdict"


def test_entry_without_a_model_is_not_screened(capability, monkeypatch):
    """Provider-default entries carry no model to screen; the walk proceeds."""
    clients = _patch_resolution(monkeypatch)
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {"fallback_chain": [
        {"provider": PROVIDER},
    ]})

    client, _model, label = aux._try_configured_fallback_chain("vision", FAILED_PROVIDER, reason="error")

    assert client is clients[""]
    assert label == f"fallback_chain[0]({PROVIDER})"


# ---------------------------------------------------------------------------
# resolve_vision_provider_client
# ---------------------------------------------------------------------------


def test_resolve_refuses_a_text_only_model(capability, monkeypatch):
    monkeypatch.setattr(
        aux, "_resolve_vision_backend_unchecked",
        lambda **kw: (PROVIDER, MagicMock(), TEXT_ONLY_MODEL))

    provider, client, model = aux.resolve_vision_provider_client(
        provider=PROVIDER, model=TEXT_ONLY_MODEL)

    assert provider == PROVIDER
    assert client is None, "a text-only model must never resolve to a vision client"
    assert model is None  # the model rides with the client


def test_resolve_passes_a_vision_capable_model(capability, monkeypatch):
    sentinel = MagicMock()
    monkeypatch.setattr(
        aux, "_resolve_vision_backend_unchecked",
        lambda **kw: (PROVIDER, sentinel, VISION_MODEL))

    _provider, client, model = aux.resolve_vision_provider_client(
        provider=PROVIDER, model=VISION_MODEL)

    assert client is sentinel
    assert model == VISION_MODEL


def test_resolve_passes_a_model_with_no_catalog_verdict(capability, monkeypatch):
    sentinel = MagicMock()
    monkeypatch.setattr(
        aux, "_resolve_vision_backend_unchecked",
        lambda **kw: (PROVIDER, sentinel, "model-with-no-catalog-verdict"))

    _provider, client, _model = aux.resolve_vision_provider_client(
        provider=PROVIDER, model="model-with-no-catalog-verdict")

    assert client is sentinel


def test_resolve_keeps_an_already_unavailable_backend_unavailable(capability, monkeypatch):
    monkeypatch.setattr(
        aux, "_resolve_vision_backend_unchecked",
        lambda **kw: (PROVIDER, None, TEXT_ONLY_MODEL))

    _provider, client, _model = aux.resolve_vision_provider_client(
        provider=PROVIDER, model=TEXT_ONLY_MODEL)

    assert client is None
