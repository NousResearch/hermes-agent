"""Regression guard: bare ``custom`` must reuse the live main runtime endpoint.

Bug (reproduced 2026-09-25): an OpenAI-compatible endpoint configured under
``providers.<name>`` (here ``cpa``) resolves to the *anonymous* ``custom``
bucket at runtime — ``resolve_runtime_provider()`` returns
``provider='custom'`` carrying the base_url. Two aux call sites passed that
bare ``custom`` to ``resolve_provider_client()`` **without** the endpoint:

  * ``_try_main_agent_model_fallback`` — last-resort aux fallback
  * ``get_available_vision_backends`` — vision capability probe

With no endpoint and no ``OPENAI_BASE_URL``, the resolver fell through to
model-name inference and left the box for the vendor's public API:

    Auxiliary vision: connection error on cpa — falling back to main agent
      model main-agent(custom) (deepseek-v4.1-flash)
    400 The supported API model names are deepseek-flash, deepseek-v4-pro,
        but you passed deepseek-v4.1-flash

(That 400 is DeepSeek's own error text — the request was sent to
api.deepseek.com instead of the local CPA endpoint.)

Fix: ``_resolve_custom_branch`` reuses the context-local live runtime
(``_normalize_main_runtime(None)``) when the caller passed no explicit
endpoint, so every bare-``custom`` caller inherits this session's real
endpoint instead of re-deriving one from the model name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent import auxiliary_client as ac

CPA_BASE = "http://192.168.2.3:8318/v1"
MODEL = "deepseek-v4.1-flash"


@pytest.fixture
def cpa_runtime(monkeypatch):
    """Live runtime exactly as the gateway publishes it for this install."""
    # Keep the env clean so the OLD code path cannot accidentally succeed via OPENAI_BASE_URL:
    # the bug is precisely that it inferred an endpoint from the model name instead.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    token = ac.set_runtime_main(
        "custom", MODEL, base_url=CPA_BASE, api_key="sk-test-key", api_mode="responses",
    )
    try:
        yield
    finally:
        ac.reset_runtime_main(token)


@pytest.fixture
def fake_client_factory(monkeypatch):
    """Build the client without touching the network or the operator's HOME.

    ``_create_openai_client`` pulls in ``agent.process_bootstrap`` (keepalive transport), which
    activates the payload venv and reads the real ``$HERMES_HOME/manifest.json`` — both refused by
    ``tests/home_io_guard``. The route under test only has to hand the right ``base_url`` over.
    """

    class _Client:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key
            self.base_url = base_url

    def _make(api_key=None, base_url=None, **_kwargs):
        return _Client(api_key, base_url)

    monkeypatch.setattr(ac, "_create_openai_client", _make)
    return _Client


def _base_of(client) -> str:
    return str(getattr(client, "base_url", "") or "")


@pytest.mark.parametrize("caller", ["fallback", "vision_probe"])
def test_bare_custom_reuses_live_endpoint(cpa_runtime, fake_client_factory, monkeypatch, caller):
    """Both bare-``custom`` call sites must stay on the configured endpoint."""
    if caller == "fallback":
        # Pin the *main* agent identity instead of reading the operator's config.yaml: this test
        # only cares that the fallback route hands the live endpoint to the resolver. (Upstream
        # later added a vision-capability gate; leaving it real would let a text-only main model
        # decline before the resolver is ever reached.)
        monkeypatch.setattr(ac, "_read_main_provider", lambda: "custom")
        monkeypatch.setattr(ac, "_read_main_model", lambda: MODEL)
        monkeypatch.setattr(ac, "_main_model_supports_vision", lambda *_a, **_k: True)
        client, _model, label = ac._try_main_agent_model_fallback(
            "cpa", task="vision", reason="connection error")
        assert label == "main-agent(custom)"
    else:
        client, _model = ac.resolve_provider_client("custom", MODEL)

    assert client is not None, (
        f"{caller}: bare 'custom' resolved to no client; the live runtime endpoint "
        "should have been reused"
    )
    resolved = _base_of(client)
    assert "deepseek.com" not in resolved, (
        f"{caller}: bare 'custom' leaked to DeepSeek's public endpoint ({resolved!r}). "
        "Without the live base_url the router infers a provider from the model name, "
        "which is how a CPA-hosted deepseek-v4.1-flash ended up 400ing against the "
        "real api.deepseek.com."
    )
    assert "8318" in resolved, f"{caller}: expected the CPA endpoint, got {resolved!r}"
