"""OpenRouter's wire must not depend on how credentials reached the resolver.

``resolve_runtime_provider`` has two routes to an OpenRouter runtime:

* the pooled/env route, taken at startup (no explicit credentials), and
* ``_resolve_openrouter_runtime``, taken when the caller passes
  ``explicit_api_key`` / ``explicit_base_url`` — which is what
  ``_ensure_runtime_credentials`` does after a ``/model`` switch.

OpenRouter serves BOTH wires (``/v1/chat/completions`` and the
Anthropic-compatible ``/v1/messages``), so ``model.api_mode`` is a real user
choice there. Both routes must honor it identically, and both must refuse a
mode inherited from a DIFFERENT provider's config block.

Regression guard — the pooled route ignored ``model.api_mode`` entirely, so
the same config produced different wires depending on call shape.
"""

from __future__ import annotations

import pytest

BOTH_WIRES = ["anthropic_messages", "chat_completions"]
FOREIGN_PROVIDERS = ["anthropic", "openai", "copilot", "ollama"]


@pytest.fixture(autouse=True)
def _dummy_credentials(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-dummy")


def _resolve(monkeypatch, cfg, *, explicit):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(rp, "load_config", lambda *a, **k: {"model": dict(cfg)})
    kwargs = {"requested": "openrouter", "target_model": "moonshotai/kimi-k3"}
    if explicit:
        kwargs["explicit_base_url"] = "https://openrouter.ai/api/v1"
        kwargs["explicit_api_key"] = "sk-or-test-dummy"
    return rp.resolve_runtime_provider(**kwargs)


@pytest.mark.parametrize("mode", BOTH_WIRES)
def test_own_api_mode_honored_on_both_routes(monkeypatch, mode):
    """A config that genuinely records OpenRouter is honored either way."""
    cfg = {"default": "moonshotai/kimi-k3", "provider": "openrouter", "api_mode": mode}

    pooled = _resolve(monkeypatch, cfg, explicit=False)["api_mode"]
    explicit = _resolve(monkeypatch, cfg, explicit=True)["api_mode"]

    assert pooled == explicit == mode, (
        f"api_mode={mode!r} must survive both routes — "
        f"pooled={pooled!r} explicit={explicit!r}"
    )


@pytest.mark.parametrize("mode", BOTH_WIRES)
@pytest.mark.parametrize("cfg_provider", FOREIGN_PROVIDERS)
def test_foreign_api_mode_rejected_on_both_routes(monkeypatch, mode, cfg_provider):
    """A mode belonging to another provider never reaches OpenRouter."""
    cfg = {"default": "claude-sonnet-5", "provider": cfg_provider, "api_mode": mode}

    pooled = _resolve(monkeypatch, cfg, explicit=False)["api_mode"]
    explicit = _resolve(monkeypatch, cfg, explicit=True)["api_mode"]

    assert pooled == explicit == "chat_completions", (
        f"a {cfg_provider!r} config's api_mode={mode!r} must not select "
        f"OpenRouter's wire — pooled={pooled!r} explicit={explicit!r}"
    )


def test_routes_agree_when_no_api_mode_recorded(monkeypatch):
    cfg = {"default": "moonshotai/kimi-k3", "provider": "openrouter"}

    pooled = _resolve(monkeypatch, cfg, explicit=False)["api_mode"]
    explicit = _resolve(monkeypatch, cfg, explicit=True)["api_mode"]

    assert pooled == explicit == "chat_completions"
