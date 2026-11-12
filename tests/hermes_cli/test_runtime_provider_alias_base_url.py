"""``model.provider`` alias spellings must resolve exactly like their canonical id.

``lm-studio`` / ``lm_studio`` are documented aliases of ``lmstudio`` (``hermes_cli/providers.py``
``_ALIAS_GROUPS``, mirrored by ``hermes_cli/auth.py::_PROVIDER_ALIASES``), but
``runtime_provider._same_registered_provider`` compared the raw spellings against
``PROVIDER_REGISTRY``, which is keyed by canonical id only. An alias therefore never matched its
own registry row, so ``_config_base_url_for_provider`` dropped the saved ``model.base_url`` and the
provider's default endpoint won — for LM Studio, ``http://127.0.0.1:1234/v1`` instead of the
configured LAN host (#154243).
"""

from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp


def _resolve_with_saved_base_url(monkeypatch, spelling: str, base_url: str) -> dict:
    """Resolve ``lmstudio`` against a config whose ``model.provider`` carries *spelling*."""
    monkeypatch.delenv("LM_API_KEY", raising=False)
    monkeypatch.delenv("LM_BASE_URL", raising=False)
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "lmstudio")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": spelling, "base_url": base_url, "default": "qwen/qwen3-coder-30b"},
    )
    monkeypatch.setattr(
        rp, "load_pool", lambda provider: type("Pool", (), {"has_credentials": lambda self: False})()
    )
    return rp.resolve_runtime_provider(requested="lmstudio")


@pytest.mark.parametrize("spelling", ["lmstudio", "lm-studio", "lm_studio"])
def test_alias_spelling_honors_saved_base_url(monkeypatch, spelling):
    """Every documented spelling keeps the configured endpoint (a remote LM Studio host)."""
    resolved = _resolve_with_saved_base_url(monkeypatch, spelling, "http://192.168.1.50:1234/v1")

    assert resolved["provider"] == "lmstudio"
    assert resolved["base_url"] == "http://192.168.1.50:1234/v1"


@pytest.mark.parametrize(
    ("provider", "configured"),
    [
        ("lmstudio", "lm-studio"),
        ("lmstudio", "lm_studio"),
        ("lmstudio", "lmstudio"),
        ("xai", "grok"),
        ("anthropic", "claude"),
        ("actual", "actual-computer"),
    ],
)
def test_alias_ids_share_the_registry_row(provider, configured):
    """A spelling from the alias table names the same provider as its canonical id."""
    assert rp._same_registered_provider(provider, configured) is True


@pytest.mark.parametrize(
    ("provider", "configured"),
    [
        ("lmstudio", "openrouter"),
        ("openrouter", "lmstudio"),
        ("custom:one", "custom:two"),
        # The generic ``custom`` row is one auth ID for several endpoints (ollama, vllm, …):
        # a stored base_url names one of them and must not answer for another.
        ("custom", "ollama"),
        ("custom", "vllm"),
        ("custom", "lmstudio"),
        # ``providers.ALIASES`` maps the legacy catalog spelling ``openai`` onto ``openrouter``;
        # resolution does not, and a saved api.openai.com base_url must not reach OpenRouter.
        ("openrouter", "openai"),
    ],
)
def test_unrelated_routes_stay_distinct(provider, configured):
    """Route identity is still a real comparison — aliasing must not merge other providers."""
    assert rp._same_registered_provider(provider, configured) is False
