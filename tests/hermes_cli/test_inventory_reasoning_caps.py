"""Tests for the reasoning detail inventory._apply_capabilities puts on the

picker payload. The desktop model picker hides its Thinking toggle from this,
so a route that can't disable reasoning must be describable here — otherwise
the UI offers an off switch whose setting the upstream rejects.

The catalog's `supported_efforts` is intentionally absent from the payload:
the Portal honors levels a route doesn't advertise, so publishing it would
invite a picker filter that hides working levels.
"""

import hermes_cli.inventory as inv
import hermes_cli.models as models_mod
from hermes_cli import models_reasoning_caps


def _patch_catalog(monkeypatch, caps_by_model, *, provider="nous"):
    """Point the Nous/OpenRouter catalog readers at a fixed capability map."""
    monkeypatch.setattr(models_mod, "model_supports_fast_mode", lambda model: False)
    monkeypatch.setattr(models_reasoning_caps, "warm_nous_reasoning_caps_async", lambda: None)
    monkeypatch.setattr(models_reasoning_caps, "warm_openrouter_reasoning_caps_async", lambda: None)
    monkeypatch.setattr(
        models_reasoning_caps,
        f"{provider}_model_reasoning_capabilities",
        lambda model, **kw: caps_by_model.get(model),
    )


def test_optional_reasoning_route_can_disable(monkeypatch):
    """A route that accepts a disable says so."""
    _patch_catalog(monkeypatch, {
        "deepseek/deepseek-v4-pro": {
            "supports_reasoning": True,
            "supported_efforts": ["xhigh", "high"],
            "mandatory": False,
        },
    })
    rows = [{"slug": "nous", "models": ["deepseek/deepseek-v4-pro"]}]
    inv._apply_capabilities(rows)

    assert rows[0]["capabilities"]["deepseek/deepseek-v4-pro"]["can_disable_reasoning"] is True


def test_advertised_efforts_never_reach_the_picker(monkeypatch):
    """The catalog's level list stays off the wire even when it is published.

    It under-reports what the Portal serves, so forwarding it would let the
    picker hide levels that work. Only the disable verdict crosses.
    """
    _patch_catalog(monkeypatch, {
        "deepseek/deepseek-v4-pro": {
            "supports_reasoning": True,
            "supported_efforts": ["xhigh", "high"],
            "mandatory": False,
        },
    })
    rows = [{"slug": "nous", "models": ["deepseek/deepseek-v4-pro"]}]
    inv._apply_capabilities(rows)

    assert "supported_efforts" not in rows[0]["capabilities"]["deepseek/deepseek-v4-pro"]


def test_non_reasoning_route_offers_no_reasoning_controls(monkeypatch):
    """The serving provider's catalog outranks the models.dev inference.

    models.dev defaults an uncatalogued model to "has reasoning"; when the
    aggregator actually serving the route says it takes no reasoning
    parameter, that is the definitive answer and the picker shows no
    reasoning controls at all — so there is no disable to describe either.
    """
    _patch_catalog(monkeypatch, {
        "moonshotai/kimi-k3-instruct": {"supports_reasoning": False},
    })
    rows = [{"slug": "nous", "models": ["moonshotai/kimi-k3-instruct"]}]
    inv._apply_capabilities(rows)

    caps = rows[0]["capabilities"]["moonshotai/kimi-k3-instruct"]
    assert caps["reasoning"] is False
    assert "can_disable_reasoning" not in caps


def test_reasoning_mandatory_route_cannot_disable(monkeypatch):
    """`mandatory` inverts into the flag the Thinking toggle keys off.

    The Portal answers a disable on these routes with HTTP 400, so offering
    the toggle would be offering a control that cannot work.
    """
    _patch_catalog(monkeypatch, {
        "z-ai/glm-5.3": {
            "supports_reasoning": True,
            "supported_efforts": ["max", "high", "low"],
            "mandatory": True,
        },
    })
    rows = [{"slug": "nous", "models": ["z-ai/glm-5.3"]}]
    inv._apply_capabilities(rows)

    assert rows[0]["capabilities"]["z-ai/glm-5.3"]["can_disable_reasoning"] is False


def test_unlisted_model_states_no_restriction(monkeypatch):
    """A model the catalog doesn't cover omits both keys rather than guessing.

    The UI reads "absent" as no known restriction and offers the full scale,
    which is the right failure: over-offering beats hiding levels a model
    actually accepts.
    """
    _patch_catalog(monkeypatch, {})
    rows = [{"slug": "nous", "models": ["mystery/model"]}]
    inv._apply_capabilities(rows)

    caps = rows[0]["capabilities"]["mystery/model"]
    assert "supported_efforts" not in caps
    assert "can_disable_reasoning" not in caps
    assert caps["reasoning"] is True


def test_providers_without_a_reasoning_catalog_are_untouched(monkeypatch):
    """Only aggregators that publish per-model detail gain the extra keys."""
    _patch_catalog(monkeypatch, {
        "gpt-5.6": {"supports_reasoning": True, "supported_efforts": ["high"], "mandatory": True},
    })
    rows = [{"slug": "openai-api", "models": ["gpt-5.6"]}]
    inv._apply_capabilities(rows)

    caps = rows[0]["capabilities"]["gpt-5.6"]
    assert "supported_efforts" not in caps
    assert "can_disable_reasoning" not in caps


def test_openrouter_uses_its_own_catalog(monkeypatch):
    """The reader is chosen per provider row, not hardcoded to one aggregator."""
    _patch_catalog(
        monkeypatch,
        {"x-ai/grok-5": {"supports_reasoning": True, "mandatory": True}},
        provider="openrouter",
    )
    rows = [{"slug": "openrouter", "models": ["x-ai/grok-5"]}]
    inv._apply_capabilities(rows)

    assert rows[0]["capabilities"]["x-ai/grok-5"]["can_disable_reasoning"] is False


def test_catalog_failure_never_breaks_the_picker(monkeypatch):
    """A raising catalog reader degrades to "unknown", not to a broken payload."""
    monkeypatch.setattr(models_mod, "model_supports_fast_mode", lambda model: False)
    monkeypatch.setattr(models_reasoning_caps, "warm_nous_reasoning_caps_async", lambda: None)

    def _boom(model, **kw):
        raise RuntimeError("catalog exploded")

    monkeypatch.setattr(models_reasoning_caps, "nous_model_reasoning_capabilities", _boom)
    rows = [{"slug": "nous", "models": ["deepseek/deepseek-v4-pro"]}]
    inv._apply_capabilities(rows)

    caps = rows[0]["capabilities"]["deepseek/deepseek-v4-pro"]
    assert "supported_efforts" not in caps
    assert caps["reasoning"] is True


def test_output_modalities_are_forwarded_for_picker_filters(monkeypatch):
    """The picker receives normalized output modalities without losing unknown models."""
    _patch_catalog(monkeypatch, {})

    class Info:
        def __init__(self, modalities):
            self.output_modalities = modalities

    monkeypatch.setattr(
        "agent.models_dev.get_model_info",
        lambda provider, model, config=None: Info(("IMAGE", "text")) if model == "vision" else None,
    )
    rows = [{"slug": "openrouter", "models": ["vision", "unknown"]}]
    inv._apply_capabilities(rows)

    assert rows[0]["output_modalities"] == {"vision": ["image", "text"]}


def test_output_modalities_fall_back_to_openrouter_namespace(monkeypatch):
    """Aggregator rows carry vendor-prefixed ids their own namespace lacks; the read
    falls back to openrouter like ``_apply_featured``, instead of filing the model
    under the picker's Unknown bucket."""
    _patch_catalog(monkeypatch, {})

    calls = []

    class Info:
        output_modalities = ("text",)

    def fake_get_model_info(provider, model, config=None):
        calls.append((provider, model))
        if provider == "nous" and model == "openai/gpt-5.6":
            return None  # warm cache: models.dev doesn't namespace this id under nous
        if provider == "openrouter" and model == "openai/gpt-5.6":
            return Info()
        return None

    monkeypatch.setattr("agent.models_dev.get_model_info", fake_get_model_info)
    rows = [{"slug": "nous", "models": ["openai/gpt-5.6"]}]
    inv._apply_capabilities(rows)

    assert rows[0]["output_modalities"] == {"openai/gpt-5.6": ["text"]}
    assert ("nous", "openai/gpt-5.6") in calls
    assert ("openrouter", "openai/gpt-5.6") in calls
