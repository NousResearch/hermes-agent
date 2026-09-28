"""MPX-001-P1a — catalog resolution for picker context + price.

Before this, ``_apply_capabilities`` looked a model up on ``"openrouter"`` whenever its id had no
``/``, so every direct-provider row (xiaomi, deepseek, zai) resolved to nothing and rendered blank.
These tests pin the replacement: the provider's own models.dev id, then a cross-vendor scan by model
id for resellers that models.dev does not list at all.
"""
from unittest.mock import patch

import agent.models_dev as models_dev
import hermes_cli.inventory as inventory


def _entry(ctx, inp=None, out=None):
    e = {"limit": {"context": ctx}}
    if inp is not None or out is not None:
        e["cost"] = {"input": inp, "output": out}
    return e


# glm-5.3-flash: three vendors quote the $0.15/$0.50 list, one discounter undercuts it.
# mimo-v2.5 exists on its own vendor and, at a wrong price, on an aggregator.
FAKE_REGISTRY = {
    "xiaomi": {"models": {"mimo-v2.5": _entry(1048576, 0.14, 0.28)}},
    "openrouter": {"models": {"mimo-v2.5": _entry(131072, 9.99, 9.99),
                              "glm-5.3-flash": _entry(1000000, 0.15, 0.5)}},
    "zai": {"models": {"glm-5.3-flash": _entry(1000000, 0.15, 0.5)}},
    "ollama-cloud": {"models": {"glm-5.3-flash": _entry(1000000, 0.15, 0.5)}},
    "302ai": {"models": {"glm-5.3-flash": _entry(1000000, 0.075, 0.25)}},
    "deepseek": {"models": {"deepseek-v4-pro": _entry(1000000, 0.435, 0.87)}},
}


def _patched():
    inventory._XVENDOR_INDEX_CACHE.clear()
    return patch.object(models_dev, "fetch_models_dev", lambda *a, **k: FAKE_REGISTRY)


def test_bare_model_id_resolves_to_its_own_provider_not_openrouter():
    """The regression this phase exists for: a slash-less id must not be asked of openrouter."""
    with _patched():
        info = inventory._catalog_model_info("xiaomi", "mimo-v2.5")
    assert info is not None
    assert info.context_window == 1048576
    assert (info.cost_input, info.cost_output) == (0.14, 0.28)
    # openrouter also lists mimo-v2.5, at a different window and price — proving the old guess is gone.
    assert info.context_window != 131072


def test_direct_provider_row_gets_context_and_price():
    row = {"slug": "deepseek", "models": ["deepseek-v4-pro"]}
    with _patched(), \
            patch("hermes_cli.models.model_supports_fast_mode", return_value=False), \
            patch.object(models_dev, "get_model_capabilities", return_value=None), \
            patch.object(inventory, "_reasoning_catalog_reader", return_value=None):
        inventory._apply_capabilities([row])
    assert row["capabilities"]["deepseek-v4-pro"]["context_window"] == 1000000
    priced = row["pricing"]["deepseek-v4-pro"]
    assert (priced["input"], priced["output"]) == ("0.435", "0.87")
    assert priced["source"] == "catalog"


def test_reseller_absent_from_models_dev_resolves_cross_vendor_at_the_list_price():
    """`apikey-fan-glm` is not a models.dev vendor; only the model id can resolve it.

    Three vendors quote $0.15/$0.50 and one discounter $0.075/$0.25 — the modal price is the
    vendor's list price, so the discounter must not be shown as if it were.
    """
    with _patched():
        info = inventory._catalog_model_info("apikey-fan-glm", "glm-5.3-flash")
    assert info is not None
    assert (info.cost_input, info.cost_output) == (0.15, 0.5)
    assert info.provider_id != "302ai"


def test_unknown_model_stays_blank():
    """Failure is silence: no context, no price, never a synthesized default."""
    with _patched():
        assert inventory._catalog_model_info("custom:lab", "no-such-model-x") is None

    row = {"slug": "custom:lab", "models": ["no-such-model-x"]}
    with _patched(), \
            patch("hermes_cli.models.model_supports_fast_mode", return_value=False), \
            patch.object(models_dev, "get_model_capabilities", return_value=None), \
            patch.object(inventory, "_reasoning_catalog_reader", return_value=None):
        inventory._apply_capabilities([row])
    assert row["capabilities"]["no-such-model-x"]["context_window"] == 0
    assert row["pricing"] == {}


def test_live_provider_price_is_never_overwritten_by_the_catalog():
    live = {"input": "0.01", "output": "0.02", "free": False, "source": "live"}
    row = {"slug": "xiaomi", "models": ["mimo-v2.5"], "pricing": {"mimo-v2.5": dict(live)}}
    with _patched(), \
            patch("hermes_cli.models.model_supports_fast_mode", return_value=False), \
            patch.object(models_dev, "get_model_capabilities", return_value=None), \
            patch.object(inventory, "_reasoning_catalog_reader", return_value=None):
        inventory._apply_capabilities([row])
    assert row["pricing"]["mimo-v2.5"] == live
    # context still fills, since only the price was already known
    assert row["capabilities"]["mimo-v2.5"]["context_window"] == 1048576
