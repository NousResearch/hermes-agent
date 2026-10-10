"""Catalog context window, vision and price for model-picker rows.

``_apply_capabilities`` resolves each row model's models.dev entry: the provider's own models.dev id,
then the vendor prefix of a slash-style id, then a cross-vendor scan by model id for resellers that
models.dev does not list at all. It must never guess ``"openrouter"`` for a slash-less id (which made
every direct-provider row resolve to nothing), never invent a value for a model the catalog does not
know, and never replace a price the provider itself reported.
"""
from contextlib import ExitStack
from unittest.mock import patch

from agent import models_dev
from hermes_cli import inventory
from hermes_cli.models_pricing import _format_price_per_mtok


def _entry(ctx, inp=None, out=None, modalities=("text",), tools=False, max_out=None):
    e = {"limit": {"context": ctx}, "modalities": {"input": list(modalities), "output": ["text"]},
         "tool_call": tools}
    if max_out:
        e["limit"]["output"] = max_out
    if inp is not None or out is not None:
        e["cost"] = {"input": inp, "output": out}
    return e


# glm-5.3-flash: three vendors quote the $0.15/$0.50 list, one discounter undercuts it.
# mimo-v2.5 exists on its own vendor and, at a wrong price, on an aggregator.
FAKE_REGISTRY = {
    "xiaomi": {"models": {"mimo-v2.5": _entry(1048576, 0.14, 0.28, modalities=("text", "image"),
                                              tools=True, max_out=131072)}},
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


def _apply(row, *, catalog_pricing=True):
    """Run the capabilities pass with the catalog faked and the unrelated per-model probes inert."""
    with ExitStack() as stack:
        stack.enter_context(_patched())
        stack.enter_context(patch("hermes_cli.models.resolve_fast_mode_overrides", return_value=None))
        stack.enter_context(patch.object(models_dev, "get_model_capabilities", return_value=None))
        stack.enter_context(patch.object(inventory, "_reasoning_catalog_reader", return_value=None))
        inventory._apply_capabilities([row], catalog_pricing=catalog_pricing)
    return row


def test_bare_model_id_resolves_to_its_own_provider_not_openrouter():
    with _patched():
        info = inventory._catalog_model_info("xiaomi", "mimo-v2.5")
    assert info is not None
    # openrouter also lists mimo-v2.5 at a different window and price: the own vendor must win.
    assert (info.context_window, info.cost_input, info.cost_output) == (1048576, 0.14, 0.28)


def test_direct_provider_row_gets_context_capabilities_and_a_catalog_price():
    row = _apply({"slug": "xiaomi", "models": ["mimo-v2.5"]})
    caps = row["capabilities"]["mimo-v2.5"]
    assert (caps["context_window"], caps["max_output"]) == (1048576, 131072)
    assert caps["supports_vision"] is True
    assert caps["supports_tools"] is True
    priced = row["pricing"]["mimo-v2.5"]
    assert priced["source"] == "catalog"
    # Rendered by the same formatter as a live price, so both sources look alike in the picker.
    assert (priced["input"], priced["output"]) == (
        _format_price_per_mtok(str(0.14 / 1_000_000)), _format_price_per_mtok(str(0.28 / 1_000_000)))


def test_a_payload_built_without_pricing_gets_no_catalog_price():
    """The payload's ``pricing`` flag owns pricing; the capabilities pass must not add it on its own."""
    row = _apply({"slug": "xiaomi", "models": ["mimo-v2.5"]}, catalog_pricing=False)
    assert "pricing" not in row
    assert row["capabilities"]["mimo-v2.5"]["context_window"] == 1048576


def test_a_capability_the_catalog_denies_reads_false_not_absent():
    caps = _apply({"slug": "deepseek", "models": ["deepseek-v4-pro"]})["capabilities"]["deepseek-v4-pro"]
    assert (caps["supports_vision"], caps["supports_tools"]) == (False, False)
    # The catalog lists no output limit for this entry: unknown stays absent, never 0.
    assert "max_output" not in caps


def test_reseller_absent_from_models_dev_resolves_cross_vendor_at_the_list_price():
    """A reseller slug is not a models.dev vendor; only the model id can resolve it.

    Three vendors quote $0.15/$0.50 and one discounter $0.075/$0.25 — the modal price is the
    vendor's list price, so the discounter must not be shown as if it were.
    """
    with _patched():
        info = inventory._catalog_model_info("some-reseller", "glm-5.3-flash")
    assert info is not None
    assert (info.cost_input, info.cost_output) == (0.15, 0.5)
    assert info.provider_id != "302ai"


def test_unknown_model_stays_blank():
    """Failure is silence: no context, no vision verdict, no price, never a synthesized default."""
    with _patched():
        assert inventory._catalog_model_info("custom:lab", "no-such-model-x") is None

    row = _apply({"slug": "custom:lab", "models": ["no-such-model-x"]})
    caps = row["capabilities"]["no-such-model-x"]
    assert caps["context_window"] == 0
    assert not {"supports_vision", "supports_tools", "max_output"} & caps.keys()
    assert row["pricing"] == {}


def _payload_row(*, pricing):
    """One provider row from the real ``build_models_payload``, with no live price source."""
    rows = [{"slug": "xiaomi", "name": "Xiaomi", "models": ["mimo-v2.5"], "total_models": 1,
             "is_current": False, "is_user_defined": False, "source": "built-in"}]
    ctx = inventory.ConfigContext(current_provider="xiaomi", current_model="mimo-v2.5", current_base_url="",
                                  user_providers={}, custom_providers=[])
    with ExitStack() as stack:
        stack.enter_context(_patched())
        stack.enter_context(patch("hermes_cli.model_switch.list_authenticated_providers", return_value=rows))
        stack.enter_context(patch.object(inventory, "_local_runtime_row", return_value=None))
        stack.enter_context(patch.object(inventory, "_moa_provider_row", return_value=None))
        stack.enter_context(patch.object(inventory, "_apply_pricing", lambda *a, **k: None))
        stack.enter_context(patch("hermes_cli.models.resolve_fast_mode_overrides", return_value=None))
        stack.enter_context(patch.object(inventory, "_reasoning_catalog_reader", return_value=None))
        payload = inventory.build_models_payload(ctx, pricing=pricing, capabilities=True)
    return next(r for r in payload["providers"] if r["slug"] == "xiaomi")


def test_payload_pricing_flag_decides_whether_the_catalog_price_appears():
    assert _payload_row(pricing=True)["pricing"]["mimo-v2.5"]["source"] == "catalog"
    unpriced = _payload_row(pricing=False)
    assert "mimo-v2.5" not in (unpriced.get("pricing") or {})
    assert unpriced["capabilities"]["mimo-v2.5"]["context_window"] == 1048576


def test_live_provider_price_is_never_overwritten_by_the_catalog():
    live = {"input": "$0.01", "output": "$0.02", "free": False}
    row = _apply({"slug": "xiaomi", "models": ["mimo-v2.5"], "pricing": {"mimo-v2.5": dict(live)}})
    assert row["pricing"]["mimo-v2.5"] == live
    # The context window still fills: only the price was already known.
    assert row["capabilities"]["mimo-v2.5"]["context_window"] == 1048576
