"""Tests for the OpenRouter live free/router tail (hermes_cli.models_openrouter)."""

from unittest.mock import patch

from hermes_cli.models import fetch_openrouter_models
import hermes_cli.models as _models_mod
from hermes_cli.models_openrouter import append_live_free_routers


class TestAppendLiveFreeRouters:
    """The curated manifest is a hand-picked subset that lags the live catalog: a whole
    generation of ``:free`` SKUs and OpenRouter's own router family (``openrouter/free``)
    can ship before any manifest PR. Live-only entries in those two groups must surface
    without waiting for curation (observed 2026-11-06: ``openrouter/free`` was live and
    $0/tool-capable for months while absent from the 74-id manifest)."""

    _TOOLS = {"supported_parameters": ["temperature", "tools", "tool_choice"]}

    @staticmethod
    def _install_live(monkeypatch, live_items, curated):
        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        by_id = {item.get("id"): item for item in live_items}
        monkeypatch.setattr(
            _models_mod,
            "_fetch_live_catalog_index",
            lambda *args, **kwargs: (live_items, by_id),
        )
        monkeypatch.setattr(
            "hermes_cli.model_catalog.get_curated_openrouter_models", lambda: curated
        )

    def test_appends_live_free_skus_and_routers_without_the_manifest(self, monkeypatch):
        live_items = [
            {"id": "qwen/qwen3.7-max", "pricing": {"prompt": "0.0001", "completion": "0.0002"}, **self._TOOLS},
            # live-only :free SKU → tailed and labeled free
            {"id": "vendor/flash:free", "pricing": {"prompt": "0", "completion": "0"}, **self._TOOLS},
            # live-only non-curated router, $0 → openrouter/free's own case, labeled free
            {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"}, **self._TOOLS},
            # live-only paid router → labeled auto-router
            {"id": "openrouter/auto", "pricing": {"prompt": "0.00001", "completion": "0.00002"}, **self._TOOLS},
            # paid non-router vendor model → NOT tailed
            {"id": "deadvendor/paid", "pricing": {"prompt": "0.001", "completion": "0.002"}, **self._TOOLS},
            # :free without tool support → stays hidden (tool-calling-first)
            {"id": "nontools/free:free", "pricing": {"prompt": "0", "completion": "0"},
             "supported_parameters": ["temperature"]},
        ]
        self._install_live(monkeypatch, live_items, curated=[("qwen/qwen3.7-max", "")])

        models = fetch_openrouter_models(force_refresh=True)

        assert [mid for mid, _ in models] == [
            "qwen/qwen3.7-max",
            "vendor/flash:free",
            "openrouter/free",
            "openrouter/auto",
        ]
        descriptions = dict(models)
        assert descriptions["vendor/flash:free"] == "free"
        assert descriptions["openrouter/free"] == "free"
        assert descriptions["openrouter/auto"] == "auto-router"

    def test_curated_ids_are_never_duplicated_by_the_live_tail(self, monkeypatch):
        live_items = [
            {"id": "z-ai/glm-5.2:free", "pricing": {"prompt": "0", "completion": "0"}, **self._TOOLS},
        ]
        self._install_live(monkeypatch, live_items, curated=[("z-ai/glm-5.2:free", "free")])

        models = fetch_openrouter_models(force_refresh=True)

        assert models == [("z-ai/glm-5.2:free", "free")]


class TestAppendLiveFreeRoutersDirect:
    """Direct coverage of the helper contract (items, delight edge cases)."""

    _TOOLS = {"supported_parameters": ["tools"]}

    def test_non_dict_items_and_missing_ids_are_skipped(self):
        curated = [("a/kept", "")]
        live_items = [
            None,
            "not-a-dict",
            {"no-id": True},
            {"id": "   ", "pricing": {"prompt": "0", "completion": "0"}, **self._TOOLS},
            {"id": "vendor/flash:free", "pricing": {"prompt": "0", "completion": "0"}, **self._TOOLS},
        ]
        assert append_live_free_routers(curated, live_items, lambda _item: True, lambda _pricing: True) == [
            ("a/kept", ""),
            ("vendor/flash:free", "free"),
        ]

    def test_curated_only_payload_returns_unchanged(self):
        curated = [("a/kept", "recommended")]
        out = append_live_free_routers(curated, [], lambda _item: True, lambda _pricing: True)
        assert out == curated and out is curated
