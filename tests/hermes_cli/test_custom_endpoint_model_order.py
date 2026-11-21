"""Configured custom endpoint models stay first when live catalogs are large."""

from hermes_cli.model_switch_providers import _merge_live_models, list_authenticated_providers


def test_merge_live_models_keeps_only_served_configured_models_first():
    assert _merge_live_models(["pinned", "gone", "pinned"], ["first", "pinned", "last"]) == [
        "pinned", "first", "last",
    ]


def test_picker_lists_configured_models_first(monkeypatch):
    url = "http://127.0.0.1:9999/v1"
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers._fetch_picker_live_models",
        lambda *_args, **_kwargs: ["catalog-1", "catalog-2", "pinned"],
    )

    rows = list_authenticated_providers(
        current_provider="my-router", current_base_url=url,
        user_providers={"my-router": {"base_url": url, "api_key": "test", "models": {"pinned": {}}}},
        probe_custom_providers=False, probe_current_custom_provider=True,
    )

    row = next(row for row in rows if row["slug"] == "my-router")
    assert row["models"] == ["pinned", "catalog-1", "catalog-2"]
    assert row["total_models"] == 3
