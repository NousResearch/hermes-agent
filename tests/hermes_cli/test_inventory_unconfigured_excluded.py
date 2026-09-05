"""``model_catalog.excluded_providers`` must also hide the unconfigured
"setup skeleton" rows that ``build_models_payload(include_unconfigured=True)``
appends for the TUI / desktop / web pickers.

``list_authenticated_providers`` already honours the exclusion for
authenticated rows; the skeleton path bypassed it, so an excluded provider
reappeared in every GUI picker as an empty row.
"""

from hermes_cli.inventory import ConfigContext, _append_unconfigured_rows


def _ctx(**kw) -> ConfigContext:
    base = dict(
        current_provider="9router",
        current_model="glm",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        excluded_providers=[],
    )
    base.update(kw)
    return ConfigContext(**base)


def test_unconfigured_rows_honor_excluded_providers():
    rows = _append_unconfigured_rows([], _ctx(excluded_providers=["Anthropic ", "openrouter"]))
    slugs = {r["slug"] for r in rows}
    assert "anthropic" not in slugs
    assert "openrouter" not in slugs
    assert "deepseek" in slugs  # unrelated canonical providers still listed


def test_unconfigured_rows_keep_current_provider_even_if_excluded():
    rows = _append_unconfigured_rows(
        [], _ctx(current_provider="anthropic", excluded_providers=["anthropic"])
    )
    cur = [r for r in rows if r["slug"] == "anthropic"]
    assert cur and cur[0]["is_current"] and cur[0]["source"] == "configured-current"
