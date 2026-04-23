from __future__ import annotations

from types import MappingProxyType

import pytest

from agent.route_categories import (
    BUILTIN_LITERAL_CATEGORIES,
    BUILTIN_ROUTE_CATEGORIES,
    DEFAULT_LITERAL_CATEGORY,
    DEFAULT_ROUTE_CATEGORY,
    LiteralCategory,
    RouteCategory,
    get_literal_categories,
    get_literal_category,
    get_route_categories,
    get_route_category,
    resolve_literal_category,
    resolve_literal_category_to_route_category,
    resolve_route_category,
    validate_literal_categories,
    validate_literal_category,
    validate_route_categories,
    validate_route_category,
)


EXPECTED_NAMES = (
    "ultrabrain",
    "deep",
    "quick",
    "visual",
    "writing",
    "artistry",
    "unspecified_low",
    "unspecified_high",
)

EXPECTED_LITERAL_NAMES = (
    "ultrabrain",
    "deep",
    "quick",
    "visual-engineering",
    "writing",
    "artistry",
    "unspecified-low",
    "unspecified-high",
)


def test_builtin_route_categories_are_preserved_exactly() -> None:
    assert tuple(BUILTIN_ROUTE_CATEGORIES) == EXPECTED_NAMES
    assert DEFAULT_ROUTE_CATEGORY == "unspecified_low"


def test_builtin_literal_categories_expose_upstream_product_names_exactly() -> None:
    assert tuple(BUILTIN_LITERAL_CATEGORIES) == EXPECTED_LITERAL_NAMES
    assert DEFAULT_LITERAL_CATEGORY == "unspecified-low"
    assert BUILTIN_LITERAL_CATEGORIES["visual-engineering"].route_category == "visual"
    assert BUILTIN_LITERAL_CATEGORIES["unspecified-low"].route_category == "unspecified_low"
    assert BUILTIN_LITERAL_CATEGORIES["unspecified-high"].route_category == "unspecified_high"


@pytest.mark.parametrize(
    ("category", "message"),
    [
        (LiteralCategory(name="", summary="summary", route_category="deep"), "name"),
        (LiteralCategory(name="deep", summary="", route_category="deep"), "summary"),
        (LiteralCategory(name="deep", summary="summary", route_category=""), "route_category"),
        (LiteralCategory(name=" deep ", summary="summary", route_category="deep"), "normalized"),
        (LiteralCategory(name="deep", summary=" summary ", route_category="deep"), "normalized"),
        (LiteralCategory(name="deep", summary="summary", route_category=" deep "), "normalized"),
    ],
)
def test_validate_literal_category_rejects_invalid_values(category: LiteralCategory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_literal_category(category)


def test_validate_literal_categories_rejects_duplicate_names() -> None:
    categories = (
        LiteralCategory(name="deep", summary="summary", route_category="deep"),
        LiteralCategory(name="deep", summary="another", route_category="quick"),
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_literal_categories(categories)


def test_get_literal_category_is_strict_on_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown literal category"):
        get_literal_category("unknown")


def test_get_literal_categories_returns_immutable_registry_view() -> None:
    registry = get_literal_categories()

    assert isinstance(registry, MappingProxyType)
    with pytest.raises(TypeError):
        registry["new"] = LiteralCategory(name="new", summary="summary", route_category="deep")


def test_resolve_literal_category_normalizes_compatibility_aliases_to_canonical_upstream_names() -> None:
    assert resolve_literal_category(None) is BUILTIN_LITERAL_CATEGORIES[DEFAULT_LITERAL_CATEGORY]
    assert resolve_literal_category("visual") is BUILTIN_LITERAL_CATEGORIES["visual-engineering"]
    assert resolve_literal_category("unspecified_low") is BUILTIN_LITERAL_CATEGORIES["unspecified-low"]
    assert resolve_literal_category("unspecified-high") is BUILTIN_LITERAL_CATEGORIES["unspecified-high"]
    assert resolve_literal_category("unknown-literal-category") is BUILTIN_LITERAL_CATEGORIES[DEFAULT_LITERAL_CATEGORY]


def test_literal_category_resolution_truthfully_delegates_fallbacks_to_mapped_route_categories() -> None:
    assert resolve_literal_category_to_route_category("visual-engineering") is BUILTIN_ROUTE_CATEGORIES["visual"]
    assert resolve_literal_category_to_route_category("visual").fallback_models == ("quick",)
    assert resolve_literal_category_to_route_category("unspecified-low").fallback_models == ("default",)


@pytest.mark.parametrize(
    ("category", "message"),
    [
        (RouteCategory(name="", summary="summary", intensity="high"), "name"),
        (RouteCategory(name="deep", summary="", intensity="high"), "summary"),
        (RouteCategory(name="deep", summary="summary", intensity=""), "intensity"),
        (RouteCategory(name="deep", summary="summary", intensity="high", fallback_models=(" default ",)), "normalized"),
        (RouteCategory(name=" deep ", summary="summary", intensity="high"), "normalized"),
        (RouteCategory(name="deep", summary=" summary ", intensity="high"), "normalized"),
        (RouteCategory(name="deep", summary="summary", intensity=" high "), "normalized"),
    ],
)
def test_validate_route_category_rejects_invalid_values(category: RouteCategory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_route_category(category)


def test_validate_route_categories_rejects_duplicate_names() -> None:
    categories = (
        RouteCategory(name="deep", summary="summary", intensity="high"),
        RouteCategory(name="deep", summary="another", intensity="low"),
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_route_categories(categories)


def test_get_route_category_is_strict_on_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown route category"):
        get_route_category("unknown")


def test_get_route_categories_returns_immutable_registry_view() -> None:
    registry = get_route_categories()

    assert isinstance(registry, MappingProxyType)
    with pytest.raises(TypeError):
        registry["new"] = RouteCategory(name="new", summary="summary", intensity="low")


def test_resolve_route_category_falls_back_to_default_without_reusing_other_taxonomy() -> None:
    assert resolve_route_category(None) is BUILTIN_ROUTE_CATEGORIES[DEFAULT_ROUTE_CATEGORY]
    assert resolve_route_category("") is BUILTIN_ROUTE_CATEGORIES[DEFAULT_ROUTE_CATEGORY]
    assert resolve_route_category("  deep  ") is BUILTIN_ROUTE_CATEGORIES["deep"]
    assert resolve_route_category("unknown-route-category") is BUILTIN_ROUTE_CATEGORIES[DEFAULT_ROUTE_CATEGORY]


def test_builtin_route_categories_include_default_fallback_chains() -> None:
    assert BUILTIN_ROUTE_CATEGORIES["ultrabrain"].fallback_models == ("deep",)
    assert BUILTIN_ROUTE_CATEGORIES["deep"].fallback_models == ("quick",)
    assert BUILTIN_ROUTE_CATEGORIES["quick"].fallback_models == ("default",)
    assert BUILTIN_ROUTE_CATEGORIES["visual"].fallback_models == ("quick",)
    assert BUILTIN_ROUTE_CATEGORIES["writing"].fallback_models == ("quick",)
    assert BUILTIN_ROUTE_CATEGORIES["artistry"].fallback_models == ("visual",)
    assert BUILTIN_ROUTE_CATEGORIES["unspecified_low"].fallback_models == ("default",)
    assert BUILTIN_ROUTE_CATEGORIES["unspecified_high"].fallback_models == ("deep",)
