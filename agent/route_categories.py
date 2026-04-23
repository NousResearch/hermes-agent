"""Wave 1 route-category schema.

Route categories are routing labels, not delegation profiles.
They intentionally use a distinct data model so later config and resolution work
can map route categories onto delegation behavior without collapsing the two.

DG2 adds a bounded literal-category product facade on top of the existing
route-category substrate. Literal categories expose upstream-facing names while
truthfully delegating route/fallback behavior to mapped internal route
categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


@dataclass(frozen=True)
class RouteCategory:
    """First-class routing concept used to classify delegation lanes."""

    name: str
    summary: str
    intensity: str
    fallback_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiteralCategory:
    """Bounded upstream-facing category facade mapped onto route categories."""

    name: str
    summary: str
    route_category: str
    default_runtime_mode: str | None = None


def _require_non_empty_normalized_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must be non-empty")
    if normalized_value != value:
        raise ValueError(f"{field_name} must use canonical normalized form")
    return normalized_value


def _normalize_literal_lookup_name(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    return token.replace("_", "-").replace(" ", "-")


def validate_route_category(category: RouteCategory) -> RouteCategory:
    if not isinstance(category, RouteCategory):
        raise ValueError("route category must be a RouteCategory instance")

    _require_non_empty_normalized_string(category.name, "name")
    _require_non_empty_normalized_string(category.summary, "summary")
    _require_non_empty_normalized_string(category.intensity, "intensity")
    if not isinstance(category.fallback_models, tuple):
        raise ValueError("fallback_models must be a tuple")
    for fallback_model in category.fallback_models:
        _require_non_empty_normalized_string(fallback_model, "fallback_models")
    return category


def validate_literal_category(category: LiteralCategory) -> LiteralCategory:
    if not isinstance(category, LiteralCategory):
        raise ValueError("literal category must be a LiteralCategory instance")

    _require_non_empty_normalized_string(category.name, "name")
    _require_non_empty_normalized_string(category.summary, "summary")
    _require_non_empty_normalized_string(category.route_category, "route_category")
    if category.default_runtime_mode is not None:
        _require_non_empty_normalized_string(category.default_runtime_mode, "default_runtime_mode")
    return category


def validate_route_categories(
    categories: tuple[RouteCategory, ...],
) -> tuple[RouteCategory, ...]:
    seen_names: set[str] = set()

    for category in categories:
        validate_route_category(category)
        if category.name in seen_names:
            raise ValueError(f"duplicate route category name: {category.name}")
        seen_names.add(category.name)

    return categories


def validate_literal_categories(
    categories: tuple[LiteralCategory, ...],
) -> tuple[LiteralCategory, ...]:
    seen_names: set[str] = set()

    for category in categories:
        validate_literal_category(category)
        if category.name in seen_names:
            raise ValueError(f"duplicate literal category name: {category.name}")
        seen_names.add(category.name)

    return categories


_ROUTE_CATEGORY_SPECS: Final[tuple[tuple[str, str, str, tuple[str, ...]], ...]] = (
    ("ultrabrain", "Highest-depth routing lane for especially demanding reasoning.", "highest", ("deep",)),
    ("deep", "High-depth routing lane for careful multi-step work.", "high", ("quick",)),
    ("quick", "Fast-path routing lane for lighter-weight execution.", "low", ("default",)),
    ("visual", "Routing lane for image- or screenshot-centered work.", "medium", ("quick",)),
    ("writing", "Routing lane for drafting, editing, and communication-heavy work.", "medium", ("quick",)),
    ("artistry", "Routing lane for creative generation, aesthetic exploration, and design-led work.", "medium", ("visual",)),
    ("unspecified_low", "Fallback routing lane when intent is underspecified but appears lightweight.", "low", ("default",)),
    ("unspecified_high", "Fallback routing lane when intent is underspecified and appears demanding.", "high", ("deep",)),
)


_LITERAL_CATEGORY_SPECS: Final[tuple[tuple[str, str, str, str | None], ...]] = (
    ("ultrabrain", "Highest-depth upstream category for especially demanding reasoning.", "ultrabrain", None),
    ("deep", "High-depth upstream category for careful multi-step work.", "deep", None),
    ("quick", "Fast-path upstream category for lighter-weight execution.", "quick", None),
    ("visual-engineering", "Upstream category for image-, screenshot-, and visual-engineering-centered work.", "visual", None),
    ("writing", "Upstream category for drafting, editing, and communication-heavy work.", "writing", None),
    ("artistry", "Upstream category for creative generation, aesthetic exploration, and design-led work.", "artistry", None),
    ("unspecified-low", "Fallback upstream category when intent is underspecified but appears lightweight.", "unspecified_low", None),
    ("unspecified-high", "Fallback upstream category when intent is underspecified and appears demanding.", "unspecified_high", None),
)


_BUILTIN_ROUTE_CATEGORY_LIST: Final[tuple[RouteCategory, ...]] = validate_route_categories(
    tuple(
        RouteCategory(name=name, summary=summary, intensity=intensity, fallback_models=fallback_models)
        for name, summary, intensity, fallback_models in _ROUTE_CATEGORY_SPECS
    )
)

_BUILTIN_LITERAL_CATEGORY_LIST: Final[tuple[LiteralCategory, ...]] = validate_literal_categories(
    tuple(
        LiteralCategory(
            name=name,
            summary=summary,
            route_category=route_category,
            default_runtime_mode=default_runtime_mode,
        )
        for name, summary, route_category, default_runtime_mode in _LITERAL_CATEGORY_SPECS
    )
)

BUILTIN_ROUTE_CATEGORIES: Final[Mapping[str, RouteCategory]] = MappingProxyType(
    {category.name: category for category in _BUILTIN_ROUTE_CATEGORY_LIST}
)

BUILTIN_LITERAL_CATEGORIES: Final[Mapping[str, LiteralCategory]] = MappingProxyType(
    {category.name: category for category in _BUILTIN_LITERAL_CATEGORY_LIST}
)

DEFAULT_ROUTE_CATEGORY: Final[str] = "unspecified_low"
DEFAULT_LITERAL_CATEGORY: Final[str] = "unspecified-low"

_LITERAL_CATEGORY_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        name: name
        for name in BUILTIN_LITERAL_CATEGORIES
    }
    | {
        "visual": "visual-engineering",
        "unspecified-low": "unspecified-low",
        "unspecified-high": "unspecified-high",
        "unspecified_low": "unspecified-low",
        "unspecified_high": "unspecified-high",
    }
)


def get_route_categories() -> Mapping[str, RouteCategory]:
    """Return the built-in route-category registry."""

    return BUILTIN_ROUTE_CATEGORIES


def get_literal_categories() -> Mapping[str, LiteralCategory]:
    """Return the built-in literal-category registry."""

    return BUILTIN_LITERAL_CATEGORIES


def get_route_category(name: str) -> RouteCategory:
    """Resolve a built-in route category by name."""

    try:
        return BUILTIN_ROUTE_CATEGORIES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown route category: {name}") from exc


def get_literal_category(name: str) -> LiteralCategory:
    """Resolve a built-in literal category by exact canonical name."""

    try:
        return BUILTIN_LITERAL_CATEGORIES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown literal category: {name}") from exc


def resolve_route_category(name: str | None) -> RouteCategory:
    """Resolve a route category, falling back to the canonical Wave 1 default.

    Route categories remain first-class routing labels. Unknown, empty, or
    whitespace-padded values should degrade to the compatibility-safe default
    instead of silently reusing delegation-profile semantics.
    """

    normalized_name = str(name or "").strip()
    if not normalized_name:
        return BUILTIN_ROUTE_CATEGORIES[DEFAULT_ROUTE_CATEGORY]
    return BUILTIN_ROUTE_CATEGORIES.get(normalized_name, BUILTIN_ROUTE_CATEGORIES[DEFAULT_ROUTE_CATEGORY])


def resolve_literal_category(name: str | None) -> LiteralCategory:
    """Resolve a literal product category with deterministic compatibility aliases."""

    normalized_name = _normalize_literal_lookup_name(name)
    if not normalized_name:
        return BUILTIN_LITERAL_CATEGORIES[DEFAULT_LITERAL_CATEGORY]
    canonical_name = _LITERAL_CATEGORY_ALIASES.get(normalized_name, DEFAULT_LITERAL_CATEGORY)
    return BUILTIN_LITERAL_CATEGORIES[canonical_name]


def resolve_literal_category_from_route_category(route_category_name: str | None) -> LiteralCategory:
    """Translate a route category into the bounded literal category facade."""

    resolved_route_category = resolve_route_category(route_category_name)
    for literal_category in BUILTIN_LITERAL_CATEGORIES.values():
        if literal_category.route_category == resolved_route_category.name:
            return literal_category
    return BUILTIN_LITERAL_CATEGORIES[DEFAULT_LITERAL_CATEGORY]


def resolve_literal_category_to_route_category(name: str | None) -> RouteCategory:
    """Resolve literal category fallback/runtime behavior via the mapped route category."""

    literal_category = resolve_literal_category(name)
    return BUILTIN_ROUTE_CATEGORIES[literal_category.route_category]


__all__ = [
    "BUILTIN_LITERAL_CATEGORIES",
    "BUILTIN_ROUTE_CATEGORIES",
    "DEFAULT_LITERAL_CATEGORY",
    "DEFAULT_ROUTE_CATEGORY",
    "LiteralCategory",
    "RouteCategory",
    "get_literal_categories",
    "get_literal_category",
    "get_route_categories",
    "get_route_category",
    "resolve_literal_category",
    "resolve_literal_category_from_route_category",
    "resolve_literal_category_to_route_category",
    "resolve_route_category",
    "validate_literal_categories",
    "validate_literal_category",
    "validate_route_categories",
    "validate_route_category",
]
