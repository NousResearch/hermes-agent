"""The active provider's row survives aggregator-overlap dedup.

``_strip_aggregator_overlaps`` exists so picking a model from a routing
aggregator (OpenRouter, custom:* proxies) cannot silently swap the provider
the user expected when a more specific provider serves the same id. Applied
to the row that IS the active provider, the same rule backfires: the model
the session is running vanishes from its own row (e.g. openrouter's
``deepseek/deepseek-v4.1-flash`` while ``custom:novita`` also lists it), so
the picker misreports the active backend and the model cannot be re-selected.

Contract: a current row keeps its full model list; non-current aggregator
rows still drop overlaps; user-defined rows stay untouched either way.
"""

from __future__ import annotations

import hermes_cli.inventory as inv


SHARED = "deepseek/deepseek-v4.1-flash"
ONLY_ROUTER = "anthropic/claude-fable-5"


def _rows(*, router_is_current: bool) -> list[dict]:
    return [
        {
            "slug": "openrouter",
            "name": "OpenRouter",
            "is_current": router_is_current,
            "is_user_defined": False,
            "models": [SHARED, ONLY_ROUTER],
            "total_models": 2,
        },
        {
            "slug": "custom:novita",
            "name": "novita",
            "is_current": not router_is_current,
            "is_user_defined": True,
            "models": [SHARED],
            "total_models": 1,
        },
    ]


def test_current_aggregator_row_keeps_overlapping_models():
    rows = _rows(router_is_current=True)

    inv._strip_aggregator_overlaps(rows)

    assert rows[0]["models"] == [SHARED, ONLY_ROUTER], (
        "the active provider's row must keep every model it serves")
    assert rows[0]["total_models"] == 2


def test_non_current_aggregator_row_still_drops_overlaps():
    rows = _rows(router_is_current=False)

    inv._strip_aggregator_overlaps(rows)

    assert rows[0]["models"] == [ONLY_ROUTER], (
        "an inactive aggregator row must still hide ids a custom provider serves")
    assert rows[0]["total_models"] == 1


def test_user_defined_row_is_never_stripped():
    rows = _rows(router_is_current=True)

    inv._strip_aggregator_overlaps(rows)

    assert rows[1]["models"] == [SHARED], (
        "a user's own provider is not a duplicate of itself")
