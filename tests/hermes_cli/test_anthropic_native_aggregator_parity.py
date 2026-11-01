"""Native Anthropic curated parity with the aggregator catalogs.

Anthropic models are added to ``OPENROUTER_MODELS`` first (the Nous Portal list is derived from
that same tuple), and the native ``_PROVIDER_MODELS["anthropic"]`` list is edited separately —
so a release lands on the aggregator routes and silently misses the native one. Opus 5.5 shipped
that way: present for OpenRouter/Nous, absent from the native curated list.

The native list is not cosmetic. ``merge_profile_catalog`` puts curated entries FIRST and appends
live-only extras, so a model missing from it either vanishes (live ``/v1/models`` lags a freshly
routed alias) or sorts below every older 4.x id — at the bottom of the picker, in Anthropic's wire
spelling rather than the public slug.

This is a relationship between two lists, not a snapshot of either: it names no model, so it keeps
holding as the catalogs turn over, and it fails the moment the two drift apart again.
"""

from hermes_cli.models_catalog_static import (
    OPENROUTER_MODELS,
    _OPENROUTER_ONLY,
    _PROVIDER_MODELS,
)

_ANTHROPIC_PREFIX = "anthropic/"


def _fold(model_id: str) -> str:
    """Fold the separators the two catalogs disagree on.

    The aggregator lists carry the public slug (``claude-opus-5.5``); the native list carries
    Anthropic's wire spelling (``claude-opus-5-5``). Same model, and a parity check must not be
    fooled by the punctuation.
    """
    return model_id.strip().lower().replace("-", " ").replace("_", " ").replace(".", " ")


def _aggregator_anthropic_slugs() -> list[str]:
    """Anthropic models the aggregator catalogs offer, minus the OpenRouter-only SKUs.

    ``_OPENROUTER_ONLY`` holds relay-exclusive variants (the ``-fast`` SKUs); Anthropic does not
    serve those ids natively, so they are not parity failures.
    """
    return [
        mid[len(_ANTHROPIC_PREFIX):]
        for mid, _desc in OPENROUTER_MODELS
        if mid.startswith(_ANTHROPIC_PREFIX) and mid not in _OPENROUTER_ONLY
    ]


def _covered_by(native_folded: str, slug_folded: str) -> bool:
    """True when a native entry is the aggregator slug, or a dated snapshot of it.

    The native list pins dated snapshots (``claude-haiku-4-5-20251001``) where the aggregators
    carry the bare alias (``claude-haiku-4.5``). The boundary space matters: without it
    ``claude-fable-5`` would match ``claude-fable-5-1`` and a genuinely missing model would pass.
    """
    return native_folded == slug_folded or native_folded.startswith(slug_folded + " ")


def test_every_aggregator_anthropic_model_is_in_the_native_curated_list():
    """A model routed on OpenRouter/Nous must also be curated natively.

    Without this the native picker falls back to whatever live ``/v1/models`` returns, where a new
    flagship is appended below every older 4.x id — or missing entirely while the endpoint lags.
    """
    native_folded = [_fold(m) for m in _PROVIDER_MODELS["anthropic"]]

    missing = [
        slug
        for slug in _aggregator_anthropic_slugs()
        if not any(_covered_by(n, _fold(slug)) for n in native_folded)
    ]

    assert not missing, (
        "Anthropic models offered on the aggregator routes but absent from the native curated "
        f"list: {missing}. Add them to _PROVIDER_MODELS['anthropic'] in models_catalog_static.py "
        "(newest first) so the native picker leads with them instead of appending the live wire id "
        "below the older 4.x entries."
    )
