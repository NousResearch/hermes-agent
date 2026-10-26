"""The MiniMax flagship must be offered on every catalog surface that carries MiniMax.

A contract between two pieces of data (the direct provider row and the aggregator
rows), not a snapshot of either: it names no model id, so the next MiniMax release
keeps it meaningful. Regression for #142648 (new models never surface in the picker)
and the aggregator half of #134659.
"""

from hermes_cli.models_catalog_static import OPENROUTER_MODELS, _PROVIDER_MODELS


def test_minimax_flagship_is_also_offered_on_the_aggregator_rows():
    """The model the direct MiniMax row leads with must also be offered, under the
    vendor's aggregator slug, on the OpenRouter and Nous picker rows.

    Before this landed, ``MiniMax-M3.1-Flash`` was selectable on the direct MiniMax
    routes but absent from the aggregator pickers — a model the user could reach on
    one surface and not on the other.
    """
    flagship = _PROVIDER_MODELS["minimax"][0]
    slug = f"minimax/{flagship.lower()}"

    openrouter_ids = {mid for mid, _ in OPENROUTER_MODELS}
    nous_ids = set(_PROVIDER_MODELS["nous"])

    assert slug in openrouter_ids, (
        f"the MiniMax flagship {flagship} is missing from the OpenRouter curated row "
        f"(expected {slug})"
    )
    assert slug in nous_ids, (
        f"the MiniMax flagship {flagship} is missing from the Nous curated row "
        f"(expected {slug})"
    )
