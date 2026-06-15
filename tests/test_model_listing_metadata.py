"""/models listing metadata: which per-model fields survive to the picker."""

from hermes_cli.models import _extract_api_model_listing


def test_description_survives_model_listing_extraction():
    models, metadata = _extract_api_model_listing(
        {
            "data": [
                {
                    "id": "auto",
                    "object": "model",
                    "owned_by": "ai-router",
                    "description": "Picks the best model for each request.",
                    "internal_only": "must be dropped",
                },
                {"id": "plain", "object": "model"},
            ]
        }
    )
    assert models == ["auto", "plain"]
    assert metadata["auto"]["description"] == "Picks the best model for each request."
    assert "internal_only" not in metadata["auto"]
    # No allowlisted fields at all -> no metadata row for the model.
    assert "plain" not in metadata


def test_empty_description_is_dropped():
    _models, metadata = _extract_api_model_listing({"data": [{"id": "m", "description": ""}]})
    assert "m" not in metadata
