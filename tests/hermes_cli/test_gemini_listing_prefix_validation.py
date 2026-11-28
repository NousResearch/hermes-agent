"""``/model`` validation for Gemini ids spelled with the listing's ``models/`` prefix (#152786).

Gemini's native ``v1beta`` ListModels — and its OpenAI-compatible surface — answer every id as
``models/<id>``, while the curated catalog, the picker and the request builders use the bare id.
Validation normalised only the *listing* side, so an id handed back by a client that echoes the
listing could never match, even though the same model sat in that listing.
"""

from unittest.mock import patch

from hermes_cli.models_validate import validate_requested_model


LISTING = ["models/gemini-4-carbon", "models/gemini-4-neon"]


def _validate(model, provider="gemini", listing=LISTING):
    """Validate *model* against a Gemini listing, with the probe stubbed (no network)."""
    probe_payload = {
        "models": listing or [],
        "probed_url": "",
        "resolved_base_url": "",
        "suggested_base_url": None,
        "used_fallback": False,
    }
    with patch("hermes_cli.models.fetch_api_models", return_value=listing), \
            patch("hermes_cli.models.probe_api_models", return_value=probe_payload):
        return validate_requested_model(model, provider, api_key="fake-gemini-key")


def test_prefixed_id_from_the_listing_validates_like_the_bare_id():
    """The reported defect: the id the listing itself returns was rejected."""
    prefixed = _validate("models/gemini-4-carbon")
    bare = _validate("gemini-4-carbon")

    assert prefixed["accepted"] is True
    assert prefixed["recognized"] is True
    assert prefixed["message"] is None
    assert prefixed == bare


def test_bare_id_still_validates():
    assert _validate("gemini-4-neon")["accepted"] is True


def test_provider_aliases_accept_the_prefixed_form():
    assert _validate("models/gemini-4-carbon", provider="google")["accepted"] is True


def test_unknown_prefixed_id_is_still_rejected_with_bare_suggestions():
    """Stripping the prefix for comparison must not turn validation into accept-all."""
    verdict = _validate("models/gemini-9-nonexistent")

    assert verdict["accepted"] is False
    assert "gemini-4-carbon" in verdict["message"]
    assert "models/gemini-4-carbon" not in verdict["message"]


def test_curated_fallback_recognizes_the_prefixed_form_when_the_listing_is_unreachable():
    """With /models down, the curated catalog decides — and it lists bare ids only."""
    verdict = _validate("models/gemini-4-carbon", listing=None)

    assert verdict["accepted"] is True
    assert verdict["recognized"] is True
    assert verdict["message"] is None
