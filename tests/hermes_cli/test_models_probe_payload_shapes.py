"""Regression: ``/models`` payload shapes that are not the ``{"data": [...]}`` envelope.

Together AI answers ``GET /models`` with a bare JSON array of model objects. ``probe_api_models``
read ``data.get("data", [])`` unconditionally, so that shape raised ``AttributeError``; the picker's
discovery caller swallows probe exceptions, so the provider's row silently stayed on its persisted
catalog and new models never appeared. Both shapes are valid wire responses and must yield ids.
"""

from unittest.mock import patch

import hermes_cli.models as models_mod


def _probe(payload, base_url):
    """Run a real probe against a stubbed ``/models`` response."""
    with patch.object(models_mod, "_get_json", return_value=payload):
        return models_mod.probe_api_models("dummy-key", base_url)


def test_probe_accepts_bare_array_payload():
    result = _probe(
        [{"id": "acme/chat-large"}, {"id": "acme/chat-small"}],
        "https://bare-array.example.com/v1",
    )
    assert result["models"] == ["acme/chat-large", "acme/chat-small"]


def test_probe_accepts_envelope_payload():
    result = _probe(
        {"object": "list", "data": [{"id": "acme/chat-large"}]},
        "https://envelope.example.com/v1",
    )
    assert result["models"] == ["acme/chat-large"]


def test_probe_drops_non_object_entries_in_either_shape():
    array_shape = _probe(
        [{"id": "acme/chat-large"}, "acme/not-an-object", None],
        "https://mixed-array.example.com/v1",
    )
    envelope_shape = _probe(
        {"data": [{"id": "acme/chat-large"}, "acme/not-an-object"]},
        "https://mixed-envelope.example.com/v1",
    )
    assert array_shape["models"] == ["acme/chat-large"]
    assert envelope_shape["models"] == ["acme/chat-large"]
