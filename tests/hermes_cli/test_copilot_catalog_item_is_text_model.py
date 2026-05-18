"""Regression tests for _copilot_catalog_item_is_text_model.

Verifies that the Copilot model catalog filter does not exclude models
that are actually usable via hermes setup model (the interactive CLI picker).

The ``model_picker_enabled`` field in the GitHub Copilot API response is NOT
a reliable signal — models like gpt-5.5 and claude-opus-4.7 are returned with
``model_picker_enabled: false`` yet work correctly when selected in
``hermes setup model`` and when used as the active model in conversations.
We therefore do not filter on this field.
"""

import pytest

from hermes_cli.models import _copilot_catalog_item_is_text_model


class TestCopilotCatalogItemIsTextModel:
    def test_model_picker_enabled_false_still_included(self):
        """Models with model_picker_enabled=false must NOT be filtered out.

        Regression: gpt-5.5 and claude-opus-4.7 carry model_picker_enabled=false
        in the live Copilot catalog, yet they are fully usable.  The setup flow
        (main.py _model_flow_copilot) shows ALL catalog entries regardless of
        this field.  The /model picker must match that behaviour.
        """
        item = {
            "id": "gpt-5.5",
            "model_picker_enabled": False,
            "capabilities": {"type": "chat"},
            "supported_endpoints": ["/chat/completions"],
        }
        assert _copilot_catalog_item_is_text_model(item) is True

    def test_claude_opus_47_model_picker_enabled_false_included(self):
        """Claude Opus-4.7 regression: same pattern as gpt-5.5 above."""
        item = {
            "id": "claude-opus-4.7",
            "model_picker_enabled": False,
            "capabilities": {"type": "chat"},
            "supported_endpoints": ["/v1/messages"],
        }
        assert _copilot_catalog_item_is_text_model(item) is True

    def test_model_picker_enabled_true_still_included(self):
        """Models with model_picker_enabled=true (the happy path) remain included."""
        item = {
            "id": "gpt-4o",
            "model_picker_enabled": True,
            "capabilities": {"type": "chat"},
            "supported_endpoints": ["/chat/completions"],
        }
        assert _copilot_catalog_item_is_text_model(item) is True

    def test_no_model_picker_enabled_field_still_included(self):
        """When the field is absent entirely the model must still be included."""
        item = {
            "id": "claude-sonnet-4.6",
            "capabilities": {"type": "chat"},
            "supported_endpoints": ["/chat/completions", "/responses"],
        }
        assert _copilot_catalog_item_is_text_model(item) is True

    def test_non_chat_type_filtered(self):
        """Non-chat model types (e.g. voice, embedding) are correctly excluded."""
        item = {
            "id": "gpt-4o-audio",
            "model_picker_enabled": True,
            "capabilities": {"type": "voice"},
            "supported_endpoints": ["/audio/transcriptions"],
        }
        assert _copilot_catalog_item_is_text_model(item) is False

    def test_unsupported_endpoint_filtered(self):
        """Models that expose no supported chat endpoint are correctly excluded."""
        item = {
            "id": "gpt-4o-embedding-only",
            "model_picker_enabled": True,
            "capabilities": {"type": "chat"},
            "supported_endpoints": ["/embeddings"],
        }
        assert _copilot_catalog_item_is_text_model(item) is False

    def test_empty_id_filtered(self):
        """Items without an id are filtered even if other fields are present."""
        item = {
            "id": "",
            "model_picker_enabled": True,
            "capabilities": {"type": "chat"},
            "supported_endpoints": ["/chat/completions"],
        }
        assert _copilot_catalog_item_is_text_model(item) is False

    def test_responses_endpoint_included(self):
        """/responses endpoint (used by newer models) is recognised."""
        item = {
            "id": "claude-opus-4.6",
            "model_picker_enabled": False,
            "capabilities": {"type": "chat"},
            "supported_endpoints": ["/responses"],
        }
        assert _copilot_catalog_item_is_text_model(item) is True

    def test_v1_messages_endpoint_included(self):
        """/v1/messages endpoint (Anthropic-style) is recognised."""
        item = {
            "id": "claude-sonnet-4.5",
            "model_picker_enabled": False,
            "capabilities": {"type": "chat"},
            "supported_endpoints": ["/v1/messages"],
        }
        assert _copilot_catalog_item_is_text_model(item) is True
