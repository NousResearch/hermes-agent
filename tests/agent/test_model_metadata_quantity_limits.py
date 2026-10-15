"""Quantity-object limit wrappers in ``/models`` payloads.

Vultr Inference wraps limits as ``{"value": N, "unit": "token"}`` under
``input_modalities[].supported_inputs``. The extraction seam must unwrap them
instead of falling through to DEFAULT_FALLBACK_CONTEXT (256k): an undetected
1M window makes the 50% compressor threshold fire at 128k, four times the
intended compaction frequency.
"""

from unittest.mock import MagicMock, patch


def _streamed(response):
    """A ``model_metadata_http.stream`` result: a context manager yielding *response*."""
    ctx = MagicMock()
    ctx.__enter__.return_value = response
    return ctx


class TestQuantityObjectLimits:
    """``{"value": N, "unit": "token"}`` wrappers must unwrap at the extraction
    seam without weakening existing shapes."""

    def test_nested_quantity_object_resolves(self):
        from agent.model_metadata import _CONTEXT_LENGTH_KEYS, _extract_first_int

        payload = {
            "input_modalities": [
                {"type": "text", "supported_inputs": {"max_context_length": {"value": 1_048_576, "unit": "token"}}},
            ]
        }
        assert _extract_first_int(payload, _CONTEXT_LENGTH_KEYS) == 1_048_576

    def test_non_token_unit_is_rejected(self):
        from agent.model_metadata import _CONTEXT_LENGTH_KEYS, _extract_first_int

        payload = {"max_context_length": {"value": 20_000, "unit": "characters"}}
        assert _extract_first_int(payload, _CONTEXT_LENGTH_KEYS) is None

    def test_missing_unit_is_accepted(self):
        from agent.model_metadata import _CONTEXT_LENGTH_KEYS, _extract_first_int

        payload = {"max_context_length": {"value": 262_144}}
        assert _extract_first_int(payload, _CONTEXT_LENGTH_KEYS) == 262_144

    def test_parameter_schema_descriptor_is_not_misread(self):
        """``supported_parameters`` entries describe a parameter's shape, not a
        quantity instance — no ``value`` key, so nothing to unwrap."""
        from agent.model_metadata import _MAX_COMPLETION_KEYS, _extract_first_int

        payload = {
            "output_modalities": [
                {
                    "type": "text",
                    "max_length": {"value": 1_048_576, "unit": "token"},
                    "supported_parameters": {
                        "max_tokens": {"type": "integer", "min": 1, "max": 1_048_576, "unit": "token"},
                    },
                }
            ]
        }
        assert _extract_first_int(payload, _MAX_COMPLETION_KEYS) is None

    def test_flat_scalar_still_resolves(self):
        from agent.model_metadata import _CONTEXT_LENGTH_KEYS, _extract_first_int

        assert _extract_first_int({"context_length": 131_072}, _CONTEXT_LENGTH_KEYS) == 131_072


class TestEndpointQuantityObjectProbe:
    def setup_method(self):
        import agent.model_metadata as mm
        mm._endpoint_model_metadata_cache.clear()
        mm._endpoint_model_metadata_cache_time.clear()

    def test_vultr_quantity_object_window_resolves(self):
        """End-to-end through the endpoint probe: /models → cache entry →
        resolution. Pre-fix the scalar coercion saw a dict, the probe missed, and
        the model fell through to the 256k probe-down default."""
        import agent.model_metadata as mm

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "data": [
                {
                    "schema_version": "2.4",
                    "id": "glm-5.3",
                    "name": "GLM 5.3",
                    "input_modalities": [
                        {
                            "type": "text",
                            "supported_inputs": {
                                "max_context_length": {"value": 1_048_576, "unit": "token"},
                            },
                        },
                        {
                            "type": "image",
                            "supported_inputs": {"sources": {"type": "enum", "values": ["url", "base64"]}},
                        },
                    ],
                    "output_modalities": [
                        {
                            "type": "text",
                            "streaming": True,
                            "max_length": {"value": 1_048_576, "unit": "token"},
                            "supported_parameters": {
                                "max_tokens": {"type": "integer", "min": 1, "max": 1_048_576, "unit": "token"},
                            },
                        }
                    ],
                }
            ]
        }

        with patch(
            "agent.model_metadata.model_metadata_http.stream",
            return_value=_streamed(response),
        ) as mock_stream:
            base = "https://api.vultrinference.example/v1"
            metadata = mm.fetch_endpoint_model_metadata(base)

        assert metadata["glm-5.3"]["context_length"] == 1_048_576
        # The ``supported_parameters.max_tokens`` schema descriptor has no ``value``
        # key — it must not be misread as a completion limit.
        assert "max_completion_tokens" not in metadata["glm-5.3"]
        # Resolution goes through the in-memory memo: no second probe.
        assert mm._resolve_endpoint_context_length("glm-5.3", base) == 1_048_576
        mock_stream.assert_called_once()
