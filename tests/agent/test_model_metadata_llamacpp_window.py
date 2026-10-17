"""The window llama.cpp actually enforces behind a gateway: /props reads and overflow wording.

The models payload advertises the model's native length while the loader may have shrunk the
allocated window. Unsloth Studio is the live case: it brands ``owned_by`` as "unsloth-studio"
and answers /props with the allocated n_ctx. These tests pin both halves — taking the /props
value for a local endpoint without the llama.cpp brand, and parsing the window quoted in the
gateway's overflow error — plus the output-cap guard on the new wording tolerance.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.model_metadata import parse_context_limit_from_error


def _streamed(response):
    """A ``model_metadata_http.stream`` result: a context manager yielding *response*."""
    ctx = MagicMock()
    ctx.__enter__.return_value = response
    return ctx


class TestGatewayOverflowWindow:
    @pytest.mark.parametrize("msg,expected", [
        ("Message too long: 22589 tokens exceeds the 15872-token context window. "
         "Try increasing the Context Length in Model settings, or shorten the conversation.",
         15872),
        ("Message too long: 25496 tokens exceeds the 24832-token context window. "
         "Try increasing the Context Length in Model settings, or shorten the conversation.",
         24832),
    ])
    def test_llamacpp_gateway_window_variants(self, msg, expected):
        """llama.cpp behind a gateway quotes the allocated window hyphenated, with the offending
        input count first. Missing this shape kept the advertised native window (262144) in the
        compressor, which then could not shrink a session to the limit the server enforced."""
        assert parse_context_limit_from_error(msg) == expected

    def test_llamacpp_gateway_window_recalibrates(self):
        from agent.model_metadata import get_context_length_from_provider_error

        msg = "Message too long: 22589 tokens exceeds the 15872-token context window."
        assert get_context_length_from_provider_error(msg, 262144) == 15872
        # Parsed limit not below the current window → no recalibration.
        assert get_context_length_from_provider_error(msg, 15872) is None

    def test_hyphenated_output_cap_is_not_a_context_limit(self):
        """Hyphen tolerance stays anchored on "context"/"window": an "<N>-token limit"
        phrase is an output cap and must never be cached as the context window."""
        assert parse_context_limit_from_error("requested 8192-token limit for this turn") is None


class TestLocalGatewayPropsWindow:
    def test_local_gateway_without_llamacpp_brand_takes_props_window(self, monkeypatch):
        """A local llama.cpp-speaking gateway that brands ``owned_by`` differently (Unsloth
        Studio reports "unsloth-studio") must still take the /props window. The models payload
        carries the model's native length (262144) while /props reports the n_ctx the loader
        actually allocated; believing the payload made Hermes plan for a window 16x larger than
        the server enforced, so sessions died on "Message too long" while compression could not
        reach the real limit."""
        import agent.model_metadata as mm

        models = MagicMock()
        models.status_code = 200
        models.json.return_value = {"data": [{
            "id": "Qwen3.8-27B-OBLITERATED-tools-Q8_0",
            "object": "model",
            "owned_by": "unsloth-studio",
            "context_length": 262_144,
            "max_context_length": 262_144,
        }]}
        props = MagicMock()
        props.is_error = False
        props.json.return_value = {
            "default_generation_settings": {"n_ctx": 15_872},
            "model_alias": "Qwen3.8-27B-OBLITERATED-tools-Q8_0",
        }

        monkeypatch.setattr(mm, "detect_local_server_type", lambda *a, **k: None)
        with patch("agent.model_metadata.model_metadata_http.stream", return_value=_streamed(models)), \
                patch("agent.model_metadata.model_metadata_http.get", return_value=props) as mock_get:
            result = mm.fetch_endpoint_model_metadata(
                "http://127.0.0.1:8899/v1", api_key="k", force_refresh=True
            )

        assert result["Qwen3.8-27B-OBLITERATED-tools-Q8_0"]["context_length"] == 15_872
        assert mock_get.call_args.args[0] == "http://127.0.0.1:8899/v1/props"

    def test_remote_provider_without_llamacpp_brand_is_not_props_probed(self):
        """Scoping guard: the /props read stays local — a remote OpenAI-compatible provider
        must not pay two extra probes, and its payload value stands."""
        import agent.model_metadata as mm

        models = MagicMock()
        models.status_code = 200
        models.json.return_value = {"data": [{
            "id": "test/remote-model",
            "owned_by": "some-provider",
            "context_length": 262_144,
        }]}

        with patch("agent.model_metadata.model_metadata_http.stream", return_value=_streamed(models)), \
                patch("agent.model_metadata.model_metadata_http.get") as mock_get:
            result = mm.fetch_endpoint_model_metadata(
                "https://remote-provider.example/v1", api_key="k", force_refresh=True
            )

        assert result["test/remote-model"]["context_length"] == 262_144
        mock_get.assert_not_called()
