"""Regression tests for #135837 — local llama.cpp VLMs behind a custom OpenAI-compatible
provider (``llama-server --mmproj``, absent from models.dev) fell through every capability
probe, so ``computer_use`` screenshots were fail-closed to ``auxiliary.vision`` prose and
the model had to guess click coordinates from a description. The llama.cpp ``/props``
``modalities`` field is the server's own attestation; the new probe closes the gap after
the Ollama probe, under the same LOCAL-only boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.image_routing import _lookup_supports_vision, decide_image_input_mode

# A custom OpenAI-compatible provider pointing at a local llama-server.
CFG = {"model": {"provider": "custom", "base_url": "http://127.0.0.1:8080/v1"}}


class TestLlamaCppVisionRouting:
    """End-to-end: props modalities flows through _lookup_supports_vision
    → decide_image_input_mode."""

    def test_props_vision_true_routes_native(self):
        with patch("agent.models_dev.get_model_capabilities", return_value=None), \
             patch("agent.image_routing._should_probe_ollama_vision", return_value=False), \
             patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata_llamacpp.query_llamacpp_supports_vision", return_value=True) as q:
            assert _lookup_supports_vision("custom", "qwen3-vl-local", CFG) is True
            assert decide_image_input_mode("custom", "qwen3-vl-local", CFG) == "native"
            # decide_image_input_mode re-runs the lookup, so the probe fires once per call
            # with the resolved base_url and no credential leak from an empty config.
            q.assert_called_with("http://127.0.0.1:8080/v1", api_key="")

    def test_props_vision_false_stays_text(self):
        with patch("agent.models_dev.get_model_capabilities", return_value=None), \
             patch("agent.image_routing._should_probe_ollama_vision", return_value=False), \
             patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata_llamacpp.query_llamacpp_supports_vision", return_value=False):
            assert _lookup_supports_vision("custom", "qwen3-vl-local", CFG) is False
            assert decide_image_input_mode("custom", "qwen3-vl-local", CFG) == "text"

    def test_props_unknown_stays_text_fail_closed(self):
        # A build predating the modalities field returns None — aux routing stays the default.
        with patch("agent.models_dev.get_model_capabilities", return_value=None), \
             patch("agent.image_routing._should_probe_ollama_vision", return_value=False), \
             patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata_llamacpp.query_llamacpp_supports_vision", return_value=None):
            assert _lookup_supports_vision("custom", "qwen3-vl-local", CFG) is None
            assert decide_image_input_mode("custom", "qwen3-vl-local", CFG) == "text"

    def test_remote_endpoint_never_probed(self):
        remote_cfg = {"model": {"provider": "custom", "base_url": "https://api.example.com/v1"}}
        with patch("agent.models_dev.get_model_capabilities", return_value=None), \
             patch("agent.image_routing._should_probe_ollama_vision", return_value=False), \
             patch("agent.model_metadata.is_local_endpoint", return_value=False), \
             patch("agent.model_metadata_llamacpp.query_llamacpp_supports_vision") as q:
            assert _lookup_supports_vision("custom", "some-model", remote_cfg) is None
            q.assert_not_called()

    def test_computer_use_capture_gate_opens(self):
        # _native_tool_result_images is THE gate _capture_response consults (#24015): with the
        # props verdict True the screenshot lane turns native end-to-end.
        from tools.vision_tools import _native_tool_result_images

        with patch("agent.models_dev.get_model_capabilities", return_value=None), \
             patch("agent.image_routing._should_probe_ollama_vision", return_value=False), \
             patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata_llamacpp.query_llamacpp_supports_vision", return_value=True):
            assert _native_tool_result_images("custom", "qwen3-vl-local", CFG) is True


def _props_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def _client_with(get_responses):
    client = MagicMock()
    client.get.side_effect = list(get_responses)
    client.__enter__.return_value = client
    return client


class TestQueryLlamaCppSupportsVision:
    """Unit tests for the /props parser in agent.model_metadata_llamacpp."""

    def _query(self, payload):
        from agent.model_metadata_llamacpp import query_llamacpp_supports_vision

        client = _client_with([_props_response(payload)])
        with patch("agent.model_metadata_llamacpp._is_llamacpp_server", return_value=True), \
             patch("agent.model_metadata._endpoint_blackholed", return_value=False), \
             patch("httpx.Client", return_value=client):
            return query_llamacpp_supports_vision("http://127.0.0.1:8080/v1")

    def test_modality_dict_true(self):
        props = {"default_generation_settings": {}, "modalities": {"vision": True}}
        assert self._query(props) is True

    def test_modality_dict_false(self):
        props = {"default_generation_settings": {}, "modalities": {"vision": False}}
        assert self._query(props) is False

    def test_modality_list_true(self):
        # Older builds emitted a plain list of modality names.
        assert self._query({"modalities": ["vision"]}) is True

    def test_modality_list_without_vision(self):
        assert self._query({"modalities": ["audio"]}) is False

    def test_missing_modalities_is_unknown(self):
        # Pre-modalities build: stay fail-closed (None), never guess.
        assert self._query({"default_generation_settings": {}}) is None

    def test_non_bool_vision_value_is_unknown(self):
        assert self._query({"modalities": {"vision": "yes"}}) is None

    def test_props_fallback_after_v1_404(self):
        # Older llama.cpp builds serve /props without the /v1 prefix.
        from agent.model_metadata_llamacpp import query_llamacpp_supports_vision

        not_found = MagicMock(status_code=404)
        ok = _props_response({"modalities": {"vision": True}})
        client = _client_with([not_found, ok])
        with patch("agent.model_metadata_llamacpp._is_llamacpp_server", return_value=True), \
             patch("agent.model_metadata._endpoint_blackholed", return_value=False), \
             patch("httpx.Client", return_value=client):
            assert query_llamacpp_supports_vision("http://127.0.0.1:8080/v1") is True
        assert [c.args[0] for c in client.get.call_args_list] == [
            "http://127.0.0.1:8080/v1/props", "http://127.0.0.1:8080/props"]

    def test_not_llamacpp_server_returns_none_without_request(self):
        from agent.model_metadata_llamacpp import query_llamacpp_supports_vision

        with patch("agent.model_metadata_llamacpp._is_llamacpp_server", return_value=False), \
             patch("httpx.Client") as client_cls:
            assert query_llamacpp_supports_vision("http://127.0.0.1:8080/v1") is None
            client_cls.assert_not_called()
