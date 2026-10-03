"""Unit tests for the Infersia provider profile's model discovery.

``fallback_models`` is empty, so ``fetch_models`` *is* the picker. The
catalogue served at ``/v1/models`` describes the whole account, so it also
carries models answering ``/v1/rerank``, ``/v1/audio/transcriptions`` and
``/v1/audio/speech``. Only the chat models may reach the picker.

The regression these tests exist for is the one-sided filter: speech-to-text
is ``audio->text``, so its ``output_modalities`` is ``["text"]`` and any test
written on the output side alone waves it through.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

import pytest

# Verbatim ``/v1/models`` entries, trimmed to the fields under test. Keeping
# the real shapes means a change to what the catalogue publishes shows up here
# as a failure rather than as a model appearing in someone's picker.
_CATALOGUE: list[dict[str, Any]] = [
    {
        "id": "deepseek/deepseek-v4-flash-0731",
        "architecture": {
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        },
    },
    {
        "id": "hexgrad/kokoro-82m",
        "architecture": {
            "modality": "text->audio",
            "input_modalities": ["text"],
            "output_modalities": ["audio"],
        },
    },
    {
        "id": "openai/whisper-large-v3-turbo",
        "architecture": {
            "modality": "audio->text",
            "input_modalities": ["audio"],
            "output_modalities": ["text"],
        },
    },
    {
        "id": "zeroentropy/zerank-2",
        "architecture": {
            "modality": "text->embedding",
            "input_modalities": ["text"],
            "output_modalities": ["embedding"],
        },
    },
    {
        "id": "qwen/qwen3.6-35b-a3b",
        "architecture": {
            "modality": "text+image->text",
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
    },
    {
        "id": "qwen/qwen3-8b",
        "architecture": {
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        },
    },
]


class _CatalogueHandler(BaseHTTPRequestHandler):
    """Serves ``/models`` with a configurable catalogue."""

    models: list = _CATALOGUE

    def do_GET(self):
        if self.path.rstrip("/") == "/models":
            body = json.dumps({"data": type(self).models}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress noise


def _serve(models):
    _CatalogueHandler.models = models
    server = HTTPServer(("127.0.0.1", 0), _CatalogueHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


@pytest.fixture
def infersia_profile():
    """Resolve the registered Infersia profile via the provider registry.

    Going through ``get_provider_profile`` keeps the test honest: if the
    registered object is ever swapped back for a plain ``ProviderProfile`` the
    filter disappears and these assertions collapse.
    """
    import model_tools  # noqa: F401  (importing triggers plugin discovery)
    import providers

    profile = providers.get_provider_profile("infersia")
    assert profile is not None, "infersia provider profile must be registered"
    return profile


def _fetch(profile, models):
    server, port = _serve(models)
    try:
        return profile.fetch_models(
            api_key="test-key", base_url=f"http://127.0.0.1:{port}"
        )
    finally:
        server.shutdown()


class TestInfersiaModelDiscovery:
    def test_only_chat_models_reach_the_picker(self, infersia_profile):
        assert _fetch(infersia_profile, _CATALOGUE) == [
            "deepseek/deepseek-v4-flash-0731",
            "qwen/qwen3.6-35b-a3b",
            "qwen/qwen3-8b",
        ]

    def test_speech_to_text_is_excluded_despite_text_output(self, infersia_profile):
        """The regression guard: ``audio->text`` outputs text but is not chat."""
        entry = [m for m in _CATALOGUE if m["id"] == "openai/whisper-large-v3-turbo"]
        assert entry[0]["architecture"]["output_modalities"] == ["text"]
        assert _fetch(infersia_profile, entry) == []

    @pytest.mark.parametrize(
        "model_id",
        ["hexgrad/kokoro-82m", "zeroentropy/zerank-2"],
    )
    def test_non_text_output_is_excluded(self, infersia_profile, model_id):
        entry = [m for m in _CATALOGUE if m["id"] == model_id]
        assert _fetch(infersia_profile, entry) == []

    def test_unknown_modality_is_excluded_by_default(self, infersia_profile):
        """The filter is an allow-list, so a modality it has never seen drops."""
        assert (
            _fetch(
                infersia_profile,
                [
                    {
                        "id": "example/video-model",
                        "architecture": {
                            "input_modalities": ["video"],
                            "output_modalities": ["video"],
                        },
                    }
                ],
            )
            == []
        )

    def test_default_aux_model_survives_the_filter(self, infersia_profile):
        """``default_aux_model`` is the one hardcoded id; it must still be live."""
        assert infersia_profile.default_aux_model in _fetch(
            infersia_profile, _CATALOGUE
        )

    def test_entry_without_architecture_is_kept(self, infersia_profile):
        """Absence is no information, and an empty picker explains nothing."""
        assert _fetch(infersia_profile, [{"id": "some/model"}]) == ["some/model"]

    def test_unreachable_endpoint_returns_none(self, infersia_profile):
        """None means "no catalogue", which is distinct from "no chat models"."""
        assert (
            infersia_profile.fetch_models(
                api_key="test-key", base_url="http://127.0.0.1:1", timeout=1.0
            )
            is None
        )
