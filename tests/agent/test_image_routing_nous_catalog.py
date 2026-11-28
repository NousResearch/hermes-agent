"""Image routing for models served through Nous Portal.

models.dev has no Nous catalog, so the capability lookup answered "unknown" for
every Portal model and ``image_input_mode: auto`` described each attachment
through the auxiliary vision model, even for a chat model the Portal itself
lists with image input. The Portal's ``/v1/models`` is the authority for what
its own routes accept; these tests drive the real lookup against a served
catalog.
"""

import json
import threading

import pytest

import hermes_cli.models as models_mod
from agent.image_routing import _lookup_supports_vision, decide_image_input_mode
from hermes_cli import models_reasoning_caps


SEES = "vendor/sees-images-0101"
BLIND = "vendor/text-only"
SILENT = "vendor/no-architecture"

_CATALOG = json.dumps({
    "data": [
        {
            "id": SEES,
            "supported_parameters": ["reasoning", "tools"],
            "architecture": {"input_modalities": ["text", "image"]},
        },
        {
            "id": BLIND,
            "supported_parameters": ["tools"],
            "architecture": {"input_modalities": ["text"]},
        },
        {"id": SILENT, "supported_parameters": ["tools"]},
    ]
}).encode()

STAGING = "https://staging-portal.example.test"
# The same model id, served text-only by another Portal endpoint.
_TEXT_ONLY_CATALOG = json.dumps({
    "data": [{"id": SEES, "supported_parameters": ["tools"], "architecture": {"input_modalities": ["text"]}}]
}).encode()


class _Portal:
    """Stands in for the Portal's ``/v1/models`` and counts the reads."""

    def __init__(self, monkeypatch):
        self.fetches = 0
        self.online = True
        self.body = _CATALOG
        self.reading = threading.Event()
        self.release = threading.Event()
        self.release.set()
        monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", self._open)

    def _open(self, req, *, timeout):
        self.fetches += 1
        self.reading.set()
        self.release.wait(10)
        if not self.online:
            raise OSError("portal unreachable")
        body = _TEXT_ONLY_CATALOG if req.full_url.startswith(STAGING) else self.body

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return body

        return _Resp()


@pytest.fixture
def cold_process(monkeypatch):
    """Put the catalog caches in the state of a freshly started process."""

    def _reset():
        for name in ("_nous_reasoning_caps_cache", "_nous_reasoning_caps_failed_at"):
            monkeypatch.setattr(models_mod, name, None)
        monkeypatch.setattr(models_reasoning_caps, "_CATALOG_URL_BY_HOME", {}, raising=False)
        for name in ("_nous_caps_disk_checked", "_nous_caps_warm_started", "_nous_caps_vision_refreshed"):
            monkeypatch.setattr(models_mod, name, False, raising=False)

    _reset()
    return _reset


@pytest.fixture
def portal(cold_process, monkeypatch):
    return _Portal(monkeypatch)


def test_portal_catalog_decides_auto_image_routing(portal, monkeypatch):
    """Image-capable Portal model gets the image; text-only and unstated ones are described;
    other providers and other endpoints are not answered from it; a user override still wins."""
    assert decide_image_input_mode("nous", SEES, {}) == "native"
    assert decide_image_input_mode("nous", BLIND, {}) == "text"
    assert _lookup_supports_vision("nous", SILENT, {}) is None
    assert _lookup_supports_vision("custom-gateway", SEES, {}) is None
    other = {"model": {"base_url": "https://elsewhere.example.test/v1"}}
    assert _lookup_supports_vision("nous", SEES, other) is None
    assert portal.fetches == 1
    assert decide_image_input_mode("nous", SEES, {"model": {"supports_vision": False}}) == "text"

    # Same process and profile, Portal A -> B -> A, one model id B serves text-only: each
    # endpoint's own catalog decides, never the one cached for the other.
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", f"{STAGING}/v1")
    assert decide_image_input_mode("nous", SEES, {}) == "text"
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL")
    assert decide_image_input_mode("nous", SEES, {}) == "native"


def test_mirror_written_before_modalities_were_recorded_is_refreshed(portal, cold_process):
    """An install that upgrades already holds a mirror with reasoning data only.

    Serving it as-is would keep every Portal model "unknown" until the mirror's
    TTL, or for the whole life of a gateway that hydrated it at boot. Reasoning
    answers from the same catalog keep working after the refresh.
    """
    url = models_reasoning_caps.nous_catalog_url()
    models_reasoning_caps._save_reasoning_caps_disk(
        url, {SEES: {"supports_reasoning": True, "mandatory": False}, BLIND: {"supports_reasoning": False}}
    )
    cold_process()

    assert decide_image_input_mode("nous", SEES, {}) == "native"
    assert _lookup_supports_vision("nous", BLIND, {}) is False
    assert portal.fetches == 1
    reasoning = models_reasoning_caps.nous_model_reasoning_capabilities(SEES)
    assert reasoning is not None and reasoning["supports_reasoning"] is True
