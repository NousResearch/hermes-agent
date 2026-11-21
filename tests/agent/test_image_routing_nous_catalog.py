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
from concurrent.futures import ThreadPoolExecutor

import pytest

import hermes_cli.models as models_mod
from agent.image_routing import _lookup_supports_vision, decide_image_input_mode
from hermes_cli import models_reasoning_caps


SEES = "vendor/sees-images-0101"
BLIND = "vendor/text-only"
SILENT = "vendor/no-architecture"
VISION_ONLY = "vendor/modalities-only"

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
        {"id": VISION_ONLY, "architecture": {"input_modalities": ["text", "image"]}},
    ]
}).encode()

_CATALOG_WITHOUT_MODALITIES = json.dumps({
    "data": [{"id": SEES, "supported_parameters": ["reasoning"]}]
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
        body = self.body

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
        for name in ("_nous_caps_disk_checked", "_nous_caps_warm_started", "_nous_caps_vision_refreshed"):
            monkeypatch.setattr(models_mod, name, False)

    _reset()
    return _reset


@pytest.fixture
def portal(cold_process, monkeypatch):
    return _Portal(monkeypatch)


def test_auto_attaches_images_to_a_portal_model_that_lists_image_input(portal):
    assert decide_image_input_mode("nous", SEES, {}) == "native"


def test_auto_describes_images_for_a_portal_model_that_lists_text_only(portal):
    assert _lookup_supports_vision("nous", BLIND, {}) is False
    assert decide_image_input_mode("nous", BLIND, {}) == "text"


def test_models_the_catalog_says_nothing_about_stay_unknown(portal):
    assert _lookup_supports_vision("nous", SILENT, {}) is None
    assert _lookup_supports_vision("nous", "vendor/not-listed", {}) is None


def test_image_input_does_not_depend_on_the_entry_stating_reasoning_support(portal):
    assert decide_image_input_mode("nous", VISION_ONLY, {}) == "native"
    # That entry still has no reasoning verdict for the consumers that ask for one.
    assert models_reasoning_caps.nous_model_reasoning_capabilities(VISION_ONLY) is None
    reasoning = models_reasoning_caps.nous_model_reasoning_capabilities(SEES)
    assert reasoning is not None
    assert reasoning["supports_reasoning"] is True


def test_catalog_without_modalities_is_read_once_not_per_image(portal):
    portal.body = _CATALOG_WITHOUT_MODALITIES
    for _ in range(3):
        assert decide_image_input_mode("nous", SEES, {}) == "text"
    assert portal.fetches == 1


def test_old_mirror_is_read_once_when_the_portal_lists_no_modalities(portal, cold_process):
    url = models_reasoning_caps.nous_catalog_url()
    models_reasoning_caps._save_reasoning_caps_disk(url, {SEES: {"supports_reasoning": True}})
    cold_process()
    portal.body = _CATALOG_WITHOUT_MODALITIES

    for _ in range(3):
        assert _lookup_supports_vision("nous", SEES, {}) is None
    assert portal.fetches == 1


def test_images_arriving_together_share_one_catalog_read(portal, cold_process, monkeypatch):
    """Two attachments on a mirror without modalities: the second waits for the first's read."""
    url = models_reasoning_caps.nous_catalog_url()
    models_reasoning_caps._save_reasoning_caps_disk(url, {SEES: {"supports_reasoning": True}})
    cold_process()
    portal.release.clear()
    second_checked_cache = threading.Event()
    records_vision = models_reasoning_caps._records_vision

    def observe_cache_check(caps):
        result = records_vision(caps)
        if portal.reading.is_set():
            second_checked_cache.set()
        return result

    monkeypatch.setattr(models_reasoning_caps, "_records_vision", observe_cache_check)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_lookup_supports_vision, "nous", SEES, {})
        try:
            assert portal.reading.wait(10)
            second = pool.submit(_lookup_supports_vision, "nous", SEES, {})
            assert second_checked_cache.wait(10)
        finally:
            portal.release.set()
        assert first.result(timeout=10) is True
        assert second.result(timeout=10) is True
    assert portal.fetches == 1


def test_user_override_still_wins_over_the_catalog(portal):
    cfg = {"model": {"supports_vision": False}}
    assert decide_image_input_mode("nous", SEES, cfg) == "text"
    assert portal.fetches == 0


def test_other_providers_never_read_the_portal_catalog(portal):
    assert _lookup_supports_vision("custom-gateway", SEES, {}) is None
    assert portal.fetches == 0


def test_one_catalog_read_answers_later_images_and_the_next_process(portal, cold_process):
    assert decide_image_input_mode("nous", SEES, {}) == "native"
    assert decide_image_input_mode("nous", BLIND, {}) == "text"
    assert portal.fetches == 1

    cold_process()
    portal.online = False
    assert decide_image_input_mode("nous", SEES, {}) == "native"
    assert portal.fetches == 1


def test_unreachable_portal_reads_as_unknown(portal):
    portal.online = False
    assert _lookup_supports_vision("nous", SEES, {}) is None
    assert decide_image_input_mode("nous", SEES, {}) == "text"


def test_mirror_written_before_modalities_were_recorded_is_refreshed(portal, cold_process):
    """An install that upgrades already holds a mirror with reasoning data only.

    Serving it as-is would keep every Portal model "unknown" until the mirror's
    TTL, or for the whole life of a gateway that hydrated it at boot.
    """
    url = models_reasoning_caps.nous_catalog_url()
    models_reasoning_caps._save_reasoning_caps_disk(
        url, {SEES: {"supports_reasoning": True, "mandatory": False}, BLIND: {"supports_reasoning": False}}
    )
    cold_process()

    assert decide_image_input_mode("nous", SEES, {}) == "native"
    assert portal.fetches == 1
    assert _lookup_supports_vision("nous", BLIND, {}) is False
    assert portal.fetches == 1


def test_old_mirror_keeps_answering_reasoning_when_the_refresh_fails(portal, cold_process):
    url = models_reasoning_caps.nous_catalog_url()
    models_reasoning_caps._save_reasoning_caps_disk(
        url, {SEES: {"supports_reasoning": True, "mandatory": False}}
    )
    cold_process()
    portal.online = False

    assert _lookup_supports_vision("nous", SEES, {}) is None
    reasoning = models_reasoning_caps.nous_model_reasoning_capabilities(SEES)
    assert reasoning is not None
    assert reasoning["supports_reasoning"] is True
