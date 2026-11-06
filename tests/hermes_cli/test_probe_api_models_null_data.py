"""``/models`` catalogs that return ``"data": null`` must degrade to an empty list.

An Ollama daemon with no models pulled answers ``GET /v1/models`` with
``{"object": "list", "data": null}``. ``data.get("data", [])`` yields ``None`` when the key is
present with a null value (the default only covers a *missing* key), so
``probe_api_models`` raised ``TypeError: 'NoneType' object is not iterable``. Every caller that
probes a catalogue inherited the crash — the ``/model`` picker swallowed it and reported
"No authenticated providers found", which points the reader at credentials instead of the
endpoint.

Contract: null, missing or malformed ``data`` probes as an empty catalogue; never raises.
"""

import pytest

from hermes_cli import models


def _probe(payload):
    """``probe_api_models`` against a stubbed ``/models`` body."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(models, "_get_json", lambda *a, **k: payload)
        return models.probe_api_models("test-key", "http://localhost:11434/v1", timeout=1)


@pytest.mark.parametrize(
    "payload",
    [
        {"object": "list", "data": None},   # empty Ollama: the shape that crashed
        {},                                 # key missing entirely
        [],                                 # non-dict body
        {"data": "nope"},                   # data present but not a list
    ],
)
def test_probe_api_models_treats_malformed_catalog_as_empty(payload):
    assert _probe(payload).get("models") == []


def test_probe_api_models_ignores_non_dict_items():
    result = _probe({"data": [None, "bad", 7, {"id": "model-a"}]})
    assert result.get("models") == ["model-a"]


def test_probe_api_models_still_reports_ids():
    result = _probe({"data": [{"id": "model-a"}, {"id": "model-b"}]})
    assert result.get("models") == ["model-a", "model-b"]
