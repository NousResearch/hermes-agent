"""``probe_api_models()`` must return alphabetically sorted, non-empty IDs.

Custom OpenAI-compatible endpoints (``custom_providers`` rows, bare
``provider: custom``) previously returned the provider's raw ``/v1/models``
response order unchanged, so the ``/model`` picker showed a different,
non-alphabetical ordering every time the cache refreshed or a new session
re-fetched. Built-in providers (anthropic / xai / github) already sort their
catalogs; these tests pin the same stable ordering on the generic custom
path.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from hermes_cli.models import probe_api_models


class _Resp:
    """Minimal file-like context manager matching the ``with ... as resp``
    seam inside ``probe_api_models``."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload


def _payload(models: list[dict]) -> bytes:
    return json.dumps({"data": models}).encode()


def test_probe_api_models_sorts_ids_case_insensitively():
    """Unordered, mixed-case IDs come back lowercase-alphabetical."""
    unordered = [
        {"id": "zulu-model"},
        {"id": "Alpha-model"},
        {"id": "mike-model"},
        {"id": "bravo-model"},
    ]
    with patch(
        "hermes_cli.models._urlopen_model_catalog_request",
        return_value=_Resp(_payload(unordered)),
    ):
        result = probe_api_models("sk-key", "https://gw.example.com/v1")

    assert result["models"] == [
        "Alpha-model",
        "bravo-model",
        "mike-model",
        "zulu-model",
    ]


def test_probe_api_models_filters_empty_ids():
    """Empty (falsy) IDs from the provider are dropped before sorting."""
    unordered = [
        {"id": "zulu-model"},
        {"id": ""},
        {"id": "alpha-model"},
        {"id": None},
        {"id": "bravo-model"},
    ]
    with patch(
        "hermes_cli.models._urlopen_model_catalog_request",
        return_value=_Resp(_payload(unordered)),
    ):
        result = probe_api_models("sk-key", "https://gw.example.com/v1")

    assert result["models"] == ["alpha-model", "bravo-model", "zulu-model"]


def test_probe_api_models_sort_is_stable_on_already_sorted_input():
    """An already-sorted response is unchanged (sort is idempotent)."""
    ordered = [
        {"id": "alpha-model"},
        {"id": "bravo-model"},
        {"id": "charlie-model"},
    ]
    with patch(
        "hermes_cli.models._urlopen_model_catalog_request",
        return_value=_Resp(_payload(ordered)),
    ):
        result = probe_api_models("sk-key", "https://gw.example.com/v1")

    assert result["models"] == ["alpha-model", "bravo-model", "charlie-model"]
