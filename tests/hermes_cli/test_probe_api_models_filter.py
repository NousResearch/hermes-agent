"""Tests for probe_api_models() catalog filtering (#68536).

ARK-style providers (e.g. Volcengine) keep returning decommissioned models
from ``/models`` with ``status: "Shutdown"`` / ``"Retiring"``. The probe used
to map ``data[].id`` straight through, flooding the ``/model`` picker with
unusable entries (and surfacing malformed rows as ``""``).
"""

import io
import json
from unittest.mock import patch

from hermes_cli.models import _is_model_entry_available, probe_api_models


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _probe_with_catalog(entries):
    payload = json.dumps({"data": entries}).encode()
    with patch(
        "hermes_cli.models._urlopen_model_catalog_request",
        return_value=_FakeResponse(payload),
    ):
        return probe_api_models("sk-test", "https://ark.example.com/api/v3")


def test_shutdown_and_retiring_models_are_filtered():
    result = _probe_with_catalog(
        [
            {"id": "doubao-pro", "status": "Active"},
            {"id": "old-model-1", "status": "Shutdown"},
            {"id": "old-model-2", "status": "Retiring"},
            {"id": "old-model-3", "status": "Retired"},
        ]
    )
    assert result["models"] == ["doubao-pro"]


def test_status_matching_is_case_insensitive():
    result = _probe_with_catalog(
        [
            {"id": "gone-1", "status": "SHUTDOWN"},
            {"id": "gone-2", "status": "retiring "},
            {"id": "kept", "status": "active"},
        ]
    )
    assert result["models"] == ["kept"]


def test_entries_without_status_pass_through():
    """OpenAI-style catalogs carry no status field — never filter those."""
    result = _probe_with_catalog([{"id": "gpt-5.5"}, {"id": "gpt-5.5-mini"}])
    assert result["models"] == ["gpt-5.5", "gpt-5.5-mini"]


def test_unknown_status_values_pass_through():
    """The filter must never hide a working model on an unrecognized tag."""
    result = _probe_with_catalog(
        [{"id": "beta-model", "status": "Preview"}, {"id": "kept", "status": ""}]
    )
    assert result["models"] == ["beta-model", "kept"]


def test_malformed_entries_are_dropped():
    """Missing/empty ids used to surface as '' in the picker."""
    result = _probe_with_catalog(
        [{"status": "Active"}, {"id": ""}, "not-a-dict", None, {"id": "kept"}]
    )
    assert result["models"] == ["kept"]


def test_predicate_directly():
    assert _is_model_entry_available({"id": "m"}) is True
    assert _is_model_entry_available({"id": "m", "status": None}) is True
    assert _is_model_entry_available({"id": "m", "status": "Shutdown"}) is False
    assert _is_model_entry_available({"id": ""}) is False
    assert _is_model_entry_available({}) is False
    assert _is_model_entry_available(["id"]) is False
