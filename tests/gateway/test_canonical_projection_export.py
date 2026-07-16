from __future__ import annotations

import copy

import pytest

from gateway.canonical_projection_export import (
    PROJECTION_EXPORT_SCHEMA,
    ProjectionExportError,
    validate_projection_export,
)


EVENT_ID = "11111111-1111-4111-8111-111111111111"
OCCURRED_AT = "2026-07-15T10:00:00+00:00"


def _export() -> dict:
    event = {
        "event_id": EVENT_ID,
        "occurred_at": OCCURRED_AT,
        "source": {
            "observed_session": {
                "request_id": "projection-export-one",
                "platform": "discord",
                "thread_id": "thread-one",
            }
        },
        "decision": {"decided_by": "model_event_append"},
        "payload": {"canonical_content_sha256": "a" * 64},
    }
    return {
        "schema": PROJECTION_EXPORT_SCHEMA,
        "events": [event],
        "provenance": [
            {
                "event_id": EVENT_ID,
                "canonical_content_sha256": "a" * 64,
                "origin": "model_event_append",
                "trusted_runtime": {
                    "request_id": "projection-export-one",
                    "platform": "discord",
                    "thread_id": "thread-one",
                },
                "appended_at": OCCURRED_AT,
            }
        ],
    }


def test_v2_export_requires_exact_parallel_database_provenance() -> None:
    events, provenance = validate_projection_export(_export(), maximum_events=10)

    assert events[0]["event_id"] == EVENT_ID
    assert provenance[0]["event_id"] == EVENT_ID


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda value: value.update(schema="canonical-writer-projection-export.v1"),
            "projection_export_envelope_invalid",
        ),
        (
            lambda value: value["provenance"].append(
                copy.deepcopy(value["provenance"][0])
            ),
            "projection_export_rows_invalid",
        ),
        (
            lambda value: value["provenance"][0].update(
                canonical_content_sha256="b" * 64
            ),
            "projection_export_provenance_content_mismatch",
        ),
        (
            lambda value: value["provenance"][0]["trusted_runtime"].update(
                thread_id="substituted"
            ),
            "projection_export_provenance_runtime_mismatch",
        ),
        (
            lambda value: (
                value["provenance"][0]["trusted_runtime"].update(
                    api_key="must-not-export"
                ),
                value["events"][0]["source"]["observed_session"].update(
                    api_key="must-not-export"
                ),
            ),
            "projection_export_provenance_runtime_invalid",
        ),
        (
            lambda value: value["provenance"][0].update(
                origin="routeback_finalize_sent"
            ),
            "projection_export_provenance_origin_mismatch",
        ),
        (
            lambda value: value["provenance"][0].update(
                appended_at="2026-07-15T10:00:01+00:00"
            ),
            "projection_export_provenance_time_mismatch",
        ),
    ],
)
def test_v2_export_fails_closed_on_envelope_or_join_tamper(mutate, error) -> None:
    value = _export()
    mutate(value)

    with pytest.raises(ProjectionExportError, match=error):
        validate_projection_export(value, maximum_events=10)


def test_v2_export_rejects_duplicate_event_identity() -> None:
    value = _export()
    value["events"].append(copy.deepcopy(value["events"][0]))
    value["provenance"].append(copy.deepcopy(value["provenance"][0]))

    with pytest.raises(ProjectionExportError, match="projection_export_event_duplicate"):
        validate_projection_export(value, maximum_events=10)


@pytest.mark.parametrize("maximum_events", [True, 10.0, "10", None, -1, 1_000_001])
def test_v2_export_rejects_invalid_maximum_events(maximum_events) -> None:
    with pytest.raises(ProjectionExportError, match="projection_export_rows_invalid"):
        validate_projection_export(_export(), maximum_events=maximum_events)
