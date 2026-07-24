"""Teams pipeline model timestamps must tolerate malformed input.

Every model is built from external Graph API / webhook payloads and from
stored pipeline state via ``from_dict()``; each timestamp field is run through
``models._parse_datetime`` in ``__post_init__``. A malformed or non-ISO value
(a corrupt stored job, an unexpected Graph payload) previously raised an
uncaught ``ValueError`` from ``datetime.fromisoformat`` and crashed model
construction. It now degrades to ``None`` — like the empty value the function
already handles — matching the defensive timestamp parsers used elsewhere in
the codebase (skill_usage, curator, status, nous_account, web_server).
"""

from datetime import datetime, timezone

import pytest

from plugins.teams_pipeline.models import MeetingArtifact, _parse_datetime


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-date",
        "2026-13-45T00:00:00Z",   # invalid month/day
        "06/10/2026",             # non-ISO format
        "tomorrow",
        "+00:00",                 # offset only, no date
        "2026-06-10T25:00:00Z",   # invalid hour
    ],
)
def test_parse_datetime_returns_none_on_malformed(bad):
    # Must not raise ValueError out of __post_init__.
    assert _parse_datetime(bad) is None


def test_parse_datetime_still_parses_valid_iso_and_z():
    assert _parse_datetime("2026-06-10T12:00:00Z") == datetime(
        2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc
    )
    # 7-digit fractional seconds (Microsoft Graph style) parse on 3.11+.
    assert _parse_datetime("2026-06-10T12:00:00.0000000Z") is not None
    # A naive timestamp is assumed UTC.
    assert _parse_datetime("2026-06-10T12:00:00").tzinfo == timezone.utc


def test_parse_datetime_passthrough_and_empty():
    now = datetime.now(timezone.utc)
    assert _parse_datetime(now) is now      # already a datetime → unchanged
    assert _parse_datetime(None) is None
    assert _parse_datetime("   ") is None    # empty after strip


def test_meeting_artifact_from_dict_survives_malformed_timestamp():
    # A stored/replayed artifact with a corrupt createdDateTime must
    # deserialize instead of raising ValueError out of __post_init__.
    art = MeetingArtifact.from_dict(
        {
            "artifact_type": "transcript",
            "artifact_id": "a-1",
            "createdDateTime": "not-a-real-timestamp",
            "lastModifiedDateTime": "2026-06-10T12:00:00Z",
        }
    )
    assert art.created_at is None
    assert art.available_at == datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
