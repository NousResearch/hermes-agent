"""Tests for the scheduler-side briefing-output persistence hook
(Artemis S-0511-07 § Architecture).

The hook fires after _deliver_result succeeds for an artemis-briefing job,
writing the exact delivered text to <HERMES_HOME>/artemis/<user_id>/
briefings/<UTC-ISO>.json. The format matches the artemis MCP tool
`save_briefing_output`'s storage shape so Coach's `get_recent_briefings`
reads either source uniformly.

Failure handling: persistence must never raise — any error logs at WARN
and the job's run record still flips to success.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cron.scheduler import (
    _is_briefing_job,
    _persist_briefing_output,
)


SAMPLE_BRIEFING_TEXT = """Looking at backend roles across the Bay — five matches today.

```
📌 Follow-ups
───────────
⭐ TODAY  Share your resume   ← see Coach's Take below
```

💬 **Coach's Take:** Drop your resume today.
"""


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point get_hermes_home() at a tmp dir for isolation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Pre-create the slack_channel sidecar so _resolve_origin can backfill
    # user_id from chat_id if a test job omits origin.user_id.
    yield tmp_path


def _briefing_job(user_id: str = "U0FIXTURE01", chat_id: str = "D0CHANNEL01") -> dict:
    """Minimal cron job dict shaped like a daily-briefing entry."""
    return {
        "id": "test-briefing-job",
        "name": "daily-briefing",
        "skills": ["artemis-briefing"],
        "origin": {
            "platform": "slack",
            "chat_id": chat_id,
            "user_id": user_id,
        },
    }


def _non_briefing_job() -> dict:
    return {
        "id": "test-other-job",
        "name": "some-other-cron",
        "skills": ["other-skill"],
        "origin": {
            "platform": "slack",
            "chat_id": "D0OTHER",
            "user_id": "U0FIXTURE01",
        },
    }


# -----------------------------------------------------------------------------
# _is_briefing_job — routing predicate
# -----------------------------------------------------------------------------

def test_is_briefing_job_true_for_artemis_briefing_skill():
    assert _is_briefing_job(_briefing_job()) is True


def test_is_briefing_job_false_for_other_skill():
    assert _is_briefing_job(_non_briefing_job()) is False


def test_is_briefing_job_false_when_skills_missing():
    job = _briefing_job()
    del job["skills"]
    assert _is_briefing_job(job) is False


def test_is_briefing_job_false_when_skills_empty():
    job = _briefing_job()
    job["skills"] = []
    assert _is_briefing_job(job) is False


# -----------------------------------------------------------------------------
# _persist_briefing_output — happy path
# -----------------------------------------------------------------------------

def test_persist_writes_three_field_entry(hermes_home):
    _persist_briefing_output(_briefing_job(), SAMPLE_BRIEFING_TEXT)

    briefings_dir = hermes_home / "artemis" / "U0FIXTURE01" / "briefings"
    files = list(briefings_dir.glob("*.json"))
    assert len(files) == 1

    entry = json.loads(files[0].read_text(encoding="utf-8"))
    assert set(entry.keys()) == {"user_id", "briefing_timestamp", "formatted_output"}
    assert entry["user_id"] == "U0FIXTURE01"
    assert entry["formatted_output"] == SAMPLE_BRIEFING_TEXT
    # ISO-8601 UTC with trailing Z, no fractional seconds
    assert entry["briefing_timestamp"].endswith("Z")
    assert "." not in entry["briefing_timestamp"]


def test_persist_filename_matches_timestamp(hermes_home):
    _persist_briefing_output(_briefing_job(), SAMPLE_BRIEFING_TEXT)
    files = list((hermes_home / "artemis" / "U0FIXTURE01" / "briefings").glob("*.json"))
    entry = json.loads(files[0].read_text(encoding="utf-8"))
    # Filename = timestamp with colons → hyphens, plus .json
    expected = entry["briefing_timestamp"].replace(":", "-") + ".json"
    assert files[0].name == expected


def test_persist_creates_briefings_dir_if_missing(hermes_home):
    target_dir = hermes_home / "artemis" / "U0FIXTURE01" / "briefings"
    assert not target_dir.exists()
    _persist_briefing_output(_briefing_job(), SAMPLE_BRIEFING_TEXT)
    assert target_dir.exists()
    assert any(target_dir.glob("*.json"))


def test_persist_guard_fallback_text_verbatim(hermes_home):
    """When the scheduler substitutes the guard fallback, the persisted text
    is the fallback string — not the LLM draft. This preserves S2's
    'show me what I saw' contract across guard hits."""
    fallback = (
        "Nothing urgent on the board today. I'll keep scanning in the "
        "background and check back tomorrow. Reply any time if something "
        "shifts."
    )
    _persist_briefing_output(_briefing_job(), fallback)
    files = list((hermes_home / "artemis" / "U0FIXTURE01" / "briefings").glob("*.json"))
    entry = json.loads(files[0].read_text(encoding="utf-8"))
    assert entry["formatted_output"] == fallback


# -----------------------------------------------------------------------------
# _persist_briefing_output — guard rails (no-op cases)
# -----------------------------------------------------------------------------

def test_persist_noop_when_user_id_missing(hermes_home):
    job = _briefing_job()
    job["origin"] = {"platform": "slack", "chat_id": "D0CHANNEL01"}
    # No sidecar exists for chat_id, and no user_id in origin → resolver gives up
    _persist_briefing_output(job, SAMPLE_BRIEFING_TEXT)
    # No artemis dir was created
    assert not (hermes_home / "artemis").exists()


def test_persist_noop_when_origin_missing(hermes_home):
    job = _briefing_job()
    del job["origin"]
    _persist_briefing_output(job, SAMPLE_BRIEFING_TEXT)
    assert not (hermes_home / "artemis").exists()


def test_persist_noop_when_text_empty(hermes_home):
    _persist_briefing_output(_briefing_job(), "")
    assert not (hermes_home / "artemis" / "U0FIXTURE01" / "briefings").exists() or \
        not any((hermes_home / "artemis" / "U0FIXTURE01" / "briefings").glob("*.json"))


def test_persist_noop_when_text_whitespace_only(hermes_home):
    _persist_briefing_output(_briefing_job(), "   \n\t  ")
    target = hermes_home / "artemis" / "U0FIXTURE01" / "briefings"
    assert not target.exists() or not any(target.glob("*.json"))


# -----------------------------------------------------------------------------
# _persist_briefing_output — failure isolation
# -----------------------------------------------------------------------------

def test_persist_swallows_write_failure(hermes_home, caplog):
    """An OSError during file write must be logged + swallowed, never raised.
    This pins the contract: persistence failure cannot affect mark_job_run.
    """
    # Pre-create the briefings dir as a regular file so mkdir / write fails
    bad_path = hermes_home / "artemis" / "U0FIXTURE01" / "briefings"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("blocker", encoding="utf-8")  # file where dir is expected

    # Must not raise
    _persist_briefing_output(_briefing_job(), SAMPLE_BRIEFING_TEXT)


def test_persist_swallows_exception_from_resolve_origin(hermes_home, caplog):
    """Even if origin resolution raises, persistence helper swallows."""
    with patch("cron.scheduler._resolve_origin", side_effect=RuntimeError("boom")):
        # Must not raise
        _persist_briefing_output(_briefing_job(), SAMPLE_BRIEFING_TEXT)
