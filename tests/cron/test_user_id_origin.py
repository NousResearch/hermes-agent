"""G1 (S-0429-01) — cron scheduler propagates user_id through origin.

When a cron job fires, the gateway respawn injects ``HERMES_SESSION_USER_ID``
into the agent process env so the MCP subprocess (downstream of
``_run_stdio``) can bind to the right user. This requires the scheduler to
know which user_id to inject — sourced either from ``origin.user_id`` (jobs
created post-G1) or via reverse-lookup from ``slack_channel.txt`` sidecar
files (legacy jobs).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestResolveOriginUserIdForward:
    """New jobs persist ``origin.user_id`` directly; ``_resolve_origin``
    surfaces it unchanged."""

    def test_origin_with_user_id_is_returned(self):
        from cron.scheduler import _resolve_origin

        job = {
            "origin": {
                "platform": "slack",
                "chat_id": "D1ABC",
                "user_id": "U0AQW54L1UN",
            }
        }
        origin = _resolve_origin(job)
        assert origin is not None
        assert origin.get("user_id") == "U0AQW54L1UN"


class TestResolveOriginUserIdReverseLookup:
    """Legacy jobs lack ``origin.user_id``. Reverse-resolve via the
    per-user ``slack_channel.txt`` sidecar files written by the Slack
    gateway adapter."""

    @pytest.fixture
    def hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Plant two users; only one matches the job's chat_id.
        (tmp_path / "artemis" / "U0ALICE").mkdir(parents=True)
        (tmp_path / "artemis" / "U0ALICE" / "slack_channel.txt").write_text(
            "D1ALICE"
        )
        (tmp_path / "artemis" / "U0BOB").mkdir(parents=True)
        (tmp_path / "artemis" / "U0BOB" / "slack_channel.txt").write_text("D1BOB")
        return tmp_path

    def test_legacy_job_reverse_resolves_user_id(self, hermes_home):
        from cron.scheduler import _resolve_origin

        job = {"origin": {"platform": "slack", "chat_id": "D1ALICE"}}
        origin = _resolve_origin(job)
        assert origin is not None
        assert origin.get("user_id") == "U0ALICE"

    def test_legacy_job_no_match_leaves_user_id_none(self, hermes_home):
        from cron.scheduler import _resolve_origin

        job = {"origin": {"platform": "slack", "chat_id": "D1UNKNOWN"}}
        origin = _resolve_origin(job)
        assert origin is not None
        assert origin.get("user_id") is None

    def test_explicit_user_id_wins_over_reverse_lookup(self, hermes_home):
        """When origin already has user_id, don't second-guess it via
        sidecar lookup. The persisted value is authoritative."""
        from cron.scheduler import _resolve_origin

        job = {
            "origin": {
                "platform": "slack",
                "chat_id": "D1ALICE",  # would resolve to U0ALICE via sidecar
                "user_id": "U0EXPLICIT",
            }
        }
        origin = _resolve_origin(job)
        assert origin.get("user_id") == "U0EXPLICIT"


class TestEmptyOriginUnchanged:
    """Jobs with no origin at all (script-created, no platform context)
    should keep returning None — reverse-lookup must not invent one."""

    def test_no_origin_returns_none(self):
        from cron.scheduler import _resolve_origin

        assert _resolve_origin({}) is None
        assert _resolve_origin({"origin": None}) is None
