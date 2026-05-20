"""Tests for Slack job-match-card button handlers (S-0511-08, Artemis pilot).

Mirrors test_slack_approval_buttons.py shape for the new job_card_save and
job_card_skip handlers. Shortlist writes go to ~/.hermes/artemis/<user_id>/
shortlist.json — schema kept in sync with Artemis mcp-server/tools/shortlist.py.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from gateway.platforms.slack import SlackAdapter
from gateway.config import Platform, PlatformConfig


def _make_adapter():
    config = PlatformConfig(enabled=True, token="xoxb-test-token")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"D1": "T1"}
    return adapter


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point hermes_constants.get_hermes_home() at a tmp dir for isolation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The handler imports get_hermes_home inside the function body, so the
    # env change is picked up by the next call. Reset any cached module
    # state if it ever caches.
    import hermes_constants
    if hasattr(hermes_constants, "_HERMES_HOME_CACHE"):
        hermes_constants._HERMES_HOME_CACHE = None
    return tmp_path


def _click_body(user_id="U0FIXTURE01", channel_id="D1"):
    return {
        "user": {"id": user_id, "name": "howiehuang"},
        "channel": {"id": channel_id},
        "message": {"ts": "1.0"},
    }


def _save_value(job_id="job-A", title="Senior PM", company="Plaid",
                location="SF, CA", url="https://plaid.com/x"):
    return json.dumps({
        "job_id": job_id,
        "title": title,
        "company": company,
        "location": location,
        "url": url,
    })


class TestJobCardSave:
    @pytest.mark.asyncio
    async def test_writes_new_entry(self, hermes_home):
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        action = {"action_id": "job_card_save", "value": _save_value("job-A")}

        await adapter._handle_job_card_save(ack, _click_body(), action)
        ack.assert_called_once()

        path = hermes_home / "artemis" / "U0FIXTURE01" / "shortlist.json"
        assert path.exists()
        entries = json.loads(path.read_text())
        assert len(entries) == 1
        assert entries[0]["job_id"] == "job-A"
        assert entries[0]["title"] == "Senior PM"
        assert entries[0]["saved_at"].endswith("Z")

        # Hard-coded ack message posted
        post_kwargs = adapter._team_clients["T1"].chat_postMessage.call_args[1]
        assert post_kwargs["text"] == "Saved to your shortlist."
        assert post_kwargs["channel"] == "D1"

    @pytest.mark.asyncio
    async def test_dedupes_by_job_id(self, hermes_home):
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        action = {"action_id": "job_card_save", "value": _save_value("job-A")}
        await adapter._handle_job_card_save(ack, _click_body(), action)
        await adapter._handle_job_card_save(ack, _click_body(), action)

        path = hermes_home / "artemis" / "U0FIXTURE01" / "shortlist.json"
        entries = json.loads(path.read_text())
        assert len(entries) == 1  # no duplicate appended

    @pytest.mark.asyncio
    async def test_atomic_write_via_temp_rename(self, hermes_home, monkeypatch):
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        seen = []
        from pathlib import Path
        original_replace = Path.replace

        def spy(self, target):
            seen.append((str(self), str(target)))
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", spy)
        action = {"action_id": "job_card_save", "value": _save_value("job-A")}
        await adapter._handle_job_card_save(ack, _click_body(), action)

        assert len(seen) == 1
        src, dst = seen[0]
        assert src.endswith(".tmp")
        assert dst.endswith("shortlist.json")

    @pytest.mark.asyncio
    async def test_malformed_value_swallowed(self, hermes_home):
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        # value is not valid JSON
        action = {"action_id": "job_card_save", "value": "not-json"}
        # Must not raise
        await adapter._handle_job_card_save(ack, _click_body(), action)
        ack.assert_called_once()

        path = hermes_home / "artemis" / "U0FIXTURE01" / "shortlist.json"
        assert not path.exists()


class TestJobCardSkip:
    @pytest.mark.asyncio
    async def test_posts_ack_no_persistence(self, hermes_home):
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        action = {"action_id": "job_card_skip", "value": "job-A"}
        await adapter._handle_job_card_skip(ack, _click_body(), action)
        ack.assert_called_once()

        post_kwargs = adapter._team_clients["T1"].chat_postMessage.call_args[1]
        assert post_kwargs["text"] == "Dropped from this list."
        assert post_kwargs["channel"] == "D1"

        # No file write on Skip
        path = hermes_home / "artemis" / "U0FIXTURE01" / "shortlist.json"
        assert not path.exists()
