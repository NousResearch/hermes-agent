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


def _click_body(user_id="U0FIXTURE01", channel_id="D1", message_ts="1.0", thread_ts=None):
    """Build a Slack interaction body. By default ts is the cards message;
    thread_ts is set when the cards landed inside a thread (post-S-0511-08
    thread fix). When thread_ts is None, ack lands in the same thread as
    the card's parent (handler falls back to message ts)."""
    message = {"ts": message_ts}
    if thread_ts is not None:
        message["thread_ts"] = thread_ts
    return {
        "user": {"id": user_id, "name": "howiehuang"},
        "channel": {"id": channel_id},
        "message": message,
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
    async def test_ack_threads_under_card_when_thread_ts_present(self, hermes_home):
        """When the card was posted in a thread, the ack stays in that thread."""
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        action = {"action_id": "job_card_save", "value": _save_value("job-A")}
        body = _click_body(message_ts="2.0", thread_ts="1.5")
        await adapter._handle_job_card_save(ack, body, action)

        post_kwargs = adapter._team_clients["T1"].chat_postMessage.call_args[1]
        assert post_kwargs.get("thread_ts") == "1.5"

    @pytest.mark.asyncio
    async def test_ack_falls_back_to_card_ts_when_no_thread_ts(self, hermes_home):
        """If card was posted as a root message (legacy), ack threads under the card itself."""
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        action = {"action_id": "job_card_save", "value": _save_value("job-A")}
        body = _click_body(message_ts="3.0", thread_ts=None)
        await adapter._handle_job_card_save(ack, body, action)

        post_kwargs = adapter._team_clients["T1"].chat_postMessage.call_args[1]
        assert post_kwargs.get("thread_ts") == "3.0"

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
    async def test_posts_ack_no_persistence_when_not_in_shortlist(self, hermes_home):
        """Skip on a job that was never Saved → 'Dropped from this list.', no file."""
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        action = {"action_id": "job_card_skip", "value": "job-A"}
        await adapter._handle_job_card_skip(ack, _click_body(message_ts="2.0", thread_ts="1.5"), action)
        ack.assert_called_once()

        post_kwargs = adapter._team_clients["T1"].chat_postMessage.call_args[1]
        assert post_kwargs["text"] == "Dropped from this list."
        assert post_kwargs.get("thread_ts") == "1.5"

        # No file write when there was nothing to remove
        path = hermes_home / "artemis" / "U0FIXTURE01" / "shortlist.json"
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_removes_job_from_shortlist_when_present(self, hermes_home):
        """Skip after Save → removes the entry; ack says 'Removed from your shortlist.'"""
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        # Pre-seed shortlist with the job
        save_action = {"action_id": "job_card_save", "value": _save_value("job-A")}
        await adapter._handle_job_card_save(ack, _click_body(), save_action)

        path = hermes_home / "artemis" / "U0FIXTURE01" / "shortlist.json"
        assert len(json.loads(path.read_text())) == 1

        # Now Skip should remove it
        skip_action = {"action_id": "job_card_skip", "value": "job-A"}
        await adapter._handle_job_card_skip(ack, _click_body(message_ts="2.0", thread_ts="1.5"), skip_action)

        entries = json.loads(path.read_text())
        assert len(entries) == 0, "Skip after Save should remove the entry"

        # Ack text reflects the removal
        last_call = adapter._team_clients["T1"].chat_postMessage.call_args[1]
        assert last_call["text"] == "Removed from your shortlist."
        assert last_call.get("thread_ts") == "1.5"

    @pytest.mark.asyncio
    async def test_skip_preserves_other_entries(self, hermes_home):
        """Skip removes only the matching job_id, leaves others untouched."""
        adapter = _make_adapter()
        adapter._team_clients["T1"].chat_postMessage = AsyncMock()
        ack = AsyncMock()

        # Pre-seed two entries
        await adapter._handle_job_card_save(ack, _click_body(), {"action_id": "job_card_save", "value": _save_value("job-A")})
        await adapter._handle_job_card_save(ack, _click_body(), {"action_id": "job_card_save", "value": _save_value("job-B", title="Other PM")})

        path = hermes_home / "artemis" / "U0FIXTURE01" / "shortlist.json"
        assert len(json.loads(path.read_text())) == 2

        # Skip job-A only
        await adapter._handle_job_card_skip(ack, _click_body(), {"action_id": "job_card_skip", "value": "job-A"})

        entries = json.loads(path.read_text())
        assert len(entries) == 1
        assert entries[0]["job_id"] == "job-B"
