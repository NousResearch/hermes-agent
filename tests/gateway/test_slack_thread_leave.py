import asyncio
import json

from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.thread_participation import SlackThreadParticipationStore


def _event(text: str, *, ts: str, thread_ts: str, team: str = "T1") -> dict:
    return {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "team": team,
        "user": "U123",
        "client_msg_id": f"msg-{team}-{ts}",
        "text": text,
        "ts": ts,
        "thread_ts": thread_ts,
    }


def _adapter(tmp_path) -> SlackAdapter:
    adapter = SlackAdapter(
        PlatformConfig(
            extra={
                "allowed_channels": ["C123"],
                "require_mention": True,
                "reply_in_thread": True,
            }
        )
    )
    adapter._bot_user_id = "UBOT"
    adapter._team_bot_user_ids.update({"T1": "UBOT", "T2": "UBOT"})
    adapter._thread_participation = SlackThreadParticipationStore(tmp_path / "left.json")
    adapter._has_active_session_for_thread = lambda **_: True
    adapter.send = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


def test_direct_leave_mutes_active_thread_until_direct_mention_rejoins(tmp_path):
    adapter = _adapter(tmp_path)

    asyncio.run(
        adapter._handle_slack_message(
            _event("<@UBOT> !leave", ts="99.000", thread_ts="99.000")
        )
    )
    adapter.handle_message.assert_awaited_once()
    adapter.handle_message.reset_mock()

    asyncio.run(
        adapter._handle_slack_message(
            _event("!leave", ts="100.500", thread_ts="100.000")
        )
    )
    adapter.handle_message.assert_awaited_once()
    adapter.handle_message.reset_mock()

    asyncio.run(
        adapter._handle_slack_message(
            _event("<@UBOT> !leave", ts="101.000", thread_ts="100.000")
        )
    )
    adapter.handle_message.assert_not_awaited()
    adapter.send.assert_awaited_once_with(
        "C123", "Left this thread. Mention me to rejoin.", reply_to="100.000",
        metadata={"team_id": "T1"},
    )

    adapter = _adapter(tmp_path)  # Restart: profile-scoped state must survive reconstruction.
    asyncio.run(
        adapter._handle_slack_message(
            _event(
                "same ids, other workspace", ts="101.500", thread_ts="100.000", team="T2"
            )
        )
    )
    adapter.handle_message.assert_awaited_once()
    adapter.handle_message.reset_mock()

    asyncio.run(
        adapter._handle_slack_message(
            _event("ordinary follow-up", ts="102.000", thread_ts="100.000")
        )
    )
    adapter.handle_message.assert_not_awaited()

    asyncio.run(
        adapter._handle_slack_message(
            _event("<@UBOT> come back", ts="103.000", thread_ts="100.000")
        )
    )
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].text == "come back"


def test_leave_ack_clears_every_direct_wake_marker_and_mute_overrides_them(tmp_path):
    adapter = _adapter(tmp_path)
    root = "100.000"
    marker = adapter._workspace_message_marker("T1", root)
    adapter._mentioned_threads.update({marker, root})
    adapter._bot_message_ts.update({marker, root})

    async def send_ack(*args, **kwargs):
        # The real send path records the reply's root as bot participation.
        adapter._bot_message_ts.add(marker)

    adapter.send.side_effect = send_ack

    asyncio.run(adapter._handle_slack_message(
        _event("<@UBOT> !leave", ts="101.000", thread_ts=root)))

    assert marker not in adapter._mentioned_threads
    assert root not in adapter._mentioned_threads
    assert marker not in adapter._bot_message_ts
    assert root not in adapter._bot_message_ts
    assert adapter._thread_participation.is_muted("T1", "C123", root)

    adapter.handle_message.reset_mock()
    asyncio.run(adapter._handle_slack_message(
        _event("ordinary follow-up", ts="102.000", thread_ts=root)))
    adapter.handle_message.assert_not_awaited()


def test_leave_reports_failure_when_mute_cannot_be_persisted(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "plugins.platforms.slack.thread_participation.atomic_json_write", fail_write)

    asyncio.run(adapter._handle_slack_message(
        _event("<@UBOT> !leave", ts="101.000", thread_ts="100.000")))

    adapter.send.assert_awaited_once_with(
        "C123",
        "Couldn't leave this thread because the leave state could not be saved. Please try again.",
        reply_to="100.000",
        metadata={"team_id": "T1"},
    )
    assert adapter._thread_participation.is_muted("T1", "C123", "100.000")


def test_participation_store_prunes_oversized_state_on_load(tmp_path):
    path = tmp_path / "left.json"
    path.write_text(json.dumps({f"thread-{i}": i for i in range(6)}), encoding="utf-8")

    SlackThreadParticipationStore(path, max_entries=4)

    assert len(json.loads(path.read_text(encoding="utf-8"))) <= 4
