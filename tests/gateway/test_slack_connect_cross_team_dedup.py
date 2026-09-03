"""
Tests for Slack Connect cross-workspace duplicate suppression.

A message posted in a Slack Connect shared channel is delivered once per
workspace context: the inner event carries the sender's home ``team`` while
the outer payload carries the local ``team_id``. The workspace-scoped dedup
id (``team:ts``) therefore resolves to two different keys for one physical
message, and the bot replies twice.

The fix suppresses the re-delivery on the sender's ``client_msg_id`` — a
per-message client UUID that is stable across workspace contexts but never
shared by two distinct messages — scoped by event_ts so message_changed
events are not swallowed. client_msg_id is absent from the synthetic
multi-workspace collision events guarded by
TestSlackWorkspaceCollisionIsolation, so that isolation behavior is
unchanged.

Follows the slack-bolt mocking pattern from test_slack_mention.py.
"""

import inspect
import sys
from unittest.mock import MagicMock


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.helpers import MessageDeduplicator  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402

LOCAL_TEAM = "T0LOCAL0001"     # workspace the bot is installed in
REMOTE_TEAM = "T0REMOTE0002"   # Slack Connect sender's home workspace
SHARED_CHANNEL = "C0SHARED001"
TS = "1788318023.505649"
CLIENT_MSG_ID = "4f1a2b3c-aaaa-bbbb-cccc-1234567890ab"


def _make_dedup():
    return MessageDeduplicator(ttl_seconds=3600)


def _passes_dedup_gate(dedup, event, payload):
    """Mirror of the dedup gate at the top of ``_handle_slack_message_impl``.

    Returns True when the event would be processed, False when suppressed.
    Kept in lockstep with production by
    ``test_cross_team_dedup_pinned_to_production_source`` below.
    """
    event_ts = event.get("_slack_changed_event_ts") or event.get("ts", "")
    dedup_team_id = SlackAdapter._event_team_id(event, payload)
    if event_ts and dedup.is_duplicate(
        SlackAdapter._workspace_event_id(dedup_team_id, event_ts)
    ):
        return False
    client_msg_id = str(event.get("client_msg_id") or "")
    if (
        event_ts
        and client_msg_id
        and dedup.is_duplicate(f"cmid:{client_msg_id}:{event_ts}")
    ):
        return False
    return True


def _slack_connect_deliveries():
    """The two envelopes one shared-channel message produces.

    First delivery: inner event carries the sender's home team.
    Second delivery: no inner team; the outer payload carries the local team.
    Channel id, ts, and client_msg_id are identical — one physical message.
    """
    first_event = {
        "type": "message",
        "text": "hi",
        "user": "U0EXTERNAL1",
        "team": REMOTE_TEAM,
        "channel": SHARED_CHANNEL,
        "ts": TS,
        "client_msg_id": CLIENT_MSG_ID,
    }
    first_payload = {"team_id": REMOTE_TEAM}
    second_event = {
        "type": "message",
        "text": "hi",
        "user": "U0EXTERNAL1",
        "channel": SHARED_CHANNEL,
        "ts": TS,
        "client_msg_id": CLIENT_MSG_ID,
    }
    second_payload = {"team_id": LOCAL_TEAM}
    return (first_event, first_payload), (second_event, second_payload)


def test_slack_connect_second_delivery_suppressed_by_client_msg_id():
    # The whole point of the fix: the cross-workspace re-delivery of the SAME
    # message (same client_msg_id/ts, different resolved team) must not
    # produce a second bot reply.
    dedup = _make_dedup()
    (e1, p1), (e2, p2) = _slack_connect_deliveries()
    assert _passes_dedup_gate(dedup, e1, p1) is True
    assert _passes_dedup_gate(dedup, e2, p2) is False


def test_socket_mode_redelivery_still_suppressed_by_team_key():
    # Pre-existing behavior (#4777) preserved: an identical envelope replayed
    # after a reconnect dedups on the workspace-scoped key.
    dedup = _make_dedup()
    (e1, p1), _ = _slack_connect_deliveries()
    assert _passes_dedup_gate(dedup, e1, p1) is True
    assert _passes_dedup_gate(dedup, dict(e1), dict(p1)) is False


def test_workspace_collision_isolation_preserved():
    # The invariant TestSlackWorkspaceCollisionIsolation pins: two workspaces'
    # coincidentally identical Slack-local ids (same channel, same ts, no
    # client_msg_id, no event_id) are two DISTINCT messages and both must be
    # delivered. The new gate keys off ids those events do not share, so the
    # isolation behavior is unchanged.
    dedup = _make_dedup()
    event = {
        "text": "same Slack-local ids",
        "user": "U_SHARED",
        "channel": "D_SHARED",
        "channel_type": "im",
        "ts": "171.000",
    }
    assert _passes_dedup_gate(dedup, dict(event), {"team_id": "T_ONE"}) is True
    assert _passes_dedup_gate(dedup, dict(event), {"team_id": "T_TWO"}) is True


def test_distinct_messages_never_share_client_msg_id():
    # Two different user messages carry different client UUIDs — same ts in
    # two channels/workspaces stays independently deliverable.
    dedup = _make_dedup()
    event_a = {"type": "message", "text": "a", "user": "U1", "team": "T0TEAMA0001",
               "channel": "C0CHANA0001", "ts": TS,
               "client_msg_id": "11111111-aaaa-bbbb-cccc-000000000001"}
    event_b = {"type": "message", "text": "b", "user": "U2", "team": "T0TEAMB0002",
               "channel": "C0CHANB0002", "ts": TS,
               "client_msg_id": "22222222-aaaa-bbbb-cccc-000000000002"}
    assert _passes_dedup_gate(dedup, event_a, {"team_id": "T0TEAMA0001"}) is True
    assert _passes_dedup_gate(dedup, event_b, {"team_id": "T0TEAMB0002"}) is True


def test_message_changed_not_swallowed_as_duplicate_of_original():
    # An edit re-carries the original client_msg_id but a different effective
    # event_ts (_slack_changed_event_ts); scoping the cmid key by ts keeps
    # the edit deliverable while its own cross-workspace re-delivery still
    # collapses.
    dedup = _make_dedup()
    (e1, p1), _ = _slack_connect_deliveries()
    assert _passes_dedup_gate(dedup, e1, p1) is True

    changed = dict(e1)
    changed["_slack_changed_event_ts"] = "1788318099.000200"
    assert _passes_dedup_gate(dedup, changed, dict(p1)) is True

    changed_redelivery = dict(changed)
    changed_redelivery.pop("team")
    assert _passes_dedup_gate(dedup, changed_redelivery, {"team_id": LOCAL_TEAM}) is False


def test_cross_team_dedup_pinned_to_production_source():
    """Regression teeth: pin the cross-workspace dedup ids in the impl.

    ``_passes_dedup_gate`` above mirrors the production gate; this test pins
    the production source so a revert of the Slack Connect fix fails here
    instead of silently passing a self-referential mirror.
    """
    src = inspect.getsource(SlackAdapter._handle_slack_message_impl)
    assert 'f"cmid:{client_msg_id}:{event_ts}"' in src, (
        "client_msg_id dedup removed — a Slack Connect shared-channel message "
        "would be processed once per workspace context and the bot would "
        "reply twice (see the cross-team duplicate RCA)"
    )
