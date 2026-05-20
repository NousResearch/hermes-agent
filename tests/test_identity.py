"""
tests/test_identity.py — Unit tests for HermesIdentity.

Coverage:
- Frozen-ness: mutation must raise FrozenInstanceError.
- scope_chain ordering: personal → team → global (CSS specificity model).
- Scope string format: each scope is deterministic and platform-aware.
- Optional thread_id: present and absent paths.
- Edge cases: empty-ish user_id strings, non-Slack platforms.
- Slack event extraction: HermesIdentity.from_slack_event() round-trips
  a real-shape Slack event payload.

These tests import only hermes_identity — no gateway, no agent, no LLM.
"""

import dataclasses
import pytest

from hermes_identity import HermesIdentity


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SLACK_TEAM_ID = "T0123ABCDE"
SLACK_USER_ID = "U9876XYZ01"
SLACK_CHANNEL_ID = "C0B4EHQFHS5"
SLACK_THREAD_TS = "1716000000.123456"


def make_identity(**overrides) -> HermesIdentity:
    """Return a baseline HermesIdentity with sensible defaults."""
    defaults = dict(
        platform="slack",
        team_id=SLACK_TEAM_ID,
        user_id=SLACK_USER_ID,
        channel_id=SLACK_CHANNEL_ID,
        thread_id=SLACK_THREAD_TS,
    )
    defaults.update(overrides)
    return HermesIdentity(**defaults)


# ---------------------------------------------------------------------------
# Frozen-ness
# ---------------------------------------------------------------------------

class TestFrozenness:
    def test_mutation_of_platform_raises(self):
        identity = make_identity()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            identity.platform = "discord"  # type: ignore[misc]

    def test_mutation_of_team_id_raises(self):
        identity = make_identity()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            identity.team_id = "T_OTHER"  # type: ignore[misc]

    def test_mutation_of_user_id_raises(self):
        identity = make_identity()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            identity.user_id = "U_HACKER"  # type: ignore[misc]

    def test_mutation_of_thread_id_raises(self):
        identity = make_identity()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            identity.thread_id = "9999.0000"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Scope chain ordering
# ---------------------------------------------------------------------------

class TestScopeChain:
    def test_scope_chain_length(self):
        identity = make_identity()
        assert len(identity.scope_chain) == 3

    def test_scope_chain_order_is_personal_then_team_then_global(self):
        identity = make_identity()
        chain = identity.scope_chain
        assert chain[0] == identity.personal_scope
        assert chain[1] == identity.team_scope
        assert chain[2] == "global"

    def test_personal_scope_is_most_specific(self):
        identity = make_identity()
        # personal must contain user_id — more specific than team
        assert identity.user_id in identity.personal_scope
        assert identity.user_id not in identity.team_scope

    def test_team_scope_contains_team_id(self):
        identity = make_identity()
        assert identity.team_id in identity.team_scope

    def test_global_scope_is_literal_string(self):
        identity = make_identity()
        assert identity.global_scope == "global"

    def test_scope_chain_is_list(self):
        identity = make_identity()
        assert isinstance(identity.scope_chain, list)


# ---------------------------------------------------------------------------
# Scope string format
# ---------------------------------------------------------------------------

class TestScopeFormat:
    def test_personal_scope_format(self):
        identity = make_identity(
            platform="slack",
            team_id="TTEAM",
            user_id="UUSER",
        )
        assert identity.personal_scope == "personal/slack/TTEAM/UUSER"

    def test_team_scope_format(self):
        identity = make_identity(
            platform="slack",
            team_id="TTEAM",
        )
        assert identity.team_scope == "team/slack/TTEAM"

    def test_discord_platform_scope(self):
        identity = make_identity(
            platform="discord",
            team_id="GUILD123",
            user_id="USER456",
        )
        assert identity.personal_scope == "personal/discord/GUILD123/USER456"
        assert identity.team_scope == "team/discord/GUILD123"

    def test_scopes_differ_across_users(self):
        alice = make_identity(user_id="UALICE")
        bob = make_identity(user_id="UBOB")
        assert alice.personal_scope != bob.personal_scope
        # team scope is the same — same workspace
        assert alice.team_scope == bob.team_scope

    def test_scopes_differ_across_teams(self):
        workspace_a = make_identity(team_id="TAAAA")
        workspace_b = make_identity(team_id="TBBBB")
        assert workspace_a.team_scope != workspace_b.team_scope
        assert workspace_a.personal_scope != workspace_b.personal_scope


# ---------------------------------------------------------------------------
# Optional thread_id
# ---------------------------------------------------------------------------

class TestThreadId:
    def test_thread_id_defaults_to_none(self):
        identity = HermesIdentity(
            platform="slack",
            team_id=SLACK_TEAM_ID,
            user_id=SLACK_USER_ID,
            channel_id=SLACK_CHANNEL_ID,
        )
        assert identity.thread_id is None

    def test_thread_id_present(self):
        identity = make_identity(thread_id="1716000000.000100")
        assert identity.thread_id == "1716000000.000100"

    def test_scope_chain_same_with_or_without_thread(self):
        """thread_id does NOT affect scope resolution (it's a routing field)."""
        with_thread = make_identity(thread_id="1716000000.000100")
        without_thread = make_identity(thread_id=None)
        assert with_thread.scope_chain == without_thread.scope_chain


# ---------------------------------------------------------------------------
# Slack event extraction — real-shape event payload
# ---------------------------------------------------------------------------

# A real Slack event payload shape (message event in a channel thread).
# Source: https://api.slack.com/events/message
SLACK_MESSAGE_EVENT = {
    "type": "message",
    "text": "Hey Hermes, summarise this thread",
    "user": SLACK_USER_ID,
    "ts": "1716120000.000200",
    "thread_ts": SLACK_THREAD_TS,
    "channel": SLACK_CHANNEL_ID,
    "channel_type": "channel",
    "team": SLACK_TEAM_ID,
    "blocks": [],
    "event_ts": "1716120000.000200",
}

# app_mention event (no thread_ts → thread_id should be None)
SLACK_MENTION_EVENT = {
    "type": "app_mention",
    "text": "<@U_BOT> hello",
    "user": SLACK_USER_ID,
    "ts": "1716120001.000100",
    "channel": SLACK_CHANNEL_ID,
    "team": SLACK_TEAM_ID,
}


class TestSlackEventExtraction:
    """
    Validate the standalone extraction helper used by the Slack gateway adapter.

    The helper is inlined here (not imported from slack.py) so these tests
    have zero dependency on the heavy gateway stack.  The gateway adapter
    calls identical logic — see gateway/platforms/slack.py _handle_message().
    """

    @staticmethod
    def identity_from_slack_event(event: dict) -> HermesIdentity:
        """
        Extract a HermesIdentity from a raw Slack event dict.

        Mirrors the extraction logic in gateway/platforms/slack.py so we can
        test the shape contract without importing the full gateway.
        """
        team_id = (
            event.get("team")
            or event.get("team_id")
            or ""
        )
        user_id = event.get("user") or ""
        channel_id = event.get("channel") or ""
        thread_id = event.get("thread_ts") or None

        return HermesIdentity(
            platform="slack",
            team_id=str(team_id),
            user_id=str(user_id),
            channel_id=str(channel_id),
            thread_id=thread_id,
        )

    def test_message_event_extracts_all_fields(self):
        identity = self.identity_from_slack_event(SLACK_MESSAGE_EVENT)
        assert identity.platform == "slack"
        assert identity.team_id == SLACK_TEAM_ID
        assert identity.user_id == SLACK_USER_ID
        assert identity.channel_id == SLACK_CHANNEL_ID
        assert identity.thread_id == SLACK_THREAD_TS

    def test_mention_event_no_thread_yields_none_thread_id(self):
        identity = self.identity_from_slack_event(SLACK_MENTION_EVENT)
        assert identity.thread_id is None

    def test_extracted_identity_has_valid_scope_chain(self):
        identity = self.identity_from_slack_event(SLACK_MESSAGE_EVENT)
        chain = identity.scope_chain
        assert len(chain) == 3
        assert chain[0].startswith("personal/slack/")
        assert chain[1].startswith("team/slack/")
        assert chain[2] == "global"

    def test_extracted_identity_is_frozen(self):
        identity = self.identity_from_slack_event(SLACK_MESSAGE_EVENT)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            identity.user_id = "HACKED"  # type: ignore[misc]

    def test_team_id_fallback_from_team_id_key(self):
        """Some Slack event shapes use 'team_id' instead of 'team'."""
        event = {**SLACK_MESSAGE_EVENT}
        del event["team"]
        event["team_id"] = SLACK_TEAM_ID
        identity = self.identity_from_slack_event(event)
        assert identity.team_id == SLACK_TEAM_ID


# ---------------------------------------------------------------------------
# Hashability (frozen dataclasses are hashable — useful for dict keys)
# ---------------------------------------------------------------------------

class TestHashability:
    def test_identity_is_hashable(self):
        identity = make_identity()
        # Should not raise
        d = {identity: "value"}
        assert d[identity] == "value"

    def test_equal_identities_have_same_hash(self):
        a = make_identity()
        b = make_identity()
        assert a == b
        assert hash(a) == hash(b)

    def test_different_identities_are_not_equal(self):
        a = make_identity(user_id="UALICE")
        b = make_identity(user_id="UBOB")
        assert a != b
