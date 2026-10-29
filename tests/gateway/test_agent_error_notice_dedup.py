"""The notification path must not amplify a deterministic failure.

On 2026-10-28 a profile's ``config.yaml`` became unparseable (one duplicate YAML key).
Every turn that touched the terminal raised ``TerminalPolicyUnavailable``, and the gateway
sent the user **85 identical 282-character DMs** in a row — the notification hammer Bryan
reported.

The fix: a DETERMINISTIC failure is notified once per (session, error class) within a
cooldown window. A TRANSIENT failure (rate limit, timeout) is still notified every time —
a crash loop is the thing that must not become a DM storm, not a flaky provider.
"""

from __future__ import annotations

import pytest


class TerminalPolicyUnavailable(Exception):
    """The exception the 2026-10-28 incident raised on every turn."""


class _RateLimitError(Exception):
    """A transient failure: a retry can succeed."""

    status_code = 429


class _AuthError(Exception):
    """A deterministic failure the provider reported."""

    status_code = 401


class TestErrorClassKey:
    def test_same_failure_gives_the_same_class(self):
        from gateway.error_notice_dedup import error_class_key

        assert error_class_key(TerminalPolicyUnavailable()) == error_class_key(
            TerminalPolicyUnavailable())

    def test_different_failures_give_different_classes(self):
        from gateway.error_notice_dedup import error_class_key

        assert error_class_key(TerminalPolicyUnavailable()) != error_class_key(_RateLimitError())

    def test_the_incident_error_is_deterministic(self):
        from gateway.error_notice_dedup import is_deterministic

        assert is_deterministic(TerminalPolicyUnavailable()) is True


class TestDeterminism:
    def test_transient_failures_are_not_deterministic(self):
        from gateway.error_notice_dedup import is_deterministic

        assert is_deterministic(_RateLimitError()) is False

    def test_auth_failure_is_deterministic(self):
        from gateway.error_notice_dedup import is_deterministic

        assert is_deterministic(_AuthError()) is True

    def test_unclassifiable_failure_is_treated_as_transient(self):
        """Suppressing a message the user needed is worse than one duplicate message."""
        from gateway.error_notice_dedup import is_deterministic

        class Weird(Exception):
            pass

        # An exception with no status_code and an unknown module is gateway-layer and
        # retryable in the surface's taxonomy — so it keeps notifying.
        assert is_deterministic(Weird()) is False


class TestLedgerCooldown:
    def test_85_consecutive_failures_send_at_most_one_notice(self):
        """The regression test for the 85-DM hammer."""
        from gateway.error_notice_dedup import AgentErrorNoticeLedger

        ledger = AgentErrorNoticeLedger(cooldown_seconds=300.0)
        sent = 0
        for _ in range(85):
            should_send, _class = ledger.notify("session-1", TerminalPolicyUnavailable())
            if should_send:
                sent += 1

        assert sent == 1, (
            f"85 consecutive identical failures produced {sent} notices; "
            f"a deterministic failure must notify once per cooldown window"
        )

    def test_transient_failures_keep_notifying(self):
        """A flaky provider's errors must not be swallowed."""
        from gateway.error_notice_dedup import AgentErrorNoticeLedger

        ledger = AgentErrorNoticeLedger(cooldown_seconds=300.0)
        sent = sum(
            1 for _ in range(10)
            if ledger.notify("session-1", _RateLimitError())[0]
        )
        assert sent == 10

    def test_distinct_error_classes_each_get_a_notice(self):
        from gateway.error_notice_dedup import AgentErrorNoticeLedger

        ledger = AgentErrorNoticeLedger(cooldown_seconds=300.0)
        first = ledger.notify("session-1", TerminalPolicyUnavailable())[0]
        second = ledger.notify("session-1", _AuthError())[0]
        assert first is True
        assert second is True

    def test_the_cooldown_expires_and_the_notice_returns(self):
        """After the window passes the user hears about it again."""
        from gateway.error_notice_dedup import AgentErrorNoticeLedger

        ledger = AgentErrorNoticeLedger(cooldown_seconds=300.0)
        assert ledger.notify("session-1", TerminalPolicyUnavailable(), now=1000.0)[0] is True
        assert ledger.notify("session-1", TerminalPolicyUnavailable(), now=1010.0)[0] is False
        assert ledger.notify("session-1", TerminalPolicyUnavailable(), now=1300.0)[0] is True

    def test_the_ledger_is_per_session(self):
        """One session's failure must not silence another session's notice."""
        from gateway.error_notice_dedup import AgentErrorNoticeLedger

        ledger = AgentErrorNoticeLedger(cooldown_seconds=300.0)
        assert ledger.notify("session-1", TerminalPolicyUnavailable())[0] is True
        assert ledger.notify("session-2", TerminalPolicyUnavailable())[0] is True

    def test_reset_clears_one_session(self):
        from gateway.error_notice_dedup import AgentErrorNoticeLedger

        ledger = AgentErrorNoticeLedger(cooldown_seconds=300.0)
        ledger.notify("session-1", TerminalPolicyUnavailable())
        ledger.reset("session-1")
        assert ledger.notify("session-1", TerminalPolicyUnavailable())[0] is True


class TestGatewayWiring:
    """The ledger must actually gate the USER-FACING reply, not just exist."""

    def _runner(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        from gateway.error_notice_dedup import AgentErrorNoticeLedger

        runner._agent_error_notice_ledger = AgentErrorNoticeLedger(cooldown_seconds=300.0)
        return runner

    def test_first_failure_notifies_and_the_rest_do_not(self):
        runner = self._runner()
        decisions = [
            runner._hmwa_should_notify_agent_error("session-1", TerminalPolicyUnavailable())[0]
            for _ in range(85)
        ]
        assert decisions.count(True) == 1
        assert decisions.count(False) == 84

    def test_a_missing_ledger_never_suppresses(self):
        """A partially constructed runner must not silently swallow errors."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        send, _class = runner._hmwa_should_notify_agent_error(
            "session-1", TerminalPolicyUnavailable())
        assert send is True

    @pytest.mark.asyncio
    async def test_the_gateway_reply_is_empty_after_the_first_notice(self):
        """The reply the adapter receives must be empty so no DM is sent."""
        runner = self._runner()

        class _Prepared:
            history = []
            message_text = None
            persistence_session_id = "p"
            persistence_owner = "o"

        class _Session:
            session_id = "s"

        # The transcript/persistence calls must not run in this test; stub them out.
        async def _stop(event, source):
            return None

        runner._hmwa_stop_typing_for_turn = _stop

        first = await runner._hmwa_agent_error_reply(
            TerminalPolicyUnavailable(), None, None, _Session(), "session-1", _Prepared())
        second = await runner._hmwa_agent_error_reply(
            TerminalPolicyUnavailable(), None, None, _Session(), "session-1", _Prepared())

        assert first, "the first failure must produce a user-facing notice"
        assert second == "", (
            "the second identical failure must produce an EMPTY reply so the adapter "
            "sends no DM"
        )
