"""Tests for the WS-upgrade ticket store (Phase 5 task 5.1).

The store is process-local and threading-safe. Tests run with xdist so
each worker has its own module instance — no cross-worker bleed — but we
call ``_reset_for_tests`` between tests to keep things deterministic.
"""

from __future__ import annotations

import threading

import pytest

from hermes_cli.dashboard_auth import ws_tickets
from hermes_cli.dashboard_auth.ws_tickets import (
    TTL_SECONDS,
    TicketInvalid,
    _reset_for_tests,
    consume_ticket,
    mint_ticket,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_for_tests()
    yield
    _reset_for_tests()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestMintAndConsume:
    def test_round_trip(self):
        ticket = mint_ticket(user_id="u1", provider="nous")
        info = consume_ticket(ticket)
        assert info["user_id"] == "u1"
        assert info["provider"] == "nous"
        assert "minted_at" in info

    def test_ticket_has_minimum_length(self):
        # ``secrets.token_urlsafe(32)`` produces ~43 chars; enforce a floor
        # so a future refactor can't accidentally shrink the entropy.
        ticket = mint_ticket(user_id="u1", provider="nous")
        assert len(ticket) >= 32

    def test_ticket_values_are_unique(self):
        seen = {mint_ticket(user_id="u1", provider="x") for _ in range(50)}
        assert len(seen) == 50

    def test_scoped_ticket_preserves_its_authorization_grant(self):
        ticket = mint_ticket(
            user_id="u1",
            provider="nous",
            audience="hermes.mobile",
            scopes=("conversation.read", "conversation.write"),
        )

        info = consume_ticket(ticket)

        assert info["audience"] == "hermes.mobile"
        assert info["scopes"] == ("conversation.read", "conversation.write")

    def test_scoped_ticket_can_explicitly_grant_conversation_deletion(self):
        ticket = mint_ticket(
            user_id="u1",
            provider="nous",
            audience="hermes.mobile",
            scopes=("conversation.read", "conversation.delete"),
        )

        info = consume_ticket(ticket)

        assert info["scopes"] == ("conversation.read", "conversation.delete")

    def test_ticket_store_rejects_unknown_audiences(self):
        with pytest.raises(ValueError, match="unsupported WebSocket audience"):
            mint_ticket(
                user_id="u1",
                provider="nous",
                audience="hermes.future",
                scopes=("*",),
            )

    def test_ticket_store_rejects_mobile_grants_without_read_scope(self):
        with pytest.raises(
            ValueError,
            match="conversation.read is required for every mobile WebSocket grant",
        ):
            mint_ticket(
                user_id="u1",
                provider="nous",
                audience="hermes.mobile",
                scopes=("conversation.write",),
            )


# ---------------------------------------------------------------------------
# Single-use
# ---------------------------------------------------------------------------


class TestSingleUse:
    def test_second_consume_raises(self):
        ticket = mint_ticket(user_id="u1", provider="stub")
        consume_ticket(ticket)
        with pytest.raises(TicketInvalid, match="unknown"):
            consume_ticket(ticket)

    def test_unknown_ticket_rejected(self):
        with pytest.raises(TicketInvalid, match="unknown"):
            consume_ticket("nope-never-minted")

    def test_empty_ticket_rejected(self):
        with pytest.raises(TicketInvalid):
            consume_ticket("")


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


class TestTTL:
    def test_constant_is_30_seconds(self):
        # Pinned so a refactor that doubled the lifetime would surface here.
        assert TTL_SECONDS == 30

    def test_expired_ticket_rejected(self, monkeypatch):
        # Mock time inside the ws_tickets module so mint and consume see
        # different clocks. We have to patch the symbol the module actually
        # binds; ``time`` is module-level there.
        clock = {"now": 1_000_000}

        def fake_time():
            return clock["now"]

        monkeypatch.setattr(ws_tickets.time, "time", fake_time)

        ticket = mint_ticket(user_id="u1", provider="stub")
        clock["now"] += TTL_SECONDS + 1
        with pytest.raises(TicketInvalid, match="expired"):
            consume_ticket(ticket)

    def test_at_exact_ttl_boundary_still_valid(self, monkeypatch):
        clock = {"now": 1_000_000}
        monkeypatch.setattr(ws_tickets.time, "time", lambda: clock["now"])

        ticket = mint_ticket(user_id="u1", provider="stub")
        clock["now"] += TTL_SECONDS  # exactly at boundary; expires_at == now
        # Implementation: ``expires_at < now`` (strict), so == passes.
        info = consume_ticket(ticket)
        assert info["user_id"] == "u1"


# ---------------------------------------------------------------------------
# Truncated value in error message (secret hygiene)
# ---------------------------------------------------------------------------


class TestErrorMessages:
    def test_unknown_ticket_error_truncates_value(self):
        long_value = "a" * 100
        with pytest.raises(TicketInvalid) as exc_info:
            consume_ticket(long_value)
        # Never log more than the first 8 chars of an opaque ticket.
        message = str(exc_info.value)
        assert long_value not in message
        assert long_value[:8] in message


# ---------------------------------------------------------------------------
# Thread safety: mint + consume from many threads doesn't deadlock or
# return duplicates.
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_mint_and_consume_concurrent(self):
        results: list[dict] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(i: int):
            try:
                t = mint_ticket(user_id=f"u{i}", provider="stub")
                info = consume_ticket(t)
                with lock:
                    results.append(info)
            except Exception as exc:  # noqa: BLE001 — collect for assert
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "thread deadlocked"

        assert errors == []
        assert len(results) == 20
        # Every consume returns a distinct user_id (no cross-thread bleed).
        assert {r["user_id"] for r in results} == {f"u{i}" for i in range(20)}


# ---------------------------------------------------------------------------
# Process-lifetime internal credential (server-spawned PTY child auth).
# Direct unit coverage for internal_ws_credential / consume_internal_credential
# — _ws_auth_ok exercises these indirectly, but the mint-once, unminted, and
# empty-value branches are only reachable via direct calls.
# ---------------------------------------------------------------------------


class TestInternalCredential:
    def test_minted_once_is_stable(self):
        """Successive calls return the same process-lifetime value."""
        first = ws_tickets.internal_ws_credential()
        second = ws_tickets.internal_ws_credential()
        assert first == second
        assert len(first) >= 32  # token_urlsafe(32)

    def test_round_trip_identity(self):
        cred = ws_tickets.internal_ws_credential()
        info = ws_tickets.consume_internal_credential(cred)
        assert info["user_id"] == ws_tickets.INTERNAL_USER_ID
        assert info["provider"] == ws_tickets.INTERNAL_PROVIDER

    def test_multi_use(self):
        """Unlike a single-use ticket, the credential survives repeated consume."""
        cred = ws_tickets.internal_ws_credential()
        for _ in range(5):
            assert (
                ws_tickets.consume_internal_credential(cred)["provider"]
                == ws_tickets.INTERNAL_PROVIDER
            )

    def test_rejected_before_mint(self):
        """With nothing minted yet, any value is rejected (expected is None)."""
        # autouse _reset leaves _internal_credential == None at test start.
        with pytest.raises(TicketInvalid):
            ws_tickets.consume_internal_credential("anything")

    def test_empty_value_rejected(self):
        ws_tickets.internal_ws_credential()  # mint so expected is non-None
        with pytest.raises(TicketInvalid):
            ws_tickets.consume_internal_credential("")

    def test_wrong_value_rejected(self):
        ws_tickets.internal_ws_credential()
        with pytest.raises(TicketInvalid):
            ws_tickets.consume_internal_credential("not-the-credential")

    def test_reset_clears_and_remints(self):
        first = ws_tickets.internal_ws_credential()
        _reset_for_tests()
        # The old value no longer validates after reset.
        with pytest.raises(TicketInvalid):
            ws_tickets.consume_internal_credential(first)
        # A fresh mint produces a different value.
        second = ws_tickets.internal_ws_credential()
        assert second != first
        assert ws_tickets.consume_internal_credential(second)["user_id"] == (
            ws_tickets.INTERNAL_USER_ID
        )

    def test_independent_of_ticket_store(self):
        """The internal credential is not a ticket — minting tickets doesn't
        touch it, and consuming the credential doesn't consume tickets."""
        cred = ws_tickets.internal_ws_credential()
        ticket = mint_ticket(user_id="u1", provider="nous")
        # Consuming the internal credential leaves the ticket intact.
        ws_tickets.consume_internal_credential(cred)
        assert consume_ticket(ticket)["user_id"] == "u1"
