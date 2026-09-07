"""Tests for the shared-metrics sender.

Covers the four contract responses, the period-based consent gate, frozen
identity across rotation, transactional claiming, and the invariant that
matters most: a package file is never deleted, because the outbox is the
user's local history rather than a send queue.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_sender import (
    MAX_ATTEMPTS,
    MAX_PACKAGES_PER_PASS,
    MAX_SEND_ATTEMPTS,
    REQUEST_TIMEOUT_SECONDS,
    SharedMetricsSender,
    reconcile_send_consent,
)
from hermes_cli.sqlite_util import write_txn

INSTALL_ID = "12a73e97-4de9-4766-830d-9ca1192c0420"
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
ENDPOINT = "https://telemetry.test/v1/telemetry"


class FakeResponse:
    def __init__(self, status, retry_after=None, body=""):
        self.status = status
        self.retry_after = retry_after
        self.body = body


class FakeTransport:
    """Records every POST and replays a scripted sequence of responses."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, endpoint, payload, *, timeout):
        self.calls.append({"endpoint": endpoint, "payload": payload, "timeout": timeout})
        if not self._responses:
            return FakeResponse(202)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def bodies(self):
        return [json.loads(c["payload"].decode("utf-8")) for c in self.calls]


@pytest.fixture
def store(tmp_path):
    """A store with a broad consent window already open.

    Most tests exercise claiming/retry/transport, not the consent gate, and
    the interval gate fails closed with no window. One window opened before
    every test package and confirmed well past NOW keeps those tests about
    what they are about. Gate tests clear it via _clear_consent.
    """
    built = SharedMetricsStore(
        database_path=tmp_path / "metrics.sqlite3",
        outbox_directory=tmp_path / "outbox",
    )
    _grant_consent(built)
    return built


def _grant_consent(
    store,
    opened=datetime(2026, 8, 20, tzinfo=timezone.utc),
    confirmed_through=datetime(2026, 10, 1, tzinfo=timezone.utc),
):
    """Open a consent window and heartbeat it forward, via the real writer."""
    with store._connection() as connection:
        with write_txn(connection):
            reconcile_send_consent(connection, True, now=opened)
            reconcile_send_consent(connection, True, now=confirmed_through)


def _revoke_consent(store, at):
    with store._connection() as connection:
        with write_txn(connection):
            reconcile_send_consent(connection, False, now=at)


def _clear_consent(store):
    """Remove all consent state, for tests of the fail-closed default."""
    with store._connection() as connection:
        with write_txn(connection):
            connection.execute("DELETE FROM send_consent_windows")
            connection.execute("DELETE FROM consent_marks")


def _add_package(store, package_id, period_day, *, exported=True, install_id=INSTALL_ID):
    payload = {
        "schema_version": "hermes.shared_metrics.v2",
        "package_id": package_id,
        "install_id": install_id,
        "period_start": f"{period_day}T00:00:00Z",
        "period_end": f"{period_day}T23:59:59Z",
        "metrics": [{"name": "hermes.client.active", "type": "counter", "value": 1}],
    }
    with store._connection() as connection:
        connection.execute(
            """
            INSERT INTO package_outbox(
                package_id, period_start, period_end, payload_json,
                created_at, exported_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                package_id,
                f"{period_day}T00:00:00Z",
                f"{period_day}T23:59:59Z",
                json.dumps(payload),
                f"{period_day}T01:00:00Z",
                f"{period_day}T01:00:01Z" if exported else None,
            ),
        )
    path = store.outbox_directory / f"{package_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _row(store, package_id):
    with store._connection() as connection:
        row = connection.execute(
            """
            SELECT send_state, sent_at, send_attempts, next_attempt_at,
                   last_error, sent_install_id
            FROM package_outbox WHERE package_id = ?
            """,
            (package_id,),
        ).fetchone()
    return dict(
        send_state=row[0],
        sent_at=row[1],
        send_attempts=row[2],
        next_attempt_at=row[3],
        last_error=row[4],
        sent_install_id=row[5],
    )


def _iso(moment):
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sender(store, transport, **kwargs):
    return SharedMetricsSender(
        store,
        ENDPOINT,
        post=transport,
        sleep=lambda _s: None,
        now=lambda: NOW,
        **kwargs,
    )


class TestContractResponses:
    def test_202_marks_sent(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(202))
        outcome = _sender(store, transport).send_pending()
        assert outcome.sent == 1
        row = _row(store, "pkg-1")
        assert row["send_state"] == "sent"
        assert row["sent_at"] is not None

    def test_400_is_permanent_and_never_retried(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(400, body='{"error":"invalid_envelope"}'))
        outcome = _sender(store, transport).send_pending()
        assert outcome.rejected == 1
        assert len(transport.calls) == 1, "a 400 must not be retried"
        assert _row(store, "pkg-1")["send_state"] == "rejected"

        # A later pass must not pick it up again.
        transport2 = FakeTransport(FakeResponse(202))
        _sender(store, transport2).send_pending()
        assert transport2.calls == []

    @pytest.mark.parametrize("status", [401, 403, 404, 422, 500, 503])
    def test_unspecified_statuses_are_retried_not_discarded(self, store, status):
        """403 is the ingest origin guard; a bad edge config must not lose data."""
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(*[FakeResponse(status)] * 3)
        outcome = _sender(store, transport).send_pending()
        assert outcome.deferred == 1
        assert _row(store, "pkg-1")["send_state"] == "pending"

    def test_413_is_permanent(self, store):
        """A package over the 1 MiB cap cannot shrink by being retried."""
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(413))
        outcome = _sender(store, transport).send_pending()
        assert outcome.rejected == 1
        assert len(transport.calls) == 1

    def test_429_defers_using_retry_after(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(429, retry_after="120"))
        outcome = _sender(store, transport).send_pending()
        assert outcome.deferred == 1
        assert len(transport.calls) == 1, "429 waits rather than burning attempts"
        row = _row(store, "pkg-1")
        assert row["send_state"] == "pending"
        assert row["next_attempt_at"] == "2026-08-26T12:02:00Z"

    def test_429_without_retry_after_still_defers(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(429))
        _sender(store, transport).send_pending()
        assert _row(store, "pkg-1")["next_attempt_at"] > "2026-08-26T12:00:00Z"

    def test_absurd_retry_after_is_clamped(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(429, retry_after="99999999"))
        _sender(store, transport).send_pending()
        # clamped to 24h, not years
        assert _row(store, "pkg-1")["next_attempt_at"] <= "2026-08-27T12:00:00Z"

    def test_5xx_retries_then_defers(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(
            FakeResponse(503), FakeResponse(503), FakeResponse(503)
        )
        outcome = _sender(store, transport).send_pending()
        assert outcome.deferred == 1
        assert len(transport.calls) == 3, "three in-process attempts"
        assert _row(store, "pkg-1")["send_state"] == "pending"

    def test_5xx_then_success_within_the_same_pass(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(503), FakeResponse(202))
        outcome = _sender(store, transport).send_pending()
        assert outcome.sent == 1
        assert len(transport.calls) == 2

    def test_transport_failure_is_retryable(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(
            OSError("offline"), OSError("offline"), FakeResponse(202)
        )
        outcome = _sender(store, transport).send_pending()
        assert outcome.sent == 1

    def test_persistent_offline_defers_without_raising(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(*[OSError("offline")] * 3)
        outcome = _sender(store, transport).send_pending()
        assert outcome.deferred == 1
        assert "OSError" in _row(store, "pkg-1")["last_error"]


class TestConsentGate:
    def test_packages_from_before_opt_in_are_never_sent(self, store):
        # Consent opens on Aug 24; the "old" package's period predates it.
        _clear_consent(store)
        _grant_consent(store, opened=datetime(2026, 8, 24, tzinfo=timezone.utc))
        _add_package(store, "old", "2026-08-20")
        _add_package(store, "new", "2026-08-26")
        transport = FakeTransport(FakeResponse(202))
        _sender(store, transport).send_pending()
        assert [b["package_id"] for b in transport.bodies] == ["new"]

    def test_a_period_straddling_opt_in_day_is_sent_whole(self, store):
        """The head/tail bug: both packages for the opt-in period must go."""
        _add_package(store, "head", "2026-08-26")
        _add_package(store, "tail", "2026-08-26")  # created later, same period
        transport = FakeTransport(FakeResponse(202), FakeResponse(202))
        _sender(store, transport).send_pending()
        assert sorted(b["package_id"] for b in transport.bodies) == ["head", "tail"]

    def test_opt_in_is_immortalised_as_a_window_not_a_day(self, store):
        """The window survives replayed observations without moving."""
        with store._connection() as connection:
            rows = connection.execute(
                "SELECT opened_at, closed_at FROM send_consent_windows"
            ).fetchall()
        assert len(rows) == 1 and rows[0][1] is None
        _grant_consent(store)  # replay: must not create a second window
        with store._connection() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM send_consent_windows"
            ).fetchone()[0]
        assert count == 1

    def test_no_consent_window_means_nothing_is_sent(self, store):
        """The gate fails closed: absence of a window is absence of consent."""
        _clear_consent(store)
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(202))
        _sender(store, transport).send_pending()
        assert transport.calls == []

    def test_unexported_packages_are_skipped(self, store):
        _add_package(store, "pending-export", "2026-08-26", exported=False)
        transport = FakeTransport(FakeResponse(202))
        _sender(store, transport).send_pending()
        assert transport.calls == []

    def test_revoking_then_re_enabling_never_releases_the_off_window(self, store):
        """The R3/R5 leak: re-opt-in must not release the refused interval.

        Under the interval model the refused days fall BETWEEN two windows;
        no later observation can place them inside one, so the property holds
        for any number of on/off cycles — not just the single cycle the old
        moving day-stamp was patched to survive.
        """
        _clear_consent(store)
        _grant_consent(store, opened=NOW - timedelta(days=2), confirmed_through=NOW)
        _add_package(store, "consented", "2026-08-25")

        # User turns sending off; packages keep being collected for 3 days.
        _revoke_consent(store, at=NOW)
        for day in ("2026-08-27", "2026-08-28", "2026-08-29"):
            _add_package(store, f"refused-{day}", day)

        # User re-enables 5 days later; heartbeat confirms past the horizon.
        later = NOW + timedelta(days=5)
        with store._connection() as connection:
            with write_txn(connection):
                reconcile_send_consent(connection, True, now=later)
                reconcile_send_consent(
                    connection, True, now=later + timedelta(days=30)
                )

        transport = FakeTransport(*[FakeResponse(202)] * 10)
        SharedMetricsSender(
            store, ENDPOINT, post=transport, sleep=lambda _s: None, now=lambda: later
        ).send_pending()

        sent = [json.loads(c["payload"])["package_id"] for c in transport.calls]
        assert not any("refused" in pid for pid in sent), (
            f"transmitted packages collected while sending was off: {sent}"
        )
        # And the interval model's improvement over the day-stamp: the
        # pre-revocation consented package is NOT collateral damage.
        assert "consented" in sent, (
            "the consented backlog was destroyed by the revoke/re-enable cycle"
        )

    def test_a_package_from_after_re_enabling_is_sent(self, store):
        """The revocation handling must not wedge sending off permanently."""
        _clear_consent(store)
        _grant_consent(store, opened=NOW - timedelta(days=2), confirmed_through=NOW)
        _revoke_consent(store, at=NOW)

        later = NOW + timedelta(days=5)
        with store._connection() as connection:
            with write_txn(connection):
                reconcile_send_consent(connection, True, now=later)
                reconcile_send_consent(
                    connection, True, now=later + timedelta(days=10)
                )
        _add_package(store, "after-re-optin", (later + timedelta(days=1)).date().isoformat())
        transport = FakeTransport(FakeResponse(202))
        SharedMetricsSender(
            store, ENDPOINT, post=transport, sleep=lambda _s: None,
            now=lambda: later + timedelta(days=2),
        ).send_pending()
        assert len(transport.calls) == 1


class TestIdentity:
    def test_the_stable_install_id_is_transmitted_as_is(self, store):
        """Product decision 2026-08-27: no pseudonymization.

        The wire body carries the profile-scoped install_id verbatim. This
        test is the deliberate inversion of the pre-decision assertion that
        the raw id never crossed the wire.
        """
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(202))
        _sender(store, transport).send_pending()
        assert transport.bodies[0]["install_id"] == INSTALL_ID

    def test_transmitted_id_is_frozen_on_the_row(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(503), FakeResponse(202))
        _sender(store, transport).send_pending()
        assert _row(store, "pkg-1")["sent_install_id"] == transport.bodies[0]["install_id"]

    def test_retries_send_identical_bytes(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(503), FakeResponse(503), FakeResponse(202))
        _sender(store, transport).send_pending()
        payloads = {c["payload"] for c in transport.calls}
        assert len(payloads) == 1, "a resend must be byte-identical per the contract"

    def test_only_install_id_differs_from_the_stored_package(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(202))
        _sender(store, transport).send_pending()
        sent = transport.bodies[0]
        with store._connection() as connection:
            stored = json.loads(
                connection.execute(
                    "SELECT payload_json FROM package_outbox WHERE package_id = 'pkg-1'"
                ).fetchone()[0]
            )
        assert set(sent) == set(stored)
        for key in stored:
            if key != "install_id":
                assert sent[key] == stored[key]


class TestOutboxIsNotAQueue:
    def test_a_sent_package_file_is_not_deleted(self, store):
        path = _add_package(store, "pkg-1", "2026-08-26")
        _sender(store, FakeTransport(FakeResponse(202))).send_pending()
        assert path.exists(), "the outbox is the user's history, not a send queue"

    def test_a_rejected_package_file_is_not_deleted(self, store):
        path = _add_package(store, "pkg-1", "2026-08-26")
        _sender(store, FakeTransport(FakeResponse(400))).send_pending()
        assert path.exists()

    def test_the_package_row_survives_sending(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        _sender(store, FakeTransport(FakeResponse(202))).send_pending()
        with store._connection() as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM package_outbox WHERE package_id = 'pkg-1'"
            ).fetchone()[0] == 1


class TestClaimingAndBounds:
    def test_a_sent_package_is_not_resent(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        _sender(store, FakeTransport(FakeResponse(202))).send_pending()
        second = FakeTransport(FakeResponse(202))
        _sender(store, second).send_pending()
        assert second.calls == []

    def test_a_deferred_package_is_skipped_until_due(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        _sender(store, FakeTransport(FakeResponse(429, retry_after="600"))).send_pending()
        second = FakeTransport(FakeResponse(202))
        _sender(store, second).send_pending()
        assert second.calls == [], "backoff must survive within the same process"

    def test_a_deferred_package_is_retried_once_due(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        _sender(store, FakeTransport(FakeResponse(429, retry_after="60"))).send_pending()

        later = SharedMetricsSender(
            store,
            ENDPOINT,
            post=(transport := FakeTransport(FakeResponse(202))),
            sleep=lambda _s: None,
            now=lambda: NOW + timedelta(minutes=5),
        )
        later.send_pending()
        assert len(transport.calls) == 1

    def test_attempts_are_counted(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        _sender(store, FakeTransport(FakeResponse(429))).send_pending()
        assert _row(store, "pkg-1")["send_attempts"] == 1

    def test_a_pass_is_bounded(self, store):
        for i in range(MAX_PACKAGES_PER_PASS + 5):
            _add_package(store, f"pkg-{i:02d}", "2026-08-26")
        transport = FakeTransport(*[FakeResponse(202)] * 40)
        outcome = _sender(store, transport).send_pending()
        assert outcome.sent == MAX_PACKAGES_PER_PASS

    def test_two_concurrent_passes_do_not_double_send(self, store):
        """Claiming is what stops two Hermes processes duplicating work.

        The second pass must RECORD what it saw rather than raise: _send_one
        catches every exception as a retryable transport failure, so an
        assertion thrown inside a transport would be swallowed and this test
        would pass no matter what the claim did.
        """
        _add_package(store, "pkg-1", "2026-08-26")

        first_calls = []
        second_calls = []

        def second_transport(endpoint, payload, *, timeout):
            second_calls.append(payload)
            return FakeResponse(202)

        def transport(endpoint, payload, *, timeout):
            first_calls.append(payload)
            # A second sender runs while the first is mid-flight.
            SharedMetricsSender(
                store,
                ENDPOINT,
                post=second_transport,
                sleep=lambda _s: None,
                now=lambda: NOW,
            ).send_pending()
            return FakeResponse(202)

        _sender(store, transport).send_pending()
        assert len(first_calls) == 1
        assert second_calls == [], (
            "a concurrent pass claimed a package already in flight"
        )

    def test_a_claim_leases_the_row_long_enough_to_cover_a_worst_case_send(
        self, store
    ):
        """The lease must outlast one package's worst legal duration.

        Asserting merely "in the future" passed for a 1-second lease, which is
        useless: a package can legally take three 30s timeouts plus backoff.
        """
        _add_package(store, "pkg-1", "2026-08-26")
        claimed = _sender(store, FakeTransport())._claim_next(NOW, set())
        assert claimed is not None

        worst_case = REQUEST_TIMEOUT_SECONDS * MAX_ATTEMPTS + 1 + 5 + 25
        deadline = NOW + timedelta(seconds=worst_case)
        assert _row(store, "pkg-1")["next_attempt_at"] >= _iso(deadline), (
            "lease expires before a single package can legally finish"
        )

    def test_a_slow_multi_package_pass_does_not_lose_its_lease(self, store):
        """Regression: a batch-wide lease expired while later rows were sent.

        One package can legally take ~96s (three 30s timeouts plus backoff).
        With 20 rows claimed under one shared lease, the later rows' leases
        expired mid-pass and a second process re-sent them. Packages are now
        claimed one at a time, immediately before transmission.
        """
        for i in range(3):
            _add_package(store, f"pkg-{i}", "2026-08-26")

        clock = {"t": NOW}
        first_posts, second_posts = [], []


        def transport(endpoint, payload, *, timeout):
            pid = json.loads(payload)["package_id"]
            first_posts.append(pid)
            # Burn the worst-case time budget for a single package.
            clock["t"] += timedelta(seconds=96)
            # A concurrent process probes for work while this package is still
            # in flight. It must not be able to claim the package we hold.
            # Restricted to that package so the probe cannot legitimately pick
            # up the OTHER pending rows and make the assertion ambiguous.
            held = _row(store, pid)
            if held["next_attempt_at"] is not None:
                eligible = held["next_attempt_at"] <= _iso(clock["t"])
                if eligible and held["send_state"] != "sent":
                    second_posts.append(pid)
            return FakeResponse(202)

        SharedMetricsSender(
            store,
            ENDPOINT,
            post=transport,
            sleep=lambda _s: None,
            now=lambda: clock["t"],
        ).send_pending()

        assert sorted(first_posts) == ["pkg-0", "pkg-1", "pkg-2"]
        assert second_posts == [], (
            f"a concurrent pass re-sent {second_posts} after a lease expired"
        )

    def test_a_re_eligible_head_row_does_not_starve_the_tail(self, store):
        """Regression: `seen` terminated the pass instead of skipping a row.

        The claim query is LIMIT 1. When the oldest row was already handled
        this pass but had become eligible again (short Retry-After, or a pass
        outliving the 15-minute failure backoff), _claim_next returned None
        and send_pending read that as "queue empty", abandoning every healthy
        package behind it. Measured: 10 of 19 delivered.
        """
        _add_package(store, "aaa-head", "2026-08-26")
        for i in range(5):
            _add_package(store, f"zzz-{i}", "2026-08-26")
        # Order by created_at puts the head first.
        with store._connection() as connection:
            connection.execute(
                "UPDATE package_outbox SET created_at = '2026-08-26T00:00:00Z'"
                " WHERE package_id = 'aaa-head'"
            )

        posts = []

        def transport(endpoint, payload, *, timeout):
            pid = json.loads(payload)["package_id"]
            posts.append(pid)
            if pid == "aaa-head":
                # Well-behaved service: retry in one second, so the head is
                # eligible again immediately.
                return FakeResponse(429, retry_after="1")
            return FakeResponse(202)

        clock = {"t": NOW}
        SharedMetricsSender(
            store,
            ENDPOINT,
            post=transport,
            sleep=lambda _s: None,
            now=lambda: clock["t"] + timedelta(seconds=30 * len(posts)),
        ).send_pending()

        delivered = {p for p in posts if p.startswith("zzz")}
        assert delivered == {f"zzz-{i}" for i in range(5)}, (
            f"tail starved by a re-eligible head row; delivered {delivered}"
        )

    def test_a_poisoned_package_is_abandoned_eventually(self, store):
        """Without a ceiling a doomed row is retried ~160 times over 30 days.

        Drives the real loop rather than pre-setting a counter: a row seeded
        at exactly the limit is also excluded by other predicates, so that
        version of this test passed even with the ceiling removed.
        """
        _add_package(store, "pkg-1", "2026-08-26")

        clock = {"t": NOW}
        attempts = []

        def transport(endpoint, payload, *, timeout):
            attempts.append(1)
            return FakeResponse(503)

        # Run many passes, always well past any backoff, as a month of hook
        # fires against a permanently failing package would.
        for i in range(60):
            SharedMetricsSender(
                store,
                ENDPOINT,
                post=transport,
                sleep=lambda _s: None,
                now=lambda: clock["t"] + timedelta(hours=i),
            ).send_pending()

        row = _row(store, "pkg-1")
        assert row["send_attempts"] <= MAX_SEND_ATTEMPTS, (
            f"package retried {row['send_attempts']} times with no ceiling"
        )
        assert len(attempts) < 100, (
            f"{len(attempts)} requests burned on one doomed package"
        )

    def test_a_lapsed_claimant_yields_even_before_anyone_reclaims(self, store):
        """Seventh review: the check-to-POST expiry race.

        A claims, sleeps past its own lease, and wakes BEFORE any other
        process reclaims. Its token is still in the row, so a read-only
        ownership check passes — and then B reclaims while A's POST is in
        flight: both send. The pre-POST renewal must instead REJECT a
        claimant whose lease already expired, whether or not anyone has
        reclaimed yet, because expiry alone means another process may claim
        at any moment.
        """
        _add_package(store, "pkg-1", "2026-08-26")

        posts = []
        sender_a = SharedMetricsSender(
            store, ENDPOINT,
            post=lambda e, p, *, timeout: (posts.append("A"), FakeResponse(202))[1],
            sleep=lambda _s: None,
            now=lambda: clock["t"],
        )
        clock = {"t": NOW}
        claimed = sender_a._claim_next(NOW, set())
        assert claimed is not None and not claimed["skip"]

        # Suspended past the 300s lease; wakes with the row NOT yet reclaimed.
        clock["t"] = NOW + timedelta(seconds=400)
        result = sender_a._send_one(claimed)

        assert posts == [], (
            "a claimant with an expired lease transmitted before renewal"
        )
        assert result == "deferred"
        # The row must remain claimable by the next process.
        row = _row(store, "pkg-1")
        assert row["send_state"] == "pending"

    def test_renewal_extends_the_lease_across_the_post(self, store):
        """A healthy in-lease claimant renews and its POST is covered.

        Round-8 review: the original assertion was `>=` under a frozen
        clock, which a renewal that matches the row but never extends the
        lease also satisfies — the exact mutant that double-POSTs (the
        un-extended lease expires mid-POST and a second process reclaims).
        The renewal must move the deadline STRICTLY forward to now + lease,
        so renew from a later clock and require the exact new deadline.
        """
        _add_package(store, "pkg-1", "2026-08-26")
        clock = {"t": NOW}
        sender = SharedMetricsSender(
            store,
            ENDPOINT,
            post=lambda e, p, *, timeout: FakeResponse(202),
            sleep=lambda _s: None,
            now=lambda: clock["t"],
        )
        claimed = sender._claim_next(NOW, set())
        assert claimed is not None
        lease_before = _row(store, "pkg-1")["next_attempt_at"]

        # 100s into the (300s) lease: still healthy, renews mid-flight.
        clock["t"] = NOW + timedelta(seconds=100)
        assert sender._renew_claim("pkg-1", claimed["claim_token"]) is True
        lease_after = _row(store, "pkg-1")["next_attempt_at"]
        assert lease_after > lease_before, (
            "renewal granted authority without extending the lease"
        )
        # And not just 'later': the full fresh lease from the renewal clock.
        expected = (NOW + timedelta(seconds=100 + 300)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        assert lease_after == expected

    def test_a_lapsed_claimant_resuming_after_reclaim_cannot_double_post(
        self, store
    ):
        """PR-review P1: expiry -> reclaim -> old claimant resumes.

        A claims, then is suspended (laptop lid) BEFORE its POST. The lease
        expires; B reclaims and POSTs; A wakes and proceeds. The pre-POST
        ownership check must make A yield without transmitting.

        Scope note: the check closes the claim->POST gap. A suspension that
        lands mid-POST (bytes already leaving) is not client-fixable — that
        residual needs server-side dedupe and is documented on _send_one.
        """
        _add_package(store, "pkg-1", "2026-08-26")

        posts = []

        def post_a(endpoint, payload, *, timeout):
            posts.append("A")
            return FakeResponse(202)

        def post_b(endpoint, payload, *, timeout):
            posts.append("B")
            return FakeResponse(202)

        sender_a = SharedMetricsSender(
            store, ENDPOINT, post=post_a, sleep=lambda _s: None, now=lambda: NOW
        )
        # A claims, then the process is suspended before _send_one runs.
        claimed_a = sender_a._claim_next(NOW, set())
        assert claimed_a is not None and not claimed_a["skip"]

        # 400s later (past the 300s lease) B claims and completes the send.
        later = NOW + timedelta(seconds=400)
        sender_b = SharedMetricsSender(
            store, ENDPOINT, post=post_b, sleep=lambda _s: None, now=lambda: later
        )
        outcome_b = sender_b.send_pending()
        assert outcome_b.sent == 1

        # A resumes exactly where it left off.
        result_a = sender_a._send_one(claimed_a)

        row = _row(store, "pkg-1")
        assert posts == ["B"], (
            f"a lapsed claimant transmitted after reclaim: {posts}"
        )
        assert result_a == "deferred"
        assert row["send_state"] == "sent", "B's settlement must stand"

    def test_a_lapsed_claimants_backoff_cannot_clobber_the_new_claim(self, store):
        """The token must fence DEFERS too, not just the 202 settlement.

        A's transport fails after B has reclaimed; A's backoff write must
        not move next_attempt_at under B's live lease.
        """
        _add_package(store, "pkg-1", "2026-08-26")
        sender_a = SharedMetricsSender(
            store, ENDPOINT,
            post=FakeTransport(OSError("net"), OSError("net"), OSError("net")),
            sleep=lambda _s: None, now=lambda: NOW,
        )
        claimed_a = sender_a._claim_next(NOW, set())
        assert claimed_a is not None and not claimed_a["skip"]

        later = NOW + timedelta(seconds=400)
        sender_b = SharedMetricsSender(
            store, ENDPOINT, post=FakeTransport(),
            sleep=lambda _s: None, now=lambda: later,
        )
        claimed_b = sender_b._claim_next(later, set())
        assert claimed_b is not None and not claimed_b["skip"]
        lease_b = _row(store, "pkg-1")["next_attempt_at"]

        # A's exhausted retries try to write a 15-minute backoff.
        result = sender_a._send_one(claimed_a)
        assert result == "deferred"
        assert _row(store, "pkg-1")["next_attempt_at"] == lease_b, (
            "a lapsed claimant's backoff overwrote the live claim's lease"
        )

    def test_an_expired_lease_is_reclaimed(self, store):
        """A process killed mid-pass must not strand its packages."""
        _add_package(store, "pkg-1", "2026-08-26")
        _sender(store, FakeTransport(OSError("killed"), OSError(""), OSError(""))).send_pending()

        later = SharedMetricsSender(
            store,
            ENDPOINT,
            post=(transport := FakeTransport(FakeResponse(202))),
            sleep=lambda _s: None,
            now=lambda: NOW + timedelta(hours=2),
        )
        later.send_pending()
        assert len(transport.calls) == 1

    def test_a_lapsed_sender_cannot_resurrect_a_sent_package(self, store):
        """Terminal state must win over a straggler's write."""
        _add_package(store, "pkg-1", "2026-08-26")
        _sender(store, FakeTransport(FakeResponse(202))).send_pending()
        assert _row(store, "pkg-1")["send_state"] == "sent"

        # A straggler from an earlier pass tries to defer the same row.
        _sender(store, FakeTransport())._defer("pkg-1", 600, "stale")
        assert _row(store, "pkg-1")["send_state"] == "sent", (
            "a lapsed pass overwrote a completed send"
        )


class TestResilience:
    def test_a_corrupt_row_does_not_stop_the_pass(self, store):
        _add_package(store, "good", "2026-08-26")
        with store._connection() as connection:
            connection.execute(
                """
                INSERT INTO package_outbox(
                    package_id, period_start, period_end, payload_json,
                    created_at, exported_at
                ) VALUES ('bad', '2026-08-26T00:00:00Z', '2026-08-26T23:59:59Z',
                          'not json', '2026-08-26T00:00:00Z', '2026-08-26T01:00:00Z')
                """
            )
        transport = FakeTransport(*[FakeResponse(202)] * 5)
        outcome = _sender(store, transport).send_pending()
        assert outcome.sent >= 1

    @pytest.mark.parametrize(
        "payload_json",
        [
            '["a", "list"]',
            "null",
            '"a string"',
            "42",
            '{"no_install_id": true}',
            '{"install_id": ""}',
            '{"install_id": null}',
        ],
    )
    def test_valid_json_that_is_not_a_usable_package_is_skipped(
        self, store, payload_json
    ):
        """Regression: a top-level array parsed fine, then .get() raised.

        The AttributeError escaped the claim transaction and blocked every
        healthy package behind it.
        """
        with store._connection() as connection:
            connection.execute(
                """
                INSERT INTO package_outbox(
                    package_id, period_start, period_end, payload_json,
                    created_at, exported_at
                ) VALUES ('bad', '2026-08-26T00:00:00Z', '2026-08-26T23:59:59Z',
                          ?, '2026-08-26T00:00:00Z', '2026-08-26T01:00:00Z')
                """,
                (payload_json,),
            )
        _add_package(store, "good", "2026-08-26")

        transport = FakeTransport(*[FakeResponse(202)] * 5)
        outcome = _sender(store, transport).send_pending()

        assert outcome.sent == 1, "the healthy package must still go out"
        assert [json.loads(c["payload"])["package_id"] for c in transport.calls] == [
            "good"
        ]
        assert _row(store, "bad")["send_state"] == "rejected"

    def test_send_pending_never_raises_on_a_broken_database(self, store, tmp_path):
        store.database_path.write_text("this is not a database")
        outcome = _sender(store, FakeTransport(FakeResponse(202))).send_pending()
        assert outcome.sent == 0


class TestConsentRevocation:
    """`send: false` must stop an in-flight pass, not just the next one."""

    def test_revoking_consent_mid_pass_stops_further_sends(self, store):
        for i in range(4):
            _add_package(store, f"pkg-{i}", "2026-08-26")

        consented = {"value": True}
        posts = []

        def transport(endpoint, payload, *, timeout):
            posts.append(json.loads(payload)["package_id"])
            consented["value"] = False  # user flips send off during the pass
            return FakeResponse(202)

        outcome = SharedMetricsSender(
            store,
            ENDPOINT,
            post=transport,
            sleep=lambda _s: None,
            now=lambda: NOW,
            consent_check=lambda: consented["value"],
        ).send_pending()

        assert len(posts) == 1, f"kept sending after consent was revoked: {posts}"
        assert outcome.sent == 1

    def test_no_send_at_all_when_consent_is_already_false(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        posts = []
        SharedMetricsSender(
            store,
            ENDPOINT,
            post=lambda *a, **k: posts.append(1) or FakeResponse(202),
            sleep=lambda _s: None,
            now=lambda: NOW,
            consent_check=lambda: False,
        ).send_pending()
        assert posts == []

    def test_an_unreadable_consent_check_fails_closed(self, store):
        """If consent cannot be established, do not transmit."""
        _add_package(store, "pkg-1", "2026-08-26")
        posts = []

        def explode():
            raise OSError("config unreadable")

        SharedMetricsSender(
            store,
            ENDPOINT,
            post=lambda *a, **k: posts.append(1) or FakeResponse(202),
            sleep=lambda _s: None,
            now=lambda: NOW,
            consent_check=explode,
        ).send_pending()
        assert posts == []


class TestCompression:
    """Compression lives in the real transport, so exercise _post directly."""

    def _captured_request(self, payload: bytes):
        import urllib.request

        from hermes_cli.observability import shared_metrics_sender as mod

        captured = {}

        class FakeConn:
            status = 202
            headers = {}

            def read(self, _n=None):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(request, timeout=None):
            captured["data"] = request.data
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            return FakeConn()

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            mod._post(ENDPOINT, payload, timeout=5)
        finally:
            urllib.request.urlopen = original
        return captured

    def test_large_payloads_are_gzipped(self):
        payload = json.dumps({"filler": "x" * 20000}).encode("utf-8")
        captured = self._captured_request(payload)
        assert captured["data"][:2] == b"\x1f\x8b", "gzip magic bytes"
        assert captured["headers"].get("Content-encoding".lower()) == "gzip"

    def test_gzip_actually_shrinks_the_body(self):
        payload = json.dumps({"filler": "x" * 20000}).encode("utf-8")
        captured = self._captured_request(payload)
        assert len(captured["data"]) < len(payload)

    def test_gzip_is_deterministic_across_time(self):
        """Kills the mtime footgun: gzip embeds a timestamp by default.

        The in-pass retry test cannot catch this — both attempts compress
        within the same second. Compressing the same bytes at two different
        wall-clock seconds is what actually exercises mtime=0.
        """
        import time as _time

        payload = json.dumps({"filler": "x" * 20000}).encode("utf-8")
        first = self._captured_request(payload)["data"]
        _time.sleep(1.1)
        second = self._captured_request(payload)["data"]
        assert first == second, (
            "gzip output changed between seconds — mtime is being embedded"
        )

    def test_small_payloads_are_sent_plain(self):
        payload = b'{"small": true}'
        captured = self._captured_request(payload)
        assert captured["data"] == payload
        assert "content-encoding" not in captured["headers"]
