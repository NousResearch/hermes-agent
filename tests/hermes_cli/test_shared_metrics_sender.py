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
    MAX_PACKAGES_PER_PASS,
    OPT_IN_PERIOD_KEY,
    SharedMetricsSender,
    opt_in_period,
)

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
    return SharedMetricsStore(
        database_path=tmp_path / "metrics.sqlite3",
        outbox_directory=tmp_path / "outbox",
    )


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

    def test_opt_in_day_is_recorded_once_and_does_not_move(self, store):
        with store._connection() as connection:
            first = opt_in_period(connection, now=NOW)
            later = opt_in_period(connection, now=NOW + timedelta(days=10))
        assert first == later == "2026-08-26"

    def test_opt_in_day_is_persisted(self, store):
        with store._connection() as connection:
            opt_in_period(connection, now=NOW)
            value = connection.execute(
                "SELECT value FROM telemetry_state WHERE key = ?", (OPT_IN_PERIOD_KEY,)
            ).fetchone()[0]
        assert value == "2026-08-26"

    def test_unexported_packages_are_skipped(self, store):
        _add_package(store, "pending-export", "2026-08-26", exported=False)
        transport = FakeTransport(FakeResponse(202))
        _sender(store, transport).send_pending()
        assert transport.calls == []


class TestIdentity:
    def test_install_id_is_never_transmitted(self, store):
        _add_package(store, "pkg-1", "2026-08-26")
        transport = FakeTransport(FakeResponse(202))
        _sender(store, transport).send_pending()
        raw = transport.calls[0]["payload"].decode("utf-8")
        assert INSTALL_ID not in raw
        assert transport.bodies[0]["install_id"] != INSTALL_ID

    def test_derived_id_is_frozen_on_the_row(self, store):
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

    def test_a_claim_leases_the_row_into_the_future(self, store):
        """The lease, not the send result, is what blocks a concurrent pass."""
        _add_package(store, "pkg-1", "2026-08-26")
        claimed = _sender(store, FakeTransport())._claim_next(NOW, set())
        assert claimed is not None
        assert _row(store, "pkg-1")["next_attempt_at"] > "2026-08-26T12:00:00Z"

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

    def test_gzip_round_trips_to_the_original_bytes(self):
        import gzip as gziplib

        payload = json.dumps({"filler": "x" * 20000}).encode("utf-8")
        captured = self._captured_request(payload)
        assert gziplib.decompress(captured["data"]) == payload

    def test_small_payloads_are_sent_plain(self):
        payload = b'{"small": true}'
        captured = self._captured_request(payload)
        assert captured["data"] == payload
        assert "content-encoding" not in captured["headers"]
