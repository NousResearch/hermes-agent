"""Property tests for the consent-interval model.

Ported from the /tmp validation harness that gated the redesign: every
scenario here is a defect that actually occurred (rounds 3-5) or a clock
adversary the day-stamp model could not survive. The v1 and v2 drafts of the
redesign each FAILED scenarios in this file before shipping — that is the
harness working, and why these run against the real store and the real
reconciler rather than a model of them.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_sender import (
    CONSENT_GATE_SQL,
    reconcile_send_consent,
)
from hermes_cli.sqlite_util import write_txn

T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)


def ts(days=0, hours=0):
    return (T0 + timedelta(days=days, hours=hours)).isoformat().replace(
        "+00:00", "Z"
    )


def dt(days=0, hours=0):
    return T0 + timedelta(days=days, hours=hours)


@pytest.fixture
def store(tmp_path):
    return SharedMetricsStore(
        database_path=tmp_path / "m.db", outbox_directory=tmp_path / "o"
    )


def _add(store, pid, start, end):
    """Store a package the way the generator does: at period end."""
    with store._connection() as connection:
        with write_txn(connection):
            connection.execute(
                "INSERT INTO package_outbox(package_id, period_start, period_end,"
                " payload_json, created_at, exported_at) VALUES (?, ?, ?, ?, ?, ?)",
                (pid, start, end, json.dumps({"package_id": pid}), end, end),
            )
            connection.execute(
                """INSERT INTO consent_marks(name, stamp) VALUES ('data', ?)
                   ON CONFLICT(name) DO UPDATE SET stamp = MAX(stamp, excluded.stamp)""",
                (end,),
            )


def _observe(store, send_enabled, when):
    with store._connection() as connection:
        with write_txn(connection):
            reconcile_send_consent(connection, send_enabled, now=when)


def _eligible(store):
    with store._connection() as connection:
        return sorted(
            row[0]
            for row in connection.execute(
                f"SELECT package_id FROM package_outbox WHERE {CONSENT_GATE_SQL}"
            )
        )


def _windows(store):
    with store._connection() as connection:
        return [
            tuple(row)
            for row in connection.execute(
                "SELECT opened_at, last_confirmed_at, closed_at"
                " FROM send_consent_windows ORDER BY opened_at"
            )
        ]


class TestRefusedWindowIsNeverReleased:
    def test_on_off_on_with_realistic_interleaving(self, store):
        """Rounds 3 and 5: the refused middle must never transmit, and
        neither consented era may be lost."""
        _observe(store, True, dt(0))
        for n in range(5):
            _add(store, f"d{n:02d}", ts(days=n), ts(days=n + 1))
            _observe(store, True, dt(days=n + 1))
        _observe(store, False, dt(5))
        for n in range(5, 10):
            _add(store, f"d{n:02d}", ts(days=n), ts(days=n + 1))
        _observe(store, True, dt(10))
        for n in range(10, 15):
            _add(store, f"d{n:02d}", ts(days=n), ts(days=n + 1))
            _observe(store, True, dt(days=n + 1))

        eligible = _eligible(store)
        assert not [p for p in eligible if 5 <= int(p[1:]) < 10], eligible
        assert [f"d{n:02d}" for n in range(5)] == eligible[:5], (
            "pre-revocation consented backlog was destroyed"
        )
        assert [f"d{n:02d}" for n in range(10, 15)] == eligible[5:], eligible

    def test_hand_edit_with_a_90_day_silent_gap(self, store):
        """Round 5 D1, strongest form: NOTHING observes the off window.

        The close back-dates to the last confirmed moment, so the unobserved
        gap is outside every window and fails closed.
        """
        _observe(store, True, dt(0))
        _add(store, "consented", ts(0, 1), ts(0, 2))
        _observe(store, True, dt(0, 6))
        for n in range(1, 90, 10):
            _add(store, f"REFUSED-d{n}", ts(days=n), ts(days=n, hours=1))
        _observe(store, False, dt(90))   # first observation: boot on day 90
        _observe(store, True, dt(91))
        _observe(store, True, dt(92))

        eligible = _eligible(store)
        assert not [p for p in eligible if p.startswith("REFUSED")], eligible
        assert "consented" in eligible, (
            "the confirmed-morning package must survive the reconciliation"
        )


class TestClockAdversaries:
    def test_forward_poison_then_revoke_releases_nothing(self, store):
        """Round 6 D1: one glitched-forward sample must not defeat a close.

        Unfixed, the poisoned obs mark dragged last_confirmed_at to 2099, a
        later revoke stamped closed_at = 2099, and the closed window then
        CONTAINED every refused period that followed — all 8 refused
        packages became eligible. The close now clamps to the closing
        observation's own raw stamp, so an honest clock at revoke time pulls
        the window back to the true revoke moment.
        """
        _observe(store, True, dt(0))
        _observe(store, True, datetime(2099, 1, 1, tzinfo=timezone.utc))
        _observe(store, False, dt(1))            # honest clock at revoke
        for n in range(2, 10):
            _add(store, f"REFUSED-{n}", ts(days=n), ts(days=n, hours=2))

        leaked = [p for p in _eligible(store) if p.startswith("REFUSED")]
        assert not leaked, f"poisoned horizon released refused data: {leaked}"

    def test_forward_poison_cannot_wedge_consent_forever(self, store):
        """The obs-advance cap bounds the damage of one insane sample.

        Uncapped, a 2099 sample would clamp every future window open at
        2099, suppressing consented data for decades (fail-closed but
        permanent). Capped, the mark moves at most MAX_OBS_ADVANCE_SECONDS
        past its previous value, so honest time overtakes it.
        """
        from hermes_cli.observability.shared_metrics_sender import (
            MAX_OBS_ADVANCE_SECONDS,
        )

        _observe(store, True, dt(0))
        _observe(store, True, datetime(2099, 1, 1, tzinfo=timezone.utc))
        with store._connection() as connection:
            stamp = connection.execute(
                "SELECT stamp FROM consent_marks WHERE name = 'obs'"
            ).fetchone()[0]
        ceiling = ts(days=MAX_OBS_ADVANCE_SECONDS // 86_400)
        assert stamp <= ceiling, (
            f"one glitched sample advanced the mark unboundedly: {stamp}"
        )

        # Consented data from shortly after the cap horizon still flows once
        # honest observations catch the marks up.
        horizon_days = MAX_OBS_ADVANCE_SECONDS // 86_400
        _add(
            store,
            "post-glitch",
            ts(days=horizon_days + 1),
            ts(days=horizon_days + 1, hours=4),
        )
        _observe(store, True, dt(days=horizon_days + 2))
        assert "post-glitch" in _eligible(store), (
            "consent wedged after a forward glitch"
        )

    def test_rollback_at_re_enable_releases_nothing(self, store):
        """Round 5 D2: the data mark clamps opens above existing packages."""
        _observe(store, True, dt(0))
        _observe(store, True, dt(5))
        _observe(store, False, dt(5))
        for n in range(1, 4):
            _add(store, f"REFUSED-{n}", ts(days=5, hours=n), ts(days=5, hours=n + 1))
        _observe(store, True, dt(-12))   # 12-day rollback at re-enable
        _observe(store, True, dt(-11))

        during = [p for p in _eligible(store) if p.startswith("REFUSED")]
        assert not during, f"rollback released refused packages: {during}"

        _observe(store, True, dt(20))    # clock recovers
        _observe(store, True, dt(21))
        after = [p for p in _eligible(store) if p.startswith("REFUSED")]
        assert not after, f"recovery released refused packages: {after}"

    def test_recovery_does_not_wedge_future_sending(self, store):
        _observe(store, True, dt(0))
        _observe(store, False, dt(5))
        _observe(store, True, dt(-12))
        _observe(store, True, dt(20))
        _add(store, "post-recovery", ts(21), ts(21, 4))
        _observe(store, True, dt(22))
        assert "post-recovery" in _eligible(store)


class TestSubDayGranularity:
    def test_intra_day_refusal_holds_back_the_whole_day_package(self, store):
        """Round 5 D3: a day package spanning a refused stretch must wait."""
        _observe(store, True, dt(0))
        _observe(store, True, dt(10, 9))
        _observe(store, False, dt(10, 9))
        _observe(store, True, dt(10, 18))
        _observe(store, True, dt(11, 2))
        _add(store, "halfday", ts(10), ts(11))
        assert "halfday" not in _eligible(store)


class TestReconcilerProperties:
    def test_idempotent_under_replay(self, store):
        for _ in range(4):
            _observe(store, True, dt(0))
        _observe(store, False, dt(2))
        for _ in range(5):
            _observe(store, False, dt(3))
        _observe(store, True, dt(4))
        for _ in range(3):
            _observe(store, True, dt(5))
        assert len(_windows(store)) == 2

    def test_the_observation_mark_is_monotonic(self, store):
        """A rolled-back clock must never lower the observation high-water.

        Every downstream guarantee leans on this: closes clamp to it via
        last_confirmed_at, and opens clamp to max(obs, data). Found as a
        surviving mutant (obs upsert rewritten from MAX to overwrite) —
        the leak scenarios happen to be covered by the data mark whenever a
        leakable package exists, but the property itself must hold on its
        own, not by coincidence of the sibling mark.
        """
        _observe(store, True, dt(5))
        _observe(store, True, dt(0))   # rollback
        with store._connection() as connection:
            stamp = connection.execute(
                "SELECT stamp FROM consent_marks WHERE name = 'obs'"
            ).fetchone()[0]
        assert stamp == ts(5), f"obs mark moved backwards: {stamp}"

    def test_the_real_package_writer_advances_the_data_mark(self, store):
        """Round 6 D2: the harness's _add re-implements the data-mark insert,
        so deleting the advance from the REAL writer survived 314 tests.
        This drives the production exporter instead.
        """
        from datetime import date, timedelta as _td

        yesterday = (date.today() - _td(days=1)).isoformat()
        with store._connection() as connection:
            with write_txn(connection):
                connection.execute(
                    "INSERT INTO counter_aggregates("
                    " period_start, metric_name, hermes_version, os_family,"
                    " architecture, install_method, dimensions_json, value,"
                    " packaged_value"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        yesterday, "hermes.client.active", "0.0.0-test",
                        "macos", "arm64", "git", "{}", 1, 0,
                    ),
                )

        exported = store.create_and_export_package_if_due()
        assert exported, "the generator was expected to export yesterday's period"

        with store._connection() as connection:
            row = connection.execute(
                "SELECT stamp FROM consent_marks WHERE name = 'data'"
            ).fetchone()
        assert row is not None and row[0] >= yesterday, (
            "the production package writer did not advance the data mark"
        )

    def test_the_gate_is_read_only(self, store):
        _observe(store, True, dt(0))
        before = _windows(store)
        for _ in range(10):
            _eligible(store)
        assert _windows(store) == before

    def test_no_window_fails_closed(self, store):
        _add(store, "orphan", ts(0), ts(1))
        assert _eligible(store) == []

    def test_fresh_package_waits_one_heartbeat_then_releases(self, store):
        """The documented latency cost of confirmation-based windows."""
        _observe(store, True, dt(0))
        _add(store, "fresh", ts(0, 1), ts(0, 2))
        assert _eligible(store) == []
        _observe(store, True, dt(0, 3))
        assert _eligible(store) == ["fresh"]
