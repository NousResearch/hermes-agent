"""Transmit exported shared-metrics packages to the Nous telemetry service.

Implements the sender side of the ingest contract (see the telemetry repo's
``CONTRACT.md``):

* ``202`` — durably stored. Mark sent.
* ``400`` — permanently malformed. Never retry.
* ``429`` — keep, retry after ``Retry-After``.
* ``5xx`` / timeout / connection error — keep, retry with backoff.

Two properties are load-bearing and easy to get wrong:

**The outbox directory is the user's local history, not a queue.** Packages
are pruned by age; a ``202`` marks send state in SQLite and never deletes a
file. See Appendix A.7 of ``docs/observability/relay-shared-metrics.md``.

**Consent is gated on the package's PERIOD, not its creation time.** One
period is split across packages created on different days, so a created-at
gate would send a period's tail while dropping its head and silently
undercount the first consented day. The gate itself is interval containment:
the period must fall entirely inside a recorded consent window
(``send_consent_windows``), maintained by the single ``reconcile_send_consent``
writer below.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from hermes_cli.sqlite_util import write_txn

from .shared_metrics_identity import (
    current_salt,
    derive_install_id,
    substitute_install_id,
)

logger = logging.getLogger(__name__)

#: Contract recommends timing out at 30s and treating a timeout as retryable.
REQUEST_TIMEOUT_SECONDS = 30

#: In-process attempts per package per pass, then the package waits for a
#: later pass. Backoff is 1s/5s/25s with full jitter.
MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1
_BACKOFF_FACTOR = 5

#: Contract recommends gzip above roughly this size.
GZIP_THRESHOLD_BYTES = 4096

#: Packages per pass. Bounds work on an interactive hook even after an outage.
MAX_PACKAGES_PER_PASS = 20

#: How long a claimed row is held. The claim writes a LEASE INTO THE FUTURE:
#: selection requires `next_attempt_at <= now`, so for the length of the lease
#: no other process can take the package.
#:
#: This must exceed the worst case for ONE package — three 30s request
#: timeouts plus 1s+5s of backoff, about 96s — which is why packages are
#: claimed one at a time, immediately before being sent. An earlier revision
#: claimed up to 20 rows under a single shared lease; a full batch can legally
#: run ~1900s, so the later rows' leases expired while the pass still held
#: them in memory and another process re-sent them.
_CLAIM_LEASE_SECONDS = 300

#: Floor applied after a pass fails to deliver, so a hard-down service is not
#: retried on every task completion.
_FAILURE_BACKOFF_SECONDS = 15 * 60

#: Statuses that are permanent per the ingest contract. Deliberately narrow:
#: 400 means the envelope is malformed and will never validate. 413 is added
#: because a package over the service's 1 MiB cap cannot shrink on retry.
#: Everything else — including 403 from the origin guard and 404 from a bad
#: path — is retried, because those are usually deployment or edge
#: misconfiguration that resolves without the package changing.
_PERMANENT_STATUSES = frozenset({400, 413})

#: Attempts after which a package is abandoned. Without a ceiling a
#: permanently-poisoned row is retried until 30-day retention deletes it —
#: measured at ~160 requests — which wastes the user's bandwidth and keeps a
#: doomed package at the head of the queue.
MAX_SEND_ATTEMPTS = 25


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_stamp(value: str) -> datetime:
    """Parse a stamp this module itself wrote (Z-suffixed ISO-8601, UTC)."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


@dataclass
class SendOutcome:
    """What one pass did. Returned for tests and diagnostics."""

    sent: int = 0
    rejected: int = 0
    deferred: int = 0


class _Response:
    __slots__ = ("status", "retry_after", "body")

    def __init__(self, status: int, retry_after: str | None, body: str) -> None:
        self.status = status
        self.retry_after = retry_after
        self.body = body


def _post(endpoint: str, payload: bytes, *, timeout: int) -> _Response:
    """POST one package. Raises on transport failure; never on HTTP status."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "hermes-agent-shared-metrics/1",
    }
    body = payload
    if len(payload) > GZIP_THRESHOLD_BYTES:
        # mtime=0: gzip embeds a timestamp by default, which would make two
        # sends of one package differ on the wire. The service decompresses
        # before storing so it would not change what lands in S3, but a
        # deterministic body keeps "a resend is byte-identical" true at the
        # transport layer too, and makes the property testable.
        body = gzip.compress(payload, mtime=0)
        headers["Content-Encoding"] = "gzip"

    request = urllib.request.Request(
        endpoint, data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _Response(
                response.status,
                response.headers.get("Retry-After"),
                response.read(2048).decode("utf-8", "replace"),
            )
    except urllib.error.HTTPError as exc:
        # An HTTP error status is a normal contract outcome, not a failure.
        return _Response(
            exc.code,
            exc.headers.get("Retry-After") if exc.headers else None,
            exc.read(2048).decode("utf-8", "replace") if exc.fp else "",
        )


def _retry_after_seconds(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        # Contract sends seconds. Clamp so a hostile or bogus value cannot
        # park a package for years, and never go below one second.
        return max(1, min(int(float(value)), 86_400))
    except (TypeError, ValueError):
        return default


#: Maximum distance one reconcile call can advance the 'obs' mark. Honest
#: heartbeats arrive hours apart at most, so the cap never binds in normal
#: operation; a machine legitimately off for months catches up in a few
#: hook fires (fail-closed latency only). What it bounds is FORWARD clock
#: poison: without it, a single glitched sample (NTP flap reading 2099)
#: permanently drags the mark — and with it every window open and every
#: confirmation horizon — decades ahead, which round 6 reproduced as a
#: refused-data leak. Capped, one insane sample moves the mark at most
#: this far, and real time overtakes it again.
MAX_OBS_ADVANCE_SECONDS = 30 * 24 * 3600


def reconcile_send_consent(
    connection: sqlite3.Connection,
    send_enabled: bool,
    *,
    now: datetime | None = None,
) -> None:
    """Reconcile the consent-window table with the observed config state.

    THE ONLY writer of consent state. Must run inside a write transaction.
    A pure function of (config, now, store): call it from anywhere, any
    number of times, in any order — the resulting windows are the same. This
    replaces the previous edge-detection design, whose three partial
    observers (wizard, relay, mid-pass) each covered a different subset of
    transitions and repeatedly leaked the transitions between the subsets.

    Timestamp discipline (each rule is load-bearing; see the validation
    harness in tests/hermes_cli/test_shared_metrics_consent_windows.py):

    - The 'obs' mark advances to every observation stamp, monotonically —
      but by at most ``MAX_OBS_ADVANCE_SECONDS`` per call. Unbounded, the
      mark is monotonic in the LEAK direction: one glitched-forward sample
      would drag ``last_confirmed_at`` decades ahead, a later close would
      stamp that horizon, and the closed window would contain every future
      refused period (reproduced in round 6). Bounded, a poisoned sample
      costs at most one cap's width, and real time overtakes it.
      An open window's ``last_confirmed_at`` follows the mark: consent is
      asserted only for time that was actually observed.
    - A close is stamped at ``last_confirmed_at`` — never "now" — so an
      unobserved gap (hand-edited config, machine off for 90 days) is never
      inside a window and fails closed.
    - An open clamps to ``max(now, obs, data)``: a rolled-back clock cannot
      open a window underneath refused packages already on disk, and cannot
      make the new window adjacent to the previous close.
    """
    stamp = _isoformat(now or _utc_now())
    raw_stamp = stamp  # pre-cap observation time, used to clamp closes
    previous_obs = connection.execute(
        "SELECT stamp FROM consent_marks WHERE name = 'obs'"
    ).fetchone()
    if previous_obs is not None:
        ceiling = _isoformat(
            _parse_stamp(str(previous_obs[0]))
            + timedelta(seconds=MAX_OBS_ADVANCE_SECONDS)
        )
        stamp = min(stamp, ceiling)
    connection.execute(
        """
        INSERT INTO consent_marks(name, stamp) VALUES ('obs', ?)
        ON CONFLICT(name) DO UPDATE SET stamp = MAX(stamp, excluded.stamp)
        """,
        (stamp,),
    )
    marks = dict(
        connection.execute("SELECT name, stamp FROM consent_marks").fetchall()
    )
    obs = marks["obs"]  # >= stamp; immune to clock rollback
    data = marks.get("data")

    open_row = connection.execute(
        "SELECT rowid FROM send_consent_windows WHERE closed_at IS NULL"
    ).fetchone()

    if send_enabled:
        if open_row is None:
            opened = max(x for x in (obs, data) if x is not None)
            connection.execute(
                "INSERT INTO send_consent_windows(opened_at, last_confirmed_at)"
                " VALUES (?, ?)",
                (opened, opened),
            )
        else:
            connection.execute(
                "UPDATE send_consent_windows"
                " SET last_confirmed_at = MAX(last_confirmed_at, ?)"
                " WHERE rowid = ?",
                (obs, open_row[0]),
            )
    elif open_row is not None:
        # Close at the last CONFIRMED moment, but never after the closing
        # observation's own raw stamp. The two clamps serve different
        # adversaries and both are load-bearing:
        # - min with last_confirmed_at: an unobserved gap (machine off,
        #   hand-edited config) is never asserted as consented (v1's leak).
        # - min with the RAW stamp (pre-cap, pre-MAX): if last_confirmed_at
        #   was poisoned by a glitched-forward sample, an honest clock at
        #   revoke time pulls the close back to the true revoke moment, so
        #   the refused era that follows falls OUTSIDE the closed window
        #   (round 6's D1 leak). A rolled-back clock at close time only
        #   closes EARLIER — fail-closed.
        connection.execute(
            "UPDATE send_consent_windows"
            " SET closed_at = MIN(last_confirmed_at, ?)"
            " WHERE rowid = ?",
            (raw_stamp, open_row[0]),
        )


#: Claim-time consent predicate: the package's period must fall entirely
#: inside SOME recorded consent window. An open window vouches only up to its
#: last confirmed moment, so a package whose period runs past it waits for
#: the next reconcile heartbeat (fail-closed; released within one hook fire).
CONSENT_GATE_SQL = """EXISTS (
    SELECT 1 FROM send_consent_windows w
    WHERE package_outbox.period_start >= w.opened_at
      AND package_outbox.period_end <=
          CASE WHEN w.closed_at IS NULL THEN w.last_confirmed_at
               ELSE w.closed_at END
)"""


def _state_get(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM telemetry_state WHERE key = ?", (key,)
    ).fetchone()
    return str(row[0]) if row is not None else None


def _state_set(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO telemetry_state(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


class SharedMetricsSender:
    """Sends exported packages, one bounded pass at a time."""

    def __init__(
        self,
        store,
        endpoint: str,
        *,
        post=_post,
        sleep=time.sleep,
        now=_utc_now,
        max_attempts: int = MAX_ATTEMPTS,
        consent_check=None,
    ) -> None:
        self._store = store
        self._endpoint = endpoint
        self._post = post
        self._sleep = sleep
        self._now = now
        self._max_attempts = max_attempts
        # Called before every package. None disables the check for callers
        # that have already established consent out of band (tests, E2E).
        self._consent_check = consent_check

    # -- selection ---------------------------------------------------------

    def _claim_next(self, now: datetime, seen: set[str]) -> dict | None:
        """Claim exactly ONE package, immediately before it is sent.

        Claiming a whole batch up front does not work: a single shared lease
        has to cover the entire pass, and 20 retrying packages can legally run
        far longer than any sane lease (three 30s timeouts plus backoff each).
        The later rows' leases then expire while this pass still holds them in
        memory, and another process re-sends them. Taking one row at a time
        keeps the lease covering only the package actually in flight.

        ``seen`` holds packages this pass has already finished with. They are
        excluded IN SQL rather than by rejecting the fetched row: with
        ``LIMIT 1``, returning None for an already-seen row would make the
        caller believe the queue was empty and abandon every healthy package
        behind it. A row can legitimately become eligible again mid-pass (a
        short Retry-After, or a pass that outlives the 15-minute failure
        backoff), so this is reachable in normal operation, not just in tests.
        """
        with self._store._connection() as connection:
            with write_txn(connection):
                stamp = _isoformat(now)
                lease_until = now + timedelta(seconds=_CLAIM_LEASE_SECONDS)

                placeholders = ",".join("?" for _ in seen)
                exclusion = (
                    f" AND package_id NOT IN ({placeholders})" if seen else ""
                )
                # Consent is a READ here — the claim must never mutate the
                # window table. The old design's opt_in_period() call at this
                # exact spot meant selecting a row could rewrite what was
                # permitted to be sent (and did, under a rolled-back clock).
                row = connection.execute(
                    f"""
                    SELECT package_id, payload_json, sent_install_id
                    FROM package_outbox
                    WHERE exported_at IS NOT NULL
                      AND (send_state IS NULL OR send_state = 'pending')
                      AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                      AND {CONSENT_GATE_SQL}
                      AND send_attempts < ?
                      {exclusion}
                    ORDER BY created_at, package_id
                    LIMIT 1
                    """,
                    (stamp, MAX_SEND_ATTEMPTS, *sorted(seen)),
                ).fetchone()
                if row is None:
                    return None

                package_id = str(row[0])
                derived = row[2]
                if not derived:
                    derived = self._freeze_identity(
                        connection, package_id, row[1], now
                    )
                    if derived is None:
                        # Unusable row, already marked rejected. Signal the
                        # caller to continue rather than stop.
                        return {"package_id": package_id, "skip": True}

                token = str(uuid.uuid4())
                connection.execute(
                    """
                    UPDATE package_outbox
                    SET send_state = 'pending',
                        send_attempts = send_attempts + 1,
                        next_attempt_at = ?,
                        claim_token = ?
                    WHERE package_id = ?
                    """,
                    # Lease INTO THE FUTURE: selection requires
                    # next_attempt_at <= now, so no other process can take
                    # this row while it is in flight. Success or a real
                    # backoff overwrites it; if this process dies, it expires.
                    # The token is this claim's identity: a reclaim after
                    # expiry mints a new one, and every later write by THIS
                    # claimant is compare-and-set against it, so a lapsed
                    # claimant that resumes cannot settle or transmit.
                    (_isoformat(lease_until), token, package_id),
                )
                return {
                    "package_id": package_id,
                    "payload_json": str(row[1]),
                    "derived": str(derived),
                    "claim_token": token,
                    "skip": False,
                }

    def _freeze_identity(
        self,
        connection: sqlite3.Connection,
        package_id: str,
        payload_json,
        now: datetime,
    ) -> str | None:
        """Derive and persist the transmitted id, or reject an unusable row.

        Returns None when the package can never be sent. Rejecting rather than
        raising matters: an exception here rolls back the claim transaction
        and blocks every healthy package behind this one.
        """
        reason = None
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError):
            reason = "unreadable payload"
        else:
            # Valid JSON is not enough: a top-level array, string, number or
            # null parses cleanly and then has no .get().
            if not isinstance(payload, dict):
                reason = f"payload is {type(payload).__name__}, expected object"
            else:
                install_id = payload.get("install_id")
                if not isinstance(install_id, str) or not install_id.strip():
                    reason = "payload has no usable install_id"

        if reason is not None:
            logger.warning(
                "Shared-metrics package %s cannot be sent (%s)", package_id, reason
            )
            connection.execute(
                """
                UPDATE package_outbox
                SET send_state = 'rejected', last_error = ?
                WHERE package_id = ?
                """,
                (reason, package_id),
            )
            return None

        salt = current_salt(connection, now=now)
        derived = derive_install_id(payload["install_id"], salt)
        connection.execute(
            "UPDATE package_outbox SET sent_install_id = ? WHERE package_id = ?",
            (derived, package_id),
        )
        return derived

    # -- transmission ------------------------------------------------------

    def _body(self, payload_json: str, derived: str) -> bytes:
        """Rebuild the exact bytes to send.

        The payload is recomputed from the stored package rather than kept as
        a second copy: json.dumps with these options is deterministic, and the
        only mutable input (the derived id) is frozen in the row.
        """
        payload = substitute_install_id(json.loads(payload_json), derived)
        return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")

    def _mark(
        self,
        package_id: str,
        *,
        only_if_pending: bool = True,
        token: str | None = None,
        **columns,
    ) -> None:
        """Write send state for one package.

        Guarded on send_state so a pass whose lease lapsed cannot resurrect a
        row another process has already finished: without this, a slow sender
        could overwrite 'sent' back to 'pending' and cause a re-send.

        When ``token`` is given, the write is additionally compare-and-set on
        claim_token: it lands only if THIS claim is still the current one. A
        claimant that lapsed and was superseded writes zero rows — its
        settlement, backoff, and error strings all silently lose to the
        newer claim's, which is the correct outcome.
        """
        assignments = ", ".join(f"{name} = ?" for name in columns)
        predicate = (
            " AND (send_state IS NULL OR send_state = 'pending')"
            if only_if_pending
            else ""
        )
        params: list = [*columns.values(), package_id]
        if token is not None:
            predicate += " AND claim_token = ?"
            params.append(token)
        with self._store._connection() as connection:
            with write_txn(connection):
                connection.execute(
                    f"UPDATE package_outbox SET {assignments} "
                    f"WHERE package_id = ?{predicate}",
                    params,
                )

    def _renew_claim(self, package_id: str, token: str | None) -> bool:
        """Atomically re-assert ownership and extend the lease. CAS, one row.

        A read-only ownership check is not enough: a claimant whose lease
        expired while suspended can pass the check (its token is still in
        the row if no one reclaimed yet) and then POST while another process
        legitimately reclaims — the check-to-POST expiry race a seventh
        review reproduced. Renewal closes it by requiring, in ONE statement:

        - the token still matches (nobody reclaimed), AND
        - the current lease is UNEXPIRED (this claimant is not stale), AND
        - the row is still pending,

        and only then pushing next_attempt_at a fresh lease into the future,
        so the upcoming POST (30s timeout, well under the 300s lease) runs
        entirely inside renewed authority. rowcount == 1 is the only grant.
        A claimant that wakes past its own lease fails the unexpired
        condition and yields even though its token was never replaced.
        """
        if token is None:
            return False
        try:
            now = self._now()
            lease_until = now + timedelta(seconds=_CLAIM_LEASE_SECONDS)
            with self._store._connection() as connection:
                with write_txn(connection):
                    cursor = connection.execute(
                        """
                        UPDATE package_outbox
                        SET next_attempt_at = ?
                        WHERE package_id = ?
                          AND claim_token = ?
                          AND (send_state IS NULL OR send_state = 'pending')
                          AND next_attempt_at > ?
                        """,
                        (
                            _isoformat(lease_until),
                            package_id,
                            token,
                            _isoformat(now),
                        ),
                    )
                    return cursor.rowcount == 1
        except Exception:
            # If renewal itself fails, do not transmit on unproven authority.
            logger.warning(
                "Unable to renew shared-metrics claim", exc_info=True
            )
            return False

    def _defer(
        self,
        package_id: str,
        delay_seconds: int,
        reason: str,
        *,
        token: str | None = None,
    ) -> None:
        # Defence in depth: no current caller can pass a non-positive delay
        # (Retry-After is already clamped to [1, 86400] when parsed, and every
        # other call site passes a positive constant), so this clamp is
        # deliberately unreachable today and no test can distinguish it. It
        # stays because a past deadline would make the row instantly
        # re-eligible and let a pass spin on it — a cheap guard against a
        # future caller that forgets.
        delay = max(1, int(delay_seconds))
        retry_at = self._now().timestamp() + delay
        self._mark(
            package_id,
            token=token,
            send_state="pending",
            next_attempt_at=_isoformat(
                datetime.fromtimestamp(retry_at, tz=timezone.utc)
            ),
            last_error=reason[:500],
        )

    def _send_one(self, package: dict) -> str:
        """Try one package. Returns 'sent', 'rejected', or 'deferred'.

        Delivery is at-least-once. The pre-POST ownership check plus the
        token-fenced writes close the claim->POST and settle-after-reclaim
        gaps, but a suspension landing MID-POST (bytes already on the wire
        when the machine sleeps) can still duplicate: no client-side check
        can revoke a request in flight. The body is byte-identical across
        retries by construction, so the residual duplicate is exactly one
        redundant copy of identical content; collapsing it fully would need
        package_id-keyed dedupe at the ingest service.
        """
        package_id = package["package_id"]
        token = package.get("claim_token")
        body = self._body(package["payload_json"], package["derived"])

        for attempt in range(1, self._max_attempts + 1):
            # Atomically renew the claim before EVERY external POST. The
            # renewal is compare-and-set on (token, pending, lease unexpired)
            # and extends the lease past the request, so a suspended-then-
            # resumed claimant whose lease lapsed yields here even if nobody
            # has reclaimed yet — a read-only ownership check passed in that
            # state and still double-sent (check-to-POST expiry race). The
            # ingest key is minute-prefixed, so duplicates become distinct
            # stored objects, not overwrites.
            if not self._renew_claim(package_id, token):
                logger.info(
                    "Shared-metrics claim on %s superseded or expired; yielding",
                    package_id,
                )
                return "deferred"
            try:
                response = self._post(
                    self._endpoint, body, timeout=REQUEST_TIMEOUT_SECONDS
                )
            except Exception as exc:  # transport failure: offline, DNS, TLS
                reason = f"{type(exc).__name__}: {exc}"
                if attempt >= self._max_attempts:
                    self._defer(
                        package_id, _FAILURE_BACKOFF_SECONDS, reason, token=token
                    )
                    return "deferred"
                self._sleep(self._backoff(attempt))
                continue

            if response.status == 202:
                self._mark(
                    package_id,
                    token=token,
                    send_state="sent",
                    sent_at=_isoformat(self._now()),
                    last_error=None,
                )
                return "sent"

            if response.status in _PERMANENT_STATUSES:
                # Only statuses the contract (or the envelope schema) makes
                # terminal. Everything else retries: 403 in particular is the
                # ingest service's origin guard, which returns 403 during an
                # edge/Transform-Rule misconfiguration — treating that as
                # permanent would discard every package sent during the
                # incident instead of retrying after recovery.
                logger.warning(
                    "Telemetry package %s rejected with HTTP %s; not retrying",
                    package_id,
                    response.status,
                )
                self._mark(
                    package_id,
                    token=token,
                    send_state="rejected",
                    last_error=f"HTTP {response.status}: {response.body[:400]}",
                )
                return "rejected"

            if response.status == 429:
                self._defer(
                    package_id,
                    _retry_after_seconds(response.retry_after, _FAILURE_BACKOFF_SECONDS),
                    "rate limited",
                    token=token,
                )
                return "deferred"

            # 5xx and anything unexpected: retryable.
            reason = f"HTTP {response.status}"
            if attempt >= self._max_attempts:
                self._defer(
                    package_id, _FAILURE_BACKOFF_SECONDS, reason, token=token
                )
                return "deferred"
            self._sleep(self._backoff(attempt))

        self._defer(
            package_id, _FAILURE_BACKOFF_SECONDS, "attempts exhausted", token=token
        )
        return "deferred"

    @staticmethod
    def _backoff(attempt: int) -> float:
        """1s, 5s, 25s with full jitter."""
        ceiling = _BACKOFF_BASE_SECONDS * (_BACKOFF_FACTOR ** (attempt - 1))
        return random.uniform(0, ceiling)

    # -- entry point -------------------------------------------------------

    def send_pending(self) -> SendOutcome:
        """Run one bounded pass. Never raises.

        Claims and sends ONE package at a time so each row's lease only has to
        cover its own transmission, and re-checks consent before every send so
        revoking `send` mid-pass stops the remaining packages.
        """
        outcome = SendOutcome()
        seen: set[str] = set()

        for _ in range(MAX_PACKAGES_PER_PASS):
            if not self._still_consented():
                # The user turned sending off while this pass was running.
                # Stop without transmitting anything further, and reconcile
                # so the window closes at its last confirmed moment. This is
                # the same single writer every other observation point uses —
                # not a separate recording mechanism.
                logger.info("Shared-metrics sending disabled mid-pass; stopping")
                self._reconcile(send_enabled=False)
                break
            try:
                package = self._claim_next(self._now(), seen)
            except Exception:
                logger.warning(
                    "Unable to select shared-metrics packages", exc_info=True
                )
                break
            if package is None:
                break

            seen.add(package["package_id"])
            if package.get("skip"):
                # Unusable row already marked rejected during the claim.
                outcome.rejected += 1
                continue

            try:
                result = self._send_one(package)
            except Exception:
                logger.warning("Unable to send shared-metrics package", exc_info=True)
                outcome.deferred += 1
                continue
            if result == "sent":
                outcome.sent += 1
            elif result == "rejected":
                outcome.rejected += 1
            else:
                outcome.deferred += 1
        return outcome

    def _reconcile(self, *, send_enabled: bool) -> None:
        """Run the single consent writer from within a pass."""
        try:
            with self._store._connection() as connection:
                with write_txn(connection):
                    reconcile_send_consent(
                        connection, send_enabled, now=self._now()
                    )
        except Exception:
            logger.warning(
                "Unable to reconcile shared-metrics consent", exc_info=True
            )

    def _still_consented(self) -> bool:
        """Re-read profile-owned send consent.

        Consent is a boundary, not cached configuration: the documentation
        promises that setting `send: false` stops transmission immediately,
        and a pass can run for minutes. Injected senders (tests, the staging
        E2E) opt out by passing consent_check=None.
        """
        if self._consent_check is None:
            return True
        try:
            return bool(self._consent_check())
        except Exception:
            # Fail CLOSED: if consent cannot be established, do not transmit.
            logger.warning(
                "Unable to confirm shared-metrics send consent; stopping",
                exc_info=True,
            )
            return False
