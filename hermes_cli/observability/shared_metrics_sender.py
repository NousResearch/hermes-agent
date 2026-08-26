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
undercount the opt-in day.
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

OPT_IN_PERIOD_KEY = "send_opt_in_period"

#: Set when sending is turned off, cleared by the next enabled pass (which
#: also advances OPT_IN_PERIOD_KEY). This is what makes consent revocation
#: permanent for the packages collected while it was off.
SEND_REVOKED_KEY = "send_revoked"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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


def opt_in_period(connection: sqlite3.Connection, *, now: datetime | None = None) -> str:
    """Return the day (UTC) from which packages may be sent.

    Must run inside a write transaction.

    This is the CURRENT consent window's start, not a permanent first-ever
    opt-in date. If the user previously turned sending off, ``record_revoked``
    stamps that; the next enabled pass advances the gate to the day sending
    resumed, so packages collected during the opted-out window are never
    transmitted. Without that advance, re-enabling would retroactively release
    the entire period the user had explicitly refused.
    """
    today = (now or _utc_now()).date().isoformat()

    revoked = _state_get(connection, SEND_REVOKED_KEY)
    if revoked:
        # Sending resumed after a revocation: the new window starts today.
        _state_set(connection, OPT_IN_PERIOD_KEY, today)
        connection.execute(
            "DELETE FROM telemetry_state WHERE key = ?", (SEND_REVOKED_KEY,)
        )
        return today

    existing = _state_get(connection, OPT_IN_PERIOD_KEY)
    if existing:
        return existing

    _state_set(connection, OPT_IN_PERIOD_KEY, today)
    return today


def record_revoked(connection: sqlite3.Connection) -> None:
    """Mark that sending was turned off, closing the current consent window.

    Idempotent. The marker is only cleared by the next enabled pass, which
    also advances the gate — so any package collected between the two events
    stays local permanently.
    """
    if _state_get(connection, OPT_IN_PERIOD_KEY):
        _state_set(connection, SEND_REVOKED_KEY, "1")


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
                period = opt_in_period(connection, now=now)
                stamp = _isoformat(now)
                lease_until = now + timedelta(seconds=_CLAIM_LEASE_SECONDS)

                placeholders = ",".join("?" for _ in seen)
                exclusion = (
                    f" AND package_id NOT IN ({placeholders})" if seen else ""
                )
                row = connection.execute(
                    f"""
                    SELECT package_id, payload_json, sent_install_id
                    FROM package_outbox
                    WHERE exported_at IS NOT NULL
                      AND (send_state IS NULL OR send_state = 'pending')
                      AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                      AND substr(period_start, 1, 10) >= ?
                      AND send_attempts < ?
                      {exclusion}
                    ORDER BY created_at, package_id
                    LIMIT 1
                    """,
                    (stamp, period, MAX_SEND_ATTEMPTS, *sorted(seen)),
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

                connection.execute(
                    """
                    UPDATE package_outbox
                    SET send_state = 'pending',
                        send_attempts = send_attempts + 1,
                        next_attempt_at = ?
                    WHERE package_id = ?
                    """,
                    # Lease INTO THE FUTURE: selection requires
                    # next_attempt_at <= now, so no other process can take
                    # this row while it is in flight. Success or a real
                    # backoff overwrites it; if this process dies, it expires.
                    (_isoformat(lease_until), package_id),
                )
                return {
                    "package_id": package_id,
                    "payload_json": str(row[1]),
                    "derived": str(derived),
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

    def _mark(self, package_id: str, *, only_if_pending: bool = True, **columns) -> None:
        """Write send state for one package.

        Guarded on send_state so a pass whose lease lapsed cannot resurrect a
        row another process has already finished: without this, a slow sender
        could overwrite 'sent' back to 'pending' and cause a re-send.
        """
        assignments = ", ".join(f"{name} = ?" for name in columns)
        predicate = (
            " AND (send_state IS NULL OR send_state = 'pending')"
            if only_if_pending
            else ""
        )
        with self._store._connection() as connection:
            with write_txn(connection):
                connection.execute(
                    f"UPDATE package_outbox SET {assignments} "
                    f"WHERE package_id = ?{predicate}",
                    (*columns.values(), package_id),
                )

    def _defer(self, package_id: str, delay_seconds: int, reason: str) -> None:
        # Never write a deadline in the past: that would make the row instantly
        # re-eligible and let a pass spin on it.
        delay = max(1, int(delay_seconds))
        retry_at = self._now().timestamp() + delay
        self._mark(
            package_id,
            send_state="pending",
            next_attempt_at=_isoformat(
                datetime.fromtimestamp(retry_at, tz=timezone.utc)
            ),
            last_error=reason[:500],
        )

    def _send_one(self, package: dict) -> str:
        """Try one package. Returns 'sent', 'rejected', or 'deferred'."""
        package_id = package["package_id"]
        body = self._body(package["payload_json"], package["derived"])

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._post(
                    self._endpoint, body, timeout=REQUEST_TIMEOUT_SECONDS
                )
            except Exception as exc:  # transport failure: offline, DNS, TLS
                reason = f"{type(exc).__name__}: {exc}"
                if attempt >= self._max_attempts:
                    self._defer(package_id, _FAILURE_BACKOFF_SECONDS, reason)
                    return "deferred"
                self._sleep(self._backoff(attempt))
                continue

            if response.status == 202:
                self._mark(
                    package_id,
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
                    send_state="rejected",
                    last_error=f"HTTP {response.status}: {response.body[:400]}",
                )
                return "rejected"

            if response.status == 429:
                self._defer(
                    package_id,
                    _retry_after_seconds(response.retry_after, _FAILURE_BACKOFF_SECONDS),
                    "rate limited",
                )
                return "deferred"

            # 5xx and anything unexpected: retryable.
            reason = f"HTTP {response.status}"
            if attempt >= self._max_attempts:
                self._defer(package_id, _FAILURE_BACKOFF_SECONDS, reason)
                return "deferred"
            self._sleep(self._backoff(attempt))

        self._defer(package_id, _FAILURE_BACKOFF_SECONDS, "attempts exhausted")
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
                # Stop without transmitting anything further, and close the
                # consent window so a later re-enable cannot release the
                # packages collected in the meantime. Recorded here as well as
                # in the setup wizard because config.yaml can be edited by
                # hand, which the wizard never sees.
                logger.info("Shared-metrics sending disabled mid-pass; stopping")
                self._record_revocation()
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

    def _record_revocation(self) -> None:
        """Close the consent window after an observed revocation."""
        try:
            with self._store._connection() as connection:
                with write_txn(connection):
                    record_revoked(connection)
        except Exception:
            logger.debug("Unable to record consent revocation", exc_info=True)

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
