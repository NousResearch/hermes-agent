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
from datetime import datetime, timezone

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

#: Floor applied after a pass fails to deliver, so a hard-down service is not
#: retried on every task completion.
_FAILURE_BACKOFF_SECONDS = 15 * 60

OPT_IN_PERIOD_KEY = "send_opt_in_period"


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
    skipped_not_due: int = 0


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
        body = gzip.compress(payload)
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
    """Return the opt-in day (UTC date), recording it on first use.

    Must run inside a write transaction. The value is written once and then
    never moves, so turning sending off and on again does not re-open the
    pre-consent backlog.
    """
    row = connection.execute(
        "SELECT value FROM telemetry_state WHERE key = ?", (OPT_IN_PERIOD_KEY,)
    ).fetchone()
    if row is not None:
        return str(row[0])
    today = (now or _utc_now()).date().isoformat()
    connection.execute(
        "INSERT OR IGNORE INTO telemetry_state(key, value) VALUES (?, ?)",
        (OPT_IN_PERIOD_KEY, today),
    )
    return today


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
    ) -> None:
        self._store = store
        self._endpoint = endpoint
        self._post = post
        self._sleep = sleep
        self._now = now
        self._max_attempts = max_attempts

    # -- selection ---------------------------------------------------------

    def _claim(self, connection: sqlite3.Connection, now: datetime) -> list[dict]:
        """Atomically take ownership of the packages this pass will try.

        Claiming inside the write transaction is what stops two Hermes
        processes sharing one database from sending the same package twice.
        Duplicates would be harmless (the service dedupes by package_id and
        the bytes are identical) but they waste the user's bandwidth.
        """
        period = opt_in_period(connection, now=now)
        stamp = _isoformat(now)
        rows = connection.execute(
            """
            SELECT package_id, payload_json, sent_install_id
            FROM package_outbox
            WHERE exported_at IS NOT NULL
              AND (send_state IS NULL OR send_state = 'pending')
              AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
              AND substr(period_start, 1, 10) >= ?
            ORDER BY created_at, package_id
            LIMIT ?
            """,
            (stamp, period, MAX_PACKAGES_PER_PASS),
        ).fetchall()

        claimed: list[dict] = []
        salt: str | None = None
        for row in rows:
            package_id = str(row[0])
            derived = row[2]
            if not derived:
                # Freeze the derived identity on first attempt so a later salt
                # rotation cannot change the bytes sent under this package_id.
                if salt is None:
                    salt = current_salt(connection, now=now)
                try:
                    payload = json.loads(row[1])
                    install_id = str(payload.get("install_id", ""))
                except (TypeError, ValueError):
                    # A row we cannot parse can never be sent. Mark it and move
                    # on: one unreadable package must not block every other
                    # package behind it, and aborting here would roll back the
                    # whole claim transaction.
                    logger.warning(
                        "Shared-metrics package %s is unreadable; not sending",
                        package_id,
                    )
                    connection.execute(
                        """
                        UPDATE package_outbox
                        SET send_state = 'rejected', last_error = 'unreadable payload'
                        WHERE package_id = ?
                        """,
                        (package_id,),
                    )
                    continue
                derived = derive_install_id(install_id, salt)
                connection.execute(
                    "UPDATE package_outbox SET sent_install_id = ? WHERE package_id = ?",
                    (derived, package_id),
                )
            connection.execute(
                """
                UPDATE package_outbox
                SET send_state = 'pending',
                    send_attempts = send_attempts + 1,
                    next_attempt_at = ?
                WHERE package_id = ?
                """,
                # Hold the row for the duration of this pass; success or a
                # real backoff overwrite this immediately below.
                (_isoformat(now), package_id),
            )
            claimed.append(
                {
                    "package_id": package_id,
                    "payload_json": str(row[1]),
                    "derived": str(derived),
                }
            )
        return claimed

    # -- transmission ------------------------------------------------------

    def _body(self, payload_json: str, derived: str) -> bytes:
        """Rebuild the exact bytes to send.

        The payload is recomputed from the stored package rather than kept as
        a second copy: json.dumps with these options is deterministic, and the
        only mutable input (the derived id) is frozen in the row.
        """
        payload = substitute_install_id(json.loads(payload_json), derived)
        return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")

    def _mark(self, package_id: str, **columns) -> None:
        assignments = ", ".join(f"{name} = ?" for name in columns)
        with self._store._connection() as connection:
            with write_txn(connection):
                connection.execute(
                    f"UPDATE package_outbox SET {assignments} WHERE package_id = ?",
                    (*columns.values(), package_id),
                )

    def _defer(self, package_id: str, delay_seconds: int, reason: str) -> None:
        retry_at = self._now().timestamp() + delay_seconds
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

            if response.status == 400:
                # Permanent per the contract. Keep the file (it is the user's
                # history) but never try again.
                logger.warning(
                    "Telemetry package %s rejected as malformed; not retrying",
                    package_id,
                )
                self._mark(
                    package_id,
                    send_state="rejected",
                    last_error=response.body[:500],
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
        """Run one bounded pass. Never raises."""
        outcome = SendOutcome()
        try:
            now = self._now()
            with self._store._connection() as connection:
                with write_txn(connection):
                    claimed = self._claim(connection, now)
        except Exception:
            logger.warning("Unable to select shared-metrics packages", exc_info=True)
            return outcome

        for package in claimed:
            try:
                result = self._send_one(package)
            except Exception:
                logger.warning(
                    "Unable to send shared-metrics package", exc_info=True
                )
                outcome.deferred += 1
                continue
            if result == "sent":
                outcome.sent += 1
            elif result == "rejected":
                outcome.rejected += 1
            else:
                outcome.deferred += 1
        return outcome
