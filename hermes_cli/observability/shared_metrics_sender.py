"""Transmit exported shared-metrics packages to the Nous telemetry service.

Sender side of the ingest contract (telemetry repo ``CONTRACT.md``): ``202`` durably stored,
mark sent; ``400`` permanently malformed, never retry; ``429`` keep, retry after
``Retry-After``; ``5xx`` / timeout / connection error keep, retry with backoff.
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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from hermes_cli.sqlite_util import write_txn

from .shared_metrics import _isoformat, _utc_now

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

#: Claim lease, written INTO THE FUTURE as next_attempt_at so no other process selects
#: the row meanwhile. Must exceed one package's worst case (3x30s timeouts + 1s+5s
#: backoff, ~96s) — hence packages are claimed one at a time right before sending; a
#: batch of 20 under one lease can run ~1900s and get re-sent by another process.
_CLAIM_LEASE_SECONDS = 300

#: Floor applied after a pass fails to deliver, so a hard-down service is not
#: retried on every task completion.
_FAILURE_BACKOFF_SECONDS = 15 * 60

#: Permanent statuses per the ingest contract, deliberately narrow: 400 never validates,
#: 413 (over the 1 MiB cap) cannot shrink on retry. Everything else — including 403 from
#: the origin guard and 404 from a bad path — is deployment/edge misconfiguration that
#: resolves without the package changing, so it is retried.
_PERMANENT_STATUSES = frozenset({400, 413})

#: Attempts after which a package is abandoned; otherwise a poisoned row is retried until
#: 30-day retention deletes it (~160 requests) and pins the head of the queue.
MAX_SEND_ATTEMPTS = 25

#: Max distance one reconcile can advance the 'obs' mark. Never binds for honest
#: heartbeats (hours apart); bounds FORWARD clock poison, where one glitched sample (NTP
#: flap reading 2099) would drag every window open and confirmation horizon decades ahead.
MAX_OBS_ADVANCE_SECONDS = 30 * 24 * 3600


def _parse_stamp(value: str) -> datetime:
    """Parse a stamp this module itself wrote (Z-suffixed ISO-8601, UTC)."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass
class SendOutcome:
    """What one pass did. Returned for tests and diagnostics."""

    sent: int = 0
    rejected: int = 0
    deferred: int = 0


@dataclass(slots=True)
class _Response:
    status: int
    retry_after: str | None
    body: str


def _post(endpoint: str, payload: bytes, *, timeout: int) -> _Response:
    """POST one package. Raises on transport failure; never on HTTP status."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "hermes-agent-shared-metrics/1",
    }
    body = payload
    if len(payload) > GZIP_THRESHOLD_BYTES:
        # mtime=0 keeps two sends of one package byte-identical on the wire too.
        body = gzip.compress(payload, mtime=0)
        headers["Content-Encoding"] = "gzip"

    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
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
        # Contract sends seconds. Clamp so a bogus value cannot park a package for
        # years, and never go below one second.
        return max(1, min(int(float(value)), 86_400))
    except (TypeError, ValueError):
        return default


def reconcile_send_consent(
    connection: sqlite3.Connection,
    send_enabled: bool,
    *,
    now: datetime | None = None,
) -> None:
    """Reconcile the consent-window table with the observed config state.

    THE ONLY writer of consent state; must run inside a write transaction. Idempotent in
    (config, now, store): call it from anywhere, any number of times, in any order.
    """
    stamp = _isoformat(now or _utc_now())
    raw_stamp = stamp  # pre-cap observation time, used to clamp closes
    previous_obs = connection.execute(
        "SELECT stamp FROM consent_marks WHERE name = 'obs'"
    ).fetchone()
    if previous_obs is not None:
        ceiling = _isoformat(
            _parse_stamp(str(previous_obs[0])) + timedelta(seconds=MAX_OBS_ADVANCE_SECONDS)
        )
        stamp = min(stamp, ceiling)
    connection.execute(
        """
        INSERT INTO consent_marks(name, stamp) VALUES ('obs', ?)
        ON CONFLICT(name) DO UPDATE SET stamp = MAX(stamp, excluded.stamp)
        """,
        (stamp,),
    )
    marks = dict(connection.execute("SELECT name, stamp FROM consent_marks").fetchall())
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
        # Close at the last CONFIRMED moment, never after the closing observation's RAW
        # stamp. Both clamps are load-bearing: last_confirmed_at means an unobserved gap
        # (machine off, hand-edited config) is never asserted as consented; the raw
        # (pre-cap) stamp pulls a glitched-forward last_confirmed_at back to the true
        # revoke moment. A rolled-back clock only closes EARLIER — fail-closed.
        connection.execute(
            "UPDATE send_consent_windows"
            " SET closed_at = MIN(last_confirmed_at, ?)"
            " WHERE rowid = ?",
            (raw_stamp, open_row[0]),
        )


#: Claim-time consent predicate: the package's period must fall entirely inside SOME
#: recorded window. An open window vouches only up to its last confirmed moment, so a
#: package running past it waits for the next reconcile heartbeat (fail-closed).
CONSENT_GATE_SQL = """EXISTS (
    SELECT 1 FROM send_consent_windows w
    WHERE package_outbox.period_start >= w.opened_at
      AND package_outbox.period_end <=
          CASE WHEN w.closed_at IS NULL THEN w.last_confirmed_at
               ELSE w.closed_at END
)"""


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
        # Called before every package. None disables the check for callers that have
        # already established consent out of band (tests, E2E).
        self._consent_check = consent_check

    @contextmanager
    def _write(self):
        """One store connection inside a write transaction."""
        with self._store._connection() as connection:
            with write_txn(connection):
                yield connection

    # -- selection ---------------------------------------------------------

    def _claim_next(self, now: datetime, seen: set[str]) -> dict | None:
        """Claim exactly ONE package, immediately before it is sent (see _CLAIM_LEASE_SECONDS).

        ``seen`` (packages this pass finished with) is excluded IN SQL: with LIMIT 1,
        returning None for a seen row would look like an empty queue and abandon everything
        behind it, and rows can legitimately become eligible again mid-pass.
        """
        with self._write() as connection:
            stamp = _isoformat(now)
            lease_until = now + timedelta(seconds=_CLAIM_LEASE_SECONDS)

            placeholders = ",".join("?" for _ in seen)
            exclusion = f" AND package_id NOT IN ({placeholders})" if seen else ""
            # Consent is a READ here: the claim must never mutate the window table.
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
            derived = row[2] or self._freeze_identity(connection, package_id, row[1])
            if derived is None:
                # Unusable row, already marked rejected. Tell the caller to continue.
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
                # The token is this claim's identity: a reclaim after expiry mints a new one
                # and every later write by THIS claimant is compare-and-set against it.
                (_isoformat(lease_until), token, package_id),
            )
            return {
                "package_id": package_id, "payload_json": str(row[1]), "derived": str(derived),
                "claim_token": token, "skip": False,
            }

    @staticmethod
    def _freeze_identity(
        connection: sqlite3.Connection, package_id: str, payload_json
    ) -> str | None:
        """Record the transmitted install_id on the row, or reject an unusable one.

        Rejecting rather than raising matters: an exception would roll back the claim
        transaction and block every healthy package behind this one.
        """
        install_id = None
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError):
            reason = "unreadable payload"
        else:
            # Valid JSON is not enough: a top-level array/string/number parses cleanly.
            install_id = payload.get("install_id") if isinstance(payload, dict) else None
            if not isinstance(payload, dict):
                reason = f"payload is {type(payload).__name__}, expected object"
            elif not isinstance(install_id, str) or not install_id.strip():
                reason = "payload has no usable install_id"
            else:
                reason = None

        if reason is not None:
            logger.warning("Shared-metrics package %s cannot be sent (%s)", package_id, reason)
            connection.execute(
                """
                UPDATE package_outbox
                SET send_state = 'rejected', last_error = ?
                WHERE package_id = ?
                """,
                (reason, package_id),
            )
            return None

        connection.execute(
            "UPDATE package_outbox SET sent_install_id = ? WHERE package_id = ?",
            (install_id, package_id),
        )
        return str(install_id)

    # -- transmission ------------------------------------------------------

    @staticmethod
    def _body(payload_json: str, transmitted_id: str) -> bytes:
        """Rebuild the exact bytes to send: deterministic json.dumps of the stored package
        with install_id from the frozen ``sent_install_id`` column (resends are byte-identical)."""
        payload = dict(json.loads(payload_json))
        payload["install_id"] = transmitted_id
        return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")

    def _mark(self, package_id: str, *, token: str | None = None, **columns) -> None:
        """Write send state for one package.

        Guarded on send_state so a lapsed pass cannot resurrect a row another process already
        finished. With ``token`` the write is also compare-and-set on claim_token: a
        superseded claimant writes zero rows.
        """
        assignments = ", ".join(f"{name} = ?" for name in columns)
        predicate = " AND (send_state IS NULL OR send_state = 'pending')"
        params: list = [*columns.values(), package_id]
        if token is not None:
            predicate += " AND claim_token = ?"
            params.append(token)
        with self._write() as connection:
            connection.execute(
                f"UPDATE package_outbox SET {assignments} WHERE package_id = ?{predicate}",
                params,
            )

    def _renew_claim(self, package_id: str, token: str | None) -> bool:
        """Atomically re-assert ownership and extend the lease (CAS, one row; rowcount == 1
        is the only grant).

        A read-only check is not enough: a claimant whose lease expired while suspended still
        has its token in the row and would POST while another process reclaims. Renewing in
        the same CAS keeps the 30s POST inside the 300s lease.
        """
        if token is None:
            return False
        try:
            now = self._now()
            lease_until = now + timedelta(seconds=_CLAIM_LEASE_SECONDS)
            with self._write() as connection:
                cursor = connection.execute(
                    """
                        UPDATE package_outbox
                        SET next_attempt_at = ?
                        WHERE package_id = ?
                          AND claim_token = ?
                          AND (send_state IS NULL OR send_state = 'pending')
                          AND next_attempt_at > ?
                        """,
                    (_isoformat(lease_until), package_id, token, _isoformat(now)),
                )
                return cursor.rowcount == 1
        except Exception:
            # If renewal itself fails, do not transmit on unproven authority.
            logger.warning("Unable to renew shared-metrics claim", exc_info=True)
            return False

    def _defer(
        self, package_id: str, delay_seconds: int, reason: str, *, token: str | None = None
    ) -> None:
        # Clamp to >= 1s so a past deadline can never make the row instantly re-eligible.
        retry_at = self._now().timestamp() + max(1, int(delay_seconds))
        self._mark(
            package_id,
            token=token,
            send_state="pending",
            next_attempt_at=_isoformat(datetime.fromtimestamp(retry_at, tz=timezone.utc)),
            last_error=reason[:500],
        )

    def _send_one(self, package: dict) -> str:
        """Try one package. Returns 'sent', 'rejected', or 'deferred'.

        Delivery is at-least-once: renewal plus token-fenced writes close the claim->POST
        and settle-after-reclaim gaps, but a suspension MID-POST can still duplicate.
        """
        package_id = package["package_id"]
        token = package.get("claim_token")
        body = self._body(package["payload_json"], package["derived"])

        def defer(delay: int, reason: str) -> str:
            self._defer(package_id, delay, reason, token=token)
            return "deferred"

        for attempt in range(1, self._max_attempts + 1):
            # Renew before EVERY external POST so a lapsed claimant yields even before anyone
            # reclaims. The ingest key is minute-prefixed: duplicates are distinct objects.
            if not self._renew_claim(package_id, token):
                logger.info(
                    "Shared-metrics claim on %s superseded or expired; yielding", package_id
                )
                return "deferred"
            try:
                response = self._post(self._endpoint, body, timeout=REQUEST_TIMEOUT_SECONDS)
            except Exception as exc:  # transport failure: offline, DNS, TLS
                reason = f"{type(exc).__name__}: {exc}"
            else:
                if response.status == 202:
                    self._mark(
                        package_id, token=token, send_state="sent",
                        sent_at=_isoformat(self._now()), last_error=None,
                    )
                    return "sent"
                if response.status in _PERMANENT_STATUSES:
                    logger.warning(
                        "Telemetry package %s rejected with HTTP %s; not retrying",
                        package_id,
                        response.status,
                    )
                    self._mark(
                        package_id, token=token, send_state="rejected",
                        last_error=f"HTTP {response.status}: {response.body[:400]}",
                    )
                    return "rejected"
                if response.status == 429:
                    return defer(
                        _retry_after_seconds(response.retry_after, _FAILURE_BACKOFF_SECONDS),
                        "rate limited",
                    )
                # 5xx and anything unexpected: retryable.
                reason = f"HTTP {response.status}"
            if attempt >= self._max_attempts:
                return defer(_FAILURE_BACKOFF_SECONDS, reason)
            self._sleep(self._backoff(attempt))

        return defer(_FAILURE_BACKOFF_SECONDS, "attempts exhausted")

    @staticmethod
    def _backoff(attempt: int) -> float:
        """1s, 5s, 25s with full jitter."""
        return random.uniform(0, _BACKOFF_BASE_SECONDS * (_BACKOFF_FACTOR ** (attempt - 1)))

    # -- entry point -------------------------------------------------------

    def send_pending(self) -> SendOutcome:
        """Run one bounded pass, one claim+send at a time, re-checking consent before each
        send so revoking `send` mid-pass stops the remainder. Never raises."""
        outcome = SendOutcome()
        seen: set[str] = set()

        for _ in range(MAX_PACKAGES_PER_PASS):
            if not self._still_consented():
                # Reconcile through the single consent writer so the window closes at its
                # last confirmed moment.
                logger.info("Shared-metrics sending disabled mid-pass; stopping")
                self._reconcile(send_enabled=False)
                break
            try:
                package = self._claim_next(self._now(), seen)
            except Exception:
                logger.warning("Unable to select shared-metrics packages", exc_info=True)
                break
            if package is None:
                break

            seen.add(package["package_id"])
            if package.get("skip"):
                # Unusable row already marked rejected during the claim.
                result = "rejected"
            else:
                try:
                    result = self._send_one(package)
                except Exception:
                    logger.warning("Unable to send shared-metrics package", exc_info=True)
                    result = "deferred"
            setattr(outcome, result, getattr(outcome, result) + 1)
        return outcome

    def _reconcile(self, *, send_enabled: bool) -> None:
        """Run the single consent writer from within a pass."""
        try:
            with self._write() as connection:
                reconcile_send_consent(connection, send_enabled, now=self._now())
        except Exception:
            logger.warning("Unable to reconcile shared-metrics consent", exc_info=True)

    def _still_consented(self) -> bool:
        """Re-read profile-owned send consent: docs promise ``send: false`` stops transmission
        immediately, and a pass can run for minutes."""
        if self._consent_check is None:
            return True
        try:
            return bool(self._consent_check())
        except Exception:
            # Fail CLOSED: if consent cannot be established, do not transmit.
            logger.warning("Unable to confirm shared-metrics send consent; stopping", exc_info=True)
            return False
