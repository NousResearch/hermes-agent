"""Drain worker for mem0 salient auto-capture (Track A-lite).

Runs PLUGIN-SIDE (in the Hermes gateway process, alongside the mem0 plugin — the plugin reaches the
store over HTTP, so the queue + worker live here, not in the mem0 container). A single background
loop that pulls queued turns and does the slow/failable work off the critical path:

    lease a due row  ->  client.add(messages, prompt=GATE, model=MODEL)   [server-side extraction+gate]
                     ->  record model_verdict (D-10)
                     ->  reconcile by capture_idem: if rows already exist, mark done (exactly-once, D-8)
                     ->  else scrub the just-written facts (INV-4 defense-in-depth) — A-lite has no
                         server redaction seam yet, so the drainer scrubs post-write and FORGETS any
                         row that carries a secret
                     ->  mark done  |  on transient failure: record fault + mark_retry (bounded, D-10)

INV-1 (no silent loss): a lease that dies mid-flight is recovered by the queue reaper; a model fault
requeues with backoff up to MAX then dead-letters + alerts. INV-3 (never blocks the turn): this runs
in its own thread; sync_turn only does the tiny enqueue.

The extraction+gate happen SERVER-SIDE (mem0 runs ADDITIVE_EXTRACTION_PROMPT + the gate as
custom_instructions) — same as mem0 cloud. The worker is the client-side reliability wrapper only.
"""

from __future__ import annotations

import logging
import re as _re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# HTTP status appearing in an add() error string. Two anchored shapes, so a stray 3-digit number in
# the body (a memory id fragment, a byte count) can't be misread as a status:
#   (a) after an explicit marker:  "HTTP 400: ...", "HTTPError 413", "status_code=422", "status: 400"
#   (b) inside an httpx/requests-style quoted status phrase, where the code is followed by a
#       recognised HTTP reason word: "Client error '400 Bad Request' for url ...",
#       "Server error \"502 Bad Gateway\"".  We require the reason word so a quoted '404 files' in
#       free text can't trip it.
_STATUS_MARKER_RE = _re.compile(
    r"(?:http(?:\s*error)?|status(?:[_ ]?code)?)\s*[:=]?\s*(\d{3})", _re.IGNORECASE)
_STATUS_QUOTED_RE = _re.compile(
    r"['\"]\s*(\d{3})\s+(?:bad request|unauthorized|forbidden|not found|method not allowed|"
    r"not acceptable|request timeout|conflict|gone|length required|precondition|payload too large|"
    r"request entity too large|uri too long|unsupported media|unprocessable|too many requests|"
    r"internal server error|not implemented|bad gateway|service unavailable|gateway timeout)",
    _re.IGNORECASE)
# Words that indicate the request never left the client (nothing was written -> bounded retry is safe).
_NOT_SENT_MARKERS = (
    "refused", "name or service", "nodename", "no route", "getaddrinfo",
    "failed to establish", "cannot connect", "connection error",
    "connection reset", "connection aborted", "broken pipe on send",
)


def _status_from_msg(msg: str):
    """Extract an HTTP status code from an error string, or None. Tries the explicit-marker shape
    first, then the quoted httpx/requests reason-phrase shape."""
    m = _STATUS_MARKER_RE.search(msg) or _STATUS_QUOTED_RE.search(msg)
    return int(m.group(1)) if m else None


def _classify_add_error(exc: Exception) -> str:
    """Classify an add() exception into one of three fail-closed buckets. THE SAFETY BIAS: only an
    EXPLICIT 4xx client status is treated as "nothing written / retrying is pointless"; everything
    ambiguous defaults to 'possibly_written' so a maybe-committed row is never abandoned (Greptile P1).

      'deterministic_client_error' -> the server rejected the payload (explicit 4xx, except 408/429):
          nothing stored AND the same payload can never succeed -> bounded/dead-letter (poison row).
      'not_sent'                   -> the request never reached the server (connection-level failure):
          nothing stored, but a RETRY may succeed -> bounded retry.
      'possibly_written'           -> anything else (timeout, 5xx, opaque error): the add MAY have
          committed on the server -> requeue fail-closed forever, never dead-letter (never abandon a
          possible secret-bearing row).
    """
    msg = f"{type(exc).__name__} {exc}".lower()
    code = _status_from_msg(msg)
    if code is not None:
        # 408 Request Timeout / 429 Too Many Requests are transient 4xx -> retryable, and the
        # server may have partially processed -> treat as possibly_written (fail-closed).
        if code in (408, 429):
            return "possibly_written"
        if 400 <= code < 500:
            return "deterministic_client_error"
        return "possibly_written"  # 5xx (incl. 502 gateway/provider) -> maybe committed
    # No explicit status: only a clear connection-level pre-send failure is safe to bounded-retry.
    if any(k in msg for k in _NOT_SENT_MARKERS):
        return "not_sent"
    return "possibly_written"  # unknown/opaque -> fail closed


class CaptureDrainWorker:
    def __init__(
        self,
        queue,                                   # CaptureQueue
        *,
        add_fn: Callable[[List[Dict[str, str]], Dict[str, Any]], Any],
        recall_idem_fn: Callable[[str], int],    # -> count of existing rows with this capture_idem
        scrub_fn: Callable[[List[str]], "tuple[List[str], List[dict]]"],
        forget_fn: Optional[Callable[[str], None]] = None,   # forget a secret-bearing memory by id
        get_written_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,  # rows for a capture_idem
        gate: str = "",
        model: str = "",
        write_filters: Optional[Dict[str, Any]] = None,
        poll_interval_s: float = 2.0,
        lease_s: float = 120.0,
        backoff_base_s: float = 30.0,
        max_attempts: int = 5,
        breaker_open_fn: Optional[Callable[[], bool]] = None,
        alert_fn: Optional[Callable[[str], None]] = None,
        router: Optional[Any] = None,
    ):
        self._q = queue
        self._add = add_fn
        self._recall_idem = recall_idem_fn
        self._scrub = scrub_fn
        self._forget = forget_fn
        self._get_written = get_written_fn
        self._gate = gate
        self._model = model
        self._write_filters = dict(write_filters or {})
        self._poll = poll_interval_s
        self._lease_s = lease_s
        self._backoff = backoff_base_s
        self._max_attempts = max_attempts
        self._breaker_open = breaker_open_fn or (lambda: False)
        self._alert = alert_fn or (lambda m: logger.error("mem0 capture alert: %s", m))
        # Arm-B two-pass capture router (Phase 2.5), flag-gated: None => today's behavior EXACTLY
        # (byte-identical). When injected, it runs ADDITIVELY after the unchanged mem0 add() path:
        # two dedicated extraction passes -> deterministic class router -> world/event facts staged
        # to disk. It NEVER touches mem0 and is fully fail-soft (a router error never fails a turn).
        self._router = router
        # Scrub failures never dead-letter (a secret must not be abandoned): requeue indefinitely
        # with a capped backoff, and escalate once at this attempt threshold.
        self._scrub_backoff_cap_s = 3600.0
        self._scrub_alert_after = max(self._max_attempts, 3)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # observability counters (read by the digest). scrub_dead = a secret MAY be live in the store.
        self.stats = {"drained": 0, "dead": 0, "retried": 0, "reaped": 0, "scrubbed": 0, "scrub_dead": 0}

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mem0-capture-drain")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    # ---- the loop ----------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                worked = self.drain_once()
            except Exception as e:  # a loop iteration must never kill the worker
                logger.warning("capture drain iteration error: %s", e)
                worked = False
            # reaper sweep each idle pass (cheap)
            try:
                r = self._q.reap(backoff_s=self._backoff, max_attempts=self._max_attempts)
                self.stats["reaped"] += r.get("requeued_env", 0) + r.get("requeued_fault", 0)
                self.stats["dead"] += r.get("dead", 0)
            except Exception as e:
                logger.debug("reaper error: %s", e)
            if not worked:
                self._stop.wait(self._poll)

    def drain_once(self) -> bool:
        """Process at most one due row. Returns True if a row was handled."""
        if self._breaker_open():
            return False
        row = self._q.lease_one(lease_s=self._lease_s)
        if row is None:
            return False
        key = row["idem_key"]
        payload = row["payload"]
        messages = [
            {"role": "user", "content": payload.get("user", "")},
            {"role": "assistant", "content": payload.get("assistant", "")},
        ]
        # EXACTLY-ONCE (D-8): if the add already ran on a prior lease (crash before mark_done),
        # rows exist -> mark done WITHOUT re-adding.
        # FAIL-CLOSED (Greptile P1): if the idem check itself FAILS (transient search fault), we
        # cannot tell "new" from "already-written". Re-adding on an unknown = duplicate rows the
        # SQLite queue can't prevent. So on an idem-check error, REQUEUE (bounded) instead of adding.
        try:
            if self._recall_idem(key) > 0:
                # rows already exist. Still run the scrub before completing — a prior lease may have
                # added but crashed/failed BEFORE the scrub ran, so the shortcut must not skip it.
                if self._scrub_written_or_requeue(key, row):
                    return True
                self._q.mark_done(key)
                self.stats["drained"] += 1
                return True
        except Exception as e:
            # The idem check FAILED (transient search fault) so we can't tell "new" from
            # "already-written". Two cases (Greptile P1):
            #  - FIRST lease (attempts==0): nothing was written yet, so a bounded retry that may
            #    eventually dead-letter is fine — there's no secret to abandon.
            #  - a LATER lease (attempts>0) OR a row whose add already COMMITTED on a prior lease
            #    (add_committed=1, which survives a crash+reap that reset attempts to 0): a row
            #    (possibly secret-bearing) may already be live. We must NOT dead-letter and abandon
            #    it — requeue indefinitely (like the scrub path) so it's eventually read+scrubbed.
            prior_write_possible = int(row.get("attempts", 0)) > 0 or bool(row.get("add_committed"))
            attempts = int(row.get("attempts", 0)) + 1
            backoff = min(self._backoff * (2 ** min(attempts - 1, 12)), self._scrub_backoff_cap_s)
            if prior_write_possible:
                self._q.mark_scrub_retry(key, backoff_s=backoff,
                                         error=f"idem-check-failed(post-write): {str(e)[:260]}")
                self.stats["retried"] += 1
                if attempts == self._scrub_alert_after:
                    self.stats["scrub_dead"] += 1
                    self._alert(
                        f"CAPTURE IDEM-CHECK STUCK for row {key!r} after {attempts} attempts after a "
                        f"possible prior write — a memory (maybe secret-bearing) may be live+unscanned; "
                        f"auto-retry continues but manual check advised: {e}")
                logger.warning("capture idem pre-check failed after possible prior write; requeued "
                               "(never dead-letter, no re-add): %s", e)
                return True
            status = self._q.mark_retry(key, backoff_s=self._backoff * (2 ** (attempts - 1)),
                                        error=f"idem-check-failed: {str(e)[:280]}",
                                        max_attempts=self._max_attempts)
            if status == "dead":
                self.stats["dead"] += 1
                logger.error("capture idem-check unresolved after %d attempts on first lease; "
                             "dead-lettered (nothing written yet — no secret abandoned): %s", attempts, e)
            else:
                self.stats["retried"] += 1
                logger.warning("capture idem pre-check failed; requeued (fail-closed, no re-add): %s", e)
            return True

        # SERVER-SIDE extraction + gate. Stamp capture_idem so the reconcile can find the rows.
        kwargs = dict(self._write_filters)
        md = dict(kwargs.get("metadata") or {})
        md["capture_idem"] = key
        kwargs["metadata"] = md
        if self._gate:
            kwargs["prompt"] = self._gate
        if self._model:
            kwargs["model"] = self._model
        try:
            added_count = self._add(messages, kwargs)
            # STICKY signal FIRST (Greptile P1): the remote row now exists. Persist add_committed
            # BEFORE record_verdict so a crash in this window can't leave a written row with no
            # durable "a write happened" marker. It survives a crash+reap that resets attempts to 0,
            # so a later idem-check failure knows a possibly-secret-bearing row is live and must not
            # be abandoned.
            self._q.mark_add_committed(key)
            self._q.record_verdict(key, "ok")
        except Exception as e:
            self._q.record_verdict(key, "fault")
            attempts = int(row.get("attempts", 0)) + 1
            # FAIL-CLOSED classification (Greptile P1). A timeout / read error / 5xx / opaque error can
            # occur AFTER the server already committed /memories, so the add may be live+unscanned — we
            # must NOT dead-letter and abandon it. Only two classes are provably safe to bound/dead-
            # letter (nothing was written, no secret to abandon):
            #   - 'not_sent'    : a connection-level PRE-SEND failure -> bounded retry (may recover), and
            #   - 'deterministic_client_error' : the server rejected the payload with an explicit 4xx
            #     -> nothing stored AND retrying the identical payload can never succeed -> DEAD-LETTER
            #     immediately (poison row; e.g. a malformed/oversized payload).
            # Everything else defaults to 'possibly_written' -> requeue forever + escalate, never dead.
            klass = _classify_add_error(e)
            if klass == "possibly_written":
                backoff = min(self._backoff * (2 ** min(attempts - 1, 12)), self._scrub_backoff_cap_s)
                self._q.mark_scrub_retry(key, backoff_s=backoff, error=f"add-ambiguous: {str(e)[:270]}")
                self.stats["retried"] += 1
                if attempts == self._scrub_alert_after:
                    self.stats["scrub_dead"] += 1
                    self._alert(
                        f"CAPTURE ADD AMBIGUOUS for row {key!r} after {attempts} attempts (a timeout/"
                        f"read error may have left a written, unscanned memory live); auto-retry "
                        f"continues but manual check advised: {e}")
                logger.warning("capture add failed AMBIGUOUSLY (maybe committed); requeued "
                               "fail-closed (never dead-letter): %s", e)
                return True
            if klass == "deterministic_client_error":
                # Nothing written + never-succeeds -> dead-letter now (max_attempts=1 forces it at the
                # current attempt) rather than spinning bounded retries on a payload the server refuses.
                self._q.mark_retry(key, backoff_s=0.0, max_attempts=1,
                                   error=f"deterministic-reject: {str(e)[:270]}")
                self.stats["dead"] += 1
                logger.warning("capture turn dead-lettered (deterministic client rejection, nothing "
                               "written, retry cannot succeed): %s", e)
                return True
            # klass == "not_sent": pre-send connection failure, safe to bounded-retry then dead-letter.
            status = self._q.mark_retry(key, backoff_s=self._backoff * (2 ** (attempts - 1)),
                                        error=str(e)[:300], max_attempts=self._max_attempts)
            if status == "dead":
                self.stats["dead"] += 1
                logger.warning("capture turn dead-lettered after %d attempts (pre-send failure, "
                               "nothing written): %s", attempts, e)
            else:
                self.stats["retried"] += 1
            return True

        # POST-WRITE SCRUB (INV-4). See _scrub_written_or_requeue — fail-closed: never complete a row
        # whose scrub boundary we could not prove clean. require_rows when the server said it wrote
        # >=1 memory: an empty read then means index-lag (eventual consistency), not "nothing written".
        if self._scrub_written_or_requeue(key, row, require_rows=bool(added_count)):
            return True

        # ARM-B ROUTER (Phase 2.5), flag-gated + ADDITIVE + fail-soft. The mem0 add() above already
        # handled the preference/ops_state facts (unchanged path). Here — only when a router is wired
        # (flag ON) — the two dedicated passes extract world/event facts and stage them to disk. This
        # runs AFTER the row's mem0 write+scrub is proven clean and BEFORE mark_done, but a router
        # failure must NEVER requeue or fail the turn: the mem0 write is already durable and complete.
        self._maybe_route(key, payload, messages)

        self._q.mark_done(key)
        self.stats["drained"] += 1
        return True

    def _maybe_route(self, key, payload, messages) -> None:
        """Invoke the Arm-B router for this turn if one is wired (flag ON). Fully fail-soft: any
        error is swallowed (the mem0 write path is already complete and durable). No-op when the
        router is None -> byte-identical to today."""
        if self._router is None:
            return
        try:
            session = payload.get("session_id") or "default"
            res = self._router.route_turn(
                payload.get("user", ""), payload.get("assistant", ""),
                turn_id=key[:16], session=session, ts=payload.get("ts"))
            if res.get("error"):
                logger.warning("capture-router: turn %s routed with extract error: %s",
                               key[:16], res["error"])
        except Exception as e:
            logger.warning("capture-router: routing failed for turn %s (mem0 write unaffected): %s",
                           key[:16], e)


    def _scrub_written_or_requeue(self, key, row, *, require_rows: bool = False) -> bool:
        """Deterministically scrub the rows written for `key` and FORGET any secret-bearing one.
        The salience gate is NOT a reliable secret boundary (it leaked a bot token in the eval), so
        this is defense-in-depth (INV-4). FAIL-CLOSED (Greptile P1): if the rows can't be READ or a
        FORGET fails, requeue (bounded) instead of completing — a secret must never be left
        recallable behind a done row. The scrub is idempotent (scanning a clean row is a no-op), so
        it's safe to re-run on the exactly-once shortcut path too.

        require_rows: set True when THIS drain just committed add() — the server's metadata search is
        eventually-consistent, so an EMPTY read right after the write is INCONCLUSIVE (the row exists
        but isn't index-visible yet). Treat empty-as-inconclusive and requeue rather than completing
        and leaving a just-written secret unscanned (Greptile P1). On the exactly-once shortcut path
        (require_rows=False) an empty read is fine — the row genuinely extracted nothing.

        Returns True if the row was requeued/dead-lettered (caller must stop); False if clean.
        """
        if not (self._get_written and self._forget):
            return False
        try:
            written = self._get_written(key)   # [{id, memory}] — may raise on transient search fault
            if require_rows and not written:
                raise RuntimeError(
                    "post-add scrub read returned 0 rows for a just-written add (metadata search "
                    "not yet consistent) — inconclusive, failing closed until the row is visible")
            for r in written:
                txt = r.get("memory", "") or ""
                _, dropped = self._scrub([txt])
                if dropped:
                    self._forget(r.get("id", ""))   # may raise; requeue below if so
                    self.stats["scrubbed"] += 1
                    logger.warning("capture scrubbed a secret-bearing memory (reason=%s)",
                                   dropped[0].get("reason"))
            return False
        except Exception as e:
            # A scrub that can't be PROVEN clean must NOT be abandoned: a dead-letter here would
            # leave a secret-bearing memory live+recallable with no automatic path to scrub it
            # (Greptile P1). So scrub failures requeue INDEFINITELY with a capped backoff (the scrub
            # is cheap + idempotent), and escalate LOUDLY once they cross a threshold so an operator
            # can intervene — but the automatic retry never stops.
            attempts = int(row.get("attempts", 0)) + 1
            backoff = min(self._backoff * (2 ** min(attempts - 1, 12)), self._scrub_backoff_cap_s)
            self._q.mark_scrub_retry(key, backoff_s=backoff, error=f"scrub-failed: {str(e)[:280]}")
            self.stats["retried"] += 1
            if attempts == self._scrub_alert_after:
                self.stats["scrub_dead"] += 1   # "a secret may be live" signal for the digest
                self._alert(
                    f"SECRET SCRUB STUCK for capture row {key!r} after {attempts} attempts — a "
                    f"secret-bearing memory may remain recallable in mem0; auto-retry continues but "
                    f"manual scrub advised: {e}")
                logger.error("capture SCRUB stuck after %d attempts (secret may be live — escalated, "
                             "still retrying): %s", attempts, e)
            else:
                logger.warning("capture post-write scrub failed; requeued to retry the scrub: %s", e)
            return True
