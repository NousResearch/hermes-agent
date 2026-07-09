"""Capture pipeline orchestration for mem0 salient auto-capture (Track A-lite, A3 wiring).

Owns the queue + drain-worker lifecycle and the two guards the review demanded, so the change to
the plugin __init__ is minimal (compose, don't inline):
  - the DURABLE QUEUE (capture_queue.CaptureQueue) + the DRAIN WORKER (capture_drain.CaptureDrainWorker)
  - the SALIENCE GATE string + its VERSION (D-11 gate-version guard): capture PAUSES if the live gate
    version != the eval-certified version, so we never write through an uncertified gate
  - the CROSS-PROCESS bg-review INTERLOCK (D-7): a single resolver both the plugin and the bgr writer
    read at DECISION TIME, so both-ON is impossible across processes

sync_turn() becomes: if capture on AND gate certified -> enqueue (tiny, non-blocking) and let the
worker do the slow server-side extraction. Everything here is degrade-safe: any failure disables
capture (falls back to today's off state) rather than breaking a turn (INV-3).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
_DEFAULT_QUEUE_PATH = "~/.hermes/state/mem0-capture/capture_queue.db"

# The gate version that was actually CERTIFIED by the A0 eval (gpt-5.4-mini, P=0.97 LB 0.92, n=90).
# This is PINNED IN CODE on purpose (Greptile P1): the D-11 guard must compare the shipped asset
# against a fixed, human-reviewed value — not against the asset's own self-reported version, which
# would let any swapped-in gate self-certify. The <hash> segment is the sha256[:12] of the certified
# gate STRING, so editing capture_gate_v3.txt without re-running the eval + bumping this constant
# breaks certification and auto-capture fail-safes OFF.
PINNED_GATE_VERSION = "v3:a1f60b86c7d6"


def load_certified_gate() -> tuple[str, str]:
    """Return (gate_string, gate_version). Empty gate ('','') if assets missing -> capture stays off
    (fail-safe: no certified gate => don't auto-capture)."""
    gate_path = os.path.join(_ASSET_DIR, "capture_gate_v3.txt")
    ver_path = os.path.join(_ASSET_DIR, "gate_version.txt")
    try:
        gate = open(gate_path, encoding="utf-8").read()
        raw = open(ver_path, encoding="utf-8").read().strip()
        # gate_version.txt is "GATE_VERSION=v3:<hash>"
        version = raw.split("=", 1)[1] if "=" in raw else raw
        if gate.strip() and version:
            return gate, version
    except Exception as e:
        logger.warning("mem0 capture: certified gate assets not loadable (%s) — capture disabled", e)
    return "", ""


def gate_string_matches_version(gate: str, version: str) -> bool:
    """True iff the gate STRING actually hashes to the hash embedded in its version tag
    (v<n>:<sha256[:12]>). Prevents a hand-edited gate keeping a certified-looking version tag."""
    try:
        import hashlib
        want = version.split(":", 1)[1]
        got = hashlib.sha256(gate.encode("utf-8")).hexdigest()[:len(want)]
        return bool(want) and got == want
    except Exception:
        return False


def bgr_write_allowed(capture_is_on: bool) -> bool:
    """Cross-process interlock (D-7): the bg-review mem0 writer calls this at DECISION TIME (each
    write attempt). If foreground auto-capture is ON, the bgr writer must NOT also write (both-ON
    impossible). Read fresh each call so a live capture flip is honored without a restart."""
    return not capture_is_on


class CapturePipeline:
    def __init__(
        self,
        *,
        capture_on_fn: Callable[[], bool],       # reads the LIVE capture state each call
        add_fn: Callable[[List[Dict[str, str]], Dict[str, Any]], Any],
        recall_idem_fn: Callable[[str], int],
        scrub_fn: Callable[[List[str]], "tuple[List[str], List[dict]]"],
        forget_fn: Optional[Callable[[str], None]],
        get_written_fn: Optional[Callable[[str], List[Dict[str, Any]]]],
        write_filters: Dict[str, Any],
        model: str,
        breaker_open_fn: Optional[Callable[[], bool]] = None,
        alert_fn: Optional[Callable[[str], None]] = None,
        queue_path: Optional[str] = None,
        expected_gate_version: Optional[str] = None,  # None => the code-pinned PINNED_GATE_VERSION
        router: Optional[Any] = None,   # Arm-B capture router (Phase 2.5); None => flag OFF (no-op)
    ):
        try:
            from .capture_queue import CaptureQueue
            from .capture_drain import CaptureDrainWorker
        except ImportError:  # flat import (unit tests run with PYTHONPATH=<dir>)
            from capture_queue import CaptureQueue
            from capture_drain import CaptureDrainWorker

        self._capture_on = capture_on_fn
        self._alert = alert_fn or (lambda m: logger.warning("mem0 capture alert: %s", m))
        self._gate, self._gate_version = load_certified_gate()
        # D-11 gate-version guard (Greptile P1): compare the shipped asset against the CODE-PINNED
        # certified version (default), NOT the asset's own self-reported version — otherwise any
        # swapped-in gate self-certifies. Also verify the gate STRING actually hashes to the version
        # tag, so a hand-edited gate that kept a certified-looking tag is rejected.
        self._expected_version = expected_gate_version or PINNED_GATE_VERSION
        self._certified = (
            bool(self._gate)
            and self._gate_version == self._expected_version
            and gate_string_matches_version(self._gate, self._gate_version)
        )
        if not self._certified:
            self._alert(
                f"mem0 auto-capture DISABLED: gate version mismatch/absent "
                f"(live={self._gate_version!r} expected={self._expected_version!r})")
        self._model = model
        qp = os.path.expanduser(queue_path or _DEFAULT_QUEUE_PATH)
        self._queue = CaptureQueue(qp)
        self._worker = CaptureDrainWorker(
            self._queue,
            add_fn=add_fn,
            recall_idem_fn=recall_idem_fn,
            scrub_fn=scrub_fn,
            forget_fn=forget_fn,
            get_written_fn=get_written_fn,
            gate=self._gate,
            model=model,
            write_filters=write_filters,
            breaker_open_fn=breaker_open_fn,
            alert_fn=self._alert,
            router=router,
        )
        self._started = False
        self._lock = threading.Lock()
        # Startup drain path (Greptile P1): if this process inherited pending/expired rows from a
        # prior lifetime AND capture is active, start the worker+reaper NOW — don't wait for a new
        # turn to arrive (an idle agent would otherwise never drain durable rows).
        self.maybe_start_pending()

    def maybe_start_pending(self) -> None:
        """Start the drain+reaper worker if capture is active and there is un-drained work already in
        the durable queue. Safe to call any time; no-op if already started, inactive, or empty."""
        try:
            if self._started or not self.active:
                return
            counts = self._queue.counts()
            if (counts.get("pending", 0) + counts.get("inflight", 0)) > 0:
                self.start()
        except Exception as e:
            logger.warning("mem0 capture: startup-drain check failed (non-fatal): %s", e)

    @property
    def active(self) -> bool:
        """Capture is active only if it's on AND the gate is certified (D-11)."""
        return self._certified and self._capture_on()

    def start(self) -> None:
        with self._lock:
            if self._started or not self._certified:
                return
            self._worker.start()
            self._started = True
            logger.info("mem0 capture pipeline started (gate %s, model %s, queue depth %s)",
                        self._gate_version, self._model, self._queue.counts())

    def stop(self) -> None:
        with self._lock:
            if self._started:
                self._worker.stop()
                self._started = False

    def enqueue_turn(self, user_content: str, assistant_content: str, *,
                     session_id: str = "", turn_ordinal: int = 0) -> bool:
        """The ONLY synchronous step (INV-3): a tiny durable INSERT. Returns True if enqueued.
        Degrade-safe: any failure is swallowed (never breaks the turn)."""
        if not self.active:
            return False
        try:
            try:
                from .capture_queue import idem_key
            except ImportError:
                from capture_queue import idem_key
            key = idem_key(session_id, turn_ordinal, user_content, assistant_content)
            enq = self._queue.enqueue(key, {"user": user_content, "assistant": assistant_content,
                                            "session_id": session_id})
            # Start the worker whenever capture is active and it isn't running yet — NOT only on a
            # brand-new insert (Greptile P1). After a restart with pending/expired rows already in
            # SQLite, a duplicate enqueue returns False; gating start on `enq` would leave those
            # durable rows (and the reaper) idle until some later unique turn. Reaching an active
            # enqueue means there is work to drain, so ensure the drain+reaper loop is up.
            if not self._started:
                self.start()
            return enq
        except Exception as e:
            logger.warning("mem0 capture enqueue failed (turn not captured, not broken): %s", e)
            return False

    def stats(self) -> Dict[str, Any]:
        out = {"certified": self._certified, "gate_version": self._gate_version,
               "queue": self._queue.counts()}
        out.update(self._worker.stats)
        return out
