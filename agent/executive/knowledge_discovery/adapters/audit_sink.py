"""Production monitoring audit sink adapter for the EvidencePackEngine.

The engine emits dict-shaped audit events when high-severity knowledge
conflicts are detected during ``discover()`` / ``dry_run()``. The
production wiring forwards those events into the process monitoring
emitter (``agent.monitoring.emitter.MonitoringEmitter``) under a fixed
audit schema.

This adapter is the seam between the engine and the monitoring
emitter. It is intentionally tiny and constrained:

* The emitter is *injected* via the constructor — the adapter never
  resolves a singleton (``get_emitter()``) and never imports from
  ``hermes_cli``, the gateway, or the TUI gateway. Composition is the
  engine's responsibility.
* The emitter is *borrowed* — the adapter stores the reference, calls
  ``emit()`` on it once per accepted event, and never calls or exposes
  ``close()``. The emitter's lifetime is owned by the caller.
* ``emit(event)`` is *synchronous* and the hot-path invariant is
  absolute: an audit failure NEVER propagates back into the engine and
  NEVER changes the EvidencePack result. Normalization errors, hashing
  errors, emitter errors, shutdown errors, and diagnostic errors are
  all swallowed.
* Input is *never mutated*. Unknown input fields are dropped. Missing,
  malformed, or oversized identifier fields cause the entire event to
  be dropped. Oversized values are NEVER truncated — the upstream
  payload is rejected wholesale.
* Output is *exactly* the nine canonical fields:

      event, gate_type, severity, conflict_id, objective_id,
      detected_at, schema_version, source, event_id

  No prompt text, no objective text, no snippets, no secrets,
  credentials, tokens, environment data, or filesystem paths ever
  leave the adapter.

* ``event_id`` is *deterministic* — it is derived from the canonical
  five-field tuple ``(schema_version, event, objective_id,
  conflict_id, detected_at)`` via a stable SHA-256 digest, so two
  adapters receiving the same logical event produce the same
  ``event_id``.

* The adapter holds *no resources*: no filesystem handle, no network
  socket, no thread, no queue, no background worker. There is nothing
  for ``close()`` to release.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping, Protocol

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# Canonical audit schema identity. The combination of these three
# fields uniquely identifies the audit surface for downstream
# monitoring consumers (OTLP exporters, log aggregators, dashboards).
SCHEMA_VERSION = "evidence_pack.audit.v1"
EVENT = "knowledge_conflict"
SOURCE = "evidence_pack"

# Allowlist values. The adapter REJECTS any event that does not match
# the single supported gate/severity pair — this is a known-only
# surface, not an open telemetry pipe.
ALLOWED_GATE_TYPE = "knowledge_conflict"
ALLOWED_SEVERITY = "high"

# Output field order. The wire payload is emitted in this order so the
# canonical representation is byte-stable across runs. Downstream
# consumers rely on the order for deterministic diffing / hashing.
OUTPUT_FIELD_ORDER: tuple[str, ...] = (
    "event",
    "gate_type",
    "severity",
    "conflict_id",
    "objective_id",
    "detected_at",
    "schema_version",
    "source",
    "event_id",
)

# Required bounded string fields. Each must be a non-empty string of
# at most ``_IDENTIFIER_MAX_LEN`` characters. Anything else — missing,
# wrong type, empty, oversized — drops the event outright. The cap is
# generous enough for any realistic identifier (UUIDs, ULIDs, ISO
# timestamps, hex digests) while preventing pathological inputs from
# inflating the audit payload.
IDENTIFIER_FIELDS: tuple[str, ...] = (
    "conflict_id",
    "objective_id",
    "detected_at",
)
_IDENTIFIER_MAX_LEN = 256

# Length of the SHA-256 hex prefix used as the event_id suffix.
# 16 hex chars = 64 bits of entropy — enough to be effectively unique
# within a single objective_id × conflict_id pair, and short enough to
# keep audit payloads terse.
_EVENT_ID_HEX_PREFIX = 16


# ─────────────────────────────────────────────────────────────────────
# Emitter protocol
# ─────────────────────────────────────────────────────────────────────


class MonitoringEmitterLike(Protocol):
    """Structural type for the injected emitter.

    The adapter only ever calls ``emit(payload)`` on the injected
    object. It does not call ``close()``, ``flush()``, ``subscribe()``,
    ``stats()``, ``reset_emitter_for_tests()``, or any other attribute.
    Any object that exposes a synchronous ``emit`` callable accepting a
    mapping/dict-shaped payload is a valid emitter for this adapter.
    """

    def emit(self, event: Any) -> None:
        ...


# ─────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────


class EvidencePackMonitoringAuditSink:
    """Synchronous monitoring audit sink adapter.

    The adapter accepts the engine's audit dicts, projects them onto a
    fixed nine-field schema, derives a deterministic ``event_id``, and
    delegates the resulting payload to the injected emitter exactly
    once per accepted call.

    The adapter is the seam between the engine and the monitoring
    emitter. It is deliberately small and side-effect free outside of
    the single delegated ``emit()`` call.

    Parameters
    ----------
    emitter
        A monitoring emitter (or any object structurally compatible
        with ``MonitoringEmitterLike``) supplied by the caller. The
        adapter borrows the reference — the caller retains ownership
        and is responsible for the emitter's lifecycle. The adapter
        does NOT call ``get_emitter()`` and does NOT resolve any
        singleton.

    Notes
    -----
    The adapter never propagates exceptions out of ``emit()``. Every
    failure mode — input validation, identifier normalization, hashing,
    emitter rejection, shutdown, or diagnostic logging — is swallowed
    internally. An audit failure MUST NOT affect the EvidencePack
    result.
    """

    def __init__(self, emitter: Any) -> None:
        # Borrow, never own. The adapter never closes the emitter and
        # never calls get_emitter() to resolve a singleton. The caller
        # owns the emitter's lifetime; this adapter is a pure pass-through.
        self._emitter = emitter

    # ── public API ─────────────────────────────────────────────────────

    def emit(self, event: Mapping[str, Any]) -> None:
        """Project ``event`` onto the audit schema and delegate.

        Accepts only ``gate_type='knowledge_conflict'`` events with
        ``severity='high'``. Required bounded string identifiers
        (``conflict_id``, ``objective_id``, ``detected_at``) are
        validated. The event is dropped — not truncated — when any
        required field is missing, malformed, empty, or oversized.
        Unknown input fields are dropped silently.

        The emitted payload contains exactly the canonical nine fields
        in the canonical order. ``event_id`` is derived deterministically
        from the canonical five-field tuple.

        This method never propagates an exception. All internal
        failures (validation, hashing, emitter rejection, shutdown,
        diagnostic) are swallowed.
        """
        try:
            payload = self._build_payload(event)
        except Exception:
            # Normalization / hashing / validation failures must never
            # escape — the engine relies on this seam being total.
            logger.debug("audit sink payload build failed", exc_info=True)
            return

        if payload is None:
            # Event was rejected by validation. Drop silently — this
            # is the normal path for unsupported gate/severity,
            # missing identifiers, or oversized values.
            return

        try:
            self._emitter.emit(payload)
        except Exception:
            # The emitter is itself expected to be non-raising (its
            # documented contract), but we swallow defensively so a
            # monitoring-side failure can NEVER affect the engine.
            logger.debug("audit sink emitter raised", exc_info=True)
            return

    # ── internals ─────────────────────────────────────────────────────

    def _build_payload(
        self, event: Mapping[str, Any]
    ) -> "dict[str, Any] | None":
        """Project the input onto the canonical audit schema.

        Returns ``None`` when the event should be dropped. Returns a
        fresh dict (never the input) on success. The returned dict is
        the exact payload that will be handed to the emitter.
        """
        if not isinstance(event, Mapping):
            return None

        # Allowlist gate_type / severity — the single supported pair.
        gate_type = event.get("gate_type")
        if gate_type != ALLOWED_GATE_TYPE:
            return None
        severity = event.get("severity")
        if severity != ALLOWED_SEVERITY:
            return None

        # Required bounded string identifiers. Any missing, non-string,
        # empty-after-strip, or oversized value drops the entire event.
        identifiers: dict[str, str] = {}
        for field in IDENTIFIER_FIELDS:
            value = event.get(field)
            if not _is_bounded_identifier(value):
                return None
            identifiers[field] = value  # type: ignore[assignment]

        # Compute the deterministic event_id from the canonical tuple.
        # The tuple order is fixed (schema_version, event, objective_id,
        # conflict_id, detected_at) so identical inputs always produce
        # identical event_ids regardless of dict ordering.
        event_id = _compute_event_id(
            schema_version=SCHEMA_VERSION,
            event=EVENT,
            objective_id=identifiers["objective_id"],
            conflict_id=identifiers["conflict_id"],
            detected_at=identifiers["detected_at"],
        )

        # Build a fresh dict in the canonical field order. The input
        # mapping is never touched — every output value comes from
        # the adapter's validated state, not from ``event`` directly.
        payload: dict[str, Any] = {
            "event": EVENT,
            "gate_type": ALLOWED_GATE_TYPE,
            "severity": ALLOWED_SEVERITY,
            "conflict_id": identifiers["conflict_id"],
            "objective_id": identifiers["objective_id"],
            "detected_at": identifiers["detected_at"],
            "schema_version": SCHEMA_VERSION,
            "source": SOURCE,
            "event_id": event_id,
        }
        # Defensive: assert exact shape. This is a hard contract — any
        # addition or omission breaks the audit schema. Cheap enough to
        # run on every accepted event and catches accidental drift
        # before it reaches the monitoring pipeline.
        assert set(payload.keys()) == set(OUTPUT_FIELD_ORDER)
        return payload


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _is_bounded_identifier(value: Any) -> bool:
    """Return True iff ``value`` is a non-empty bounded string.

    A valid identifier is a Python ``str`` whose ``strip()`` result is
    non-empty and whose raw length is at most ``_IDENTIFIER_MAX_LEN``.
    Bytes, ints, and other non-string types are rejected. Empty,
    whitespace-only, and oversized values are rejected. Values are
    never truncated — an oversized identifier causes the event to be
    dropped.
    """
    if not isinstance(value, str):
        return False
    if not value.strip():
        return False
    if len(value) > _IDENTIFIER_MAX_LEN:
        return False
    return True


def _compute_event_id(
    *,
    schema_version: str,
    event: str,
    objective_id: str,
    conflict_id: str,
    detected_at: str,
) -> str:
    """Derive a deterministic ``event_id`` from the canonical tuple.

    The event_id is a stable hash of the five canonical fields in a
    fixed order, prefixed with the schema slug for human readability:

        ``epa1-<16 hex chars of SHA-256>``

    The canonical-tuple hashing means two adapters — or two
    invocations of the same adapter — that receive the same logical
    event produce the same event_id, regardless of how the input
    dict ordered its keys.
    """
    payload = (
        f"{schema_version}|{event}|{objective_id}|"
        f"{conflict_id}|{detected_at}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"epa1-{digest[:_EVENT_ID_HEX_PREFIX]}"


__all__ = [
    "EvidencePackMonitoringAuditSink",
    "MonitoringEmitterLike",
    "SCHEMA_VERSION",
    "EVENT",
    "SOURCE",
    "ALLOWED_GATE_TYPE",
    "ALLOWED_SEVERITY",
    "OUTPUT_FIELD_ORDER",
    "IDENTIFIER_FIELDS",
]