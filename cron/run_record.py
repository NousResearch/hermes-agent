"""Plan 056-D — the Hermes-side run-record emitter (vendored ``RunRecord``).

Why this exists (agentic-hub ``docs/system/STATEFULNESS.md §4.1, §4.3, §5.4-5``):
the agentic system has THREE producers of work — the orchestrator (cloud drain),
Hermes (cron + Slack-triggered tasks), and Claude-Code (the harness) — but only
the orchestrator emits a typed, queryable ``RunRecord``. 056-D brings Hermes into
the **shared ``run_id`` join key** so a Hermes-initiated → orchestrator-executed
chain is one queryable subgraph.

This fork (NousResearch/hermes-agent) CANNOT import agentic-hub, so we **vendor a
minimal ``RunRecord``** that emits the SAME JSON shape as the canonical
``orchestrator/activities/run_record.py::RunRecord`` (056-B). Field-for-field
parity (``schema_version``, ``run_id``, ``parent_run_id``, ``producer``, ``kind``,
``started``, ``ended``, ``status``, ``plan_id``, ``phase_id``, ``commit_sha``,
``pr_url``, ``branch``, ``spend_usd``, ``artifact_keys``, ``memory_refs``,
``notes``) so the orchestrator's validate-on-read accepts it unchanged.

Design constraints (mirroring 056-B + the cron module ethos):

- **Stdlib only.** Like ``cron/daily_brief.py`` / ``cron/follow_up_sweep.py`` this
  is a thin orchestration layer; no optional deps. A plain ``dataclass`` builds the
  dict; ``json`` serializes it.
- **Pure deterministic builder — never reads the clock.** Every value
  (timestamps, status, spend) is PASSED IN by the caller. ``new_run_id()`` is the
  ONE place an id is minted, and it is called by the *initiator* of an action, not
  inside the record builder — so the record itself stays a pure snapshot (parity
  with 056-B, which never calls ``datetime.now()`` inside the emitter).
- **Fail-soft everywhere.** A run-record emit must NEVER break a cron run. The
  emit helper swallows every exception and logs (mirrors the daily-brief "honest
  failure" posture, but for the *observability* side-channel, not the brief
  itself).
- **Shared ``run_id`` only — no new id scheme.** ``parent_run_id`` is carried
  (for the future cross-producer campaign join, G4) but never acted on.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Parity with agentic-hub ``run_record.RUN_RECORD_SCHEMA_VERSION``. Bump in LOCKSTEP
# with the canonical schema if a field's meaning changes incompatibly — the
# orchestrator's validate-on-read rejects any unknown ``schema_version``.
RUN_RECORD_SCHEMA_VERSION = "1"

# This producer. Parity with the canonical ``Producer`` Literal
# (``orchestrator | hermes | claude-code``).
PRODUCER_HERMES = "hermes"


def new_run_id() -> str:
    """Mint a fresh ``run_id`` for a Hermes-initiated action.

    Called by the INITIATOR of an action (a cron tick, a Slack-triggered task),
    NOT inside the record builder — so the record stays a pure snapshot. A plain
    uuid4 hex; the only constraint (056-A) is that all three producers share ONE
    id *namespace* (an opaque string), which uuid4 satisfies.
    """
    return uuid.uuid4().hex


@dataclass
class RunRecord:
    """One typed, versioned record of a single unit of Hermes work.

    Vendored parity with ``orchestrator/activities/run_record.py::RunRecord``
    (056-B). ``to_dict()`` / ``to_json()`` emit the SAME field shape so the
    orchestrator (or Atlas) can validate-on-read a Hermes record unchanged.
    Every field is supplied by the caller — the builder generates nothing.
    """

    run_id: str
    producer: str = PRODUCER_HERMES
    kind: str = "brief"
    status: str = "ok"
    schema_version: str = RUN_RECORD_SCHEMA_VERSION
    parent_run_id: Optional[str] = None
    started: Optional[str] = None
    ended: Optional[str] = None
    plan_id: Optional[str] = None
    phase_id: Optional[str] = None
    commit_sha: Optional[str] = None
    pr_url: Optional[str] = None
    branch: Optional[str] = None
    spend_usd: float = 0.0
    artifact_keys: list[str] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Emit the canonical 056-B JSON object (key order is parity-stable)."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "producer": self.producer,
            "kind": self.kind,
            "started": self.started,
            "ended": self.ended,
            "status": self.status,
            "plan_id": self.plan_id,
            "phase_id": self.phase_id,
            "commit_sha": self.commit_sha,
            "pr_url": self.pr_url,
            "branch": self.branch,
            "spend_usd": self.spend_usd,
            "artifact_keys": list(self.artifact_keys),
            "memory_refs": list(self.memory_refs),
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def build_run_record(
    *,
    run_id: str,
    kind: str = "brief",
    status: str = "ok",
    started: Optional[str] = None,
    ended: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    notes: str = "",
    memory_refs: Optional[list[str]] = None,
    spend_usd: float = 0.0,
    **extra: Any,
) -> RunRecord:
    """Build a ``producer="hermes"`` ``RunRecord``. Caller supplies every value.

    A thin keyword constructor so cron call-sites read declaratively. ``extra``
    forwards any remaining 056-B fields (e.g. ``plan_id``, ``pr_url``) without
    bloating the signature.
    """
    return RunRecord(
        run_id=run_id,
        producer=PRODUCER_HERMES,
        kind=kind,
        status=status,
        started=started,
        ended=ended,
        parent_run_id=parent_run_id,
        notes=notes,
        memory_refs=list(memory_refs or []),
        spend_usd=spend_usd,
        **extra,
    )


# ---------------------------------------------------------------------------
# Fail-soft emit
# ---------------------------------------------------------------------------

# Where Hermes writes its run records locally. The LIVE wiring (shipping these to
# the shared S3 artifact store / Atlas so they join the orchestrator's records) is
# a Blake-gated ``[manual]`` step — this autonomous build writes a local JSONL
# breadcrumb under $HERMES_HOME so a record is durable + inspectable with zero
# infra. The path is overridable via ``HERMES_RUN_RECORD_PATH`` (an operator
# pointing at a synced/uploaded location).
_DEFAULT_RUN_RECORD_RELPATH = "run_records.jsonl"


def _default_run_record_path() -> str:
    override = os.environ.get("HERMES_RUN_RECORD_PATH")
    if override:
        return override
    # Lazy import so importing this module never triggers a config.yaml read.
    try:
        from hermes_constants import get_hermes_home

        return str(get_hermes_home().resolve() / _DEFAULT_RUN_RECORD_RELPATH)
    except Exception:  # pragma: no cover — config not available
        return _DEFAULT_RUN_RECORD_RELPATH


def emit_run_record(
    record: RunRecord,
    *,
    sink: Optional[Callable[[str], None]] = None,
    path: Optional[str] = None,
) -> bool:
    """Persist one ``RunRecord``. STRICTLY fail-soft → never raises.

    Appends the record as one JSON line. The ``sink`` is injectable for tests (a
    callable receiving the JSON string); the default appends to
    ``path`` / ``HERMES_RUN_RECORD_PATH`` / ``$HERMES_HOME/run_records.jsonl``.

    Returns True on a successful write, False if anything failed (logged, never
    raised) — so a cron job can call this and ignore the result without risking
    its own success.
    """
    try:
        line = record.to_json()
        if sink is not None:
            sink(line)
            return True
        target = path or _default_run_record_path()
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return True
    except Exception as exc:  # noqa: BLE001 — observability side-channel is best-effort
        logger.warning(
            "056-D: Hermes run-record emit failed (swallowed) run_id=%s: %s",
            getattr(record, "run_id", "?"),
            exc,
        )
        return False
