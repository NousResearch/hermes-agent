"""kanban_crash_sweep_guard — the reclaim sweep may not act on a liveness substrate it cannot vouch for.

One module, one purpose: the two clauses of the platform-stl ruling ``t_63a20c59`` (2026-10-10)
that stand between a *bad liveness evaluation* and a fleet-wide outage. Both are decided
DETERMINISTICALLY -- a count, a shape and a timestamp, never a model and never prose a reader has to
interpret. Carrier card: ``t_fc2cf9ee``.

R4 -- the mass-crash abort
==========================

A reclaim sweep that would close ``>= threshold`` (default 3) runs in ONE tick is a
**host-systemic signature, not N card failures**. The sweep must refuse the whole write, stay
handed-off (nothing booked, no run ended, no claim released) and fail LOUD: one stderr line with a
deterministic signature, one board event on every card it would have taken, and one acting card
filed to the design authority. The failure mode this exists to stop is measured: 2026-10-10, all 21
``crashed`` runs on board ``migrations`` fell into three instants (8 at ``ended_at`` 14:20:37, 5 at
13:35:34, 8 at 00:01:23) and every one carried ``pid N not alive``; pid 70697 was measured ALIVE
eight minutes after the 14:20:37 sweep and still writing the live tree. NOT ONE of the 21 was a
genuine crash. With this guard the reclaim would have refused all three sweeps and left 21 cards and
~21 live workers alone.

**Which closures are counted, and why the class is narrowed.** The counted class is the
*unexplained-death* family -- a closure whose verdict is that the process is gone but which carries
no evidence of its own:

* ``unknown`` (``pid N not alive`` -- the measured census),
* ``nonzero_exit`` (``pid N exited with code N``),
* ``signaled`` (``pid N killed by signal N``).

That is exactly the class in which a broken liveness witness masquerades as N card failures.
``protocol_violation`` (the worker exited rc=0 and only skipped its paperwork) and
``terminal_provider`` (the provider rejected the credential/model) each carry their own evidence and
their own bounded budget, and ``rate_limited`` is not a crash at all. Counting those would refuse a
*legitimate* sweep of N cards whose workers each finished successfully -- the false-refusal storm
this guard must not become. The narrow class is named in code (:data:`UNEXPLAINED_DEATH_KINDS`), not
left to a reader of the sweep.

**One tick IS one ``ended_at``.** The ruling names two spellings of one signature: ">= 3 runs closed
``crashed`` in one tick, OR with one identical ``ended_at``". Within a single tick every closure is
stamped with the same instant by construction, so the two are one measurement and the guard is one
comparison. Nothing is inferred from card identity, prose or error text.

The abort is a verdict reached BEFORE the sweep writes, at the granularity of the sweep: a wrong
refusal costs a loud card and is reversible (raise the threshold, or disable it with
``KANBAN_MASS_CRASH_ABORT_THRESHOLD``), never a silent board mutation.

R6 -- the deployed-identity generation
=====================================

Deployment is the deliverable, and the *deployed* generation must be checked. This class has been
fixed and has recurred (t_1e6a58c3 2026-09-25, the 2026-09-26 six-worker tick, the 2026-09-29
regression window pinned by ``tests/hermes_cli/test_kanban_liveness_witness.py``, t_ae80d242,
2026-10-10) because each fix lived as an override or an in-memory build and vanished on the next tree
write or interpreter change.

Two mechanisms, both deterministic:

1. **The generation stamp.** The identity shape a dispatcher WRITES is stamped on disk
   (``<kanban_home>/kanban/dispatcher_identity_generation.json``) on the first tick and compared on
   every later tick. A divergence -- an interpreter move that changes the shape, a tree write that
   drops the witness bytes -- is detected in ONE tick and alerted, instead of being discovered by a
   false crash sweep hours later. Measured 2026-10-10, three incomparable shapes in circulation at
   once: ``'|<abs-epoch-cs>'`` (``.venv`` 3.11.16 + psutil), ``None`` -> ``'unverified'`` (the 3.14
   gateway interpreter, no psutil), and ``bootsession:<uuid>|<ref_cs>|<boot-relative start>`` (the
   live rows).

2. **The per-row hold.** The same comparison is made against the fingerprints actually READ off the
   running rows. When the shape the sweep READS was built by a different identity generation than
   the shape it would WRITE, the two values were never comparable and the row is HELD -- not
   reclaimed, not counted, its worker's claim left alone. This is R2's ``UNKNOWN -> HOLD`` at the
   granularity of one row, and it is the clause the 2026-10-10 outage needed: the live rows carried
   a three-field ``bootsession:`` fingerprint while the tree wrote a two-field one, so
   ``_pid_recycled`` compared two different kinds of value, called the difference "recycled", and
   booked 21 live workers as crashed. A row with no fingerprint (legacy) or an ``unverified`` marker
   is NOT held -- the existing bare-existence rule owns those, and holding them would strand every
   pre-fingerprint card.

Boundaries, deliberately:

* **The refusal path never mutates board state.** A refused sweep writes nothing; the loud path is a
  log line, an evidence event per card it would have taken, and one acting card for the design
  authority. A board that refuses the actor card is logged with the key it could not file and never
  takes the dispatcher tick down with it.
* **The per-row hold is the narrow interlock, not a whole-sweep refusal.** It never strands a
  genuinely dead worker of the current generation, because such rows are comparable by definition.
* **No import of :mod:`hermes_cli.kanban_db` at module scope** (import-cycle discipline, as in
  :mod:`hermes_cli.tree_identity`); the two helpers that need it late-bind.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Optional

logger = logging.getLogger(__name__)

# --- identity shapes (shared by R4's hold and R6's stamp) ------------------------------------

IDENTITY_SHAPE_NONE = "none"
IDENTITY_SHAPE_UNVERIFIED = "unverified"
IDENTITY_SHAPE_BOOT_WITNESS = "boot-witness"
IDENTITY_SHAPE_EPOCH_CS = "epoch-cs"
IDENTITY_SHAPE_COMPOSED = "composed"
IDENTITY_SHAPE_LEGACY_INT = "legacy-int"
IDENTITY_SHAPE_OTHER = "other"

#: Families whose values are *identity-bearing*: a comparison between two of these is a real
#: liveness comparison only if the shapes match exactly. ``none`` (a legacy row with no fingerprint)
#: and ``unverified`` are deliberately excluded -- they carry no identity to compare, and the
#: existing bare-existence rule owns them.
IDENTITY_SHAPE_FAMILIES: frozenset[str] = frozenset({
    IDENTITY_SHAPE_BOOT_WITNESS, IDENTITY_SHAPE_EPOCH_CS, IDENTITY_SHAPE_COMPOSED,
})


def shape_family(shape: str) -> str:
    """``"boot-witness/3"`` -> ``"boot-witness"``."""
    return str(shape).split("/", 1)[0]


def identity_shape(sample: Any) -> str:
    """Classify one ``worker_started_at`` fingerprint into a comparable generation token.

    The token names the STRUCTURE, not the value -- the family and the FIELD COUNT, because the two
    generations measured on 2026-10-10 differ exactly there (``bootsession:<uuid>|<ref>|<start>``,
    three fields, versus ``bootsession:<uuid>|<start>``, two). Two fingerprints whose tokens differ
    were never comparable; the caller holds the row instead of comparing them.
    """
    if sample is None:
        return IDENTITY_SHAPE_NONE
    if isinstance(sample, bool):
        return IDENTITY_SHAPE_OTHER
    if isinstance(sample, int):
        return IDENTITY_SHAPE_LEGACY_INT
    text = str(sample)
    if text == "":
        return IDENTITY_SHAPE_NONE
    if text == IDENTITY_SHAPE_UNVERIFIED:
        return IDENTITY_SHAPE_UNVERIFIED
    fields = text.count("|") + 1
    if text.startswith("bootsession:"):
        return f"{IDENTITY_SHAPE_BOOT_WITNESS}/{fields}"
    if text.isdigit():
        return IDENTITY_SHAPE_LEGACY_INT
    if "|" in text:
        head, _, tail = text.partition("|")
        if head == "" and tail.isdigit():
            return f"{IDENTITY_SHAPE_EPOCH_CS}/{fields}"
        return f"{IDENTITY_SHAPE_COMPOSED}/{fields}"
    return IDENTITY_SHAPE_OTHER


def shapes_comparable(write_shape: str, read_shape: str) -> bool:
    """False when a liveness comparison between these two fingerprints is meaningless.

    Equal shapes are always comparable. A pair that names two different identity-bearing families,
    or two field counts of the same family, is not -- that is the measured 2026-10-10 condition.
    Anything involving ``none`` / ``unverified`` / ``legacy-int`` is left to the existing
    bare-existence rule.
    """
    if write_shape == read_shape:
        return True
    if shape_family(write_shape) not in IDENTITY_SHAPE_FAMILIES:
        return True
    if shape_family(read_shape) not in IDENTITY_SHAPE_FAMILIES:
        return True
    return False


# --- R4: the mass-crash abort ----------------------------------------------------------------

#: Default number of unexplained-death closures in ONE tick that makes a sweep host-systemic.
MASS_CRASH_ABORT_THRESHOLD_DEFAULT = 3

#: Operator override. ``0`` (or any value < 2) disables the guard entirely -- the documented
#: rollback for a wrong refusal. Read at decision time, so a long-lived dispatcher picks it up
#: without a restart.
MASS_CRASH_ABORT_THRESHOLD_ENV = "KANBAN_MASS_CRASH_ABORT_THRESHOLD"

#: Board event kind appended to every card a refused sweep would have taken. Evidence, never a
#: status change: the card keeps its ``running`` status and its claim.
MASS_CRASH_ABORT_EVENT = "crash_sweep_aborted"

#: Stable, greppable prefix of the deterministic refusal signature.
MASS_CRASH_SIGNATURE_PREFIX = "mass-crash-sweep:"

#: The design authority the acting card is filed to.
MASS_CRASH_ACTOR_ASSIGNEE = "platform-stl"
MASS_CRASH_ACTOR_TITLE = (
    "[guard] reclaim sweep REFUSED: >= {count} runs would close `crashed` in one tick"
)
MASS_CRASH_ACTOR_EVENT = "mass_crash_sweep_actor_filed"
MASS_CRASH_ACTOR_IDEMPOTENCY_PREFIX = "mass-crash:"

#: The verdicts that mean "the process is gone" WITHOUT evidence of their own -- the only class a
#: broken liveness witness can forge. See the module docstring.
UNEXPLAINED_DEATH_KINDS = frozenset({"unknown", "nonzero_exit", "signaled"})

#: One acting card per signature per hour per board, so a persistent condition pages once, not once
#: per tick.
_ACTOR_WINDOW_SECONDS = 3600


@dataclass(frozen=True)
class SweepClosure:
    """One run a reclaim sweep is about to close, built by the caller from the sweep's own rows."""

    task_id: str
    pid: Optional[int]
    kind: str
    shape: str = IDENTITY_SHAPE_NONE
    protocol_violation: bool = False
    terminal_provider: bool = False
    rate_limited: bool = False
    claim_lock: Optional[str] = None

    @property
    def is_unexplained_death(self) -> bool:
        """True when this closure is the class a bad liveness evaluation can forge."""
        if self.protocol_violation or self.terminal_provider or self.rate_limited:
            return False
        return self.kind in UNEXPLAINED_DEATH_KINDS

    def incomparable_with(self, write_shape: str) -> bool:
        """True when this row's fingerprint cannot be compared with the shape we would write."""
        return not shapes_comparable(write_shape, self.shape)


@dataclass(frozen=True)
class MassCrashVerdict:
    """The R4 count decision. Deterministic: a count, a threshold and one timestamp."""

    abort: bool
    count: int
    threshold: int
    ended_at: int
    signature: str
    task_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SweepAssessment:
    """The whole R4/R6 answer for one sweep: what to refuse, and what to hold."""

    verdict: MassCrashVerdict
    held: tuple[str, ...] = ()
    """Task ids whose read/write identity shapes are incomparable -- not reclaimed, not counted."""
    held_shapes: tuple[str, ...] = ()
    """Distinct ``"<read->write>"`` shape pairs behind :attr:`held`, for the evidence line."""

    @property
    def refusal_needed(self) -> bool:
        return self.verdict.abort

    @property
    def alerting(self) -> bool:
        return self.verdict.abort or bool(self.held)

    @property
    def signature(self) -> str:
        return self.verdict.signature


def mass_crash_threshold(environ: Optional[Mapping[str, str]] = None) -> int:
    """The configured threshold; a value below 2 disables the guard (operator override)."""
    raw = (environ or os.environ).get(MASS_CRASH_ABORT_THRESHOLD_ENV)
    if raw is None or raw == "":
        return MASS_CRASH_ABORT_THRESHOLD_DEFAULT
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "kanban crash-sweep guard: %s=%r is not an integer; using %d",
            MASS_CRASH_ABORT_THRESHOLD_ENV, raw, MASS_CRASH_ABORT_THRESHOLD_DEFAULT,
        )
        return MASS_CRASH_ABORT_THRESHOLD_DEFAULT
    return 0 if value < 2 else value


def sweep_signature(count: int, ended_at: int) -> str:
    """``mass-crash-sweep:count=<n>,ended_at=<epoch>`` -- the deterministic refusal identity."""
    return f"{MASS_CRASH_SIGNATURE_PREFIX}count={int(count)},ended_at={int(ended_at)}"


def assess_sweep(
    closures: Iterable[SweepClosure],
    *,
    write_shape: Optional[str] = None,
    threshold: Optional[int] = None,
    now: Optional[int] = None,
) -> SweepAssessment:
    """Decide, before any write, what this sweep may not do.

    Rows whose fingerprint was built by a different identity generation than ``write_shape`` are HELD
    (R6/R2) and excluded from the count; the remaining unexplained-death closures are counted and the
    sweep is refused in full at ``>= threshold`` (R4). A threshold below 2 disables only the count
    guard -- the per-row hold still applies, because an incomparable comparison is a fact about the
    values, not a policy setting.
    """
    effective = mass_crash_threshold() if threshold is None else int(threshold)
    ended_at = int(time.time() if now is None else now)
    shape = write_shape or IDENTITY_SHAPE_NONE

    held: list[str] = []
    pairs: list[str] = []
    counted: list[str] = []
    for closure in closures:
        if closure.incomparable_with(shape):
            held.append(closure.task_id)
            pair = f"{closure.shape}->{shape}"
            if pair not in pairs:
                pairs.append(pair)
            continue
        if closure.is_unexplained_death:
            counted.append(closure.task_id)

    abort = bool(effective >= 2 and len(counted) >= effective)
    return SweepAssessment(
        verdict=MassCrashVerdict(
            abort=abort,
            count=len(counted),
            threshold=effective,
            ended_at=ended_at,
            signature=sweep_signature(len(counted), ended_at),
            task_ids=tuple(counted),
        ),
        held=tuple(held),
        held_shapes=tuple(pairs),
    )


def refusal_lines(assessment: SweepAssessment) -> list[str]:
    """The LOUD one-liners. Deterministic; suitable for the log, the event payload and the card."""
    verdict = assessment.verdict
    lines: list[str] = []
    if verdict.abort:
        lines.append(
            f"{MASS_CRASH_SIGNATURE_PREFIX} REFUSING this reclaim sweep: {verdict.count} runs would "
            f"close `crashed` in one tick (threshold {verdict.threshold}, ended_at {verdict.ended_at})."
        )
        lines.append(
            "A sweep this size is a host-systemic signature, not N card failures -- a broken liveness "
            "witness, not N broken cards. NOTHING was written: no run ended, no claim released, no "
            "failure counted. Cards taken: " + (", ".join(verdict.task_ids) or "(none)") + "."
        )
    if assessment.held:
        lines.append(
            f"{MASS_CRASH_SIGNATURE_PREFIX} HOLDING {len(assessment.held)} row(s) whose identity "
            f"shape is not comparable with this dispatcher's: "
            + ", ".join(assessment.held_shapes)
            + "."
        )
        lines.append(
            "A liveness comparison across two identity generations is meaningless, so these rows are "
            "not reclaimed and not counted (R2 UNKNOWN -> HOLD). Cards held: "
            + ", ".join(assessment.held) + "."
        )
    return lines


def _bullets(values: Iterable[str], *, code: bool = True) -> str:
    items = list(values)
    if not items:
        return "(none)"
    return ", ".join(f"`{v}`" if code else v for v in items)


def refusal_body(assessment: SweepAssessment, *, board: Optional[str] = None, tree: Any = None) -> str:
    """The acting card's body: the counts, the shapes, the timestamp, the cards, the tree."""
    verdict = assessment.verdict
    resolved_tree = tree if tree is not None else dispatching_tree()
    lines = [
        "## Observed",
        "",
        "```",
        *refusal_lines(assessment),
        "```",
        "",
        "Filed by the dispatcher's mass-crash sweep guard (R4/R6 of ruling `t_63a20c59`) --",
        "deterministic, no model was consulted.",
        "",
        f"- refused: `{verdict.abort}`",
        f"- signature: `{verdict.signature}`",
        f"- counted unexplained-death closures: {verdict.count} (threshold {verdict.threshold})",
        f"- tick instant (`ended_at`): {verdict.ended_at}",
        f"- board: `{board or '(current)'}`",
        f"- tree: `{resolved_tree}`",
        "",
        f"- cards taken ({len(verdict.task_ids)}): {_bullets(verdict.task_ids)}",
        f"- rows held on an incomparable identity shape ({len(assessment.held)}): "
        + _bullets(assessment.held),
        f"- shape pairs held (`<read->write>`): {_bullets(assessment.held_shapes)}",
        "",
        "## What was NOT done",
        "",
        "No run was ended, no claim released, no failure counted, no circuit breaker tripped. The",
        "cards keep `running` and their workers keep their claims. A refusal is loud and reversible;",
        "a silent board mutation is not.",
        "",
        "## Operator move",
        "",
        "Confirm the liveness witness (R1-R3 of the ruling; `tests/hermes_cli/test_kanban_liveness_witness.py`)",
        "before re-arming the sweep. To accept a genuine mass crash, raise or disable the count guard",
        f"with `{MASS_CRASH_ABORT_THRESHOLD_ENV}` (a value < 2 disables it). The half that cannot be",
        "disabled from here is the identity-shape hold: two generations of fingerprint were never",
        "comparable, so re-land the witness bytes (or re-record the generation stamp) instead.",
    ]
    return "\n".join(lines)


def _board_key(conn: Any, board: Optional[str]) -> str:
    """The board slug the actor's idempotency key is scoped to (connection first, argument second)."""
    from hermes_cli import kanban_db as kb

    if board:
        try:
            explicit = kb._normalize_board_slug(board)
        except Exception:
            explicit = None
        if explicit:
            return explicit
    try:
        for row in conn.execute("PRAGMA database_list").fetchall():
            if str(row[1]) == "main" and row[2]:
                db_file = Path(str(row[2]))
                if db_file.parent == Path(kb.kanban_home()):
                    return "default"
                if db_file.parent.name:
                    return db_file.parent.name
    except Exception:
        pass
    try:
        return kb._slug_or_default(board)
    except Exception:
        return "default"


def _actor_key(prefix: str, signature: str, board_key: str, now: int) -> str:
    bucket = int(now) // _ACTOR_WINDOW_SECONDS * _ACTOR_WINDOW_SECONDS
    return f"{prefix}{signature}:{board_key}:{bucket}"


def _file_actor(
    conn: Any,
    *,
    prefix: str,
    signature: str,
    title: str,
    body: str,
    event: str,
    event_payload: dict,
    board: Optional[str] = None,
    now: Optional[int] = None,
) -> Optional[str]:
    """File EXACTLY ONE acting card for a signature, idempotently, best-effort.

    The idempotency key (``<prefix><signature>:<board>:<hour-bucket>``) is the whole mechanism: N
    ticks hitting the same condition resolve to one card, and the condition can page again in the
    NEXT hour. A board that refuses the write is logged with the key it could not file and never
    takes the tick down.
    """
    from hermes_cli import kanban_db as kb

    moment = int(time.time() if now is None else now)
    board_key = _board_key(conn, board)
    key = _actor_key(prefix, signature, board_key, moment)
    try:
        existing = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1",
            (key,),
        ).fetchone()
        if existing is not None:
            return existing["id"]
        actor_id = kb.create_task(
            conn,
            title=title,
            body=body,
            assignee=MASS_CRASH_ACTOR_ASSIGNEE,
            created_by="kanban-dispatcher",
            idempotency_key=key,
        )
        with kb.write_txn(conn):
            kb._append_event(conn, actor_id, event, dict(event_payload, board=board_key))
        return actor_id
    except Exception as exc:
        logger.warning(
            "kanban crash-sweep guard: could not file the actor card for %s (key=%s): %s",
            signature, key, exc,
        )
        return None


def record_refusal_evidence(conn: Any, verdict: MassCrashVerdict) -> None:
    """Append the refusal event to EVERY card the sweep would have taken. Never mutates status."""
    from hermes_cli import kanban_db as kb

    payload = {
        "signature": verdict.signature,
        "count": verdict.count,
        "threshold": verdict.threshold,
        "ended_at": verdict.ended_at,
    }
    for task_id in verdict.task_ids:
        try:
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, MASS_CRASH_ABORT_EVENT, dict(payload))
        except Exception as exc:
            logger.warning(
                "kanban crash-sweep guard: could not append refusal evidence to %s: %s", task_id, exc,
            )


def alert_sweep_guard(
    conn: Any,
    assessment: SweepAssessment,
    *,
    board: Optional[str] = None,
    tree: Any = None,
    now: Optional[int] = None,
) -> Optional[str]:
    """The whole loud path for a refused sweep or a held row: log, per-card evidence, one card."""
    if not assessment.alerting:
        return None
    for line in refusal_lines(assessment):
        logger.error("kanban crash-sweep guard: %s", line)
    if assessment.verdict.abort:
        record_refusal_evidence(conn, assessment.verdict)
    verdict = assessment.verdict
    actor_signature = verdict.signature if verdict.abort else (
        "held:" + ",".join(assessment.held_shapes)
    )
    return _file_actor(
        conn,
        prefix=MASS_CRASH_ACTOR_IDEMPOTENCY_PREFIX,
        signature=actor_signature,
        title=MASS_CRASH_ACTOR_TITLE.format(
            count=verdict.count, threshold=verdict.threshold,
        ),
        body=refusal_body(assessment, board=board, tree=tree),
        event=MASS_CRASH_ACTOR_EVENT,
        event_payload={
            "signature": verdict.signature,
            "count": verdict.count,
            "threshold": verdict.threshold,
            "ended_at": verdict.ended_at,
            "task_ids": list(verdict.task_ids),
            "held": list(assessment.held),
            "held_shapes": list(assessment.held_shapes),
        },
        board=board,
        now=now,
    )


# --- R6: the deployed-identity generation stamp ----------------------------------------------

#: ``<kanban_home>/kanban/dispatcher_identity_generation.json`` -- machine-global, like the board
#: itself and like :func:`hermes_cli.tree_identity.record_path` (unchanged: this is a different
#: record, so it is a different file).
GENERATION_STAMP_FILENAME = "dispatcher_identity_generation.json"

GENERATION_STAMP_VERSION = 1

GENERATION_DIVERGENCE_EVENT = "identity_generation_divergence"
GENERATION_ACTOR_EVENT = "identity_generation_actor_filed"
GENERATION_ACTOR_IDEMPOTENCY_PREFIX = "identity-generation:"
GENERATION_ACTOR_TITLE = "[guard] dispatcher identity generation changed: {reasons}"

#: The witness symbols R1-R3 land. Recorded for evidence; only their LOSS alerts (an arrival is a
#: fix landing, not a regression).
WITNESS_SYMBOLS: tuple[str, ...] = (
    "_worker_liveness",
    "WORKER_ALIVE",
    "WORKER_DEAD",
    "WORKER_UNKNOWN",
    "host_boot_witness",
)


def dispatching_tree() -> Path:
    """The tree THIS guard is executing from (never a hardcoded path)."""
    return Path(__file__).resolve().parent.parent


def witness_symbols_present(module: Any) -> list[str]:
    """Which of :data:`WITNESS_SYMBOLS` the module the dispatcher executes actually carries."""
    return sorted(name for name in WITNESS_SYMBOLS if hasattr(module, name))


def stamp_path(path: Any = None) -> Path:
    """The generation stamp's path, or the explicit one a caller or a test supplies."""
    if path is not None:
        return Path(path)
    from hermes_cli import kanban_db as kb

    return kb.kanban_home() / "kanban" / GENERATION_STAMP_FILENAME


def _psutil_available() -> bool:
    """Is ``psutil`` importable in THIS interpreter? Measured, never assumed."""
    try:
        import importlib.util

        return importlib.util.find_spec("psutil") is not None
    except Exception:
        return False


def _interpreter_tag(version: Optional[str] = None) -> str:
    """``major.minor`` -- the part of an interpreter change that can move the identity shape.

    Patch releases are recorded but not compared: a routine patch bump must not page a lane, while a
    3.11 -> 3.14 move (psutil present -> absent, ``'|<epoch-cs>'`` -> ``'unverified'``) must.
    """
    raw = version or platform.python_version()
    parts = str(raw).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(raw)


def generation_stamp(
    *,
    sample: Any,
    tree: Any = None,
    interpreter: Optional[str] = None,
    psutil: Optional[bool] = None,
    witness_symbols: Optional[Iterable[str]] = None,
    now: Optional[int] = None,
) -> dict:
    """The deployed-generation record: the shape this process produces, and its substrate."""
    return {
        "version": GENERATION_STAMP_VERSION,
        "shape": identity_shape(sample),
        "sample": sample if isinstance(sample, (str, int)) else repr(sample),
        "interpreter": _interpreter_tag(interpreter),
        "interpreter_full": interpreter or platform.python_version(),
        "psutil": bool(_psutil_available() if psutil is None else psutil),
        "witness_symbols": sorted(witness_symbols or []),
        "tree": str(tree) if tree is not None else str(dispatching_tree()),
        "pid": os.getpid(),
        "recorded_at": int(time.time() if now is None else now),
    }


def compare_generation(recorded: Optional[dict], current: dict) -> list[str]:
    """Divergence reasons, in a deterministic order. Empty means the generation is unchanged.

    Direction matters: an interpreter MINOR change, a shape change, a lost psutil, and a LOST witness
    symbol all alert. A GAINED witness symbol does not -- that is the R1-R3 fix landing.
    """
    if not recorded:
        return []
    reasons: list[str] = []
    if recorded.get("shape") != current.get("shape"):
        reasons.append(f"identity-shape-changed:{recorded.get('shape')}->{current.get('shape')}")
    if recorded.get("interpreter") != current.get("interpreter"):
        reasons.append(
            f"interpreter-changed:{recorded.get('interpreter')}->{current.get('interpreter')}"
        )
    if recorded.get("psutil") and not current.get("psutil"):
        reasons.append("psutil-lost")
    lost = sorted(set(recorded.get("witness_symbols") or []) - set(current.get("witness_symbols") or []))
    if lost:
        reasons.append("witness-bytes-lost:" + ",".join(lost))
    return reasons


def read_generation_stamp(path: Any = None) -> Optional[dict]:
    """The recorded generation, or ``None`` when absent, empty or unreadable (treated as absent)."""
    try:
        raw = stamp_path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def write_generation_stamp(gen: dict, path: Any = None) -> Optional[Path]:
    """Record a generation. Best-effort: an unwritable kanban home must not stop a tick."""
    target = stamp_path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(gen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return target
    except (OSError, ValueError) as exc:
        logger.warning("kanban crash-sweep guard: could not record the identity generation: %s", exc)
        return None


@dataclass(frozen=True)
class GenerationTick:
    """One tick's answer to "is the deployed generation the one we recorded?"."""

    stamp: dict
    reasons: list[str] = field(default_factory=list)
    recorded: bool = False
    reference: Optional[dict] = None

    @property
    def alerting(self) -> bool:
        return bool(self.reasons)

    @property
    def signature(self) -> str:
        return "|".join(self.reasons)


def check_deployed_generation(
    *,
    sample: Any,
    tree: Any = None,
    interpreter: Optional[str] = None,
    psutil: Optional[bool] = None,
    witness_symbols: Optional[Iterable[str]] = None,
    path: Any = None,
    now: Optional[int] = None,
) -> GenerationTick:
    """Record the generation on the first tick; on every later tick compare and report divergence.

    A clean tick REFRESHES the reference, so the record tracks the latest known-good generation and a
    loss is measured against it. A diverging tick LEAVES the reference alone, so the divergence keeps
    being reported every tick until it is fixed -- detected in ONE tick, not discovered hours later by
    a false crash sweep.
    """
    current = generation_stamp(
        sample=sample, tree=tree, interpreter=interpreter, psutil=psutil,
        witness_symbols=witness_symbols, now=now,
    )
    recorded = read_generation_stamp(path)
    if recorded is None:
        write_generation_stamp(current, path)
        return GenerationTick(stamp=current, reasons=[], recorded=True, reference=None)
    reasons = compare_generation(recorded, current)
    if reasons:
        return GenerationTick(stamp=current, reasons=reasons, recorded=False, reference=recorded)
    write_generation_stamp(current, path)
    return GenerationTick(stamp=current, reasons=[], recorded=False, reference=recorded)


def divergence_body(tick: GenerationTick, *, board: Optional[str] = None, tree: Any = None) -> str:
    """The acting card's body for a generation divergence: both generations, side by side."""
    recorded = tick.reference or {}
    current = tick.stamp
    resolved_tree = tree if tree is not None else dispatching_tree()
    lines = [
        "## Observed",
        "",
        "```",
        "identity-generation divergence: " + ", ".join(tick.reasons),
        "```",
        "",
        "Filed by the dispatcher's deployed-identity generation check (R6 of ruling `t_63a20c59`) --",
        "deterministic, no model was consulted. The dispatcher WRITES one identity shape and READS",
        "another, so a liveness comparison across them is meaningless; the reclaim sweep is where that",
        "error becomes a fleet-wide outage.",
        "",
        f"- board: `{board or '(current)'}`",
        f"- tree: `{resolved_tree}`",
        "",
        "| field | recorded | current |",
        "| --- | --- | --- |",
    ]
    for key in ("shape", "interpreter", "interpreter_full", "psutil", "witness_symbols", "sample", "tree"):
        lines.append(f"| {key} | `{recorded.get(key)}` | `{current.get(key)}` |")
    lines += [
        "",
        f"- recorded_at: {recorded.get('recorded_at')}",
        f"- observed_at: {current.get('recorded_at')}",
        "",
        "## Operator move",
        "",
        "Land the witness bytes (R1-R3) so the write and the read shape agree, or re-record the",
        "reference deliberately once the new generation is the intended one: remove",
        f"`{stamp_path()}` and let the next tick record it. Deleting the stamp is the",
        "acknowledgement; it does not touch board data.",
    ]
    return "\n".join(lines)


def record_divergence_evidence(conn: Any, tick: GenerationTick) -> None:
    """Append the divergence event to every card this process is running. Never mutates status."""
    from hermes_cli import kanban_db as kb

    payload = {
        "signature": tick.signature,
        "reasons": list(tick.reasons),
        "recorded": {
            k: (tick.reference or {}).get(k)
            for k in ("shape", "interpreter", "psutil", "witness_symbols")
        },
        "current": {
            k: tick.stamp.get(k) for k in ("shape", "interpreter", "psutil", "witness_symbols")
        },
    }
    for task_id in _cards_owned_by_this_process(conn):
        try:
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, GENERATION_DIVERGENCE_EVENT, dict(payload))
        except Exception as exc:
            logger.warning(
                "kanban crash-sweep guard: could not append generation evidence to %s: %s",
                task_id, exc,
            )


def _cards_owned_by_this_process(conn: Any) -> list[str]:
    """Task ids whose recorded worker PID is THIS process -- the cards running on these bytes."""
    try:
        rows = conn.execute(
            "SELECT id FROM tasks WHERE status = 'running' AND worker_pid = ?", (os.getpid(),),
        ).fetchall()
    except Exception:
        return []
    return [row["id"] for row in rows]


def alert_generation_divergence(
    conn: Any,
    tick: GenerationTick,
    *,
    board: Optional[str] = None,
    tree: Any = None,
    now: Optional[int] = None,
) -> Optional[str]:
    """The whole loud path for a divergence: log, per-card evidence, one acting card."""
    if not tick.alerting:
        return None
    signature = hashlib.sha256(tick.signature.encode("utf-8")).hexdigest()[:12]
    logger.error(
        "kanban crash-sweep guard: DEPLOYED IDENTITY GENERATION CHANGED (%s) -- recorded %s, "
        "current %s. A liveness comparison across the two shapes is meaningless; hold the reclaim.",
        tick.signature, tick.reference, tick.stamp,
    )
    record_divergence_evidence(conn, tick)
    return _file_actor(
        conn,
        prefix=GENERATION_ACTOR_IDEMPOTENCY_PREFIX,
        signature=signature,
        title=GENERATION_ACTOR_TITLE.format(reasons=", ".join(tick.reasons)),
        body=divergence_body(tick, board=board, tree=tree),
        event=GENERATION_ACTOR_EVENT,
        event_payload={
            "reasons": list(tick.reasons),
            "recorded": tick.reference,
            "current": tick.stamp,
        },
        board=board,
        now=now,
    )
