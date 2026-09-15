"""Deterministic, dispatch-time gate precheck for kanban tasks.

Wave2/2c: re-implements ONLY the mechanically-checkable subset of the
``atomic-task-gate`` and ``ears-sensibility-gate`` skills' discriminants as a
pure, importable function — no LLM calls, no ``decision_hud`` calls, no file
I/O, no skill-file inspection at runtime (per this repo's AGENTS.md "never
read source code in tests" rule: the judgment lives in the skills' prose,
this module only checks the SHAPE of evidence a task record already carries).

Every genuinely judgment-laden discriminant from either skill (oracle
correctness, proportionality, actor/ownership, "is this really one oracle",
conflict-with-prior-requirement, ...) is explicitly OUT of scope here and
stays a prose gate routed through ``decision_hud`` exactly as the skills
already specify. This module never re-derives that judgment — it only
verifies that the routing already happened (a field is present and
plausibly shaped).

Enforcement point: ``hermes_cli/kanban_db_dispatch.py``'s per-task dispatch
guard (``_dispatch_lane_task``), gated by the config flag
``kanban.gate_precheck_enabled`` in ``config.yaml`` (default ``False`` —
opt-in; this is a new, unproven heuristic, so an operator must explicitly
turn it on).
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------

STATUS_PASS = "pass"
STATUS_FAST_LANE = "fast_lane"
STATUS_NOT_GATED = "not_gated"
STATUS_BLOCKED = "blocked"

VALID_STATUSES = (STATUS_PASS, STATUS_FAST_LANE, STATUS_NOT_GATED, STATUS_BLOCKED)

# Statuses under which dispatch may proceed. ``blocked`` is the only status
# that stops a spawn; everything else — including "not gated yet" — is a
# pass-through, per the owner's dispatch-time-only enforcement decision:
# this precheck validates shape WHEN evidence is present, it never forces
# Rule 6 (EARS) or Rule 2 (atomic-task-gate) population as a hard gate.
DISPATCHABLE_STATUSES = (STATUS_PASS, STATUS_FAST_LANE, STATUS_NOT_GATED)


class TaskLike(Protocol):
    """Minimal shape ``gate_precheck`` needs. ``kanban_db.Task`` satisfies
    this structurally; tests may pass any object/namespace with these
    attributes without importing the DB layer."""

    title: str
    body: Optional[str]
    ears_sentence: Optional[str]
    scope_paths: Optional[list]


@dataclass
class GatePrecheckResult:
    """Outcome of :func:`gate_precheck`. Plain dict-like access via
    :meth:`to_dict` so dispatch code (and tests) can treat it as
    ``{"status": ..., "reason": ...}`` without importing this dataclass."""

    status: str
    reason: str
    # Diagnostic detail, never load-bearing for the pass/block decision —
    # e.g. which single check failed, useful in board UI / logs.
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "detail": dict(self.detail)}


# ---------------------------------------------------------------------------
# (1) Fast lane: constraint-tier short-circuit for trivial doc/config edits.
#
# Per critique_06 (gate proliferation): a one-line README typo forced
# through all four gates is the reproducing failure case this exists to
# avoid. A task qualifies for the fast lane ONLY when ALL of:
#   - scope_paths has EXACTLY one entry (single-file, unambiguous blast
#     radius — the same "single_file" discriminant atomic-task-gate's Q4
#     asks, just checked mechanically instead of by prose judgment)
#   - that one path matches a doc/config/metadata glob (below)
#   - the title/body contain none of the behavior-change keywords (below)
# ---------------------------------------------------------------------------

# Deliberately narrow: these globs name file KINDS that are near-certainly
# non-executable prose/config, never source code that runs. Extensions were
# chosen to cover the common "trivial edit" cases named in critique_06
# (README typo, changelog entry, a YAML/JSON config tweak, a license/notice
# file) while excluding anything that could plausibly carry behavior
# (.py/.ts/.js/.sh/.sql/... are never in this list, on purpose — even a
# one-line code change can alter behavior in a way prose/config cannot).
FAST_LANE_GLOBS: tuple[str, ...] = (
    "*.md",
    "*.mdx",
    "*.txt",
    "*.rst",
    "*.json",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.ini",
    "*.cfg",
    "LICENSE",
    "LICENSE.*",
    "NOTICE",
    "CHANGELOG*",
)

# Keywords whose presence in the title/body signal the edit is NOT a pure
# doc/config tweak even though the touched file's extension matches the
# fast-lane glob (e.g. a ".json" file that is actually a fixture asserted on
# in a test, or a "config.yaml" change that flips a feature's runtime
# behavior). Chosen to be broad rather than clever: false negatives here
# (missing a real behavior-change word) fail safe because the task still
# falls through to the full gate check below, not to an unchecked bypass;
# the fast lane never blocks anything, it only skips extra scrutiny.
BEHAVIOR_CHANGE_KEYWORDS: tuple[str, ...] = (
    "fix",
    "fixes",
    "bug",
    "behavior",
    "behaviour",
    "logic",
    "algorithm",
    "feature",
    "refactor",
    "implement",
    "regression",
    "race condition",
    "deadlock",
    "crash",
    "security",
    "vulnerability",
    "performance",
    "endpoint",
    "api change",
    "schema",
    "migration",
    "breaking change",
)


def _matches_fast_lane_glob(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, pattern) for pattern in FAST_LANE_GLOBS)


def _contains_behavior_change_keyword(text: str) -> Optional[str]:
    lowered = text.lower()
    for kw in BEHAVIOR_CHANGE_KEYWORDS:
        if kw in lowered:
            return kw
    return None


def _is_fast_lane(task: TaskLike) -> tuple[bool, str]:
    scope_paths = getattr(task, "scope_paths", None)
    if not scope_paths or len(scope_paths) != 1:
        return False, "fast_lane requires exactly one scope_paths entry"
    (only_path,) = scope_paths
    if not _matches_fast_lane_glob(str(only_path)):
        return False, f"{only_path!r} does not match a doc/config fast-lane glob"
    combined_text = " ".join(
        str(part) for part in (getattr(task, "title", "") or "", getattr(task, "body", "") or "") if part
    )
    hit = _contains_behavior_change_keyword(combined_text)
    if hit is not None:
        return False, f"title/body contains behavior-change keyword {hit!r}"
    return True, f"single doc/config file {only_path!r}, no behavior-change language"


# ---------------------------------------------------------------------------
# (2)/(3) EARS-shape heuristic.
#
# The five canonical EARS templates (ears-sensibility-gate's Rule 6):
#   Ubiquitous : "The <system> shall <response>."
#   Event      : "When <trigger>, the <system> shall <response>."
#   State      : "While <state>, the <system> shall <response>."
#   Conditional: "If <condition>, then the <system> shall <response>."
#   Optional   : "Where <feature>, the <system> shall <response>."
#
# This is a shape/keyword heuristic, not NLP judgment (per the plan's own
# risk note: it WILL false-positive on a form-fitting but semantically
# empty sentence, and false-negative on an oddly-phrased legitimate one).
# Per the gates' own verification rule ("never claim resolved while owner
# judgment remains open") this fails CLOSED: anything that doesn't match
# one of the five shapes is BLOCKED, not silently passed.
# ---------------------------------------------------------------------------

_EARS_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("when", re.compile(r"^when\b.+,\s*.+\bshall\b.+", re.IGNORECASE)),
    ("while", re.compile(r"^while\b.+,\s*.+\bshall\b.+", re.IGNORECASE)),
    ("if_then", re.compile(r"^if\b.+,\s*then\b.+\bshall\b.+", re.IGNORECASE)),
    ("where", re.compile(r"^where\b.+,\s*.+\bshall\b.+", re.IGNORECASE)),
    # Ubiquitous is checked last: it is the least constrained shape (no
    # leading trigger clause), so trigger-bearing sentences must be ruled
    # out by the more specific patterns above first to avoid mis-labeling
    # a malformed "when"/"if" sentence as ubiquitous just because it also
    # happens to contain "shall".
    ("ubiquitous", re.compile(r"^the\b.+\bshall\b.+", re.IGNORECASE)),
)

# Sequencing conjunctions that indicate the sentence bundles more than one
# requirement — a single EARS sentence is supposed to state ONE trigger/
# response pair (atomic-task-gate's Q1: "single EARS sentence" shape).
_SEQUENCING_MARKERS: tuple[str, ...] = ("and then", ";", " then also ")


def _ears_shape(sentence: str) -> tuple[bool, str]:
    stripped = sentence.strip()
    if not stripped:
        return False, "ears_sentence is empty/whitespace"
    lowered = stripped.lower()
    for marker in _SEQUENCING_MARKERS:
        if marker in lowered:
            return False, f"contains sequencing conjunction {marker!r} (not a single requirement)"
    if "shall" not in lowered:
        return False, "missing the mandatory 'shall' response keyword"
    for name, pattern in _EARS_PATTERNS:
        if pattern.match(stripped):
            return True, f"matches EARS '{name}' template"
    return False, "does not match any of the 5 EARS templates (Ubiquitous/When/While/If-then/Where)"


# ---------------------------------------------------------------------------
# (4) Scope-size threshold.
#
# Chosen threshold: > 5 files. atomic-task-gate's Q4 asks "is this a single
# atomic change to a single conceptual unit"; a task whose own declared
# scope already spans more than five files is treated as evidence the task
# should have been decomposed (matches the plan's own "e.g. >5 files"
# example). This is a blunt proxy — a 6-file mechanical rename is not
# inherently non-atomic — but per the gates' fail-closed philosophy an
# overshoot here costs a manual decompose/override, not a silent miss.
# ---------------------------------------------------------------------------

SCOPE_SIZE_THRESHOLD = 5


def gate_precheck(task: TaskLike) -> dict[str, Any]:
    """Pure function: evaluate one task against the deterministic subset of
    the atomic-task-gate / ears-sensibility-gate discriminants.

    Returns a plain dict ``{"status": ..., "reason": ..., "detail": {...}}``
    with ``status`` one of :data:`VALID_STATUSES`. Never raises for a
    well-shaped ``task`` object; never touches the filesystem, the DB, or
    any skill file. No side effects.
    """
    fast_lane, fast_lane_reason = _is_fast_lane(task)
    if fast_lane:
        return GatePrecheckResult(
            status=STATUS_FAST_LANE, reason=fast_lane_reason,
        ).to_dict()

    ears_sentence = getattr(task, "ears_sentence", None)
    if ears_sentence is None:
        return GatePrecheckResult(
            status=STATUS_NOT_GATED,
            reason="ears_sentence not yet populated — this task has not "
                   "gone through the EARS gate; precheck does not force it",
        ).to_dict()

    shaped, shape_reason = _ears_shape(str(ears_sentence))
    if not shaped:
        return GatePrecheckResult(
            status=STATUS_BLOCKED,
            reason=f"ears_sentence does not look EARS-shaped: {shape_reason}",
            detail={"ears_sentence": ears_sentence},
        ).to_dict()

    scope_paths = getattr(task, "scope_paths", None)
    if scope_paths is not None and len(scope_paths) > SCOPE_SIZE_THRESHOLD:
        return GatePrecheckResult(
            status=STATUS_BLOCKED,
            reason=f"scope_paths has {len(scope_paths)} entries (> "
                   f"{SCOPE_SIZE_THRESHOLD}) — likely decomposable into "
                   "smaller tasks",
            detail={"scope_paths_count": len(scope_paths)},
        ).to_dict()

    return GatePrecheckResult(status=STATUS_PASS, reason="EARS-shaped, scope within threshold").to_dict()
