"""Deterministic execution-budget preflight for Kanban tasks.

The analyzer is intentionally lexical and side-effect free: it never calls an
LLM, reads the filesystem, or mutates the board.  It is a conservative routing
signal, not a quality verdict.  ``split`` means the card spans enough lifecycle
families that dispatching it as one worker run is likely to waste iterations;
it does *not* mean any requested operation is forbidden.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable


CAUTION_THRESHOLD = 3
SPLIT_THRESHOLD = 6
_REVIEW_KEY_RE = re.compile(
    r"^(?P<postimage>[0-9a-f]{64}):(?P<claims>[0-9a-f]{64})"
    r"(?::round-(?P<round>[1-9][0-9]*))?$"
)
_FILE_REF_RE = re.compile(
    r"(?<![\w.-])(?:[\w.-]+/)*[\w.-]+\."
    r"(?:py|pyi|js|jsx|ts|tsx|go|rs|java|kt|swift|rb|php|sh|bash|zsh|"
    r"json|ya?ml|toml|md|sql|html|css|scss|xml|plist)\b",
    re.IGNORECASE,
)
_ACCEPTANCE_RE = re.compile(
    r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+|\[[ xX]\]\s+)",
)


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _signal_text(title: str, body: str) -> str:
    text = _plain(f"{title}\n{body}")
    # A safety constraint such as "without activating" must not make a
    # report-only card look like it also contains a deployment phase.
    return re.sub(
        r"\b(?:without|do not|dont|never|sin|no)\s+"
        r"(?:\w+\s+){0,3}(?:activat\w*|deploy\w*|publish\w*|"
        r"roll\s+out|make\s+(?:\w+\s+){0,2}live|go\s+live|"
        r"make\s+(?:\w+\s+){0,2}available|send\w*|submit\w*|"
        r"despl(?:eg|ieg)\w*|"
        r"public(?:ar\w*|acion\w*|ad[oa]s?|and[oa]\w*|[oa]s?|an)|"
        r"publiqu\w*|lanz\w*|"
        r"(?:hacer(?:la|lo|las|los)?|hace\w*|haz(?:la|lo|las|los)?|dejar)\s+"
        r"(?:\w+\s+){0,2}disponible|"
        r"enviar\w*|presentar\w*)",
        "",
        text,
    )


_ACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "discovery",
        re.compile(
            r"\b(?:investig\w*|inspect\w*|diagnos\w*|analys\w*|analyz\w*|"
            r"disen\w*|design\w*|architect\w*|planif\w*|plan\w*)\b"
        ),
    ),
    (
        "implementation",
        re.compile(
            r"\b(?:implement\w*|fix\w*|corrig\w*|correct\w*|patch\w*|"
            r"modif\w*|creat\w*|regener\w*|harden\w*|remedi\w*|escrib\w*)\b"
        ),
    ),
    (
        "verification",
        re.compile(
            r"\b(?:test\w*|prueb\w*|verif\w*|valid\w*|pytest|suite\w*|"
            r"matrix|matriz|dual[- ]?runtime|compil\w*|lint\w*|ruff)\b"
        ),
    ),
    (
        "evidence",
        re.compile(
            r"\b(?:receipt\w*|recibo\w*|hash\w*|freeze\w*|congel\w*|"
            r"manifest\w*|artifact\w*|artefact\w*|report\w*|informe\w*|"
            r"binding\w*|evidenc\w*|checkpoint\w*)\b"
        ),
    ),
    (
        "review",
        re.compile(
            r"\b(?:review\w*|revision\w*|rereview\w*|adversarial\w*|"
            r"audit\w*|independent\w*|independiente\w*)\b"
        ),
    ),
    (
        "controller",
        re.compile(
            r"\b(?:controller\w*|controlador\w*|approv\w*|aprob\w*|"
            r"verdict\w*|gate\w*|go/no-go)\b"
        ),
    ),
    (
        "activation",
        re.compile(
            r"\b(?:deploy\w*|despl(?:eg|ieg)\w*|activat\w*|activar\w*|publish\w*|"
            r"public(?:ar\w*|acion\w*|ad[oa]s?|and[oa]\w*|[oa]s?|an)|"
            r"publiqu\w*|roll\s+out|rollout\w*|"
            r"make\s+(?:\w+\s+){0,2}live|go\s+live|"
            r"make\s+(?:\w+\s+){0,2}available|"
            r"(?:hacer(?:la|lo|las|los)?|hace\w*|haz(?:la|lo|las|los)?|dejar)\s+"
            r"(?:\w+\s+){0,2}disponible|"
            r"releas\w*\s+(?:(?:version|build|package|release)\b|"
            r"(?:\w+\s+){0,4}(?:to|into)\s+production)|"
            r"ship\w*\s+(?:(?:version|build|package|release)\b|"
            r"(?:\w+\s+){0,4}(?:to\s+production|live))|"
            r"promot\w*\s+(?:\w+\s+){0,4}(?:to\s+production|live)|"
            r"send\w*|submit\w*|enviar\w*|presentar\w*|"
            r"lanz\w*\s+(?:\w+\s+){0,3}(?:version|produccion|usuarios)|"
            r"liberar\w*\s+(?:\w+\s+){0,4}(?:version|produccion)|"
            r"promover\w*\s+(?:\w+\s+){0,4}(?:a\s+produccion|entorno\s+vivo)|"
            r"poner\s+(?:\w+\s+){0,3}en\s+produccion|"
            r"pasar\s+(?:\w+\s+){0,3}a\s+produccion)\b"
        ),
    ),
)


_SENSITIVE_FAMILY_KINDS: dict[str, frozenset[str]] = {
    "review": frozenset({"review", "review_scheduler", "controller"}),
    "controller": frozenset({"review", "review_scheduler", "controller", "activation"}),
    "activation": frozenset({"activation"}),
}


def incompatible_decomposition_family(
    kind: str,
    action_families: Iterable[str],
) -> str | None:
    """Return the first sensitive family incompatible with ``kind``.

    Decomposition kinds are an authorization boundary, not descriptive labels.
    Lexically detected review/controller/activation work must therefore never be
    materialized as generic ``work`` merely because an LLM supplied that label.
    ``activation`` is additionally explicit: callers may choose it even when
    conservative lexical detection finds no activation phrase, and the DB will
    still hold the task behind human approval.
    """
    normalized_kind = str(kind or "work").strip().lower()
    present = set(action_families)
    for family in ("activation", "controller", "review"):
        allowed_kinds = _SENSITIVE_FAMILY_KINDS[family]
        if family in present and normalized_kind not in allowed_kinds:
            return family
    return None


@dataclass(frozen=True)
class TaskBudgetAssessment:
    """Stable, JSON-friendly result of a task scope preflight."""

    verdict: str
    score: int
    caution_threshold: int
    split_threshold: int
    estimated_context_tokens: int
    estimated_turns: int
    max_turns: int
    body_chars: int
    acceptance_items: int
    file_references: int
    action_families: tuple[str, ...]
    reasons: tuple[str, ...]
    suggested_shards: tuple[str, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        # asdict preserves tuples; lists are friendlier and stable at JSON/API
        # boundaries, and avoid consumers relying on Python-only tuple semantics.
        for key in ("action_families", "reasons", "suggested_shards"):
            value[key] = list(value[key])
        return value


def _ordered_families(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in _ACTION_PATTERNS if pattern.search(text))


def _suggest_shards(families: Iterable[str]) -> tuple[str, ...]:
    present = set(families)
    shards: list[str] = []
    if "discovery" in present:
        shards.append("discovery")
    if "implementation" in present:
        shards.append("implementation")
    if {"verification", "evidence"} & present:
        shards.append("verification_and_evidence")
    if "review" in present:
        shards.append("independent_review")
    if "controller" in present:
        shards.append("controller")
    if "activation" in present:
        shards.append("human_activation_gate")
    return tuple(shards)


def assess_task(
    title: str, body: str | None, *, max_turns: int = 60
) -> TaskBudgetAssessment:
    """Assess semantic task breadth without spending model tokens.

    The score deliberately keys on *lifecycle diversity* more heavily than raw
    text length.  A concise card can still be oversized when it asks one worker
    to discover, implement, run a full matrix, write receipts, review itself,
    and make a controller decision.
    """

    safe_title = str(title or "")
    safe_body = str(body or "")
    signals = _signal_text(safe_title, safe_body)
    families = _ordered_families(signals)
    family_set = set(families)
    acceptance_items = len(_ACCEPTANCE_RE.findall(safe_body))
    file_references = len({
        m.group(0).lower() for m in _FILE_REF_RE.finditer(safe_body)
    })
    body_chars = len(safe_body)
    arrow_count = safe_body.count("→") + safe_body.count("->")

    score = 0
    reasons: list[str] = []

    if body_chars > 6000:
        score += 2
        reasons.append("large_body")
    elif body_chars > 3000:
        score += 1
        reasons.append("medium_body")

    if acceptance_items > 10:
        score += 3
        reasons.append("many_acceptance_items")
    elif acceptance_items > 6:
        score += 1
        reasons.append("several_acceptance_items")

    if file_references > 8:
        score += 2
        reasons.append("many_file_surfaces")
    elif file_references > 4:
        score += 1
        reasons.append("several_file_surfaces")

    family_count = len(families)
    if family_count >= 6:
        score += 5
        reasons.append("many_lifecycle_families")
    elif family_count == 5:
        score += 4
        reasons.append("many_lifecycle_families")
    elif family_count == 4:
        score += 3
        reasons.append("multiple_lifecycle_families")
    elif family_count == 3:
        score += 1
        reasons.append("multiple_lifecycle_families")

    # Four lifecycle owners already imply at least one handoff boundary. Keep
    # this structural rule independent of prose length and available budget:
    # more turns do not make self-review or controller coupling atomic.
    if family_count >= 4:
        score = max(score, SPLIT_THRESHOLD)
        reasons.append("four_or_more_lifecycle_families")

    if {"implementation", "verification", "review"}.issubset(family_set):
        score += 2
        reasons.append("compound_lifecycle")
    if {"review", "controller"}.issubset(family_set):
        score += 1
        reasons.append("review_controller_coupled")
    if {"implementation", "activation"}.issubset(family_set):
        score += 1
        reasons.append("implementation_activation_coupled")

    if arrow_count >= 4:
        score += 2
        reasons.append("long_explicit_pipeline")
    elif arrow_count >= 2:
        score += 1
        reasons.append("explicit_pipeline")

    full_matrix = bool(
        re.search(
            r"\b(?:full (?:project )?suite|suite completa|matriz completa|full matrix)\b",
            signals,
        )
    )
    if full_matrix and "implementation" in family_set:
        score += 1
        reasons.append("implementation_plus_full_matrix")

    normalized_max_turns = max(1, int(max_turns))
    # This is a planning estimate, not a hard cap.  It intentionally errs on
    # the high side for lifecycle handoffs and acceptance-heavy cards.
    estimated_turns = (
        4
        + (4 * family_count)
        + math.ceil(acceptance_items / 2)
        + min(file_references, 10)
        + (4 if full_matrix else 0)
    )
    if estimated_turns > normalized_max_turns:
        reasons.append("estimated_turns_exceed_budget")
        if estimated_turns > normalized_max_turns * 1.5:
            score = max(score, SPLIT_THRESHOLD)
        else:
            score = max(score, CAUTION_THRESHOLD)

    if score >= SPLIT_THRESHOLD:
        verdict = "split"
    elif score >= CAUTION_THRESHOLD:
        verdict = "caution"
    else:
        verdict = "ok"

    estimated_context_tokens = math.ceil((len(safe_title) + body_chars) / 4)

    return TaskBudgetAssessment(
        verdict=verdict,
        score=score,
        caution_threshold=CAUTION_THRESHOLD,
        split_threshold=SPLIT_THRESHOLD,
        estimated_context_tokens=estimated_context_tokens,
        estimated_turns=estimated_turns,
        max_turns=normalized_max_turns,
        body_chars=body_chars,
        acceptance_items=acceptance_items,
        file_references=file_references,
        action_families=families,
        reasons=tuple(reasons),
        suggested_shards=_suggest_shards(families),
    )


def canonical_review_idempotency_key(review_key: str) -> str:
    """Map an exact-postimage review key onto Kanban's existing dedup key.

    A caller should normally use ``<postimage-sha256>:<claims-sha256>`` or a
    canonical composite digest, with an optional ``:round-N`` suffix. Review
    rounds stay possible, while accidental duplicate reviewers for the exact
    same immutable postimage and claim contract collapse to one non-archived task.
    """

    value = str(review_key or "").strip().lower()
    if not _REVIEW_KEY_RE.fullmatch(value):
        raise ValueError(
            "review_key must bind postimage and claims as "
            "<postimage-sha256>:<claims-sha256>[:round-N]"
        )
    return f"review:v1:{value}"


__all__ = [
    "CAUTION_THRESHOLD",
    "SPLIT_THRESHOLD",
    "TaskBudgetAssessment",
    "assess_task",
    "canonical_review_idempotency_key",
]
