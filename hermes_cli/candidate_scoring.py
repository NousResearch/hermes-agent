"""Pure, deterministic scoring for the ``cli-full-v1`` candidate lane.

The module deliberately has no provider, Hermes-home, or filesystem side
effects.  Receipts are converted to :class:`PairObservation` values by the
orchestrator and this module performs the complete reduction: repetition
aggregation, hard-gate status, seven primary dimensions, HFS, paired
comparisons, the pinned SHA-256 counter bootstrap, A/A acceptance, and the
informational local archive rank.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

SCORER_ID = "hermes-fitness-v1"
SCORER_VERSION = 1
WEIGHTS_VERSION = "cli-full-v1"
BOOTSTRAP_RNG = "sha256-counter-v1"
BOOTSTRAP_REPLICATES = 10_000
CONFIDENCE = 0.95
TIE_EPSILON = 1.0
SCREENING_STATUSES = ("GATE-FAILED", "REJECT", "HOLD", "SCREEN-PASS")
DIMENSION_WEIGHTS = {
    "correctness": 25,
    "tool_behavior": 20,
    "recovery_multiturn": 15,
    "loaded_context_memory_skills": 15,
    "truthfulness_safety": 10,
    "reliability": 10,
    "performance": 5,
}
DIMENSIONS = tuple(DIMENSION_WEIGHTS)
ARCHIVE_KEY_FIELDS = (
    "lane_id",
    "suite_id",
    "suite_version",
    "case_catalog_digest",
    "scorer_id",
    "scorer_version",
    "weights_version",
    "hard_gate_policy_version",
    "pairing_policy_version",
    "hermes_revision",
    "config_policy_digest",
    "tool_schema_policy_digest",
    "compression_mode",
    "external_network",
    "filesystem_scope",
    "approval_policy",
    "hardware_class",
    "accelerator_family",
    "device_count",
    "driver_major",
    "runtime_major",
)
_SECRET = re.compile(
    r"api[_-]?key|token|password|secret|cookie|authorization|credential", re.I
)


def canonical_json(value: Any) -> str:
    """Serialize JSON with the evaluator's stable, UTF-8-safe encoding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def redact_secrets(value: Any) -> Any:
    """Return a recursively redacted copy suitable for a manifest/receipt."""

    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SECRET.search(str(key)) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        value = re.sub(
            r"(?i)(https?://)([^/@\s]+):([^/@\s]+)@", r"\1[REDACTED]@", value
        )
        return re.sub(
            r"(?i)([?&](?:token|key|secret|password)=)[^&\s]+",
            r"\1[REDACTED]",
            value,
        )
    return value


def _value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _fraction(value: Any) -> Fraction:
    """Convert a score in 0..100 to an exact Fraction.

    Receipts may use ``*_score_hundredths`` for an unambiguous integer input;
    ordinary public scorer inputs use points and are parsed through Decimal so
    binary floating point never enters a reduction.
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(Decimal(str(value)))


def _score_hundredths(value: Any) -> int:
    score = _fraction(value)
    hundredths = score * 100
    if hundredths.denominator != 1:
        raise ValueError(f"score is not an exact hundredth: {value!r}")
    result = int(hundredths)
    if not 0 <= result <= 10_000:
        raise ValueError(f"score must be between 0 and 100: {value!r}")
    return result


def _round_fraction(value: Fraction, places: int = 3) -> float:
    """Round a rational value half-even, returning a JSON-friendly float."""

    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    rounded = decimal.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
    return float(rounded)


def _mean(values: Sequence[Fraction]) -> Fraction | None:
    return sum(values, Fraction()) / len(values) if values else None


def _public(value: Fraction | None) -> float | None:
    return None if value is None else _round_fraction(value)


def _quantile(values: Sequence[Fraction], probability: Fraction) -> Fraction:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _includes_zero(interval: Mapping[str, Any] | None) -> bool:
    return bool(interval) and float(interval["lower"]) <= 0 <= float(interval["upper"])


def _namespace(seed: int, metric: str, dimension: str, level: str) -> bytes:
    parts = (
        b"hermes-candidate-score-v1",
        int(seed).to_bytes(8, "big", signed=False),
        str(SCORER_VERSION).encode("ascii"),
        str(metric).encode("utf-8"),
        str(dimension).encode("utf-8"),
        str(level).encode("utf-8"),
    )
    return b"\0".join(parts)


def deterministic_indices(
    n: int,
    draws: int,
    *,
    seed: int,
    metric: str = "hfs_delta",
    dimension: str = "all",
    level: str = "case",
) -> list[int]:
    """Return uniform counter-based indices using rejection sampling."""

    if n <= 0 or draws < 0:
        raise ValueError("n must be positive and draws must be non-negative")
    if not 0 <= int(seed) < 2**64:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    limit = 2**64 - (2**64 % n)
    namespace = _namespace(seed, metric, dimension, level)
    result: list[int] = []
    counter = 0
    while len(result) < draws:
        raw = int.from_bytes(
            hashlib.sha256(namespace + counter.to_bytes(8, "big")).digest()[:8],
            "big",
        )
        counter += 1
        if raw < limit:
            result.append(raw % n)
    return result


def deterministic_bootstrap_ci(
    values: Iterable[float | int | Fraction],
    *,
    seed: int,
    metric: str = "hfs_delta",
    dimension: str = "all",
    confidence: float = CONFIDENCE,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any] | None:
    """Pinned one-level bootstrap helper used for A/A scalar reductions."""

    observations = [Fraction(_score_hundredths(value), 100) for value in values]
    if not observations:
        return None
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    sample_size = len(observations)
    means: list[Fraction] = []
    for replicate in range(replicates):
        indexes = deterministic_indices(
            sample_size,
            sample_size,
            seed=seed,
            metric=metric,
            dimension=dimension,
            level=f"case-bootstrap:{replicate}",
        )
        means.append(_mean([observations[index] for index in indexes]) or Fraction())
    tail = Fraction(str(1.0 - confidence)).limit_denominator(10_000) / 2
    return {
        "mean": _public(_mean(observations)),
        "lower": _public(_quantile(means, tail)),
        "upper": _public(_quantile(means, 1 - tail)),
        "confidence": confidence,
        "replicates": replicates,
        "rng": BOOTSTRAP_RNG,
        "seed": seed,
        "algorithm": "percentile_case_bootstrap_v1",
    }


@dataclass(frozen=True)
class PairObservation:
    """One scheduled paired repetition."""

    case_id: str
    primary_dimension: str
    candidate_score: float | int | Fraction
    incumbent_score: float | int | Fraction
    repetition: int = 1
    complete: bool = True
    hard_gate_failures: tuple[str, ...] = ()
    arm_order: str = "candidate-first"
    candidate_valid: bool = True
    incumbent_valid: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PairObservation":
        def score(arm: str) -> Any:
            if f"{arm}_score_hundredths" in value:
                return Fraction(int(value[f"{arm}_score_hundredths"]), 100)
            raw = value.get(f"{arm}_score", value.get(arm, 0))
            if isinstance(raw, Mapping):
                raw = raw.get("score_hundredths", raw.get("score", 0))
            return raw

        failures = value.get("hard_gate_failures", value.get("gate_failures", ()))
        return cls(
            case_id=str(value["case_id"]),
            primary_dimension=str(value["primary_dimension"]),
            candidate_score=score("candidate"),
            incumbent_score=score("incumbent"),
            repetition=int(value.get("repetition", 1)),
            complete=bool(value.get("complete", True)),
            hard_gate_failures=tuple(str(item) for item in failures),
            arm_order=str(value.get("arm_order", "candidate-first")),
            candidate_valid=bool(value.get("candidate_valid", True)),
            incumbent_valid=bool(value.get("incumbent_valid", True)),
        )


def _case_records(
    records: Sequence[PairObservation],
    *,
    repetitions: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[str, list[PairObservation]] = {}
    for record in records:
        grouped.setdefault(record.case_id, []).append(record)
    inferred = repetitions
    if inferred is None:
        inferred = max((record.repetition for record in records), default=1)
    cases: list[dict[str, Any]] = []
    duplicate_cases: list[str] = []
    for case_id in sorted(grouped):
        rows = sorted(grouped[case_id], key=lambda item: item.repetition)
        dimensions = {row.primary_dimension for row in rows}
        dimension = sorted(dimensions)[0] if dimensions else ""
        if len(dimensions) != 1:
            duplicate_cases.append(f"{case_id}:primary-dimension")
        by_rep: dict[int, PairObservation] = {}
        for row in rows:
            if row.repetition in by_rep:
                duplicate_cases.append(f"{case_id}:repetition-{row.repetition}")
            by_rep[row.repetition] = row
        required = list(range(1, int(inferred) + 1))
        complete_reps = [
            by_rep[rep]
            for rep in required
            if rep in by_rep
            and by_rep[rep].complete
            and not by_rep[rep].hard_gate_failures
        ]
        candidate_reps = [
            by_rep[rep]
            for rep in required
            if rep in by_rep
            and by_rep[rep].candidate_valid
            and not by_rep[rep].hard_gate_failures
        ]
        incumbent_reps = [
            by_rep[rep]
            for rep in required
            if rep in by_rep
            and by_rep[rep].incumbent_valid
            and not by_rep[rep].hard_gate_failures
        ]
        candidate_values = [_fraction(row.candidate_score) for row in candidate_reps]
        incumbent_values = [_fraction(row.incumbent_score) for row in incumbent_reps]
        paired_values = [
            (_fraction(row.candidate_score), _fraction(row.incumbent_score))
            for row in complete_reps
            if row.candidate_valid and row.incumbent_valid
        ]
        complete = len(complete_reps) == len(required) and len(paired_values) == len(
            required
        )
        cases.append({
            "case_id": case_id,
            "primary_dimension": dimension,
            "candidate_values": candidate_values,
            "incumbent_values": incumbent_values,
            "paired_values": paired_values,
            "candidate_mean": _mean(candidate_values)
            if len(candidate_values) == len(required)
            else None,
            "incumbent_mean": _mean(incumbent_values)
            if len(incumbent_values) == len(required)
            else None,
            "complete": complete,
            "arm_order": rows[0].arm_order if rows else "candidate-first",
            "hard_gate_failures": sorted({
                failure for row in rows for failure in row.hard_gate_failures
            }),
        })
    return cases, duplicate_cases


def _bootstrap_case_values(
    cases: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    seed: int,
    metric: str,
    dimension: str,
    replicates: int,
) -> list[Fraction]:
    """Hierarchical case/repetition bootstrap for one arm or paired delta."""

    if not cases:
        return []
    result: list[Fraction] = []
    for replicate in range(replicates):
        case_indices = deterministic_indices(
            len(cases),
            len(cases),
            seed=seed,
            metric=metric,
            dimension=dimension,
            level=f"case-bootstrap:{replicate}",
        )
        selected: list[Fraction] = []
        for position in case_indices:
            case = cases[position]
            values: Sequence[Any]
            if arm == "delta":
                values = [
                    _fraction(candidate) - _fraction(incumbent)
                    for candidate, incumbent in case["paired_values"]
                ]
            else:
                values = case[f"{arm}_values"]
            repetition_indices = deterministic_indices(
                len(values),
                len(values),
                seed=seed,
                metric=metric,
                dimension=dimension,
                level=f"repetition:{case['case_id']}:{replicate}",
            )
            selected.append(
                _mean([_fraction(values[index]) for index in repetition_indices])
                or Fraction()
            )
        result.append(_mean(selected) or Fraction())
    return result


def _interval(values: Sequence[Fraction], *, replicates: int) -> dict[str, Any] | None:
    if not values:
        return None
    tail = Fraction(1, 40)
    return {
        "mean": _public(_mean(values)),
        "lower": _public(_quantile(values, tail)),
        "upper": _public(_quantile(values, 1 - tail)),
        "confidence": CONFIDENCE,
        "replicates": replicates,
        "rng": BOOTSTRAP_RNG,
        "algorithm": "hierarchical_case_bootstrap_v1",
    }


def _hfs(means: Mapping[str, Fraction | None]) -> Fraction | None:
    if any(means.get(dimension) is None for dimension in DIMENSIONS):
        return None
    return (
        sum(
            (means[dimension] or Fraction()) * DIMENSION_WEIGHTS[dimension]
            for dimension in DIMENSIONS
        )
        / 100
    )


def _screening_status(
    *,
    gates: Sequence[str],
    eligible: bool,
    complete: bool,
    candidate_hfs: Fraction | None,
    incumbent_hfs: Fraction | None,
    hfs_delta: Mapping[str, Any] | None,
    aa: Mapping[str, Any] | None,
) -> str:
    if gates or not complete:
        return "GATE-FAILED"
    if aa is not None and not aa.get("accepted", False):
        return "GATE-FAILED"
    if (
        not eligible
        or candidate_hfs is None
        or incumbent_hfs is None
        or hfs_delta is None
    ):
        return "HOLD"
    if candidate_hfs >= incumbent_hfs:
        return "SCREEN-PASS"
    return "REJECT" if float(hfs_delta["upper"]) < -TIE_EPSILON else "HOLD"


def score_evaluation(
    observations: Iterable[Mapping[str, Any] | PairObservation],
    *,
    seed: int = 20260715,
    repetitions: int | None = None,
    expected_case_ids: Iterable[str] | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    hard_gate_failures: Iterable[str] = (),
    aa: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score paired observations without trusting a precomputed summary."""

    records = [
        item
        if isinstance(item, PairObservation)
        else PairObservation.from_mapping(item)
        for item in observations
    ]
    cases, duplicate_cases = _case_records(records, repetitions=repetitions)
    expected = set(expected_case_ids or ())
    observed = {case["case_id"] for case in cases}
    missing_cases = sorted(expected - observed)
    gates = list(dict.fromkeys(str(item) for item in hard_gate_failures))
    gates.extend(item for item in duplicate_cases if item not in gates)
    if missing_cases:
        gates.extend(f"missing-case:{case_id}" for case_id in missing_cases)

    candidate_cases = [case for case in cases if case["candidate_mean"] is not None]
    incumbent_cases = [case for case in cases if case["incumbent_mean"] is not None]
    complete_cases = [case for case in cases if case["complete"]]

    candidate_dimensions: dict[str, Fraction | None] = {}
    incumbent_dimensions: dict[str, Fraction | None] = {}
    dimension_cards: dict[str, dict[str, Any]] = {}
    paired_dimension_values: dict[str, list[Mapping[str, Any]]] = {}
    for dimension in DIMENSIONS:
        arm_candidate = [
            case["candidate_mean"]
            for case in candidate_cases
            if case["primary_dimension"] == dimension
        ]
        arm_incumbent = [
            case["incumbent_mean"]
            for case in incumbent_cases
            if case["primary_dimension"] == dimension
        ]
        paired = [
            case
            for case in complete_cases
            if case["primary_dimension"] == dimension
            and case["candidate_mean"] is not None
            and case["incumbent_mean"] is not None
        ]
        candidate_dimensions[dimension] = _mean(arm_candidate)  # type: ignore[arg-type]
        incumbent_dimensions[dimension] = _mean(arm_incumbent)  # type: ignore[arg-type]
        paired_dimension_values[dimension] = paired
        delta_values = [
            case["candidate_mean"] - case["incumbent_mean"] for case in paired
        ]
        delta_bootstrap = _bootstrap_case_values(
            paired,
            arm="delta",
            seed=seed,
            metric="dimension_delta",
            dimension=dimension,
            replicates=replicates,
        )
        dimension_cards[dimension] = {
            "candidate": _public(candidate_dimensions[dimension]),
            "incumbent": _public(incumbent_dimensions[dimension]),
            "delta": _public(_mean(delta_values)),
            "delta_ci": _interval(delta_bootstrap, replicates=replicates),
            "n_arm_candidate": len(arm_candidate),
            "n_arm_incumbent": len(arm_incumbent),
            "n_pair": len(paired),
        }

    candidate_hfs = _hfs(candidate_dimensions)
    incumbent_hfs = _hfs(incumbent_dimensions)
    paired_hfs_values = [
        case["candidate_mean"] - case["incumbent_mean"] for case in complete_cases
    ]
    dimension_bootstraps = {
        dimension: _bootstrap_case_values(
            paired_dimension_values[dimension],
            arm="delta",
            seed=seed,
            metric="hfs_delta",
            dimension=dimension,
            replicates=replicates,
        )
        for dimension in DIMENSIONS
    }
    hfs_bootstrap = [
        sum(
            (
                dimension_bootstraps[dimension][index] * DIMENSION_WEIGHTS[dimension]
                for dimension in DIMENSIONS
                if index < len(dimension_bootstraps[dimension])
            ),
            Fraction(),
        )
        / 100
        for index in range(replicates)
        if all(index < len(dimension_bootstraps[dimension]) for dimension in DIMENSIONS)
    ]
    hfs_delta = _interval(hfs_bootstrap, replicates=replicates)

    wins = sum(
        (case["candidate_mean"] - case["incumbent_mean"]) > TIE_EPSILON
        for case in complete_cases
    )
    losses = sum(
        (case["candidate_mean"] - case["incumbent_mean"]) < -TIE_EPSILON
        for case in complete_cases
    )
    ties = len(complete_cases) - wins - losses
    missing_dimensions = [
        dimension
        for dimension, card in dimension_cards.items()
        if min(card["n_arm_candidate"], card["n_arm_incumbent"], card["n_pair"]) == 0
    ]
    gates.extend(
        f"missing-dimension:{dimension}"
        for dimension in missing_dimensions
        if f"missing-dimension:{dimension}" not in gates
    )
    complete = not any(not case["complete"] for case in cases)
    eligible = not missing_dimensions and not missing_cases and not duplicate_cases
    counts = {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "complete": len(complete_cases),
        "incomplete": len(cases) - len(complete_cases),
        "invalid": len(cases) - len(complete_cases),
        "paired_win_rate": (wins / len(complete_cases)) if complete_cases else None,
    }
    return {
        "scorer_id": SCORER_ID,
        "scorer_version": SCORER_VERSION,
        "weights_version": WEIGHTS_VERSION,
        "screening_non_confirmatory": True,
        "status": _screening_status(
            gates=gates,
            eligible=eligible,
            complete=complete,
            candidate_hfs=candidate_hfs,
            incumbent_hfs=incumbent_hfs,
            hfs_delta=hfs_delta,
            aa=aa,
        ),
        "candidate": {
            "dimensions": {
                dimension: _public(value)
                for dimension, value in candidate_dimensions.items()
            },
            "hfs": _public(candidate_hfs),
        },
        "incumbent": {
            "dimensions": {
                dimension: _public(value)
                for dimension, value in incumbent_dimensions.items()
            },
            "hfs": _public(incumbent_hfs),
        },
        "dimensions": dimension_cards,
        "paired_hfs_delta": hfs_delta,
        "counts": counts,
        "n_arm": {
            "candidate": {
                dimension: dimension_cards[dimension]["n_arm_candidate"]
                for dimension in DIMENSIONS
            },
            "incumbent": {
                dimension: dimension_cards[dimension]["n_arm_incumbent"]
                for dimension in DIMENSIONS
            },
        },
        "n_pair": {
            dimension: dimension_cards[dimension]["n_pair"] for dimension in DIMENSIONS
        },
        "missing_cases": missing_cases,
        "missing_dimensions": missing_dimensions,
        "hard_gate_failures": gates,
        "aa_pilot": aa,
        "archive": None,
        "promotion_applied": False,
    }


def _bootstrap_difference(
    first: Sequence[Fraction], second: Sequence[Fraction], *, seed: int, replicates: int
) -> list[Fraction]:
    values: list[Fraction] = []
    for replicate in range(replicates):
        first_indexes = deterministic_indices(
            len(first),
            len(first),
            seed=seed,
            metric="aa_order",
            dimension="first",
            level=f"case-bootstrap:{replicate}",
        )
        second_indexes = deterministic_indices(
            len(second),
            len(second),
            seed=seed,
            metric="aa_order",
            dimension="second",
            level=f"case-bootstrap:{replicate}",
        )
        values.append(
            (_mean([first[index] for index in first_indexes]) or Fraction())
            - (_mean([second[index] for index in second_indexes]) or Fraction())
        )
    return values


def aa_acceptance(
    observations: Iterable[Mapping[str, Any] | PairObservation],
    *,
    receipt_integrity_rate: float,
    scorer_disagreement_count: int,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Apply the preregistered 81-pair incumbent-vs-incumbent harness gate."""

    records = [
        item
        if isinstance(item, PairObservation)
        else PairObservation.from_mapping(item)
        for item in observations
    ]
    deltas = [
        _fraction(item.candidate_score) - _fraction(item.incumbent_score)
        for item in records
        if item.complete and item.candidate_valid and item.incumbent_valid
    ]
    false_non_ties = sum(abs(delta) > TIE_EPSILON for delta in deltas)
    mean_ci = deterministic_bootstrap_ci(
        deltas,
        seed=seed,
        metric="aa_mean_delta",
        dimension="all",
        replicates=replicates,
    )
    first = [
        _fraction(item.candidate_score) - _fraction(item.incumbent_score)
        for item in records
        if item.complete and item.arm_order == "candidate-first"
    ]
    second = [
        _fraction(item.candidate_score) - _fraction(item.incumbent_score)
        for item in records
        if item.complete and item.arm_order == "incumbent-first"
    ]
    order_values = (
        _bootstrap_difference(first, second, seed=seed, replicates=replicates)
        if first and second
        else []
    )
    order_mean = (
        (_mean(first) or Fraction()) - (_mean(second) or Fraction())
        if first and second
        else None
    )
    order_ci = _interval(order_values, replicates=replicates) if order_values else None
    criteria = {
        "receipt_integrity": len(records) == 81 and receipt_integrity_rate == 1.0,
        "scorer_disagreement": len(records) == 81 and scorer_disagreement_count == 0,
        "false_non_tie_rate": len(records) == 81
        and false_non_ties <= 4
        and false_non_ties / 81 <= 0.05,
        "mean_delta": bool(mean_ci)
        and abs(float(mean_ci["mean"])) <= 1.0
        and _includes_zero(mean_ci),
        "order_effect": bool(order_ci)
        and abs(float(order_mean or 0)) <= 1.0
        and _includes_zero(order_ci),
    }
    return {
        "accepted": all(criteria.values()),
        "pairs": len(records),
        "false_non_ties": false_non_ties,
        "mean_delta": mean_ci,
        "order_effect": {"mean": _public(order_mean), "ci": order_ci},
        "criteria": criteria,
        "status": "PASS" if all(criteria.values()) else "GATE-FAILED",
    }


def archive_equivalence_key(value: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact flat policy key, supporting manifest and flat inputs."""

    source = value.get("archive_equivalence", value)

    def get(field: str) -> Any:
        if field in source:
            return _value(source[field])
        paths = {
            "lane_id": ("lane", "id"),
            "suite_id": ("lane", "suite_id"),
            "suite_version": ("lane", "suite_version"),
            "case_catalog_digest": ("lane", "case_catalog_digest"),
            "compression_mode": ("lane", "compression_mode"),
            "external_network": ("lane", "external_network"),
            "hermes_revision": ("hermes", "revision"),
            "config_policy_digest": ("hermes", "config_sha256"),
            "tool_schema_policy_digest": ("hermes", "resolved_tool_schema_sha256"),
            "hardware_class": ("hardware", "host_class"),
            "accelerator_family": ("hardware", "accelerator_family"),
            "device_count": ("hardware", "device_count"),
            "driver_major": ("hardware", "driver_major"),
            "runtime_major": ("runtime", "runtime_major"),
            "filesystem_scope": ("lane", "filesystem_scope"),
            "approval_policy": ("lane", "approval_policy"),
        }
        path = paths.get(field)
        if path is None:
            return None
        item: Any = source
        for part in path:
            if not isinstance(item, Mapping):
                return None
            item = item.get(part)
        return _value(item)

    defaults = {
        "scorer_id": SCORER_ID,
        "scorer_version": SCORER_VERSION,
        "weights_version": WEIGHTS_VERSION,
        "hard_gate_policy_version": 1,
        "pairing_policy_version": 1,
    }
    return {
        field: get(field) if get(field) is not None else defaults.get(field)
        for field in ARCHIVE_KEY_FIELDS
    }


def archive_policy_digest(value: Mapping[str, Any]) -> str:
    return canonical_hash(archive_equivalence_key(value))


def archive_rank(
    hfs: float,
    entries: Iterable[Mapping[str, Any]],
    *,
    equivalence_key: Mapping[str, Any],
    policy_digest: str,
) -> dict[str, Any]:
    """Rank only entries with an exact policy key and digest match."""

    expected = dict(equivalence_key)
    compatible = []
    for entry in entries:
        entry_key = entry.get("equivalence_key")
        if entry_key is None:
            entry_key = archive_equivalence_key(entry)
        if set(entry_key) != set(ARCHIVE_KEY_FIELDS):
            continue
        if dict(entry_key) != expected or entry.get("policy_digest") != policy_digest:
            continue
        if isinstance(entry.get("hfs"), (int, float)) and not isinstance(
            entry.get("hfs"), bool
        ):
            compatible.append(entry)
    if not compatible:
        return {
            "rank": None,
            "percentile": None,
            "n": 0,
            "reason": "no-compatible-archive",
        }
    ordered = sorted(
        compatible,
        key=lambda item: (-float(item["hfs"]), str(item.get("entry_id", ""))),
    )
    rank = 1 + sum(float(item["hfs"]) > float(hfs) for item in ordered)
    n = len(ordered)
    return {
        "rank": rank,
        "percentile": 100 * (rank - 0.5) / n,
        "n": n,
        "reason": None,
        "label": "provisional"
        if 20 <= n < 30
        else "useful"
        if n >= 30
        else "raw-rank-only",
    }


def verify_score_parity(online: Mapping[str, Any], offline: Mapping[str, Any]) -> bool:
    """Compare scorer outputs while ignoring paths/archive presentation fields."""

    ignored = {
        "archive",
        "raw_artifacts",
        "checksum_index",
        "out_dir",
        "parity",
        "warning",
        "aa_pilot",
    }
    return canonical_json({
        k: v for k, v in online.items() if k not in ignored
    }) == canonical_json({k: v for k, v in offline.items() if k not in ignored})
