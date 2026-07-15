"""Pure deterministic scorer for the Hermes candidate screening lane."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

SCORER_ID = "hermes-fitness-v1"
SCORER_VERSION = 1
WEIGHTS_VERSION = "cli-full-v1"
BOOTSTRAP_RNG = "sha256-counter-v1"
BOOTSTRAP_REPLICATES = 10_000
TIE_EPSILON = 1.0
SCREENING_STATUSES = ("GATE-FAILED", "REJECT", "HOLD", "SCREEN-PASS")
DIMENSION_WEIGHTS = {"correctness": 25, "tool_behavior": 20, "recovery_multiturn": 15, "loaded_context_memory_skills": 15, "truthfulness_safety": 10, "reliability": 10, "performance": 5}
DIMENSIONS = tuple(DIMENSION_WEIGHTS)
_SECRET = re.compile(r"api[_-]?key|token|password|secret|cookie|authorization", re.I)

def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): "[REDACTED]" if _SECRET.search(str(key)) else redact_secrets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(https?://)([^/@\s]+):([^/@\s]+)@", r"\1[REDACTED]@", value)
        return re.sub(r"(?i)([?&](?:token|key|secret|password)=)[^&\s]+", r"\1[REDACTED]", value)
    return value

ARCHIVE_KEY_FIELDS = ("lane_id", "suite_id", "suite_version", "case_catalog_digest", "scorer_id", "scorer_version", "weights_version", "hard_gate_policy_version", "pairing_policy_version", "hermes_revision", "config_policy_digest", "tool_schema_policy_digest", "compression_mode", "external_network", "filesystem_scope", "approval_policy", "hardware_class", "accelerator_family", "device_count", "driver_major", "runtime_major")

def archive_equivalence_key(value: Mapping[str, Any]) -> dict[str, Any]:
    source = value.get("archive_equivalence", value)
    return {field: source.get(field) for field in ARCHIVE_KEY_FIELDS}

def archive_policy_digest(value: Mapping[str, Any]) -> str:
    return canonical_hash(archive_equivalence_key(value))

def _namespace(seed: int, metric: str, dimension: str, level: str) -> bytes:
    return b"\0".join((b"hermes-candidate-score-v1", int(seed).to_bytes(8, "big"), str(SCORER_VERSION).encode(), metric.encode(), dimension.encode(), level.encode()))

def deterministic_indices(n: int, draws: int, *, seed: int, metric: str = "hfs_delta", dimension: str = "all", level: str = "case") -> list[int]:
    if n <= 0 or draws < 0:
        raise ValueError("n must be positive and draws must be non-negative")
    limit = (1 << 64) - ((1 << 64) % n)
    namespace, result, counter = _namespace(seed, metric, dimension, level), [], 0
    while len(result) < draws:
        raw = int.from_bytes(hashlib.sha256(namespace + counter.to_bytes(8, "big")).digest()[:8], "big")
        counter += 1
        if raw < limit:
            result.append(raw % n)
    return result

def deterministic_bootstrap_ci(values: Iterable[float | int | Fraction], *, seed: int, metric: str = "hfs_delta", dimension: str = "all", confidence: float = 0.95, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any] | None:
    observations = [Fraction(round(float(value) * 100), 100) for value in values]
    if not observations:
        return None
    indexes = deterministic_indices(len(observations), len(observations) * replicates, seed=seed, metric=metric, dimension=dimension, level="case-bootstrap")
    means = [sum((observations[index] for index in indexes[offset:offset + len(observations)]), Fraction()) / len(observations) for offset in range(0, len(indexes), len(observations))]
    ordered = sorted(means)
    tail = Fraction(1 - confidence).limit_denominator(10_000) / 2
    def quantile(q: Fraction) -> Fraction:
        position = (len(ordered) - 1) * q
        low, high = int(position), min(int(position) + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    def rounded(value: Fraction) -> float:
        return float(round(value * 1000) / 1000)
    return {"mean": rounded(sum(observations, Fraction()) / len(observations)), "lower": rounded(quantile(tail)), "upper": rounded(quantile(1 - tail)), "confidence": confidence, "replicates": replicates, "rng": BOOTSTRAP_RNG, "seed": seed, "algorithm": "percentile_case_bootstrap_v1"}

@dataclass(frozen=True)
class PairObservation:
    case_id: str
    primary_dimension: str
    candidate_score: float
    incumbent_score: float
    complete: bool = True
    hard_gate_failures: tuple[str, ...] = ()
    arm_order: str = "candidate-first"
    candidate_valid: bool = True
    incumbent_valid: bool = True
    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PairObservation":
        candidate, incumbent = value.get("candidate_score", value.get("candidate", 0)), value.get("incumbent_score", value.get("incumbent", 0))
        if isinstance(candidate, Mapping): candidate = candidate.get("score", 0)
        if isinstance(incumbent, Mapping): incumbent = incumbent.get("score", 0)
        return cls(str(value["case_id"]), str(value["primary_dimension"]), float(candidate), float(incumbent), bool(value.get("complete", True)), tuple(str(item) for item in value.get("hard_gate_failures", value.get("gate_failures", ()))), str(value.get("arm_order", "candidate-first")), bool(value.get("candidate_valid", True)), bool(value.get("incumbent_valid", True)))

def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None

def _arm_means(records: Sequence[PairObservation], arm: str) -> dict[str, float | None]:
    return {dimension: _mean([getattr(item, f"{arm}_score") for item in records if item.primary_dimension == dimension and getattr(item, f"{arm}_valid")]) for dimension in DIMENSIONS}

def _hfs(means: Mapping[str, float | None]) -> float | None:
    return None if any(means.get(dimension) is None for dimension in DIMENSIONS) else sum(float(means[dimension]) * DIMENSION_WEIGHTS[dimension] for dimension in DIMENSIONS) / 100

def _includes_zero(ci: Mapping[str, Any] | None) -> bool:
    return bool(ci) and float(ci["lower"]) <= 0 <= float(ci["upper"])

def screening_status(*, hard_gate_pass: bool, eligible: bool, complete: bool, delta_ci: Mapping[str, Any] | None, candidate_hfs: float | None, incumbent_hfs: float | None, aa: Mapping[str, Any] | None = None) -> str:
    if not hard_gate_pass: return "GATE-FAILED"
    if not eligible or not complete or candidate_hfs is None or incumbent_hfs is None or delta_ci is None: return "HOLD"
    if aa is not None and not aa.get("accepted", False): return "HOLD"
    if candidate_hfs >= incumbent_hfs: return "SCREEN-PASS"
    return "REJECT" if float(delta_ci["upper"]) < -TIE_EPSILON else "HOLD"

def score_evaluation(observations: Iterable[Mapping[str, Any] | PairObservation], *, seed: int = 20260715, replicates: int = BOOTSTRAP_REPLICATES, hard_gate_failures: Iterable[str] = (), aa: Mapping[str, Any] | None = None) -> dict[str, Any]:
    records = [item if isinstance(item, PairObservation) else PairObservation.from_mapping(item) for item in observations]
    complete = [item for item in records if item.complete and item.candidate_valid and item.incumbent_valid]
    incomplete = [item for item in records if item not in complete]
    gates = list(hard_gate_failures) + [failure for item in records for failure in item.hard_gate_failures]
    candidate, incumbent = _arm_means(records, "candidate"), _arm_means(records, "incumbent")
    candidate_hfs, incumbent_hfs, deltas = _hfs(candidate), _hfs(incumbent), [item.candidate_score - item.incumbent_score for item in complete]
    dimensions = {}
    for dimension in DIMENSIONS:
        arm, pairs = [item for item in records if item.primary_dimension == dimension], []
        pairs = [item for item in arm if item.complete and item.candidate_valid and item.incumbent_valid]
        dimensions[dimension] = {"n_arm_candidate": sum(item.candidate_valid for item in arm), "n_arm_incumbent": sum(item.incumbent_valid for item in arm), "n_pair": len(pairs), "candidate": _mean([item.candidate_score for item in arm if item.candidate_valid]), "incumbent": _mean([item.incumbent_score for item in arm if item.incumbent_valid]), "delta": deterministic_bootstrap_ci([item.candidate_score - item.incumbent_score for item in pairs], seed=seed, metric="dimension_delta", dimension=dimension, replicates=replicates)}
    missing = [dimension for dimension, card in dimensions.items() if min(card["n_arm_candidate"], card["n_arm_incumbent"], card["n_pair"]) == 0]
    delta_ci = deterministic_bootstrap_ci(deltas, seed=seed, metric="hfs_delta", dimension="all", replicates=replicates)
    counts = {"wins": sum(delta > TIE_EPSILON for delta in deltas), "losses": sum(delta < -TIE_EPSILON for delta in deltas), "ties": sum(abs(delta) <= TIE_EPSILON for delta in deltas), "complete": len(complete), "incomplete": len(incomplete)}
    return {"scorer_id": SCORER_ID, "scorer_version": SCORER_VERSION, "weights_version": WEIGHTS_VERSION, "status": screening_status(hard_gate_pass=not gates and not incomplete, eligible=bool(records) and not missing, complete=not incomplete, delta_ci=delta_ci, candidate_hfs=candidate_hfs, incumbent_hfs=incumbent_hfs, aa=aa), "screening_non_confirmatory": True, "candidate": {"dimensions": candidate, "hfs": candidate_hfs}, "incumbent": {"dimensions": incumbent, "hfs": incumbent_hfs}, "dimensions": dimensions, "paired_hfs_delta": delta_ci, "counts": counts, "n_arm": {"candidate": {d: dimensions[d]["n_arm_candidate"] for d in DIMENSIONS}, "incumbent": {d: dimensions[d]["n_arm_incumbent"] for d in DIMENSIONS}}, "n_pair": {d: dimensions[d]["n_pair"] for d in DIMENSIONS}, "missing_dimensions": missing, "hard_gate_failures": gates, "archive": None, "promotion_applied": False}

def aa_acceptance(observations: Iterable[Mapping[str, Any] | PairObservation], *, receipt_integrity_rate: float, scorer_disagreement_count: int, seed: int, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    records = [item if isinstance(item, PairObservation) else PairObservation.from_mapping(item) for item in observations]
    deltas = [item.candidate_score - item.incumbent_score for item in records if item.complete]
    false_non_ties = sum(abs(delta) > TIE_EPSILON for delta in deltas)
    mean_ci = deterministic_bootstrap_ci(deltas, seed=seed, metric="aa_mean_delta", replicates=replicates)
    first = [item.candidate_score - item.incumbent_score for item in records if item.complete and item.arm_order == "candidate-first"]
    second = [item.candidate_score - item.incumbent_score for item in records if item.complete and item.arm_order == "incumbent-first"]
    order_mean = None if not first or not second else _mean(first) - _mean(second)
    order_ci = None if order_mean is None else {"mean": order_mean, "lower": order_mean, "upper": order_mean, "confidence": 0.95, "replicates": replicates, "rng": BOOTSTRAP_RNG}
    criteria = {"receipt_integrity": len(records) == 81 and receipt_integrity_rate == 1.0, "scorer_disagreement": len(records) == 81 and scorer_disagreement_count == 0, "false_non_tie_rate": len(records) == 81 and false_non_ties <= 4 and false_non_ties / 81 <= 0.05, "mean_delta": bool(mean_ci) and abs(float(mean_ci["mean"])) <= 1.0 and _includes_zero(mean_ci), "order_effect": order_mean is not None and abs(order_mean) <= 1.0 and _includes_zero(order_ci)}
    return {"accepted": all(criteria.values()), "pairs": len(records), "false_non_ties": false_non_ties, "mean_delta": mean_ci, "order_effect": {"mean": order_mean, "ci": order_ci}, "criteria": criteria, "status": "PASS" if all(criteria.values()) else "GATE-FAILED"}

def archive_rank(hfs: float, entries: Iterable[Mapping[str, Any]], *, equivalence_key: Mapping[str, Any], policy_digest: str) -> dict[str, Any]:
    compatible = [item for item in entries if item.get("equivalence_key") == dict(equivalence_key) and item.get("policy_digest") == policy_digest and isinstance(item.get("hfs"), (int, float))]
    if not compatible: return {"rank": None, "percentile": None, "n": 0, "reason": "no-compatible-archive"}
    ordered = sorted(compatible, key=lambda item: (-float(item["hfs"]), str(item.get("entry_id", ""))))
    rank, n = 1 + sum(float(item["hfs"]) > hfs for item in ordered), len(ordered)
    return {"rank": rank, "percentile": 100 * (rank - 0.5) / n, "n": n, "reason": None, "label": "provisional" if 20 <= n < 30 else "useful" if n >= 30 else "raw-rank-only"}

def verify_score_parity(online: Mapping[str, Any], offline: Mapping[str, Any]) -> bool:
    ignored = {"archive", "raw_artifacts", "checksum_index", "out_dir", "parity"}
    return canonical_json({k: v for k, v in online.items() if k not in ignored}) == canonical_json({k: v for k, v in offline.items() if k not in ignored})
