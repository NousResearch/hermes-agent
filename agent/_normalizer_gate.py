"""Producer Normalizer operational gate.

Wraps ProducerNormalizer (v1.0) / ProducerNormalizerV1_1 with shadow-mode
and enforce-mode semantics for the live operational flow:

  Producer → ProducerNormalizer gate → conditional Reviewer

Shadow mode (default):
  - normalizer executes;
  - normalizer verdict + metrics recorded;
  - reviewer ALWAYS continues (no skipping);
  - reviewer verdict unchanged.

Enforce mode:
  - normalizer verdict ∈ {PASS, PARTIAL} → continue reviewer;
  - normalizer verdict ∈ {BLOCKED, NO_EVIDENCE} → skip reviewer call
    (saves HTTP cost and latency).

Configuration via environment variables:
  HERMES_PRODUCER_NORMALIZER_ENABLED   (1/0; default 0 = disabled)
  HERMES_PRODUCER_NORMALIZER_VERSION  ("1.0.0" | "1.1.0"; default "1.1.0")
  HERMES_PRODUCER_NORMALIZER_SHADOW_MODE (1/0; default 1 = shadow)

NO HTTP. NO LLM. The actual producer + reviewer HTTP calls are NOT made
by this module — it only decides whether the caller SHOULD call the
reviewer. The caller is responsible for invoking the reviewer (or not).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


_VALID_VERDICTS_PASS_PARTIAL = ("PASS", "PARTIAL")
_VALID_VERDICTS_BLOCK = ("BLOCKED", "NO_EVIDENCE")


@dataclass
class GateConfig:
    """Resolved gate configuration."""
    enabled: bool
    version: str
    shadow_mode: bool
    limited_enforce: bool

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "GateConfig":
        e = env if env is not None else dict(os.environ)
        enabled_raw = e.get("HERMES_PRODUCER_NORMALIZER_ENABLED", "0")
        version = e.get("HERMES_PRODUCER_NORMALIZER_VERSION", "1.1.0")
        shadow_raw = e.get("HERMES_PRODUCER_NORMALIZER_SHADOW_MODE", "1")
        limited_raw = e.get("HERMES_PRODUCER_NORMALIZER_LIMITED_ENFORCE", "0")
        return cls(
            enabled=bool(int(enabled_raw or 0)),
            version=str(version or "1.1.0"),
            shadow_mode=bool(int(shadow_raw or 1)),
            limited_enforce=bool(int(limited_raw or 0)),
        )


@dataclass
class GateDecision:
    """Output of the gate."""
    gate_status: str              # "DISABLED" | "SHADOW" | "ENFORCE_PASS" | "ENFORCE_BLOCK" | "ENFORCE_REVIEW" | "ENFORCE_LIMITED_BLOCK" | "ENGINE_STOP"
    shadow_mode: bool
    limited_enforce: bool
    normalizer_engine_status: str
    normalizer_verdict: str       # "PASS" | "PARTIAL" | "BLOCKED" | "NO_EVIDENCE" | "NOT_RUN"
    reviewer_should_run: bool
    reviewer_call_saved: bool
    reason: str
    limited_enforce_reason: Optional[str] = None
    report_path: Optional[str] = None
    metrics_path: Optional[str] = None
    metrics: dict = field(default_factory=dict)


def _load_normalizer(version: str, ruleset_path: Path, config_path: Path):
    """Lazy import + instantiate the right ProducerNormalizer class for `version`."""
    from agent.producer_normalizer import ProducerNormalizer, ProducerNormalizerV1_1

    if version == "1.0.0":
        return ProducerNormalizer(ruleset_path, config_path)
    if version == "1.1.0":
        return ProducerNormalizerV1_1(ruleset_path, config_path)
    raise ValueError(f"unsupported normalizer version: {version!r}")


def _default_ruleset_paths(version: str) -> tuple[Path, Path]:
    repo = Path("/home/jr-ubuntu/.hermes/hermes-agent")
    if version == "1.0.0":
        return (
            repo / "tests/contract_tests/normalizer_ruleset.v1.0.0.yaml",
            repo / "tests/contract_tests/normalizer_config.v1.0.0.yaml",
        )
    return (
        repo / "tests/contract_tests/normalizer_ruleset.v1.1.0.yaml",
        repo / "tests/contract_tests/normalizer_config.v1.1.0.yaml",
    )


def evaluate(
    bundle: dict[str, Any],
    bundle_root: Path,
    output_dir: Path,
    *,
    config: Optional[GateConfig] = None,
    ruleset_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> GateDecision:
    """Evaluate the gate against a bundle and return a GateDecision.

    Args:
        bundle: the EvidenceBundle v1.0.0 dict.
        bundle_root: filesystem root for the bundle artifacts.
        output_dir: where the normalizer writes normalizer_report +
                    normalizer_metrics + (optionally) the ruleset/config.
        config: optional pre-resolved GateConfig. If None, read from env.
        ruleset_path: optional override for the ruleset file.
        config_path: optional override for the config file.

    Returns:
        GateDecision with gate_status, normalizer_verdict, reviewer_should_run,
        and report/metrics paths.
    """
    cfg = config or GateConfig.from_env()

    # DISABLED path: gate off, always continue, no normalizer artifacts.
    if not cfg.enabled:
        return GateDecision(
            gate_status="DISABLED",
            shadow_mode=cfg.shadow_mode,
            limited_enforce=cfg.limited_enforce,
            normalizer_engine_status="OFF",
            normalizer_verdict="NOT_RUN",
            reviewer_should_run=True,
            reviewer_call_saved=False,
            reason="HERMES_PRODUCER_NORMALIZER_ENABLED=0",
        )

    # ENABLED path: instantiate the right normalizer class.
    if ruleset_path is None or config_path is None:
        ruleset_path, config_path = _default_ruleset_paths(cfg.version)
    ruleset_path = Path(ruleset_path)
    config_path = Path(config_path)

    if not ruleset_path.exists() or not config_path.exists():
        return GateDecision(
            gate_status="ENGINE_STOP",
            shadow_mode=cfg.shadow_mode,
            limited_enforce=cfg.limited_enforce,
            normalizer_engine_status="STOP",
            normalizer_verdict="NOT_RUN",
            reviewer_should_run=True,
            reviewer_call_saved=False,
            reason=f"ruleset/config not found: {ruleset_path} or {config_path}",
        )

    try:
        normalizer = _load_normalizer(cfg.version, ruleset_path, config_path)
        result = normalizer.run(bundle, bundle_root, output_dir)
    except Exception as e:
        return GateDecision(
            gate_status="ENGINE_STOP",
            shadow_mode=cfg.shadow_mode,
            limited_enforce=cfg.limited_enforce,
            normalizer_engine_status="STOP",
            normalizer_verdict="NOT_RUN",
            reviewer_should_run=True,
            reviewer_call_saved=False,
            reason=f"normalizer raised: {e!r}",
        )

    engine_status = result.get("engine_status", "STOP")
    verdict = result.get("normalizer_verdict", "NOT_RUN")
    if engine_status != "OK":
        return GateDecision(
            gate_status="ENGINE_STOP",
            shadow_mode=cfg.shadow_mode,
            limited_enforce=cfg.limited_enforce,
            normalizer_engine_status=engine_status,
            normalizer_verdict=verdict,
            reviewer_should_run=True,
            reviewer_call_saved=False,
            reason=f"normalizer engine_status={engine_status}",
            report_path=result.get("report_path"),
            metrics_path=result.get("metrics_path"),
            metrics=result.get("metrics", {}),
        )

    # Engine OK. Apply shadow vs enforce vs limited_enforce logic.
    if cfg.shadow_mode:
        # Shadow: log but always continue to reviewer.
        return GateDecision(
            gate_status="SHADOW",
            shadow_mode=True,
            limited_enforce=cfg.limited_enforce,
            normalizer_engine_status=engine_status,
            normalizer_verdict=verdict,
            reviewer_should_run=True,
            reviewer_call_saved=False,
            reason=f"shadow_mode=true; would have {'skipped' if verdict in _VALID_VERDICTS_BLOCK else 'continued'} reviewer",
            report_path=result.get("report_path"),
            metrics_path=result.get("metrics_path"),
            metrics=result.get("metrics", {}),
        )

    # Enforce mode.
    if cfg.limited_enforce:
        # Limited enforce: skip reviewer only for NO_EVIDENCE and BLOCKED (deterministic).
        # PARTIAL/PASS continue to reviewer.
        if verdict == "NO_EVIDENCE":
            return GateDecision(
                gate_status="ENFORCE_LIMITED_BLOCK",
                shadow_mode=False,
                limited_enforce=True,
                normalizer_engine_status=engine_status,
                normalizer_verdict=verdict,
                reviewer_should_run=False,
                reviewer_call_saved=True,
                reason=f"limited_enforce: verdict={verdict} (no artifacts) skips reviewer",
                limited_enforce_reason="no_evidence_deterministic",
                report_path=result.get("report_path"),
                metrics_path=result.get("metrics_path"),
                metrics=result.get("metrics", {}),
            )
        if verdict == "BLOCKED":
            # BLOCKED with engine_status=OK is deterministic by construction
            # (heuristic rules cannot emit blockers — validation enforces this).
            return GateDecision(
                gate_status="ENFORCE_LIMITED_BLOCK",
                shadow_mode=False,
                limited_enforce=True,
                normalizer_engine_status=engine_status,
                normalizer_verdict=verdict,
                reviewer_should_run=False,
                reviewer_call_saved=True,
                reason=f"limited_enforce: verdict=BLOCKED (deterministic) skips reviewer",
                limited_enforce_reason="blocked_deterministic",
                report_path=result.get("report_path"),
                metrics_path=result.get("metrics_path"),
                metrics=result.get("metrics", {}),
            )
        # PARTIAL/PASS: continue to reviewer.
        if verdict in _VALID_VERDICTS_PASS_PARTIAL:
            return GateDecision(
                gate_status="ENFORCE_REVIEW",
                shadow_mode=False,
                limited_enforce=True,
                normalizer_engine_status=engine_status,
                normalizer_verdict=verdict,
                reviewer_should_run=True,
                reviewer_call_saved=False,
                reason=f"limited_enforce: verdict={verdict} continues to reviewer",
                limited_enforce_reason="semantic_value",
                report_path=result.get("report_path"),
                metrics_path=result.get("metrics_path"),
                metrics=result.get("metrics", {}),
            )
        # Unknown verdict — fail-open: continue reviewer.
        return GateDecision(
            gate_status="ENFORCE_REVIEW",
            shadow_mode=False,
            limited_enforce=True,
            normalizer_engine_status=engine_status,
            normalizer_verdict=verdict,
            reviewer_should_run=True,
            reviewer_call_saved=False,
            reason=f"limited_enforce: unknown verdict={verdict!r}; fail-open: continue reviewer",
            limited_enforce_reason="unknown_verdict_fail_open",
            report_path=result.get("report_path"),
            metrics_path=result.get("metrics_path"),
            metrics=result.get("metrics", {}),
        )

    # Global enforce mode.
    if verdict in _VALID_VERDICTS_BLOCK:
        return GateDecision(
            gate_status="ENFORCE_BLOCK",
            shadow_mode=False,
            limited_enforce=False,
            normalizer_engine_status=engine_status,
            normalizer_verdict=verdict,
            reviewer_should_run=False,
            reviewer_call_saved=True,
            reason=f"enforce: verdict={verdict} blocks reviewer call",
            report_path=result.get("report_path"),
            metrics_path=result.get("metrics_path"),
            metrics=result.get("metrics", {}),
        )
    if verdict in _VALID_VERDICTS_PASS_PARTIAL:
        return GateDecision(
            gate_status="ENFORCE_REVIEW",
            shadow_mode=False,
            limited_enforce=cfg.limited_enforce,
            normalizer_engine_status=engine_status,
            normalizer_verdict=verdict,
            reviewer_should_run=True,
            reviewer_call_saved=False,
            reason=f"enforce: verdict={verdict} continues to reviewer",
            report_path=result.get("report_path"),
            metrics_path=result.get("metrics_path"),
            metrics=result.get("metrics", {}),
        )
    # Unknown verdict — fail safe: continue reviewer.
    return GateDecision(
        gate_status="ENFORCE_REVIEW",
        shadow_mode=False,
        limited_enforce=cfg.limited_enforce,
        normalizer_engine_status=engine_status,
        normalizer_verdict=verdict,
        reviewer_should_run=True,
        reviewer_call_saved=False,
        reason=f"enforce: unknown verdict={verdict!r}; fail-safe: continue reviewer",
        report_path=result.get("report_path"),
        metrics_path=result.get("metrics_path"),
        metrics=result.get("metrics", {}),
    )


def write_decision_log(decision: GateDecision, log_path: Path) -> None:
    """Append the GateDecision to a JSONL audit log (append-only)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate_status": decision.gate_status,
        "shadow_mode": decision.shadow_mode,
        "limited_enforce": decision.limited_enforce,
        "normalizer_engine_status": decision.normalizer_engine_status,
        "normalizer_verdict": decision.normalizer_verdict,
        "reviewer_should_run": decision.reviewer_should_run,
        "reviewer_call_saved": decision.reviewer_call_saved,
        "reason": decision.reason,
        "limited_enforce_reason": decision.limited_enforce_reason,
        "report_path": decision.report_path,
        "metrics_path": decision.metrics_path,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")