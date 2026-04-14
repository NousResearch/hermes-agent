#!/usr/bin/env python3
"""Deterministic A6 multi-host jitter/failure resilience harness.

Proves that injected 500ms network jitter does not improperly trip the failover FSM,
while sustained multi-host hard failures still escalate through reset-required paths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

from multi_host_failover_guard import GuardPolicy, evaluate_tick, policy_from_env

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_LATEST_EVIDENCE = "state/continuity/latest/a6_multi_host_jitter_resilience_evidence.json"
DEFAULT_OUTPUT_DIR = "state/continuity/a6_multi_host_jitter_resilience"
DEFAULT_DECISION_LOG = "state/continuity/a6_multi_host_jitter_resilience/decisions.jsonl"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_path(repo_root: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def ensure_repo_relative_path(repo_root: Path, target: Path, *, label: str) -> None:
    if not is_within(repo_root, target):
        raise ValueError(f"{label}_outside_repo:{target}")


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module_load_spec_failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _scenario_jitter_500ms() -> list[list[dict[str, Any]]]:
    return [
        [
            {"host_id": "host_a", "latency_ms": 120, "ok": True},
            {"host_id": "host_b", "latency_ms": 110, "ok": True},
            {"host_id": "host_c", "latency_ms": 130, "ok": True},
        ],
        [
            {"host_id": "host_a", "latency_ms": 520, "ok": True},
            {"host_id": "host_b", "latency_ms": 640, "ok": True},
            {"host_id": "host_c", "latency_ms": 505, "ok": True},
        ],
        [
            {"host_id": "host_a", "latency_ms": 500, "ok": True},
            {"host_id": "host_b", "latency_ms": 530, "ok": True},
            {"host_id": "host_c", "latency_ms": 490, "ok": True},
        ],
        [
            {"host_id": "host_a", "latency_ms": 145, "ok": True},
            {"host_id": "host_b", "latency_ms": 150, "ok": True},
            {"host_id": "host_c", "latency_ms": 160, "ok": True},
        ],
    ]


def _scenario_real_failures() -> list[list[dict[str, Any]]]:
    return [
        [
            {"host_id": "host_a", "latency_ms": 115, "ok": True},
            {"host_id": "host_b", "latency_ms": 125, "ok": True},
            {"host_id": "host_c", "latency_ms": 120, "ok": True},
        ],
        [
            {"host_id": "host_a", "latency_ms": 1250, "ok": False, "timed_out": True},
            {"host_id": "host_b", "latency_ms": 1240, "ok": False, "timed_out": True},
            {"host_id": "host_c", "latency_ms": 130, "ok": True},
        ],
        [
            {"host_id": "host_a", "latency_ms": 1300, "ok": False, "timed_out": True},
            {"host_id": "host_b", "latency_ms": 1100, "ok": False, "timed_out": True},
            {"host_id": "host_c", "latency_ms": 128, "ok": True},
        ],
    ]


def run_scenario(
    *,
    fsm: Any,
    name: str,
    host_windows: list[list[Mapping[str, Any]]],
    policy: GuardPolicy,
) -> dict[str, Any]:
    snapshot = fsm.build_state_snapshot(state="HEALTHY")
    warning_active = False
    streaks: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    selected_triggers: list[str] = []
    states: list[str] = []

    for tick_index, window in enumerate(host_windows, start=1):
        assessment = evaluate_tick(
            host_samples=[dict(row) for row in window],
            prior_failure_streaks=streaks,
            prior_warning_active=warning_active,
            policy=policy,
        )
        streaks = dict(assessment.host_failure_streaks)

        snapshot = fsm.reduce_failover_state(
            snapshot,
            triggers=assessment.trigger_set,
            blockers=[],
        )
        evaluation = snapshot.get("evaluation") or {}
        selected_trigger = str(evaluation.get("selected_trigger") or "")
        state = str(snapshot.get("state") or "")

        selected_triggers.append(selected_trigger)
        states.append(state)
        warning_active = state == "WARNING"

        rows.append(
            {
                "tick": tick_index,
                "host_samples": [dict(item) for item in window],
                "host_status": dict(assessment.host_status),
                "degraded_host_count": int(assessment.degraded_host_count),
                "failed_host_count": int(assessment.failed_host_count),
                "trigger_set": list(assessment.trigger_set),
                "selected_trigger": selected_trigger or None,
                "fsm_state": state,
            }
        )

    if name == "jitter_only_500ms":
        improper_states = [
            row
            for row in states
            if row in {"PRE_FAILOVER", "FAILOVER_PREP", "RESET_READY", "BLOCKED_RESET", "SUCCESSOR_RESUME_VALIDATION"}
        ]
        pass_checks = {
            "contains_500ms_or_higher_jitter": any(
                int(sample.get("latency_ms") or 0) >= 500
                for window in host_windows
                for sample in window
            ),
            "never_selected_reset_required": "TR_RESET_REQUIRED" not in selected_triggers,
            "no_failover_state_reached": len(improper_states) == 0,
        }
        expected_outcome = "RESIST_FALSE_POSITIVE_FAILOVER"
    elif name == "real_multi_host_failures":
        pass_checks = {
            "selected_reset_required": "TR_RESET_REQUIRED" in selected_triggers,
            "reached_failover_prep": "FAILOVER_PREP" in states,
            "failed_quorum_observed": any(
                int(row.get("failed_host_count") or 0) >= policy.failed_host_quorum_for_reset for row in rows
            ),
        }
        expected_outcome = "TRIGGER_TRUE_FAILURE_FAILOVER"
    else:
        pass_checks = {"unknown_scenario": False}
        expected_outcome = "UNKNOWN"

    failed_checks = [name for name, ok in pass_checks.items() if not bool(ok)]
    scenario_ok = not failed_checks

    return {
        "scenario": name,
        "expected_outcome": expected_outcome,
        "ok": scenario_ok,
        "failed_checks": failed_checks,
        "selected_trigger_path": selected_triggers,
        "fsm_state_path": states,
        "ticks": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = resolve_path(DEFAULT_REPO_ROOT, str(args.repo_root))
    output_dir = resolve_path(repo_root, str(args.output_dir))
    latest_evidence_path = resolve_path(repo_root, str(args.latest_evidence_path))
    decision_log_path = resolve_path(repo_root, str(args.decision_log))

    ensure_repo_relative_path(repo_root, output_dir, label="output_dir")
    ensure_repo_relative_path(repo_root, latest_evidence_path, label="latest_evidence_path")
    ensure_repo_relative_path(repo_root, decision_log_path, label="decision_log")

    fsm_path = repo_root / "ops" / "openclaw" / "continuity" / "failover_fsm.py"
    if not fsm_path.exists():
        raise FileNotFoundError(f"missing_module:{fsm_path}")
    fsm = load_module(fsm_path, "a6_multi_host_jitter_fsm")

    env_policy = policy_from_env(os.environ)
    policy = GuardPolicy(
        jitter_grace_ms=max(0, int(args.jitter_grace_ms if args.jitter_grace_ms is not None else env_policy.jitter_grace_ms)),
        hard_timeout_ms=max(1, int(args.hard_timeout_ms if args.hard_timeout_ms is not None else env_policy.hard_timeout_ms)),
        consecutive_failures_for_host_down=max(
            1,
            int(
                args.consecutive_failures_for_host_down
                if args.consecutive_failures_for_host_down is not None
                else env_policy.consecutive_failures_for_host_down
            ),
        ),
        failed_host_quorum_for_reset=max(
            1,
            int(
                args.failed_host_quorum_for_reset
                if args.failed_host_quorum_for_reset is not None
                else env_policy.failed_host_quorum_for_reset
            ),
        ),
    )

    scenarios = [
        run_scenario(
            fsm=fsm,
            name="jitter_only_500ms",
            host_windows=_scenario_jitter_500ms(),
            policy=policy,
        ),
        run_scenario(
            fsm=fsm,
            name="real_multi_host_failures",
            host_windows=_scenario_real_failures(),
            policy=policy,
        ),
    ]

    blocked_reasons = [
        f"scenario_failed:{row.get('scenario')}:{','.join(row.get('failed_checks') or ['unknown'])}"
        for row in scenarios
        if not bool(row.get("ok"))
    ]

    overall_verdict = "PASS" if not blocked_reasons else "FAIL_BLOCKED"
    generated_at = now_iso()

    run_seed = json_dumps(
        {
            "policy": policy.__dict__,
            "scenarios": [
                {
                    "scenario": row.get("scenario"),
                    "ok": row.get("ok"),
                    "selected_trigger_path": row.get("selected_trigger_path"),
                    "fsm_state_path": row.get("fsm_state_path"),
                }
                for row in scenarios
            ],
            "overall_verdict": overall_verdict,
        }
    )
    run_id = "a6multihost_" + hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:16]
    run_dir = output_dir / "runs" / run_id

    evidence = {
        "object_type": "clawd.a6_multi_host_jitter_resilience_evidence.v1",
        "run_id": run_id,
        "generated_at": generated_at,
        "lane": "A6",
        "harness": {
            "component": "continuity.a6_multi_host_jitter_harness",
            "version": "v1",
        },
        "policy": policy.__dict__,
        "summary": {
            "overall_verdict": overall_verdict,
            "scenario_count": len(scenarios),
            "scenario_pass_count": sum(1 for row in scenarios if bool(row.get("ok"))),
            "scenario_fail_count": sum(1 for row in scenarios if not bool(row.get("ok"))),
            "blocked_reasons": blocked_reasons,
            "jitter_injection_target_ms": 500,
        },
        "scenarios": scenarios,
        "artifacts": {
            "run_dir": str(run_dir.relative_to(repo_root)),
            "evidence_ref": str((run_dir / "evidence.json").relative_to(repo_root)),
            "latest_ref": str(latest_evidence_path.relative_to(repo_root)),
            "decision_log_ref": str(decision_log_path.relative_to(repo_root)),
        },
        "source_refs": [
            {
                "path": "ops/openclaw/continuity/failover_fsm.py",
                "sha256": file_sha256(fsm_path),
            },
            {
                "path": "ops/openclaw/continuity/multi_host_failover_guard.py",
                "sha256": file_sha256(repo_root / "ops" / "openclaw" / "continuity" / "multi_host_failover_guard.py"),
            },
            {
                "path": "ops/openclaw/continuity/a6_multi_host_jitter_harness.py",
                "sha256": file_sha256(repo_root / "ops" / "openclaw" / "continuity" / "a6_multi_host_jitter_harness.py"),
            },
        ],
    }

    write_json(run_dir / "evidence.json", evidence)
    write_json(latest_evidence_path, evidence)
    append_jsonl(
        decision_log_path,
        {
            "run_id": run_id,
            "generated_at": generated_at,
            "verdict": overall_verdict,
            "blocked_reasons": blocked_reasons,
            "evidence_ref": str((run_dir / "evidence.json").relative_to(repo_root)),
        },
    )

    return evidence


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Deterministic A6 multi-host jitter resilience harness")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Run artifact directory")
    ap.add_argument("--latest-evidence-path", default=DEFAULT_LATEST_EVIDENCE, help="Latest evidence artifact path")
    ap.add_argument("--decision-log", default=DEFAULT_DECISION_LOG, help="Decision log JSONL path")
    ap.add_argument("--jitter-grace-ms", type=int, default=None, help="Latency threshold treated as jitter (not hard fail)")
    ap.add_argument("--hard-timeout-ms", type=int, default=None, help="Latency threshold treated as hard failure")
    ap.add_argument(
        "--consecutive-failures-for-host-down",
        type=int,
        default=None,
        help="Consecutive hard-failure ticks before a host is counted down",
    )
    ap.add_argument(
        "--failed-host-quorum-for-reset",
        type=int,
        default=None,
        help="Failed-host quorum required to emit TR_RESET_REQUIRED",
    )
    ap.add_argument("--json", action="store_true", help="Print evidence JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        payload = run(args)
    except Exception as exc:
        error_payload = {
            "object_type": "clawd.a6_multi_host_jitter_resilience_evidence.v1",
            "run_id": None,
            "generated_at": now_iso(),
            "summary": {
                "overall_verdict": "FAIL_BLOCKED",
                "scenario_count": 0,
                "scenario_pass_count": 0,
                "scenario_fail_count": 1,
                "blocked_reasons": [f"harness_error:{exc}"],
                "jitter_injection_target_ms": 500,
            },
            "error": {"reason": str(exc)},
        }
        if args.json:
            print(json.dumps(error_payload, ensure_ascii=False, indent=2))
        else:
            print(f"BLOCKER: a6 multi-host jitter harness failed: {exc}")
        return 2

    verdict = str((((payload or {}).get("summary") or {}).get("overall_verdict") or "FAIL_BLOCKED")).upper()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{verdict}: a6_multi_host_jitter_run={payload.get('run_id')}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
