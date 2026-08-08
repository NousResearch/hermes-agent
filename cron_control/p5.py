"""P5 rollout pack helpers.

This module ties together the phase-5 rollout artifacts:

* labeled dataset summary
* shadow quarantine verification
* reset-job canary
* openai-codex model-switch canary
* rollback rehearsal
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from cron.jobs import get_job, save_jobs, use_cron_store
from cron.provider_recovery import execute_rollback
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.model_switch import switch_model

from .actions import execute_verdict_action
from .evaluator import evaluate_job_verdict

ROOT = Path(__file__).resolve().parents[1]
P5_DIR = ROOT / "docs" / "cron-control" / "p5"
P0_DIR = ROOT / "docs" / "cron-control" / "p0"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_p5_dataset() -> list[dict[str, Any]]:
    return _load_jsonl(P5_DIR / "labeled-dataset.jsonl")


def load_p5_allowlist() -> dict[str, Any]:
    return _load_json(P5_DIR / "canary-allowlist.json")


def _dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("label") or "") for row in rows)
    return {
        "rows": len(rows),
        "date_span": [rows[0]["date"], rows[-1]["date"]] if rows else [None, None],
        "label_counts": dict(sorted(counts.items())),
        "fixtures": sorted({str(row.get("source_fixture") or "") for row in rows}),
    }


def _base_job(job_id: str, *, model: str) -> dict[str, Any]:
    job = _load_json(P0_DIR / "examples" / "job-metadata.agent.example.json")
    job = dict(job)
    job["id"] = job_id
    job["name"] = f"P5 canary {job_id}"
    job["model"] = model
    job["provider"] = "openai-codex"
    job["provider_snapshot"] = "openai-codex"
    job["model_snapshot"] = model
    job["enabled"] = True
    job["state"] = "running"
    job["run_claim"] = {"claimed_at": "2026-07-29T00:00:00+08:00", "fencing_token": 7}
    job["fire_claim"] = {"claimed_at": "2026-07-29T00:00:00+08:00", "fencing_token": 7}
    job["last_run_at"] = "2026-07-29T00:00:00+08:00"
    job["last_status"] = "running"
    job["last_error"] = None
    job["last_delivery_error"] = None
    job["control_policy"] = dict(job.get("control_policy") or {})
    job["control_policy"]["fallback_chain"] = [
        {
            "route_id": "p5-openai-codex-gpt-5.4",
            "provider": "openai-codex",
            "model": "gpt-5.4",
        },
        {
            "route_id": "p5-openai-codex-gpt-5.6-terra",
            "provider": "openai-codex",
            "model": "gpt-5.6-terra",
        },
        {
            "route_id": "p5-openai-codex-gpt-5.6-sol",
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
        },
    ]
    return job


def _seed_temp_home(temp_home: Path, jobs: list[dict[str, Any]]) -> None:
    cron_dir = temp_home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    config_text = """fallback_providers:
  - provider: openai-codex
    model: gpt-5.4
  - provider: openai-codex
    model: gpt-5.6-terra
  - provider: openai-codex
    model: gpt-5.6-sol
"""
    (temp_home / "config.yaml").write_text(config_text, encoding="utf-8")
    with use_cron_store(temp_home):
        save_jobs(jobs)


def _run_quarantine_canary() -> dict[str, Any]:
    fixture = _load_json(P0_DIR / "fixtures" / "receipt-conflict-429.json")
    job = fixture["job"]
    evidence = fixture["evidence"]
    verdict = evaluate_job_verdict(job, evidence)
    return {
        "fixture": "receipt-conflict-429.json",
        "state": verdict["state"],
        "recommended_action": verdict["recommended_action"],
        "automatic_action_allowed": verdict["automatic_action_allowed"],
        "rule_id": verdict["rule_id"],
        "blocked_by": verdict["blocked_by"],
        "ok": verdict["state"] == "quarantined" and verdict["recommended_action"] == "escalate_to_human",
    }


def _run_reset_canary(temp_home: Path) -> dict[str, Any]:
    job_id = "p5-reset-canary"
    with ExitStack() as stack:
        token = set_hermes_home_override(temp_home)
        stack.callback(reset_hermes_home_override, token)
        stack.enter_context(use_cron_store(temp_home))
        save_jobs([_base_job(job_id, model="gpt-5.4")])
        verdict = {
            "verdict_id": "vd_p5_reset",
            "incident_id": "inc_p5_reset",
            "job_id": job_id,
            "state": "stale_running",
            "evidence_state": "complete",
            "rule_id": "P5_STALE_RUNNING_RESET_JOB",
            "evidence_refs": ["ev_p5_reset"],
            "recommended_action": "reset_job",
            "automatic_action_allowed": True,
            "blocked_by": [],
            "classified_at": "2026-07-29T00:00:00+08:00",
            "classifier_version": "cron_control.p5/v1",
        }
        outcome = execute_verdict_action(verdict, approved=True)
        job_after = get_job(job_id) or {}
        return {
            "fixture": "stale-running-reset",
            "action_outcome": outcome,
            "job_after": {
                "id": job_after.get("id"),
                "state": job_after.get("state"),
                "enabled": job_after.get("enabled"),
                "run_claim": job_after.get("run_claim"),
                "fire_claim": job_after.get("fire_claim"),
            },
            "ok": outcome.get("status") == "verified" and job_after.get("state") == "scheduled" and not job_after.get("run_claim") and not job_after.get("fire_claim"),
        }


def _run_model_switch_canary(temp_home: Path) -> dict[str, Any]:
    job_id = "p5-switch-canary"
    with ExitStack() as stack:
        token = set_hermes_home_override(temp_home)
        stack.callback(reset_hermes_home_override, token)
        stack.enter_context(use_cron_store(temp_home))
        save_jobs([_base_job(job_id, model="gpt-5.4")])
        verdict = {
            "verdict_id": "vd_p5_switch",
            "incident_id": "inc_p5_switch",
            "job_id": job_id,
            "state": "recoverable",
            "evidence_state": "complete",
            "rule_id": "P5_OPENAI_CODEX_MODEL_SWITCH",
            "evidence_refs": ["ev_p5_switch"],
            "recommended_action": "switch_provider",
            "automatic_action_allowed": True,
            "blocked_by": [],
            "classified_at": "2026-07-29T00:00:00+08:00",
            "classifier_version": "cron_control.p5/v1",
        }
        stack.enter_context(
            patch(
                "cron_control.actions._fallback_for_provider",
                return_value=("openai-codex", "gpt-5.6-terra"),
            )
        )
        outcome = execute_verdict_action(verdict, approved=True)
        job_after = get_job(job_id) or {}
        rollback = execute_rollback(job_id)
        job_rolled_back = get_job(job_id) or {}
        switch_ok = bool(
            switch_model(
                "gpt-5.6-terra",
                current_provider="openai-codex",
                current_model="gpt-5.4",
                current_base_url="",
                current_api_key="",
                user_providers=None,
            ).success
        )
        return {
            "fixture": "openai-codex-model-switch",
            "action_outcome": outcome,
            "switch_model_result": switch_ok,
            "job_after_switch": {
                "provider": job_after.get("provider"),
                "model": job_after.get("model"),
                "recovery_state": job_after.get("recovery_state"),
            },
            "rollback_record": {
                "primary_provider": rollback.primary_provider if rollback else None,
                "primary_model": rollback.primary_model if rollback else None,
                "fallback_provider": rollback.fallback_provider if rollback else None,
                "fallback_model": rollback.fallback_model if rollback else None,
            },
            "job_after_rollback": {
                "provider": job_rolled_back.get("provider"),
                "model": job_rolled_back.get("model"),
                "recovery_state": job_rolled_back.get("recovery_state"),
            },
                "ok": (
                    outcome.get("status") == "verified"
                and switch_ok
                and job_after.get("provider") == "openai-codex"
                and job_after.get("model") == "gpt-5.6-terra"
                and rollback is not None
                and job_rolled_back.get("provider") == "openai-codex"
                and job_rolled_back.get("model") == "gpt-5.4"
            ),
        }


def run_p5_canaries() -> dict[str, Any]:
    dataset = load_p5_dataset()
    allowlist = load_p5_allowlist()

    with tempfile.TemporaryDirectory(prefix="hermes-p5-") as tmp:
        temp_home = Path(tmp)
        token = set_hermes_home_override(temp_home)
        try:
            _seed_temp_home(temp_home, [_base_job("p5-reset-canary", model="gpt-5.4"), _base_job("p5-switch-canary", model="gpt-5.4")])
            quarantine = _run_quarantine_canary()
            reset = _run_reset_canary(temp_home)
            model_switch = _run_model_switch_canary(temp_home)
            summary = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": _dataset_summary(dataset),
                "allowlist": {
                    "policy_id": allowlist.get("policy_id"),
                    "routes": allowlist.get("routes"),
                    "actions": list((allowlist.get("actions") or {}).keys()),
                },
                "canaries": {
                    "quarantine": quarantine,
                    "reset": reset,
                    "model_switch": model_switch,
                },
                "all_passed": quarantine["ok"] and reset["ok"] and model_switch["ok"],
            }
            return summary
        finally:
            reset_hermes_home_override(token)
