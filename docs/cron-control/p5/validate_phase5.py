#!/usr/bin/env python3
"""Validate the P5 rollout contract pack."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


class ValidationError(RuntimeError):
    pass


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - surfaced by CLI
        raise ValidationError(f"{path.name}: cannot read file ({exc})") from exc


def _load_json(path: Path) -> Any:
    try:
        return json.loads(_read_text(path))
    except Exception as exc:  # pragma: no cover - surfaced by CLI
        raise ValidationError(f"{path.name}: invalid JSON ({exc})") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(_read_text(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:  # pragma: no cover - surfaced by CLI
            raise ValidationError(f"{path.name}:{lineno}: invalid JSONL ({exc})") from exc
        if not isinstance(row, dict):
            raise ValidationError(f"{path.name}:{lineno}: expected object row")
        rows.append(row)
    return rows


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 30:
        raise ValidationError(f"dataset: expected 30 rows, got {len(rows)}")

    start = date(2026, 6, 30)
    allowed_labels = {
        "healthy",
        "quarantined",
        "human_required",
        "stale_running",
        "systemic_failure",
    }
    allowed_fixtures = {
        "hmm-policy-block.json",
        "cc-audit-stale-running.json",
        "receipt-conflict-429.json",
        "state-store-unavailable.json",
        "suppressed-receipt-success.json",
        "unknown-nonidempotent.json",
        "action-readback-failed.json",
    }

    seen_dates: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        for key in ("date", "incident_id", "job_id", "source_fixture", "scenario", "verdict_state", "recommended_action", "evidence_state", "automatic_action_allowed", "label"):
            if key not in row:
                raise ValidationError(f"dataset row {idx}: missing {key!r}")
        if row["label"] != row["verdict_state"]:
            raise ValidationError(f"dataset row {idx}: label must match verdict_state")
        if row["source_fixture"] not in allowed_fixtures:
            raise ValidationError(f"dataset row {idx}: unknown source_fixture {row['source_fixture']!r}")
        if row["verdict_state"] not in allowed_labels and row["verdict_state"] != "recoverable":
            raise ValidationError(f"dataset row {idx}: unsupported verdict_state {row['verdict_state']!r}")
        seen_dates.add(str(row["date"]))

    expected_dates = {
        (start + timedelta(days=offset)).isoformat() for offset in range(30)
    }
    if seen_dates != expected_dates:
        raise ValidationError("dataset: dates must cover the contiguous 30-day window 2026-06-30..2026-07-29")


def validate_allowlist(allowlist: dict[str, Any]) -> None:
    if allowlist.get("policy_id") != "p5-canary-allowlist-v1":
        raise ValidationError("allowlist: unexpected policy_id")
    actions = allowlist.get("actions")
    if not isinstance(actions, dict):
        raise ValidationError("allowlist: actions must be an object")
    expected_actions = {"shadow_diff", "auto_quarantine", "auto_reset", "model_switch"}
    if set(actions.keys()) != expected_actions:
        raise ValidationError("allowlist: actions must cover shadow_diff, auto_quarantine, auto_reset, model_switch")
    routes = allowlist.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValidationError("allowlist: routes must be a non-empty list")
    for idx, route in enumerate(routes, start=1):
        if not isinstance(route, dict):
            raise ValidationError(f"allowlist.routes[{idx}]: expected object")
        route_id = str(route.get("route_id") or "")
        if not route_id.startswith("openai-codex/"):
            raise ValidationError(f"allowlist.routes[{idx}]: route_id must start with openai-codex/")
    hard_limits = allowlist.get("hard_limits")
    if not isinstance(hard_limits, dict):
        raise ValidationError("allowlist: hard_limits must be an object")
    if hard_limits.get("cross_provider_fallback") is not False:
        raise ValidationError("allowlist: cross_provider_fallback must be false")


def validate_markdown(path: Path, required_fragments: list[str]) -> None:
    text = _read_text(path)
    for fragment in required_fragments:
        if fragment not in text:
            raise ValidationError(f"{path.name}: missing required fragment {fragment!r}")


def main() -> int:
    validate_dataset(_load_jsonl(ROOT / "labeled-dataset.jsonl"))
    validate_allowlist(_load_json(ROOT / "canary-allowlist.json"))
    validate_markdown(ROOT / "README.md", ["openai-codex/<model>", "validate_phase5.py"])
    validate_markdown(ROOT / "auto-quarantine-canary.md", ["quarantined", "delivery_receipt_conflict"])
    validate_markdown(ROOT / "auto-reset-canary.md", ["reset_job", "scheduled"])
    validate_markdown(ROOT / "model-switch-canary.md", ["switch_provider", "openai-codex", "gpt-5.6-terra"])
    validate_markdown(ROOT / "rollback-rehearsal.md", ["execute_rollback", "primary_model"])
    validate_markdown(ROOT / "production-readiness-review.md", ["Status: approved", "P5-T08"])
    validate_markdown(ROOT / "production-readiness-signoff.md", ["Status: approved", "Name: ryanchao", "Date: 2026-07-29"])
    print("phase-5 rollout pack validation passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
