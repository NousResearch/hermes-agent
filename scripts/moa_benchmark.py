#!/usr/bin/env python3
"""Локальный воспроизводимый benchmark маршрутизации и готовности MoA.

По умолчанию не выполняет модельных запросов. Live-режим требует одновременно
``--live`` и ``--confirm-live``, запрещает MoA traces, проверяет суммарный
reference budget до первого provider call и сохраняет только обезличенные
метрики и hash ответа.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from hermes_cli.config import load_config, reload_env
from hermes_cli.moa_config import (
    classify_moa_auto_route,
    evaluate_moa_runtime_config,
    resolve_moa_preset_for_messages,
    validate_moa_config,
)


CASES = [
    ("translate", "Translate this short sentence into Spanish.", "fast"),
    ("rewrite", "Rewrite this paragraph in one concise sentence.", "fast"),
    ("format", "Format these three values as a Markdown table.", "fast"),
    ("current_docs", "Check the latest official API documentation and cite sources.", "research"),
    ("news", "Find current news about this product and compare primary sources.", "research"),
    ("paper", "Research this paper online and verify its main claim.", "research"),
    ("bug", "Fix the bug in parser.py and run targeted tests.", "code_heavy"),
    ("refactor", "Refactor this TypeScript API without changing its public contract.", "code_heavy"),
    ("traceback", "Diagnose this traceback and implement a verified fix.", "code_heavy"),
    ("architecture", "Review the repository architecture and propose a migration.", "code_heavy"),
    ("max_explicit", "Use preset max and provide the deepest possible analysis.", "max"),
    ("maximum_quality", "I need maximum quality and the most thorough cross-check.", "max"),
    ("product_plan", "Prepare a practical launch plan for a new market.", "balanced"),
    ("decision", "Compare these options and recommend one with tradeoffs.", "balanced"),
    ("analysis", "Analyze the proposal and identify the main risks.", "balanced"),
    ("summary", "Summarize the supplied report and suggest next actions.", "balanced"),
    ("email", "Draft a professional response to this customer request.", "balanced"),
    ("strategy", "Develop a strategy for improving retention next quarter.", "balanced"),
    ("requirements", "Turn this idea into a clear implementation plan.", "balanced"),
    ("review", "Review this decision memo for weak assumptions.", "balanced"),
]
LIVE_CASE_IDS = ["translate", "current_docs", "bug", "max_explicit", "product_plan"]
DEFAULT_REFERENCE_BUDGET_USD = 0.75


def _response_text(response: Any) -> str:
    try:
        return str(response.choices[0].message.content or "")
    except Exception:
        return ""


def selected_live_cases(max_cases: int) -> list[tuple[str, str, str]]:
    """Вернуть стабильный ограниченный набор только синтетических кейсов."""
    limit = max(1, min(int(max_cases), len(LIVE_CASE_IDS)))
    return [case for case in CASES if case[0] in LIVE_CASE_IDS][:limit]


def validate_live_preflight(
    config: dict[str, Any],
    preset: str,
    max_cases: int,
    reference_budget_usd: float = DEFAULT_REFERENCE_BUDGET_USD,
) -> dict[str, Any]:
    """Fail closed до provider call, если budget/privacy нельзя доказать."""
    if bool(config.get("save_traces", False)):
        raise ValueError(
            "live benchmark requires moa.save_traces=false because responses "
            "must not be persisted"
        )
    if not math.isfinite(reference_budget_usd) or reference_budget_usd < 0:
        raise ValueError("reference budget must be finite and non-negative")

    errors = [
        issue
        for issue in validate_moa_config(config)
        if issue.get("severity") == "error"
    ]
    if errors:
        raise ValueError("MoA config has validation errors")

    allocations = []
    total = 0.0
    for case_id, prompt, _expected in selected_live_cases(max_cases):
        try:
            resolved_name, resolved = resolve_moa_preset_for_messages(
                config,
                preset,
                [{"role": "user", "content": prompt}],
            )
        except KeyError as exc:
            raise ValueError(f"unknown MoA preset: {preset}") from exc
        references = resolved.get("reference_models") or []
        configured_cap = resolved.get("max_reference_cost_usd")
        if references and configured_cap is None:
            raise ValueError(
                f"preset {resolved_name!r} has references but no "
                "max_reference_cost_usd"
            )
        case_budget = float(configured_cap or 0.0)
        total += case_budget
        allocations.append({
            "id": case_id,
            "resolved_preset": resolved_name,
            "reference_budget_usd": round(case_budget, 6),
        })

    if total > reference_budget_usd + 1e-9:
        raise ValueError(
            f"planned reference budget ${total:.6f} exceeds limit "
            f"${reference_budget_usd:.6f}"
        )
    return {
        "planned_reference_budget_usd": round(total, 6),
        "reference_budget_limit_usd": round(float(reference_budget_usd), 6),
        "cases": allocations,
        "save_traces": False,
    }


def run_dry(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case_id, prompt, expected in CASES:
        predicted = classify_moa_auto_route([{"role": "user", "content": prompt}])
        rows.append({
            "id": case_id,
            "expected": expected,
            "predicted": predicted,
            "pass": predicted == expected,
        })
    runtime = evaluate_moa_runtime_config(config)
    passed = sum(int(row["pass"]) for row in rows)
    return {
        "mode": "dry-run",
        "cases": len(rows),
        "route_accuracy": round(passed / max(1, len(rows)), 4),
        "route_failures": [row for row in rows if not row["pass"]],
        "validation_issues": validate_moa_config(config),
        "runtime_degraded": runtime["degraded"],
        "degraded_presets": [
            name for name, status in runtime["presets"].items() if status["degraded"]
        ],
    }


def run_live(config: dict[str, Any], preset: str, max_cases: int) -> dict[str, Any]:
    from agent.moa_loop import MoAChatCompletions

    rows = []
    live_cases = selected_live_cases(max_cases)
    for case_id, prompt, expected in live_cases:
        resolved_name, _resolved = resolve_moa_preset_for_messages(
            config,
            preset,
            [{"role": "user", "content": prompt}],
        )
        started = time.monotonic()
        facade = MoAChatCompletions(preset)
        try:
            response = facade.create(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                stream=False,
            )
            text = _response_text(response)
            status = "ok" if text.strip() else "empty"
        except Exception as exc:
            text = ""
            status = f"error:{type(exc).__name__}"
        rows.append({
            "id": case_id,
            "expected_route": expected,
            "resolved_preset": resolved_name,
            "status": status,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "response_chars": len(text),
            "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else "",
            "runtime": facade.last_runtime_status,
        })
    return {
        "mode": "live",
        "preset": preset,
        "cases": len(rows),
        "successful": sum(int(row["status"] == "ok") for row in rows),
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument(
        "--reference-budget-usd",
        type=float,
        default=DEFAULT_REFERENCE_BUDGET_USD,
        help="maximum summed per-preset reference budget for this live run",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    reload_env()
    config = load_config().get("moa") or {}
    if args.live and not args.confirm_live:
        parser.error("--live requires --confirm-live because it performs billed/provider requests")
    if args.live:
        try:
            preflight = validate_live_preflight(
                config,
                args.preset,
                args.max_cases,
                args.reference_budget_usd,
            )
        except ValueError as exc:
            parser.error(str(exc))
        result = run_live(config, args.preset, args.max_cases)
        result["preflight"] = preflight
    else:
        result = run_dry(config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if any(
        issue.get("severity") == "error"
        for issue in result.get("validation_issues", [])
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
