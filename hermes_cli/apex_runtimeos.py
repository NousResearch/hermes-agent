"""CLI helpers for APEX RuntimeOS diagnostics."""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

from agent.apex_runtimeos_audit_summary import render_markdown, summarize_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="/apex-runtimeos",
        description="Show APEX RuntimeOS audit summary diagnostics",
    )
    parser.add_argument("command", nargs="?", default="summary", choices=("summary", "status", "feishu", "autopromote", "rollback", "autonomy", "cron-ledger"))
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--limit", type=int, default=10000, help="max audit lines to read")
    parser.add_argument("--target", default="memory", choices=("memory", "skill", "all"), help="autopromote/rollback target")
    parser.add_argument("--execute", action="store_true", help="execute side effects; default is dry-run/disabled")
    parser.add_argument("--repair", action="store_true", help="quarantine bad cron-ledger lines; requires enforce mode")
    parser.add_argument("--min-occurrences", type=int, default=2, help="minimum repeated candidate count for autopromote/autonomy")
    return parser


def _top_rows(items: Mapping[str, Any], *, limit: int = 6) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in items.items():
        if isinstance(value, Mapping):
            rows.append((str(key), value))
    return sorted(rows, key=lambda item: int(item[1].get("count") or 0), reverse=True)[:limit]


def _status_icon(summary: Mapping[str, Any]) -> str:
    if int(summary.get("blocking_records") or 0) > 0:
        return "⚠️"
    if int(summary.get("bad_lines") or 0) > 0:
        return "🟡"
    return "✅"


def _feishu_markdown(summary: Mapping[str, Any]) -> str:
    organs = summary.get("organs") if isinstance(summary.get("organs"), Mapping) else {}
    stages = summary.get("stages") if isinstance(summary.get("stages"), Mapping) else {}
    recommendations = summary.get("recommendations") if isinstance(summary.get("recommendations"), Mapping) else {}
    recommendations_count = int(recommendations.get("count") or 0) if isinstance(recommendations, Mapping) else 0
    recommendations_status = recommendations.get("status", {}) if isinstance(recommendations, Mapping) else {}
    recommendation_gate = str(summary.get("recommendation_gate", "OK"))
    icon = _status_icon(summary)
    lines = [
        f"# {icon} APEX RuntimeOS 体征摘要",
        "",
        "**状态：手动只读摘要，未自动推送，未修改运行状态。**",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 有效记录 | {summary.get('records', 0)} |",
        f"| 坏行 | {summary.get('bad_lines', 0)} |",
        f"| 阻断记录 | {summary.get('blocking_records', 0)} |",
        f"| 建议数 | {recommendations_count} |",
        f"| 建议状态 | {recommendations_status} |",
        f"| 建议门禁 | {recommendation_gate} |",
        f"| audit 文件存在 | {summary.get('audit_path_exists', False)} |",
        "",
        "## 建议概览",
        "",
        "| code | severity | applied | mutates_runtime | reason |",
        "|---|---|---:|---:|---|",
    ]
    top_recommendations = recommendations.get("items") if isinstance(recommendations.get("items"), list) else []
    if top_recommendations:
        for item in top_recommendations[:6]:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"| {item.get('code', '')} | {item.get('severity', '')} | {item.get('applied', False)} | {item.get('mutates_runtime', False)} | {item.get('reason', '')} |"
            )
    else:
        lines.append("| - | - | False | False | - |")
    lines.extend([
        "",
        "## Cron dry-run 账本",
        "",
        "| 字段 | 数值 |",
        "|---|---:|",
    ])
    autonomy_raw = summary.get("autonomy") if isinstance(summary.get("autonomy"), Mapping) else {}
    autonomy: Mapping[str, Any] = autonomy_raw if isinstance(autonomy_raw, Mapping) else {}
    health_raw = autonomy.get("health_report") if isinstance(autonomy.get("health_report"), Mapping) else {}
    health: Mapping[str, Any] = health_raw if isinstance(health_raw, Mapping) else {}
    cron_raw = autonomy.get("cron_dryrun") if isinstance(autonomy.get("cron_dryrun"), Mapping) else {}
    cron: Mapping[str, Any] = cron_raw if isinstance(cron_raw, Mapping) else {}
    lines.extend([
        f"| health_status | {health.get('status', 'OK')} |",
        f"| alert_count | {health.get('alert_count', 0)} |",
        f"| ledger_exists | {cron.get('ledger_exists', False)} |",
        f"| unique_keys | {cron.get('unique_keys', 0)} |",
        f"| bad_lines | {cron.get('bad_lines', 0)} |",
        f"| last_seen_at | {cron.get('last_seen_at', 0)} |",
    ])
    alerts = health.get("alerts") if isinstance(health.get("alerts"), list) else []
    if alerts:
        lines.extend(["", "### 阈值提醒", "", "| code | severity | count | message |", "|---|---|---:|---|"])
        for alert in alerts[:6]:
            if isinstance(alert, Mapping):
                lines.append(f"| {alert.get('code')} | {alert.get('severity')} | {alert.get('count')} | {alert.get('message')} |")
    lines.extend([
        "",
        "## Organ 概览",
        "",
        "| organ | 数量 | 阻断 | 状态 | 平均耗时 ms |",
        "|---|---:|---:|---|---:|",
    ])
    top_organs = _top_rows(organs)
    if top_organs:
        for organ, data in top_organs:
            lines.append(f"| {organ} | {data.get('count', 0)} | {data.get('blocking', 0)} | {data.get('status', {})} | {data.get('avg_elapsed_ms', 0)} |")
    else:
        lines.append("| - | 0 | 0 | {} | 0 |")
    lines.extend(["", "## Stage 概览", "", "| stage | 数量 | 阻断 | 状态 | 平均耗时 ms |", "|---|---:|---:|---|---:|"])
    top_stages = _top_rows(stages)
    if top_stages:
        for stage, data in top_stages:
            lines.append(f"| {stage} | {data.get('count', 0)} | {data.get('blocking', 0)} | {data.get('status', {})} | {data.get('avg_elapsed_ms', 0)} |")
    else:
        lines.append("| - | 0 | 0 | {} | 0 |")
    lines.extend([
        "",
        "## 边界",
        "",
        "- 只读聚合：不读取、不输出原始对话正文。",
        "- 不输出本地 audit 路径、原始错误堆栈、凭据。",
        "- 本命令只生成文本，不自动发送飞书。",
    ])
    return "\n".join(lines) + "\n"


def _cn_markdown(summary: Mapping[str, Any]) -> str:
    organs = summary.get("organs") or {}
    stages = summary.get("stages") or {}
    lines = [
        "# APEX RuntimeOS 诊断摘要",
        "",
        "| 字段 | 数值 |",
        "|---|---:|",
        f"| 有效记录 | {summary.get('records', 0)} |",
        f"| 坏行 | {summary.get('bad_lines', 0)} |",
        f"| 阻断记录 | {summary.get('blocking_records', 0)} |",
        f"| audit 文件存在 | {summary.get('audit_path_exists', False)} |",
        "",
        "## Organs",
        "",
        "| organ | 数量 | 阻断 | 状态 | 平均耗时 ms |",
        "|---|---:|---:|---|---:|",
    ]
    if organs:
        for organ, data in sorted(organs.items()):
            lines.append(
                f"| {organ} | {data.get('count', 0)} | {data.get('blocking', 0)} | "
                f"{data.get('status', {})} | {data.get('avg_elapsed_ms', 0)} |"
            )
    else:
        lines.append("| - | 0 | 0 | {} | 0 |")
    lines.extend([
        "",
        "## Stages",
        "",
        "| stage | 数量 | 阻断 | 状态 | 平均耗时 ms |",
        "|---|---:|---:|---|---:|",
    ])
    if stages:
        for stage, data in sorted(stages.items()):
            lines.append(
                f"| {stage} | {data.get('count', 0)} | {data.get('blocking', 0)} | "
                f"{data.get('status', {})} | {data.get('avg_elapsed_ms', 0)} |"
            )
    else:
        lines.append("| - | 0 | 0 | {} | 0 |")
    return "\n".join(lines) + "\n"


def run_apex_runtimeos_cli(argv: list[str] | None = None) -> str:
    """Return CLI output for the /apex-runtimeos command.

    Output is aggregate-only: no local paths, prompts, messages, raw errors, or
    credentials are rendered.
    """
    parser = _parser()
    ns = parser.parse_args(argv or [])
    limit = max(1, min(int(ns.limit), 100000))
    if ns.command == "autonomy":
        from agent.apex_runtimeos_autonomy import summarize_autonomy_status
        status = summarize_autonomy_status(limit=limit, min_occurrences=max(1, int(ns.min_occurrences)))
        if ns.json:
            return json.dumps({"object": "hermes.apex_runtimeos.autonomy_status", "status": status}, ensure_ascii=False, indent=2)
        cron = status.get("cron_dryrun") if isinstance(status.get("cron_dryrun"), dict) else {}
        return "\n".join([
            "# APEX RuntimeOS 自主化状态",
            "",
            "| 字段 | 数值 |",
            "|---|---:|",
            f"| mode | {status.get('mode')} |",
            f"| autopromote_enabled | {status.get('autopromote_enabled')} |",
            f"| rollback_enabled | {status.get('rollback_enabled')} |",
            f"| stable_ready_count | {status.get('stable_ready_count')} |",
            f"| candidate_groups | {status.get('candidate_groups')} |",
            f"| promotion_count | {status.get('promotion_count')} |",
            f"| pending_rollbacks | {status.get('pending_rollbacks')} |",
            f"| rollback_events | {status.get('rollback_events', {}).get('count', 0)} |",
            f"| health_status | {status.get('health_report', {}).get('status', 'OK')} |",
            f"| alert_count | {status.get('health_report', {}).get('alert_count', 0)} |",
            f"| cron_ledger_keys | {cron.get('unique_keys', 0)} |",
            f"| cron_ledger_last_seen | {cron.get('last_seen_at', 0)} |",
            "",
        ])
    if ns.command == "cron-ledger":
        import os
        from agent.apex_runtimeos_autonomy import summarize_cron_dryrun_ledger
        previous_repair = os.environ.get("APEX_RUNTIMEOS_CRON_LEDGER_REPAIR_ENABLED")
        if ns.repair:
            os.environ["APEX_RUNTIMEOS_CRON_LEDGER_REPAIR_ENABLED"] = "1"
        try:
            ledger = summarize_cron_dryrun_ledger(limit=limit)
        finally:
            if ns.repair:
                if previous_repair is None:
                    os.environ.pop("APEX_RUNTIMEOS_CRON_LEDGER_REPAIR_ENABLED", None)
                else:
                    os.environ["APEX_RUNTIMEOS_CRON_LEDGER_REPAIR_ENABLED"] = previous_repair
        if ns.json:
            return json.dumps({"object": "hermes.apex_runtimeos.cron_dryrun_ledger", "ledger": ledger}, ensure_ascii=False, indent=2)
        return "\n".join([
            "# APEX RuntimeOS Cron Dry-run 账本",
            "",
            "| 字段 | 数值 |",
            "|---|---:|",
            f"| ledger_exists | {ledger.get('ledger_exists')} |",
            f"| unique_keys | {ledger.get('unique_keys')} |",
            f"| bad_lines | {ledger.get('bad_lines')} |",
            f"| repair_enabled | {ledger.get('repair_enabled')} |",
            f"| quarantine_exists | {ledger.get('quarantine_exists')} |",
            f"| last_seen_at | {ledger.get('last_seen_at')} |",
            "",
        ])
    if ns.command in {"autopromote", "rollback"}:
        import os
        from agent.apex_runtimeos_autonomy import execute_promotion_rollbacks, run_autopromotion_scheduler
        previous_auto = os.environ.get("APEX_RUNTIMEOS_AUTOPROMOTE_ENABLED")
        previous_rollback = os.environ.get("APEX_RUNTIMEOS_ROLLBACK_ENABLED")
        if ns.execute:
            if ns.command == "autopromote":
                os.environ["APEX_RUNTIMEOS_AUTOPROMOTE_ENABLED"] = "1"
            else:
                os.environ["APEX_RUNTIMEOS_ROLLBACK_ENABLED"] = "1"
        try:
            if ns.command == "autopromote":
                result = run_autopromotion_scheduler(target=ns.target if ns.target != "all" else "memory", limit=limit, min_occurrences=max(1, int(ns.min_occurrences)))
            else:
                result = execute_promotion_rollbacks(target=ns.target, dry_run=not ns.execute, limit=limit)
        finally:
            if previous_auto is None:
                os.environ.pop("APEX_RUNTIMEOS_AUTOPROMOTE_ENABLED", None)
            else:
                os.environ["APEX_RUNTIMEOS_AUTOPROMOTE_ENABLED"] = previous_auto
            if previous_rollback is None:
                os.environ.pop("APEX_RUNTIMEOS_ROLLBACK_ENABLED", None)
            else:
                os.environ["APEX_RUNTIMEOS_ROLLBACK_ENABLED"] = previous_rollback
        if ns.json:
            return json.dumps({"object": f"hermes.apex_runtimeos.{ns.command}", "result": result}, ensure_ascii=False, indent=2)
        return "\n".join([
            f"# APEX RuntimeOS {ns.command}",
            "",
            f"| 字段 | 数值 |",
            "|---|---:|",
            f"| execute | {bool(ns.execute)} |",
            f"| target | {ns.target} |",
            f"| promoted | {result.get('promoted', 0)} |",
            f"| rolled_back | {result.get('rolled_back', 0)} |",
            f"| skipped | {result.get('skipped', 0)} |",
            f"| reason | {result.get('reason', '')} |",
            "",
        ])
    summary = summarize_audit(limit=limit)
    summary.pop("audit_path", None)
    try:
        from agent.apex_runtimeos_autonomy import summarize_autonomy_status
        summary["autonomy"] = summarize_autonomy_status(limit=limit, min_occurrences=max(1, int(ns.min_occurrences)))
    except Exception:
        summary["autonomy"] = {"health_report": {"status": "OK", "alert_count": 0, "alerts": []}, "cron_dryrun": {}}
    if ns.json:
        return json.dumps({"object": "hermes.apex_runtimeos.audit_summary", "summary": summary}, ensure_ascii=False, indent=2)
    if ns.command == "feishu":
        return _feishu_markdown(summary)
    _ = render_markdown(summary)
    return _cn_markdown(summary)
