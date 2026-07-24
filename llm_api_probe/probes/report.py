"""报告生成: 控制台表格 + JSON + Markdown 三种输出。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import ProbeResult, Provider


PROBE_ORDER = ["connectivity", "context", "speed", "stability", "security", "ability"]


def _short_metric_value(metric) -> str:
    """精简指标值用于表格。"""
    name = metric.name
    val = metric.value
    # 限长
    if isinstance(val, str) and len(val) > 30:
        val = val[:27] + "..."
    return f"{val}{metric.unit}"


def render_console_table(
    providers: list[Provider],
    results: list[ProbeResult],
) -> str:
    """生成控制台对比表。"""
    by_pp: dict[tuple[str, str], ProbeResult] = {(r.provider, r.probe): r for r in results}

    lines: list[str] = []
    lines.append("=" * 88)
    lines.append(f"LLM API Probe — 报告  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append("=" * 88)

    # Provider 概览
    lines.append("\n## Provider 概览")
    lines.append(f"  {'name':<22} {'label':<22} {'category':<14} {'base_url'}")
    lines.append(f"  {'-' * 22} {'-' * 22} {'-' * 14} {'-' * 40}")
    for p in providers:
        lines.append(f"  {p.name:<22} {p.label[:22]:<22} {p.category:<14} {p.base_url}")

    # 按 probe 分块
    for probe in PROBE_ORDER:
        probe_results = [r for r in results if r.probe == probe]
        if not probe_results:
            continue
        lines.append(f"\n## {probe.upper()}")
        # 表头
        lines.append(f"  {'provider':<22} {'ok':<4} {'metrics'}")
        lines.append(f"  {'-' * 22} {'-' * 4} {'-' * 56}")
        for r in probe_results:
            ok_mark = "✓" if r.ok else "✗"
            metrics_str = "; ".join(f"{m.name}={_short_metric_value(m)}" for m in r.metrics[:6])
            if len(r.metrics) > 6:
                metrics_str += f" ... (+{len(r.metrics) - 6})"
            lines.append(f"  {r.provider:<22} {ok_mark:<4} {metrics_str}")
        # 警告 / 结论
        for r in probe_results:
            for w in r.warnings:
                lines.append(f"    ⚠ [{r.provider}] {w}")
            for f in r.findings:
                lines.append(f"    · [{r.provider}] {f}")
            if r.error:
                lines.append(f"    ✗ [{r.provider}] {r.error}")

    # 横向对比表 (只看 speed 模块的核心指标)
    speed_results = [r for r in results if r.probe == "speed" and r.ok]
    if len(speed_results) >= 2:
        lines.append("\n## 速度横向对比")
        lines.append(f"  {'provider':<22} {'TTFT中位':>10} {'输出tok/s中位':>16} {'并发吞吐':>14} {'成功率':>10}")
        lines.append(f"  {'-' * 22} {'-' * 10} {'-' * 16} {'-' * 14} {'-' * 10}")
        for r in speed_results:
            m = {x.name: x.value for x in r.metrics}
            lines.append(
                f"  {r.provider:<22} {m.get('ttft_median_ms', '-'):>10} "
                f"{m.get('output_tps_median', '-'):>16} "
                f"{m.get('concurrency_total_out_tps', '-'):>14} "
                f"{m.get('seq_success_rate', '-'):>10}"
            )

    # 能力横向对比
    ability_results = [r for r in results if r.probe == "ability" and r.ok]
    if len(ability_results) >= 1:
        lines.append("\n## 能力基线对比")
        lines.append(f"  {'provider':<22} {'总数':>6} {'通过':>6} {'通过率':>8}")
        lines.append(f"  {'-' * 22} {'-' * 6} {'-' * 6} {'-' * 8}")
        for r in ability_results:
            m = {x.name: x.value for x in r.metrics}
            lines.append(
                f"  {r.provider:<22} {m.get('total_questions', '-'):>6} "
                f"{m.get('total_pass', '-'):>6} {m.get('overall_pass_rate', '-'):>7}%"
            )

    # 安全审计警告汇总
    security_results = [r for r in results if r.probe == "security"]
    all_warnings: list[str] = []
    for r in security_results:
        for w in r.warnings:
            all_warnings.append(f"[{r.provider}] {w}")
    if all_warnings:
        lines.append("\n## ⚠ 安全审计警告汇总")
        for w in all_warnings:
            lines.append(f"  - {w}")
    else:
        lines.append("\n## ✓ 安全审计: 无异常")

    lines.append("\n" + "=" * 88)
    return "\n".join(lines)


def write_json(
    providers: list[Provider],
    results: list[ProbeResult],
    path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "providers": [p.to_dict() for p in providers],
        "results": [r.to_dict() for r in results],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_markdown(
    providers: list[Provider],
    results: list[ProbeResult],
    path: Path,
) -> None:
    """Markdown 报告, 适合贴到 wiki / PR。"""
    lines: list[str] = []
    lines.append(f"# LLM API Probe 报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Provider 概览\n")
    lines.append("| name | label | category | base_url | note |")
    lines.append("|------|-------|----------|----------|------|")
    for p in providers:
        lines.append(f"| `{p.name}` | {p.label} | {p.category} | {p.base_url} | {p.note} |")

    for probe in PROBE_ORDER:
        probe_results = [r for r in results if r.probe == probe]
        if not probe_results:
            continue
        lines.append(f"\n## {probe}\n")
        lines.append("| provider | ok | 指标 | 警告 | 结论 |")
        lines.append("|----------|----|----|------|------|")
        for r in probe_results:
            ok_mark = "✓" if r.ok else "✗"
            metrics_md = "<br>".join(
                f"`{m.name}`={_short_metric_value(m)}" for m in r.metrics
            )
            warnings_md = "<br>".join(f"⚠ {w}" for w in r.warnings) or "—"
            findings_md = "<br>".join(f"· {f}" for f in r.findings) or "—"
            lines.append(
                f"| `{r.provider}` | {ok_mark} | {metrics_md} | {warnings_md} | {findings_md} |"
            )

    path.write_text("\n".join(lines), encoding="utf-8")