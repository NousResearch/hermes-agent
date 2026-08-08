#!/usr/bin/env python3
"""Compliant social-platform comment analysis and Hermes handoff pipeline.

This script intentionally starts from user-authorized exports / official API dumps
(JSON/JSONL/CSV files). It does not bypass login walls, CAPTCHA, anti-bot controls,
or platform rate limits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TEXT_KEYS = ("text", "content", "comment", "body", "message", "评论", "内容")
ID_KEYS = ("comment_id", "id", "cid", "评论ID")
POST_KEYS = ("post_id", "aweme_id", "note_id", "video_id", "thread_id", "帖子ID")
AUTHOR_KEYS = ("author", "user", "nickname", "username", "用户", "昵称")
TIME_KEYS = ("created_at", "time", "timestamp", "date", "发布时间")
URL_KEYS = ("url", "link", "permalink", "链接")
PLATFORM_KEYS = ("platform", "source", "site", "平台")

INTENT_KEYWORDS = {
    "feature_request": ["希望", "能不能", "可不可以", "建议", "增加", "支持", "想要", "需求", "最好", "希望可以", "need", "want", "feature", "support"],
    "bug_report": ["bug", "崩", "闪退", "报错", "打不开", "失败", "卡住", "不能用", "异常", "错误", "crash", "error", "fail", "broken"],
    "pain_point": ["麻烦", "太慢", "不好用", "复杂", "难用", "贵", "费劲", "痛点", "不方便", "slow", "hard", "expensive", "annoying"],
    "praise": ["好用", "喜欢", "不错", "赞", "稳定", "流畅", "推荐", "love", "great", "nice", "awesome"],
    "question": ["怎么", "如何", "为什么", "吗", "?", "？", "how", "why", "what", "when"],
}

AREA_KEYWORDS = {
    "登录/账号": ["登录", "注册", "账号", "密码", "验证码", "权限", "login", "account", "password"],
    "支付/价格": ["价格", "付费", "会员", "订阅", "退款", "优惠", "贵", "payment", "price", "subscribe"],
    "性能/稳定性": ["慢", "卡", "闪退", "崩", "延迟", "加载", "crash", "slow", "lag"],
    "内容/推荐": ["推荐", "内容", "搜索", "筛选", "排序", "标签", "search", "recommend"],
    "通知/消息": ["通知", "消息", "提醒", "私信", "推送", "notification", "message"],
    "界面/交互": ["界面", "按钮", "入口", "操作", "交互", "UI", "UX", "页面", "看不懂"],
    "数据/导出": ["数据", "导出", "报表", "同步", "备份", "csv", "excel", "api"],
}

ROLE_TEMPLATES = {
    "product_manager": "梳理用户需求、定义验收标准、排优先级，并维护需求归档。",
    "developer": "基于产品需求完成最小可用实现，提交可运行代码和自测记录。",
    "tester": "设计测试用例，执行功能/回归/边界测试，输出缺陷与风险。",
    "acceptance": "按验收标准验证最终交付，确认是否可发布或退回修改。",
}


@dataclass
class Comment:
    platform: str
    post_id: str
    comment_id: str
    author: str
    text: str
    created_at: str
    url: str
    source_file: str


@dataclass
class Insight:
    title: str
    intent: str
    area: str
    score: int
    evidence_count: int
    examples: list[str]
    suggested_owner: str
    acceptance: list[str]


def _first(record: dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _stable_id(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("comments", "data", "items", "records", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [data]
    return []


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是合法 JSONL: {exc}") from exc
        if isinstance(item, dict):
            records.append(item)
    return records


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl_records(path)
    if suffix == ".json":
        return _read_json_records(path)
    if suffix == ".csv":
        return _read_csv_records(path)
    return []


def normalize_record(record: dict[str, Any], source_file: Path) -> Comment | None:
    text = re.sub(r"\s+", " ", _first(record, TEXT_KEYS)).strip()
    if not text:
        return None
    platform = _first(record, PLATFORM_KEYS, source_file.stem.split("_")[0] or "unknown")
    post_id = _first(record, POST_KEYS, "unknown-post")
    comment_id = _first(record, ID_KEYS) or _stable_id(platform, post_id, text)
    return Comment(
        platform=platform,
        post_id=post_id,
        comment_id=comment_id,
        author=_first(record, AUTHOR_KEYS, "unknown"),
        text=text,
        created_at=_first(record, TIME_KEYS),
        url=_first(record, URL_KEYS),
        source_file=str(source_file),
    )


def load_comments(input_path: Path) -> list[Comment]:
    paths = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in {".json", ".jsonl", ".csv"}
    )
    comments: list[Comment] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for record in read_records(path):
            comment = normalize_record(record, path)
            if not comment:
                continue
            key = (comment.platform, comment.post_id, comment.comment_id)
            if key in seen:
                continue
            seen.add(key)
            comments.append(comment)
    return comments


def classify(text: str) -> tuple[str, str, int]:
    lowered = text.lower()
    intent_scores = {
        intent: sum(1 for kw in kws if kw.lower() in lowered)
        for intent, kws in INTENT_KEYWORDS.items()
    }
    intent = max(intent_scores, key=lambda key: intent_scores[key])
    if intent_scores[intent] == 0:
        intent = "other"
    area_scores = {
        area: sum(1 for kw in kws if kw.lower() in lowered)
        for area, kws in AREA_KEYWORDS.items()
    }
    area = max(area_scores, key=lambda key: area_scores[key])
    if area_scores[area] == 0:
        area = "未分类"
    score = intent_scores.get(intent, 0) * 2 + area_scores.get(area, 0) + min(len(text) // 30, 3)
    return intent, area, max(score, 1)


def make_title(area: str, intent: str, examples: list[str]) -> str:
    intent_cn = {
        "feature_request": "新增/优化需求",
        "bug_report": "缺陷反馈",
        "pain_point": "体验痛点",
        "question": "用户疑问",
        "praise": "正向反馈",
        "other": "评论洞察",
    }.get(intent, intent)
    sample = examples[0] if examples else ""
    sample = re.sub(r"[，。！？!?].*$", "", sample)[:24]
    return f"{area}：{intent_cn}" + (f"（{sample}）" if sample else "")


def acceptance_for(intent: str, area: str) -> list[str]:
    base = [
        "有明确用户问题、目标用户和业务价值说明",
        "保留至少 2 条原始评论证据或说明证据不足",
        "定义可观察的成功指标或验收口径",
    ]
    if intent == "bug_report":
        base.append("能复现或说明无法复现的环境、步骤和日志/截图需求")
    elif intent == "feature_request":
        base.append("给出 MVP 范围、非目标范围和上线后验证指标")
    elif intent == "pain_point":
        base.append("给出当前流程痛点、改进方案和预期减少的用户成本")
    else:
        base.append(f"针对“{area}”给出下一步处理建议")
    return base


def analyze(comments: list[Comment], min_evidence: int = 1) -> list[Insight]:
    grouped: dict[tuple[str, str], list[tuple[Comment, int]]] = defaultdict(list)
    for comment in comments:
        intent, area, score = classify(comment.text)
        if intent == "praise":
            owner = "product_manager"
        elif intent == "question":
            owner = "product_manager"
        elif intent == "bug_report":
            owner = "developer"
        else:
            owner = "product_manager"
        grouped[(intent, area)].append((comment, score + (2 if owner == "developer" else 0)))

    insights: list[Insight] = []
    for (intent, area), rows in grouped.items():
        if len(rows) < min_evidence:
            continue
        rows.sort(key=lambda item: item[1], reverse=True)
        examples = [item[0].text for item in rows[:5]]
        total_score = sum(item[1] for item in rows) + min(len(rows), 10)
        insights.append(
            Insight(
                title=make_title(area, intent, examples),
                intent=intent,
                area=area,
                score=total_score,
                evidence_count=len(rows),
                examples=examples,
                suggested_owner="developer" if intent == "bug_report" else "product_manager",
                acceptance=acceptance_for(intent, area),
            )
        )
    insights.sort(key=lambda item: (item.score, item.evidence_count), reverse=True)
    return insights


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_escape(text: str) -> str:
    return text.replace("\n", " ").strip()


def render_insights(comments: list[Comment], insights: list[Insight]) -> str:
    by_platform = Counter(c.platform for c in comments)
    lines = [
        "# 社交平台评论需求洞察",
        "",
        f"生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"评论总数：{len(comments)}",
        "平台分布：" + "、".join(f"{k} {v}" for k, v in by_platform.most_common()) if by_platform else "平台分布：无",
        "",
        "## Top 洞察",
    ]
    for idx, insight in enumerate(insights, 1):
        lines.extend([
            "",
            f"### {idx}. {insight.title}",
            f"- 意图：{insight.intent}",
            f"- 领域：{insight.area}",
            f"- 优先级分：{insight.score}",
            f"- 证据数：{insight.evidence_count}",
            f"- 建议负责人：{insight.suggested_owner}",
            "- 代表评论：",
        ])
        for example in insight.examples:
            lines.append(f"  - {markdown_escape(example)}")
        lines.append("- 验收口径：")
        for item in insight.acceptance:
            lines.append(f"  - {item}")
    return "\n".join(lines) + "\n"


def render_pm_brief(insights: list[Insight]) -> str:
    lines = [
        "# 给产品经理 Agent 的需求归档包",
        "",
        "你的任务：基于评论洞察完成需求澄清、排期建议、任务拆分和验收标准维护。",
        "",
        "## 处理原则",
        "- 优先处理高频、高痛点、可验证的问题。",
        "- 保留原始评论作为证据，不要把单条吐槽放大成确定需求。",
        "- 对缺证据需求先补充调研任务，不直接进入开发。",
        "- 所有开发任务必须带验收标准和测试口径。",
        "",
        "## 候选需求",
    ]
    for idx, insight in enumerate(insights, 1):
        lines.extend([
            "",
            f"### PM-{idx:02d} {insight.title}",
            f"- 意图：{insight.intent}",
            f"- 领域：{insight.area}",
            f"- 优先级分：{insight.score}",
            f"- 证据数：{insight.evidence_count}",
            "- PM 下一步：确认问题定义、用户场景、MVP 范围、成功指标。",
            "- 验收标准：",
        ])
        for item in insight.acceptance:
            lines.append(f"  - {item}")
    return "\n".join(lines) + "\n"


def task_body(role: str, insight: Insight, idx: int) -> str:
    examples = "\n".join(f"- {markdown_escape(e)}" for e in insight.examples)
    acceptance = "\n".join(f"- [ ] {a}" for a in insight.acceptance)
    return f"""角色：{role}
职责：{ROLE_TEMPLATES[role]}

关联需求：REQ-{idx:02d} {insight.title}
意图：{insight.intent}
领域：{insight.area}
优先级分：{insight.score}
证据数：{insight.evidence_count}

代表评论：
{examples}

验收标准：
{acceptance}

工作要求：
- 若信息不足，先输出澄清问题或调研任务。
- 不绕过平台登录、验证码、风控或限流；采集只使用官方 API、导出文件或用户授权数据。
- 交付物必须可验证：代码、测试结果、文档路径或验收记录。
"""


def write_task_packages(out_dir: Path, insights: list[Insight], limit: int) -> list[dict[str, Any]]:
    tasks_dir = out_dir / "agent_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for idx, insight in enumerate(insights[:limit], 1):
        for role in ("product_manager", "developer", "tester", "acceptance"):
            title = f"[{role}] REQ-{idx:02d} {insight.title}"
            body = task_body(role, insight, idx)
            file_name = f"REQ-{idx:02d}-{role}.md"
            (tasks_dir / file_name).write_text(f"# {title}\n\n{body}", encoding="utf-8")
            tasks.append({
                "title": title,
                "body": body,
                "role": role,
                "requirement_id": f"REQ-{idx:02d}",
                "priority": insight.score,
                "file": str(tasks_dir / file_name),
            })
    (out_dir / "kanban_tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return tasks


def run_kanban_create(task: dict[str, Any], *, board: str | None, workspace: str, dry_run: bool) -> dict[str, Any]:
    cmd = ["hermes", "kanban"]
    if board:
        cmd += ["--board", board]
    cmd += [
        "create",
        task["title"],
        "--body", task["body"],
        "--assignee", task["role"],
        "--workspace", workspace,
        "--priority", str(task.get("priority", 0)),
        "--idempotency-key", _stable_id(task["title"], task["body"]),
        "--json",
    ]
    if dry_run:
        return {"dry_run": True, "command": cmd}
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
    return {"command": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def dispatch_tasks(tasks: list[dict[str, Any]], out_dir: Path, *, board: str | None, workspace: str, dry_run: bool) -> None:
    results = [run_kanban_create(task, board=board, workspace=workspace, dry_run=dry_run) for task in tasks]
    (out_dir / "kanban_dispatch_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def build_run_dir(output_dir: Path, run_name: str | None) -> Path:
    stamp = run_name or datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    comments = load_comments(Path(args.input))
    out_dir = build_run_dir(Path(args.output), args.run_name)
    write_jsonl(out_dir / "normalized_comments.jsonl", (asdict(c) for c in comments))
    insights = analyze(comments, min_evidence=args.min_evidence)
    (out_dir / "insights.json").write_text(json.dumps([asdict(i) for i in insights], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "insights.md").write_text(render_insights(comments, insights), encoding="utf-8")
    (out_dir / "product_manager_brief.md").write_text(render_pm_brief(insights), encoding="utf-8")
    tasks = write_task_packages(out_dir, insights, args.max_requirements)
    if args.dispatch_kanban or args.dry_run_kanban:
        dispatch_tasks(tasks, out_dir, board=args.board, workspace=args.workspace, dry_run=not args.dispatch_kanban)
    summary = {
        "output_dir": str(out_dir),
        "comment_count": len(comments),
        "insight_count": len(insights),
        "task_count": len(tasks),
        "dispatched": bool(args.dispatch_kanban),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze social comment exports and hand off tasks to Hermes agents.")
    parser.add_argument("--input", required=True, help="Input export file or directory (.json/.jsonl/.csv).")
    parser.add_argument("--output", default="artifacts/social-comment-runs", help="Archive output directory.")
    parser.add_argument("--run-name", help="Deterministic run directory name, useful for tests/cron.")
    parser.add_argument("--min-evidence", type=int, default=1, help="Minimum comments per insight cluster.")
    parser.add_argument("--max-requirements", type=int, default=5, help="How many top insights become agent task packages.")
    parser.add_argument("--dry-run-kanban", action="store_true", help="Write kanban create commands without executing them.")
    parser.add_argument("--dispatch-kanban", action="store_true", help="Actually create Hermes Kanban tasks.")
    parser.add_argument("--board", help="Hermes Kanban board slug.")
    parser.add_argument("--workspace", default="scratch", help="Kanban workspace strategy: scratch, worktree, dir:<path>.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
