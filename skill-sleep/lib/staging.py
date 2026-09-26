"""Staging operations for skill-sleep REVIEW stage."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def now_ts() -> str:
    """UTC timestamp string safe for directory names: YYYYmmdd-HHMMSS."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def skill_slug(skill_path: str) -> str:
    """Derive a filesystem-safe slug from a skill path."""
    p = Path(skill_path)
    # e.g. ~/.hermes/skills/autonomous-ai-agents/hermes-agent/SKILL.md -> hermes-agent
    # fallback: parent dir name or stem
    if p.name == "SKILL.md" and p.parent.name:
        return p.parent.name
    if p.stem:
        return p.stem
    return "skill"


def staging_dir_name(skill_path: str, ts: str | None = None) -> str:
    slug = skill_slug(skill_path)
    ts = ts or now_ts()
    return f"{slug}-{ts}"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_file(src: str | Path, dst: str | Path) -> None:
    dst_p = Path(dst)
    ensure_dir(dst_p.parent)
    shutil.copy2(str(src), str(dst_p))


def append_rejected_jsonl(rejected_dir: str | Path, entry: dict) -> str:
    """Append one JSON line to rejected.jsonl under rejected_dir."""
    p = Path(rejected_dir)
    ensure_dir(p)
    jsonl = p / "rejected.jsonl"
    line = json.dumps(entry, ensure_ascii=False)
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return str(jsonl)


def build_review_md(
    *,
    ts: str,
    skill_path: str,
    staging_dir: str,
    validation: dict,
    proposal: dict | None,
    candidate_diff: str,
) -> str:
    overall = validation.get("overall_passed")
    passed_str = "PASS" if overall else "FAIL"
    total = validation.get("total_tasks", 0)
    passed_tasks = validation.get("passed_tasks", 0)
    pass_rate = validation.get("pass_rate", 0)
    threshold = validation.get("threshold", 70)
    # Try to derive an average score
    items = validation.get("items", [])
    avg_score = ""
    if items:
        try:
            avg = sum(int(it.get("score", 0)) for it in items) / len(items)
            avg_score = f"{avg:.1f}"
        except Exception:
            avg_score = ""
    summary = ""
    if proposal and proposal.get("summary"):
        summary = str(proposal["summary"])[:2000]
    else:
        summary = "(no summary — proposal.json missing or empty)"

    score_line = f"{avg_score}" if avg_score else "—"
    lines: list[str] = []
    lines.append("# Skill 审查请求")
    lines.append("")
    lines.append(f"- **时间**: {ts}")
    lines.append(f"- **目标 skill**: `{skill_path}`")
    lines.append(f"- **候选修改文件**: `candidate.diff`")
    lines.append(f"- **验证结果**: {passed_str}, 评分 {score_line}, 通过率 {pass_rate} ({passed_tasks}/{total}), 阈值 {threshold}")
    if validation.get("rejected_reason"):
        lines.append(f"- **拒绝原因**: {validation.get('rejected_reason')}")
    lines.append(f"- **修改摘要**: {summary}")
    lines.append("")
    lines.append("## Diff 内容")
    lines.append("")
    lines.append("```diff")
    lines.append(candidate_diff.rstrip() or "(empty diff)")
    lines.append("```")
    lines.append("")
    lines.append("## 验证明细")
    lines.append("")
    if items:
        for it in items:
            idx = it.get("task_index", "?")
            sc = it.get("score", "?")
            ps = "PASS" if it.get("passed") else "FAIL"
            rs = str(it.get("reason", ""))[:800]
            lines.append(f"- task {idx}: score={sc} {ps} — {rs}")
    else:
        lines.append("(no validation items)")
    lines.append("")
    if proposal and proposal.get("focused_on"):
        lines.append("## 关注点")
        lines.append("")
        for f in proposal["focused_on"][:10]:
            lines.append(f"- {f}")
        lines.append("")
    lines.append("## 采纳 / 拒绝")
    lines.append("")
    lines.append(f"- 采纳：`python3 pipeline/review.py apply --staging-dir {staging_dir} --skill {skill_path}`")
    lines.append(f"- 拒绝：`python3 pipeline/review.py reject --staging-dir {staging_dir} --reason \"...\"`")
    lines.append("")
    return "\n".join(lines)
