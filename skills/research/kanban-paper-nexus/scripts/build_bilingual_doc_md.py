#!/usr/bin/env python3
"""Build bilingual (ZH + EN) Feishu markdown skeleton from paper metadata (generic)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _canonical_id(paper_id: str) -> str:
    pid = (paper_id or "").strip()
    if pid.lower().startswith("s2:"):
        return pid.lower()
    return re.sub(r"v\d+$", "", pid, flags=re.I)


def _zh_title_short(title: str, max_len: int = 48) -> str:
    t = title.replace("\n", " ").strip()
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    m = re.split(r"(?<=[.!?。！？])\s+", text, maxsplit=1)
    return m[0] if m else text[:200]


def build(meta: dict, marker: str = "", *, stage_notes: str = "") -> str:
    """Generic skeleton — workers replace 【待填】 sections via pipeline handoffs."""
    title = meta.get("title_zh") or meta["title"]
    pid = meta.get("paper_id") or meta.get("canonical_id") or ""
    cid = meta.get("canonical_id") or _canonical_id(pid)
    link = meta.get("arxiv_abs") or meta.get("s2_url") or "—"
    pdf = meta.get("arxiv_pdf") or "—"
    doi = meta.get("doi") or "—"
    venue = meta.get("venue") or "—"
    authors = ", ".join(meta["authors"][:6])
    if len(meta["authors"]) > 6:
        authors += " et al."
    cats = ", ".join(meta.get("categories") or [])[:120]
    summary_zh = meta["summary"].strip()
    thesis_seed = _first_sentence(summary_zh)
    summary_en = summary_zh[:500] + ("…" if len(summary_zh) > 500 else "")

    notes_block = f"\n\n> {stage_notes}\n" if stage_notes else ""
    marker_row = f"| 批次 Run | {marker} |\n" if marker else ""

    return f"""# 论文精读 / Paper Brief：{_zh_title_short(title)}

| 项 Item | 内容 |
|---------|------|
| canonical_id | `{cid}` |
| 链接 Link | {link} |
| DOI | {doi} |
| 期刊 Venue | {venue} |
| 发表 Published | {meta['published']} |
| 领域 Categories | {cats or '—'} |
| 作者 Authors | {authors} |
| PDF | {pdf} |
| 流水线 Pipeline | `paper-nexus` · `/kanban-paper-nexus` |
{marker_row}{notes_block}
---

## 核心总结（30 秒读懂）/ Executive Summary

**中文（工人 T1/T5 填写，此处为种子句）：**
1. **论点：** {thesis_seed}（T0 应精炼为 ≤40 字）
2. **方法：** 【待填：数据 / 模型 / 训练或推理机制】
3. **结果：** 【待填：主基准 + 数字 + 设定】
4. **局限：** 【待填：证据弱点或外推风险】
5. **参考方向：** 【待填：复现 / 产品 / 对标 各 1 条】

**English:** 【待填：mirror bullets 2–4 lines each】

---

## 主张–证据–局限 / Claims–Evidence–Limits

| ID | 主张 (ZH) | 证据 Evidence | 强度 | 局限 |
|----|-----------|---------------|------|------|
| C1 | 【T1 填写】 | § / Fig / Table | 强/中/弱 | 【T1】 |

*English: One row summary after table is finalized.*

---

## 阅读地图 / Reading Map

**中文（T0）：** 【待填：Abstract → 图1 → Method § → 主实验表 → Appendix 要点】

*English: Section walk order.*

---

## 问题与动机 / Problem & Motivation

**中文：** 【T1/T2 填写】

*English: 【待填】*

---

## 方法与复现要点 / Method & Reproducibility

**中文：**
- 数据：【待填】
- 模型/算法：【待填】
- 复现清单：代码 / 权重 / 算力 / 关键超参 【待填】

*English: 【待填】*

---

## 对标与开源地图 / Related Work & Open Source

| 工作/项目 | 关系 | 为何相关 |
|-----------|------|----------|
| 【T3】 | 竞争/工具 | … |

---

## 实验审计 / Experiment Audit

| 检查项 | 结论 (ZH) |
|--------|-----------|
| Baseline 公平性 | 【T4】 |
| 指标对口 | 【T4】 |
| 方差/多次运行 | 【T4】 |
| 消融 | 【T4】 |
| 可复现性 | 【T4】 |

---

## 参考方向 / Where to Go Next

**中文（必填，可执行）：**
- **复现：** 【读哪一节、要什么资源】
- **产品化：** 【场景 + 风险】
- **对标：** 【2–3 篇后续工作或 repo】

**English:** 【mirror】

---

## 开源与工具 / Tools

| 项目 | 用途 | 链接 |
|------|------|------|
| PaperQA | PDF RAG | https://github.com/Future-House/paper-qa |
| Hermes arxiv | 元数据 | `skills/research/arxiv` |
| GROBID | PDF 结构 | https://github.com/kermitt2/grobid |

---

## 摘要摘录 / Abstract Excerpt

**中文：** {summary_zh[:700]}{'…' if len(summary_zh) > 700 else ''}

**English (placeholder):** {summary_en}
"""


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: build_bilingual_doc_md.py <metadata.json> [marker]", file=sys.stderr)
        return 2
    meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    marker = sys.argv[2] if len(sys.argv) > 2 else ""
    sys.stdout.write(build(meta, marker))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
