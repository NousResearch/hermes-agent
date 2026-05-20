---
name: kanban-paper-nexus
description: Paper-to-doc Kanban DAG on Hermes; Feishu via lark-cli.
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, paper, arxiv, research, feishu, lark-cli]
    category: research
    related_skills: [kanban, kanban-orchestrator, kanban-worker, arxiv, lark-doc, literature-review]
---

# Kanban Paper Nexus — 论文精读 → 可决策文档

**入口：`/kanban-paper-nexus <arXiv ID 或 URL>`**（必须 `/` 触发；普通聊天不自动进板）。

编排者 **只建卡**。工人按研究员框架产出 **CEL 主张表 + 实验审计 + 中英双语飞书 doc**（见 `references/paper-reading-framework.md`）。

## 设计原则（研究员视角）

1. **先论点，后段落** — T0 一句话 Thesis；T1 填 Claims–Evidence–Limits，禁止只翻译 Abstract。
2. **证据分级** — 强/中/弱；无 Fig/Table/§ 指针的主张不算完成。
3. **实验可审计** — T4 用五问检查 baseline/指标/方差/消融/复现，不是复述表格。
4. **文档服务决策** — 中文写清「能否信、能否用、下一步」；English 为 mirror。
5. **一篇论文一篇 doc** — registry 管 canonical id；同论文只 append（见下）。

## 文档 1:1 规则

- 新 `canonical_id` → `docs +create`（标题含 `[{canonical_id}]`）
- 同论文（`2402.03300` ≈ `2402.03300v3`）→ 仅 `append`
- 登记：`~/.hermes/kanban/boards/paper-nexus/paper_doc_registry.json`
- 同步：`paper_feishu_doc_sync.py`（T4 / E2E）

飞书 IM：**一条**中文摘要 + doc 链接（新建/追加二选一）。

## When to Use

- `/kanban-paper-nexus 2402.03300`
- `/kanban-paper-nexus https://arxiv.org/abs/1706.03762`

不用本 skill：单次问答、只要 bib、不落地飞书文档。

## Prerequisites

- 看板 `paper-nexus`；gateway + `kanban.dispatch_in_gateway: true`
- Workers：`kanban-researcher`（T0–T3）、`kanban-coder`（T4）、`kanban-writer`（T5）、`kanban-qa`（T6）
- `lark-cli`、`arxiv` / `web_extract`（深度读 PDF 时）
- 深度阅读可选：`references/paper-reading-framework.md` + PaperQA

## Parse User Input

| 字段 | 规则 |
|------|------|
| `paper_id` | arXiv id 或 URL |
| `canonical_id` | 去 `vN` 后缀 |
| `deep` | 含「深度」「精读」「full pdf」→ T0/T1 必须 `web_extract` PDF，CEL ≥5 行 |
| `feishu_doc` | `paper_doc_registry.py resolve` |
| `idempotency_key` | `paper-{canonical_id}-{YYYYMMDD}` |

```bash
python3 skills/research/kanban-paper-nexus/scripts/paper_nexus_metadata.py <id>
python3 skills/research/kanban-paper-nexus/scripts/paper_doc_registry.py resolve <id>
```

## Fixed DAG（`[paper]` 前缀不可改）

| 卡 | 标题 | assignee | parents | 产出 |
|----|------|----------|---------|------|
| T0 | `[paper] {id} 论点与阅读地图` | kanban-researcher | — | thesis_one_liner, reading_map |
| T1 | `[paper] {id} 主张-证据链 CEL` | kanban-researcher | T0 | CEL 表 ≥3 行 |
| T2 | `[paper] {id} 方法与复现要点` | kanban-researcher | T0 | 数据/算法/复现清单 |
| T3 | `[paper] {id} 对标与开源地图` | kanban-researcher | T0 | 相关论文 + OSS |
| T4 | `[paper] {id} 实验审计与局限` | kanban-coder | T1,T2 | 五问审计表 |
| T5 | `[paper] {id} 飞书精读文档` | kanban-writer | T1–T4 | `paper_feishu_doc_sync` |
| T6 | `[paper] {id} QA 门禁` | kanban-qa | T5 | `references/qa-rubric.md` |

T2∥T3 可并行；T4 必须等 T1+T2。

## Orchestrator Procedure

1. `skill_view` 本 skill + `kanban-orchestrator`
2. `hermes kanban boards switch paper-nexus`
3. 解析 `paper_id`、`deep`；`resolve` 告知将 **create** 或 **append** doc
4. `kanban_create` 全表；`parents` 用返回的 task_id
5. 确认 `notify_subscribed: true`
6. 回复（中文）：canonical_id、task_ids、doc 策略、arXiv/PDF、预计 T5 交付

## Worker 必读

| 文档 | 阶段 |
|------|------|
| `paper-reading-framework.md` | T0–T5 |
| `feishu-doc-bilingual-template.md` | T5 |
| `paper-kanban-pipeline.md` | 全流程 + handoff schema |
| `qa-rubric.md` | T6 |

`handoff.json` 必须含：`canonical_id`, `stage`, `thesis_one_liner`, `claims[]`, `feishu_doc_url`。

## Forbidden

- 编排器写正文、调 `lark-cli`、读完全文 PDF
- 无 Evidence 列的 CEL、全是「强」无局限
- 跨论文共用 doc、同论文重复 create
- `feishu-finance-kanban` / `kanban-stock-nexus`

## Verification

```bash
hermes kanban --board paper-nexus list
scripts/paper_kanban_lark_e2e.sh
scripts/run_tests.sh tests/skills/test_kanban_paper_nexus_skill.py -q
```

## Pitfalls

| 问题 | 处理 |
|------|------|
| 读后感式摘要 | 改填 CEL；见 framework |
| 文档与 handoff 数字不一致 | T5 只写 handoff 已核实数字 |
| 飞书双消息 | E2E/脚本连跑两次；每轮一条 IM |
| 同论文第二 doc | 必须 `resolve`→update |
