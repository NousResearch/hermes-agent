---
name: kanban-paper-nexus
description: Paper-to-doc Kanban DAG on Hermes; Feishu via lark-cli.
version: 1.5.0
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

**入口：`/kanban-paper-nexus <论文链接或 ID>`**（arXiv、DOI、OpenAlex 或 Semantic Scholar；必须 `/` 触发）。

### 禁止自行否决（编排硬规则）

1. **必须先跑** `paper_nexus_metadata.py <用户输入>`；退出码 0 → **必须**继续建板（`kanban_create` 等），**禁止**在未跑脚本前声称「无 arXiv 无法进流水线」。
2. 无 arXiv 时优先取 `s2:<40位hash>`；若 S2 临时不可用则退化为 `doi:<doi>`，也必须继续建板并打通飞书实时。
3. PDF 来源：优先 `arxiv_pdf`；否则 S2 `openAccessPdf` / `s2_url` / DOI 落地页，T0/T1 用 `web_extract`。
4. 仅当脚本 **非 0 退出** 或 API 持续失败时，才向用户报告失败并给替代方案。
5. **禁止凭对话上下文臆断“已有流程”**。在声称 workflow 已存在、或声称 “T0–T4 已完成 / T5 ready” 前，必须先跑 `paper_nexus_status.py <paper_id>` 读取真实看板状态；若 `exists=false`，就按新论文建卡，不能口头编造现成流程。

编排者 **只建卡**。工人按研究员框架产出 **CEL 主张表 + 实验审计 + 中英双语飞书 doc**（见 `references/paper-reading-framework.md`）。

## 设计原则（研究员视角）

1. **先论点，后段落** — T0 一句话 Thesis；T1 填 Claims–Evidence–Limits，禁止只翻译 Abstract。
2. **证据分级** — 强/中/弱；无 Fig/Table/§ 指针的主张不算完成。
3. **实验可审计** — T4 用五问检查 baseline/指标/方差/消融/复现，不是复述表格。
4. **文档服务决策** — 中文写清「能否信、能否用、下一步」；English 为 mirror。
5. **方法必须可迁移** — 若论文含公式/目标函数，T5 必须补“核心公式 + 符号释义”；同时补“运行时策略细节 + 边界分析 + 软硬件 Delta 对照表”。
6. **一篇论文一篇 doc** — registry 管 canonical id；同论文只 append（见下）。
7. **Memory OS 可追溯** — 跨会话用 `workflow_id: paper-nexus:<canonical_id>` 回忆 CEL、doc、QA 结论（见 `references/memory-os.md`）。

## Memory OS（跨会话记忆）

| 时机 | 工具 | 说明 |
|------|------|------|
| 建卡前 | `search_memory` | **仅** canonical_id + 论文名（`paper_memory_search_query.py`，`limit=3`） |
| 每阶段完成 | `store_memory_markdown` | 用 `paper_memory_markdown.py` 生成 entry |
| 可选背景 | `search_existing_knowledge` | 公司 wiki 是否已有该主题（不替代 CEL） |

```bash
python3 skills/research/kanban-paper-nexus/scripts/paper_memory_markdown.py \
  --stage T5 --handoff handoff.json --task-id t_xxx
```

**workflow_id：** `paper-nexus:2402.03300`（canonical，无 `vN`）。详表见 `references/memory-os.md`。

## 文档 1:1 规则

- 新 `canonical_id` → `lark-cli docs +create --api-version v2`；**在线文档名**=`[{canonical_id}] {中文题名}`（`title_zh`，非英文 arXiv 标题）
- 同论文（`2402.03300` ≈ `2402.03300v3`）→ 仅 `append`
- 登记：`~/.hermes/kanban/boards/paper-nexus/paper_doc_registry.json`
- 同步：`paper_feishu_doc_sync.py`（T4 / E2E）

飞书 IM：**阶段短更新**（`paper_feishu_stage_notify.py`，对齐 finance-nexus 实时）+ T5 后 doc 链接；长文只进 doc，不进 IM。

## When to Use

- `/kanban-paper-nexus 2402.03300`
- `/kanban-paper-nexus 10.1126/scirobotics.aau4984`
- `/kanban-paper-nexus https://openalex.org/W2741809807`
- `/kanban-paper-nexus https://arxiv.org/abs/1706.03762`
- `/kanban-paper-nexus https://www.semanticscholar.org/paper/ceced53f...`（无 arXiv 时用 `s2:<id>` 作 canonical）

不用本 skill：单次问答、只要 bib、不落地飞书文档。

## Prerequisites

- 看板 `paper-nexus`；gateway + `kanban.dispatch_in_gateway: true`
- Workers：`kanban-researcher`（T0–T3）、`kanban-coder`（T4）、`kanban-writer`（T5）、`kanban-qa`（T6）
- `lark-cli`、`arxiv` / `web_extract`（深度读 PDF 时）
- **`unified-memory-os` MCP**（`search_memory` + `store_memory_markdown`）
- 深度阅读可选：`references/paper-reading-framework.md` + PaperQA
- 可选 `OPENALEX_API_KEY`（OpenAlex polite pool / OA enrichment）
- 可选 `OPENALEX_MAILTO` / `CROSSREF_MAILTO`（DOI / OpenAlex 联系邮箱）

## Parse User Input

| 字段 | 规则 |
|------|------|
| `paper_id` | arXiv id/URL、DOI/doi.org URL、OpenAlex work URL/id，或 Semantic Scholar 论文链接/40 位 corpus id |
| `canonical_id` | arXiv 去 `vN`；优先 `s2:<hash>`；其后允许 `doi:<doi>` 或 `openalex:<wid>`（见 metadata JSON） |
| `deep` | 含「深度」「精读」「full pdf」→ T0/T1 必须 `web_extract` PDF，CEL ≥5 行 |
| `feishu_doc` | `paper_doc_registry.py resolve` |
| `idempotency_key` | `paper-{canonical_id}-{YYYYMMDD}` |

```bash
python3 skills/research/kanban-paper-nexus/scripts/paper_nexus_metadata.py <id>
python3 skills/research/kanban-paper-nexus/scripts/paper_doc_registry.py resolve <id>
python3 skills/research/kanban-paper-nexus/scripts/paper_nexus_metadata.py <id> > /tmp/paper-meta.json
python3 skills/research/kanban-paper-nexus/scripts/paper_memory_search_query.py <id> --meta-json /tmp/paper-meta.json
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
2. `paper_nexus_metadata.py <id> > /tmp/paper-meta.json` → `paper_memory_search_query.py <id> --meta-json /tmp/paper-meta.json` → `search_memory`（**禁止全文 query**；`--meta-json` 必填，避免重复打 S2 API）
3. `hermes kanban boards switch paper-nexus`
4. 解析 `paper_id`、`deep`；`resolve` 告知将 **create** 或 **append** doc
5. 先跑 `paper_nexus_status.py <paper_id>`：`exists=true` 才允许总结现有 workflow；`exists=false` 才进入建卡分支
6. `kanban_create` 全表；`parents` 用返回的 task_id
7. **飞书实时（必做）** — 优先使用 `paper_feishu_live_init.py` 一次完成 `init + subscribe + pipeline_started`，见 `references/feishu-live-updates.md`
8. `store_memory_markdown`（stage=orchestrator，含 task_ids、doc 策略）
9. 回复（中文）：canonical_id、task_ids、doc 策略、Memory 是否命中、arXiv/PDF（说明后续阶段完成会再推短 IM）

## Worker 必读

| 文档 | 阶段 |
|------|------|
| `paper-reading-framework.md` | T0–T5 |
| `feishu-doc-bilingual-template.md` | T5 |
| `paper-kanban-pipeline.md` | 全流程 + handoff schema |
| `memory-os.md` | 何时 search/store、workflow_id |
| `feishu-live-updates.md` | 每阶段 `lark-cli` 推送（finance 同款，无 core patch） |
| `qa-rubric.md` | T6 |

`handoff.json` 必须含：`canonical_id`, `title_zh`（论文中文名，用于飞书在线文档名）, `stage`, `thesis_one_liner`, `claims[]`, `feishu_doc_url`。

T5 写文档时，`docs +create / +fetch / +update` **一律显式带** `--api-version v2`；不要依赖 CLI 默认值。
T5 创建/追加飞书文档时，**优先且默认** 调 `scripts/paper_feishu_doc_sync.py --handoff handoff.json`；不要直接手写 `lark-cli docs +create ...` 生成整篇正文。

## Forbidden

- 编排器写正文、调 `lark-cli`、读完全文 PDF
- 无 Evidence 列的 CEL、全是「强」无局限
- 跨论文共用 doc、同论文重复 create
- `feishu-finance-kanban` / `kanban-stock-nexus`
- 只写 Kanban/SQLite 不写 Memory OS（跨会话会丢）

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
| 飞书双消息 | E2E/脚本连跑两次；每阶段 notify 一条，勿手写第二遍长文 |
| 无阶段推送 | 编排未 `init` / 工人未 `notify stage_done` | 见 `feishu-live-updates.md` |
| 同论文第二 doc | 必须 `resolve`→update |
