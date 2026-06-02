---
name: paper-search
description: Ranked paper search with bilingual Feishu quick view.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [paper, search, literature, semantic-scholar, openalex, feishu, arxiv]
    category: research
    related_skills: [arxiv, literature-review, kanban-paper-nexus]
---

# Paper Literature Search — 高价值论文检索 → 飞书

**入口：`/paper-search <主题或类别>`**（必须 `/` 触发）。

按 **相关性 + 引用 + 高影响引用 + 时效 + 可下载** 综合排序，把 Top 论文列表推到飞书 IM。候选主来源为 **Semantic Scholar + OpenAlex**，不足时用 arXiv 补齐。默认输出是 **双语 IM 快览**：前 5 篇给中文题意 + 英文原题 + 中英一句话导读，剩余候选用简表折叠。**不**默认创建飞书在线文档、**不**走 Kanban 精读流水线。

## 打分方案（默认）

见 `references/scoring-rubric.md`。简述：

| 信号 | 权重 |
|------|------|
| Semantic Scholar 相关性 | 35% |
| 被引（批内归一化） | 30% |
| 高影响引用 | 15% |
| 发表年份（ML 偏新） | 15% |
| 有 PDF/arXiv | 5% |

用户说「最新 / SOTA」→ 编排加 `--boost-recency`；「经典 / 必读」→ `--min-citations 50`。

## When to Use

- `/paper-search 多模态大模型`
- `/paper-search RLHF alignment 2024`
- `/paper-search 图神经网络 综述`（加 `--profile survey`）

**不用本 skill：** 单篇精读、要飞书 doc → `/kanban-paper-nexus`；要股票 → `/kanban-stock-nexus`。

## Prerequisites

- `lark-cli` 已登录；Gateway 飞书 DM 有 `HERMES_SESSION_CHAT_ID`
- 网络可访问 Semantic Scholar、OpenAlex、arXiv
- 可选 `SEMANTIC_SCHOLAR_API_KEY`（降低 429、带引用数排序）
- 可选 `OPENALEX_API_KEY`（进入 polite pool / 提高 OpenAlex 稳定性）
- 可选 `OPENALEX_MAILTO` / `CROSSREF_MAILTO`（学术 API 联系邮箱）

## Procedure（编排 / 单轮 agent）

1. 解析检索式（保留用户原话，可补 1 行英文关键词）
2. 一键闭环（推荐）：
```bash
python3 skills/research/paper-literature-search/scripts/paper_search_pipeline.py \
  "<检索式>" --chat-id "$HERMES_SESSION_CHAT_ID" --top 8
```
3. 或分步：`paper_search_rank.py` → 检查 JSON → `paper_search_feishu_deliver.py --json-in`
4. 若 deliver stdout 返回 `no_hit=true` / `next_action=stop_no_manual_fallback`：**到此为止**，只告诉用户“当前论文库未命中精确候选”，建议缩检索式；**不要**继续 broaden、不要自己手工补榜单。
5. 若有命中：回复用户已推送 Top N；精读某篇用 `/kanban-paper-nexus <arxiv_id>`

## 飞书实时（与 paper-nexus 同款）

`kanban-feishu-live` board=`paper-search`：进度 3 条 + 结果 1 条 **双语快览消息**。见 `references/feishu-delivery.md`。

## Forbidden

- `docs +create` / `paper_feishu_doc_sync`
- `kanban_create` 全套 T0–T6（本 skill 无看板精读）
- 把 PDF/摘要全文贴进 IM
- 在 `no_hit=true` 后继续 `web_search` / `browser_*` / `send_message` 手工补榜单
- 在飞书 IM 里输出 LaTeX / MathJax（如 `$\\pi$`）；一律改成纯文本 `pi` 或 `π`

## Verification

```bash
bash scripts/paper_search_lark_e2e.sh
scripts/run_tests.sh tests/skills/test_paper_literature_search_skill.py -q
```

## Pitfalls

| 问题 | 处理 |
|------|------|
| S2 限流 | 降 `--candidate-limit`；仍不足靠 arXiv 补候选 |
| 中文查询命中少 | 检索式附英文同义词 |
| 结果太旧 | `--boost-recency` 或 `--year-floor 2023` |
