---
name: feishu-finance-nexus
description: A-share Nexus DAG on finance-kanban with live Feishu + Base sync.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [finance, feishu, kanban, nexus, a-share]
    category: finance
    related_skills: [feishu-finance-kanban, china-stock-analysis, stock-monitoring]
prerequisites:
  commands: [python3, lark-cli]
---

# Feishu Finance Nexus

在 **`feishu-finance-kanban`**（外部 `financial-kanban` MCP）上跑 **A 股个股 Nexus 流水线**，解决 Hermes 内置 `kanban-stock-nexus` **派工慢、飞书不实时回帖** 的问题。

## 与 kanban-stock-nexus 的差异

| | `kanban-stock-nexus` | **本 skill** |
|---|----------------------|--------------|
| 引擎 | Hermes 内置 Kanban + gateway dispatcher | `financial-kanban` MCP + SQLite |
| 飞书回复 | 依赖 notify / 工人完成 | **每阶段** `render_feishu_card_update` → `send_message` |
| 股票池 Base | 可选 batch 脚本 | **`sync_symbol_to_stock_pool`** 聚合写入 |
| 触发 | `/kanban-stock-nexus` | `/feishu-finance-nexus 分析 002236` |

## When to Use

- `/feishu-finance-nexus 分析 002236`
- `/feishu-finance-nexus 深度分析 600519`
- 用户要在飞书**实时看到**各阶段进展，且结果写入**股票池多维表格**

不要用本 skill 做美股/港股（需 6 位 A 股代码）；全球标的用 `feishu-finance-kanban` 通用三卡（earnings/valuation/risk）。

## Prerequisites

1. MCP `financial_kanban` 已启用（`~/.hermes/config.yaml` → `mcp_servers.financial_kanban`）。
2. Base 同步环境变量已配置：`FINANCE_STOCK_POOL_SYNC=true` + `FINANCE_STOCK_POOL_*`。
3. 推荐 `FINANCE_STOCK_POOL_SYNC_MODE=artifact`（有 artifact 才写 Base，避免空行）。
4. 先 `skill_view feishu-finance-kanban` 了解 MCP 工具名。

## Fixed DAG（MCP 一次建好）

| 阶段 | work_type | 标题后缀 |
|------|-----------|----------|
| T0 | research | 上下文时间线 |
| T1 | research | 宏观/行业/资金 |
| T2 | valuation | 技术面/量价 |
| T3 | risk | 避雷审查 |
| T4 | risk | 规则校准 |
| T5 | trade-idea | 投资决策合成 |
| T6 | portfolio-review | QA门禁 |

`create_finance_nexus_analysis(symbol, chat_id=…)` 返回 7 张卡及 `workflow_id`。

## Procedure（编排 agent）

1. **建卡** — `create_finance_nexus_analysis`（带当前 `chat_id` / `thread_id`）。
2. **按 T0→T6 执行**（可并行 T1+T2，T3 等 T1/T2）：
   - `web_search` / `china-stock-analysis` 拉数据
   - `add_finance_artifact` 写入真实分析（非空）
   - `transition_finance_card` → `in_progress` / `done`
   - **`render_feishu_card_update` + `send_message`** — 每完成一阶段即发飞书（实时）
3. **收尾** — 全部完成后：
   - `sync_symbol_to_stock_pool(symbol=…)` — **聚合**写入股票池 Base（催化剂/估值/风险/评分）
   - `render_symbol_feishu_summary` + `send_message` — 总览
   - `store_memory_markdown` — `workflow_id: finance-kanban:<symbol>`
4. **硬避雷** — T3 若 ST/净利&lt;0/负债&gt;70% 等，`transition_finance_card` → `blocked`，Base 状态保持「观察中」。

分析脚本可参考 `stock-monitoring/references/nexus-kanban-pipeline.md`；本 skill **不**依赖 gateway 派工。

## Quick Reference

| 步骤 | MCP 工具 |
|------|----------|
| 建 Nexus 卡 | `create_finance_nexus_analysis` |
| 写阶段产出 | `add_finance_artifact` |
| 飞书进度 | `render_feishu_card_update` → `send_message` |
| 写股票池 | `sync_symbol_to_stock_pool` |
| 总览 | `render_symbol_feishu_summary` |

CLI 等价：

```bash
python3 ~/.hermes/skills/finance/feishu-finance-kanban/scripts/financial_kanban_server.py create-nexus --symbol 002236 --chat-id oc_xxx
python3 .../financial_kanban_server.py sync-symbol-pool --symbol 002236
```

## Pitfalls

- 建卡后立刻 sync 会得到空摘要 — 等 artifact 后再 `sync_symbol_to_stock_pool`。
- Base 的 `日期`/`入池日期`/`最后复盘日` 必须是 **Unix 毫秒**（服务端已自动处理）。
- 字段名用 `股票代码` 而非 `代码` — 见 `feishu-finance-kanban/references/stock-pool-sync.md`。
- 不要与 `kanban-stock-nexus` 对同一标的同一天各跑一套完整流水线。

## Verification

```bash
python3 ~/.hermes/skills/finance/feishu-finance-kanban/scripts/financial_kanban_server.py create-nexus --symbol 002236 --chat-id oc_demo
python3 ~/.hermes/skills/finance/feishu-finance-kanban/scripts/financial_kanban_server.py list-cards --symbol 002236
```
