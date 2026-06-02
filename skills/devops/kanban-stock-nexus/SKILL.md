---
name: kanban-stock-nexus
description: A-share stock Nexus Kanban DAG via /kanban-stock-nexus.
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, stock, nexus, a-share, feishu]
    category: devops
    related_skills: [kanban, kanban-orchestrator, stock-monitoring, kanban-feishu-live]
---

# Kanban Stock Nexus — 固定编排契约

**飞书/CLI 入口：`/kanban-stock-nexus <指令>`**（必须 `/` 触发）。

编排 agent **只建卡、不分析**。工人按 `stock-monitoring` → `references/nexus-kanban-pipeline.md` 执行。

## 飞书实时（对齐 feishu-finance-nexus，不改 Hermes 核心）

| 时机 | 动作 |
|------|------|
| 建卡后 | `kanban_feishu_stage_notify.py init` → `kanban_feishu_subscribe.py` → `notify pipeline_started` |
| 每阶段 | 工人 `notify --event stage_done --stage Tn` |

脚本路径：`skills/devops/kanban-feishu-live/scripts/`（`--board stock-nexus`）。详见 `references/feishu-live-updates.md`。

## When to Use

- `/kanban-stock-nexus 分析 002236`
- `/kanban-stock-nexus 深度分析 600519`
- `/kanban-stock-nexus 批量分析候选股`

普通聊天 **不要** 用本 skill。

## Prerequisites

- 看板 **`stock-nexus`**；`kanban.dispatch_in_gateway: true`
- Workers：`kanban-researcher`、`kanban-coder`、`kanban-writer`、`kanban-qa`
- `lark-cli` 已登录

## Parse User Input

| 字段 | 规则 |
|------|------|
| `symbol` | 第一个 6 位数字 `\\d{6}` |
| `deep` | 含「深度」「deep」「TradingAgents」→ 建 `T_deep` |
| `batch` | 「批量」「候选」「bitable」→ `stock_nexus_bitable_batch.py` |
| `horizon` | 默认 `4w`；含「长线/26周」→ `26w` |

`idempotency_key`: `stock-{symbol}-{YYYYMMDD}`

## Fixed DAG（禁止改标题前缀）

| 卡 | 标题 | assignee | parents |
|----|------|----------|---------|
| T0 | `[stock] {code} 上下文时间线` | kanban-researcher | — |
| T1 | `[stock] {code} 宏观/行业/资金` | kanban-researcher | T0 |
| T2 | `[stock] {code} 技术面/量价` | kanban-coder | T0 |
| T_deep | `[stock] {code} TradingAgents深度` | kanban-coder | T0（仅 deep） |
| T3 | `[stock] {code} 避雷审查` | kanban-qa | T1,T2[,T_deep] |
| T4 | `[stock] {code} 规则校准` | kanban-qa | T3 |
| T5 | `[stock] {code} 投资决策合成` | kanban-writer | T1,T2,T3,T4[,T_deep] |
| T6 | `[stock] {code} QA门禁` | kanban-qa | T5 |

## Orchestrator Procedure

1. `skill_view` 本 skill + `kanban-orchestrator`
2. `hermes kanban boards switch stock-nexus`
3. 解析 symbol / deep / batch；`kanban_create` 全表
4. **飞书实时（必做）**：
```bash
python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_stage_notify.py \
  --board stock-nexus init 002236 --stock-name "大华股份" --deep \
  --tasks-inline '{"T0":"t_...",...}'
python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_subscribe.py --board stock-nexus 002236
python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_stage_notify.py \
  --board stock-nexus notify --entity-id 002236 --event pipeline_started
```
5. 回复：symbol、task_ids、是否 deep、飞书将按阶段推送

## Forbidden

- 自然语言关键词自动进 Kanban
- 编排器自己跑分析脚本
- 工人省略 `stage_done` 飞书 notify（除非 `[feishu-notify-skipped]`）

## Verification

```bash
hermes kanban --board stock-nexus list
python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_stage_notify.py \
  --board stock-nexus notify --entity-id 002236 --event stage_done --stage T0 --summary test --dry-run
```
