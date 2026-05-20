# Memory OS — 论文流水线持久化

Kanban SQLite + 飞书 doc 解决**当次协作**；**Memory OS** 解决**跨会话回忆**（同一篇论文是否读过、CEL 结论、doc 链接、QA 建议）。

## 三层存储分工

| 层 | 存什么 | 生命周期 |
|----|--------|----------|
| Kanban `paper-nexus` | 任务状态、handoff 文件 | 看板运维 |
| `paper_doc_registry.json` | canonical_id → 飞书 doc URL | 永久 1:1 |
| **Memory OS** | 论点、CEL、审计结论、doc、task_ids | 跨 session / 跨 agent |

## MCP 工具（优先）

| 动作 | 工具 |
|------|------|
| 开工前回忆 | `mcp_unified-memory-os_search_memory` |
| 企业/wiki 背景 | `mcp_unified-memory-os_search_existing_knowledge`（可选） |
| 写入 | `mcp_unified-memory-os_store_memory_markdown`（长文/表格/JSON 用这个） |
| 短注 | `mcp_unified-memory-os_store_memory` |

`unified-memory-os` 不可用时：不要改用全盘 grep；在 `kanban_comment` 留 `[memory-skipped]` 并继续。

## workflow_id（硬规则）

```text
paper-nexus:<canonical_id>
```

示例：`paper-nexus:2402.03300`（**无** `v3` 后缀）。同一论文所有阶段共用此 id。

## 何时写（必做节点）

| 时机 | 谁 | importance | 写什么 |
|------|-----|------------|--------|
| 编排建卡后 | orchestrator | 0.55 | task_ids、doc 策略 create/update、arxiv 链接 |
| T0 完成 | researcher | 0.6 | thesis_one_liner、reading_map |
| T1 完成 | researcher | 0.75 | CEL 表全文（JSON 或 markdown 表） |
| T4 完成 | coder | 0.7 | experiment_audit 五问结论 |
| T5 完成 | writer | 0.85 | feishu_doc_url、核心总结 5 条（中文） |
| T6 通过 | qa | 0.9 | qa_pass、recommendation_zh、最终 doc |

**不要**每个 heartbeat 都写；阶段完成写一次即可。

## 开工前：先搜再干（query 硬规则）

**只允许**用 **canonical 代号（arXiv 或 `s2:<hash>`）+ 论文标题（截断）** 作为 `search_memory` 的 `query`。

| 允许 | 禁止 |
|------|------|
| `2402.03300` | skill / pipeline / qa-rubric 全文 |
| `s2:ceced53f349f7e425352ecf4813b307667cd8aa6` | PDF、Abstract、CEL 表、handoff JSON |
| `2402.03300 DeepSeekMath Pushing the Limits` | 同上 |
| `limit=3` | `kanban-feishu-design`、`finance-kanban` 等无关 Kanban 设计记忆 |
| 单行 ≤120 字符 | 多段粘贴、换行、>24 个 token |

用脚本生成（编排 **必跑**，禁止手写长 query）：

```bash
python3 skills/research/kanban-paper-nexus/scripts/paper_nexus_metadata.py <id> > /tmp/meta.json
python3 skills/research/kanban-paper-nexus/scripts/paper_memory_search_query.py <id> \
  --meta-json /tmp/meta.json
```

将 JSON 里的 `query`、`limit` 传给 `search_memory`；`workflow_id` 字段写入 handoff 供核对（`paper-nexus:<canonical_id>`）。

**推荐两次短搜（均遵守上表）：**

1. `query=<canonical_id>`，`limit=3`
2. 无命中再 `query=<canonical_id> <title≤80字符>`，`limit=3`

若命中：
- 复用已有 `feishu_doc_url`（`paper_doc_registry.py resolve` 应一致）
- 在飞书回复注明「Memory 命中：上次精读 YYYY-MM-DD，本次 append 更新」
- **仍可按用户要求重跑 DAG**；避免重复 create 第二篇 doc

## store_memory_markdown 格式

用脚本生成条目（推荐）：

```bash
python3 skills/research/kanban-paper-nexus/scripts/paper_memory_markdown.py \
  --stage T1 --handoff "$HERMES_KANBAN_WORKSPACE/handoff.json" \
  --session-id "$HERMES_SESSION_ID" --task-id "$HERMES_KANBAN_TASK"
```

将 stdout **整段**传给 `store_memory_markdown` 的 `entry` 参数。

手写模板：

```text
session_id: <HERMES_SESSION_ID 或稳定 session>
agent_name: hermes-agent
title: [paper-nexus] 2402.03300 T1 CEL
workflow_id: paper-nexus:2402.03300
tags: paper, arxiv, kanban-paper-nexus, 2402.03300, cel
importance_score: 0.75

# T1 主张-证据链
- canonical_id: 2402.03300
- thesis: …
- claims: …
- feishu_doc: https://my.feishu.cn/docx/...
- kanban_task: t_xxx
```

## 字段约定

- `session_id`：gateway 会话 ID；工人用 `HERMES_SESSION_ID` 环境变量（若无则 `paper-nexus-<canonical_id>`）
- `request_id`：可选，填 kanban `task_id`（如 `t_abc123`）
- **禁止**把 PDF 全文、上万字 appendix 写入 Memory；只写 CEL + 审计 + 链接 + ≤800 字摘要

## T6 QA 附加项

见 `qa-rubric.md` §E：流水线结束后应能在 Memory OS 搜到 `workflow_id:paper-nexus:<id>` 且含 `feishu_doc_url`。
