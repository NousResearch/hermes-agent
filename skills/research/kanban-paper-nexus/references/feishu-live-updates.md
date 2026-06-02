# 飞书实时进度（skill 层，对齐 feishu-finance-nexus）

**不改 Hermes 核心。** 内置 Kanban + Dispatcher 照旧；飞书「每阶段可见」由 **脚本 + `lark-cli`** 完成，模式与 `feishu-finance-nexus` 的 `render_feishu_card_update` → `send_message` 一致。

## 两层推送

| 层 | 机制 | 何时 |
|----|------|------|
| **主动进度**（本 skill） | `kanban_feishu_stage_notify.py`（`--board paper-nexus`）→ `lark-cli` | 编排建卡后、工人每阶段开始/完成 |
| **兜底终端**（Gateway 内置） | `notify_subscribe` + Notifier ~5s | `completed` / `blocked` / `crashed` 等 |

主动层负责「像 finance 一样实时」；兜底层不依赖改 `gateway/run.py`。

## 编排器（Gateway 飞书会话）

建齐 T0–T6 且拿到各 `task_id` 后 **必须**：

共享脚本：`skills/devops/kanban-feishu-live/scripts/`（paper shim：`paper_feishu_*.py` 自动带 `--board paper-nexus`）。

**推荐单入口（避免漏步骤）：**

```bash
python3 skills/research/kanban-paper-nexus/scripts/paper_feishu_live_init.py \
  2402.03300 \
  --title-zh "DeepSeekMath 开放数学推理" \
  --tasks-inline '{"T0":"t_xxx","T1":"t_yyy",...}'
```

它会顺序执行：`init` → `subscribe` → `pipeline_started`。

```bash
# 手动分步版（仅当你需要逐步 debug 时）
python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_stage_notify.py \
  --board paper-nexus init 2402.03300 \
  --title-zh "DeepSeekMath 开放数学推理" \
  --tasks-inline '{"T0":"t_xxx","T1":"t_yyy",...}'
python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_subscribe.py --board paper-nexus 2402.03300
python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_stage_notify.py \
  --board paper-nexus notify --entity-id 2402.03300 --event pipeline_started
```

`init` 从 `HERMES_SESSION_CHAT_ID` 读 chat（或 `--chat-id`）。会话文件：

`~/.hermes/kanban/boards/paper-nexus/feishu_sessions/<canonical_id>.json`

## 工人（每个 profile：researcher / coder / writer / qa）

在 `kanban_complete` **之前**（handoff 已写好）：

```bash
python3 skills/research/kanban-paper-nexus/scripts/paper_feishu_stage_notify.py notify \
  --canonical-id "$(jq -r .canonical_id handoff.json)" \
  --event stage_done --stage T1 \
  --summary "$(jq -r .thesis_one_liner handoff.json)"
```

可选开工：

```bash
--event stage_started --stage T0
```

T5 完成后带 doc：

```bash
--event stage_done --stage T5 --update-doc-url "https://my.feishu.cn/docx/..."
```

T6 收尾：

```bash
--event pipeline_done --summary "$(jq -r .recommendation_zh handoff.json)"
```

**禁止**在 IM 里贴 PDF 全文或整表 CEL；摘要 ≤200 字 + 消息里已有 DAG 状态行。

## 消息长什么样

```
📄 Paper Nexus · [2402.03300] DeepSeekMath …
✔ T1 主张-证据链 CEL 完成
摘要：…
看板：T0✅ T1✅ T2🔄 T3⏳ T4⏳ T5⏳ T6⏳
📎 文档：https://my.feishu.cn/docx/...
```

## 与 finance-nexus 对照

| | feishu-finance-nexus | paper-nexus（本方案） |
|---|----------------------|------------------------|
| 引擎 | financial-kanban MCP | Hermes Kanban + dispatcher |
| 实时 IM | MCP `render_feishu_card_update` | `kanban_feishu_stage_notify.py` |
| 飞书 doc 名 | MCP / Base | `[id] title_zh`（`paper_feishu_doc_sync.py --handoff`） |
| 谁发送 | 编排 agent 每阶段 | **工人**每阶段（编排只发 `pipeline_started`） |
| 核心改动 | 无（MCP 在 skill 目录） | 无 |

## Pitfalls

- 漏掉 `init / subscribe / pipeline_started` 任一步，飞书实时都会残缺；编排优先用 `paper_feishu_live_init.py`。
- 工人未装 `lark-cli` 或未登录 → 只写 handoff，在 comment 留 `[feishu-notify-skipped]`。
- 不要用 Memory / skill 全文当 `search_memory` query（见 `memory-os.md`）。
- T5 长文只进飞书 **doc**，IM 仍是一条短更新 + 链接。
