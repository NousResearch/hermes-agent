# Stock Nexus — 飞书实时进度

与 **feishu-finance-nexus** 相同模式：**skill 脚本 + `lark-cli`**，不改 `gateway/run.py`。

脚本目录：`skills/devops/kanban-feishu-live/scripts/`  
看板：`--board stock-nexus`

## 编排器（建卡后）

```bash
python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_stage_notify.py \
  --board stock-nexus init 002236 \
  --stock-name "大华股份" \
  --tasks-inline '{"T0":"t_a","T1":"t_b","T2":"t_c","T3":"t_d","T4":"t_e","T5":"t_f","T6":"t_g"}'

# deep 模式加上 --deep 且 tasks 含 T_deep

python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_subscribe.py \
  --board stock-nexus 002236

python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_stage_notify.py \
  --board stock-nexus notify --entity-id 002236 --event pipeline_started
```

会话文件：`~/.hermes/kanban/boards/stock-nexus/feishu_sessions/002236.json`

## 工人（每阶段 kanban_complete 前）

```bash
python3 skills/devops/kanban-feishu-live/scripts/kanban_feishu_stage_notify.py \
  --board stock-nexus notify --entity-id 002236 \
  --event stage_done --stage T1 \
  --summary "$(jq -r '.stages.T1.summary' handoff.json)"
```

T3 硬否决：`--event stage_blocked --stage T3 --summary "硬避雷: ST"`

T6 收尾：`--event pipeline_done --summary "team-deep-read"`

## 与 paper-nexus 共用

同一套 `kanban_feishu_*` 脚本，仅 `--board` 与阶段表不同。paper 另见 `research/kanban-paper-nexus/references/feishu-live-updates.md`。
