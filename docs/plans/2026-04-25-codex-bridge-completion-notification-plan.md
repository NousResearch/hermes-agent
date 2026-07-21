---
title: Codex Bridge 异步完成通知 MVP 实现计划
date: 2026-04-25
status: active
origin: docs/brainstorms/2026-04-25-codex-bridge-completion-notification-requirements.md
---

# Codex Bridge 异步完成通知 MVP 实现计划

## 问题框架

Codex Bridge 已能通过 app-server stdio 启动异步 Codex turn，并把状态写入 `codex_bridge.db`。缺口在完成后的主动送达：当前没有通知目标、通知状态，也没有 watcher 入口来把 terminal 任务的摘要回发给原会话。

## 技术决策

- 在 `codex_bridge_tasks` 上新增通知元数据：`notify_target`、`notification_status`、`notified_at`、`notification_error`。
- `start` 接受可选 `notify_target`，不传时保持旧行为。
- 新增 one-shot `notify_completed` action：扫描 terminal 且尚未处理通知的任务，按目标发送或标记 `no_target`。
- 默认 notifier 复用现有 `send_message` 工具；测试和 CLI dry-run 通过注入或 dry-run 避免真实外发。
- `local` 目标作为本地消费目标：记录为已通知并返回摘要，不调用外部平台。

## 实现单元

### U1: 持久化通知目标与状态

修改文件：
- `tools/codex_bridge_tool.py`
- `tests/tools/test_codex_bridge_tool.py`

做法：
- 数据库初始化时对旧库执行兼容迁移。
- `CodexBridgeTask.snapshot()`、`list_tasks()`、`get_task_snapshot()` 暴露通知字段。
- `start_task()` 接受 `notify_target` 并保存。

测试场景：
- 启动任务时传入 `notify_target`，状态快照和持久化查询都能看到该值。

### U2: 完成通知 one-shot watcher

修改文件：
- `tools/codex_bridge_tool.py`
- `tests/tools/test_codex_bridge_tool.py`

做法：
- 增加扫描 terminal 任务的方法。
- 对无 target 的任务标记 `no_target`，不调用 notifier。
- 对有 target 的任务构造简洁摘要，调用 notifier 后标记 `sent` 和 `notified_at`。
- 已 `sent` 或 `no_target` 的任务不再重复处理。
- 支持 `dry_run`，只返回会处理的任务，不写通知状态，不发送。

测试场景：
- completed 任务只通知一次。
- 无 target completed 任务不发送，并标记 `no_target`。
- dry-run 不发送且不改变通知状态。

### U3: 工具 schema 与 CLI 入口

修改文件：
- `tools/codex_bridge_tool.py`
- `skills/codex-bridge/references/cli.py`
- `skills/codex-bridge/references/validator.py`
- `tests/skills/test_codex_bridge_skill.py`

做法：
- schema 加入 `notify_completed` action、`notify_target`、`dry_run`。
- CLI `start`/`smoke-test` 增加 `--notify-target`。
- CLI 增加 `notify-completed` one-shot 命令。
- validator 校验 notify 输出的基本结构。

测试场景：
- CLI start 能把 `--notify-target` 传给工具。
- CLI notify-completed dry-run 调用 bridge 且不依赖真实平台。

## 验证

- `python -m py_compile tools/codex_bridge_tool.py skills/codex-bridge/references/cli.py skills/codex-bridge/references/validator.py`
- `scripts/run_tests.sh tests/tools/test_codex_bridge_tool.py tests/skills/test_codex_bridge_skill.py`

## 风险

- 默认 notifier 依赖 `send_message` 的运行环境；没有 gateway 或目标不可达时会记录 `notification_error` 并保留可重试状态。
- 当前只处理 terminal completion，不处理实时 approval/input；后续应在同一 target 模型上扩展。
