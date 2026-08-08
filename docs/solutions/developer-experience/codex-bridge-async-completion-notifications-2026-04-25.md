---
title: Codex Bridge 异步任务需要持久化完成通知状态
date: 2026-04-25
category: docs/solutions/developer-experience/
module: Codex Bridge
problem_type: developer_experience
component: assistant
severity: medium
applies_when:
  - 异步 agent 任务由本地 bridge 启动，但完成结果需要回到原平台会话
  - 任务状态已经持久化，但缺少完成后主动送达能力
  - 测试不能向真实外部平台发送消息
tags: [codex-bridge, async-notification, app-server, send-message, watcher]
---

# Codex Bridge 异步任务需要持久化完成通知状态

## Context

Codex Bridge 已经通过 app-server stdio 启动 Codex 任务，并把状态写入 `codex_bridge.db`。dogfood 暴露出的体验问题是：异步任务完成后没有主动通知原 Feishu/平台会话，用户必须再次触发 Hermes 查询 `status` 或 `list` 才能知道结果。

这类问题不需要先做多租户调度系统。MVP 的关键是让任务在启动时可选记录通知目标，并让一个 one-shot watcher 可以可靠地处理 terminal 任务。

## Guidance

在已有任务表上补齐三个概念，而不是重写底层通信协议：

- `notify_target`：启动时可选记录目标，例如 `local` 或 `feishu:<chat_id>`。
- `notification_status`：记录通知生命周期，例如 `pending`、`sent`、`failed`、`no_target`。
- `notified_at` / `notification_error`：让 watcher 重启后能防重复，并保留失败原因。

watcher 应该只扫描 terminal 状态任务，并做幂等处理：

- 有目标：构造简洁完成摘要，调用可注入 notifier，成功后标记 `sent`。
- 无目标：标记 `no_target`，不发送，避免每次扫描重复捞到同一任务。
- dry-run：返回预览，不发送，也不写通知状态。

默认 notifier 可以复用现有 `send_message` 能力，但核心 manager 方法要允许注入 notifier。这样单元测试可以用 fake notifier 验证行为，避免真实平台副作用。

## Why This Matters

异步 bridge 的产品承诺不是“能启动后台任务”，而是“任务结束后用户能在原上下文看到结果”。如果只有状态表但没有通知状态，系统会卡在“完成但无人查收”的灰区；如果没有持久化防重复，watcher 或 daemon 重启又可能重复推送。

把通知状态做成任务元数据，可以在不引入 mailbox/outbox/inbox 通信机制的情况下满足 MVP，并为后续实时 approval / `requestUserInput` 扩展留下同一套 target 语义。

## When to Apply

- 异步任务生命周期已经持久化，但完成后需要跨平台送达。
- 现有平台发送能力已经存在，新增功能只需要选择目标和调用发送。
- 需要保证测试环境不触发真实外部消息。
- 需要 watcher/daemon 重启后不重复通知。

## Examples

启动时记录目标：

```python
codex_bridge(
    action="start",
    prompt="Investigate the failing tests",
    notify_target="feishu:chat-1",
)
```

one-shot watcher dry-run：

```bash
python skills/codex-bridge/references/cli.py notify-completed --dry-run
```

测试中注入 notifier：

```python
deliveries = []
manager.notify_completed(
    notifier=lambda target, message: deliveries.append((target, message)) or {"ok": True}
)
```

## Related

- `docs/brainstorms/2026-04-25-codex-bridge-completion-notification-requirements.md`
- `docs/plans/2026-04-25-codex-bridge-completion-notification-plan.md`
- `tools/codex_bridge_tool.py`
- `skills/codex-bridge/references/cli.py`
