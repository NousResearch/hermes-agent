---
title: Codex Bridge 异步完成通知 MVP 需求
date: 2026-04-25
status: accepted
scope: lightweight
---

# Codex Bridge 异步完成通知 MVP 需求

## 背景

Hermes 通过 `skills/codex-bridge/references/cli.py start` 启动 Codex app-server stdio 任务后，会把任务状态写入本地 `codex_bridge.db`。当前实现不是常驻订阅或完成回调模式：Codex turn 完成后不会主动通知原 Feishu/平台会话，用户必须再次说“继续”后 Hermes 才会手动查询 `status` 或 `list`。

这造成一个产品异味：异步任务已经启动，但完成后没有人主动查收。

## 范围决策

本次做窄范围 MVP：让 Codex Bridge 启动的异步任务在完成后能回到原会话或目标发送完成摘要。不要做多租户调度系统，不重写现有 Codex Bridge 低层协议，不引入 mailbox/outbox/inbox 作为主通信机制。

## 目标

- 启动任务时可选记录通知目标，例如 `local`、`feishu:<chat_id>` 或其他 `send_message` 支持的显式平台目标。
- 默认不改变现有 API 行为；未传通知目标时仍能正常启动和查询。
- 提供 watcher/one-shot poll 入口，发现已完成但未处理通知的任务。
- 对有目标的任务读取 final summary，生成简洁完成摘要，并通过可注入 notifier 发送。
- 对无目标的完成任务标记为 `no_target`，避免 watcher 重启后重复处理。
- 通过持久化 `notification_status` / `notified_at` 防重复通知。

## 非目标

- 不实现常驻多租户调度器。
- 不实现 pending approval / `requestUserInput` 的实时双向交互。
- 不让测试向真实 Feishu、WeChat、Telegram 等外部平台发消息。
- 不开放 `danger-full-access` 默认权限。
- 不用 mailbox/outbox/inbox 作为通信机制。

## 验收标准

- `codex_bridge(action="start", notify_target=...)` 能把目标写入任务状态。
- watcher/notify 入口只通知 terminal 状态任务一次；重启或重复运行不会重复发送。
- terminal 任务没有 target 时会被标记为 `no_target`，不会调用 notifier。
- CLI 暴露 `--notify-target` 和 one-shot `notify-completed` 入口，并支持 dry-run。
- 测试通过 mock/inject notifier 覆盖通知行为。

## 后续扩展说明

pending approval 和 `requestUserInput` 后续可复用同一通知目标字段：当任务进入 `waiting_for_approval` 或 `waiting_for_user_input` 时，watcher 可以发送带 request id 的交互提示；平台侧回复再映射到 `codex_bridge respond`。本次先只处理 terminal completion，避免把交互式审批设计混入 MVP。
