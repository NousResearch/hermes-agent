# Task Context: 实现 Slack 归档审批系统（接收端）

Session ID: 2026-06-01-archive-receiver
Created: 2026-06-01T05:45:00+08:00
Status: in_progress

## Current Request

归档卡片上的按钮点击没有任何响应。原因：缺少接收端处理组件。

## Context Files (Standards to Follow)

- `references/local-patch-and-release.md` — 本地补丁标准

## Reference Files (Source Material to Look At)

- `gateway/platforms/slack.py:2484` — `_handle_approval_action` 参考实现
- `gateway/platforms/slack.py:2386` — `_handle_slash_confirm_action` 参考实现
- `gateway/platforms/slack.py:665-682` — action 注册模式
- `tools/slack_archive_tool.py` — 发送端（已有）

## External Docs Fetched

无

## Components

1. `tools/slack_nickname.py` — 用户昵称解析
   - 优先级：SLACK_USER_NICKNAMES env → name_cache → raw user_name
   - 纯函数，无外部依赖

2. `tools/elysia_voice.py` — 爱莉希雅风格决策文案
   - `get_decision_text(action_id, nick)` — 返回按钮点击后的显示文案
   - `get_archive_post_text(action_id, nick)` — 返回 thread 内后续消息
   - 纯函数，无外部依赖

3. `gateway/platforms/slack.py` — 添加归档处理
   - `_handle_archive_action(self, ack, body, action)` — 处理按钮点击
   - 注册 `archive_approve` 和 `archive_discard` action IDs
   - 侵入式修改，需标记 `--- Local patch ---`

## Constraints

- 遵循 `local-patch-and-release.md` 标准
- 侵入式修改 slack.py 必须标记 patch point
- 使用 `SLACK_USER_NICKNAMES` 环境变量（已配置：`{"U0B6GNDF5J6":"爸爸"}`）
- chat_update 有 3000 字符限制

## Exit Criteria

- [ ] slack_nickname.py 创建完成
- [ ] elysia_voice.py 创建完成
- [ ] _handle_archive_action 添加到 slack.py
- [ ] action IDs 注册完成
- [ ] 按钮点击能正确响应
