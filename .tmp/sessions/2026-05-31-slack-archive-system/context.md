# Task Context: Slack Archive Approval System

Session ID: 2026-05-31-slack-archive-system
Created: 2026-05-31T05:00:00Z
Status: in_progress

## Current Request

为 Hermes Agent 实现 Slack 归档审批系统，包含：
1. 发送端：`send_archive_approval_block` 工具（发送带按钮的卡片）
2. 接收端：`_handle_archive_action` handler（处理按钮点击）
3. 昵称系统：`slack_nickname.py`（用户昵称解析）
4. 口吻系统：`elysia_voice.py`（爱莉希雅风格的文案）

## Context Files (Standards to Follow)

- `tools/send_message_tool.py` — 工具注册模式参考（registry.register 格式）
- `gateway/platforms/slack.py:2390` — `_handle_slash_confirm_action` — 授权检查模式参考
- `SLACK_ALLOWED_USERS` 环境变量 — 现有授权机制

## Reference Files (Source Material to Look At)

- `gateway/platforms/slack.py` — 主要的 Slack 适配器（3027行）
- `tools/registry.py` — 工具注册机制

## External Docs Fetched

### Slack Block Kit - Actions block
- URL: https://docs.slack.dev/reference/block-kit/blocks/actions-block
- 最多 25 个 elements
- block_id 最大 255 字符
- block_id 更新消息时需要新的

### Button element
- type: "button"
- text: plain_text 对象
- value: 按钮值（JSON字符串）
- action_id: 唯一标识符
- style: "primary" (绿色) / "danger" (红色)

## Components

1. **slack_archive_tool.py** — 归档审批卡片发送工具
   - registry.register() 注册
   - toolset="slack"
   - 参数：channel_id, thread_ts, summary_preview

2. **slack_nickname.py** — 昵称解析模块
   - resolve_nickname(user_id, user_name, cache)
   - 优先级：SLACK_USER_NICKNAMES > 显示名 > username

3. **elysia_voice.py** — 口吻文案模块
   - get_decision_text(action_id, nickname)
   - get_archive_post_text(action_id, nickname)

4. **slack.py - _handle_archive_action** — 接收端 handler
   - SLACK_ALLOWED_USERS 授权检查
   - 幂等保护（防双击）
   - chat_update 更新卡片

## Constraints

- 必须复用现有的 SLACK_ALLOWED_USERS 授权机制
- 必须有幂等保护（同一消息只能处理一次）
- 工具注册必须使用 registry.register() 格式
- 所有外部资源引用必须先验证存在

## Exit Criteria

- [ ] 4个文件全部创建完成
- [ ] 工具可注册并可用
- [ ] 授权检查正常工作
- [ ] 幂等保护正常工作
- [ ] 测试通过
