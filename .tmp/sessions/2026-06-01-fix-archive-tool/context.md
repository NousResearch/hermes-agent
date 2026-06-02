# Task Context: 修复 slack_archive_tool 注册问题

Session ID: 2026-06-01-fix-archive-tool
Created: 2026-06-01T05:30:00+08:00
Status: in_progress

## Current Request

用户发现触发归档能力时没有出现卡片，只有口述文字。原因：`slack_archive_tool.py` 缺少 `registry.register()` 调用，工具未被注册到系统中。

## Context Files (Standards to Follow)

- `references/local-patch-and-release.md` — 本地补丁标准

## Reference Files (Source Material to Look At)

- `tools/send_message_tool.py:126` — SEND_MESSAGE_SCHEMA 格式
- `tools/send_message_tool.py:1793` — registry.register() 调用模式
- `tools/registry.py` — 注册机制
- `tools/slack_archive_tool.py` — 需要修复的文件

## External Docs Fetched

无（内部工具）

## Components

1. 添加 `SEND_ARCHIVE_APPROVAL_BLOCK_SCHEMA` JSON Schema
2. 添加 `_send_archive_tool_handler()` wrapper 函数
3. 添加 `_check_archive_requirements()` 检查函数
4. 添加 `registry.register()` 调用

## Constraints

- 保持与 `send_message_tool.py` 一致的注册模式
- `summary` 为必需参数，`channel_id` 和 `thread_ts` 为可选（自动从 session context 获取）
- 注册到 `slack` toolset

## Exit Criteria

- [x] Schema 定义正确
- [x] Handler wrapper 函数正确
- [x] registry.register() 调用正确
- [x] 工具能被 agent 发现和调用
- [x] 发送卡片到正确的 thread（thread_ts: 1780262067.392419）
