---
name: daily-diary
description: >-
  Daily diary system — record entries via #diary-start / #diary-end commands,
  store by ISO week in markdown files, optional cron reminder that skips if
  already written. First-run setup wizard walks through directory and
  reminder time.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Diary, Journal, Note-Taking, Cron, Productivity]
    related_skills: [obsidian]
    requires_toolsets: [terminal, file]
    category: productivity
---

# Daily Diary — Markdown Journal

## Overview

`#diary-start` / `#diary-end` 命令驱动的日记系统。按 ISO 周分目录存储为 markdown 文件，可设置 cron 定时提醒（写过当天日记则不提醒）。

兼容 Obsidian、VS Code 等任何 markdown 编辑器。

## When to Use

| 用户说… | 做什么 |
|---|---|
| `#diary-start` | 开始记录。收集文字和图片，写入当天文件。 |
| `#diary-end` | 结束记录。确认保存。 |
| `#view-diary` | 读取最近日记。 |
| "帮我设置日记" / "设定日记" | 运行引导初始化（**先检查 memory 是否已有配置**）。 |
| "改日记设置" / "改提醒时间" | 调出重配流程。 |
| "删除日记系统" | 移除 cronjob + 清配置。 |

## Core Flow

```
skill_view('daily-diary') on load
  → 检查 memory 是否有 diary_storage_dir
    ├─ 有 → 进入日常模式
    │     #diary-start → 写日记（references/writing.md）
    │     #view-diary   → 读取日记
    └─ 无 → 进入初始化模式
          user: "设置日记" → 加载 references/setup.md 走引导流程
```

## Quick Reference (日常模式)

### 写日记 (#diary-start)

1. 从 memory 读取 `diary_storage_dir`
2. 确定 ISO 周：`date +%V` → `YYYY-Www`
3. 构建路径：
   - `{storage_dir}/{YYYY-Www}/{YYYY-MM-DD}.md`
   - `{storage_dir}/{YYYY-Www}/images/`
4. 如果当天已有内容 → 追加 `## HH:MM` 段
5. 如果新一天 → 写完整头（H1 + 天气）
6. 标记为"记录中"

收到文字 → 追加到文件。收到图片 → 存 `images/` 并引用。

### 结束 (#diary-end)

确认文件路径和条目数，关闭会话。

### 查看 (#view-diary)

读取当天 md 文件返回。

---

**细节参考 →** 以下场景需要时通过 `skill_view('daily-diary', file_path='references/...')` 加载：

| 场景 | 加载文件 |
|---|---|
| 首次设置 | `references/setup.md` |
| 改配置 / 重新配 / 移除 | `references/reconfiguration.md` |
| 日记格式规范、天气、图片排列 | `references/format.md` |
| cronjob 行为细节 | `references/cron-reminder.md` |
| 坑和验证清单 | `references/pitfalls.md` |
