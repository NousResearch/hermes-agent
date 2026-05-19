# Setup Wizard — First Run

仅在 SKILL.md 核心流程检测到 memory 无 `diary_storage_dir` 时加载。
（此文件通过 `skill_view('daily-diary', file_path='references/setup.md')` 加载）

## Pre-check

**重要：** 在走 wizard 前，先检查 memory 是否已有日记配置。
如果 memory 包含 `diary_storage_dir`，用户却说"设置日记"，说明想重新配 → 加载 `references/reconfiguration.md`。

## Step-by-step Wizard

### 1. 说明

> 我来帮你设置日记系统。之后用 `#diary-start` / `#diary-end` 写日记，存为 markdown 文件。

### 2. 询问存储目录

> 日记文件存在哪里？

- 默认建议：`~/Documents/Diary/`
- 若用户提到 Obsidian → 建议 `~/Documents/Obsidian Vault/Diary/`
- 若用户提到 iCloud → 建议 `~/Library/Mobile Documents/com~apple~CloudDocs/Diary/`
- 接受任意绝对路径

如果父目录不存在，告知用户首次写时会自动创建。

### 3. 询问提醒时间

> 需要每天定时提醒写日记吗？什么时间？（如 21:00）。说"不用"跳过。

接受 `21:00`、`9 PM`、`2100` 等格式，转换为 cron（`0 21 * * *`）。

### 4. 说明提醒行为

> 提醒是智能的——如果当天已经写过日记，不会发通知。只有空白日才提醒。

### 5. 创建 cronjob

用 `cronjob` 工具创建：

```json
{
  "action": "create",
  "name": "diary-reminder",
  "schedule": "0 21 * * *",
  "prompt": "Check if today ({YYYY-MM-DD}) has a diary entry.\n\nSteps:\n1. Determine ISO week: `date +\\%V`\n2. Build path: {STORAGE_DIR}/{YYYY-Www}/{YYYY-MM-DD}.md\n3. If file contains \"# {YYYY-MM-DD}\" as a heading → already written. Say NOTHING.\n4. Otherwise → send brief reminder: \"📝 该写日记啦\"\n\nDiary path: {STORAGE_DIR}\n\nCritical: If entry exists, SILENT exit.",
  "deliver": "origin",
  "skills": ["daily-diary"]
}
```

> **替换** `{STORAGE_DIR}` 为用户确认的实际路径。不要原样发送。

### 6. 保存配置到 memory

使用结构化 key-value 格式（便于解析）：

```
diary_storage_dir={PATH}
diary_reminder_time={TIME_OR_NONE}
diary_reminder_job_id={JOB_ID_OR_NONE}
diary_start_command=#diary-start
diary_end_command=#diary-end
diary_initialized=true
```

### 7. 确认

> 日记设置完成！试试说 `#diary-start` 写下第一篇。
