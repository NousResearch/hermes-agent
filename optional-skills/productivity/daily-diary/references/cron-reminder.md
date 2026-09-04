# Cron Reminder Logic

cronjob 的完整行为规范。
（此文件通过 `skill_view('daily-diary', file_path='references/cron-reminder.md')` 加载）

## 触发逻辑

```
每天 reminder_time：
  1. 获取今天 → YYYY-MM-DD
  2. 获取 ISO 周 → Www
  3. 检查 {storage_dir}/{YYYY-Www}/{YYYY-MM-DD}.md：
     ├─ 文件不存在 → 发送提醒
     ├─ 文件存在，但 grep "# YYYY-MM-DD" 无结果 → 发送提醒（当天没写）
     └─ 文件存在且包含当天标题 → 静默退出，不发任何消息
```

## 提醒消息

> 📝 该写日记啦

简短一句即可。不要加图片、MEDIA、长文。

## 条款

- cronjob 的 `deliver` 用 `"origin"` 自动适配创建时的对话
- cronjob 绑定 `skills: ["daily-diary"]` 确保 cron agent 知道如何检查
- 如果用户更换平台，需要更新 cronjob 的 deliver target
