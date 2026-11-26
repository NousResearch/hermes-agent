# Reconfiguration & Removal

当用户已设置过日记，希望改配置或移除时加载。
（此文件通过 `skill_view('daily-diary', file_path='references/reconfiguration.md')` 加载）

## 变更存储目录

1. 从 memory 读取 `diary_reminder_job_id`
2. 如果 cronjob 存在 → **先删除**（cron prompt 里 hardcode 了路径）
3. 询问新目录
4. 如果之前设过提醒时间 → **重新创建** cronjob（见 setup.md Step 5）
5. 用 `memory(action='replace', old_text='diary_storage_dir', new_content='...')` 更新

## 变更提醒时间

1. 从 memory 读取 `diary_reminder_job_id`
2. 如果存在 → `cronjob(action='remove', job_id=...)`
3. 询问新时间
4. 用新 schedule 创建 cronjob
5. 更新 `diary_reminder_time` 和 `diary_reminder_job_id`

## 变更命令别名

1. 询问新的 start/end 命令（如 `#dstart` / `#dend`）
2. 更新 `diary_start_command` / `diary_end_command`

## 移除日记系统

1. 如果 `diary_reminder_job_id` 存在 → `cronjob(action='remove', job_id=...)`
2. 移除所有 `diary_*` key（`memory(action='remove', old_text='diary_')`）
3. 确认：日记系统已移除，日记文件仍保留在磁盘上
