# Pitfalls & Verification

（此文件通过 `skill_view('daily-diary', file_path='references/pitfalls.md')` 加载）

## Common Pitfalls

1. **iCloud 同步延迟** — 文件写入本地后 iCloud 可能需要数分钟才在其他设备可见。告知用户：已保存本地，同步需要时间。

2. **模型不支持 vision** — DeepSeek 等模型不能看图片。保存图片文件，但说明"图片已保存，但这个模型看不到内容"。

3. **用户变更存储目录后 cronjob 未更新** — cron prompt 里 hardcode 了路径。必须：删旧 job → 更新路径 → 重建 job。三步缺一不可。

4. **cron 在周末边界上的 ISO 周编号** — `date +%V` 在跨年边界也正确。但注意 Www 不含年份，跨年时 `2025-W01` 和 `2026-W01` 是不同的目录。

5. **图片相对路径** — `images/YYYY-MM-DD-NNN.jpg` 是相对路径，在 Obsidian 中正常显示，但文件被移出目录树后不可用（这是设计，不是 bug）。

6. **天气获取失败** — `curl wttr.in` 可能需要网络。失败时直接跳过天气行，不要报错中断。

## Verification Checklist

- [ ] `#diary-start` 创建目录和文件（如果不存在）
- [ ] 同一天追加不重复 H1
- [ ] 新一天创建新文件，写完整 H1
- [ ] 图片保存到 `images/` 并用相对路径引用
- [ ] `#diary-end` 确认并结束会话
- [ ] `#view-diary` 读取当天内容返回
- [ ] Setup wizard 第一次运行创建 cronjob
- [ ] Setup wizard 第二次运行提示"已设置"，询问是否重配
- [ ] 重配时删除旧 cronjob 再建新
- [ ] 移除时删除 cronjob + 清 memory
- [ ] cron 提醒在已有日记时静默退出
