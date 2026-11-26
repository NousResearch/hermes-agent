---
name: continuous-task
version: 1.0.0
author: andorexu
license: MIT
metadata:
  hermes:
    tags: [automation, loop, scheduler, background]
    related_skills: []
description: "Continuous task looper with auto-interval, isolated sessions. 持续迭代任务循环器。"
category: devops
---

# Continuous Task Loop — 持续迭代任务循环器

## Overview / 概述

Run a task repeatedly at automatic intervals. Each iteration spawns a fresh session — no context bloat, no token accumulation. State passes through a lightweight JSON file (≤200 chars per round).

按自动间隔反复执行任务。每轮启动独立会话——上下文不膨胀，token 不累积。状态通过轻量 JSON 传递。

## When to Use / 触发场景

- User says "持续优化..." / "一直搜索..." / "循环执行..." / "迭代改进..." / "keep doing..." / "loop" / "continuously..."
- Long-running tasks that would overflow context if done in one session
- Monitoring, periodic search, iterative improvement, batch processing

## Auto Interval Detection / 自动间隔判定

分析任务内容，匹配默认间隔：

| Keyword / 关键词 | Interval / 间隔 |
|--------|----------|
| 监控/监听/monitor/watch | 10 min |
| 搜索/查找/收集/search/crawl | 15 min |
| 优化/改进/重构/optimize/refactor | 5 min |
| 写/生成/模板/邮件/generate/email | 30 min |
| 分析/调研/报告/analyze/research | 60 min |
| 备份/同步/归档/backup/sync | 4 hours |

含"紧急/urgent"→间隔减半。多关键词→取最短。

## 执行流程

```
1. 用户: "持续优化邮件模板"
2. 判定间隔 → 30分钟
3. 创建状态文件: ~/.hermes/task_loops/<task_id>/state.json
4. 写入停止信号占位
5. 启动 terminal(background=true):
   SKILL_DIR=$(dirname "$(readlink -f "$0")")
   python3 "$SKILL_DIR/scripts/task_looper.py" <task_id> "<prompt>" <interval_minutes>
   (Script bundled at skills/devops/continuous-task/scripts/task_looper.py)
6. 告知用户: "已启动任务循环，每30分钟一轮。回复'停止'结束。"
```

## 停止
用户说"停止"或"停止任务循环"或"停止循环" → 写停止文件 → looper检测到退出。

## 状态查看
用户问"循环状态" → 读取 state.json → 显示轮次/结果。

## 输出格式
每轮完成后推送微信（通过 send_message）：
```
🔄 循环任务 #3 完成
任务：优化邮件模板
本轮改进：缩短了开场白
下次运行：30分钟后
回复"停止"结束循环
```

## 约束
- 每轮独立会话，通过 state.json 传递上轮摘要（≤200字）
- 不占用上下文，不积累 token
- looper 退出后无残留进程
- 同时最多 3 个循环任务


## Common Pitfalls / 常见陷阱

1. **Too many concurrent loops.** Max 3 simultaneous loop tasks. Check `task_loops/` directory before starting a new one.
2. **Forgetting to stop.** Loops run until stopped. If user goes silent, the loop keeps running. Remind them how to stop.
3. **Context leaking.** Never pass full context between iterations. Use state.json with ≤200 char summary only.
4. **Wrong interval.** Double-check auto-detected interval against task type. Monitoring should be 10min, not 60min.
5. **Stale state files.** Remove state.json if loop was killed abnormally — stale state files block restart.
6. **NOT for batch queue processing (2026-05-22).** Task looper runs the SAME prompt each round; it can't iterate a list of items. If the task is "process 64 companies from a list", use **cronjob** instead. Each looper round spawns a fresh `hermes chat` process → ~60% overhead in init per round → very slow for batch work.
7. **PATH issue on WSL non-interactive shells.** `task_looper.py` hardcodes `hermes` without full path → breaks when `~/.local/bin` missing from PATH. Fix: use `/home/andore/.local/bin/hermes`.
6. **PATH 陷阱（2026-05-22 实测）**：`task_looper.py` 内 `subprocess.run(["hermes", "chat", ...])` 硬编码了 `hermes` 命令。非交互式进程（如从 cron/systemd/后台启动）不加载 `.bashrc`，`~/.local/bin` 不在 PATH → 每轮都找不到 hermes。修复：改为完整路径 `/home/andore/.local/bin/hermes`。每次修改 task_looper.py 后必须确认该行用的是完整路径。
7. **队列任务模式**：当任务是批量处理同类项（如背调N家公司），不要试图在 looper prompt 里内嵌完整队列。正确做法：①保存待处理列表到 JSON 文件（`pending_companies.json`，含 `current_index` 指针）②每轮 looper 从文件读取下一家 → 执行 → 结果追加到 CSV → 更新指针。状态文件 + JSON 队列 = looper 专长。

## Queue Task Pattern / 队列任务模式

适用于"对N个同类目标逐一执行相同操作"：

```
1. 保存目标列表 → pending_<task>.json（含 total + pending[] + completed[] + current_index）
2. 创建输出文件 → <task>_results.csv（含表头）
3. Looper prompt: "从 <pending_file> 读取第 {current_index} 个目标，执行<操作>，结果追加到 <results_csv>，更新 current_index"
4. 每轮 state.json 传递 last_summary（仅上轮结果摘要，≤200字）
5. 全部完成 → looper 自动 exit(0)
```

## Verification Checklist / 验证清单

- [ ] Interval correctly auto-detected from task keywords
- [ ] State file created at `~/.hermes/task_loops/<task_id>/state.json`
- [ ] Stop signal placeholder written
- [ ] Background process started with `terminal(background=true)`
- [ ] User informed of loop start + how to stop
- [ ] ≤3 concurrent loops (check `task_loops/` directory)


## Author / 作者

- **GitHub:** [github.com/andorexu](https://github.com/andorexu)
- **Company / 公司:** 百赛联（深圳）科技有限公司
- **Email:** andore@sina.com

