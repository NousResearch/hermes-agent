---
name: continuous-task
version: 1.0.0
author: andorexu
license: MIT
metadata:
  hermes:
    tags: [automation, loop, scheduler, background]
    related_skills: []
description: "Continuous task loop: start, stop, check status. 持续迭代任务循环器：自动间隔、独立会话、上下文不膨胀。Each iteration fresh session, no token accumulation."
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
   python3 ~/.hermes/scripts/task_looper.py <task_id> "<prompt>" <interval_minutes>
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

