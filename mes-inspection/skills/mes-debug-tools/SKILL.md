---
name: mes-debug-tools
description: 高阶调试 — GC 日志截取、线程堆栈过滤、ES 日志检索。
version: 1.0.0
author: MES AI Inspection
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [mes, inspection, debug, jvm, gc, thread, elasticsearch]
    category: devops
---

# MES Debug Tools Skill

## Overview

MES 高阶调试工具，提供三项核心调试能力：
1. **GC 日志截取** — 远程获取 JVM GC 日志，按时间段截取，解析 GC 指标（Full GC 频率、暂停时间、内存回收量）
2. **线程堆栈分析** — 远程获取 jstack dump，按关键字过滤线程，返回匹配行及上下文 ±10 行
3. **ES 日志检索** — 从 Elasticsearch 查询应用日志，支持按主机名+时间范围+关键字+日志级别过滤

## When to Use

- 巡检发现 GC 频率异常时，截取问题时段 GC 日志分析
- 发现 BLOCKED/WAITING 线程过多时，按关键字定位问题堆栈
- 需要查看某节点某时段的应用日志时，直接从 ES 检索

## How to Run

### GC 日志截取

```bash
# 获取节点 GC 日志（最近 50000 行）
python scripts/debug_tools.py gc --host 10.0.0.1

# 按时间段截取
python scripts/debug_tools.py gc --host 10.0.0.1 --start 2026-06-05T01:00:00 --end 2026-06-05T02:00:00

# 指定 GC 日志路径
python scripts/debug_tools.py gc --host 10.0.0.1 --gc-log-path /opt/app/logs/gc.log
```

### 线程堆栈分析

```bash
# 获取线程 dump 并统计状态
python scripts/debug_tools.py stack --host 10.0.0.1

# 按关键字过滤堆栈（返回匹配行 ±10 行上下文）
python scripts/debug_tools.py stack --host 10.0.0.1 --keyword DataRecordServiceImpl

# 指定 PID
python scripts/debug_tools.py stack --host 10.0.0.1 --pid 12345 --keyword Deadlock
```

### ES 日志检索

```bash
# 查询节点应用日志
python scripts/debug_tools.py log --host-name 39QEMES-Tomcat-Crontab01 --start 2026-06-05T01:00:00 --end 2026-06-05T02:00:00

# 按关键字搜索
python scripts/debug_tools.py log --host-name 39QEMES-Tomcat-Crontab01 --start 2026-06-05T01:00:00 --end 2026-06-05T02:00:00 --keyword Exception

# 只看 ERROR 级别
python scripts/debug_tools.py log --host-name 39QEMES-Tomcat-Crontab01 --start 2026-06-05T01:00:00 --end 2026-06-05T02:00:00 --level ERROR
```

## Quick Reference

| 工具 | 命令 | 关键参数 |
|------|------|---------|
| GC 日志 | `debug_tools.py gc` | `--host`, `--start`, `--end`, `--gc-log-path` |
| 线程堆栈 | `debug_tools.py stack` | `--host`, `--keyword`, `--context-lines`, `--pid` |
| ES 日志 | `debug_tools.py log` | `--host-name`, `--start`, `--end`, `--keyword`, `--level` |

## Expected Output

### GC 日志分析

```json
{
  "entries": [
    {"timestamp": "2026-06-05T01:20:00", "gc_type": "young", "before_mb": 500, "after_mb": 100, "total_mb": 1024, "pause_sec": 0.05}
  ],
  "summary": {
    "total_count": 15,
    "young_count": 12,
    "full_count": 3,
    "avg_pause_sec": 0.25,
    "max_pause_sec": 2.5,
    "total_pause_sec": 3.75,
    "reclaimed_mb": 4200
  }
}
```

### 线程堆栈分析

```json
{
  "matches": [
    {
      "thread_name": "http-nio-8080-exec-101",
      "state": "BLOCKED",
      "matched_line": "at com.example.mes.service.DataRecordServiceImpl.processRecord(DataRecordServiceImpl.java:123)",
      "context_before": ["..."],
      "context_after": ["..."]
    }
  ],
  "summary": {
    "total_threads": 245,
    "state_counts": {"RUNNABLE": 85, "WAITING": 157, "BLOCKED": 3},
    "blocked_count": 3,
    "deadlock_detected": false,
    "keyword": "DataRecordServiceImpl",
    "match_count": 1
  }
}
```

### ES 日志检索

```json
{
  "logs": [
    {
      "timestamp": "2026-06-05T05:54:11.816Z",
      "message": "2026-06-05 01:21:43.028 WARN  [taskExecutor1-307] DataRecordServiceImpl - 采集项重复",
      "host_name": "39QEMES-Tomcat-Crontab01",
      "log_file": "/u01/app/mes-app/logs/catalina.2026060501.log",
      "tags": ["tomcat-mes"]
    }
  ],
  "total": 2,
  "error": null
}
```

## Prerequisites

- SSH 免密登录到目标 MES 应用节点（GC 日志、线程堆栈）
- Elasticsearch 可访问（ES 日志检索）
- 目标节点已安装 JDK（jstack/jps 命令）
- GC 日志路径正确配置（默认 `/u01/app/mes-app/logs/gc.log`）
- ES 索引前缀正确配置（默认 `39qjmes`）
