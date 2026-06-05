---
name: mes-jvm-check
description: JVM/Tomcat 巡检 — 堆内存、GC、线程、死锁检测。
version: 1.0.0
author: MES AI Inspection
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [mes, inspection, jvm, tomcat, java]
    category: devops
---

# MES JVM Check Skill

## Overview

对 MES 系统的 JVM/Tomcat 应用进行健康巡检，包括堆内存使用、GC 频率与耗时、线程池状态、死锁检测和 Tomcat 连接池监控。适用于 Cron 定时调度场景。

## How to Run

```bash
# 在 MES 巡检项目根目录执行（自动发现本地 Java 进程）
python3 scripts/jvm_check.py

# 指定目标进程 PID
python3 scripts/jvm_check.py --pid 12345

# 指定 JMX 远程连接
python3 scripts/jvm_check.py --jmx-host 10.0.1.10 --jmx-port 9999

# 多实例批量检查
python3 scripts/jvm_check.py --pids 12345,12346,12347
```

## Quick Reference

| 检查项 | 命令/指标 | 告警阈值 | 严重阈值 |
|--------|-----------|----------|----------|
| 堆内存使用率 | `jstat -gc` | >75% | >90% |
| Old Gen 使用率 | `jstat -gc` | >80% | >95% |
| Full GC 频率 | GC 日志 | >3次/分钟 | >10次/分钟 |
| Full GC 耗时 | GC 日志 | >2s | >5s |
| 活跃线程数 | `jstack` | >500 | >800 |
| 死锁检测 | `jstack` | 存在死锁 | - |
| Tomcat 线程池 | JMX | >80% | >95% |

### 关键参数

- `--pid`: 目标 Java 进程 PID
- `--jmx-host`: JMX 远程主机地址
- `--jmx-port`: JMX 远程端口
- `--heap-threshold`: 堆内存告警阈值（百分比）
- `--gc-threshold`: GC 频率告警阈值（次/分钟）
- `--thread-threshold`: 线程数告警阈值

## Expected Output

```json
{
  "service": "jvm",
  "timestamp": "2026-06-04T10:30:00Z",
  "status": "healthy",
  "pid": 12345,
  "checks": {
    "heap_memory": {
      "status": "ok",
      "used_mb": 2048,
      "max_mb": 4096,
      "usage_percent": 50.0,
      "young_gen_percent": 45.0,
      "old_gen_percent": 52.0
    },
    "gc": {
      "status": "ok",
      "young_gc_count": 150,
      "young_gc_avg_ms": 35,
      "full_gc_count": 2,
      "full_gc_avg_ms": 450,
      "full_gc_last_1min": 0
    },
    "threads": {
      "status": "ok",
      "total": 245,
      "daemon": 180,
      "runnable": 85,
      "blocked": 3,
      "waiting": 157,
      "deadlock_detected": false,
      "deadlocked_threads": []
    },
    "tomcat_pool": {
      "status": "ok",
      "max_threads": 200,
      "current_threads": 120,
      "busy_threads": 45,
      "usage_percent": 22.5,
      "queue_size": 0
    }
  },
  "exit_code": 0
}
```

**退出码说明：** 0=全部正常, 1=存在告警, 2=存在严重问题

## Troubleshooting

### 堆内存持续增长

```bash
# 生成堆转储
jmap -dump:format=b,file=/tmp/heap_dump.hprof <pid>
# 分析大对象
jmap -histo:live <pid> | head -20
```

### Full GC 频繁

检查是否存在内存泄漏，查看 GC 日志确认回收效果。调整 JVM 参数：

```bash
-Xms4g -Xmx4g -XX:MetaspaceSize=256m -XX:+UseG1GC
```

### 死锁检测

```bash
# 获取线程转储
jstack <pid> > /tmp/thread_dump.txt
# 搜索死锁
grep -A 20 "Found.*deadlock" /tmp/thread_dump.txt
```

### JMX 连接失败

确认 JMX 远程访问已启用：

```bash
-Dcom.sun.management.jmxremote
-Dcom.sun.management.jmxremote.port=9999
-Dcom.sun.management.jmxremote.authenticate=false
-Dcom.sun.management.jmxremote.ssl=false
```

### Tomcat 线程池耗尽

检查慢请求日志，调整 `server.tomcat.threads.max` 配置。确认是否有数据库连接瓶颈。
