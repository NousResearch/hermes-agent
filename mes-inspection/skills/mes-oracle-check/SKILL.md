---
name: mes-oracle-check
description: Oracle 巡检 — 慢SQL、表空间、锁等待、会话数。
version: 1.0.0
author: MES AI Inspection
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [mes, inspection, oracle, database]
    category: devops
---

# MES Oracle Check Skill

## Overview

对 MES 系统的 Oracle 数据库进行健康巡检，包括慢 SQL 检测、表空间使用率、锁等待、会话连接数和归档日志监控。适用于 Cron 定时调度场景。

## How to Run

```bash
# 在 MES 巡检项目根目录执行
python3 scripts/oracle_check.py

# 指定连接信息
python3 scripts/oracle_check.py --host 10.0.1.30 --port 1521 --service MESDB

# 使用连接字符串
python3 scripts/oracle_check.py --dsn "10.0.1.30:1521/MESDB"

# 指定认证文件
python3 scripts/oracle_check.py --auth-file /etc/mes/oracle_credentials.json
```

## Quick Reference

| 检查项 | SQL/视图 | 告警阈值 | 严重阈值 |
|--------|----------|----------|----------|
| 表空间使用率 | `DBA_TABLESPACES` | >80% | >90% |
| 慢 SQL | `V$SQL` | >3s | >10s |
| 锁等待 | `V$LOCK` | >30s | >60s |
| 活跃会话数 | `V$SESSION` | >200 | >300 |
| 归档日志空间 | `V$ARCHIVED_LOG` | >80% | >95% |
| PGA 使用率 | `V$PGA_TARGET_ADVICE` | >80% | >95% |

### 关键参数

- `--host`: Oracle 数据库主机地址
- `--port`: 监听端口（默认 1521）
- `--service`: 服务名或 SID
- `--dsn`: 完整连接字符串
- `--auth-file`: 认证信息 JSON 文件
- `--slow-threshold`: 慢 SQL 时间阈值（秒）
- `--tablespace-threshold`: 表空间告警阈值（百分比）

## Expected Output

```json
{
  "service": "oracle",
  "timestamp": "2026-06-04T10:30:00Z",
  "status": "healthy",
  "database": "MESDB",
  "checks": {
    "tablespace": [
      {
        "name": "MES_DATA",
        "status": "ok",
        "size_mb": 51200,
        "used_mb": 35840,
        "usage_percent": 70.0,
        "autoextend": true
      },
      {
        "name": "MES_INDEX",
        "status": "ok",
        "size_mb": 20480,
        "used_mb": 14336,
        "usage_percent": 70.0,
        "autoextend": true
      }
    ],
    "slow_sql": {
      "status": "ok",
      "count": 5,
      "threshold_seconds": 3,
      "top_sql": [
        {
          "sql_id": "abc123",
          "elapsed_seconds": 4.2,
          "executions": 150,
          "avg_seconds": 0.8,
          "sql_text": "SELECT /* MES_ORDER_QUERY */ ..."
        }
      ]
    },
    "lock_wait": {
      "status": "ok",
      "blocked_sessions": 0,
      "max_wait_seconds": 0,
      "details": []
    },
    "sessions": {
      "status": "ok",
      "active": 85,
      "inactive": 120,
      "total": 205,
      "max_sessions": 500,
      "usage_percent": 41.0
    },
    "archive_log": {
      "status": "ok",
      "used_percent": 45.0,
      "space_remaining_gb": 120.5
    }
  },
  "exit_code": 0
}
```

**退出码说明：** 0=全部正常, 1=存在告警, 2=存在严重问题

## Troubleshooting

### 表空间即将满

```sql
-- 查看表空间使用情况
SELECT tablespace_name, round(used_percent, 1) FROM dba_tablespace_usage_metrics;

-- 扩展表空间
ALTER DATABASE DATAFILE '/oradata/mes_data01.dbf' RESIZE 60G;
-- 或添加数据文件
ALTER TABLESPACE mes_data ADD DATAFILE '/oradata/mes_data02.dbf' SIZE 10G AUTOEXTEND ON;
```

### 锁等待超时

```sql
-- 查看锁信息
SELECT s.sid, s.serial#, s.username, l.type, l.id1, l.id2
FROM v$session s, v$lock l WHERE s.sid = l.sid AND l.block > 0;

-- 杀掉阻塞会话
ALTER SYSTEM KILL SESSION 'sid,serial#' IMMEDIATE;
```

### 慢 SQL 优化

使用 AWR 报告分析：`@?/rdbms/admin/awrrpt.sql`

检查执行计划：`EXPLAIN PLAN FOR <sql>; SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY());`

### 连接数满

检查连接泄漏：`SELECT username, machine, count(*) FROM v$session GROUP BY username, machine;`

调整应用连接池配置，确认连接释放逻辑。
