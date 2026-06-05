---
name: mes-nginx-check
description: Nginx 健康巡检 — 进程、错误率、连接数、响应时间。
version: 1.0.0
author: MES AI Inspection
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [mes, inspection, nginx, web]
    category: devops
---

# MES Nginx Check Skill

## Overview

对 MES 系统的 Nginx 服务进行全面健康巡检，包括进程状态、错误率统计、活跃连接数、上游服务可达性和响应时间检测。适用于 Cron 定时调度场景。

## How to Run

```bash
# 在 MES 巡检项目根目录执行
python3 scripts/nginx_check.py

# 指定 Nginx 状态端点（默认 http://localhost:80/nginx_status）
python3 scripts/nginx_check.py --status-url http://10.0.1.5:8080/nginx_status

# 指定日志路径分析错误率
python3 scripts/nginx_check.py --access-log /var/log/nginx/mes_access.log

# 自定义阈值
python3 scripts/nginx_check.py --error-rate-threshold 5.0 --conn-threshold 500
```

## Quick Reference

| 检查项 | 命令/指标 | 告警阈值 | 严重阈值 |
|--------|-----------|----------|----------|
| 进程存活 | `systemctl status nginx` | 进程不存在 | - |
| 5xx 错误率 | 解析 access.log | >2% | >5% |
| 活跃连接数 | stub_status 接口 | >300 | >500 |
| 上游可达性 | HTTP HEAD 请求 | 超时 3s | 不可达 |
| 响应时间 | 平均响应延迟 | >500ms | >2000ms |

### 关键参数

- `--status-url`: Nginx stub_status 端点 URL
- `--access-log`: 访问日志路径（用于错误率分析）
- `--error-rate-threshold`: 错误率告警阈值（百分比）
- `--conn-threshold`: 连接数告警阈值
- `--timeout`: 上游探活超时秒数

## Expected Output

```json
{
  "service": "nginx",
  "timestamp": "2026-06-04T10:30:00Z",
  "status": "healthy",
  "checks": {
    "process_alive": {
      "status": "ok",
      "pid": 12345,
      "uptime_seconds": 864000
    },
    "error_rate": {
      "status": "ok",
      "rate_percent": 0.8,
      "total_requests": 150000,
      "error_5xx_count": 1200,
      "sample_window": "1h"
    },
    "connections": {
      "status": "ok",
      "active": 125,
      "reading": 3,
      "writing": 12,
      "waiting": 110
    },
    "upstream_reachable": {
      "status": "ok",
      "targets": [
        {"name": "mes-api-1", "status": "up", "response_ms": 45},
        {"name": "mes-api-2", "status": "up", "response_ms": 52}
      ]
    },
    "response_time": {
      "status": "ok",
      "avg_ms": 85,
      "p95_ms": 210,
      "p99_ms": 450
    }
  },
  "exit_code": 0
}
```

**退出码说明：** 0=全部正常, 1=存在告警, 2=存在严重问题

## Troubleshooting

### 进程不存在

```bash
# 检查 Nginx 是否安装
which nginx
# 检查配置语法
nginx -t
# 启动服务
systemctl start nginx
```

### stub_status 未启用

在 Nginx 配置中添加：

```nginx
location /nginx_status {
    stub_status on;
    allow 127.0.0.1;
    deny all;
}
```

### 日志路径错误

确认日志路径：`nginx -T | grep access_log`

### 连接数异常高

检查是否有慢连接或 DDoS：`ss -s` 查看 socket 统计，`netstat -an | grep ESTABLISHED | wc -l` 确认连接数。

### 上游服务不可达

检查上游配置：`nginx -T | grep upstream`，确认后端服务是否正常运行。
