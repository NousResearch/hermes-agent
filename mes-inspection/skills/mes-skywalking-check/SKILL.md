---
name: mes-skywalking-check
description: SkyWalking 巡检 — SLA、P95响应时间、慢接口、告警。
version: 1.0.0
author: MES AI Inspection
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [mes, inspection, skywalking, apm, tracing]
    category: devops
---

# MES SkyWalking Check Skill

## Overview

对 MES 系统的 SkyWalking APM 进行健康巡检，包括服务 SLA、P95 响应时间、慢接口检测、告警状态和存储后端健康度。适用于 Cron 定时调度场景。

## How to Run

```bash
# 在 MES 巡检项目根目录执行
python3 scripts/skywalking_check.py

# 指定 SkyWalking OAP 地址
python3 scripts/skywalking_check.py --oap-url http://10.0.1.50:12800

# 指定 GraphQL 端点
python3 scripts/skywalking_check.py --graphql-url http://10.0.1.50:12800/graphql

# 检查指定服务
python3 scripts/skywalking_check.py --services "mes-api,mes-gateway,mes-worker"

# 设置时间窗口（最近 N 分钟）
python3 scripts/skywalking_check.py --time-range 30
```

## Quick Reference

| 检查项 | GraphQL 查询 | 告警阈值 | 严重阈值 |
|--------|--------------|----------|----------|
| 服务 SLA | `getTopNServiceThroughput` | <99% | <95% |
| P95 响应时间 | `getServicePercentile` | >1000ms | >3000ms |
| 慢接口数量 | `getSlowEndpoints` | >5 | >20 |
| 活跃告警 | `readAlarms` | >0 | >3 |
| 存储健康 | OAP REST API | 延迟高 | 不可达 |
| Trace 采样率 | OAP 配置 | <100% | <50% |

### 关键参数

- `--oap-url`: SkyWalking OAP 服务地址
- `--graphql-url`: GraphQL API 端点
- `--services`: 指定要检查的服务列表（逗号分隔）
- `--time-range`: 检查时间窗口（分钟，默认 30）
- `--sla-threshold`: SLA 告警阈值（百分比）
- `--p95-threshold`: P95 响应时间告警阈值（毫秒）

## Expected Output

```json
{
  "service": "skywalking",
  "timestamp": "2026-06-04T10:30:00Z",
  "status": "healthy",
  "time_range_minutes": 30,
  "checks": {
    "oap_status": {
      "status": "ok",
      "version": "9.7.0",
      "storage_type": "elasticsearch",
      "response_ms": 25
    },
    "services": [
      {
        "name": "mes-api",
        "status": "ok",
        "sla_percent": 99.95,
        "avg_response_ms": 85,
        "p50_ms": 45,
        "p95_ms": 210,
        "p99_ms": 520,
        "throughput_per_min": 12500,
        "error_rate_percent": 0.05
      },
      {
        "name": "mes-gateway",
        "status": "ok",
        "sla_percent": 99.98,
        "avg_response_ms": 120,
        "p50_ms": 80,
        "p95_ms": 350,
        "p99_ms": 800,
        "throughput_per_min": 15000,
        "error_rate_percent": 0.02
      }
    ],
    "slow_endpoints": {
      "status": "ok",
      "count": 2,
      "endpoints": [
        {
          "service": "mes-api",
          "endpoint": "/api/v1/orders/batch-query",
          "avg_ms": 2500,
          "p95_ms": 4200,
          "calls_per_min": 150
        }
      ]
    },
    "alerts": {
      "status": "ok",
      "active_count": 0,
      "recent_alerts": []
    },
    "storage": {
      "status": "ok",
      "type": "elasticsearch",
      "response_ms": 15,
      "index_size_gb": 120.5
    }
  },
  "exit_code": 0
}
```

**退出码说明：** 0=全部正常, 1=存在告警, 2=存在严重问题

## Troubleshooting

### OAP 服务不可达

```bash
# 检查 OAP 进程
systemctl status skywalking-oap
# 查看日志
tail -f /var/log/skywalking/oap.log
# 检查端口
ss -tlnp | grep 12800
```

### SLA 突然下降

查看具体错误：SkyWalking UI → 服务 → 错误追踪

检查上游依赖是否异常（数据库、MQ、外部 API）。

### 慢接口排查

```bash
# 查看 Trace 详情
curl -XPOST http://localhost:12800/graphql -H 'Content-Type: application/json' -d '{
  "query": "query { queryBasicTraces(condition: {serviceId: \"mes-api\", queryDuration: {start: \"2026-06-04 1000\", end: \"2026-06-04 1030\", step: MINUTE}, paging: {pageNum: 1, pageSize: 10, needTotal: true}, queryOrder: BY_DURATION}) { traces { key: segmentId endpointNames duration start } } }"
}'
```

### 存储后端问题

如果是 Elasticsearch 存储：

```bash
# 检查 ES 集群状态
curl -s "http://localhost:9200/_cluster/health"
# 检查 SkyWalking 索引
curl -s "http://localhost:9200/_cat/indices/sw*?v"
```

### 告警规则配置

检查告警规则文件：`config/alarm-settings.yml`

确认告警 webhook 是否正常：`curl -XPOST http://localhost:12800/alarm/test`
