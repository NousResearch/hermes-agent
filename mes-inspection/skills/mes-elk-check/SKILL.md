---
name: mes-elk-check
description: ELK 巡检 — 集群状态、分片、JVM堆、磁盘使用率。
version: 1.0.0
author: MES AI Inspection
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [mes, inspection, elk, elasticsearch, kibana]
    category: devops
---

# MES ELK Check Skill

## Overview

对 MES 系统的 ELK（Elasticsearch + Logstash + Kibana）集群进行健康巡检，包括集群状态、分片分配、JVM 堆内存、磁盘使用率和索引健康度。适用于 Cron 定时调度场景。

## How to Run

```bash
# 在 MES 巡检项目根目录执行
python3 scripts/elk_check.py

# 指定 Elasticsearch 节点
python3 scripts/elk_check.py --es-url http://10.0.1.40:9200

# 多节点集群
python3 scripts/elk_check.py --es-nodes "http://10.0.1.40:9200,http://10.0.1.41:9200"

# 指定认证
python3 scripts/elk_check.py --user elastic --password-file /etc/mes/es_password

# 检查特定索引模式
python3 scripts/elk_check.py --index-pattern "mes-*"
```

## Quick Reference

| 检查项 | API 端点 | 告警阈值 | 严重阈值 |
|--------|----------|----------|----------|
| 集群状态 | `/_cluster/health` | yellow | red |
| 未分配分片 | `/_cluster/health` | >0 | >5 |
| JVM 堆使用率 | `/_nodes/stats` | >75% | >90% |
| 磁盘使用率 | `/_nodes/stats` | >80% | >90% |
| 索引数量 | `/_cat/indices` | >500 | >1000 |
| 待处理任务 | `/_cluster/pending_tasks` | >10 | >50 |

### 关键参数

- `--es-url`: Elasticsearch 主节点 URL
- `--es-nodes`: 多节点 URL（逗号分隔）
- `--user`: 认证用户名
- `--password-file`: 密码文件路径
- `--index-pattern`: 索引匹配模式（默认 `mes-*`）
- `--disk-threshold`: 磁盘告警阈值（百分比）

## Expected Output

```json
{
  "service": "elk",
  "timestamp": "2026-06-04T10:30:00Z",
  "status": "healthy",
  "checks": {
    "cluster_health": {
      "status": "ok",
      "cluster_name": "mes-elk-cluster",
      "cluster_status": "green",
      "active_nodes": 3,
      "active_primary_shards": 150,
      "active_shards": 300,
      "relocating_shards": 0,
      "initializing_shards": 0,
      "unassigned_shards": 0
    },
    "nodes": [
      {
        "name": "es-node-1",
        "status": "ok",
        "role": "data",
        "jvm_heap_used_mb": 8192,
        "jvm_heap_max_mb": 16384,
        "jvm_heap_percent": 50.0,
        "disk_used_percent": 65.0,
        "disk_available_gb": 250.5,
        "cpu_percent": 35.0,
        "index_count": 120,
        "docs_count": 50000000
      }
    ],
    "indices": {
      "status": "ok",
      "total_count": 120,
      "green_count": 118,
      "yellow_count": 2,
      "red_count": 0,
      "total_size_gb": 450.5
    },
    "pending_tasks": {
      "status": "ok",
      "count": 0
    }
  },
  "exit_code": 0
}
```

**退出码说明：** 0=全部正常, 1=存在告警, 2=存在严重问题

## Troubleshooting

### 集群状态为 RED

```bash
# 查看未分配分片原因
curl -s "http://localhost:9200/_cluster/allocation/explain?pretty"

# 检查节点是否掉线
curl -s "http://localhost:9200/_cat/nodes?v"

# 手动重试分配
curl -XPOST "http://localhost:9200/_cluster/reroute?retry_failed=true"
```

### 磁盘空间不足

```bash
# 清理旧索引
curl -XDELETE "http://localhost:9200/mes-logs-2026.01.*"

# 调整水位线
curl -XPUT "http://localhost:9200/_cluster/settings" -H 'Content-Type: application/json' -d '{
  "persistent": {
    "cluster.routing.allocation.disk.watermark.low": "85%",
    "cluster.routing.allocation.disk.watermark.high": "90%"
  }
}'
```

### JVM 堆内存过高

检查是否有大查询或聚合：`GET /_nodes/hot_threads`

调整 `jvm.options` 中的堆大小：`-Xms16g -Xmx16g`

### Logstash 管道卡住

```bash
# 检查 Logstash 状态
curl -s "http://localhost:9600/_node/stats/pipelines?pretty"
# 重启 Logstash
systemctl restart logstash
```

### Kibana 不可用

检查 Kibana 状态：`curl -s "http://localhost:5601/api/status"`

确认 Kibana 可连接 Elasticsearch：`curl -s "http://localhost:5601/api/status" | jq .status.overall`
