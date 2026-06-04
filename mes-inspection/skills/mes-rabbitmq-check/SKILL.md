---
name: mes-rabbitmq-check
description: RabbitMQ 巡检 — 队列深度、消费者、内存、磁盘。
version: 1.0.0
author: MES AI Inspection
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [mes, inspection, rabbitmq, mq]
    category: devops
---

# MES RabbitMQ Check Skill

## Overview

对 MES 系统的 RabbitMQ 消息队列进行健康巡检，包括队列深度、消费者数量、内存使用、磁盘空间和连接数监控。适用于 Cron 定时调度场景。

## How to Run

```bash
# 在 MES 巡检项目根目录执行（使用 Management API）
python3 scripts/rabbitmq_check.py

# 指定 RabbitMQ 管理端点
python3 scripts/rabbitmq_check.py --api-url http://10.0.1.20:15672

# 指定认证信息
python3 scripts/rabbitmq_check.py --user mes_user --password-file /etc/mes/mq_password

# 检查指定队列
python3 scripts/rabbitmq_check.py --queues mes.order.queue,mes.production.queue
```

## Quick Reference

| 检查项 | API 端点 | 告警阈值 | 严重阈值 |
|--------|----------|----------|----------|
| 队列深度 | `/api/queues` | >10000 | >50000 |
| 消费者数量 | `/api/queues` | <1 | 0 |
| 内存使用 | `/api/nodes` | >80% | >90% |
| 磁盘空间 | `/api/nodes` | <2GB | <500MB |
| 连接数 | `/api/connections` | >500 | >800 |
| 未确认消息 | `/api/queues` | >1000 | >5000 |

### 关键参数

- `--api-url`: RabbitMQ Management API 地址
- `--user`: API 认证用户名
- `--password-file`: 密码文件路径（避免明文密码）
- `--queues`: 指定要检查的队列列表（逗号分隔）
- `--queue-threshold`: 队列深度告警阈值
- `--memory-threshold`: 内存告警阈值（百分比）

## Expected Output

```json
{
  "service": "rabbitmq",
  "timestamp": "2026-06-04T10:30:00Z",
  "status": "healthy",
  "checks": {
    "node_status": {
      "status": "ok",
      "node_name": "rabbit@mes-mq-01",
      "running": true,
      "uptime_seconds": 2592000,
      "erlang_version": "26.2"
    },
    "queues": [
      {
        "name": "mes.order.queue",
        "status": "ok",
        "messages": 1250,
        "messages_ready": 1200,
        "messages_unacked": 50,
        "consumers": 3,
        "consumer_utilization": 0.95
      },
      {
        "name": "mes.production.queue",
        "status": "ok",
        "messages": 320,
        "messages_ready": 300,
        "messages_unacked": 20,
        "consumers": 2,
        "consumer_utilization": 0.88
      }
    ],
    "memory": {
      "status": "ok",
      "used_mb": 2048,
      "limit_mb": 4096,
      "usage_percent": 50.0
    },
    "disk": {
      "status": "ok",
      "free_mb": 51200,
      "limit_mb": 1024,
      "alarm_active": false
    },
    "connections": {
      "status": "ok",
      "total": 125,
      "channels": 250
    }
  },
  "exit_code": 0
}
```

**退出码说明：** 0=全部正常, 1=存在告警, 2=存在严重问题

## Troubleshooting

### 队列积压

检查消费者是否存活：`rabbitmqctl list_consumers`

确认消费者应用日志是否有错误。检查是否有消息处理失败导致的死信。

### 消费者为 0

```bash
# 检查队列绑定
rabbitmqctl list_bindings
# 检查连接
rabbitmqctl list_connections
# 重启消费者服务
systemctl restart mes-consumer
```

### 内存告警

```bash
# 查看内存详情
rabbitmqctl status | grep -A 10 memory
# 清理空闲连接
rabbitmqctl close_connection "<connection_id>" "memory cleanup"
```

### 磁盘空间不足

RabbitMQ 会阻塞生产者。清理旧消息或扩容磁盘：

```bash
# 查看磁盘使用
df -h /var/lib/rabbitmq
# 清理未使用队列
rabbitmqctl delete_queue <queue_name>
```

### Management API 不可达

确认插件已启用：`rabbitmq-plugins enable rabbitmq_management`

检查防火墙是否开放 15672 端口。
