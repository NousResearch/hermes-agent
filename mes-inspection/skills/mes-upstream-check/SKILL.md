---
name: mes-upstream-check
description: 通过 nginx_upstream_check_module 检查上游后端节点存活状态。
version: "1.0"
license: MIT
metadata:
  hermes:
    tags: [mes, inspection, upstream, heartbeat, nginx]
    category: devops
---

# MES Upstream Check Skill

通过 nginx_upstream_check_module 状态页一次性获取所有 upstream 后端节点的存活状态。心跳巡检专用（2 分钟间隔）。

## Prerequisites

- Nginx 已安装 `nginx_upstream_check_module`
- Nginx 配置中已启用状态页，例如：

```nginx
location /upstream_status {
    upstream_check;
    upstream_check_type html;
}
```

JSON 格式（推荐）：访问 `/upstream_status?format=json`

## How to Run

```bash
python scripts/upstream_check.py
```

## 检查项

| 检查项 | 数据源 | 判定逻辑 |
|--------|--------|----------|
| Nginx 连通性 | HTTP GET status_url | 连接失败 → CRITICAL |
| 节点状态 | JSON/文本中的 status 字段 | `down` → CRITICAL，`up` → NORMAL |

## 配置

```yaml
upstream:
  # nginx_upstream_check_module 状态页 URL
  status_url: "http://localhost/upstream_status?format=json"
  # HTTP 超时（秒）
  timeout: 5
```

## 状态页格式

### JSON 格式（`?format=json`）

```json
{
  "mes-backend": [
    {"server": "10.0.0.1:8080", "status": "up"},
    {"server": "10.0.0.2:8080", "status": "down"}
  ]
}
```

### 文本格式

```
upstream mes-backend
    server 10.0.0.1:8080 up
    server 10.0.0.2:8080 down
```

两种格式均自动识别。
