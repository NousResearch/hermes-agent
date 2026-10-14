---
name: feishu-calendar
description: 飞书日历管理 - 创建/查询/删除日程，支持 OAuth 2.0 自动刷新 token
version: 1.0.0
author: Hermes Agent Community
license: MIT
required_credential_files:
  - path: .feishu_user_token.json
    description: 飞书 OAuth2 token（由 setup 脚本自动创建）
metadata:
  hermes:
    tags: [Feishu, Lark, Calendar, 日历，OAuth, 日程管理]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [feishu-bot-polling]
---

# 飞书日历管理

创建、查询、删除飞书日历日程，支持 OAuth 2.0 自动刷新 token。

## 架构

```
feishu_calendar.py  →  feishu_oauth.py  →  飞书开放平台 API
(日历操作)            (OAuth 管理)          (user_access_token)
```

- `feishu_oauth.py` 处理 OAuth 2.0 授权流程和 token 自动刷新
- `feishu_calendar.py` 调用飞书日历 API 创建/查询/删除日程
- Token 存储在 `~/.hermes/.feishu_user_token.json`，自动刷新

## 脚本

- `scripts/setup.py` — OAuth2 授权设置（首次使用运行一次）
- `scripts/feishu_oauth.py` — Token 管理（生成链接、刷新、状态检查）
- `scripts/feishu_calendar.py` — 日历 API 操作

## 前置配置

### 1. 飞书开放平台应用

需要在飞书开放平台创建自定义应用：

1. 访问 https://open.feishu.cn/app
2. 创建自定义应用（或编辑现有应用）
3. 记录 `App ID` 和 `App Secret`

### 2. 配置权限

在飞书开放平台 **权限管理** 中添加：

- `calendar:calendar:readonly` - 读取日历
- `calendar:calendar` - 创建/修改日历事件

### 3. 配置重定向 URL

在飞书开放平台 **安全设置** 中添加：

```
http://127.0.0.1:18080/callback
```

### 4. 配置 Hermes 环境变量

在 `~/.hermes/.env` 中添加：

```bash
FEISHU_APP_ID=cli_xxxxxxxxxxxxx
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
FEISHU_REDIRECT_URI=http://127.0.0.1:18080/callback
```

## 首次设置

定义快捷命令：

```bash
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
FEISHU_SKILL_DIR="$HERMES_HOME/skills/productivity/feishu-calendar"
PYTHON_BIN="${HERMES_PYTHON:-python3}"
if [ -x "$HERMES_HOME/hermes-agent/venv/bin/python" ]; then
  PYTHON_BIN="$HERMES_HOME/hermes-agent/venv/bin/python"
fi
FSETUP="$PYTHON_BIN $FEISHU_SKILL_DIR/scripts/setup.py"
FOAUTH="$PYTHON_BIN $FEISHU_SKILL_DIR/scripts/feishu_oauth.py"
FCAL="$PYTHON_BIN $FEISHU_SKILL_DIR/scripts/feishu_calendar.py"
```

### 步骤 1：检查配置

```bash
$FSETUP --check
```

如果显示 `NOT_CONFIGURED`，继续下一步。

### 步骤 2：生成授权链接

```bash
$FOAUTH generate_link
```

复制输出的链接到浏览器打开。

### 步骤 3：飞书授权

1. 在浏览器打开授权链接
2. 登录飞书并同意授权
3. 授权后 URL 会变成：`http://127.0.0.1:18080/callback?code=xxx&state=xxx`
4. 复制 `code=` 后面的值

### 步骤 4：换取 token

```bash
$FOAUTH exchange <code>
```

### 步骤 5：验证

```bash
$FSETUP --check
```

应显示 `AUTHENTICATED`。Token 会自动刷新，无需再次授权（refresh_token 有效期 30 天）。

## 用法

### 创建日程

```bash
$FCAL create --title "会议" --start "2026-04-16 09:00" --end "2026-04-16 10:00"
$FCAL create --title "会议" --start "2026-04-16 09:00" --end "2026-04-16 10:00" --description "项目讨论"
```

### 查询日程

```bash
# 查询今天
$FCAL list

# 查询指定日期
$FCAL list "2026-04-16"
```

### 删除日程

```bash
$FCAL delete --event_id "53bf9415-c0f2-45d3-9941-a9c7a3abfafd_0"
```

### 查看 token 状态

```bash
$FOAUTH status
```

### 手动刷新 token

```bash
$FOAUTH refresh
```

## 输出格式

所有命令返回 JSON 格式：

### 创建日程成功
```json
{
  "success": true,
  "event_id": "xxx_0",
  "message": "日程创建成功",
  "app_link": "https://applink.feishu.cn/..."
}
```

### 查询日程
```json
{
  "success": true,
  "events": [
    {
      "event_id": "xxx_0",
      "title": "会议",
      "start_time": 1776265200,
      "end_time": 1776268800
    }
  ]
}
```

## Token 管理

### 自动刷新

- `access_token` 有效期 2 小时，脚本会自动在过期前 5 分钟刷新
- `refresh_token` 有效期 30 天，每次刷新会获得新的 refresh_token
- 理论上可以一直自动续期，无需手动干预

### 重新授权

如果 refresh_token 过期（30 天后），需要重新运行授权流程：

```bash
$FOAUTH generate_link
# 浏览器授权后
$FOAUTH exchange <code>
```

## 自然语言示例

用户：帮我查看今天的飞书日程
用户：查看我明天的日历
用户：帮我在飞书日历创建一个会议，明天 14:00 到 15:00，标题是"项目评审"
用户：我下周一有什么安排？
用户：帮我创建今晚 11 点的睡觉提醒

## ⚠️ 核心陷阱：Token 类型选择

**这是最大的坑！** 飞书有两种身份验证方式：

| Token 类型 | 身份 | 访问的日历 | 用户能看到事件吗？ |
|-----------|------|-----------|------------------|
| `tenant_access_token` | 应用身份 | 机器人日历 | ❌ **看不到** |
| `user_access_token` | 用户身份 | 用户个人日历 | ✅ **看得到** |

**个人助理场景必须使用 `user_access_token`**（OAuth 2.0 用户授权）！

如果使用 `tenant_access_token`，事件会创建成功但用户看不到（在机器人日历上）。

## 注意事项

| 项目 | 说明 |
|------|------|
| Token 有效期 | access_token 约 2 小时，refresh_token 约 30 天 |
| 权限要求 | 需要 `calendar:calendar:readonly` 和 `calendar:calendar` |
| 首次使用 | 需要完成 OAuth 授权流程 |
| 自动刷新 | 每次调用 API 前自动检查并刷新 token |
| 文件权限 | token 文件自动设置为 600（仅所有者可读） |
| Token 类型 | 必须使用 `user_access_token`（OAuth），不能用 `tenant_access_token` |

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| `NOT_CONFIGURED` | 检查 ~/.hermes/.env 中的 FEISHU_APP_ID 和 FEISHU_APP_SECRET |
| `NOT_AUTHENTICATED` | 运行 `feishu_oauth.py generate_link` 重新授权 |
| `REFRESH_FAILED` | refresh_token 过期，需要重新授权 |
| `99991679 Unauthorized` | 权限不足，在飞书开放平台添加日历权限后重新授权 |
| `20029 重定向 URL 有误` | 检查飞书开放平台的重定向 URL 是否精确匹配 `http://127.0.0.1:18080/callback` |
| 事件创建成功但看不到 | 使用了 `tenant_access_token`，改用 OAuth 获取 `user_access_token` |
| 授权链接打不开 | URL 编码问题，使用 `urllib.parse.quote()` 编码参数 |

## 常见错误详解

### 错误 1：事件创建成功但用户看不到

**症状**：API 返回 success，但飞书 App 里找不到事件

**原因**：使用了 `tenant_access_token`，事件创建在机器人日历上

**解决**：改用 OAuth 2.0 获取 `user_access_token`

### 错误 2：重定向 URL 有误（错误码 20029）

**常见错误**：
- ❌ `http://localhost:18080/callback`（localhost ≠ 127.0.0.1）
- ❌ `http://127.0.0.1:18080`（缺少 /callback）
- ❌ `https://127.0.0.1:18080/callback`（必须是 http）

**解决**：飞书开放平台 → 安全设置，精确配置 `http://127.0.0.1:18080/callback`

### 错误 3：URL 编码问题

**症状**：授权链接打不开或报错

**解决**：使用 urllib.parse 编码：
```python
import urllib.parse
redirect_uri = urllib.parse.quote("http://127.0.0.1:18080/callback", safe='')
scope = urllib.parse.quote("calendar:calendar", safe='')
```

## 撤销授权

在飞书开放平台的应用管理中可以撤销应用授权，或删除 `~/.hermes/.feishu_user_token.json` 文件。

## 安全说明

- Token 存储在本地文件 `~/.hermes/.feishu_user_token.json`
- 文件权限自动设置为 600（仅所有者可读写）
- 不要将 token 文件上传到版本控制或分享给他人
- 仅申请必要的日历权限，不访问消息、联系人等其他数据
