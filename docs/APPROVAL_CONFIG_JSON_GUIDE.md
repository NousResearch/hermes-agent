# Hermes Agent JSON 配置 - 授权系统指南

## 📋 概述

本文档介绍如何在 JSON 配置文件中配置 Hermes Agent 的授权系统。JSON 配置提供更清晰的结构和更好的可维护性。

## 🎯 JSON 配置示例

### 完整示例

```json
{
  "$schema": "https://hermes-agent.dev/schemas/config.v1.json",
  "_version": 1,
  
  "approvals": {
    "mode": "blocking",
    "timeout": 120,
    "gateway_timeout": 300,
    
    "auto_allow": [
      "^git\\s+(status|log|branch)",
      "^npm\\s+install\\s+(?!-g)",
      "^cargo\\s+(build|test)"
    ],
    
    "auto_deny": [
      "^rm\\s+-rf\\s+/",
      "^chmod\\s+.*777",
      "^curl.*\\|.*sh"
    ]
  }
}
```

---

## 🔧 配置项说明

### approvals.mode - 授权模式

**类型:** `string`  
**可选值:** `"manual"` | `"blocking"` | `"smart"` | `"off"`  
**默认值:** `"manual"`

```json
{
  "approvals": {
    "mode": "blocking"
  }
}
```

**模式说明:**

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `"manual"` | 默认模式，危险命令提示用户授权 | 大多数使用场景 |
| `"blocking"` | **强制等待**用户授权，不跳过 | 生产环境、关键任务 |
| `"smart"` | AI 智能判断，低风险自动通过 | 希望减少提示的场景 |
| `"off"` | 禁用所有授权检查 | 完全可信的开发环境 |

---

### approvals.timeout - 超时配置

**类型:** `number` (秒)  
**默认值:** `60`

```json
{
  "approvals": {
    "timeout": 120,
    "gateway_timeout": 300
  }
}
```

**配置项:**
- `timeout` - CLI 模式下的授权超时（秒）
- `gateway_timeout` - Gateway 模式下的授权超时（秒）

---

### approvals.auto_allow - 自动允许列表

**类型:** `string[]` (正则表达式数组)  
**默认值:** `[]`

```json
{
  "approvals": {
    "auto_allow": [
      "^git\\s+(status|log|diff|branch)",
      "^npm\\s+install\\s+(?!-g)",
      "^yarn\\s+(install|add)",
      "^cargo\\s+(build|test|check)",
      "^make\\s+(build|test)",
      "^mkdir\\s+-p\\s+\\./",
      "^cp\\s+\\./",
      "^docker\\s+(build|run|compose)"
    ]
  }
}
```

**说明:**
- 匹配的命令会**自动执行**，无需授权提示
- 使用 JavaScript/Python 正则表达式语法
- 支持复杂匹配（否定预查、分组等）

---

### approvals.auto_deny - 自动拒绝列表

**类型:** `string[]` (正则表达式数组)  
**默认值:** `[]`

```json
{
  "approvals": {
    "auto_deny": [
      "^rm\\s+-rf\\s+/",
      "^rm\\s+-rf\\s+\\$",
      "^chmod\\s+.*777",
      "^echo.*>\\s+/etc/",
      "^curl.*\\|.*sh",
      "^wget.*\\|.*sh",
      "^mkfs",
      "^dd\\s+"
    ]
  }
}
```

**说明:**
- 匹配的命令会**自动拒绝**，甚至不会提示用户
- 优先级最高，即使在其他列表中也会被拒绝
- 用于保护极度危险的操作

---

## 📝 配置场景示例

### 场景 1: 生产环境（严格）

```json
{
  "approvals": {
    "mode": "blocking",
    "timeout": 600,
    "gateway_timeout": 600,
    
    "auto_allow": [],
    
    "auto_deny": [
      "^rm\\s+-rf\\s+/",
      "^mkfs",
      "^dd\\s+",
      "^chmod\\s+.*777"
    ]
  }
}
```

**特点:**
- ✅ 强制等待所有授权
- ✅ 10 分钟超时
- ✅ 不允许自动通过任何危险命令
- ✅ 自动拒绝极度危险操作

---

### 场景 2: 开发环境（平衡）

```json
{
  "approvals": {
    "mode": "manual",
    "timeout": 120,
    "gateway_timeout": 300,
    
    "auto_allow": [
      "^git\\s+(status|log|diff|branch|tag|remote|fetch)",
      "^npm\\s+install\\s+(?!-g)",
      "^npm\\s+ci\\b",
      "^yarn\\s+(install|add)",
      "^cargo\\s+(build|test|check|run)",
      "^make\\s+(build|test|all|clean)",
      "^mkdir\\s+-p\\s+\\./",
      "^cp\\s+\\./",
      "^mv\\s+\\./",
      "^touch\\s+",
      "^rm\\s+(?!-rf)",
      "^docker\\s+(build|run|compose)"
    ],
    
    "auto_deny": [
      "^rm\\s+-rf\\s+/",
      "^rm\\s+-rf\\s+\\$",
      "^rm\\s+-rf\\s+~",
      "^chmod\\s+.*777",
      "^chmod\\s+.*666",
      "^echo.*>\\s+/etc/",
      "^curl.*\\|.*sh",
      "^wget.*\\|.*sh"
    ]
  }
}
```

**特点:**
- ✅ 常见开发命令自动通过
- ✅ 危险命令仍然提示
- ✅ 极度危险命令自动拒绝
- ✅ 平衡安全性和效率

---

### 场景 3: 开发环境（高效）

```json
{
  "approvals": {
    "mode": "smart",
    "timeout": 120,
    
    "auto_allow": [
      "^git\\s+",
      "^npm\\s+(install|run|test|build)",
      "^yarn\\s+",
      "^pnpm\\s+",
      "^cargo\\s+(build|test|run)",
      "^make\\s+",
      "^docker\\s+(build|run|compose)"
    ],
    
    "auto_deny": [
      "^rm\\s+-rf\\s+/",
      "^sudo\\s+rm",
      "^chmod\\s+.*777"
    ]
  }
}
```

**特点:**
- ✅ AI 智能判断风险
- ✅ 低风险命令自动通过
- ✅ 高风险命令提示用户
- ✅ 最大化开发效率

---

### 场景 4: CI/CD 环境（自动化）

```json
{
  "approvals": {
    "mode": "off"
  }
}
```

**特点:**
- ⚠️ 完全禁用授权检查
- ⚠️ 仅在完全可信的环境中使用
- ✅ 适合 CI/CD 自动化流程

---

## 🔍 正则表达式语法

### 基础语法

| 模式 | 说明 | 示例 |
|------|------|------|
| `^` | 匹配命令开头 | `"^rm"` 匹配以 rm 开头的命令 |
| `$` | 匹配命令结尾 | `"test$"` 匹配以 test 结尾 |
| `\s+` | 匹配一个或多个空白 | `"rm\\s+-rf"` 匹配 `rm -rf` |
| `*` | 匹配零个或多个 | `".*"` 匹配任意字符 |
| `+` | 匹配一个或多个 | `"\\s+"` 匹配一个或多个空白 |
| `?` | 匹配零个或一个 | `"s?"` 匹配可选的 s |
| `()` | 分组 | `"git\\s+(status|log)"` |
| `[]` | 字符集合 | `"[0-9]"` 匹配数字 |
| `|` | 或 | `"build|test"` |

### 高级语法

#### 否定预查 `(?!pattern)`

```json
{
  "auto_allow": [
    "^npm\\s+install\\s+(?!-g)",
    "^pip\\s+install\\s+(?!-\\-user)"
  ]
}
```

**说明:** 匹配 `npm install` 但不匹配 `npm install -g`

#### 肯定预查 `(?=pattern)`

```json
{
  "auto_allow": [
    "^docker\\s+(?=build|run|compose)"
  ]
}
```

**说明:** 只匹配 docker build/run/compose

---

## 🧪 测试配置

### 1. 验证 JSON 格式

```bash
# 验证 JSON 语法
python -m json.tool ~/.hermes/config.json > /dev/null && echo "✓ JSON valid"
```

### 2. 测试配置加载

```bash
# 显示当前配置
python -c "
from hermes_cli.config_json import load_config_json
import json
config = load_config_json()
print(json.dumps(config.get('approvals', {}), indent=2))
"
```

### 3. 测试授权功能

```bash
# 测试危险命令检测
python -c "
from tools.approval import detect_dangerous_command

test_commands = [
    'git status',
    'rm -rf /tmp/test',
    'npm install',
    'chmod 777 /etc/passwd'
]

for cmd in test_commands:
    is_dangerous, key, desc = detect_dangerous_command(cmd)
    print(f'{cmd}: dangerous={is_dangerous}, reason={desc}')
"
```

---

## 📊 配置对比：JSON vs YAML

### JSON 格式（推荐）

```json
{
  "approvals": {
    "mode": "blocking",
    "timeout": 120,
    "auto_allow": [
      "^git\\s+status",
      "^npm\\s+install"
    ],
    "auto_deny": [
      "^rm\\s+-rf\\s+/"
    ]
  }
}
```

**优点:**
- ✅ 结构清晰，易于解析
- ✅ 支持注释（通过 `$schema`）
- ✅ 更好的 IDE 支持
- ✅ 与新的配置系统一致

### YAML 格式（传统）

```yaml
approvals:
  mode: blocking
  timeout: 120
  auto_allow:
    - "^git\\s+status"
    - "^npm\\s+install"
  auto_deny:
    - "^rm\\s+-rf\\s+/"
```

**优点:**
- ✅ 简洁，人类可读性好
- ✅ 向后兼容

---

## 🔐 安全最佳实践

### 1. 使用环境变量

```json
{
  "approvals": {
    "mode": "${APPROVALS_MODE:-blocking}",
    "timeout": "${APPROVALS_TIMEOUT:-120}"
  }
}
```

### 2. 分层配置

```json
{
  "approvals": {
    "auto_allow": [
      "^git\\s+(status|log|branch)"
    ],
    "auto_deny": [
      "^rm\\s+-rf\\s+/",
      "^chmod\\s+.*777"
    ]
  }
}
```

### 3. 定期审查

```bash
# 查看授权配置
python -c "
from hermes_cli.config_json import load_config_json
config = load_config_json()
approvals = config.get('approvals', {})
print('Mode:', approvals.get('mode'))
print('Auto Allow:', len(approvals.get('auto_allow', [])), 'patterns')
print('Auto Deny:', len(approvals.get('auto_deny', [])), 'patterns')
"
```

---

## 🛠️ 故障排除

### 问题 1: 配置不生效

**检查:**
```bash
# 1. 验证 JSON 语法
python -m json.tool ~/.hermes/config.json

# 2. 检查配置加载
python -c "from hermes_cli.config_json import load_config_json; print(load_config_json())"
```

**解决:** 确保 JSON 语法正确，使用双引号

### 问题 2: 正则表达式不匹配

**检查:**
```python
import re
pattern = "^git\\s+status"
command = "git status"
print(re.search(pattern, command))  # 应该输出 Match 对象
```

**解决:** 确保正确转义反斜杠（JSON 中需要 `\\`）

### 问题 3: Blocking 模式不等待

**检查:**
```bash
# 查看当前模式
python -c "
from hermes_cli.config_json import load_config_json
config = load_config_json()
print('Mode:', config.get('approvals', {}).get('mode'))
"
```

**解决:** 确保 `mode` 设置为 `"blocking"`

---

## 📚 相关文档

- **YAML 配置指南:** `docs/APPROVAL_CONFIG_GUIDE.md`
- **配置示例:** `config.json.approvals-example`
- **实现总结:** `APPROVAL_SYSTEM_IMPROVEMENT.md`
- **测试用例:** `tests/test_approval_features.py`

---

## 🎯 快速参考

### 最小配置

```json
{
  "approvals": {
    "mode": "blocking"
  }
}
```

### 推荐配置（开发）

```json
{
  "approvals": {
    "mode": "manual",
    "timeout": 120,
    "auto_allow": [
      "^git\\s+",
      "^npm\\s+(install|run|test)"
    ],
    "auto_deny": [
      "^rm\\s+-rf\\s+/"
    ]
  }
}
```

### 推荐配置（生产）

```json
{
  "approvals": {
    "mode": "blocking",
    "timeout": 600,
    "auto_allow": [],
    "auto_deny": [
      "^rm\\s+-rf\\s+/",
      "^mkfs",
      "^dd\\s+"
    ]
  }
}
```

---

**最后更新:** 2026-04-17  
**版本:** v0.9.0  
**格式:** JSON Configuration  
**维护者:** Hermes Agent Team
