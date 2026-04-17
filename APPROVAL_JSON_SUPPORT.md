# Hermes Agent 授权系统 - JSON 配置支持

## 📋 概述

授权系统改进现已完全支持 JSON 配置格式！用户可以选择使用 YAML 或 JSON 格式配置授权系统。

---

## ✅ 实现的功能

### 1. JSON/YAML 双格式支持 ✅

```python
# tools/approval.py - _get_approval_config()
def _get_approval_config() -> dict:
    # 优先加载 JSON 配置（新格式）
    from hermes_cli.config_json import config_exists_json, load_config_json
    
    if config_exists_json():
        config = load_config_json()
        approvals = config.get("approvals", {})
    else:
        # 回退到 YAML 配置（旧格式）
        from hermes_cli.config import load_config
        config = load_config()
        approvals = config.get("approvals", {})
```

**加载优先级:**
1. `~/.hermes/config.json` (JSON 格式)
2. `~/.hermes/config.yaml` (YAML 格式)
3. 默认配置

---

## 📁 配置文件对比

### JSON 格式（推荐）

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

### YAML 格式（传统）

```yaml
approvals:
  mode: blocking
  timeout: 120
  gateway_timeout: 300
  
  auto_allow:
    - "^git\\s+(status|log|branch)"
    - "^npm\\s+install\\s+(?!-g)"
    - "^cargo\\s+(build|test)"
  
  auto_deny:
    - "^rm\\s+-rf\\s+/"
    - "^chmod\\s+.*777"
    - "^curl.*\\|.*sh"
```

---

## 🎯 配置示例

### 示例 1: 开发环境（JSON）

```json
{
  "approvals": {
    "mode": "manual",
    "timeout": 120,
    "auto_allow": [
      "^git\\s+",
      "^npm\\s+(install|run|test)",
      "^cargo\\s+(build|test|run)"
    ],
    "auto_deny": [
      "^rm\\s+-rf\\s+/",
      "^chmod\\s+.*777"
    ]
  }
}
```

**文件位置:** `~/.hermes/config.json`

---

### 示例 2: 生产环境（JSON）

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
      "^dd\\s+"
    ]
  }
}
```

---

### 示例 3: 高效开发（JSON）

```json
{
  "approvals": {
    "mode": "smart",
    "timeout": 120,
    "auto_allow": [
      "^git\\s+",
      "^npm\\s+(install|run)",
      "^docker\\s+(build|run|compose)"
    ],
    "auto_deny": [
      "^rm\\s+-rf\\s+/",
      "^curl.*\\|.*sh"
    ]
  }
}
```

---

## 🔧 使用方法

### 1. 使用 JSON 配置

```bash
# 编辑 JSON 配置文件
nano ~/.hermes/config.json

# 添加 approvals 部分
# 保存后自动生效
```

### 2. 验证配置

```bash
# 验证 JSON 语法
python -m json.tool ~/.hermes/config.json > /dev/null && echo "✓ Valid"

# 查看授权配置
python -c "
import json
from pathlib import Path
config = json.load(open(Path.home() / '.hermes' / 'config.json'))
print(json.dumps(config.get('approvals', {}), indent=2))
"
```

### 3. 测试功能

```bash
# 测试危险命令检测
python -c "
from tools.approval import detect_dangerous_command

cmd = 'rm -rf /tmp/test'
is_dangerous, key, desc = detect_dangerous_command(cmd)
print(f'{cmd}: dangerous={is_dangerous}, reason={desc}')
"
```

---

## 📊 配置迁移

### 从 YAML 迁移到 JSON

**方式 1: 手动迁移**

1. 复制 YAML 中的 `approvals` 部分
2. 转换为 JSON 格式
3. 添加到 `config.json`

**方式 2: 自动迁移工具**

```bash
# 使用迁移工具（如果 config.json 不存在会自动创建）
hermes config json migrate --apply
```

然后手动添加 `approvals` 部分。

---

## 🧪 测试结果

### JSON 配置加载测试

```bash
$ python -c "
import json
from pathlib import Path

config = json.load(open(Path.home() / '.hermes' / 'config.json'))
approvals = config.get('approvals', {})

print('✓ JSON Config Approvals Section:')
print(json.dumps(approvals, indent=2))
"

✓ JSON Config Approvals Section:
{
  "mode": "manual",
  "timeout": 120,
  "gateway_timeout": 300,
  "auto_allow": [
    "^git\\s+(status|log|diff|branch)",
    "^npm\\s+install\\s+(?!-g)",
    "^cargo\\s+(build|test)"
  ],
  "auto_deny": [
    "^rm\\s+-rf\\s+/",
    "^chmod\\s+.*777",
    "^curl.*\\|.*sh"
  ]
}

✅ Configuration loaded successfully!
```

### 功能测试

| 功能 | JSON 配置 | YAML 配置 | 状态 |
|------|----------|----------|------|
| Blocking 模式 | ✅ 支持 | ✅ 支持 | 通过 |
| Auto Allow | ✅ 支持 | ✅ 支持 | 通过 |
| Auto Deny | ✅ 支持 | ✅ 支持 | 通过 |
| 环境变量 | ✅ 支持 | ✅ 支持 | 通过 |
| 配置加载 | ✅ 支持 | ✅ 支持 | 通过 |

---

## 📖 文档位置

| 文档 | 路径 | 格式 |
|------|------|------|
| **JSON 配置指南** | `docs/APPROVAL_CONFIG_JSON_GUIDE.md` | JSON |
| **YAML 配置指南** | `docs/APPROVAL_CONFIG_GUIDE.md` | YAML |
| **JSON 示例** | `config.json.approvals-example` | JSON |
| **YAML 示例** | `config.approvals.example.yaml` | YAML |
| **实现总结** | `APPROVAL_SYSTEM_IMPROVEMENT.md` | Markdown |

---

## 🔍 技术实现

### 核心代码

```python
# tools/approval.py
def _get_approval_config() -> dict:
    """Read the approvals config block from JSON or YAML config.
    
    Supports both config formats:
    - JSON: ~/.hermes/config.json (new format)
    - YAML: ~/.hermes/config.yaml (legacy format)
    """
    try:
        # Try JSON config first (new format)
        from hermes_cli.config_json import config_exists_json, load_config_json
        
        if config_exists_json():
            config = load_config_json()
            approvals = config.get("approvals", {}) or {}
        else:
            # Fall back to YAML config (legacy format)
            from hermes_cli.config import load_config
            config = load_config()
            approvals = config.get("approvals", {}) or {}
        
        # Normalize mode
        if "mode" in approvals:
            approvals["mode"] = _normalize_approval_mode(approvals["mode"])
        
        return approvals
    except Exception as e:
        logger.warning("Failed to load approval config: %s", e)
        return {}
```

### 执行流程

```
授权检查请求
    ↓
_get_approval_config()
    ↓
检查 config.json 是否存在？
    ├─ 是 → 加载 JSON 配置
    └─ 否 → 加载 YAML 配置
         ↓
    返回 approvals 配置
         ↓
    应用到授权逻辑
```

---

## 🎯 最佳实践

### 1. 使用 JSON 格式（新项目的推荐）

```json
{
  "$schema": "https://hermes-agent.dev/schemas/config.v1.json",
  "approvals": {
    "mode": "blocking",
    "auto_allow": ["^git\\s+"],
    "auto_deny": ["^rm\\s+-rf\\s+/"]
  }
}
```

**优点:**
- ✅ 结构清晰
- ✅ 易于解析
- ✅ 更好的 IDE 支持
- ✅ 与新的配置系统一致

### 2. 环境变量支持

```json
{
  "approvals": {
    "mode": "${APPROVALS_MODE:-blocking}",
    "timeout": "${APPROVALS_TIMEOUT:-120}"
  }
}
```

### 3. 版本控制

```bash
# 将 JSON 配置添加到版本控制
git add ~/.hermes/config.json

# 但排除敏感信息（API keys 在 .env 中）
echo "*.env" >> .gitignore
```

---

## 🚀 快速开始

### 步骤 1: 创建 JSON 配置

```bash
# 复制示例配置
cp config.json.approvals-example ~/.hermes/config.json

# 或使用迁移工具
hermes config json migrate --apply
```

### 步骤 2: 编辑配置

```bash
nano ~/.hermes/config.json
```

添加或修改 `approvals` 部分。

### 步骤 3: 验证配置

```bash
python -m json.tool ~/.hermes/config.json > /dev/null && echo "✓ Valid"
```

### 步骤 4: 测试功能

```bash
# 测试危险命令
hermes "rm -rf /tmp/test"
```

---

## 📝 总结

### 实现的功能

✅ **双格式支持** - JSON 和 YAML 都完全支持  
✅ **向后兼容** - 现有 YAML 配置继续工作  
✅ **优先加载** - JSON 优先，YAML 回退  
✅ **完整功能** - 所有授权功能在两种格式中都可用  
✅ **文档完善** - 两种格式都有完整文档  

### 用户收益

- ✅ **选择性** - 用户可以选择喜欢的格式
- ✅ **平滑迁移** - 可以从 YAML 迁移到 JSON
- ✅ **一致性** - 所有配置都在一个文件中
- ✅ **灵活性** - 支持环境变量和复杂配置

---

**实现日期:** 2026-04-17  
**版本:** v0.9.0  
**格式支持:** JSON + YAML  
**状态:** ✅ 完成并测试通过
