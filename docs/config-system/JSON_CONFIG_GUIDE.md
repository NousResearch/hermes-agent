# Hermes Agent JSON 配置系统

> 📋 新的 JSON 配置格式，灵感来自 openclaw.json，提供更清晰、更易管理的配置体验。

## 🎯 为什么要迁移到 JSON 格式？

### 当前 YAML 配置的问题

```yaml
# ❌ 问题 1: API Key 重复配置
custom_providers:
- name: bailian
  api_key: sk-xxx...  # 第一次出现

auxiliary:
  vision:
    api_key: sk-xxx...  # 又要配？
  web_extract:
    api_key: sk-xxx...  # 又要配？
  compression:
    api_key: sk-xxx...  # 第 N 次...

# ❌ 问题 2: 模型配置分散
model:
  default: qwen3.5-plus
  provider: bailian

# 想加个新模型？要在多处修改...
```

### JSON 配置的优势

```json
{
  // ✅ 优势 1: Provider 集中管理
  "providers": {
    "bailian": {
      "base_url": "https://...",
      "api_key": "${BAILIAN_API_KEY}",  // 一次配置，处处使用
      "models": [
        { "id": "qwen3.5-plus", ... },
        { "id": "qwen3.6-plus", ... },
        { "id": "glm-5", ... }
      ]
    }
  },
  
  // ✅ 优势 2: 清晰的结构
  "defaults": {
    "primary_model": "bailian/qwen3.5-plus"
  },
  
  "features": {
    "vision": { "provider": "bailian", "model": "qwen3.5-plus" },
    "compression": { "provider": "bailian", "model": "qwen3.5-plus" }
  }
}
```

## 📦 配置文件结构

### 完整示例

```json
{
  "$schema": "https://hermes-agent.dev/schemas/config.v1.json",
  "_version": 1,
  
  "providers": {
    "bailian": {
      "base_url": "https://coding.dashscope.aliyuncs.com/v1",
      "api_key": "${BAILIAN_API_KEY}",
      "models": [
        {
          "id": "qwen3.5-plus",
          "name": "Qwen 3.5 Plus",
          "context_window": 1000000,
          "max_tokens": 65536,
          "supports_vision": true
        }
      ]
    }
  },
  
  "defaults": {
    "primary_model": "bailian/qwen3.5-plus",
    "max_turns": 90,
    "personality": "kawaii"
  },
  
  "features": {
    "vision": {
      "provider": "bailian",
      "model": "qwen3.5-plus"
    },
    "session_search": {
      "provider": "bailian",
      "model": "text-embedding-v4"
    }
  },
  
  "toolsets": ["hermes-cli"],
  
  "terminal": {
    "backend": "local",
    "timeout": 180,
    "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20"
  },
  
  "display": {
    "compact": false,
    "streaming": true,
    "skin": "default"
  },
  
  "memory": {
    "enabled": true,
    "char_limit": 2200
  },
  
  "security": {
    "redact_secrets": true,
    "tirith_enabled": true
  },
  
  "platforms": {
    "feishu": {
      "enabled": true,
      "app_id": "${FEISHU_APP_ID}",
      "app_secret": "${FEISHU_APP_SECRET}"
    }
  }
}
```

## 🔧 环境变量

### ${VAR} 语法

JSON 配置支持环境变量引用：

```json
{
  "providers": {
    "bailian": {
      "api_key": "${BAILIAN_API_KEY}"
    }
  },
  "platforms": {
    "feishu": {
      "app_id": "${FEISHU_APP_ID}",
      "app_secret": "${FEISHU_APP_SECRET}"
    }
  }
}
```

### 环境变量来源

1. **系统环境变量** (`os.environ`)
2. **~/.hermes/.env 文件**

```bash
# ~/.hermes/.env
BAILIAN_API_KEY=sk-sp-xxx
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx
```

## 🚀 快速开始

### 1. 查看当前配置

```bash
# 查看当前配置（自动检测格式）
python ~/.hermes/hermes-agent/hermes_cli/config_json.py show
```

### 2. 迁移现有配置

```bash
# 预览迁移结果
python ~/.hermes/hermes-agent/scripts/migrate_config.py

# 应用迁移
python ~/.hermes/hermes-agent/scripts/migrate_config.py --apply

# 对比格式差异
python ~/.hermes/hermes-agent/scripts/migrate_config.py --compare
```

### 3. 手动创建配置

```bash
# 复制示例配置
cp ~/.hermes/config.json.example ~/.hermes/config.json

# 编辑配置
nano ~/.hermes/config.json
```

## 📊 配置项说明

### providers

集中管理所有 LLM Provider 配置。

```json
"providers": {
  "provider_name": {
    "base_url": "API 端点",
    "api_key": "${ENV_VAR}",
    "models": [
      {
        "id": "model-id",
        "name": "显示名称",
        "context_window": 1000000,
        "max_tokens": 65536,
        "supports_vision": true,
        "supports_reasoning": false
      }
    ]
  }
}
```

### defaults

默认设置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `primary_model` | string | - | 主模型 (格式：provider/model-id) |
| `fallback_model` | string | - | 备用模型 |
| `max_turns` | number | 90 | 最大对话轮数 |
| `personality` | string | "kawaii" | 默认人格 |

### features

辅助功能配置（vision、compression 等）。

```json
"features": {
  "vision": {
    "provider": "bailian",
    "model": "qwen3.5-plus",
    "timeout": 120
  },
  "compression": {
    "provider": "bailian",
    "model": "qwen3.5-plus"
  }
}
```

### terminal

终端配置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | string | "local" | 后端类型 (local/docker/ssh) |
| `timeout` | number | 180 | 命令超时时间（秒） |
| `docker_image` | string | - | Docker 镜像 |
| `persistent_shell` | boolean | true | 持久化 shell |

### display

显示配置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `compact` | boolean | false | 紧凑模式 |
| `streaming` | boolean | true | 流式输出 |
| `show_cost` | boolean | false | 显示费用 |
| `skin` | string | "default" | 主题皮肤 |

### memory

记忆配置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | true | 启用记忆 |
| `char_limit` | number | 2200 | 字符限制 |

### security

安全配置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `redact_secrets` | boolean | true | 脱敏密钥 |
| `tirith_enabled` | boolean | true | 启用安全扫描 |

### platforms

平台配置（Feishu、Telegram 等）。

```json
"platforms": {
  "feishu": {
    "enabled": true,
    "app_id": "${FEISHU_APP_ID}",
    "app_secret": "${FEISHU_APP_SECRET}",
    "domain": "feishu",
    "connection_mode": "websocket"
  }
}
```

## 🔄 迁移指南

### 自动迁移

```bash
# 1. 预览
python scripts/migrate_config.py

# 2. 对比
python scripts/migrate_config.py --compare

# 3. 应用
python scripts/migrate_config.py --apply
```

### 手动迁移步骤

1. **备份现有配置**
   ```bash
   cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak
   ```

2. **创建 config.json**
   ```bash
   cp ~/.hermes/config.json.example ~/.hermes/config.json
   ```

3. **迁移 Provider 配置**
   - 将 `custom_providers` 移到 `providers`
   - 将 API Key 改为 `${ENV_VAR}` 引用

4. **迁移辅助模型**
   - 将 `auxiliary.*` 移到 `features.*`

5. **验证配置**
   ```bash
   python hermes_cli/config_json.py show
   ```

## 🛠️ API 参考

### Python API

```python
from hermes_cli.config_json import (
    load_config_json,
    load_config_unified,
    migrate_yaml_to_json
)

# 加载 JSON 配置
config = load_config_json()

# 自动检测格式加载
config = load_config_unified()

# 迁移配置
result = migrate_yaml_to_json(dry_run=True)
if result["success"]:
    print("Migration preview:", result["config"])
```

### CLI 命令

```bash
# 显示配置
python config_json.py show

# 迁移配置
python config_json.py migrate
python config_json.py migrate --dry-run
```

## ⚠️ 注意事项

### 向后兼容

- JSON 配置和 YAML 配置可以共存
- 系统优先加载 `config.json`
- 如果 `config.json` 不存在，回退到 `config.yaml`

### API Key 安全

- ✅ 使用 `${ENV_VAR}` 引用环境变量
- ✅ 将真实密钥放在 `.env` 文件中
- ❌ 不要在 `config.json` 中硬编码密钥

### 迁移后验证

```bash
# 1. 检查配置加载
python config_json.py show

# 2. 测试 Hermes 启动
hermes --version

# 3. 测试对话
hermes "Hello"
```

## 📝 常见问题

### Q: 迁移后原有的 config.yaml 还在吗？

A: 是的，迁移工具不会删除原文件。确认新配置工作正常后，可以手动删除或保留作为备份。

### Q: 可以混合使用 JSON 和 YAML 吗？

A: 不建议。系统会优先加载 `config.json`，如果存在则忽略 `config.yaml`。

### Q: 如何回退到 YAML 格式？

A: 删除或重命名 `config.json`，系统会自动回退到 `config.yaml`。

```bash
mv ~/.hermes/config.json ~/.hermes/config.json.bak
```

### Q: 环境变量不生效怎么办？

A: 检查：
1. `.env` 文件路径是否正确 (`~/.hermes/.env`)
2. 变量名是否匹配 (`${VAR_NAME}`)
3. 重启 Hermes 使环境变量生效

## 🎉 贡献

欢迎贡献！请查看：
- [GitHub Issues](https://github.com/hermes-agent/hermes-agent/issues)
- [Pull Requests](https://github.com/hermes-agent/hermes-agent/pulls)

---

**最后更新:** 2026-04-17  
**版本:** 1.0.0  
**维护者:** Hermes Agent Team
