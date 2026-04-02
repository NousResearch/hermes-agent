# 配置管理系统

## 4.1 概述

**核心文件**: `hermes_cli/config.py`

**配置存储**: `~/.hermes/config.yaml`

---

## 4.2 默认配置

```python
# hermes_cli/config.py
DEFAULT_CONFIG = {
    # 模型配置
    "model": "claude-sonnet-4-6",
    "base_url": "https://api.anthropic.com",
    "api_key": None,

    # 对话配置
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 1.0,

    # 工具配置
    "toolsets": ["files", "terminal", "web", "browser"],
    "disabled_tools": [],

    # CLI 配置
    "theme": "default",
    "quick_sync": False,

    # 代理配置
    "http_proxy": None,
    "https_proxy": None,

    # ...
}
```

---

## 4.3 配置加载优先级

```
优先级 (高 → 低):
┌─────────────────────────────────────────────────────────────────┐
│  1. 环境变量 (HERMES_*)                                          │
│     例如: HERMES_MODEL=claude-opus-4-6                          │
├─────────────────────────────────────────────────────────────────┤
│  2. CLI 参数                                                     │
│     例如: hermes --model claude-opus-4-6                        │
├─────────────────────────────────────────────────────────────────┤
│  3. 用户配置 (~/.hermes/config.yaml)                            │
├─────────────────────────────────────────────────────────────────┤
│  4. 默认配置 (hermes_cli/config.py::DEFAULT_CONFIG)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.4 配置迁移

```python
# hermes_cli/config.py
CONFIG_VERSION = 2

def migrate_config(old_config: dict) -> dict:
    """迁移旧版本配置到新版本"""
    version = old_config.get("version", 1)

    if version < 2:
        # v1 -> v2 迁移
        old_config["toolsets"] = old_config.pop("enabled_tools", ["files"])
        old_config["disabled_tools"] = old_config.pop("disabled_tools", [])
        version = 2

    old_config["version"] = CONFIG_VERSION
    return old_config
```

---

## 4.5 关键源码索引

| 功能 | 文件 | 关键变量/函数 |
|------|------|--------------|
| 默认配置 | `hermes_cli/config.py` | `DEFAULT_CONFIG` |
| 配置加载 | `hermes_cli/config.py` | `load_config()` |
| 配置迁移 | `hermes_cli/config.py` | `migrate_config()` |
| 环境变量 | `hermes_cli/config.py` | `OPTIONAL_ENV_VARS` |

---

## 4.6 下一步

- [认证与凭证](04-auth.md) - auth.py 详解
- [主题皮肤系统](05-skin.md) - skin_engine.py 详解
- [Gateway 系统](../../05-gateway/01-overview.md) - 网关架构

---

[← 返回总目录](../SUMMARY.md) | [← 设置向导](02-setup.md) | [认证 →](04-auth.md)
