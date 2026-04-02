# 认证与凭证管理

## 4.1 概述

**核心文件**: `hermes_cli/auth.py` (~88,000 字符)

**功能**: 管理多 provider 的 API 凭证，解决凭证解析、存储、轮换等问题。

---

## 4.2 支持的 Provider

| Provider | 凭证类型 | 配置文件字段 |
|----------|---------|------------|
| OpenAI | `OPENAI_API_KEY` | `api_key` |
| Anthropic | `ANTHROPIC_API_KEY` | `api_key` |
| OpenRouter | `OPENROUTER_API_KEY` | `api_key` |
| Azure OpenAI | `AZURE_OPENAI_KEY` | `api_key` + `base_url` |
| 自定义 | 任意 | `api_key` + `base_url` |

---

## 4.3 凭证解析

```python
# hermes_cli/auth.py
def resolve_credentials(provider: str = None) -> dict:
    """
    解析凭证配置。

    支持多种来源:
    - 环境变量
    - 配置文件
    - .env 文件
    """
    if provider == "openai":
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        }
    elif provider == "anthropic":
        return {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "base_url": "https://api.anthropic.com",
        }
    # ...
```

---

## 4.4 关键源码索引

| 功能 | 文件 | 关键函数 |
|------|------|---------|
| 凭证解析 | `hermes_cli/auth.py` | `resolve_credentials()` |
| 环境变量 | `hermes_cli/auth.py` | `load_from_env()` |
| 凭证验证 | `hermes_cli/auth.py` | `validate_credentials()` |

---

## 4.5 下一步

- [主题皮肤系统](05-skin.md) - skin_engine.py 详解
- [Gateway 系统](../../05-gateway/01-overview.md) - 网关架构

---

[← 返回总目录](../SUMMARY.md) | [← 配置管理](03-config.md) | [皮肤系统 →](05-skin.md)
