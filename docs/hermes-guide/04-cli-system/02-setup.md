# 交互式设置向导

## 4.1 概述

**核心文件**: `hermes_cli/setup.py` (~107,000 字符)

**功能**: 引导用户完成首次配置，包括 API 密钥、模型选择、平台连接等。

---

## 4.2 设置流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    首次设置流程                                  │
└─────────────────────────────────────────────────────────────────┘

hermes setup
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 欢迎 & 快速介绍                                        │
│  - 显示欢迎信息                                                  │
│  - 说明需要配置的内容                                            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: 选择模型提供商                                          │
│  - OpenAI                                                       │
│  - Anthropic (Claude)                                           │
│  - OpenRouter                                                   │
│  - 自定义端点                                                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: 输入 API 密钥                                           │
│  - 安全输入 (不显示密码)                                         │
│  - 验证密钥有效性                                                │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: 选择模型                                                │
│  - 显示可用模型列表                                              │
│  - 根据提供商筛选                                                │
│  - 显示模型参数 (上下文长度、价格)                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: 配置消息平台 (可选)                                     │
│  - Telegram                                                    │
│  - Discord                                                      │
│  - Slack                                                        │
│  - ...                                                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: 完成                                                    │
│  - 保存配置到 ~/.hermes/config.yaml                            │
│  - 显示下一步指引                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.3 核心实现

```python
# hermes_cli/setup.py (简化结构)
class SetupWizard:
    """交互式设置向导"""

    def __init__(self):
        self.config = {}
        self.steps = [
            self.step_welcome,
            self.step_select_provider,
            self.step_api_key,
            self.step_select_model,
            self.step_config_platforms,
            self.step_finalize,
        ]

    def run(self):
        """运行设置向导"""
        for step in self.steps:
            result = step()
            if result is False:
                print("设置取消")
                return False

        return True

    def step_api_key(self) -> bool:
        """获取 API 密钥"""
        print("请输入您的 API 密钥:")
        key = getpass.getpass("> ")

        # 验证密钥
        if self.validate_key(key):
            self.config["api_key"] = key
            return True
        else:
            print("密钥验证失败，请重试")
            return self.step_api_key()

    def validate_key(self, key: str) -> bool:
        """验证 API 密钥"""
        # 尝试调用 API
        client = OpenAI(api_key=key)
        try:
            client.models.list()
            return True
        except:
            return False
```

---

## 4.4 关键源码索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| 设置向导 | `hermes_cli/setup.py` | `SetupWizard` |
| 配置保存 | `hermes_cli/config.py` | `save_config()` |
| API 验证 | `hermes_cli/auth.py` | `validate_credentials()` |

---

## 4.5 下一步

- [配置管理系统](03-config.md) - config.py 详解
- [认证与凭证](04-auth.md) - auth.py 详解
- [主题皮肤系统](05-skin.md) - skin_engine.py 详解

---

[← 返回总目录](../SUMMARY.md) | [← CLI 主入口](01-main.md) | [配置管理 →](03-config.md)
