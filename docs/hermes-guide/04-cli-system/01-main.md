# CLI 系统架构

## 4.1 概述

**核心文件**: `hermes_cli/main.py` (~200,000 字符)

**入口点**: `hermes` 命令行工具

**架构**: 基于 Fire 的交互式 CLI，支持多级子命令。

---

## 4.2 命令体系

```
hermes
├── (无子命令)          → 交互式 CLI
├── setup              → 首次配置向导
├── config             → 配置管理
│   ├── show           → 显示当前配置
│   ├── edit           → 编辑配置
│   └── validate       → 验证配置
├── model              → 模型切换
│   ├── list           → 列出可用模型
│   ├── set            → 设置默认模型
│   └── info           → 模型信息
├── skills             → 技能管理
│   ├── list           → 列出技能
│   ├── enable         → 启用技能
│   └── disable        → 禁用技能
├── tools              → 工具管理
│   ├── list           → 列出工具
│   ├── enable         → 启用工具
│   └── disable        → 禁用工具
├── gateway            → 消息网关
│   ├── start          → 启动网关
│   ├── stop           → 停止网关
│   ├── telegram       → Telegram 平台
│   ├── discord        → Discord 平台
│   └── status         → 网关状态
├── cron               → 定时任务
│   ├── list           → 列出任务
│   ├── add            → 添加任务
│   └── remove         → 删除任务
├── doctor             → 诊断检查
└── help               → 帮助信息
```

---

## 4.3 核心组件

### 4.3.1 主入口

```python
# hermes_cli/main.py (简化结构)
def main():
    """Hermes CLI 主入口"""
    cli = HermesCLI()
    cli.run()

class HermesCLI:
    def __init__(self):
        self.config = load_config()
        self.session = SessionManager()

    def run(self):
        """运行交互式 CLI"""
        while True:
            prompt = self.build_prompt()
            user_input = input(prompt)
            response = self.process_input(user_input)
            print(response)
```

### 4.3.2 配置加载

```python
# hermes_cli/config.py
DEFAULT_CONFIG = {
    "model": "claude-sonnet-4-6",
    "base_url": "https://api.anthropic.com",
    "max_tokens": 8192,
    "temperature": 0.7,
    # ...
}

def load_config() -> dict:
    """加载配置 (优先级: 环境变量 > 用户配置 > 默认)"""
    config = DEFAULT_CONFIG.copy()

    # 用户配置
    user_config_path = Path("~/.hermes/config.yaml")
    if user_config_path.exists():
        config.update(yaml.safe_load(user_config_path.read_text()))

    # 环境变量覆盖
    for key in config:
        env_key = f"HERMES_{key.upper()}"
        if env_key in os.environ:
            config[key] = os.environ[env_key]

    return config
```

---

## 4.4 关键模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 主入口 | `hermes_cli/main.py` | CLI 框架、命令调度 |
| 配置 | `hermes_cli/config.py` | 配置管理、迁移 |
| 命令 | `hermes_cli/commands.py` | 命令定义注册表 |
| 认证 | `hermes_cli/auth.py` | Provider 凭证管理 |
| 设置 | `hermes_cli/setup.py` | 交互式设置向导 |
| 皮肤 | `hermes_cli/skin_engine.py` | 主题系统 |
| 模型 | `hermes_cli/models.py` | 模型目录 |

---

## 4.5 关键源码索引

| 功能 | 文件 | 关键函数/类 |
|------|------|------------|
| CLI 入口 | `hermes_cli/main.py` | `main()`, `HermesCLI` |
| 配置加载 | `hermes_cli/config.py` | `load_config()`, `DEFAULT_CONFIG` |
| 命令注册 | `hermes_cli/commands.py` | `COMMAND_REGISTRY` |
| 认证 | `hermes_cli/auth.py` | `resolve_credentials()` |
| 设置向导 | `hermes_cli/setup.py` | `SetupWizard` |

---

## 4.6 下一步

- [交互式设置向导](02-setup.md) - setup.py 详解
- [配置管理系统](03-config.md) - config.py 详解
- [Gateway 系统](../05-gateway/01-overview.md) - 网关消息处理

---

[← 返回总目录](../SUMMARY.md) | [← 工具系统](../03-tool-system/03-core-tools.md) | [设置向导 →](02-setup.md)
