[根目录](../CLAUDE.md) > **hermes_cli**

# Hermes CLI 模块 - 命令行界面

> 最后更新：2026-04-13 16:35:00

## 模块职责

`hermes_cli/` 模块是 Hermes Agent 的命令行界面，负责用户交互、配置管理、命令调度和设置向导。该模块提供了完整的 TUI（终端用户界面）体验，支持多行编辑、斜杠命令自动补全、对话历史等功能。

主要职责：
- **CLI 入口**：`hermes` 命令的主入口点
- **Web UI 服务器**：FastAPI 后端服务器，提供 Web 管理界面
- **配置管理**：加载、保存、迁移配置文件
- **命令注册**：中央化的斜杠命令注册表
- **设置向导**：引导用户完成初始设置
- **皮肤引擎**：可自定义的终端主题和样式
- **环境加载**：从 `.env` 文件加载环境变量
- **多实例支持**：支持多个 Hermes 实例隔离

## 入口与启动

### 主要入口点
- **`hermes_cli/main.py`**：`main()` 函数 - CLI 主入口
- **`hermes` 命令**：通过 `setup.py` 安装的命令行工具

### 启动流程
```
hermes 命令
  ↓
hermes_cli/main.py:main()
  ↓
加载配置 (config.py)
  ↓
初始化 CLI (cli.py:HermesCLI)
  ↓
启动交互式循环 (HermesCLI.run())
```

## 对外接口

### 命令注册表 (commands.py)

**COMMAND_REGISTRY** - 所有斜杠命令的真实来源：

```python
COMMAND_REGISTRY: list[CommandDef] = [
    # Session 命令
    CommandDef("new", "Start a new session", "Session",
               aliases=("reset",)),
    CommandDef("clear", "Clear screen and start a new session", "Session",
               cli_only=True),
    CommandDef("history", "Show conversation history", "Session",
               cli_only=True),
    CommandDef("save", "Save the current conversation", "Session",
               cli_only=True),
    CommandDef("retry", "Retry the last message", "Session"),
    CommandDef("undo", "Remove the last user/assistant exchange", "Session"),
    CommandDef("branch", "Branch the current session", "Session",
               aliases=("fork",)),
    CommandDef("compress", "Manually compress conversation context", "Session"),
    CommandDef("stop", "Kill all running background processes", "Session"),

    # Configuration 命令
    CommandDef("config", "Show current configuration", "Configuration",
               cli_only=True),
    CommandDef("model", "Switch model for this session", "Configuration"),
    CommandDef("provider", "Show available providers", "Configuration"),
    CommandDef("prompt", "View/set custom system prompt", "Configuration",
               cli_only=True),
    CommandDef("personality", "Set a predefined personality", "Configuration"),
    CommandDef("yolo", "Toggle YOLO mode", "Configuration"),

    # Tools 命令
    CommandDef("tools", "Configure enabled toolsets", "Configuration"),
    CommandDef("skills", "Manage skills", "Configuration"),

    # Gateway 命令
    CommandDef("gateway", "Start the messaging gateway", "Gateway",
               cli_only=True),
    CommandDef("broadcast", "Broadcast a message to all platforms", "Gateway",
               cli_only=True),
]
```

**CommandDef 数据类**：
```python
@dataclass(frozen=True)
class CommandDef:
    name: str                          # 规范名称（不带斜杠）
    description: str                   # 人类可读的描述
    category: str                      # "Session", "Configuration" 等
    aliases: tuple[str, ...] = ()      # 替代名称
    args_hint: str = ""                # 参数占位符
    subcommands: tuple[str, ...] = ()  # Tab 补全的子命令
    cli_only: bool = False             # 仅在 CLI 中可用
    gateway_only: bool = False         # 仅在网关中可用
    gateway_config_gate: str | None = None  # 配置点路径
```

### 配置系统 (config.py)

**DEFAULT_CONFIG** - 默认配置字典：

```python
DEFAULT_CONFIG = {
    # 模型配置
    "provider": "anthropic",
    "model": "claude-sonnet-4.5",
    "api_mode": "chat",

    # 工具集配置
    "enabled_toolsets": ["terminal", "file", "web"],
    "disabled_toolsets": [],

    # 终端配置
    "terminal": {
        "backend": "local",
        "cwd": null,  # 自动检测
        "timeout": 120,
    },

    # 显示配置
    "display": {
        "emoji": true,
        "spinner": "kawaii",
        "tool_progress": "new",
    },

    # 压缩配置
    "compression": {
        "enabled": true,
        "target_ratio": 0.7,
    },

    # 记忆配置
    "memory": {
        "enabled": true,
        "provider": "builtin",
    },

    # 网关配置
    "gateway": {
        "telegram": false,
        "discord": false,
        "slack": false,
    },

    # 配置版本（用于迁移）
    "_config_version": 1,
}
```

**主要函数**：
```python
def load_cli_config(hermes_home: Path = None) -> dict:
    """加载 CLI 配置文件。"""

def save_cli_config(config: dict, hermes_home: Path = None) -> None:
    """保存 CLI 配置文件。"""

def migrate_config(config: dict) -> dict:
    """迁移旧版本配置到新版本。"""

def get_config_schema() -> dict:
    """获取配置模式的 JSON Schema。"""
```

### CLI 类 (cli.py)

**HermesCLI 类**：
```python
class HermesCLI:
    def __init__(self, config: dict = None, session_id: str = None):
        """初始化 CLI 实例。"""

    def run(self) -> None:
        """启动交互式循环。"""

    def process_command(self, command: str) -> bool:
        """处理斜杠命令。返回 True 表示应该继续循环。"""

    def display_message(self, role: str, content: str) -> None:
        """显示消息。"""

    def get_user_input(self) -> str:
        """获取用户输入。"""
```

### 设置向导 (setup.py)

**主要函数**：
```python
def run_setup_wizard(hermes_home: Path = None) -> None:
    """运行完整的设置向导。"""

def setup_provider() -> None:
    """设置 LLM 提供商。"""

def setup_model() -> None:
    """设置模型。"""

def setup_toolsets() -> None:
    """设置工具集。"""

def setup_terminal() -> None:
    """设置终端后端。"""

def setup_gateway() -> None:
    """设置消息网关。"""
```

### 皮肤引擎 (skin_engine.py)

**主要类和函数**：
```python
class Skin:
    def __init__(self, name: str, colors: dict, emojis: dict):
        """皮肤定义。"""

    def apply(self) -> None:
        """应用皮肤到 CLI。"""

def load_skin(name: str) -> Skin:
    """加载皮肤。"""

def list_skins() -> list[str]:
    """列出可用皮肤。"""
```

**内置皮肤**：
- `default` - 默认主题
- `dark` - 深色主题
- `light` - 浅色主题
- `minimal` - 极简主题
- `kawaii` - 可爱主题

### 环境加载 (env_loader.py)

**主要函数**：
```python
def load_hermes_dotenv(hermes_home: Path, project_env: Path = None) -> None:
    """加载 ~/.hermes/.env 文件。"""

def load_project_env(project_dir: Path) -> None:
    """加载项目特定的 .env 文件。"""
```

### Web UI 服务器 (web_server.py)

**FastAPI 应用**：
```python
app = FastAPI(title="Hermes Agent", version=__version__)

# 主要端点
@app.get("/api/config")
async def get_config() -> dict:
    """获取当前配置。"""

@app.post("/api/config")
async def update_config(config: dict) -> dict:
    """更新配置。"""

@app.get("/api/env")
async def get_env() -> dict:
    """获取环境变量（已脱敏）。"""

@app.post("/api/env/{key}")
async def set_env_value(key: str, value: str) -> dict:
    """设置环境变量值。"""

@app.delete("/api/env/{key}")
async def delete_env_value(key: str) -> dict:
    """删除环境变量值。"""

@app.get("/api/sessions")
async def list_sessions() -> list:
    """列出所有会话。"""

@app.get("/api/skills")
async def list_skills() -> list:
    """列出所有技能。"""

@app.get("/api/cron")
async def list_cron_jobs() -> list:
    """列出所有 cron 任务。"""

@app.post("/api/reveal/{key}")
async def reveal_secret(key: str, token: str) -> dict:
    """揭示敏感环境变量值（需要会话令牌）。"""
```

**安全特性**：
- **会话令牌**：每次启动生成唯一的 `_SESSION_TOKEN`
- **CORS 限制**：仅允许 localhost 源访问
- **速率限制**：敏感端点（reveal）有速率限制
- **静态文件服务**：从 `web_dist/` 目录提供构建的前端

**主要函数**：
```python
def start_web_server(port: int = 9119, open_browser: bool = True) -> None:
    """启动 Web UI 服务器。"""

def get_config_schema() -> dict:
    """生成配置的 JSON Schema（包含字段类型和选项）。"""
```

## 关键依赖与配置

### 依赖项
- **prompt_toolkit**：TUI 框架，提供多行编辑、自动补全等
- **Rich**：终端格式化和美化
- **PyYAML**：配置文件解析
- **python-dotenv**：环境变量加载
- **FastAPI**：Web UI 后端服务器
- **uvicorn**：ASGI 服务器（Web UI）

### 配置文件
- `~/.hermes/config.yaml` - 主配置文件
- `~/.hermes/.env` - API 密钥和秘密
- `~/.hermes/SOUL.md` - 个性文件
- `~/.hermes/skills/` - 技能目录

### 环境变量
- `HERMES_HOME` - Hermes 主目录（支持多实例）
- `HERMES_MANAGED` - 托管模式标记
- `ANTHROPIC_API_KEY` - Anthropic API 密钥
- `OPENAI_API_KEY` - OpenAI API 密钥

## 数据模型

### 配置结构
```python
{
    # 模型配置
    "provider": str,           # "anthropic", "openai", "local"
    "model": str,              # 模型名称
    "api_mode": str,           # "chat", "completion"

    # 工具集配置
    "enabled_toolsets": list[str],
    "disabled_toolsets": list[str],

    # 终端配置
    "terminal": {
        "backend": str,        # "local", "docker", "ssh", "modal"
        "cwd": str | None,     # 工作目录
        "timeout": int,        # 命令超时（秒）
        "docker_image": str,   # Docker 镜像
        "ssh_host": str,       # SSH 主机
        "ssh_user": str,       # SSH 用户
    },

    # 显示配置
    "display": {
        "emoji": bool,         # 启用表情符号
        "spinner": str,        # Spinner 类型
        "tool_progress": str,  # 工具进度显示
    },

    # 压缩配置
    "compression": {
        "enabled": bool,
        "target_ratio": float,
    },

    # 记忆配置
    "memory": {
        "enabled": bool,
        "provider": str,       # "builtin", "honcho", "mem0"
    },

    # 网关配置
    "gateway": {
        "telegram": bool,
        "discord": bool,
        "slack": bool,
        "whatsapp": bool,
    },

    # 配置版本
    "_config_version": int,
}
```

### 命令历史格式
```python
{
    "session_id": str,
    "messages": [
        {"role": "user", "content": str},
        {"role": "assistant", "content": str},
    ],
    "timestamp": str,
    "title": str,
}
```

## 测试与质量

### 测试文件
- `tests/hermes_cli/test_commands.py` - 命令注册表测试
- `tests/hermes_cli/test_config.py` - 配置管理测试
- `tests/hermes_cli/test_setup.py` - 设置向导测试
- `tests/hermes_cli/test_skin_engine.py` - 皮肤引擎测试
- `tests/hermes_cli/test_env_loader.py` - 环境加载测试
- `tests/hermes_cli/test_web_server.py` - Web UI 服务器测试（约 675 个测试用例）

### 测试覆盖
- **单元测试**：每个模块都有对应的测试文件
- **集成测试**：测试与配置文件、环境变量的交互
- **约 30 个测试文件**覆盖核心功能

### 质量指标
- 高测试覆盖率（~80%+）
- 完整的类型提示
- 详细的文档字符串
- 良好的错误处理

## 常见问题 (FAQ)

### Q: 如何添加新的斜杠命令？
A:
1. 在 `commands.py` 的 `COMMAND_REGISTRY` 中添加 `CommandDef`
2. 在 `cli.py` 的 `HermesCLI.process_command()` 中添加处理器
3. 如果网关支持，在 `gateway/run.py` 中添加处理器

### Q: 如何自定义配置？
A: 使用 `hermes config set <key> <value>` 或直接编辑 `~/.hermes/config.yaml`。

### Q: 如何创建自定义皮肤？
A: 在 `skin_engine.py` 中定义新的 `Skin` 实例，或在 `~/.hermes/skins/` 目录中创建皮肤文件。

### Q: 如何迁移配置到新版本？
A: 配置系统会自动检测 `_config_version` 并运行迁移函数。

### Q: 如何支持多实例？
A: 设置 `HERMES_HOME` 环境变量到不同的目录，每个实例有独立的配置和数据。

### Q: 如何调试配置加载？
A: 设置 `HERMES_DEBUG=1` 环境变量，查看详细的配置加载日志。

## 相关文件清单

### 核心文件
- `hermes_cli/__init__.py` - 模块初始化
- `hermes_cli/main.py` - CLI 主入口
- `hermes_cli/cli.py` - CLI 类和交互循环
- `hermes_cli/config.py` - 配置管理
- `hermes_cli/web_server.py` - Web UI 服务器（FastAPI 后端）
- `hermes_cli/commands.py` - 命令注册表
- `hermes_cli/setup.py` - 设置向导
- `hermes_cli/skin_engine.py` - 皮肤引擎
- `hermes_cli/env_loader.py` - 环境加载

### 辅助文件
- `hermes_cli/colors.py` - 颜色定义
- `hermes_cli/default_soul.py` - 默认个性
- `hermes_cli/completer.py` - 命令补全
- `hermes_cli/history.py` - 历史记录
- `hermes_cli/profiling.py` - 性能分析
- `hermes_cli/validation.py` - 配置验证

### 配置示例
- `hermes_cli/cli-config.yaml.example` - CLI 配置示例

### 测试文件
- `tests/hermes_cli/` - 完整的测试套件

## 变更记录 (Changelog)

### 2026-04-13 16:35:00 - Web UI 服务器集成 🌐
- ✅ **新增 Web UI 服务器**：FastAPI 后端，提供配置管理、环境变量、会话管理等 API
- 📊 **新增 API 端点**：config、env、sessions、skills、cron、logs 等端点
- 🔒 **安全特性**：会话令牌、CORS 限制、速率限制
- 🧪 **测试覆盖**：约 675 个 Web 服务器测试用例
- 📖 **文档更新**：添加 Web 服务器接口文档

### 2026-04-08 - 初始化模块文档 🚀
- ✅ **创建模块文档**：生成 hermes_cli 模块的完整 CLAUDE.md
- 📊 **架构分析**：详细说明命令注册表和配置系统
- 🔧 **接口文档**：记录主要对外接口
- 📖 **使用指南**：提供添加命令和自定义配置的步骤
- 🎯 **下一步**：需要补充详细的 API 文章和示例代码
