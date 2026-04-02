# 工具注册与调度

## 3.1 概述

**核心文件**:
- `tools/registry.py` - 中心化工具注册表
- `model_tools.py` - 工具编排层

**设计原则**: 每个工具文件在导入时通过 `@registry.register()` 自动注册自己。

---

## 3.2 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    工具注册流程                                   │
└─────────────────────────────────────────────────────────────────┘

                           ┌─────────────────┐
                           │  tools/registry.py │
                           │                  │
                           │  class Registry  │
                           │  - _tools: Dict  │
                           │  - register()    │
                           │  - get_handler() │
                           │  - list_tools()  │
                           └────────┬─────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
          │file_tools.py│ │terminal_tool│ │  mcp_tool   │
          │             │ │             │ │             │
          │register_   │ │register_    │ │register_    │
          │handlers(r)  │ │handlers(r)  │ │handlers(r)  │
          └─────────────┘ └─────────────┘ └─────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │ import 时自动执行
                                    ▼
                          ┌─────────────────┐
                          │ model_tools.py  │
                          │                 │
                          │ _discover_tools()│
                          │ get_tool_       │
                          │ definitions()   │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │   run_agent.py  │
                          │                 │
                          │   AIAgent       │
                          └─────────────────┘
```

---

## 3.3 ToolRegistry 类

### 3.3.1 核心数据结构

```python
# tools/registry.py:45-50
class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        # 工具名 -> ToolEntry
        self._tools: Dict[str, ToolEntry] = {}

        # 工具集 -> 检查函数
        self._toolset_checks: Dict[str, Callable] = {}
```

### 3.3.2 ToolEntry 元数据

```python
# tools/registry.py:24-42
class ToolEntry:
    """Metadata for a single registered tool."""

    __slots__ = (
        "name",        # 工具名 (如 "Read", "Write")
        "toolset",     # 所属工具集 (如 "files", "terminal")
        "schema",      # JSON Schema (用于参数验证)
        "handler",     # 处理函数
        "check_fn",    # 可用性检查函数
        "requires_env", # 所需环境变量
        "is_async",    # 是否异步执行
        "description", # 工具描述
        "emoji",       # emoji 图标
    )
```

### 3.3.3 注册方法

```python
# tools/registry.py:56-88
def register(
    self,
    name: str,           # 工具名
    toolset: str,        # 工具集
    schema: dict,        # JSON Schema
    handler: Callable,   # 处理函数
    check_fn: Callable = None,      # 可用性检查
    requires_env: list = None,      # 依赖的环境变量
    is_async: bool = False,          # 是否异步
    description: str = "",           # 描述
    emoji: str = "",                 # emoji
):
    """注册工具 - 由各工具文件在导入时调用"""
    self._tools[name] = ToolEntry(...)
```

---

## 3.4 工具文件注册示例

### 3.4.1 文件工具注册

```python
# tools/file_tools.py (简化结构)
from tools.registry import registry

def register_handlers(r: Registry):
    """文件工具处理器注册"""
    r.register(
        name="Read",
        toolset="files",
        schema={
            "name": "Read",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        },
        handler=read_file_handler,
        description="Read contents of a file"
    )

    r.register(name="Write", ...)
    r.register(name="Edit", ...)
    r.register(name="Glob", ...)
    r.register(name="Grep", ...)

# 导入时自动注册
register_handlers(registry)
```

### 3.4.2 终端工具注册

```python
# tools/terminal_tool.py (简化结构)
def register_handlers(r: Registry):
    r.register(
        name="Bash",
        toolset="terminal",
        schema={...},
        handler=bash_handler,
        is_async=True,  # 终端命令可能是长时间运行
    )
```

---

## 3.5 工具发现机制

### 3.5.1 _discover_tools()

```python
# model_tools.py (伪代码)
def _discover_tools():
    """
    触发所有工具模块的注册。

    通过导入 tools/ 下的所有模块，触发每个模块的
    register_handlers(registry) 调用。
    """
    # 工具目录
    tool_dir = Path(__file__).parent / "tools"

    # 扫描所有 .py 文件
    for tool_file in tool_dir.glob("*.py"):
        if tool_file.name in ("registry.py", "__init__.py"):
            continue

        # 动态导入模块
        module_name = f"tools.{tool_file.stem}"
        importlib.import_module(module_name)
```

### 3.5.2 获取工具定义

```python
# model_tools.py
def get_tool_definitions(
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None,
    quiet_mode: bool = False
) -> List[dict]:
    """
    返回工具定义列表，用于 API 调用。

    Args:
        enabled_toolsets: 仅包含这些工具集
        disabled_toolsets: 排除这些工具集

    Returns:
        [{name, description, parameters}, ...]
    """
    definitions = []
    for name, entry in registry._tools.items():
        # 根据工具集过滤
        if disabled_toolsets and entry.toolset in disabled_toolsets:
            continue
        if enabled_toolsets and entry.toolset not in enabled_toolsets:
            continue

        definitions.append(entry.schema)

    return definitions
```

---

## 3.6 工具执行流程

### 3.6.1 handle_function_call()

```python
# model_tools.py
def handle_function_call(
    function_name: str,
    function_args: dict,
    task_id: str,
    user_task: str = None
) -> str:
    """
    执行工具调用。

    Flow:
    1. 获取处理器
    2. 验证参数
    3. 执行处理函数
    4. 返回结果
    """
    # 1. 获取工具条目
    entry = registry._tools.get(function_name)
    if not entry:
        return f"Error: Unknown tool '{function_name}'"

    # 2. 检查可用性
    if entry.check_fn and not entry.check_fn():
        return f"Error: Tool '{function_name}' is not available"

    # 3. 执行
    try:
        if entry.is_async:
            # 异步处理 (如终端命令)
            return _run_async(entry.handler(**function_args))
        else:
            # 同步处理
            return entry.handler(**function_args)
    except Exception as e:
        return f"Error executing {function_name}: {e}"
```

### 3.6.2 异步处理

```python
# model_tools.py:39-78
_tool_loop = None          # 主线程持久化事件循环
_worker_thread_local = threading.local()  # worker 线程独立循环

def _run_async(coro):
    """从同步上下文运行异步协程"""
    global _tool_loop

    if asyncio.get_event_loop().is_running():
        # 已在事件循环中，使用 worker 线程
        loop = _get_worker_loop()
    else:
        # CLI 主线程，使用持久化循环
        loop = _get_tool_loop()

    return asyncio.run_coroutine_threadsafe(coro, loop).result()
```

---

## 3.7 工具集 (Toolsets)

### 3.7.1 工具集定义

```python
# toolsets.py
_HERMES_CORE_TOOLS = [
    "Read", "Write", "Edit", "Glob", "Grep",
    "Bash", "WebSearch", "WebExtract",
    "BrowserNavigate", "BrowserClick",
    ...
]

TOOLSETS = {
    "files": {
        "tools": ["Read", "Write", "Edit", "Glob", "Grep"],
        "requires": [],  # 无额外依赖
    },
    "terminal": {
        "tools": ["Bash"],
        "requires": [],  # 可能需要 shell 环境
    },
    "web": {
        "tools": ["WebSearch", "WebExtract"],
        "requires": [],
    },
    "browser": {
        "tools": ["BrowserNavigate", "BrowserClick", ...],
        "requires": ["browser"],
    },
    "mcp": {
        "tools": ["mcp_*"],  # MCP 动态工具
        "requires": ["mcp"],
    },
}
```

### 3.7.2 工具集管理

```python
# toolsets.py
def resolve_toolset(name: str) -> List[str]:
    """获取工具集包含的所有工具"""
    return TOOLSETS.get(name, {}).get("tools", [])

def validate_toolset(name: str) -> bool:
    """验证工具集是否可用"""
    reqs = TOOLSETS.get(name, {}).get("requires", [])
    return all(os.getenv(r) for r in reqs)
```

---

## 3.8 关键源码索引

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 注册表类 | `tools/registry.py` | 45 | `ToolRegistry` |
| 注册方法 | `tools/registry.py` | 56 | `register()` |
| 工具入口 | `tools/registry.py` | 24 | `ToolEntry` |
| 工具发现 | `model_tools.py` | `_discover_tools()` | 触发所有模块注册 |
| 工具执行 | `model_tools.py` | `handle_function_call()` | 核心调度函数 |
| 异步处理 | `model_tools.py` | 39-78 | `_get_tool_loop()` |
| 工具集 | `toolsets.py` | `_HERMES_CORE_TOOLS` | 核心工具集定义 |

---

## 3.9 工具系统全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                        工具系统全景                               │
└─────────────────────────────────────────────────────────────────┘

                           ┌─────────────────┐
                           │   AIAgent       │
                           │ run_conversation│
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  model_tools.py │
                           │                 │
                           │ get_tool_       │
                           │ definitions()   │
                           │ handle_         │
                           │ function_call() │
                           └────────┬────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
          ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
          │   registry │ │  toolsets   │ │     ...     │
          │             │ │             │ │             │
          │ _tools: Dict│ │ _HERMES_    │ │             │
          │             │ │ CORE_TOOLS  │ │             │
          └──────┬──────┘ └──────┬──────┘ └─────────────┘
                 │               │
       ┌─────────┴──────┐       │
       │                │       │
       ▼                ▼       ▼
  ┌─────────┐    ┌─────────┐ ┌─────────┐
  │  Read   │    │  Write  │ │  Bash   │
  │ (file_  │    │ (file_  │ │(terminal│
  │ tools)  │    │ tools)  │ │ _tool)  │
  └─────────┘    └─────────┘ └─────────┘
```

---

## 3.10 下一步

- [工具发现机制](02-discovery.md) - 自动扫描详解
- [核心工具集](03-core-tools.md) - 各类工具详解
- [CLI 系统架构](../04-cli-system/01-main.md) - hermes_cli 模块分析

---

[← 返回总目录](../SUMMARY.md) | [← 核心概念](../01-overview/01-core-concepts.md) | [Tool 发现 →](02-discovery.md)
