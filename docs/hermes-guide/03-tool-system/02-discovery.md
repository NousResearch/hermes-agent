# 工具发现机制

## 3.1 概述

**核心文件**: `model_tools.py`

**机制**: 通过 Python 导入机制，在首次使用时自动扫描并加载所有工具模块。

---

## 3.2 发现流程

```
┌─────────────────────────────────────────────────────────────────┐
│                   工具发现流程                                   │
└─────────────────────────────────────────────────────────────────┘

首次调用 get_tool_definitions()
         │
         ▼
_d discover_tools() 首次调用标志
         │
         ▼
import tools.registry  (初始化单例)
         │
         ▼
遍历 tools/*.py (排除 registry.py, __init__.py)
         │
         ├── tools/file_tools.py     ──► register_handlers(registry)
         ├── tools/terminal_tool.py  ──► register_handlers(registry)
         ├── tools/web_tools.py      ──► register_handlers(registry)
         ├── tools/browser_tool.py   ──► register_handlers(registry)
         ├── tools/mcp_tool.py       ──► register_handlers(registry)
         └── ...                     ──► ...
         │
         ▼
所有工具已注册到 registry._tools
         │
         ▼
返回工具定义列表
```

---

## 3.3 源码解析

### 3.3.1 发现函数

```python
# model_tools.py (伪代码结构)
def _discover_tools():
    """
    首次调用时扫描所有工具模块并注册。

    使用标志位确保只执行一次 (thread-safe)。
    """
    global _tools_discovered

    if _tools_discovered:
        return

    # 标记为已发现 (防止重复)
    _tools_discovered = True

    # 导入注册表
    from tools.registry import registry

    # 工具目录
    tool_dir = Path(__file__).parent / "tools"

    # 扫描工具模块
    for tool_file in sorted(tool_dir.glob("*.py")):
        # 跳过特殊文件
        if tool_file.stem in ("registry", "__init__", "__pycache__"):
            continue

        # 动态导入触发注册
        module_name = f"tools.{tool_file.stem}"
        importlib.import_module(module_name)
```

### 3.3.2 线程安全

```python
# model_tools.py
_tools_discovered = False
_discovery_lock = threading.Lock()

def _discover_tools():
    global _tools_discovered

    with _discovery_lock:
        if _tools_discovered:
            return
        _tools_discovered = True

    # 执行实际的导入和注册
    ...
```

---

## 3.4 工具定义获取

```python
# model_tools.py
def get_tool_definitions(
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None,
    quiet_mode: bool = False
) -> List[dict]:
    """
    获取已注册工具的 JSON Schema 列表。

    Args:
        enabled_toolsets:  白名单 - 仅返回这些工具集的工具
        disabled_toolsets: 黑名单 - 排除这些工具集的工具

    Returns:
        OpenAI 格式的工具定义列表
        [{
            "type": "function",
            "function": {
                "name": "Read",
                "description": "...",
                "parameters": {...}
            }
        }, ...]
    """
    # 确保工具已发现
    _discover_tools()

    from tools.registry import registry

    definitions = []
    for name, entry in registry._tools.items():
        # 工具集过滤
        if disabled_toolsets and entry.toolset in disabled_toolsets:
            continue
        if enabled_toolsets and entry.toolset not in enabled_toolsets:
            continue

        # 转换为 OpenAI 格式
        definitions.append({
            "type": "function",
            "function": {
                "name": entry.name,
                "description": entry.description,
                "parameters": entry.schema.get("parameters", {})
            }
        })

    return definitions
```

---

## 3.5 工具可用性检查

```python
# model_tools.py
def check_tool_availability(quiet: bool = False) -> tuple:
    """
    检查所有工具的可用性。

    Returns:
        (available_tools, unavailable_tools, check_errors)
    """
    _discover_tools()
    from tools.registry import registry

    available = []
    unavailable = []
    errors = []

    for name, entry in registry._tools.items():
        if entry.check_fn is None:
            available.append(name)
            continue

        try:
            if entry.check_fn():
                available.append(name)
            else:
                unavailable.append(name)
        except Exception as e:
            errors.append((name, str(e)))
            unavailable.append(name)

    return available, unavailable, errors
```

---

## 3.6 工具集需求检查

```python
# model_tools.py
TOOLSET_REQUIREMENTS = {
    "browser": {
        "env_vars": ["BROWSER_PROVIDER"],
        "check": lambda: os.getenv("BROWSER_PROVIDER") is not None
    },
    "mcp": {
        "env_vars": ["MCP_*"],
        "check": lambda: True  # MCP 服务器动态提供
    },
    # ...
}

def check_toolset_requirements() -> dict:
    """
    检查工具集依赖是否满足。

    Returns:
        {toolset_name: {"satisfied": bool, "missing": list}}
    """
    results = {}
    for toolset, reqs in TOOLSET_REQUIREMENTS.items():
        missing = [e for e in reqs.get("env_vars", []) if not os.getenv(e)]
        results[toolset] = {
            "satisfied": len(missing) == 0,
            "missing": missing
        }
    return results
```

---

## 3.7 MCP 动态工具

### 3.7.1 动态注册/注销

```python
# tools/registry.py:90-100
def deregister(self, name: str) -> None:
    """
    注销工具。

    MCP 工具发现变化时 (notifications/tools/list_changed)，
    会先注销旧工具再注册新工具。
    """
    entry = self._tools.pop(name, None)
    if entry is None:
        return

    # 清理工具集检查函数
    ...
```

### 3.7.2 MCP 工具处理

```python
# tools/mcp_tool.py
def register_handlers(r: Registry):
    """MCP 工具注册"""
    # MCP 服务器连接的工具有动态名称
    # 通过 MCP protocol 动态发现
    pass

# MCP 工具通过 mcp_client 动态注册
class MCPClient:
    def on_tools_changed(self, tools: List[Tool]):
        """MCP 服务器通知工具列表变化"""
        # 1. 注销旧工具
        for tool in self._registered_tools:
            registry.deregister(tool.name)

        # 2. 注册新工具
        for tool in tools:
            registry.register(
                name=tool.name,
                toolset="mcp",
                schema=tool.schema,
                handler=self._create_handler(tool)
            )
```

---

## 3.8 关键源码索引

| 功能 | 文件 | 说明 |
|------|------|------|
| 发现函数 | `model_tools.py` | `_discover_tools()` |
| 工具定义 | `model_tools.py` | `get_tool_definitions()` |
| 可用性检查 | `model_tools.py` | `check_tool_availability()` |
| 工具集检查 | `model_tools.py` | `check_toolset_requirements()` |
| 动态注销 | `tools/registry.py:90` | `deregister()` |
| MCP 处理 | `tools/mcp_tool.py` | MCP 客户端 |

---

## 3.9 下一步

- [核心工具集](03-core-tools.md) - 各类工具详解
- [CLI 系统架构](../04-cli-system/01-main.md) - hermes_cli 模块分析
- [Gateway 系统](../05-gateway/01-overview.md) - 网关消息处理

---

[← 返回总目录](../SUMMARY.md) | [← 工具注册](01-registry.md) | [核心工具集 →](03-core-tools.md)
