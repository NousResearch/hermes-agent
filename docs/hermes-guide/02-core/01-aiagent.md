# AIAgent 核心模块

## 2.1 概述

**核心文件**: `run_agent.py` (~500 行)

**主要类**: `AIAgent`

**核心职责**: 实现 Agent 对话循环，处理工具调用，管理消息历史。

---

## 2.2 AIAgent 类结构

```python
# run_agent.py:150+ (approximate line after class definition)
class AIAgent:
    """
    AI Agent with tool calling capabilities.

    Key features:
    - Automatic tool calling loop until completion
    - Configurable model parameters
    - Error handling and recovery
    - Message history management
    - Support for multiple model providers
    """

    def __init__(
        self,
        base_url: str = "http://localhost:30000/v1",
        model: str = "claude-opus-4-20250514",
        api_key: str = "not-needed",
        max_iterations: int = 100,
        iteration_budget: int = 100,
        tool_iteration_limit: int = 10,
        **kwargs
    ):
        # 初始化
        self.base_url = base_url
        self.model = model
        self.max_iterations = max_iterations
        self.iteration_budget = iteration_budget
        self.tool_iteration_limit = tool_iteration_limit

        # 工具系统
        self.tool_definitions = get_tool_definitions()

        # SessionDB
        self.session_db = SessionDB(...)

        # 上下文压缩器
        self.context_compressor = ContextCompressor(...)
```

---

## 2.3 对话循环流程

### 2.3.1 主循环 `run_conversation()`

```python
# run_agent.py: run_conversation() method
def run_conversation(self, initial_message: str, **kwargs) -> str:
    """
    Main conversation loop.

    Flow:
    1. 构建初始消息
    2. 进入迭代循环 (while budget > 0)
    3. 调用模型
    4. 处理响应 (工具调用或文本响应)
    5. 管理迭代预算
    6. 返回最终响应
    """
    messages = [{"role": "user", "content": initial_message}]
    iteration = 0
    budget = self.iteration_budget

    while budget > 0 and iteration < self.max_iterations:
        # 1. 构建请求
        request = self._build_request(messages)

        # 2. 调用模型
        response = self._call_llm(request)

        # 3. 处理响应
        result = self._handle_response(response, messages)

        # 4. 检查是否完成
        if result.get("type") == "complete":
            return result["content"]

        # 5. 预算管理
        budget -= 1
        iteration += 1

    return "迭代预算耗尽或达到最大迭代次数"
```

### 2.3.2 消息构建 `_build_request()`

```python
def _build_request(self, messages: List[dict]) -> dict:
    """
    构建模型请求。

    包括:
    - system prompt (身份、指令、记忆引导)
    - 上下文引用
    - 工具定义
    - 消息历史
    """
    system_prompt = self._build_system_prompt()

    return {
        "model": self.model,
        "messages": [system_prompt] + messages,
        "tools": self.tool_definitions,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
```

### 2.3.3 工具执行 `_handle_function_call()`

```python
def _handle_function_call(self, function_name: str, function_args: dict) -> str:
    """
    执行工具调用。

    委派给 model_tools.handle_function_call()
    """
    return handle_function_call(
        function_name,
        function_args,
        task_id=str(uuid.uuid4()),
        user_task=None
    )
```

---

## 2.4 迭代控制机制

### 2.4.1 预算管理

```python
class AIAgent:
    def __init__(self, ...):
        # 迭代预算 - 整个对话的最多迭代次数
        self.iteration_budget = iteration_budget

        # 最大迭代次数 - 防止无限循环
        self.max_iterations = max_iterations

        # 单个工具调用的最大迭代次数
        self.tool_iteration_limit = tool_iteration_limit
```

**控制流程**:
```
budget = iteration_budget (默认 100)
while budget > 0:
    budget -= 1
    if 完成:
        break
    if iteration >= max_iterations:
        break
```

### 2.4.2 上下文压缩触发

当消息历史超过阈值时，自动触发压缩：

```python
# 上下文压缩器
class ContextCompressor:
    def should_compress(self, messages: List[dict]) -> bool:
        """检查是否需要压缩"""
        total_tokens = estimate_messages_tokens(messages)
        return total_tokens > self.compression_threshold

    def compress(self, messages: List[dict]) -> List[dict]:
        """压缩消息历史，保留关键信息"""
        # 1. 识别关键消息 (工具调用结果、用户明确要求)
        # 2. 生成摘要
        # 3. 替换原始消息
```

---

## 2.5 关键源码索引

| 功能 | 文件位置 | 函数/类 |
|------|---------|---------|
| 主入口 | `run_agent.py:run_conversation()` | `AIAgent.run_conversation()` |
| 请求构建 | `run_agent.py:_build_request()` | `AIAgent._build_request()` |
| 模型调用 | `run_agent.py:_call_llm()` | `AIAgent._call_llm()` |
| 工具执行 | `run_agent.py:_handle_function_call()` | `handle_function_call()` |
| 预算管理 | `run_agent.py` | `iteration_budget`, `max_iterations` |
| 上下文压缩 | `agent/context_compressor.py` | `ContextCompressor` |
| SessionDB | `hermes_state.py` | `SessionDB` |

---

## 2.6 工具集成

### 2.6.1 工具定义获取

```python
# run_agent.py:65-70
from model_tools import (
    get_tool_definitions,      # 获取工具 schema 列表
    get_toolset_for_tool,       # 获取工具所属工具集
    handle_function_call,        # 执行工具调用
    check_toolset_requirements,  # 检查工具集依赖
)
```

### 2.6.2 工具调用流程

```
用户输入
    │
    ▼
模型响应 (包含 tool_calls)
    │
    ▼
AIAgent._handle_function_call()
    │
    ▼
model_tools.handle_function_call()
    │
    ├── registry.get_handler(tool_name)  # 获取处理器
    ├── 验证参数 (schema)
    └── 执行处理器
            │
            ├── file_tools (同步)
            ├── terminal_tool (同步)
            ├── web_tools (可能异步)
            └── mcp_tool (异步)
    │
    ▼
返回结果，追加到消息历史
```

---

## 2.7 与 CLI 的集成

```python
# cli.py 调用 AIAgent
from run_agent import AIAgent

class HermesCLI:
    def chat(self, message: str):
        agent = AIAgent(
            base_url=self.config.base_url,
            model=self.config.model,
            api_key=self.config.api_key,
        )
        return agent.run_conversation(message)
```

---

## 2.8 下一步

- [消息流转机制](02-message-flow.md) - 消息处理、上下文管理详解
- [工具注册与调度](../03-tool-system/01-registry.md) - tools/registry.py 详解
- [CLI 系统架构](../04-cli-system/01-main.md) - hermes_cli 模块分析

---

[← 返回总目录](../SUMMARY.md) | [← 核心概念](../01-overview/01-core-concepts.md) | [消息流转 →](02-message-flow.md)
