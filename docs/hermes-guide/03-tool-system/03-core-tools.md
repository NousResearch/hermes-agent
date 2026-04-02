# 核心工具集详解

## 3.1 概述

Hermes 内置 30+ 工具，分为以下类别：

| 类别 | 工具数 | 主要文件 |
|------|--------|---------|
| 文件操作 | 6+ | `tools/file_tools.py` (~44k) |
| 终端控制 | 5+ | `tools/terminal_tool.py` (~57k) |
| Web 搜索 | 3+ | `tools/web_tools.py` (~76k) |
| 浏览器自动化 | 8+ | `tools/browser_tool.py` (~77k) |
| MCP 客户端 | 动态 | `tools/mcp_tool.py` (~77k) |
| 委托代理 | 2+ | `tools/delegate_tool.py` (~32k) |
| 技能管理 | 3+ | `tools/skills_tool.py` (~48k) |
| 其他 | 10+ | `tools/*.py` |

---

## 3.2 文件操作工具 (file_tools)

**文件**: `tools/file_tools.py` (~44,000 字符)

### 3.2.1 工具列表

| 工具名 | 功能 | 签名 |
|--------|------|------|
| `Read` | 读取文件内容 | `(path: str) -> str` |
| `Write` | 写入文件 | `(path: str, content: str) -> str` |
| `Edit` | 编辑文件 | `(path: str, old_string: str, new_string: str) -> str` |
| `Glob` | 文件模式匹配 | `(pattern: str) -> List[str]` |
| `Grep` | 文本搜索 | `(pattern: str, path: str) -> List[str]` |
| `Patch` | 应用补丁 | `(path: str, patch: str) -> str` |

### 3.2.2 核心处理器

```python
# tools/file_tools.py (结构示意)
def read_file_handler(path: str) -> str:
    """
    读取文件内容。

    支持:
    - 普通文件
    - 符号链接 (跟随)
    - 二进制文件 (Base64 编码)
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        return f"Error: File not found: {path}"

    if p.is_file():
        content = p.read_text(errors="replace")
        return content

    if p.is_dir():
        return "\n".join(list(p.iterdir()))

    return f"Error: Not a file or directory: {path}"

def write_file_handler(path: str, content: str) -> str:
    """写入文件 (原子操作)"""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    # 原子写入
    tmp = p.with_suffix('.tmp')
    tmp.write_text(content)
    tmp.rename(p)

    return f"Written {len(content)} bytes to {path}"
```

---

## 3.3 终端工具 (terminal_tool)

**文件**: `tools/terminal_tool.py` (~57,000 字符)

### 3.3.1 工具列表

| 工具名 | 功能 | 签名 |
|--------|------|------|
| `Bash` | 执行 Shell 命令 | `(command: str, timeout?: int) -> str` |
| `InteractiveBash` | 交互式 Shell | `(command: str) -> str` |
| `BackgroundProcess` | 后台进程管理 | `(action: str, pid?: str) -> str` |
| `ProcessList` | 列出进程 | `() -> str` |

### 3.3.2 Bash 执行器

```python
# tools/terminal_tool.py (结构示意)
class BashExecutor:
    """跨平台 Bash 执行器"""

    def execute(self, command: str, timeout: int = 30) -> dict:
        """
        执行命令并返回结果。

        Returns:
            {
                "stdout": str,
                "stderr": str,
                "returncode": int,
                "timed_out": bool
            }
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timed_out": False
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1,
                "timed_out": True
            }
```

### 3.3.3 异步执行

```python
async def async_bash_handler(command: str, timeout: int = 30) -> str:
    """异步 Bash 执行"""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout
        )
        return stdout.decode()
    except asyncio.TimeoutError:
        proc.kill()
        return f"Command timed out after {timeout}s"
```

---

## 3.4 Web 工具 (web_tools)

**文件**: `tools/web_tools.py` (~76,000 字符)

### 3.4.1 工具列表

| 工具名 | 功能 | 签名 |
|--------|------|------|
| `WebSearch` | 网络搜索 | `(query: str, num_results?: int) -> str` |
| `WebExtract` | 提取网页内容 | `(url: str, query?: str) -> str` |
| `WebFetch` | 获取页面 | `(url: str) -> str` |

### 3.4.2 搜索处理

```python
# tools/web_tools.py (结构示意)
def web_search_handler(query: str, num_results: int = 5) -> str:
    """
    执行网络搜索。

    支持多种搜索后端:
    - Tavily (优先)
    - DuckDuckGo (备选)
    - Google (需要 API key)
    """
    # 1. 尝试 Tavily
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        return _tavily_search(query, num_results, tavily_key)

    # 2. 备选 DuckDuckGo
    return _duckduckgo_search(query, num_results)

def _tavily_search(query: str, num_results: int, api_key: str) -> str:
    """Tavily 搜索 API"""
    response = requests.post(
        "https://api.tavily.com/search",
        json={"query": query, "max_results": num_results},
        headers={"Authorization": f"Bearer {api_key}"}
    )
    data = response.json()

    results = []
    for item in data.get("results", []):
        results.append(f"## {item['title']}\n{item['url']}\n{item['content']}\n")

    return "\n---\n".join(results)
```

---

## 3.5 浏览器工具 (browser_tool)

**文件**: `tools/browser_tool.py` (~77,000 字符)

### 3.5.1 工具列表

| 工具名 | 功能 | 签名 |
|--------|------|------|
| `BrowserNavigate` | 导航到 URL | `(url: str) -> str` |
| `BrowserClick` | 点击元素 | `(selector: str) -> str` |
| `BrowserType` | 输入文本 | `(selector: str, text: str) -> str` |
| `BrowserScreenshot` | 截图 | `(full_page?: bool) -> str` |
| `BrowserWait` | 等待元素 | `(selector: str, timeout?: int) -> str` |
| `BrowserEvaluate` | 执行 JS | `(script: str) -> str` |

### 3.5.2 浏览器提供者

```python
# tools/browser_providers/
class BrowserProvider(ABC):
    """浏览器提供者抽象基类"""

    @abstractmethod
    async def navigate(self, url: str) -> str: ...

    @abstractmethod
    async def click(self, selector: str) -> str: ...

    @abstractmethod
    async def screenshot(self, full_page: bool = False) -> str: ...

# 实现
class PlaywrightProvider(BrowserProvider):
    """Playwright 实现"""
    async def navigate(self, url: str) -> str:
        await self.page.goto(url)
        return f"Navigated to {url}"

class SeleniumProvider(BrowserProvider):
    """Selenium 实现 (备选)"""
    ...

class DaytonaProvider(BrowserProvider):
    """Daytona 云浏览器"""
    ...
```

---

## 3.6 MCP 工具 (mcp_tool)

**文件**: `tools/mcp_tool.py` (~77,000 字符)

### 3.6.1 MCP 协议支持

```python
# tools/mcp_tool.py
class MCPClient:
    """
    Model Context Protocol 客户端。

    允许 Hermes 连接外部 MCP 服务器，
    动态获取工具列表。
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self._tools = []
        self._session = None

    async def connect(self):
        """连接到 MCP 服务器"""
        # 1. 建立 WebSocket 连接
        # 2. 发送 initialize 请求
        # 3. 接收工具列表
        # 4. 注册工具到 registry
        pass

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用 MCP 工具"""
        response = await self._session.send_request(
            "tools/call",
            {"name": name, "arguments": arguments}
        )
        return response["result"]
```

### 3.6.2 动态工具注册

```python
# tools/mcp_tool.py
def register_handlers(r: Registry):
    """MCP 工具处理者注册"""
    # 注意: MCP 工具名是动态的，由服务器提供
    # 这里注册的是 MCP 协议处理器
    pass

async def mcp_handler(tool_name: str, **kwargs) -> str:
    """MCP 工具调用处理"""
    # 1. 找到对应的 MCP 服务器
    server = _find_server_for_tool(tool_name)

    # 2. 转发到服务器
    result = await server.call_tool(tool_name, kwargs)

    return result
```

---

## 3.7 委托工具 (delegate_tool)

**文件**: `tools/delegate_tool.py` (~32,000 字符)

### 3.7.1 子 Agent 委托

```python
# tools/delegate_tool.py
def delegate_task_handler(task: str, context: dict = None) -> str:
    """
    将任务委托给子 Agent。

    用于:
    - 并行处理多个子任务
    - 隔离危险操作
    - 使用专门的模型处理特定任务
    """
    from run_agent import AIAgent

    agent = AIAgent(
        base_url=context.get("base_url", "http://localhost:30000/v1"),
        model=context.get("model", "claude-opus-4-20250514"),
    )

    result = agent.run_conversation(task)
    return result
```

---

## 3.8 工具统计

```
┌─────────────────────────────────────────────────────────────────┐
│                       工具系统统计                                │
└─────────────────────────────────────────────────────────────────┘

总工具数:      30+
总代码量:      ~400,000+ 字符

┌─────────────────────────────────────────────────────────────────┐
│  文件大小排名                                                   │
├─────────────────────────────────────────────────────────────────┤
│  1. agent/                    ~550,000 字符 (agent 核心)         │
│  2. gateway/                  ~287,000 字符 (网关)               │
│  3. hermes_cli/               ~200,000 字符 (CLI)                │
│  4. tools/browser_tool.py     ~77,000 字符                      │
│  5. tools/mcp_tool.py         ~77,000 字符                      │
│  6. tools/web_tools.py        ~76,000 字符                      │
│  7. tools/terminal_tool.py    ~57,000 字符                      │
│  8. tools/file_tools.py       ~44,000 字符                      │
│  9. tools/skills_tool.py      ~48,000 字符                      │
│ 10. tools/skills_hub.py       ~98,000 字符                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3.9 关键源码索引

| 工具类 | 文件 | 关键函数 |
|--------|------|---------|
| 文件操作 | `tools/file_tools.py` | `Read`, `Write`, `Edit`, `Glob`, `Grep` |
| 终端控制 | `tools/terminal_tool.py` | `Bash`, `InteractiveBash` |
| Web 搜索 | `tools/web_tools.py` | `WebSearch`, `WebExtract` |
| 浏览器 | `tools/browser_tool.py` | `BrowserNavigate`, `BrowserClick` |
| MCP | `tools/mcp_tool.py` | `mcp_handler`, `MCPClient` |
| 委托 | `tools/delegate_tool.py` | `delegate_task_handler` |
| 注册表 | `tools/registry.py` | `ToolRegistry` |

---

## 3.10 下一步

- [CLI 系统架构](../04-cli-system/01-main.md) - hermes_cli 模块分析
- [Gateway 系统](../05-gateway/01-overview.md) - 网关消息处理
- [SessionDB 与记忆](../06-memory-state/01-sessiondb.md) - SQLite FTS5

---

[← 返回总目录](../SUMMARY.md) | [← 工具发现](02-discovery.md) | [CLI 系统 →](../04-cli-system/01-main.md)
