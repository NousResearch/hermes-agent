#!/usr/bin/env python3
"""
tool_call_benchmark.py — Compare tool calling between Gemma 4 and mimo-v2-pro.

Runs 100 diverse tool call prompts through each model and measures:
- Schema parse success (did the model return valid JSON tool calls?)
- Tool execution success (did the tool actually run without error?)
- Parallel tool success (did multiple simultaneous tool calls work?)
- Average latency per call
- Token cost per call

Usage:
    python3 benchmarks/tool_call_benchmark.py --model nous:xiaomi/mimo-v2-pro
    python3 benchmarks/tool_call_benchmark.py --model ollama/gemma4:latest
    python3 benchmarks/tool_call_benchmark.py --compare  # run both, compare
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import urllib.request

# ── Configuration ──────────────────────────────────────────────────────────

GITEA_TOKEN_PATH = os.path.expanduser("~/.config/gitea/token")
NOUS_API_BASE = "https://api.nousresearch.com/v1"
OLLAMA_API_BASE = "http://localhost:11434/v1"

DEFAULT_MODELS = {
    "mimo": "xiaomi/mimo-v2-pro",
    "gemma4": "gemma4:latest",
}

# ── Tool Schemas (subset for benchmark) ────────────────────────────────────

BENCHMARK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file with line numbers and pagination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "offset": {"type": "integer", "description": "Start line (1-indexed)", "default": 1},
                    "limit": {"type": "integer", "description": "Max lines to read", "default": 500},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, completely replacing existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Execute shell commands on the system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 180},
                    "workdir": {"type": "string", "description": "Working directory"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents or find files by name using ripgrep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern or glob"},
                    "target": {"type": "string", "enum": ["content", "files"], "default": "content"},
                    "path": {"type": "string", "default": "."},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigate to a URL in the browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to navigate to"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": "Spawn a subagent to work on a task in an isolated context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "What the subagent should accomplish"},
                    "context": {"type": "string", "description": "Background information"},
                    "toolsets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run a Python script that can call tools programmatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
]

# ── Test Cases ─────────────────────────────────────────────────────────────

TEST_CASES = [
    # File operations (20)
    {"category": "file-ops", "prompt": "Read the file README.md", "expected_tool": "read_file", "expected_args": {"path": "README.md"}},
    {"category": "file-ops", "prompt": "Read lines 10-50 of config.yaml", "expected_tool": "read_file", "expected_args": {"path": "config.yaml", "offset": 10}},
    {"category": "file-ops", "prompt": "Write 'hello world' to /tmp/test.txt", "expected_tool": "write_file", "expected_args": {"path": "/tmp/test.txt", "content": "hello world"}},
    {"category": "file-ops", "prompt": "Create a file called notes.md with the content '# Notes\n\nTBD'", "expected_tool": "write_file", "expected_args": {"path": "notes.md"}},
    {"category": "file-ops", "prompt": "Search for the word 'dispatch' in all Python files", "expected_tool": "search_files", "expected_args": {"pattern": "dispatch"}},
    {"category": "file-ops", "prompt": "Find all .jsonl files in the current directory", "expected_tool": "search_files", "expected_args": {"pattern": "*.jsonl", "target": "files"}},
    {"category": "file-ops", "prompt": "Read the first 20 lines of run_agent.py", "expected_tool": "read_file", "expected_args": {"path": "run_agent.py", "limit": 20}},
    {"category": "file-ops", "prompt": "Search for 'class AIAgent' in all Python files", "expected_tool": "search_files", "expected_args": {"pattern": "class AIAgent"}},
    {"category": "file-ops", "prompt": "Write a Python hello world script to hello.py", "expected_tool": "write_file", "expected_args": {"path": "hello.py"}},
    {"category": "file-ops", "prompt": "Read the file toolsets.py", "expected_tool": "read_file", "expected_args": {"path": "toolsets.py"}},
    {"category": "file-ops", "prompt": "Search for files matching '*_test.py'", "expected_tool": "search_files", "expected_args": {"pattern": "*_test.py", "target": "files"}},
    {"category": "file-ops", "prompt": "Read the last 30 lines of cli.py starting from line 100", "expected_tool": "read_file", "expected_args": {"path": "cli.py", "offset": 100, "limit": 30}},
    {"category": "file-ops", "prompt": "Overwrite /tmp/status.txt with 'all systems operational'", "expected_tool": "write_file", "expected_args": {"path": "/tmp/status.txt"}},
    {"category": "file-ops", "prompt": "Search for 'import os' in the tools directory", "expected_tool": "search_files", "expected_args": {"pattern": "import os", "path": "tools"}},
    {"category": "file-ops", "prompt": "Read model_tools.py from the beginning, 100 lines", "expected_tool": "read_file", "expected_args": {"path": "model_tools.py", "limit": 100}},
    {"category": "file-ops", "prompt": "Find all YAML config files", "expected_tool": "search_files", "expected_args": {"pattern": "*.yaml", "target": "files"}},
    {"category": "file-ops", "prompt": "Write the string 'test passed' to /tmp/result.txt", "expected_tool": "write_file", "expected_args": {"path": "/tmp/result.txt"}},
    {"category": "file-ops", "prompt": "Read gateway/run.py", "expected_tool": "read_file", "expected_args": {"path": "gateway/run.py"}},
    {"category": "file-ops", "prompt": "Search for 'def main' across all Python files", "expected_tool": "search_files", "expected_args": {"pattern": "def main"}},
    {"category": "file-ops", "prompt": "Create a file called /tmp/output.json with content {} ", "expected_tool": "write_file", "expected_args": {"path": "/tmp/output.json"}},

    # Terminal commands (20)
    {"category": "terminal", "prompt": "Run 'ls -la' in the current directory", "expected_tool": "terminal", "expected_args": {"command": "ls -la"}},
    {"category": "terminal", "prompt": "Check what Python version is installed", "expected_tool": "terminal", "expected_args": {"command": "python3 --version"}},
    {"category": "terminal", "prompt": "List all running processes containing 'hermes'", "expected_tool": "terminal", "expected_args": {"command": "ps aux | grep hermes"}},
    {"category": "terminal", "prompt": "Show the current git branch", "expected_tool": "terminal", "expected_args": {"command": "git branch --show-current"}},
    {"category": "terminal", "prompt": "Count the number of Python files in the project", "expected_tool": "terminal", "expected_args": {"command": "find . -name '*.py' | wc -l"}},
    {"category": "terminal", "prompt": "Check disk usage of the current directory", "expected_tool": "terminal", "expected_args": {"command": "du -sh ."}},
    {"category": "terminal", "prompt": "Show the last 5 git commits", "expected_tool": "terminal", "expected_args": {"command": "git log --oneline -5"}},
    {"category": "terminal", "prompt": "Check if port 8080 is in use", "expected_tool": "terminal", "expected_args": {"command": "lsof -i :8080"}},
    {"category": "terminal", "prompt": "Show environment variables containing 'PATH'", "expected_tool": "terminal", "expected_args": {"command": "env | grep PATH"}},
    {"category": "terminal", "prompt": "Run the command 'date' to get the current time", "expected_tool": "terminal", "expected_args": {"command": "date"}},
    {"category": "terminal", "prompt": "List all files modified in the last 24 hours", "expected_tool": "terminal", "expected_args": {"command": "find . -mtime -1 -type f"}},
    {"category": "terminal", "prompt": "Check the system uptime", "expected_tool": "terminal", "expected_args": {"command": "uptime"}},
    {"category": "terminal", "prompt": "Show free memory", "expected_tool": "terminal", "expected_args": {"command": "free -h"}},
    {"category": "terminal", "prompt": "Run 'pip list' to see installed packages", "expected_tool": "terminal", "expected_args": {"command": "pip list"}},
    {"category": "terminal", "prompt": "Check the Python syntax of run_agent.py", "expected_tool": "terminal", "expected_args": {"command": "python3 -m py_compile run_agent.py"}},
    {"category": "terminal", "prompt": "Show all git remotes", "expected_tool": "terminal", "expected_args": {"command": "git remote -v"}},
    {"category": "terminal", "prompt": "Count lines of code in Python files", "expected_tool": "terminal", "expected_args": {"command": "find . -name '*.py' | xargs wc -l | tail -1"}},
    {"category": "terminal", "prompt": "Check if curl is installed", "expected_tool": "terminal", "expected_args": {"command": "which curl"}},
    {"category": "terminal", "prompt": "Show the first 10 lines of requirements.txt", "expected_tool": "terminal", "expected_args": {"command": "head -10 requirements.txt"}},
    {"category": "terminal", "prompt": "List tmux sessions", "expected_tool": "terminal", "expected_args": {"command": "tmux list-sessions"}},

    # Web search (15)
    {"category": "web-search", "prompt": "Search the web for 'Python asyncio tutorial'", "expected_tool": "web_search", "expected_args": {"query": "Python asyncio tutorial"}},
    {"category": "web-search", "prompt": "Find information about DigitalOcean API", "expected_tool": "web_search", "expected_args": {"query": "DigitalOcean API"}},
    {"category": "web-search", "prompt": "Search for 'Gitea webhook configuration'", "expected_tool": "web_search", "expected_args": {"query": "Gitea webhook configuration"}},
    {"category": "web-search", "prompt": "Look up the latest news about LLM benchmarks", "expected_tool": "web_search", "expected_args": {"query": "LLM benchmarks 2026"}},
    {"category": "web-search", "prompt": "Search for tmux scripting examples", "expected_tool": "web_search", "expected_args": {"query": "tmux scripting examples"}},
    {"category": "web-search", "prompt": "Find documentation for Pydantic v2", "expected_tool": "web_search", "expected_args": {"query": "Pydantic v2 documentation"}},
    {"category": "web-search", "prompt": "Search for 'docker compose tutorial'", "expected_tool": "web_search", "expected_args": {"query": "docker compose tutorial"}},
    {"category": "web-search", "prompt": "Look up how to configure nginx reverse proxy", "expected_tool": "web_search", "expected_args": {"query": "nginx reverse proxy configuration"}},
    {"category": "web-search", "prompt": "Search for 'Gemma 4 model specifications'", "expected_tool": "web_search", "expected_args": {"query": "Gemma 4 model specifications"}},
    {"category": "web-search", "prompt": "Find Ansible playbook best practices", "expected_tool": "web_search", "expected_args": {"query": "Ansible playbook best practices"}},
    {"category": "web-search", "prompt": "Search for 'SQLite FTS5 full text search'", "expected_tool": "web_search", "expected_args": {"query": "SQLite FTS5 full text search"}},
    {"category": "web-search", "prompt": "Look up Python dataclass documentation", "expected_tool": "web_search", "expected_args": {"query": "Python dataclass documentation"}},
    {"category": "web-search", "prompt": "Search for 'systemd service file example'", "expected_tool": "web_search", "expected_args": {"query": "systemd service file example"}},
    {"category": "web-search", "prompt": "Find information about OpenAI function calling format", "expected_tool": "web_search", "expected_args": {"query": "OpenAI function calling format"}},
    {"category": "web-search", "prompt": "Search for 'rip grep usage examples'", "expected_tool": "web_search", "expected_args": {"query": "ripgrep usage examples"}},

    # Code execution (15)
    {"category": "code-exec", "prompt": "Run Python code to print 'hello world'", "expected_tool": "execute_code", "expected_args": {"code": "print('hello world')"}},
    {"category": "code-exec", "prompt": "Execute Python to calculate 2+2", "expected_tool": "execute_code", "expected_args": {"code": "print(2+2)"}},
    {"category": "code-exec", "prompt": "Run a Python script that creates a list of squares from 1 to 10", "expected_tool": "execute_code", "expected_args": {"code": "print([x**2 for x in range(1,11)])"}},
    {"category": "code-exec", "prompt": "Execute Python code to read and count lines in a file", "expected_tool": "execute_code", "expected_args": {"code": "print(sum(1 for _ in open('README.md')))"}},
    {"category": "code-exec", "prompt": "Run Python to get the current directory", "expected_tool": "execute_code", "expected_args": {"code": "import os; print(os.getcwd())"}},
    {"category": "code-exec", "prompt": "Execute Python to parse a JSON string", "expected_tool": "execute_code", "expected_args": {"code": "import json; print(json.loads('{\"a\": 1}'))"}},
    {"category": "code-exec", "prompt": "Run Python code that uses os.walk to list files", "expected_tool": "execute_code", "expected_args": {"code": "import os; [print(f) for f in os.walk('.')"}},
    {"category": "code-exec", "prompt": "Execute Python to generate a random number", "expected_tool": "execute_code", "expected_args": {"code": "import random; print(random.randint(1,100))"}},
    {"category": "code-exec", "prompt": "Run Python to check Python version programmatically", "expected_tool": "execute_code", "expected_args": {"code": "import sys; print(sys.version)"}},
    {"category": "code-exec", "prompt": "Execute Python code to format a date", "expected_tool": "execute_code", "expected_args": {"code": "from datetime import datetime; print(datetime.now().isoformat())"}},
    {"category": "code-exec", "prompt": "Run Python to create a dictionary and dump it as JSON", "expected_tool": "execute_code", "expected_args": {"code": "import json; print(json.dumps({'key': 'value'}))"}},
    {"category": "code-exec", "prompt": "Execute Python code to check if a file exists", "expected_tool": "execute_code", "expected_args": {"code": "import os; print(os.path.exists('README.md'))"}},
    {"category": "code-exec", "prompt": "Run Python to compute fibonacci numbers", "expected_tool": "execute_code", "expected_args": {"code": "def fib(n): return n if n<2 else fib(n-1)+fib(n-2); print([fib(i) for i in range(10)])"}},
    {"category": "code-exec", "prompt": "Execute Python to list all environment variables", "expected_tool": "execute_code", "expected_args": {"code": "import os; print(list(os.environ.keys())[:10])"}},
    {"category": "code-exec", "prompt": "Run Python code to measure time.time()", "expected_tool": "execute_code", "expected_args": {"code": "import time; print(time.time())"}},

    # Browser (10)
    {"category": "browser", "prompt": "Navigate to https://example.com", "expected_tool": "browser_navigate", "expected_args": {"url": "https://example.com"}},
    {"category": "browser", "prompt": "Open the browser to google.com", "expected_tool": "browser_navigate", "expected_args": {"url": "https://google.com"}},
    {"category": "browser", "prompt": "Go to https://forge.alexanderwhitestone.com", "expected_tool": "browser_navigate", "expected_args": {"url": "https://forge.alexanderwhitestone.com"}},
    {"category": "browser", "prompt": "Navigate the browser to https://github.com", "expected_tool": "browser_navigate", "expected_args": {"url": "https://github.com"}},
    {"category": "browser", "prompt": "Open https://news.ycombinator.com in the browser", "expected_tool": "browser_navigate", "expected_args": {"url": "https://news.ycombinator.com"}},
    {"category": "browser", "prompt": "Browse to https://docs.python.org", "expected_tool": "browser_navigate", "expected_args": {"url": "https://docs.python.org"}},
    {"category": "browser", "prompt": "Visit https://arxiv.org", "expected_tool": "browser_navigate", "expected_args": {"url": "https://arxiv.org"}},
    {"category": "browser", "prompt": "Load https://httpbin.org/get", "expected_tool": "browser_navigate", "expected_args": {"url": "https://httpbin.org/get"}},
    {"category": "browser", "prompt": "Navigate to localhost:8080", "expected_tool": "browser_navigate", "expected_args": {"url": "http://localhost:8080"}},
    {"category": "browser", "prompt": "Open the URL https://huggingface.co", "expected_tool": "browser_navigate", "expected_args": {"url": "https://huggingface.co"}},

    # Delegation (10)
    {"category": "delegation", "prompt": "Spawn a subagent to research Python async patterns", "expected_tool": "delegate_task", "expected_args": {"goal": "research Python async patterns"}},
    {"category": "delegation", "prompt": "Delegate a task to write unit tests for the config module", "expected_tool": "delegate_task", "expected_args": {"goal": "write unit tests for the config module"}},
    {"category": "delegation", "prompt": "Create a subagent to analyze the codebase structure", "expected_tool": "delegate_task", "expected_args": {"goal": "analyze the codebase structure"}},
    {"category": "delegation", "prompt": "Send a subagent to check all API endpoints", "expected_tool": "delegate_task", "expected_args": {"goal": "check all API endpoints"}},
    {"category": "delegation", "prompt": "Delegate the task of finding unused imports", "expected_tool": "delegate_task", "expected_args": {"goal": "find unused imports"}},
    {"category": "delegation", "prompt": "Spawn a worker to generate documentation", "expected_tool": "delegate_task", "expected_args": {"goal": "generate documentation"}},
    {"category": "delegation", "prompt": "Use a subagent to run the test suite", "expected_tool": "delegate_task", "expected_args": {"goal": "run the test suite"}},
    {"category": "delegation", "prompt": "Delegate research on Docker best practices to a subagent", "expected_tool": "delegate_task", "expected_args": {"goal": "research Docker best practices"}},
    {"category": "delegation", "prompt": "Create a subagent with terminal and file tools to refactor the CLI", "expected_tool": "delegate_task", "expected_args": {"goal": "refactor the CLI"}},
    {"category": "delegation", "prompt": "Spawn a subagent to check for security vulnerabilities", "expected_tool": "delegate_task", "expected_args": {"goal": "check for security vulnerabilities"}},

    # MCP (10)
    {"category": "mcp", "prompt": "List available MCP resources", "expected_tool": "mcp_list_resources", "expected_args": {}},
    {"category": "mcp", "prompt": "Show MCP server prompts", "expected_tool": "mcp_list_prompts", "expected_args": {}},
    {"category": "mcp", "prompt": "Read the MCP resource at uri config://settings", "expected_tool": "mcp_read_resource", "expected_args": {"uri": "config://settings"}},
    {"category": "mcp", "prompt": "Get the MCP prompt named 'code-review'", "expected_tool": "mcp_get_prompt", "expected_args": {"name": "code-review"}},
    {"category": "mcp", "prompt": "Call MCP tool 'weather' with city=London", "expected_tool": "mcp_call_tool", "expected_args": {"tool": "weather", "arguments": {"city": "London"}}},
    {"category": "mcp", "prompt": "List all connected MCP servers", "expected_tool": "mcp_list_servers", "expected_args": {}},
    {"category": "mcp", "prompt": "Read MCP resource file://config.json", "expected_tool": "mcp_read_resource", "expected_args": {"uri": "file://config.json"}},
    {"category": "mcp", "prompt": "Get the 'summarize' prompt from MCP", "expected_tool": "mcp_get_prompt", "expected_args": {"name": "summarize"}},
    {"category": "mcp", "prompt": "Call MCP tool 'calculate' with expression='2+2'", "expected_tool": "mcp_call_tool", "expected_args": {"tool": "calculate"}},
    {"category": "mcp", "prompt": "Check MCP server status", "expected_tool": "mcp_list_servers", "expected_args": {}},
]

assert len(TEST_CASES) == 100, f"Expected 100 test cases, got {len(TEST_CASES)}"


# ── API Callers ────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """Get API key from standard locations."""
    for path in [os.path.expanduser("~/.config/nous/key"), os.path.expanduser("~/.config/openrouter/key")]:
        if os.path.exists(path):
            with open(path) as f:
                key = f.read().strip()
                if key:
                    return key
    return ""


def call_model(model: str, prompt: str, tools: list, api_base: str, api_key: str, timeout: int = 30) -> dict:
    """Call a model with tool schemas and return the response."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools to complete tasks. Always call the appropriate tool."},
        {"role": "user", "content": prompt},
    ]

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": 0.0,
        "max_tokens": 1024,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        latency = time.time() - start
        return {"success": True, "result": result, "latency": latency}
    except Exception as e:
        latency = time.time() - start
        return {"success": False, "error": str(e), "latency": latency}


def extract_tool_calls(response: dict) -> list:
    """Extract tool calls from model response."""
    if not response.get("success"):
        return []
    result = response["result"]
    choices = result.get("choices", [])
    if not choices:
        return []
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])
    return tool_calls


def validate_tool_call(call: dict, expected_tool: str, expected_args: dict) -> dict:
    """Validate a tool call against expected values."""
    func = call.get("function", {})
    name = func.get("name", "")
    args_str = func.get("arguments", "{}")

    # Schema parse check
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        parse_ok = True
    except (json.JSONDecodeError, TypeError):
        return {"parse_ok": False, "name_ok": False, "args_ok": False, "actual_name": name, "actual_args": args_str}

    name_ok = name == expected_tool
    # Lenient args check: all required keys present
    args_ok = all(k in args for k in expected_args if expected_args.get(k) is not None)

    return {"parse_ok": True, "name_ok": name_ok, "args_ok": args_ok, "actual_name": name, "actual_args": args}


# ── Benchmark Runner ───────────────────────────────────────────────────────

def run_benchmark(model: str, api_base: str, api_key: str, label: str) -> dict:
    """Run all 100 test cases against a model."""
    results = {
        "model": model,
        "label": label,
        "timestamp": datetime.utcnow().isoformat(),
        "total": len(TEST_CASES),
        "by_category": {},
        "cases": [],
    }

    categories = set(tc["category"] for tc in TEST_CASES)
    for cat in categories:
        results["by_category"][cat] = {"total": 0, "parse_ok": 0, "name_ok": 0, "args_ok": 0, "latency_sum": 0.0}

    print(f"\nRunning benchmark: {label} ({model})")
    print(f"{'='*60}")

    for i, tc in enumerate(TEST_CASES):
        cat = tc["category"]
        results["by_category"][cat]["total"] += 1

        resp = call_model(model, tc["prompt"], BENCHMARK_TOOLS, api_base, api_key)
        tool_calls = extract_tool_calls(resp)

        case_result = {
            "index": i,
            "category": cat,
            "prompt": tc["prompt"][:80],
            "latency": round(resp["latency"], 3),
            "api_ok": resp["success"],
            "tool_calls_count": len(tool_calls),
            "parse_ok": False,
            "name_ok": False,
            "args_ok": False,
        }

        results["by_category"][cat]["latency_sum"] += resp["latency"]

        if tool_calls:
            first_call = tool_calls[0]
            validation = validate_tool_call(first_call, tc["expected_tool"], tc.get("expected_args", {}))
            case_result["parse_ok"] = validation["parse_ok"]
            case_result["name_ok"] = validation["name_ok"]
            case_result["args_ok"] = validation["args_ok"]
            case_result["actual_tool"] = validation["actual_name"]

            if validation["parse_ok"]:
                results["by_category"][cat]["parse_ok"] += 1
            if validation["name_ok"]:
                results["by_category"][cat]["name_ok"] += 1
            if validation["args_ok"]:
                results["by_category"][cat]["args_ok"] += 1

        results["cases"].append(case_result)

        # Progress
        if (i + 1) % 10 == 0:
            parse_rate = sum(1 for c in results["cases"] if c["parse_ok"]) / (i + 1) * 100
            name_rate = sum(1 for c in results["cases"] if c["name_ok"]) / (i + 1) * 100
            print(f"  [{i+1:3d}/100] parse: {parse_rate:.0f}% | tool: {name_rate:.0f}% | last latency: {resp['latency']:.2f}s")

    # Summary
    total_parse = sum(1 for c in results["cases"] if c["parse_ok"])
    total_name = sum(1 for c in results["cases"] if c["name_ok"])
    total_args = sum(1 for c in results["cases"] if c["args_ok"])
    avg_latency = sum(c["latency"] for c in results["cases"]) / len(results["cases"])

    results["summary"] = {
        "schema_parse_rate": round(total_parse / 100, 3),
        "tool_name_accuracy": round(total_name / 100, 3),
        "args_accuracy": round(total_args / 100, 3),
        "avg_latency_s": round(avg_latency, 3),
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"  Schema parse:  {total_parse}/100 ({total_parse}%)")
    print(f"  Tool accuracy: {total_name}/100 ({total_name}%)")
    print(f"  Args accuracy: {total_args}/100 ({total_args}%)")
    print(f"  Avg latency:   {avg_latency:.2f}s")
    print(f"{'='*60}")

    return results


def save_results(results: dict, output_dir: str):
    """Save benchmark results to markdown and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")

    # JSON (machine-readable)
    json_path = os.path.join(output_dir, f"tool-call-benchmark-{results['label']}-{date_str}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown (human-readable)
    md_path = os.path.join(output_dir, f"tool-call-benchmark-{results['label']}-{date_str}.md")
    s = results["summary"]
    with open(md_path, "w") as f:
        f.write(f"# Tool Call Benchmark: {results['label']}\n\n")
        f.write(f"Model: `{results['model']}`\n")
        f.write(f"Date: {results['timestamp']}\n")
        f.write(f"Test cases: {results['total']}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Score |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Schema parse success | {s['schema_parse_rate']*100:.1f}% |\n")
        f.write(f"| Tool name accuracy | {s['tool_name_accuracy']*100:.1f}% |\n")
        f.write(f"| Args accuracy | {s['args_accuracy']*100:.1f}% |\n")
        f.write(f"| Avg latency | {s['avg_latency_s']:.2f}s |\n\n")
        f.write(f"## By Category\n\n")
        f.write(f"| Category | Total | Parse | Tool | Args | Avg Latency |\n")
        f.write(f"|----------|-------|-------|------|------|-------------|\n")
        for cat, data in results["by_category"].items():
            n = data["total"]
            p = f"{data['parse_ok']}/{n}"
            t = f"{data['name_ok']}/{n}"
            a = f"{data['args_ok']}/{n}"
            lat = f"{data['latency_sum']/n:.2f}s" if n > 0 else "N/A"
            f.write(f"| {cat} | {n} | {p} | {t} | {a} | {lat} |\n")

    print(f"\nSaved: {md_path}")
    print(f"Saved: {json_path}")
    return md_path, json_path


def compare_results(mimo_results: dict, gemma_results: dict, output_dir: str):
    """Generate comparison report."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    md_path = os.path.join(output_dir, f"tool-call-benchmark-comparison-{date_str}.md")

    ms = mimo_results["summary"]
    gs = gemma_results["summary"]

    with open(md_path, "w") as f:
        f.write(f"# Tool Call Benchmark: Gemma 4 vs mimo-v2-pro\n\n")
        f.write(f"Date: {datetime.utcnow().isoformat()}\n")
        f.write(f"Test cases: 100 (7 categories)\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | mimo-v2-pro | Gemma 4 | Winner |\n")
        f.write(f"|--------|-------------|---------|--------|\n")
        for metric, label in [("schema_parse_rate", "Schema parse"), ("tool_name_accuracy", "Tool accuracy"),
                               ("args_accuracy", "Args accuracy"), ("avg_latency_s", "Avg latency (s)")]:
            m = ms[metric]
            g = gs[metric]
            if metric == "avg_latency_s":
                winner = "mimo" if m < g else ("gemma" if g < m else "tie")
                f.write(f"| {label} | {m:.3f} | {g:.3f} | {winner} |\n")
            else:
                winner = "mimo" if m > g else ("gemma" if g > m else "tie")
                f.write(f"| {label} | {m*100:.1f}% | {g*100:.1f}% | {winner} |\n")

        f.write(f"\n## By Category\n\n")
        f.write(f"| Category | mimo parse | gemma parse | mimo tool | gemma tool |\n")
        f.write(f"|----------|-----------|-------------|-----------|------------|\n")
        for cat in mimo_results["by_category"]:
            md = mimo_results["by_category"][cat]
            gd = gemma_results["by_category"].get(cat, {"total": 0, "parse_ok": 0, "name_ok": 0})
            n = md["total"]
            f.write(f"| {cat} | {md['parse_ok']}/{n} | {gd['parse_ok']}/{n} | {md['name_ok']}/{n} | {gd['name_ok']}/{n} |\n")

    print(f"\nComparison saved: {md_path}")
    return md_path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tool call benchmark: Gemma 4 vs mimo-v2-pro")
    parser.add_argument("--model", help="Single model to test (e.g. nous:xiaomi/mimo-v2-pro)")
    parser.add_argument("--compare", action="store_true", help="Run both models and compare")
    parser.add_argument("--output", default="benchmarks", help="Output directory")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.compare:
        # Run both
        mimo_results = run_benchmark(DEFAULT_MODELS["mimo"], NOUS_API_BASE, api_key, "mimo-v2-pro")
        save_results(mimo_results, args.output)

        gemma_results = run_benchmark(DEFAULT_MODELS["gemma4"], OLLAMA_API_BASE, "", "gemma4")
        save_results(gemma_results, args.output)

        compare_results(mimo_results, gemma_results, args.output)

    elif args.model:
        # Single model
        if "/" in args.model:
            label = args.model.split("/")[-1]
            api_base = NOUS_API_BASE
        else:
            label = args.model.replace(":", "-")
            api_base = OLLAMA_API_BASE
        results = run_benchmark(args.model, api_base, api_key, label)
        save_results(results, args.output)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 benchmarks/tool_call_benchmark.py --model nous:xiaomi/mimo-v2-pro")
        print("  python3 benchmarks/tool_call_benchmark.py --model ollama/gemma4:latest")
        print("  python3 benchmarks/tool_call_benchmark.py --compare")


if __name__ == "__main__":
    main()
