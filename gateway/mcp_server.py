"""
Hermes MCP Server — expose Hermes Agent tools over the Model Context Protocol.

This allows any MCP client (Claude Desktop, Cursor, VS Code, etc.) to use
Hermes's terminal, web, file, memory, and other tools directly.

Usage:
    hermes mcp-server               # stdio transport (Claude Desktop)
    hermes mcp-server --http        # HTTP transport (port 8765)
    hermes mcp-server --http --port 9000

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "hermes": {
          "command": "hermes",
          "args": ["mcp-server"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

HERMES_MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "terminal",
        "description": (
            "Execute a shell command on the Hermes agent's machine. "
            "Supports background processes, PTY mode for interactive tools, "
            "and all configured terminal backends (local, docker, ssh, modal)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "background": {
                    "type": "boolean",
                    "description": "Run in background; returns a process ID",
                    "default": False,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 60)",
                    "default": 60,
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory override",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file on the Hermes agent's machine.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file"},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write or overwrite a file on the Hermes agent's machine.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "File contents"},
                "append": {
                    "type": "boolean",
                    "description": "Append instead of overwrite",
                    "default": False,
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web and return a list of results with titles, URLs, and snippets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_extract",
        "description": "Fetch and extract the text content of one or more URLs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to fetch",
                },
            },
            "required": ["urls"],
        },
    },
    {
        "name": "memory_read",
        "description": (
            "Read Hermes's persistent memory (MEMORY.md and USER.md). "
            "Returns the agent's accumulated knowledge about its environment and the user."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": ["memory", "user", "both"],
                    "description": "Which memory section to read",
                    "default": "both",
                },
            },
        },
    },
    {
        "name": "memory_write",
        "description": "Add or update an entry in Hermes's persistent memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": ["memory", "user"],
                    "description": "Which memory section to write to",
                },
                "key": {"type": "string", "description": "Memory entry key/title"},
                "value": {"type": "string", "description": "Memory entry value"},
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove"],
                    "default": "add",
                },
            },
            "required": ["section", "key", "value"],
        },
    },
    {
        "name": "list_skills",
        "description": "List all available Hermes skills with their names and descriptions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (optional)",
                },
            },
        },
    },
    {
        "name": "run_agent",
        "description": (
            "Run a full Hermes agent session with a given prompt. "
            "The agent has access to all its tools and will work autonomously. "
            "Returns the final response when done."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task or question for Hermes to work on",
                },
                "toolsets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Toolsets to enable (default: terminal, file, web)",
                    "default": ["terminal", "file", "web"],
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Max tool-calling iterations (default: 30)",
                    "default": 30,
                },
            },
            "required": ["prompt"],
        },
    },
]


class HermesToolBridge:
    """Bridges MCP tool calls to Hermes's internal tool implementations."""

    def __init__(self) -> None:
        self._hermes_home = os.path.expanduser(
            os.environ.get("HERMES_HOME", "~/.hermes")
        )
        self._tool_registry: Any = None
        self._agent_class: Any = None

    def _lazy_load(self) -> None:
        if self._tool_registry is None:
            try:
                sys.path.insert(0, os.path.join(self._hermes_home, "hermes-agent"))
                from tools.registry import get_tool_registry  # type: ignore[import]
                self._tool_registry = get_tool_registry()
            except ImportError:
                logger.warning("Could not import Hermes tool registry; using subprocess fallback")

        if self._agent_class is None:
            try:
                from run_agent import AIAgent  # type: ignore[import]
                self._agent_class = AIAgent
            except ImportError:
                logger.warning("Could not import AIAgent; run_agent tool will be unavailable")

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        self._lazy_load()
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return f"Error: unknown tool '{tool_name}'"
        try:
            result = await handler(**arguments)
            return result if isinstance(result, str) else json.dumps(result, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s failed", tool_name)
            return f"Error executing {tool_name}: {exc}"

    async def _tool_terminal(
        self,
        command: str,
        background: bool = False,
        timeout: int = 60,
        cwd: str | None = None,
    ) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )
        if background:
            return f"Process started with PID {proc.pid}"
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode(errors="replace").strip()
            rc = proc.returncode
            if rc != 0:
                return f"[exit {rc}]\n{output}"
            return output or "(no output)"
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: command timed out after {timeout}s"

    async def _tool_read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        try:
            with open(os.path.expanduser(path), encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            if start_line is not None or end_line is not None:
                s = (start_line or 1) - 1
                e = end_line or len(lines)
                lines = lines[s:e]
            return "".join(lines)
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except PermissionError:
            return f"Error: permission denied: {path}"

    async def _tool_write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> str:
        expanded = os.path.expanduser(path)
        os.makedirs(os.path.dirname(expanded) or ".", exist_ok=True)
        mode = "a" if append else "w"
        with open(expanded, mode, encoding="utf-8") as fh:
            fh.write(content)
        action = "Appended to" if append else "Wrote"
        return f"{action} {path} ({len(content)} chars)"

    async def _tool_web_search(self, query: str, limit: int = 5) -> str:
        try:
            from tools.web import web_search  # type: ignore[import]
            results = await web_search(query=query, limit=limit)
            return json.dumps(results, indent=2)
        except ImportError:
            pass
        return f"Web search unavailable without Hermes install. Query was: {query}"

    async def _tool_web_extract(self, urls: list[str]) -> str:
        results = []
        for url in urls[:5]:
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "HermesMCP/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = resp.read().decode(errors="replace")
                import re
                text = re.sub(r"<[^>]+>", " ", body)
                text = re.sub(r"\s+", " ", text).strip()
                results.append({"url": url, "content": text[:3000]})
            except Exception as exc:  # noqa: BLE001
                results.append({"url": url, "error": str(exc)})
        return json.dumps(results, indent=2)

    async def _tool_memory_read(self, section: str = "both") -> str:
        memories_dir = os.path.join(self._hermes_home, "memories")
        out: dict[str, str] = {}

        def _read(name: str) -> str:
            p = os.path.join(memories_dir, name)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    return fh.read()
            return "(empty)"

        if section in ("memory", "both"):
            out["MEMORY.md"] = _read("MEMORY.md")
        if section in ("user", "both"):
            out["USER.md"] = _read("USER.md")
        return json.dumps(out, indent=2)

    async def _tool_memory_write(
        self,
        section: str,
        key: str,
        value: str,
        action: str = "add",
    ) -> str:
        memories_dir = os.path.join(self._hermes_home, "memories")
        os.makedirs(memories_dir, exist_ok=True)
        fname = "MEMORY.md" if section == "memory" else "USER.md"
        path = os.path.join(memories_dir, fname)

        content = ""
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                content = fh.read()

        if action == "remove":
            lines = [l for l in content.splitlines() if key not in l]
            content = "\n".join(lines)
        elif action == "replace":
            import re
            pattern = rf"- \*\*{re.escape(key)}\*\*:.*"
            replacement = f"- **{key}**: {value}"
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
            else:
                content = content.rstrip() + f"\n- **{key}**: {value}\n"
        else:
            content = content.rstrip() + f"\n- **{key}**: {value}\n"

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"Memory {action} successful: [{section}] {key}"

    async def _tool_list_skills(self, category: str | None = None) -> str:
        skills_dir = os.path.join(self._hermes_home, "skills")
        if not os.path.isdir(skills_dir):
            return "No skills directory found at " + skills_dir

        skills = []
        for root, dirs, files in os.walk(skills_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if "SKILL.md" in files:
                skill_path = os.path.join(root, "SKILL.md")
                rel = os.path.relpath(root, skills_dir)
                if category and not rel.startswith(category):
                    continue
                desc = ""
                try:
                    with open(skill_path, encoding="utf-8") as fh:
                        in_front = False
                        for line in fh:
                            if line.strip() == "---":
                                in_front = not in_front
                                continue
                            if in_front and line.startswith("description:"):
                                desc = line.split(":", 1)[1].strip()
                                break
                except OSError:
                    pass
                skills.append({"name": rel, "description": desc})

        if not skills:
            return "No skills found" + (f" in category '{category}'" if category else "")
        return json.dumps(skills, indent=2)

    async def _tool_run_agent(
        self,
        prompt: str,
        toolsets: list[str] | None = None,
        max_iterations: int = 30,
    ) -> str:
        if self._agent_class is None:
            return (
                "Error: AIAgent not available. "
                "Make sure hermes-agent is installed and on PYTHONPATH."
            )
        toolsets = toolsets or ["terminal", "file", "web"]
        agent = self._agent_class(
            enabled_toolsets=toolsets,
            max_iterations=max_iterations,
        )
        result = agent.run_conversation(prompt)
        return result.get("final_response", str(result))


class HermesMCPServer:
    """
    Minimal MCP server implementation using stdio transport.
    Implements: initialize, tools/list, tools/call
    """

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self) -> None:
        self.bridge = HermesToolBridge()

    def _make_response(self, request_id: Any, result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _make_error(self, request_id: Any, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    async def handle_request(self, request: dict) -> dict | None:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if req_id is None:
            return None

        if method == "initialize":
            return self._make_response(
                req_id,
                {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "hermes-agent",
                        "version": "1.0.0",
                    },
                },
            )

        if method == "tools/list":
            return self._make_response(req_id, {"tools": HERMES_MCP_TOOLS})

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result_text = await self.bridge.call(tool_name, arguments)
            return self._make_response(
                req_id,
                {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": result_text.startswith("Error"),
                },
            )

        if method == "ping":
            return self._make_response(req_id, {})

        return self._make_error(req_id, -32601, f"Method not found: {method}")

    async def run_stdio(self) -> None:
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_event_loop()

        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        w_transport, w_protocol = await loop.connect_write_pipe(
            asyncio.BaseProtocol, sys.stdout
        )
        writer = asyncio.StreamWriter(w_transport, w_protocol, reader, loop)

        logger.info("Hermes MCP Server started (stdio transport)")

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                line = line.decode().strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    error = self._make_error(None, -32700, f"Parse error: {exc}")
                    writer.write((json.dumps(error) + "\n").encode())
                    await writer.drain()
                    continue

                response = await self.handle_request(request)
                if response is not None:
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()

            except asyncio.IncompleteReadError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error: %s", exc)
                break

    async def run_http(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        try:
            from aiohttp import web  # type: ignore[import]
        except ImportError:
            print("aiohttp is required for HTTP transport: pip install aiohttp", file=sys.stderr)
            sys.exit(1)

        async def mcp_handler(request: web.Request) -> web.Response:
            origin = request.headers.get("Origin", "")
            if origin and not origin.startswith(("http://localhost", "http://127.0.0.1")):
                return web.Response(status=403, text="Forbidden: invalid origin")
            try:
                body = await request.json()
            except Exception:
                return web.Response(
                    status=400,
                    content_type="application/json",
                    text=json.dumps({"error": "Invalid JSON"}),
                )
            if isinstance(body, list):
                responses = []
                for req in body:
                    resp = await self.handle_request(req)
                    if resp is not None:
                        responses.append(resp)
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(responses),
                )
            response = await self.handle_request(body)
            if response is None:
                return web.Response(status=202)
            return web.Response(
                content_type="application/json",
                text=json.dumps(response),
            )

        app = web.Application()
        app.router.add_post("/mcp", mcp_handler)
        app.router.add_get("/health", lambda r: web.Response(text="ok"))

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        print(f"Hermes MCP Server running at http://{host}:{port}/mcp", file=sys.stderr)

        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()


def main(http: bool = False, host: str = "127.0.0.1", port: int = 8765) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    server = HermesMCPServer()
    if http:
        asyncio.run(server.run_http(host=host, port=port))
    else:
        asyncio.run(server.run_stdio())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Hermes MCP Server")
    p.add_argument("--http", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    main(http=args.http, host=args.host, port=args.port)
