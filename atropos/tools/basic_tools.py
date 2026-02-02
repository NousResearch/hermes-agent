"""
Basic tool implementations for atropos-agent.

These tools provide simple sandbox operations:
- BashTool: Execute shell commands
- ReadFileTool: Read file contents
- WriteFileTool: Write content to files

For PoC, these run via subprocess in the local environment.
Production usage should use proper sandbox isolation.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from .base import Tool, ToolResult, ToolSchema


class BashTool(Tool):
    """
    Execute bash commands in a sandboxed environment.
    
    TODO: Nomad slot execution
    """
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout: float = 30.0,
        max_output_size: int = 10000,
    ):
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout
        self.max_output_size = max_output_size
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="bash",
            description="Execute a bash command and return stdout/stderr. Use for running shell commands, scripts, or system operations.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
            },
            required=["command"],
        )
    
    async def execute(self, command: str) -> ToolResult:
        """Execute a bash command."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {self.timeout}s",
                    metadata={"exit_code": -1, "timeout": True},
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            
            # Truncate if too long
            if len(stdout_str) > self.max_output_size:
                stdout_str = stdout_str[:self.max_output_size] + "\n... (output truncated)"
            if len(stderr_str) > self.max_output_size:
                stderr_str = stderr_str[:self.max_output_size] + "\n... (output truncated)"
            
            exit_code = process.returncode
            success = exit_code == 0
            
            output = stdout_str
            if stderr_str:
                output = f"{stdout_str}\n[stderr]\n{stderr_str}" if stdout_str else stderr_str
            
            return ToolResult(
                success=success,
                output=output.strip(),
                error="" if success else f"Exit code: {exit_code}",
                metadata={"exit_code": exit_code},
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to execute command: {str(e)}",
            )


class ReadFileTool(Tool):
    """Read the contents of a file."""
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        max_file_size: int = 100000,
    ):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.max_file_size = max_file_size
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read the contents of a file at the given path.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory)",
                },
            },
            required=["path"],
        )
    
    async def execute(self, path: str) -> ToolResult:
        """Read a file's contents."""
        try:
            file_path = self.working_dir / path
            
            # Security: prevent path traversal outside working dir
            resolved = file_path.resolve()
            if not str(resolved).startswith(str(self.working_dir.resolve())):
                return ToolResult(
                    success=False,
                    error="Access denied: path outside working directory",
                )
            
            if not resolved.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}",
                )
            
            if not resolved.is_file():
                return ToolResult(
                    success=False,
                    error=f"Not a file: {path}",
                )
            
            size = resolved.stat().st_size
            if size > self.max_file_size:
                return ToolResult(
                    success=False,
                    error=f"File too large: {size} bytes (max {self.max_file_size})",
                )
            
            content = resolved.read_text(encoding="utf-8", errors="replace")
            
            return ToolResult(
                success=True,
                output=content,
                metadata={"path": str(resolved), "size": size},
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {str(e)}",
            )


class WriteFileTool(Tool):
    """Write content to a file."""
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        max_file_size: int = 100000,
    ):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.max_file_size = max_file_size
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Write content to a file at the given path. Creates parent directories if needed.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            required=["path", "content"],
        )
    
    async def execute(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        try:
            if len(content) > self.max_file_size:
                return ToolResult(
                    success=False,
                    error=f"Content too large: {len(content)} bytes (max {self.max_file_size})",
                )
            
            file_path = self.working_dir / path
            
            # Security: prevent path traversal outside working dir
            resolved = file_path.resolve()
            if not str(resolved).startswith(str(self.working_dir.resolve())):
                return ToolResult(
                    success=False,
                    error="Access denied: path outside working directory",
                )
            
            # Create parent directories
            resolved.parent.mkdir(parents=True, exist_ok=True)
            
            resolved.write_text(content, encoding="utf-8")
            
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} bytes to {path}",
                metadata={"path": str(resolved), "size": len(content)},
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write file: {str(e)}",
            )

class FireCrawl(Tool):
    """Perform a web crawl using FireCrawl tool."""
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="firecrawl",
            description="Perform a web crawl starting from a given URL using FireCrawl.",
            parameters={
                "start_url": {
                    "type": "string",
                    "description": "The starting URL for the web crawl",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum number of pages to crawl",
                },
            },
            required=["start_url"],
        )
    
    async def execute(self, start_url: str, max_pages: int = 100) -> ToolResult:
        """Execute a web crawl using FireCrawl."""
        try:
            command = f"firecrawl --start-url {start_url} --max-pages {max_pages}"
            bash_tool = BashTool(working_dir=self.working_dir, timeout=self.timeout)
            result = await bash_tool.execute(command)
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to execute FireCrawl: {str(e)}",
            )