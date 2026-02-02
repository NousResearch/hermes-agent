"""
Base Tool abstraction for atropos-agent.

Tools follow a simple pattern:
1. Define schema (name, description, parameters)
2. Implement execute() method
3. Return ToolResult with output/error

Tool calls use Hermes-style XML tags:
<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>
"""

import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


@dataclass
class ToolSchema:
    """JSON Schema for a tool's parameters."""
    
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    external: bool = False  # Whether the tool must be executed via an external ToolServer (secret proxy) and not inside the sandbox.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }
    
    def to_prompt_description(self) -> str:
        """Convert to human-readable description for system prompt."""
        params_desc = []
        for name, spec in self.parameters.items():
            req = "(required)" if name in self.required else "(optional)"
            desc = spec.get("description", "")
            param_type = spec.get("type", "string")
            params_desc.append(f"  - {name} ({param_type}) {req}: {desc}")
        
        params_str = "\n".join(params_desc) if params_desc else "  (no parameters)"
        return f"**{self.name}**: {self.description}\nParameters:\n{params_str}"


@dataclass
class ToolCall:
    """A parsed tool call from model output."""
    
    name: str
    arguments: Dict[str, Any]
    raw_text: str = ""  # Original XML/JSON text
    uniq_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique tool-call id for traceability/reconstruction.
    
    @classmethod
    def parse_from_text(cls, text: str) -> List["ToolCall"]:
        """
        Extract tool calls from text using Hermes-style XML tags.
        
        Format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        """
        calls = []
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                uniq_id = data.get("uniq_id") or data.get("id") or str(uuid.uuid4())
                calls.append(cls(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                    raw_text=match,
                    uniq_id=uniq_id,
                ))
            except json.JSONDecodeError:
                # Skip malformed tool calls
                continue
        
        return calls
    
    @classmethod
    def has_tool_call(cls, text: str) -> bool:
        """Check if text contains any tool calls."""
        return bool(re.search(r"<tool_call>", text))


@dataclass
class ToolResult:
    """Result from executing a tool."""
    
    success: bool
    output: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    uniq_id: Optional[str] = None  # Should match ToolCall.uniq_id for async execution tracking.
    
    def to_xml(self) -> str:
        """Format as XML for including in conversation."""
        data = {
            "success": self.success,
            "output": self.output,
        }
        if self.uniq_id:
            data["uniq_id"] = self.uniq_id
        if self.error:
            data["error"] = self.error
        if self.metadata:
            data["metadata"] = self.metadata
        return f"<tool_response>{json.dumps(data)}</tool_response>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "uniq_id": self.uniq_id,
        }


class Tool(ABC):
    """
    Abstract base class for tools.
    
    Subclasses must implement:
    - schema: ToolSchema describing the tool
    - execute(): async method that performs the tool action
    """
    
    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the tool's schema."""
        pass
    
    @property
    def name(self) -> str:
        """Tool name (from schema)."""
        return self.schema.name
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult with success/failure and output
        """
        pass
    
    def is_available(self) -> tuple[bool, str | None]:
        """
        Return whether this tool should be exposed/executable in the current process.

        Tools that depend on optional binaries/services/env vars can override this
        to avoid advertising a tool that will fail at runtime.
        """
        return True, None

    async def __call__(self, **kwargs) -> ToolResult:
        """Allow calling tool instance directly."""
        return await self.execute(**kwargs)

# Note: This is only wrapping declarations for the external ToolServer (for execution on external process tools), and tools preinstalled in envs
class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def get_schemas(self) -> List[ToolSchema]:
        """Get schemas for all registered tools."""
        return [tool.schema for tool in self._tools.values()]
    
    def get_prompt_description(self) -> str:
        """Generate tool descriptions for system prompt."""
        descriptions = [tool.schema.to_prompt_description() for tool in self._tools.values()]
        return "\n\n".join(descriptions)
    
    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {call.name}",
                uniq_id=call.uniq_id,
            )
        
        try:
            result = await tool.execute(**call.arguments)
            if result.uniq_id is None:
                result.uniq_id = call.uniq_id
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Tool execution error: {str(e)}",
                uniq_id=call.uniq_id,
            )


# =============================================================================
# FastAPI / transport models
# =============================================================================


class ToolCallPayload(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    uniq_id: str

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> "ToolCallPayload":
        return cls(name=call.name, arguments=call.arguments, uniq_id=call.uniq_id)

    def to_tool_call(self) -> ToolCall:
        return ToolCall(name=self.name, arguments=self.arguments, uniq_id=self.uniq_id)


class ToolResultPayload(BaseModel):
    success: bool
    output: str = ""
    error: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    uniq_id: Optional[str] = None

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> "ToolResultPayload":
        return cls(
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=result.metadata,
            uniq_id=result.uniq_id,
        )

    def to_tool_result(self) -> ToolResult:
        return ToolResult(
            success=self.success,
            output=self.output,
            error=self.error,
            metadata=self.metadata,
            uniq_id=self.uniq_id,
        )


class ToolExecutorExecuteRequest(BaseModel):
    trajectory_id: str
    tool: ToolCallPayload
    timeout_s: Optional[float] = None


class ToolExecutorReleaseRequest(BaseModel):
    trajectory_id: str
    reset_workspace: bool = False


class ToolServerExecuteRequest(BaseModel):
    trajectory_id: Optional[str] = None
    tool: ToolCallPayload
    timeout_s: Optional[float] = None
    # Optional sandbox context for tools that need workspace artifacts.
    # This is set by ToolExecutor and is NOT model-controlled.
    slot_id: Optional[str] = None
    container_addr: Optional[str] = None


# =============================================================================
# Artifact transport models
# =============================================================================


class ArtifactReadRequestPayload(BaseModel):
    trajectory_id: str
    path: str
    encoding: Literal["text", "base64"] = "text"
    max_bytes: Optional[int] = None
    include_sha256: bool = False


class ArtifactReadResponsePayload(BaseModel):
    success: bool
    content: str = ""
    error: str = ""
    encoding: str = "text"
    truncated: bool = False
    bytes: int = 0
    file_size: Optional[int] = None
    path: str = ""
    mime: Optional[str] = None
    sha256: Optional[str] = None


class ArtifactListRequestPayload(BaseModel):
    trajectory_id: str
    path: str = "."
    recursive: bool = False
    max_entries: Optional[int] = None


class ArtifactListEntryPayload(BaseModel):
    path: str
    is_dir: bool
    size: int
    mtime: float


class ArtifactListResponsePayload(BaseModel):
    success: bool
    entries: List[ArtifactListEntryPayload] = Field(default_factory=list)
    truncated: bool = False
    error: str = ""


class ArtifactArchiveRequestPayload(BaseModel):
    trajectory_id: str
    path: str = "."
    format: Literal["tar.gz", "tgz"] = "tar.gz"
    max_bytes: Optional[int] = None
    max_entries: Optional[int] = None


class ArtifactArchiveResponsePayload(BaseModel):
    success: bool
    content: str = ""
    error: str = ""
    encoding: str = "base64"
    format: str = "tar.gz"
    bytes: int = 0
    entry_count: int = 0
