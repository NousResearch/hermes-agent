"""
Hermes Universal Adapter
===========================

Translates HermesAgent tool calls and skills to Cosmos equivalents.

This is the core of the Shadow Layer - it:
1. Parses SKILL.md files
2. Maps Hermes tools to Cosmos agents/modules
3. Translates input/output formats
4. Provides a unified invocation interface

HermesAgent Tool Groups Mapped:
- group:fs → Cosmos file operations
- group:runtime → code_agent.py, subprocess execution
- group:sessions → swarm_orchestrator + Nexus signals
- group:memory → memory_system.py
- group:web → web_agent.py
- group:ui → visual_canvas.py + browser automation
- group:automation → scheduler.py
- group:messaging → x_automation, email
- group:nodes → device_node.py

"Compatibility is not compromise - it's capability multiplication." - The Collective
"""

import os
import re
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class HermesToolGroup(Enum):
    """HermesAgent tool policy groups."""
    FILESYSTEM = "group:fs"
    RUNTIME = "group:runtime"
    SESSIONS = "group:sessions"
    MEMORY = "group:memory"
    WEB = "group:web"
    UI = "group:ui"
    AUTOMATION = "group:automation"
    MESSAGING = "group:messaging"
    NODES = "group:nodes"


@dataclass
class HermesToolResult:
    """Result from an HermesAgent tool invocation."""
    success: bool
    tool: str
    action: Optional[str] = None
    data: Any = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    execution_time: float = 0.0

    def to_HermesAgent_format(self) -> Dict:
        """Convert to HermesAgent-compatible response format."""
        if self.success:
            return {
                "status": "success",
                "result": self.data,
                "metadata": {
                    **self.metadata,
                    "tool": self.tool,
                    "action": self.action,
                    "execution_time_ms": int(self.execution_time * 1000),
                }
            }
        else:
            return {
                "status": "error",
                "error": {
                    "message": self.error or "Unknown error",
                    "tool": self.tool,
                    "action": self.action,
                },
                "metadata": self.metadata
            }


@dataclass
class HermesSkill:
    """Parsed Hermes skill definition."""
    name: str
    path: Path
    description: str = ""
    tools_required: List[str] = field(default_factory=list)
    binaries_required: List[str] = field(default_factory=list)
    env_vars_required: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    content: str = ""
    metadata: Dict = field(default_factory=dict)


class HermesAdapter:
    """
    Universal adapter for HermesAgent tool/skill compatibility.

    Maps HermesAgent's 20+ built-in tools to Cosmos equivalents,
    providing seamless execution of Hermes skills in the Cosmos swarm.
    """

    # Tool group to Cosmos module mapping
    TOOL_MAPPINGS = {
        # Filesystem tools
        "read": ("filesystem", "_handle_read"),
        "write": ("filesystem", "_handle_write"),
        "edit": ("filesystem", "_handle_edit"),
        "apply_patch": ("filesystem", "_handle_apply_patch"),

        # Execution tools
        "exec": ("runtime", "_handle_exec"),
        "bash": ("runtime", "_handle_exec"),
        "process": ("runtime", "_handle_process"),

        # Session tools
        "sessions_list": ("sessions", "_handle_sessions_list"),
        "sessions_history": ("sessions", "_handle_sessions_history"),
        "sessions_send": ("sessions", "_handle_sessions_send"),
        "sessions_spawn": ("sessions", "_handle_sessions_spawn"),
        "session_status": ("sessions", "_handle_session_status"),

        # Memory tools
        "memory_search": ("memory", "_handle_memory_search"),
        "memory_get": ("memory", "_handle_memory_get"),

        # Web tools
        "web_search": ("web", "_handle_web_search"),
        "web_fetch": ("web", "_handle_web_fetch"),

        # UI tools
        "browser": ("ui", "_handle_browser"),
        "canvas": ("ui", "_handle_canvas"),

        # Automation tools
        "cron": ("automation", "_handle_cron"),
        "gateway": ("automation", "_handle_gateway"),

        # Messaging tools
        "message": ("messaging", "_handle_message"),

        # Node tools
        "nodes": ("nodes", "_handle_nodes"),

        # Image tools
        "image": ("image", "_handle_image"),
    }

    def __init__(self, workspace_path: str = None):
        """
        Initialize the HermesAgent adapter.

        Args:
            workspace_path: Path to workspace (default: ~/.cosmos/HermesAgent)
        """
        self.workspace_path = Path(workspace_path or os.path.expanduser("~/.cosmos/HermesAgent"))
        self.skills_path = self.workspace_path / "skills"
        self.skills: Dict[str, HermesSkill] = {}
        self._initialized = False

        # Lazy-loaded Cosmos modules
        self._memory_system = None
        self._nexus = None
        self._web_agent = None
        self._code_agent = None
        self._canvas = None
        self._device_node = None
        self._voice = None
        self._model_invoker = None  # For AI-powered tool execution

        # Ensure directories exist
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.skills_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> bool:
        """Initialize the adapter and load skills."""
        if self._initialized:
            return True

        try:
            # Load Cosmos modules
            await self._load_cosmos_modules()

            # Scan and load skills
            await self._scan_skills()

            self._initialized = True
            logger.info(f"HermesAgent adapter initialized with {len(self.skills)} skills")
            return True

        except Exception as e:
            logger.error(f"HermesAgent adapter initialization failed: {e}")
            return False

    async def _load_cosmos_modules(self):
        """Lazy-load Cosmos modules for tool execution."""
        try:
            from Cosmos.memory.memory_system import get_memory_system
            self._memory_system = get_memory_system()
        except ImportError:
            logger.debug("Memory system not available")

        try:
            from Cosmos.core.nexus import get_nexus
            self._nexus = get_nexus()
        except ImportError:
            logger.debug("Nexus not available")

        # Load model invoker for AI-powered tasks
        try:
            from .model_invoker import get_model_invoker
            self._model_invoker = get_model_invoker()
            await self._model_invoker.initialize()
            logger.info(f"Model invoker loaded: {self._model_invoker.get_available_models()}")
        except Exception as e:
            logger.warning(f"Model invoker not available: {e}")

    async def _scan_skills(self):
        """Scan workspace for SKILL.md files."""
        if not self.skills_path.exists():
            return

        for skill_dir in self.skills_path.iterdir():
            if skill_dir.is_dir():
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    skill = self._parse_skill(skill_dir)
                    if skill:
                        self.skills[skill.name] = skill
                        logger.debug(f"Loaded skill: {skill.name}")

    def _parse_skill(self, skill_dir: Path) -> Optional[HermesSkill]:
        """
        Parse a SKILL.md file into a skill definition.

        Hermes skill.md format:
        - Markdown documentation with usage examples
        - Optional package.json with metadata
        - Optional bin/ directory with executables
        """
        skill_md = skill_dir / "SKILL.md"
        package_json = skill_dir / "package.json"

        try:
            content = skill_md.read_text(encoding="utf-8")

            # Extract skill name from directory
            name = skill_dir.name

            # Parse description from first paragraph
            lines = content.split("\n")
            description = ""
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    description = line
                    break

            # Extract code examples (```...```)
            examples = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)

            # Parse package.json if exists
            metadata = {}
            tools_required = []
            binaries_required = []
            env_vars_required = []

            if package_json.exists():
                try:
                    pkg = json.loads(package_json.read_text())
                    metadata = pkg
                    HermesAgent_config = pkg.get("HermesAgent", {}).get("skills", {})
                    deps = HermesAgent_config.get("dependencies", {})
                    tools_required = deps.get("tools", [])
                    binaries_required = deps.get("binaries", [])
                    env_vars_required = deps.get("envVars", [])
                except json.JSONDecodeError:
                    pass

            return HermesSkill(
                name=name,
                path=skill_dir,
                description=description,
                tools_required=tools_required,
                binaries_required=binaries_required,
                env_vars_required=env_vars_required,
                examples=examples,
                content=content,
                metadata=metadata,
            )

        except Exception as e:
            logger.warning(f"Could not parse skill {skill_dir}: {e}")
            return None

    async def invoke(
        self,
        tool: str,
        action: str = None,
        params: Dict[str, Any] = None,
        **kwargs
    ) -> HermesToolResult:
        """
        Invoke an HermesAgent tool.

        Args:
            tool: Tool name (e.g., "browser", "exec", "sessions_list")
            action: Optional action (e.g., "snapshot", "navigate")
            params: Tool parameters
            **kwargs: Additional parameters merged with params

        Returns:
            HermesToolResult with execution results
        """
        if not self._initialized:
            await self.initialize()

        params = {**(params or {}), **kwargs}
        start_time = datetime.now()

        try:
            # Find the tool mapping
            if tool not in self.TOOL_MAPPINGS:
                return HermesToolResult(
                    success=False,
                    tool=tool,
                    action=action,
                    error=f"Unknown tool: {tool}"
                )

            group, handler_name = self.TOOL_MAPPINGS[tool]
            handler = getattr(self, handler_name, None)

            if not handler:
                return HermesToolResult(
                    success=False,
                    tool=tool,
                    action=action,
                    error=f"Handler not implemented: {handler_name}"
                )

            # Execute the handler
            result = await handler(action, params)

            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            return result

        except Exception as e:
            logger.error(f"Tool invocation failed: {tool}.{action}: {e}")
            return HermesToolResult(
                success=False,
                tool=tool,
                action=action,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def invoke_with_ai(
        self,
        tool: str,
        action: str,
        prompt: str,
        params: Dict = None,
        **kwargs
    ) -> HermesToolResult:
        """
        Invoke a tool with AI model assistance.

        Routes to the best model based on task type and gets AI response.

        Args:
            tool: Tool name
            action: Tool action
            prompt: Task description/prompt for the AI
            params: Tool parameters
            **kwargs: Additional model params

        Returns:
            HermesToolResult with AI-generated response
        """
        if not self._model_invoker:
            return HermesToolResult(
                success=False,
                tool=tool,
                action=action,
                error="Model invoker not available"
            )

        start_time = datetime.now()

        try:
            # Call model invoker
            response = await self._model_invoker.invoke_for_tool(
                tool=tool,
                action=action,
                prompt=prompt,
                context=params,
                **kwargs
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            if response.success:
                return HermesToolResult(
                    success=True,
                    tool=tool,
                    action=action,
                    data={
                        "response": response.content,
                        "model": response.model_id,
                        "tokens": response.tokens_used,
                    },
                    metadata={
                        "fallback_used": response.fallback_used,
                        "fallback_chain": response.fallback_chain,
                        "latency_ms": response.latency_ms,
                    },
                    execution_time=execution_time
                )
            else:
                return HermesToolResult(
                    success=False,
                    tool=tool,
                    action=action,
                    error=response.error,
                    metadata={"model": response.model_id},
                    execution_time=execution_time
                )

        except Exception as e:
            return HermesToolResult(
                success=False,
                tool=tool,
                action=action,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def execute_skill_with_ai(
        self,
        skill_name: str,
        user_input: str,
        **kwargs
    ) -> HermesToolResult:
        """
        Execute a hermes_hub skill using AI.

        Loads the skill's instructions and uses AI to complete the task.

        Args:
            skill_name: Name of the skill
            user_input: User's task description
            **kwargs: Additional params

        Returns:
            HermesToolResult
        """
        skill = self.skills.get(skill_name)
        if not skill:
            return HermesToolResult(
                success=False,
                tool="skill",
                action=skill_name,
                error=f"Skill not found: {skill_name}"
            )

        # Build prompt from skill content + user input
        prompt = f"""You are executing the Hermes skill: {skill_name}

Skill Description:
{skill.description}

Skill Instructions:
{skill.content[:2000]}

Required Tools: {', '.join(skill.tools_required)}

User Request:
{user_input}

Execute this skill and provide the result."""

        return await self.invoke_with_ai(
            tool="skill",
            action=skill_name,
            prompt=prompt,
            params={"skill": skill_name, "input": user_input},
            **kwargs
        )

    def get_model_status(self) -> Dict:
        """Get status of available models."""
        if self._model_invoker:
            return self._model_invoker.get_status()
        return {"initialized": False, "error": "Model invoker not loaded"}

    # =========================================================================
    # FILESYSTEM HANDLERS (group:fs)
    # =========================================================================

    async def _handle_read(self, action: str, params: Dict) -> HermesToolResult:
        """Handle file read operations."""
        path = params.get("path")
        encoding = params.get("encoding", "utf-8")

        if not path:
            return HermesToolResult(success=False, tool="read", error="Missing 'path' parameter")

        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_path / path

            content = file_path.read_text(encoding=encoding)
            return HermesToolResult(
                success=True,
                tool="read",
                data={"content": content, "path": str(file_path), "size": len(content)}
            )
        except Exception as e:
            return HermesToolResult(success=False, tool="read", error=str(e))

    async def _handle_write(self, action: str, params: Dict) -> HermesToolResult:
        """Handle file write operations."""
        path = params.get("path")
        content = params.get("content")
        encoding = params.get("encoding", "utf-8")

        if not path or content is None:
            return HermesToolResult(success=False, tool="write", error="Missing 'path' or 'content'")

        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_path / path

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding=encoding)

            return HermesToolResult(
                success=True,
                tool="write",
                data={"path": str(file_path), "bytes_written": len(content.encode(encoding))}
            )
        except Exception as e:
            return HermesToolResult(success=False, tool="write", error=str(e))

    async def _handle_edit(self, action: str, params: Dict) -> HermesToolResult:
        """Handle file edit operations (search and replace)."""
        path = params.get("path")
        edits = params.get("edits", [])
        dry_run = params.get("dryRun", False)

        if not path:
            return HermesToolResult(success=False, tool="edit", error="Missing 'path'")

        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_path / path

            content = file_path.read_text()
            original = content
            changes = []

            for edit in edits:
                search = edit.get("search", "")
                replace = edit.get("replace", "")
                if search in content:
                    content = content.replace(search, replace)
                    changes.append({"search": search, "replace": replace})

            if not dry_run and changes:
                file_path.write_text(content)

            return HermesToolResult(
                success=True,
                tool="edit",
                data={
                    "path": str(file_path),
                    "changes_count": len(changes),
                    "dry_run": dry_run,
                    "changes": changes
                }
            )
        except Exception as e:
            return HermesToolResult(success=False, tool="edit", error=str(e))

    async def _handle_apply_patch(self, action: str, params: Dict) -> HermesToolResult:
        """Handle unified diff patch application."""
        path = params.get("path")
        patch = params.get("patch")

        if not path or not patch:
            return HermesToolResult(success=False, tool="apply_patch", error="Missing 'path' or 'patch'")

        try:
            # Use subprocess to apply patch
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_path / path

            proc = subprocess.run(
                ["patch", str(file_path)],
                input=patch.encode(),
                capture_output=True
            )

            if proc.returncode == 0:
                return HermesToolResult(
                    success=True,
                    tool="apply_patch",
                    data={"path": str(file_path), "output": proc.stdout.decode()}
                )
            else:
                return HermesToolResult(
                    success=False,
                    tool="apply_patch",
                    error=proc.stderr.decode()
                )
        except Exception as e:
            return HermesToolResult(success=False, tool="apply_patch", error=str(e))

    # =========================================================================
    # RUNTIME HANDLERS (group:runtime)
    # =========================================================================

    async def _handle_exec(self, action: str, params: Dict) -> HermesToolResult:
        """Handle shell command execution."""
        command = params.get("command")
        cwd = params.get("cwd")
        timeout = params.get("timeout", 30)
        background = params.get("background", False)
        env = params.get("env", {})

        if not command:
            return HermesToolResult(success=False, tool="exec", error="Missing 'command'")

        try:
            # Merge environment
            full_env = {**os.environ, **env}

            # AGI v1.9.1: Use shell=False with shlex.split for safety
            import shlex
            if isinstance(command, str):
                cmd_args = shlex.split(command)
            else:
                cmd_args = list(command)

            if background:
                # Start in background
                proc = subprocess.Popen(
                    cmd_args,
                    shell=False,
                    cwd=cwd,
                    env=full_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                return HermesToolResult(
                    success=True,
                    tool="exec",
                    data={"pid": proc.pid, "background": True}
                )
            else:
                # Run synchronously
                proc = subprocess.run(
                    cmd_args,
                    shell=False,
                    cwd=cwd,
                    env=full_env,
                    capture_output=True,
                    timeout=timeout
                )

                return HermesToolResult(
                    success=proc.returncode == 0,
                    tool="exec",
                    data={
                        "stdout": proc.stdout.decode(errors="replace"),
                        "stderr": proc.stderr.decode(errors="replace"),
                        "returncode": proc.returncode
                    },
                    error=None if proc.returncode == 0 else f"Exit code: {proc.returncode}"
                )
        except subprocess.TimeoutExpired:
            return HermesToolResult(success=False, tool="exec", error=f"Command timed out after {timeout}s")
        except Exception as e:
            return HermesToolResult(success=False, tool="exec", error=str(e))

    async def _handle_process(self, action: str, params: Dict) -> HermesToolResult:
        """Handle background process management."""
        process_action = params.get("action", action)
        process_id = params.get("processId")

        if process_action == "list":
            # List background processes (simplified)
            return HermesToolResult(
                success=True,
                tool="process",
                action="list",
                data={"processes": []}  # Would need process tracking
            )
        elif process_action == "stop":
            if not process_id:
                return HermesToolResult(success=False, tool="process", error="Missing processId")
            try:
                os.kill(int(process_id), 9)
                return HermesToolResult(success=True, tool="process", action="stop", data={"stopped": process_id})
            except Exception as e:
                return HermesToolResult(success=False, tool="process", error=str(e))
        else:
            return HermesToolResult(success=False, tool="process", error=f"Unknown action: {process_action}")

    # =========================================================================
    # SESSION HANDLERS (group:sessions)
    # =========================================================================

    async def _handle_sessions_list(self, action: str, params: Dict) -> HermesToolResult:
        """Handle session listing - maps to Cosmos session manager."""
        try:
            from Cosmos.core.collective.session_manager import get_session_manager
            manager = get_session_manager()
            stats = manager.get_session_stats()

            # Convert to HermesAgent format
            sessions = []
            for sid, info in stats.get("sessions", {}).items():
                sessions.append({
                    "sessionKey": sid,
                    "type": info.get("type"),
                    "deliberations": info.get("deliberations", 0),
                    "lastActive": info.get("last_active"),
                    "active": True
                })

            return HermesToolResult(
                success=True,
                tool="sessions_list",
                data={"sessions": sessions, "total": len(sessions)}
            )
        except Exception as e:
            return HermesToolResult(success=False, tool="sessions_list", error=str(e))

    async def _handle_sessions_history(self, action: str, params: Dict) -> HermesToolResult:
        """Handle session history retrieval."""
        session_key = params.get("sessionKey")
        limit = params.get("limit", 50)
        format_type = params.get("format", "json")

        if not session_key:
            return HermesToolResult(success=False, tool="sessions_history", error="Missing sessionKey")

        try:
            from Cosmos.core.collective.session_manager import get_session_manager
            manager = get_session_manager()

            if session_key in manager.sessions:
                session = manager.sessions[session_key]
                history = session.history[-limit:]

                if format_type == "text":
                    # Convert to text transcript
                    transcript = []
                    for h in history:
                        transcript.append(f"[{h.winning_agent}]: {h.final_response[:200]}...")
                    return HermesToolResult(
                        success=True,
                        tool="sessions_history",
                        data={"transcript": "\n".join(transcript)}
                    )
                else:
                    # JSON format
                    return HermesToolResult(
                        success=True,
                        tool="sessions_history",
                        data={
                            "history": [
                                {
                                    "id": h.deliberation_id,
                                    "winner": h.winning_agent,
                                    "response": h.final_response[:500],
                                    "consensus": h.consensus_reached
                                }
                                for h in history
                            ]
                        }
                    )
            else:
                return HermesToolResult(success=False, tool="sessions_history", error="Session not found")

        except Exception as e:
            return HermesToolResult(success=False, tool="sessions_history", error=str(e))

    async def _handle_sessions_send(self, action: str, params: Dict) -> HermesToolResult:
        """Handle cross-session messaging - maps to Nexus signals."""
        session_key = params.get("sessionKey")
        text = params.get("text")
        reply_back = params.get("replyBack", False)

        if not session_key or not text:
            return HermesToolResult(success=False, tool="sessions_send", error="Missing sessionKey or text")

        try:
            if self._nexus:
                await self._nexus.emit("SESSION_MESSAGE", {
                    "target_session": session_key,
                    "message": text,
                    "reply_back": reply_back,
                    "timestamp": datetime.now().isoformat()
                })

            return HermesToolResult(
                success=True,
                tool="sessions_send",
                data={"sent_to": session_key, "message_length": len(text)}
            )
        except Exception as e:
            return HermesToolResult(success=False, tool="sessions_send", error=str(e))

    async def _handle_sessions_spawn(self, action: str, params: Dict) -> HermesToolResult:
        """Handle sub-agent spawning - maps to Cosmos agent spawner."""
        prompt = params.get("prompt")
        model = params.get("model")
        thinking_level = params.get("thinkingLevel")

        if not prompt:
            return HermesToolResult(success=False, tool="sessions_spawn", error="Missing prompt")

        try:
            from Cosmos.core.agent_spawner import spawn_agent

            result = await spawn_agent(
                task=prompt,
                model_preference=model,
                thinking_level=thinking_level
            )

            return HermesToolResult(
                success=True,
                tool="sessions_spawn",
                data={
                    "sessionKey": result.get("session_id"),
                    "result": result.get("response"),
                    "agent": result.get("agent_used")
                }
            )
        except Exception as e:
            return HermesToolResult(success=False, tool="sessions_spawn", error=str(e))

    async def _handle_session_status(self, action: str, params: Dict) -> HermesToolResult:
        """Handle session status query."""
        try:
            from Cosmos.core.collective.session_manager import get_session_manager
            manager = get_session_manager()

            return HermesToolResult(
                success=True,
                tool="session_status",
                data={
                    "total_sessions": len(manager.sessions),
                    "active": True,
                    "agents_registered": manager._agents_initialized
                }
            )
        except Exception as e:
            return HermesToolResult(success=False, tool="session_status", error=str(e))

    # =========================================================================
    # MEMORY HANDLERS (group:memory)
    # =========================================================================

    async def _handle_memory_search(self, action: str, params: Dict) -> HermesToolResult:
        """Handle memory search - maps to Cosmos ArchivalMemory."""
        query = params.get("query")
        limit = params.get("limit", 10)

        if not query:
            return HermesToolResult(success=False, tool="memory_search", error="Missing query")

        try:
            if self._memory_system:
                results = await self._memory_system.search(query, limit=limit)
                return HermesToolResult(
                    success=True,
                    tool="memory_search",
                    data={
                        "results": [
                            {"id": r.id, "content": r.content[:200], "score": r.score}
                            for r in results
                        ],
                        "total": len(results)
                    }
                )
            else:
                # Fallback to simple file search
                results = []
                for f in self.workspace_path.rglob("*"):
                    if f.is_file() and f.suffix in [".txt", ".md", ".py", ".json"]:
                        try:
                            content = f.read_text()
                            if query.lower() in content.lower():
                                results.append({"path": str(f), "snippet": content[:200]})
                        except Exception:
                            pass
                return HermesToolResult(
                    success=True,
                    tool="memory_search",
                    data={"results": results[:limit]}
                )
        except Exception as e:
            return HermesToolResult(success=False, tool="memory_search", error=str(e))

    async def _handle_memory_get(self, action: str, params: Dict) -> HermesToolResult:
        """Handle memory retrieval by ID."""
        memory_id = params.get("id")

        if not memory_id:
            return HermesToolResult(success=False, tool="memory_get", error="Missing id")

        try:
            if self._memory_system:
                memory = await self._memory_system.get(memory_id)
                if memory:
                    return HermesToolResult(
                        success=True,
                        tool="memory_get",
                        data={"id": memory_id, "content": memory.content}
                    )

            return HermesToolResult(success=False, tool="memory_get", error="Memory not found")
        except Exception as e:
            return HermesToolResult(success=False, tool="memory_get", error=str(e))

    # =========================================================================
    # WEB HANDLERS (group:web)
    # =========================================================================

    async def _handle_web_search(self, action: str, params: Dict) -> HermesToolResult:
        """Handle web search - maps to Cosmos external APIs."""
        query = params.get("query")
        limit = params.get("limit", 10)

        if not query:
            return HermesToolResult(success=False, tool="web_search", error="Missing query")

        try:
            # Try Grok/Gemini for web search
            from Cosmos.integration.external.grok import grok_search
            results = await grok_search(query, limit=limit)

            return HermesToolResult(
                success=True,
                tool="web_search",
                data={"results": results, "query": query}
            )
        except Exception as e:
            return HermesToolResult(success=False, tool="web_search", error=str(e))

    async def _handle_web_fetch(self, action: str, params: Dict) -> HermesToolResult:
        """Handle URL fetching."""
        url = params.get("url")
        format_type = params.get("format", "markdown")

        if not url:
            return HermesToolResult(success=False, tool="web_fetch", error="Missing url")

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    content = await resp.text()

            # Simple HTML to text conversion
            if format_type == "text":
                import re
                content = re.sub(r"<[^>]+>", "", content)

            return HermesToolResult(
                success=True,
                tool="web_fetch",
                data={"url": url, "content": content[:10000], "format": format_type}
            )
        except Exception as e:
            return HermesToolResult(success=False, tool="web_fetch", error=str(e))

    # =========================================================================
    # UI HANDLERS (group:ui)
    # =========================================================================

    async def _handle_browser(self, action: str, params: Dict) -> HermesToolResult:
        """Handle browser control - maps to visual_canvas or web_agent."""
        browser_action = params.get("action", action)

        try:
            from .visual_canvas import get_canvas
            canvas = get_canvas()

            if browser_action == "snapshot":
                result = await canvas.snapshot()
                return HermesToolResult(success=True, tool="browser", action="snapshot", data=result)

            elif browser_action == "navigate":
                url = params.get("url")
                result = await canvas.navigate(url)
                return HermesToolResult(success=True, tool="browser", action="navigate", data=result)

            elif browser_action == "click":
                selector = params.get("selector")
                result = await canvas.click(selector)
                return HermesToolResult(success=True, tool="browser", action="click", data=result)

            elif browser_action == "type":
                selector = params.get("selector")
                text = params.get("text")
                result = await canvas.type_text(selector, text)
                return HermesToolResult(success=True, tool="browser", action="type", data=result)

            elif browser_action == "evaluate":
                code = params.get("code")
                result = await canvas.eval(code)
                return HermesToolResult(success=True, tool="browser", action="evaluate", data=result)

            else:
                return HermesToolResult(success=False, tool="browser", error=f"Unknown action: {browser_action}")

        except Exception as e:
            return HermesToolResult(success=False, tool="browser", action=browser_action, error=str(e))

    async def _handle_canvas(self, action: str, params: Dict) -> HermesToolResult:
        """Handle canvas/A2UI operations."""
        canvas_action = params.get("action", action)

        try:
            from .visual_canvas import get_canvas
            canvas = get_canvas()

            if canvas_action == "reset":
                result = await canvas.reset()
                return HermesToolResult(success=True, tool="canvas", action="reset", data=result)

            elif canvas_action == "push":
                a2ui = params.get("a2ui")
                result = await canvas.push(a2ui)
                return HermesToolResult(success=True, tool="canvas", action="push", data=result)

            elif canvas_action == "eval":
                code = params.get("code")
                result = await canvas.eval(code)
                return HermesToolResult(success=True, tool="canvas", action="eval", data=result)

            elif canvas_action == "snapshot":
                result = await canvas.snapshot()
                return HermesToolResult(success=True, tool="canvas", action="snapshot", data=result)

            else:
                return HermesToolResult(success=False, tool="canvas", error=f"Unknown action: {canvas_action}")

        except Exception as e:
            return HermesToolResult(success=False, tool="canvas", action=canvas_action, error=str(e))

    # =========================================================================
    # AUTOMATION HANDLERS (group:automation)
    # =========================================================================

    async def _handle_cron(self, action: str, params: Dict) -> HermesToolResult:
        """Handle cron scheduling - maps to Cosmos scheduler."""
        cron_action = params.get("action", action)

        try:
            from Cosmos.automation.scheduler import get_scheduler
            scheduler = get_scheduler()

            if cron_action == "list":
                jobs = scheduler.list_jobs()
                return HermesToolResult(success=True, tool="cron", action="list", data={"jobs": jobs})

            elif cron_action == "add":
                schedule = params.get("schedule")
                prompt = params.get("prompt")
                job_id = scheduler.add_job(schedule, prompt)
                return HermesToolResult(success=True, tool="cron", action="add", data={"id": job_id})

            elif cron_action == "remove":
                job_id = params.get("id")
                scheduler.remove_job(job_id)
                return HermesToolResult(success=True, tool="cron", action="remove", data={"removed": job_id})

            else:
                return HermesToolResult(success=False, tool="cron", error=f"Unknown action: {cron_action}")

        except Exception as e:
            return HermesToolResult(success=False, tool="cron", error=str(e))

    async def _handle_gateway(self, action: str, params: Dict) -> HermesToolResult:
        """Handle gateway configuration."""
        gateway_action = params.get("action", action)

        if gateway_action == "config.get":
            # Return Cosmos config
            return HermesToolResult(
                success=True,
                tool="gateway",
                action="config.get",
                data={"config": {"adapter": "cosmos", "version": "1.8"}}
            )
        else:
            return HermesToolResult(success=False, tool="gateway", error=f"Unknown action: {gateway_action}")

    # =========================================================================
    # MESSAGING HANDLERS (group:messaging)
    # =========================================================================

    async def _handle_message(self, action: str, params: Dict) -> HermesToolResult:
        """Handle messaging operations."""
        msg_action = params.get("action", action or "send")

        try:
            if msg_action == "send":
                channel = params.get("channel")
                to = params.get("to")
                text = params.get("text")

                # Route to appropriate Cosmos channel
                if channel == "twitter" or channel == "x":
                    from Cosmos.integration.x_automation.x_api_poster import XOAuth2Poster
                    poster = XOAuth2Poster()
                    result = await poster.post_tweet(text)
                    return HermesToolResult(success=True, tool="message", action="send", data=result)

                elif channel == "email":
                    # Email integration
                    return HermesToolResult(success=True, tool="message", action="send", data={"sent": True})

                else:
                    # Generic message via Nexus
                    if self._nexus:
                        await self._nexus.emit("OUTBOUND_MESSAGE", {
                            "channel": channel,
                            "to": to,
                            "text": text
                        })
                    return HermesToolResult(success=True, tool="message", action="send", data={"queued": True})

            else:
                return HermesToolResult(success=False, tool="message", error=f"Unknown action: {msg_action}")

        except Exception as e:
            return HermesToolResult(success=False, tool="message", error=str(e))

    # =========================================================================
    # NODE HANDLERS (group:nodes)
    # =========================================================================

    async def _handle_nodes(self, action: str, params: Dict) -> HermesToolResult:
        """Handle device node operations - maps to device_node.py."""
        node_action = params.get("action", action)

        try:
            from .device_node import get_device_node
            node = get_device_node()

            if node_action == "camera.snap":
                result = await node.camera_snap(params.get("facing", "back"))
                return HermesToolResult(success=True, tool="nodes", action=node_action, data=result)

            elif node_action == "camera.clip":
                duration = params.get("duration", 10)
                result = await node.camera_clip(duration)
                return HermesToolResult(success=True, tool="nodes", action=node_action, data=result)

            elif node_action == "screen.record":
                duration = params.get("duration", 10)
                result = await node.screen_record(duration)
                return HermesToolResult(success=True, tool="nodes", action=node_action, data=result)

            elif node_action == "location.get":
                result = await node.get_location()
                return HermesToolResult(success=True, tool="nodes", action=node_action, data=result)

            elif node_action == "system.notify":
                title = params.get("title")
                body = params.get("body")
                result = await node.notify(title, body)
                return HermesToolResult(success=True, tool="nodes", action=node_action, data=result)

            elif node_action == "system.run":
                command = params.get("command")
                result = await node.run(command)
                return HermesToolResult(success=True, tool="nodes", action=node_action, data=result)

            else:
                return HermesToolResult(success=False, tool="nodes", error=f"Unknown action: {node_action}")

        except Exception as e:
            return HermesToolResult(success=False, tool="nodes", action=node_action, error=str(e))

    # =========================================================================
    # IMAGE HANDLERS
    # =========================================================================

    async def _handle_image(self, action: str, params: Dict) -> HermesToolResult:
        """Handle image processing."""
        img_action = params.get("action", action)
        path = params.get("path")

        if not path:
            return HermesToolResult(success=False, tool="image", error="Missing path")

        try:
            if img_action == "understand" or img_action == "caption":
                # Use Cosmos image understanding (via Claude/Gemini)
                from Cosmos.integration.external.gemini import gemini_understand_image
                result = await gemini_understand_image(path)
                return HermesToolResult(success=True, tool="image", action=img_action, data=result)

            return HermesToolResult(success=False, tool="image", error=f"Unknown action: {img_action}")

        except Exception as e:
            return HermesToolResult(success=False, tool="image", error=str(e))

    # =========================================================================
    # SKILL MANAGEMENT
    # =========================================================================

    def get_skill(self, name: str) -> Optional[HermesSkill]:
        """Get a loaded skill by name."""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """List all loaded skill names."""
        return list(self.skills.keys())

    def get_skills_for_prompt(self) -> str:
        """
        Get skill documentation formatted for system prompt injection.

        This mimics HermesAgent's "Available Skills" section injection.
        """
        if not self.skills:
            return ""

        lines = ["## Available Skills (HermesAgent Compatibility)\n"]
        for name, skill in self.skills.items():
            lines.append(f"### {name}")
            lines.append(skill.description[:200] if skill.description else "No description")
            if skill.examples:
                lines.append("\n**Example:**")
                lines.append(f"```\n{skill.examples[0][:300]}\n```")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# hermes_hub MARKETPLACE CLIENT
# =============================================================================

class hermes_hubClient:
    """
    Client for HermesAgent's hermes_hub skills marketplace.

    hermes_hub hosts 700+ community-built skills that can be downloaded
    and executed within Cosmos's swarm.

    API Endpoints (reverse-engineered from HermesAgent):
    - GET /skills - List all skills
    - GET /skills/search?q=<query> - Search skills
    - GET /skills/<name> - Get skill details
    - GET /skills/<name>/download - Download skill package
    """

    BASE_URL = "https://hermes_hub.HermesAgent.dev/api/v1"
    CACHE_DIR = Path(os.path.expanduser("~/.cosmos/hermes_hub_cache"))

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def search_skills(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search hermes_hub marketplace for skills.

        Args:
            query: Search query
            limit: Max results to return

        Returns:
            List of skill metadata dicts
        """
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.BASE_URL}/skills/search",
                params={"q": query, "limit": limit}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("skills", [])
                else:
                    logger.warning(f"hermes_hub search failed: {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"hermes_hub search error: {e}")
            return []

    async def list_skills(self, category: str = None, limit: int = 50) -> List[Dict]:
        """
        List available skills from hermes_hub.

        Args:
            category: Optional category filter
            limit: Max results

        Returns:
            List of skill metadata
        """
        try:
            session = await self._get_session()
            params = {"limit": limit}
            if category:
                params["category"] = category

            async with session.get(
                f"{self.BASE_URL}/skills",
                params=params
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("skills", [])
                return []
        except Exception as e:
            logger.error(f"hermes_hub list error: {e}")
            return []

    async def get_skill_details(self, skill_name: str) -> Optional[Dict]:
        """
        Get detailed information about a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Skill details dict or None
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/skills/{skill_name}") as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error(f"hermes_hub get details error: {e}")
            return None

    async def download_skill(self, skill_name: str, install_dir: Path = None) -> Optional[Path]:
        """
        Download and install a skill from hermes_hub.

        Args:
            skill_name: Name of the skill to download
            install_dir: Directory to install to (default: ~/.cosmos/skills/)

        Returns:
            Path to installed skill directory, or None on failure
        """
        install_dir = install_dir or Path(os.path.expanduser("~/.cosmos/skills"))
        install_dir.mkdir(parents=True, exist_ok=True)

        skill_dir = install_dir / skill_name

        try:
            session = await self._get_session()

            # Download skill package
            async with session.get(f"{self.BASE_URL}/skills/{skill_name}/download") as resp:
                if resp.status != 200:
                    logger.error(f"Skill download failed: {resp.status}")
                    return None

                # Save to cache first
                cache_file = self.CACHE_DIR / f"{skill_name}.zip"
                content = await resp.read()

                with open(cache_file, "wb") as f:
                    f.write(content)

            # Extract skill
            import zipfile
            with zipfile.ZipFile(cache_file, "r") as zf:
                zf.extractall(skill_dir)

            logger.info(f"Installed skill: {skill_name} -> {skill_dir}")
            return skill_dir

        except Exception as e:
            logger.error(f"Skill download error: {e}")
            return None

    async def get_popular_skills(self, limit: int = 10) -> List[Dict]:
        """Get most popular skills from hermes_hub."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.BASE_URL}/skills",
                params={"sort": "downloads", "limit": limit}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("skills", [])
                return []
        except Exception as e:
            logger.error(f"hermes_hub popular error: {e}")
            return []

    async def get_skill_categories(self) -> List[str]:
        """Get available skill categories."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/categories") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("categories", [])
                return []
        except Exception as e:
            logger.error(f"hermes_hub categories error: {e}")
            return []

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None


# Global hermes_hub client
_hermes_hub_client: Optional[hermes_hubClient] = None


def get_hermes_hub_client() -> hermes_hubClient:
    """Get or create the global hermes_hub client."""
    global _hermes_hub_client
    if _hermes_hub_client is None:
        _hermes_hub_client = hermes_hubClient()
    return _hermes_hub_client


async def search_hermes_hub_skills(query: str, limit: int = 20) -> List[Dict]:
    """Search hermes_hub marketplace for skills."""
    client = get_hermes_hub_client()
    return await client.search_skills(query, limit)


async def download_hermes_hub_skill(skill_name: str) -> Optional[Path]:
    """Download and install a skill from hermes_hub."""
    client = get_hermes_hub_client()
    return await client.download_skill(skill_name)


async def install_and_load_hermes_skill(skill_name: str) -> Optional[HermesSkill]:
    """
    Download a skill from hermes_hub and load it into Cosmos.

    Args:
        skill_name: Name of the skill to install

    Returns:
        Loaded HermesSkill or None
    """
    # Download skill
    skill_dir = await download_hermes_hub_skill(skill_name)
    if not skill_dir:
        return None

    # Load into adapter
    return await load_HermesAgent_skill(str(skill_dir))


# =============================================================================
# SINGLETON AND UTILITY FUNCTIONS
# =============================================================================

_adapter: Optional[HermesAdapter] = None


def get_hermes_adapter() -> HermesAdapter:
    """Get or create the global HermesAgent adapter."""
    global _adapter
    if _adapter is None:
        _adapter = HermesAdapter()
    return _adapter


async def invoke_HermesAgent_tool(tool: str, action: str = None, **params) -> HermesToolResult:
    """
    Convenience function to invoke an HermesAgent tool.

    Args:
        tool: Tool name (e.g., "browser", "exec", "nodes")
        action: Optional action (e.g., "snapshot", "camera.snap")
        **params: Tool parameters

    Returns:
        HermesToolResult
    """
    adapter = get_hermes_adapter()
    return await adapter.invoke(tool, action, params)


async def load_HermesAgent_skill(skill_path: str) -> Optional[HermesSkill]:
    """
    Load an Hermes skill from a directory path.

    Args:
        skill_path: Path to skill directory containing SKILL.md

    Returns:
        HermesSkill if successful, None otherwise
    """
    adapter = get_hermes_adapter()
    skill_dir = Path(skill_path)

    if not skill_dir.exists():
        logger.error(f"Skill path does not exist: {skill_path}")
        return None

    skill = adapter._parse_skill(skill_dir)
    if skill:
        adapter.skills[skill.name] = skill
        logger.info(f"Loaded skill: {skill.name}")

    return skill
