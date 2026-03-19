"""
OpenCode Worker Integration
Spawns OpenCode CLI instances to work on development tasks
https://opencode.ai - Open source AI coding agent
"""
import asyncio
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# OpenCode configuration
OPENCODE_CONFIG = {
    "provider": "ollama",  # Use local Ollama by default
    "model": "deepseek-r1:1.5b",  # Use available model
    "max_tokens": 4096,
    "timeout": 300,  # 5 minute timeout per task
}


class OpenCodeWorker:
    """
    Wrapper for OpenCode CLI to work on development tasks.
    OpenCode is an open-source AI coding agent that supports 75+ LLM providers.
    """

    def __init__(self, workspace: str = "/workspace/Cosmos"):
        self.workspace = Path(workspace)
        self.staging_dir = self.workspace / "cosmos" / "staging" / "opencode"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.active_processes: Dict[str, subprocess.Popen] = {}

    async def is_installed(self) -> bool:
        """Check if OpenCode is installed"""
        try:
            result = await asyncio.create_subprocess_exec(
                "opencode", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            return result.returncode == 0
        except FileNotFoundError:
            return False

    async def install(self) -> bool:
        """Install OpenCode via npm"""
        logger.info("Installing OpenCode...")
        try:
            result = await asyncio.create_subprocess_exec(
                "npm", "install", "-g", "opencode-ai@latest",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            if result.returncode == 0:
                logger.info("OpenCode installed successfully")
                return True
            else:
                logger.error(f"OpenCode installation failed: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Failed to install OpenCode: {e}")
            return False

    async def execute_task(self, task_description: str, task_id: str,
                           files: list = None) -> Optional[str]:
        """
        Execute a development task using OpenCode.

        Args:
            task_description: What to build/fix
            task_id: Unique task identifier
            files: Optional list of files to focus on

        Returns:
            Generated code/result or None if failed
        """
        if not await self.is_installed():
            logger.warning("OpenCode not installed, attempting installation...")
            if not await self.install():
                return None

        # Build the prompt for OpenCode
        prompt = f"""You are a Python developer working on the Cosmos AI Swarm project.

TASK: {task_description}

REQUIREMENTS:
1. Write production-ready Python code
2. Include proper docstrings and type hints
3. Follow existing code patterns in the project
4. Output ONLY the code, no explanations

{f'Focus on these files: {", ".join(files)}' if files else ''}

Start coding:"""

        output_file = self.staging_dir / f"{task_id}_output.py"

        try:
            # Run OpenCode using 'opencode run' command (correct CLI syntax)
            # This runs non-interactively with the message as argument
            process = await asyncio.create_subprocess_exec(
                "opencode", "run", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
                env={**os.environ, "NO_COLOR": "1"}  # Disable color output for parsing
            )

            self.active_processes[task_id] = process

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=OPENCODE_CONFIG["timeout"]
                )
            except asyncio.TimeoutError:
                process.kill()
                logger.error(f"OpenCode task {task_id} timed out")
                return None
            finally:
                if task_id in self.active_processes:
                    del self.active_processes[task_id]

            result = stdout.decode().strip()
            stderr_out = stderr.decode().strip()

            # OpenCode run may return 0 even with partial results
            if result:
                # Extract code from response
                code = self._extract_code(result)
                if code:
                    output_file.write_text(code)
                    logger.info(f"OpenCode completed task {task_id}, saved to {output_file}")
                    return code
                # If no code block found, save raw result
                output_file.write_text(result)
                logger.info(f"OpenCode completed task {task_id} (raw output)")
                return result
            elif stderr_out:
                logger.error(f"OpenCode error: {stderr_out}")
                return None
            else:
                logger.warning(f"OpenCode returned empty result for task {task_id}")
                return None

        except Exception as e:
            logger.error(f"OpenCode execution failed: {e}")
            return None

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from OpenCode response"""
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        elif "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        return response

    async def run_build_agent(self, task: str) -> Optional[str]:
        """
        Run OpenCode's built-in 'build' agent for a task.
        The build agent has full access for development work.
        """
        try:
            # Use OpenCode's agent subcommand if available
            process = await asyncio.create_subprocess_exec(
                "opencode", "agent", "build",
                "-m", task,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=OPENCODE_CONFIG["timeout"]
            )

            if process.returncode == 0:
                return stdout.decode()
            return None
        except Exception as e:
            logger.error(f"OpenCode build agent failed: {e}")
            return None

    async def run_plan_agent(self, task: str) -> Optional[str]:
        """
        Run OpenCode's built-in 'plan' agent for analysis.
        The plan agent is read-only for code exploration.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "opencode", "agent", "plan",
                "-m", task,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=OPENCODE_CONFIG["timeout"]
            )

            if process.returncode == 0:
                return stdout.decode()
            return None
        except Exception as e:
            logger.error(f"OpenCode plan agent failed: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get status of active OpenCode processes"""
        return {
            "active_tasks": len(self.active_processes),
            "task_ids": list(self.active_processes.keys()),
            "staging_dir": str(self.staging_dir),
            "config": OPENCODE_CONFIG
        }


# Global instance
_opencode_worker: Optional[OpenCodeWorker] = None


def get_opencode_worker() -> OpenCodeWorker:
    global _opencode_worker
    if _opencode_worker is None:
        _opencode_worker = OpenCodeWorker()
    return _opencode_worker


async def spawn_opencode_task(task_desc: str, task_id: str) -> Optional[str]:
    """Convenience function to spawn an OpenCode task"""
    worker = get_opencode_worker()
    return await worker.execute_task(task_desc, task_id)
