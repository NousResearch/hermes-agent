#!/usr/bin/env python3
"""
Mini-SWE-Agent Runner with Hermes Trajectory Format

This module provides a runner that uses mini-swe-agent's execution environments
(local, docker, modal) but outputs trajectories in the Hermes-Agent format
compatible with batch_runner.py and trajectory_compressor.py.

Features:
- Uses mini-swe-agent's Docker, Modal, or Local environments for command execution
- Outputs trajectories in Hermes format (from/value pairs with <tool_call>/<tool_response> XML)
- Compatible with the trajectory compression pipeline
- Supports batch processing from JSONL prompt files

Usage:
    # Run a single task with local environment
    python mini_swe_runner.py --task "Create a hello world Python script" --env local

    # Run with Docker
    python mini_swe_runner.py --task "List files in /tmp" --env docker --image python:3.11-slim

    # Run with Modal (cloud)
    python mini_swe_runner.py --task "Install numpy and test it" --env modal --image python:3.11-slim

    # Batch mode from JSONL file
    python mini_swe_runner.py --prompts_file prompts.jsonl --output_file trajectories.jsonl --env docker
"""

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import fire
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add mini-swe-agent to path if not installed
mini_swe_path = Path(__file__).parent / "mini-swe-agent" / "src"
if mini_swe_path.exists():
    sys.path.insert(0, str(mini_swe_path))


# ============================================================================
# Terminal Tool Definition (matches Hermes-Agent format)
# ============================================================================

TERMINAL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": """Execute bash commands in a sandboxed environment.

**Environment:**
- Isolated execution environment (local, Docker, or Modal cloud)
- Filesystem persists between tool calls within the same task
- Internet access available

**Command Execution:**
- Provide the command to execute via the 'command' parameter
- Optional 'timeout' parameter in seconds (default: 60)

**Examples:**
- Run command: `{"command": "ls -la"}`
- With timeout: `{"command": "long_task.sh", "timeout": 300}`

**Best Practices:**
- Use non-interactive commands (avoid vim, nano, interactive python)
- Pipe to cat if output might be large
- Install tools with apt-get or pip as needed

**Completion:**
- When task is complete, output: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by your result
""",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
                "timeout": {"type": "integer", "description": "Command timeout in seconds (default: 60)"},
            },
            "required": ["command"],
        },
    },
}


# ============================================================================
# Environment Factory
# ============================================================================


def create_environment(
    env_type: str = "local", image: str = "python:3.11-slim", cwd: str = "/tmp", timeout: int = 60, **kwargs
):
    """
    Create an execution environment from mini-swe-agent.

    Args:
        env_type: One of "local", "docker", "modal"
        image: Docker/Modal image name (ignored for local)
        cwd: Working directory
        timeout: Default command timeout
        **kwargs: Additional environment-specific options

    Returns:
        Environment instance with execute() method
    """
    if env_type == "local":
        from minisweagent.environments.local import LocalEnvironment

        return LocalEnvironment(cwd=cwd, timeout=timeout)

    elif env_type == "docker":
        from minisweagent.environments.docker import DockerEnvironment

        return DockerEnvironment(image=image, cwd=cwd, timeout=timeout, **kwargs)

    elif env_type == "modal":
        from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment

        return SwerexModalEnvironment(image=image, cwd=cwd, timeout=timeout, **kwargs)

    else:
        raise ValueError(f"Unknown environment type: {env_type}. Use 'local', 'docker', or 'modal'")


# ============================================================================
# Mini-SWE Runner with Hermes Trajectory Format
# ============================================================================


class MiniSWERunner:
    """
    Agent runner that uses mini-swe-agent environments but outputs
    trajectories in Hermes-Agent format.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        base_url: str = None,
        api_key: str = None,
        env_type: str = "local",
        image: str = "python:3.11-slim",
        cwd: str = "/tmp",
        max_iterations: int = 15,
        command_timeout: int = 60,
        verbose: bool = False,
    ):
        """
        Initialize the Mini-SWE Runner.

        Args:
            model: Model name for OpenAI-compatible API
            base_url: API base URL (optional, uses env vars if not provided)
            api_key: API key (optional, uses env vars if not provided)
            env_type: Environment type - "local", "docker", or "modal"
            image: Docker/Modal image (ignored for local)
            cwd: Working directory for commands
            max_iterations: Maximum tool-calling iterations
            command_timeout: Default timeout for commands
            verbose: Enable verbose logging
        """
        self.model = model
        self.max_iterations = max_iterations
        self.command_timeout = command_timeout
        self.verbose = verbose
        self.env_type = env_type
        self.image = image
        self.cwd = cwd

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client - defaults to OpenRouter
        from openai import OpenAI

        client_kwargs = {}

        # Default to OpenRouter if no base_url provided
        if base_url:
            client_kwargs["base_url"] = base_url
        else:
            client_kwargs["base_url"] = "https://openrouter.ai/api/v1"

        # Handle API key - OpenRouter is the primary provider
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            client_kwargs["api_key"] = os.getenv(
                "OPENROUTER_API_KEY", os.getenv("ANTHROPIC_API_KEY", os.getenv("OPENAI_API_KEY", ""))
            )

        self.client = OpenAI(**client_kwargs)

        # Environment will be created per-task
        self.env = None

        # Tool definition
        self.tools = [TERMINAL_TOOL_DEFINITION]

        print(f"🤖 Mini-SWE Runner initialized")
        print(f"   Model: {self.model}")
        print(f"   Environment: {self.env_type}")
        if self.env_type != "local":
            print(f"   Image: {self.image}")
        print(f"   Max iterations: {self.max_iterations}")

    def _create_env(self):
        """Create the execution environment."""
        print(f"🔧 Creating {self.env_type} environment...")
        self.env = create_environment(
            env_type=self.env_type, image=self.image, cwd=self.cwd, timeout=self.command_timeout
        )
        print(f"✅ Environment ready")

    def _cleanup_env(self):
        """Cleanup the execution environment."""
        if self.env is not None:
            if hasattr(self.env, "cleanup"):
                self.env.cleanup()
            elif hasattr(self.env, "stop"):
                self.env.stop()
            self.env = None

    def _execute_command(self, command: str, timeout: int = None) -> Dict[str, Any]:
        """
        Execute a command in the environment.

        Args:
            command: Bash command to execute
            timeout: Optional timeout override

        Returns:
            Dict with 'output' and 'returncode'
        """
        if self.env is None:
            self._create_env()

        try:
            result = self.env.execute(command, timeout=timeout or self.command_timeout)
            return {"output": result.get("output", ""), "exit_code": result.get("returncode", 0), "error": None}
        except Exception as e:
            return {"output": "", "exit_code": -1, "error": str(e)}

    def _format_tools_for_system_message(self) -> str:
        """Format tool definitions for the system message."""
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tools.append(
                {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                    "required": None,
                }
            )
        return json.dumps(formatted_tools, ensure_ascii=False)

    def _convert_to_hermes_format(
        self, messages: List[Dict[str, Any]], user_query: str, completed: bool
    ) -> List[Dict[str, Any]]:
        """
        Convert internal message format to Hermes trajectory format.

        This produces the exact format used by batch_runner.py.
        """
        trajectory = []

        # System message with tool definitions
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )

        trajectory.append({"from": "system", "value": system_msg})
        trajectory.append({"from": "human", "value": user_query})

        # Process messages (skip first user message as we already added it)
        i = 1
        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Assistant message with tool calls
                    content = ""

                    # Add reasoning if present
                    if msg.get("reasoning"):
                        content = f"<think>{msg['reasoning']}</think>"

                    if msg.get("content"):
                        content += msg["content"] + "\n"

                    # Add tool calls in XML format
                    for tool_call in msg["tool_calls"]:
                        try:
                            arguments = (
                                json.loads(tool_call["function"]["arguments"])
                                if isinstance(tool_call["function"]["arguments"], str)
                                else tool_call["function"]["arguments"]
                            )
                        except json.JSONDecodeError:
                            arguments = {}

                        tool_call_json = {"name": tool_call["function"]["name"], "arguments": arguments}
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"

                    trajectory.append({"from": "gpt", "value": content.rstrip()})

                    # Collect subsequent tool responses
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        tool_content = tool_msg["content"]

                        # Try to parse as JSON
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass

                        tool_response = f"<tool_response>\n"
                        tool_response += json.dumps(
                            {
                                "tool_call_id": tool_msg.get("tool_call_id", ""),
                                "name": msg["tool_calls"][len(tool_responses)]["function"]["name"]
                                if len(tool_responses) < len(msg["tool_calls"])
                                else "unknown",
                                "content": tool_content,
                            },
                            ensure_ascii=False,
                        )
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1

                    if tool_responses:
                        trajectory.append({"from": "tool", "value": "\n".join(tool_responses)})
                        i = j - 1

                else:
                    # Regular assistant message (no tool calls)
                    content = ""
                    if msg.get("reasoning"):
                        content = f"<think>{msg['reasoning']}</think>"
                    content += msg.get("content") or ""
                    trajectory.append({"from": "gpt", "value": content})

            elif msg["role"] == "user":
                trajectory.append({"from": "human", "value": msg["content"]})

            i += 1

        return trajectory

    def run_task(self, task: str) -> Dict[str, Any]:
        """
        Run a single task and return the result with trajectory.

        Args:
            task: The task/prompt to execute

        Returns:
            Dict with trajectory, completion status, and metadata
        """
        print(f"\n{'=' * 60}")
        print(f"📝 Task: {task[:80]}{'...' if len(task) > 80 else ''}")
        print(f"{'=' * 60}")

        # Initialize environment
        self._create_env()

        # Message history
        messages = [{"role": "user", "content": task}]

        # System prompt for the LLM (ephemeral - not saved to trajectory)
        system_prompt = """You are an AI agent that can execute bash commands to complete tasks.

When you need to run commands, use the 'terminal' tool with your bash command.

**Important:**
- When you have completed the task successfully, run: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by a summary
- Be concise and efficient in your approach
- Install any needed tools with apt-get or pip
- Avoid interactive commands (no vim, nano, less, etc.)

Complete the user's task step by step."""

        api_call_count = 0
        completed = False
        final_response = None

        try:
            while api_call_count < self.max_iterations:
                api_call_count += 1
                print(f"\n🔄 API call #{api_call_count}/{self.max_iterations}")

                # Prepare API messages
                api_messages = [{"role": "system", "content": system_prompt}] + messages

                # Make API call
                try:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=api_messages, tools=self.tools, timeout=300.0
                    )
                except Exception as e:
                    self.logger.error(f"API call failed: {e}")
                    break

                assistant_message = response.choices[0].message

                # Log assistant response
                if assistant_message.content:
                    print(f"🤖 Assistant: {assistant_message.content[:100]}...")

                # Check for tool calls
                if assistant_message.tool_calls:
                    print(f"🔧 Tool calls: {len(assistant_message.tool_calls)}")

                    # Add assistant message with tool calls
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                                }
                                for tc in assistant_message.tool_calls
                            ],
                        }
                    )

                    # Execute each tool call
                    for tc in assistant_message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        command = args.get("command", "echo 'No command provided'")
                        timeout = args.get("timeout", self.command_timeout)

                        print(f"   📞 terminal: {command[:60]}...")

                        # Execute command
                        result = self._execute_command(command, timeout)

                        # Format result
                        result_json = json.dumps(
                            {
                                "content": {
                                    "output": result["output"],
                                    "exit_code": result["exit_code"],
                                    "error": result["error"],
                                }
                            },
                            ensure_ascii=False,
                        )

                        # Check for task completion signal
                        if "MINI_SWE_AGENT_FINAL_OUTPUT" in result["output"]:
                            print(f"   ✅ Task completion signal detected!")
                            completed = True

                        # Add tool response
                        messages.append({"role": "tool", "content": result_json, "tool_call_id": tc.id})

                        print(f"   ✅ exit_code={result['exit_code']}, output={len(result['output'])} chars")

                    # If task completed, we can stop
                    if completed:
                        final_response = assistant_message.content
                        break

                else:
                    # No tool calls - final response
                    final_response = assistant_message.content or ""
                    messages.append({"role": "assistant", "content": final_response})
                    completed = True
                    print(f"🎉 Agent finished (no more tool calls)")
                    break

            if api_call_count >= self.max_iterations:
                print(f"⚠️  Reached max iterations ({self.max_iterations})")

        finally:
            # Cleanup environment
            self._cleanup_env()

        # Convert to Hermes trajectory format
        trajectory = self._convert_to_hermes_format(messages, task, completed)

        return {
            "conversations": trajectory,
            "completed": completed,
            "api_calls": api_call_count,
            "metadata": {"model": self.model, "env_type": self.env_type, "timestamp": datetime.now().isoformat()},
        }

    def run_batch(self, prompts: List[str], output_file: str) -> List[Dict[str, Any]]:
        """
        Run multiple tasks and save trajectories to a JSONL file.

        Args:
            prompts: List of task prompts
            output_file: Output JSONL file path

        Returns:
            List of results
        """
        results = []

        print(f"\n📦 Running batch of {len(prompts)} tasks")
        print(f"📁 Output: {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(prompts, 1):
                print(f"\n{'=' * 60}")
                print(f"📋 Task {i}/{len(prompts)}")
                print(f"{'=' * 60}")

                try:
                    result = self.run_task(prompt)
                    results.append(result)

                    # Write to file immediately
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

                    print(f"✅ Task {i} completed (api_calls={result['api_calls']})")

                except Exception as e:
                    self.logger.error(f"Error on task {i}: {e}")
                    error_result = {
                        "conversations": [],
                        "completed": False,
                        "api_calls": 0,
                        "error": str(e),
                        "metadata": {"timestamp": datetime.now().isoformat()},
                    }
                    results.append(error_result)
                    f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                    f.flush()

        print(f"\n✅ Batch complete! {len(results)} trajectories saved to {output_file}")
        return results


# ============================================================================
# CLI Interface
# ============================================================================


def main(
    task: str = None,
    prompts_file: str = None,
    output_file: str = "mini-swe-agent-test1.jsonl",
    model: str = "claude-sonnet-4-20250514",
    base_url: str = None,
    api_key: str = None,
    env: str = "local",
    image: str = "python:3.11-slim",
    cwd: str = "/tmp",
    max_iterations: int = 15,
    timeout: int = 60,
    verbose: bool = False,
):
    """
    Run mini-swe-agent tasks with Hermes trajectory format output.

    Args:
        task: Single task to run (use this OR prompts_file)
        prompts_file: JSONL file with prompts (each line: {"prompt": "..."})
        output_file: Output JSONL file for trajectories
        model: Model name (default: claude-sonnet-4-20250514)
        base_url: API base URL (optional)
        api_key: API key (optional, uses env vars)
        env: Environment type - "local", "docker", or "modal"
        image: Docker/Modal image (default: python:3.11-slim)
        cwd: Working directory (default: /tmp)
        max_iterations: Maximum tool-calling iterations (default: 15)
        timeout: Command timeout in seconds (default: 60)
        verbose: Enable verbose logging

    Examples:
        # Single task with local environment
        python mini_swe_runner.py --task "Create hello.py that prints Hello World"

        # Single task with Docker
        python mini_swe_runner.py --task "List files" --env docker

        # Batch from file
        python mini_swe_runner.py --prompts_file tasks.jsonl --output_file results.jsonl
    """
    print("🚀 Mini-SWE Runner with Hermes Trajectory Format")
    print("=" * 60)

    # Initialize runner
    runner = MiniSWERunner(
        model=model,
        base_url=base_url,
        api_key=api_key,
        env_type=env,
        image=image,
        cwd=cwd,
        max_iterations=max_iterations,
        command_timeout=timeout,
        verbose=verbose,
    )

    if task:
        # Single task mode
        result = runner.run_task(task)

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\n📁 Trajectory saved to: {output_file}")
        print(f"✅ Completed: {result['completed']}")
        print(f"📞 API calls: {result['api_calls']}")
        print(f"💬 Turns: {len(result['conversations'])}")

    elif prompts_file:
        # Batch mode
        prompts = []
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        prompts.append(entry.get("prompt", entry.get("task", "")))
                    except json.JSONDecodeError:
                        prompts.append(line)

        if not prompts:
            print(f"❌ No prompts found in {prompts_file}")
            return

        runner.run_batch(prompts, output_file)

    else:
        print("❌ Please provide either --task or --prompts_file")
        print("   Example: python mini_swe_runner.py --task 'Create a hello world script'")


if __name__ == "__main__":
    fire.Fire(main)
