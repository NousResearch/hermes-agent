#!/usr/bin/env python3
"""SWE Runner with Hermes Trajectory Format

Runs tool-calling agent tasks in Hermes-Agent's execution environments (local,
docker, modal) and writes trajectories in Hermes format (from/value pairs with
<tool_call>/<tool_response> XML), compatible with batch_runner.py and
trajectory_compressor.py. Supports single tasks and JSONL batch mode.

Usage:
    python mini_swe_runner.py --task "Create a hello world Python script" --env local
    python mini_swe_runner.py --task "List files in /tmp" --env docker --image python:3.11-slim
    python mini_swe_runner.py --prompts_file prompts.jsonl --output_file trajectories.jsonl --env docker
"""

import importlib
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import fire
from dotenv import load_dotenv
from agent.tool_dispatch_helpers import make_tool_result_message
from trajectory_compressor import _effective_temperature_for_model

# Load environment variables
load_dotenv()


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

SYSTEM_PROMPT = """You are an AI agent that can execute bash commands to complete tasks.

When you need to run commands, use the 'terminal' tool with your bash command.

**Important:**
- When you have completed the task successfully, run: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by a summary
- Be concise and efficient in your approach
- Install any needed tools with apt-get or pip
- Avoid interactive commands (no vim, nano, less, etc.)

Complete the user's task step by step."""

HERMES_SYSTEM_PREFIX = (
    "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
    "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
    "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
    "into functions. After calling & executing the functions, you will be provided with function results within "
    "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
)
HERMES_SYSTEM_SUFFIX = (
    "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
    "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
    "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
    "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
    "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
)

_OPENROUTER_URL = "https://openrouter.ai/api/v1"


def create_environment(env_type: str = "local", image: str = "python:3.11-slim", cwd: str = "/tmp", timeout: int = 60, **kwargs):
    """Create a Hermes execution environment (``local`` ignores ``image``/``kwargs``)."""
    if env_type == "local":
        from tools.environments.local import LocalEnvironment
        return LocalEnvironment(cwd=cwd, timeout=timeout)
    if env_type not in ("docker", "modal"):
        raise ValueError(f"Unknown environment type: {env_type}. Use 'local', 'docker', or 'modal'")
    module = importlib.import_module(f"tools.environments.{env_type}")
    return getattr(module, f"{env_type.capitalize()}Environment")(image=image, cwd=cwd, timeout=timeout, **kwargs)


def _parse_json_args(raw: Any) -> Any:
    """Decode tool-call arguments; invalid JSON becomes ``{}``."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _gpt_content(msg: Dict[str, Any], content: str) -> str:
    """Prefix ``content`` with a ``<think>`` block when the message carries reasoning."""
    return (f"<think>{msg['reasoning']}</think>" if msg.get("reasoning") else "") + content


class MiniSWERunner:
    """Tool-calling agent loop over a Hermes execution environment, emitting Hermes trajectories."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4.6", base_url: str = None, api_key: str = None,
                 env_type: str = "local", image: str = "python:3.11-slim", cwd: str = "/tmp",
                 max_iterations: int = 15, command_timeout: int = 60, verbose: bool = False):
        self.model, self.max_iterations, self.command_timeout, self.verbose = model, max_iterations, command_timeout, verbose
        self.env_type, self.image, self.cwd = env_type, image, cwd
        self.logger = logging.getLogger(__name__)
        self.client = self._init_client(base_url, api_key)
        self.env = None  # created per-task
        self.tools = [TERMINAL_TOOL_DEFINITION]
        print("🤖 Mini-SWE Runner initialized")
        print(f"   Model: {self.model}")
        print(f"   Environment: {self.env_type}")
        if self.env_type != "local":
            print(f"   Image: {self.image}")
        print(f"   Max iterations: {self.max_iterations}")

    def _init_client(self, base_url: Optional[str], api_key: Optional[str]):
        """Explicit api_key/base_url -> direct OpenAI client; otherwise the provider router."""
        if api_key or base_url:
            from openai import OpenAI
            return OpenAI(base_url=base_url or _OPENROUTER_URL, api_key=api_key or os.getenv(
                "OPENROUTER_API_KEY", os.getenv("ANTHROPIC_API_KEY", os.getenv("OPENAI_API_KEY", ""))))
        from agent.auxiliary_client import resolve_provider_client
        client, _ = resolve_provider_client("openrouter", model=self.model)
        if client is None:
            client, _ = resolve_provider_client("auto", model=self.model)
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=_OPENROUTER_URL, api_key=os.getenv("OPENROUTER_API_KEY", ""))
        return client

    def _create_env(self):
        print(f"🔧 Creating {self.env_type} environment...")
        self.env = create_environment(env_type=self.env_type, image=self.image, cwd=self.cwd, timeout=self.command_timeout)
        print("✅ Environment ready")

    def _cleanup_env(self):
        if self.env is not None:
            stop = getattr(self.env, 'cleanup', None) or getattr(self.env, 'stop', None)
            if stop:
                stop()
            self.env = None

    def _execute_command(self, command: str, timeout: int = None) -> Dict[str, Any]:
        """Run ``command`` in the environment; returns ``{output, exit_code, error}``."""
        if self.env is None:
            self._create_env()
        try:
            result = self.env.execute(command, timeout=timeout or self.command_timeout)
            return {"output": result.get("output", ""), "exit_code": result.get("returncode", 0), "error": None}
        except Exception as e:
            return {"output": "", "exit_code": -1, "error": str(e)}

    def _format_tools_for_system_message(self) -> str:
        return json.dumps([
            {"name": t["function"]["name"], "description": t["function"].get("description", ""),
             "parameters": t["function"].get("parameters", {}), "required": None}
            for t in self.tools
        ], ensure_ascii=False)

    def _tool_response_turn(self, messages: List[Dict[str, Any]], i: int) -> tuple:
        """Fold the tool messages following assistant turn ``i`` into one ``tool`` value.

        Returns ``(value_or_None, index_of_last_consumed_message)``.
        """
        tool_calls = messages[i]["tool_calls"]
        tool_responses = []
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "tool":
            tool_msg = messages[j]
            tool_content = tool_msg["content"]
            try:
                if tool_content.strip().startswith(("{", "[")):
                    tool_content = json.loads(tool_content)
            except (json.JSONDecodeError, AttributeError):
                pass
            k = len(tool_responses)
            body = json.dumps({"tool_call_id": tool_msg.get("tool_call_id", ""),
                               "name": tool_calls[k]["function"]["name"] if k < len(tool_calls) else "unknown",
                               "content": tool_content}, ensure_ascii=False)
            tool_responses.append(f"<tool_response>\n{body}\n</tool_response>")
            j += 1
        return ("\n".join(tool_responses), j - 1) if tool_responses else (None, i)

    def _convert_to_hermes_format(self, messages: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
        """Convert the OpenAI-style message list to the Hermes trajectory format used by batch_runner.py."""
        system_msg = HERMES_SYSTEM_PREFIX + f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n" + HERMES_SYSTEM_SUFFIX
        trajectory = [{"from": "system", "value": system_msg}, {"from": "human", "value": user_query}]
        i = 1  # first user message already added
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "user":
                trajectory.append({"from": "human", "value": msg["content"]})
            elif msg["role"] == "assistant" and not msg.get("tool_calls"):
                trajectory.append({"from": "gpt", "value": _gpt_content(msg, msg.get("content") or "")})
            elif msg["role"] == "assistant":
                content = (msg["content"] + "\n") if msg.get("content") else ""
                for tool_call in msg["tool_calls"]:
                    if isinstance(tool_call, dict) and tool_call:
                        tool_call_json = {"name": tool_call["function"]["name"], "arguments": _parse_json_args(tool_call["function"]["arguments"])}
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                trajectory.append({"from": "gpt", "value": _gpt_content(msg, content).rstrip()})
                tool_value, i = self._tool_response_turn(messages, i)
                if tool_value is not None:
                    trajectory.append({"from": "tool", "value": tool_value})
            i += 1
        return trajectory

    def _call_model(self, messages: List[Dict[str, Any]]):
        """One chat completion with the ephemeral system prompt; returns the message or None on API error."""
        api_kwargs = {"model": self.model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                      "tools": self.tools, "timeout": 300.0}
        # requested_temperature=None: only fixed model contracts (Kimi omit / Arcee 0.5) apply here.
        fixed_temperature = _effective_temperature_for_model(self.model, None, str(getattr(self.client, "base_url", "") or ""))
        if fixed_temperature is not None:
            api_kwargs["temperature"] = fixed_temperature
        try:
            return self.client.chat.completions.create(**api_kwargs).choices[0].message
        except Exception as e:
            self.logger.error("API call failed: %s", e)

    def _run_tool_calls(self, assistant_message, messages: List[Dict[str, Any]]) -> bool:
        """Record the assistant turn, execute each terminal call, append results; True if the completion signal fired."""
        print(f"🔧 Tool calls: {len(assistant_message.tool_calls)}")
        messages.append({"role": "assistant", "content": assistant_message.content, "tool_calls": [
            {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in assistant_message.tool_calls
        ]})

        completed = False
        for tc in assistant_message.tool_calls:
            args = _parse_json_args(tc.function.arguments)
            command = args.get("command", "echo 'No command provided'")
            print(f"   📞 terminal: {command[:60]}...")
            result = self._execute_command(command, args.get("timeout", self.command_timeout))
            if "MINI_SWE_AGENT_FINAL_OUTPUT" in result["output"]:
                print("   ✅ Task completion signal detected!")
                completed = True
            messages.append(make_tool_result_message(tc.function.name, json.dumps({"content": result}, ensure_ascii=False), tc.id))
            print(f"   ✅ exit_code={result['exit_code']}, output={len(result['output'])} chars")
        return completed

    def run_task(self, task: str) -> Dict[str, Any]:
        """Run one task; returns ``{conversations, completed, api_calls, metadata}``."""
        print(f"\n{'='*60}")
        print(f"📝 Task: {task[:80]}{'...' if len(task) > 80 else ''}")
        print(f"{'='*60}")
        self._create_env()
        messages = [{"role": "user", "content": task}]
        api_call_count = 0
        completed = False
        try:
            while api_call_count < self.max_iterations:
                api_call_count += 1
                print(f"\n🔄 API call #{api_call_count}/{self.max_iterations}")
                assistant_message = self._call_model(messages)
                if assistant_message is None:
                    break
                if assistant_message.content:
                    print(f"🤖 Assistant: {assistant_message.content[:100]}...")
                if not assistant_message.tool_calls:
                    messages.append({"role": "assistant", "content": assistant_message.content or ""})
                    completed = True
                    print("🎉 Agent finished (no more tool calls)")
                    break
                if self._run_tool_calls(assistant_message, messages):
                    completed = True
                    break
            if api_call_count >= self.max_iterations:
                print(f"⚠️  Reached max iterations ({self.max_iterations})")
        finally:
            self._cleanup_env()
        return {"conversations": self._convert_to_hermes_format(messages, task), "completed": completed, "api_calls": api_call_count,
                "metadata": {"model": self.model, "env_type": self.env_type, "timestamp": datetime.now().isoformat()}}

    def run_batch(self, prompts: List[str], output_file: str) -> List[Dict[str, Any]]:
        """Run every prompt, appending each result to ``output_file`` as it finishes."""
        results = []
        print(f"\n📦 Running batch of {len(prompts)} tasks")
        print(f"📁 Output: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, prompt in enumerate(prompts, 1):
                print(f"\n{'='*60}")
                print(f"📋 Task {i}/{len(prompts)}")
                print(f"{'='*60}")
                try:
                    result = self.run_task(prompt)
                    print(f"✅ Task {i} completed (api_calls={result['api_calls']})")
                except Exception as e:
                    self.logger.error("Error on task %s: %s", i, e)
                    result = {"conversations": [], "completed": False, "api_calls": 0, "error": str(e),
                              "metadata": {"timestamp": datetime.now().isoformat()}}
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
        print(f"\n✅ Batch complete! {len(results)} trajectories saved to {output_file}")
        return results


def _load_prompts(prompts_file: str) -> List[str]:
    """One prompt per non-blank line: JSON ``{"prompt"|"task": ...}`` or raw text."""
    prompts = []
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                prompts.append(entry.get("prompt", entry.get("task", "")))
            except json.JSONDecodeError:
                prompts.append(line)
    return prompts


def main(
    task: str = None,
    prompts_file: str = None,
    output_file: str = "swe-runner-test1.jsonl",
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
    Run SWE tasks with Hermes trajectory format output.
    
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
    """
    print("🚀 Mini-SWE Runner with Hermes Trajectory Format")
    print("=" * 60)
    # Configure root logging at the entry point (not in library __init__).
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    runner = MiniSWERunner(model=model, base_url=base_url, api_key=api_key, env_type=env, image=image, cwd=cwd,
                           max_iterations=max_iterations, command_timeout=timeout, verbose=verbose)
    if task:
        result = runner.run_task(task)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\n📁 Trajectory saved to: {output_file}")
        print(f"✅ Completed: {result['completed']}")
        print(f"📞 API calls: {result['api_calls']}")
        print(f"💬 Turns: {len(result['conversations'])}")
    elif prompts_file:
        prompts = _load_prompts(prompts_file)
        if not prompts:
            print(f"❌ No prompts found in {prompts_file}")
            return
        runner.run_batch(prompts, output_file)
    else:
        print("❌ Please provide either --task or --prompts_file")
        print("   Example: python mini_swe_runner.py --task 'Create a hello world script'")


if __name__ == "__main__":
    fire.Fire(main)
