"""
Hermes-Agent Atropos Environments

Provides a layered integration between hermes-agent's tool-calling capabilities
and the Atropos RL training framework.

Core layers:
    - agent_loop: Reusable multi-turn agent loop with standard OpenAI-spec tool calling
    - tool_context: Per-rollout tool access handle for reward/verification functions
    - hermes_base_env: Abstract base environment (BaseEnv subclass) for Atropos
    - tool_call_parsers: Client-side tool call parser registry for Phase 2 (VLLM /generate)

Concrete environments:
    - terminal_test_env/: Simple file-creation tasks for testing the stack
    - hermes_swe_env/: SWE-bench style tasks with Modal sandboxes

Benchmarks (eval-only):
    - benchmarks/terminalbench_2/: Terminal-Bench 2.0 evaluation
"""

from environments.agent_loop import AgentResult, HermesAgentLoop
from environments.tool_context import ToolContext
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig

__all__ = [
    "AgentResult",
    "HermesAgentLoop",
    "ToolContext",
    "HermesAgentBaseEnv",
    "HermesAgentEnvConfig",
]
