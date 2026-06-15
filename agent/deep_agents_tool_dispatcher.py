"""Tool dispatcher that wraps conversation_loop.py tool dispatch.

Provides:
- execute_hermes_tools: runs a single Hermes tool via the existing
  handle_function_call pipeline (tool_executor machinery).
- build_hermes_engine_tool: constructs a StructuredTool that can be
  passed to DeepAgents/LangGraph to run the full turn loop as a sub-agent.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-tool execution (for the _HermesToolAdapter path)
# ---------------------------------------------------------------------------


def execute_hermes_tools(tool_name: str, tool_args: dict) -> str:
    """Execute a single Hermes tool (called by the tool adapter).

    This is the path for individual tool calls within the conversation_loop.
    It delegates to the standard handle_function_call to preserve all existing
    tool dispatch, error handling, and ContextVar propagation.
    """
    from model_tools import handle_function_call

    try:
        result = handle_function_call(
            function_name=tool_name,
            function_args=tool_args,
        )
        return result
    except Exception as e:
        error_str = str(e)
        return json.dumps({"error": error_str}, ensure_ascii=False)


def run_hermes_engine(
    user_message: str,
    conversation_history: str,  # JSON-serialized list of Hermes messages
    session_id: str,
    task_id: str,
) -> str:
    """Run the full Hermes conversation loop as a single tool call.

    This is the "hermes_engine" tool body. It runs inside
    thread_executor.submit() in DeepAgentsAIAgent._run_sync to avoid
    blocking LangGraph's event loop.

    Parameters are JSON-serialized strings (because tools only accept
    string arguments in LangChain); they are parsed at the entry point.

    Returns a JSON-serialized result dict matching run_conversation's output
    shape: {final_response, messages, api_calls, completed, ...}.
    """
    from run_agent import AIAgent as _NativeAIAgent
    from agent.conversation_loop import run_conversation

    try:
        history = json.loads(conversation_history) if conversation_history else []
    except json.JSONDecodeError:
        history = []

    agent = _NativeAIAgent.__new__(_NativeAIAgent)
    # Minimal attribute setup — the real init is done by DeepAgentsAIAgent
    agent.session_id = session_id
    agent.model = ""
    agent.provider = ""
    agent.base_url = ""
    agent.max_iterations = 90
    agent.quiet_mode = True
    agent.enabled_toolsets = []
    agent.disabled_toolsets = None
    agent.iteration_budget = None
    agent.interrupt_requested = False
    agent._interrupt_message = None
    agent._stream_callback = None

    result = run_conversation(
        agent=agent,
        user_message=user_message,
        system_message=None,
        conversation_history=history,
        task_id=task_id,
    )

    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# hermes_engine StructuredTool wrapper
# ---------------------------------------------------------------------------


def build_hermes_engine_tool():
    """Build the LangChain StructuredTool wrapper for the full loop."""
    from model_tools import get_tool_definitions
    from run_agent import AIAgent as _NativeAIAgent

    def _execute_hermes_engine(
        user_message: str,
        conversation_history: str,
        session_id: str,
        task_id: str,
    ) -> str:
        """Run the full Hermes conversation loop.

        Accepts a user message and conversation history, runs the complete
        agent loop (LLM calls, tool dispatch, retries, compression), and
        returns the final response with full turn metadata.

        Args:
            user_message: The user's latest message.
            conversation_history: JSON-serialized list of prior messages.
            session_id: Hermes session ID.
            task_id: Optional task identifier.

        Returns:
            JSON-serialized result dict with final_response, messages,
            api_calls, completed, etc.
        """
        return run_hermes_engine(
            user_message=user_message,
            conversation_history=conversation_history,
            session_id=session_id,
            task_id=task_id,
        )

    return StructuredTool.from_function(
        name="hermes_engine",
        description=(
            "Run the full Hermes agent conversation loop. Accepts a user message "
            "and conversation history, runs the complete agent loop (LLM calls, "
            "tool dispatch, retries, compression), and returns the final response "
            "with full turn metadata."
        ),
        func=_execute_hermes_engine,
    )
