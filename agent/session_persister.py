"""Session persistence -- JSON logs, SQLite, and trajectory files.

Extracted from AIAgent. Owns session_id and the session_log_file path,
ensuring they stay in sync (fixing the pre-existing bug where compression
updated session_id but not session_log_file).

Dependencies:
    agent.trajectory -- save_trajectory, convert_scratchpad_to_think, has_incomplete_scratchpad
    (No imports from run_agent.py)
"""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.trajectory import (
    save_trajectory as _save_trajectory_to_file,
    convert_scratchpad_to_think,
    has_incomplete_scratchpad,
)

logger = logging.getLogger(__name__)


class SessionPersister:
    """Manages session persistence to JSON files and SQLite.

    Args:
        session_id: Initial session identifier.
        session_db: Optional SQLite session database instance.
        logs_dir: Directory for session log JSON files.
        model: Model name string.
        base_url: API base URL.
        platform: Platform identifier (e.g. "cli", "telegram").
        session_start: Session start datetime.
        save_trajectories: Whether to save trajectory JSONL files.
        tools: Tool definitions list (for trajectory format).
        format_tools_fn: Callable that formats tools for system message (for trajectories).
        verbose_logging: Enable verbose logging.
        quiet_mode: Suppress output.
        save_interval: Save session log every N calls to maybe_save_session_log.
    """

    def __init__(
        self,
        *,
        session_id: str,
        session_db=None,
        logs_dir: Path,
        model: str,
        base_url: str,
        platform: str = None,
        session_start: datetime = None,
        save_trajectories: bool = False,
        tools: list = None,
        format_tools_fn: Callable = None,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        save_interval: int = 1,
    ):
        self._session_id = session_id
        self._session_db = session_db
        self._logs_dir = Path(logs_dir)
        self._model = model
        self._base_url = base_url
        self._platform = platform
        self._session_start = session_start or datetime.now()
        self._save_trajectories = save_trajectories
        self._tools = tools or []
        self._format_tools_fn = format_tools_fn
        self._verbose_logging = verbose_logging
        self._quiet_mode = quiet_mode
        self._save_interval = save_interval
        self._save_counter = 0
        # Atomic: session_log_file always matches session_id
        self._session_log_file = self._logs_dir / f"session_{session_id}.json"
        self._session_messages: List[Dict[str, Any]] = []  # Last known messages state

    # -- Properties -----------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str):
        """Atomically update both session_id and session_log_file."""
        self._session_id = value
        self._session_log_file = self._logs_dir / f"session_{value}.json"

    @property
    def session_log_file(self) -> Path:
        return self._session_log_file

    # -- Single-message logging -----------------------------------------------

    def log_message(self, msg: Dict):
        """Log a single message to SQLite immediately.

        Called after each messages.append().  No-op when session_db is None.
        """
        if not self._session_db:
            return
        try:
            role = msg.get("role", "unknown")
            content = msg.get("content")
            tool_calls_data = None
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_data = [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in msg.tool_calls
                ]
            elif isinstance(msg.get("tool_calls"), list):
                tool_calls_data = msg["tool_calls"]
            self._session_db.append_message(
                session_id=self._session_id,
                role=role,
                content=content,
                tool_name=msg.get("tool_name"),
                tool_calls=tool_calls_data,
                tool_call_id=msg.get("tool_call_id"),
                finish_reason=msg.get("finish_reason"),
            )
        except Exception as e:
            logger.debug("Session DB log_msg failed: %s", e)

    # -- Bulk persistence -----------------------------------------------------

    def persist(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save session state to both JSON log and SQLite."""
        self._session_messages = messages
        self.save_session_log(messages)
        self.flush_to_db(messages, conversation_history)

    def save_session_log(self, messages: List[Dict] = None):
        """Save the full raw session to a JSON file.

        Stores every message exactly as the agent sees it.
        REASONING_SCRATCHPAD tags are converted to <think> blocks for consistency.
        Overwritten after each turn so it always reflects the latest state.
        """
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # Clean assistant content for session logs
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                cleaned.append(msg)

            entry = {
                "session_id": self._session_id,
                "model": self._model,
                "base_url": self._base_url,
                "platform": self._platform,
                "session_start": self._session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            with open(self._session_log_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            if self._verbose_logging:
                logging.warning(f"Failed to save session log: {e}")

    def maybe_save_session_log(self, messages: List[Dict] = None):
        """Save session log at interval (every save_interval calls)."""
        self._save_counter += 1
        if self._save_counter >= self._save_interval:
            self._save_counter = 0
            self.save_session_log(messages)

    def flush_to_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Persist un-logged messages to the SQLite session store.

        Called both at the normal end of run_conversation and from every early-
        return path so that tool calls, tool responses, and assistant messages
        are never lost even when the conversation errors out.
        """
        if not self._session_db:
            return
        try:
            start_idx = (len(conversation_history) if conversation_history else 0) + 1
            for msg in messages[start_idx:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self._session_db.append_message(
                    session_id=self._session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                )
        except Exception as e:
            logger.debug("Session DB append_message failed: %s", e)

    # -- Trajectory -----------------------------------------------------------

    def save_trajectory(self, messages: List[Dict], user_query: str, completed: bool):
        """Save conversation trajectory to JSONL file."""
        if not self._save_trajectories:
            return

        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self._model, completed)

    # -- Compression session split --------------------------------------------

    def create_compression_session(self, *, platform: str = None, model: str = None,
                                   parent_session_id: str = None) -> str:
        """End old session and create new one after compression.

        Returns the new session_id.
        """
        new_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if self._session_db:
            try:
                self._session_db.end_session(self._session_id, "compression")
                self.session_id = new_id  # Atomic update via setter
                self._session_db.create_session(
                    session_id=self._session_id,
                    source=platform or self._platform or "cli",
                    model=model or self._model,
                    parent_session_id=parent_session_id or self._session_id,
                )
            except Exception as e:
                logger.debug("Session DB compression split failed: %s", e)
        else:
            self.session_id = new_id  # Still update even without DB
        return self._session_id

    def update_system_prompt(self, prompt: str):
        """Update system prompt in the session DB."""
        if self._session_db:
            try:
                self._session_db.update_system_prompt(self._session_id, prompt)
            except Exception as e:
                logger.debug("Session DB update_system_prompt failed: %s", e)

    # -- Content cleaning (static) --------------------------------------------

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        # Strip extra newlines before/after think blocks
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    # -- Trajectory format conversion -----------------------------------------

    def _format_tools_for_system_message(self) -> str:
        """Format tool definitions for the system message in the trajectory format."""
        if self._format_tools_fn:
            return self._format_tools_fn()
        if not self._tools:
            return "[]"
        # Convert tool definitions to the format expected in trajectories
        formatted_tools = []
        for tool in self._tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None,  # Match the format in the example
            }
            formatted_tools.append(formatted_tool)
        return json.dumps(formatted_tools, ensure_ascii=False)

    def _convert_to_trajectory_format(
        self, messages: List[Dict[str, Any]], user_query: str, completed: bool
    ) -> List[Dict[str, Any]]:
        """Convert internal message format to trajectory format for saving.

        Args:
            messages: Internal message history.
            user_query: Original user query.
            completed: Whether the conversation completed successfully.

        Returns:
            Messages in trajectory format (ShareGPT-style).
        """
        trajectory = []

        # Add system message with tool definitions
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

        trajectory.append({
            "from": "system",
            "value": system_msg
        })

        # Add the actual user prompt (from the dataset) as the first human message
        trajectory.append({
            "from": "human",
            "value": user_query
        })

        # Skip the first message (the user query) since we already added it above.
        # Prefill messages are injected at API-call time only (not in the messages
        # list), so no offset adjustment is needed here.
        i = 1

        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "assistant":
                # Check if this message has tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Format assistant message with tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""

                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"

                    if msg.get("content") and msg["content"].strip():
                        # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                        # (used when native thinking is disabled and model reasons via XML)
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"

                    # Add tool calls wrapped in XML tags
                    for tool_call in msg["tool_calls"]:
                        # Parse arguments - should always succeed since we validate during conversation
                        # but keep try-except as safety net
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            # This shouldn't happen since we validate and retry during conversation,
                            # but if it does, log warning and use empty dict
                            logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                            arguments = {}

                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"

                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    # so the format is consistent for training data
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content

                    trajectory.append({
                        "from": "gpt",
                        "value": content.rstrip()
                    })

                    # Collect all subsequent tool responses
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # Format tool response with XML tags
                        tool_response = "<tool_response>\n"

                        # Try to parse tool content as JSON if it looks like JSON
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # Keep as string if not valid JSON

                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": msg["tool_calls"][len(tool_responses)]["function"]["name"] if len(tool_responses) < len(msg["tool_calls"]) else "unknown",
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1

                    # Add all tool responses as a single message
                    if tool_responses:
                        trajectory.append({
                            "from": "tool",
                            "value": "\n".join(tool_responses)
                        })
                        i = j - 1  # Skip the tool messages we just processed

                else:
                    # Regular assistant message without tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""

                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"

                    # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                    # (used when native thinking is disabled and model reasons via XML)
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)

                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content

                    trajectory.append({
                        "from": "gpt",
                        "value": content.strip()
                    })

            elif msg["role"] == "user":
                trajectory.append({
                    "from": "human",
                    "value": msg["content"]
                })

            i += 1

        return trajectory
