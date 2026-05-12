import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Attempt to import agent.model_metadata.estimate_tokens_rough,
# otherwise provide a fallback locally.
try:
    from agent.model_metadata import estimate_tokens_rough
except ImportError:
    # Fallback roughly 4 chars per token if missing
    def estimate_tokens_rough(text: str) -> int:
        if not text:
            return 0
        return len(text) // 4


@dataclass
class TokenBucket:
    name: str
    tokens: int
    chars: int
    count: int = 1


@dataclass
class TokenBreakdown:
    total_estimated_tokens: int
    buckets: Dict[str, TokenBucket] = field(default_factory=dict)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return estimate_tokens_rough(text)


def estimate_message_tokens(message: dict) -> int:
    """Estimate tokens for a single OpenAI-format message dictionary."""
    if not message:
        return 0

    total_tokens = 0

    # 1. Content
    content = message.get("content")
    if isinstance(content, str):
        total_tokens += estimate_text_tokens(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "text" in block:
                total_tokens += estimate_text_tokens(block.get("text", ""))
            # Ignoring image tokens for this rough estimate, could add later if needed.

    # 2. Tool Calls
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            # Add function name and args
            func = tc.get("function", {})
            total_tokens += estimate_text_tokens(func.get("name", ""))
            args = func.get("arguments", "")
            if isinstance(args, str):
                total_tokens += estimate_text_tokens(args)
            elif isinstance(args, dict):
                total_tokens += estimate_text_tokens(json.dumps(args))

    # 3. Reasoning content (used by some models like Claude 3.7 or deepseek-reasoner)
    for reasoning_key in ["reasoning", "reasoning_content", "reasoning_details"]:
        r_content = message.get(reasoning_key)
        if isinstance(r_content, str):
            total_tokens += estimate_text_tokens(r_content)
        elif isinstance(r_content, dict) and "text" in r_content:
            total_tokens += estimate_text_tokens(r_content["text"])

    return total_tokens


def classify_message(message: dict, index: int, current_turn_user_idx: Optional[int]) -> str:
    """Classify an API message into a bucket name."""
    role = message.get("role", "")

    if role == "system":
        return "system_prompt"

    if current_turn_user_idx is not None and index == current_turn_user_idx:
        return "current_user_turn"

    if role == "user":
        return "conversation_history_user"

    if role == "assistant":
        if message.get("tool_calls"):
            return "tool_call_arguments"

        # Check if it has reasoning but no content/tool_calls
        has_content = bool(message.get("content"))
        has_reasoning = any(message.get(k) for k in ["reasoning", "reasoning_content", "reasoning_details"])
        if has_reasoning and not has_content:
            return "assistant_reasoning_replay"

        return "conversation_history_assistant"

    if role == "tool":
        return "conversation_history_tool_result"

    return "other"


def estimate_request_breakdown(
    api_messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    current_turn_user_idx: Optional[int] = None,
    injected_context_chars: int = 0
) -> TokenBreakdown:
    """Estimate total request token usage and breakdown by bucket."""
    buckets = {}
    total_tokens = 0

    def _add_to_bucket(name: str, tokens: int, chars: int):
        if name not in buckets:
            buckets[name] = TokenBucket(name=name, tokens=0, chars=0, count=0)
        buckets[name].tokens += tokens
        buckets[name].chars += chars
        buckets[name].count += 1
        nonlocal total_tokens
        total_tokens += tokens

    # 1. Process tools (tool schemas)
    if tools:
        tools_json = json.dumps(tools, sort_keys=True, separators=(",", ":"))
        tools_chars = len(tools_json)
        tools_tokens = estimate_text_tokens(tools_json)
        _add_to_bucket("tool_schemas", tools_tokens, tools_chars)

    # 2. Process injected context (added to system prompt or first user message)
    if injected_context_chars > 0:
        ctx_tokens = injected_context_chars // 4
        _add_to_bucket("injected_context", ctx_tokens, injected_context_chars)

    # 3. Process messages
    for idx, msg in enumerate(api_messages):
        msg_tokens = estimate_message_tokens(msg)
        # Approximate chars by getting raw content length for basic reporting
        content = msg.get("content", "")
        msg_chars = len(content) if isinstance(content, str) else 0

        bucket_name = classify_message(msg, idx, current_turn_user_idx)
        _add_to_bucket(bucket_name, msg_tokens, msg_chars)

    return TokenBreakdown(total_estimated_tokens=total_tokens, buckets=buckets)
