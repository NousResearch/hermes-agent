"""Anthropic Messages API adapter for Hermes Agent.

Translates between Hermes's internal OpenAI-style message format and
Anthropic's Messages API. Follows the same pattern as the codex_responses
adapter — all provider-specific logic is isolated here.

Auth supports:
  - Regular API keys (sk-ant-api*) → x-api-key header
  - OAuth setup-tokens (sk-ant-oat*) → Bearer auth + beta header
  - Claude Code credentials (~/.claude/.credentials.json) → Bearer auth
"""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

THINKING_BUDGET = {"xhigh": 32000, "high": 16000, "medium": 8000, "low": 4000}

# Beta headers for enhanced features (sent with ALL auth types)
_COMMON_BETAS = [
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
]

# Additional beta headers required for OAuth/subscription auth
_OAUTH_ONLY_BETAS = [
    "oauth-2025-04-20",
]


def _is_oauth_token(key: str) -> bool:
    """Check if the key is an OAuth/setup token (not a regular Console API key).

    Regular API keys start with 'sk-ant-api'. Everything else (setup-tokens
    starting with 'sk-ant-oat', managed keys, JWTs, etc.) needs Bearer auth.
    """
    if not key:
        return False
    # Regular Console API keys use x-api-key header
    if key.startswith("sk-ant-api"):
        return False
    # Everything else (setup-tokens, managed keys, JWTs) uses Bearer auth
    return True


def build_anthropic_client(api_key: str, base_url: str = None):
    """Create an Anthropic client, auto-detecting setup-tokens vs API keys.

    Returns an anthropic.Anthropic instance.
    """
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )
    from httpx import Timeout

    kwargs = {
        "timeout": Timeout(timeout=900.0, connect=10.0),
    }
    if base_url:
        kwargs["base_url"] = base_url

    if _is_oauth_token(api_key):
        # OAuth access token / setup-token → Bearer auth + beta headers
        all_betas = _COMMON_BETAS + _OAUTH_ONLY_BETAS
        kwargs["auth_token"] = api_key
        kwargs["default_headers"] = {"anthropic-beta": ",".join(all_betas)}
    else:
        # Regular API key → x-api-key header + common betas
        kwargs["api_key"] = api_key
        if _COMMON_BETAS:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(_COMMON_BETAS)}

    return _anthropic_sdk.Anthropic(**kwargs)


def read_claude_code_credentials() -> Optional[Dict[str, Any]]:
    """Read credentials from Claude Code's config files.

    Checks two locations (in order):
      1. ~/.claude.json — top-level primaryApiKey (native binary, v2.x)
      2. ~/.claude/.credentials.json — claudeAiOauth block (npm/legacy installs)

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
    # 1. Native binary (v2.x): ~/.claude.json with top-level primaryApiKey
    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            data = json.loads(claude_json.read_text(encoding="utf-8"))
            primary_key = data.get("primaryApiKey", "")
            if primary_key:
                return {
                    "accessToken": primary_key,
                    "refreshToken": "",
                    "expiresAt": 0,  # Managed keys don't have a user-visible expiry
                }
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude.json: %s", e)

    # 2. Legacy/npm installs: ~/.claude/.credentials.json
    cred_path = Path.home() / ".claude" / ".credentials.json"
    if cred_path.exists():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            oauth_data = data.get("claudeAiOauth")
            if oauth_data and isinstance(oauth_data, dict):
                access_token = oauth_data.get("accessToken", "")
                if access_token:
                    return {
                        "accessToken": access_token,
                        "refreshToken": oauth_data.get("refreshToken", ""),
                        "expiresAt": oauth_data.get("expiresAt", 0),
                    }
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude/.credentials.json: %s", e)

    return None


def is_claude_code_token_valid(creds: Dict[str, Any]) -> bool:
    """Check if Claude Code credentials have a non-expired access token."""
    import time

    expires_at = creds.get("expiresAt", 0)
    if not expires_at:
        # No expiry set (managed keys) — valid if token is present
        return bool(creds.get("accessToken"))

    # expiresAt is in milliseconds since epoch
    now_ms = int(time.time() * 1000)
    # Allow 60 seconds of buffer
    return now_ms < (expires_at - 60_000)


def resolve_anthropic_token() -> Optional[str]:
    """Resolve an Anthropic token from all available sources.

    Priority:
      1. ANTHROPIC_API_KEY env var (regular API key)
      2. ANTHROPIC_TOKEN env var (OAuth/setup token)
      3. Claude Code credentials (~/.claude/.credentials.json)

    Returns the token string or None.
    """
    # 1. Regular API key
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key

    # 2. OAuth/setup token env var
    token = os.getenv("ANTHROPIC_TOKEN", "").strip()
    if token:
        return token

    # Also check CLAUDE_CODE_OAUTH_TOKEN (used by Claude Code for setup-tokens)
    cc_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if cc_token:
        return cc_token

    # 3. Claude Code credential file
    creds = read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        logger.debug("Using Claude Code credentials from ~/.claude/.credentials.json")
        return creds["accessToken"]
    elif creds:
        logger.debug("Claude Code credentials expired — run 'claude' to refresh")

    return None


# ---------------------------------------------------------------------------
# Message / tool / response format conversion
# ---------------------------------------------------------------------------


def normalize_model_name(model: str) -> str:
    """Normalize a model name for the Anthropic API.

    - Strips 'anthropic/' prefix (OpenRouter format)
    """
    if model.startswith("anthropic/"):
        model = model[len("anthropic/"):]
    return model


def convert_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return []
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def convert_messages_to_anthropic(
    messages: List[Dict],
) -> Tuple[Optional[Any], List[Dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    System messages are extracted since Anthropic takes them as a separate param.
    system_prompt is a string or list of content blocks (when cache_control present).
    """
    system = None
    result = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            if isinstance(content, list):
                # Preserve cache_control markers on content blocks
                has_cache = any(
                    p.get("cache_control") for p in content if isinstance(p, dict)
                )
                if has_cache:
                    system = [p for p in content if isinstance(p, dict)]
                else:
                    system = "\n".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
            else:
                system = content
            continue

        if role == "assistant":
            blocks = []
            if content:
                text = content if isinstance(content, str) else json.dumps(content)
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else args
                except (json.JSONDecodeError, ValueError):
                    parsed_args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": parsed_args,
                })
            # Anthropic rejects empty assistant content
            effective = blocks or content
            if not effective or effective == "":
                effective = [{"type": "text", "text": "(empty)"}]
            result.append({"role": "assistant", "content": effective})
            continue

        if role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": content if isinstance(content, str) else json.dumps(content),
            }
            # Merge consecutive tool results into one user message
            if (
                result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], list)
                and result[-1]["content"]
                and result[-1]["content"][0].get("type") == "tool_result"
            ):
                result[-1]["content"].append(tool_result)
            else:
                result.append({"role": "user", "content": [tool_result]})
            continue

        # Regular user message
        result.append({"role": "user", "content": content})

    # Strip orphaned tool_use blocks (no matching tool_result follows)
    tool_result_ids = set()
    for m in result:
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_result":
                    tool_result_ids.add(block.get("tool_use_id"))
    for m in result:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            m["content"] = [
                b
                for b in m["content"]
                if b.get("type") != "tool_use" or b.get("id") in tool_result_ids
            ]
            if not m["content"]:
                m["content"] = [{"type": "text", "text": "(tool call removed)"}]

    # Enforce strict role alternation (Anthropic rejects consecutive same-role messages)
    fixed = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            if m["role"] == "user":
                # Merge consecutive user messages
                prev_content = fixed[-1]["content"]
                curr_content = m["content"]
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    fixed[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    fixed[-1]["content"] = prev_content + curr_content
                else:
                    # Mixed types — wrap string in list
                    if isinstance(prev_content, str):
                        prev_content = [{"type": "text", "text": prev_content}]
                    if isinstance(curr_content, str):
                        curr_content = [{"type": "text", "text": curr_content}]
                    fixed[-1]["content"] = prev_content + curr_content
            else:
                # Consecutive assistant messages — merge text content
                prev_blocks = fixed[-1]["content"]
                curr_blocks = m["content"]
                if isinstance(prev_blocks, list) and isinstance(curr_blocks, list):
                    fixed[-1]["content"] = prev_blocks + curr_blocks
                elif isinstance(prev_blocks, str) and isinstance(curr_blocks, str):
                    fixed[-1]["content"] = prev_blocks + "\n" + curr_blocks
                else:
                    # Keep the later message
                    fixed[-1] = m
        else:
            fixed.append(m)
    result = fixed

    return system, result


def build_anthropic_kwargs(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    max_tokens: Optional[int],
    reasoning_config: Optional[Dict[str, Any]],
    tool_choice: Optional[str] = None,
) -> Dict[str, Any]:
    """Build kwargs for anthropic.messages.create()."""
    system, anthropic_messages = convert_messages_to_anthropic(messages)
    anthropic_tools = convert_tools_to_anthropic(tools) if tools else []

    model = normalize_model_name(model)
    effective_max_tokens = max_tokens or 16384

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": effective_max_tokens,
    }

    if system:
        kwargs["system"] = system

    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
        # Map OpenAI tool_choice to Anthropic format
        if tool_choice == "auto" or tool_choice is None:
            kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            pass  # Don't send tool_choice — Anthropic will use tools if needed
        elif isinstance(tool_choice, str):
            # Specific tool name
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

    # Map reasoning_config to Anthropic's thinking parameter
    # Newer models (4.6+) prefer "adaptive" thinking; older models use "enabled"
    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is not False:
            effort = reasoning_config.get("effort", "medium")
            budget = THINKING_BUDGET.get(effort, 8000)
            # Use adaptive thinking for 4.5+ models (they deprecate type=enabled)
            if any(v in model for v in ("4-6", "4-5", "4.6", "4.5")):
                kwargs["thinking"] = {"type": "adaptive", "budget_tokens": budget}
            else:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = max(effective_max_tokens, budget + 4096)

    return kwargs


def normalize_anthropic_response(
    response,
) -> Tuple[SimpleNamespace, str]:
    """Normalize Anthropic response to match the shape expected by AIAgent.

    Returns (assistant_message, finish_reason) where assistant_message has
    .content, .tool_calls, and .reasoning attributes.
    """
    text_parts = []
    reasoning_parts = []
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            reasoning_parts.append(block.thinking)
        elif block.type == "tool_use":
            tool_calls.append(
                SimpleNamespace(
                    id=block.id,
                    type="function",
                    function=SimpleNamespace(
                        name=block.name,
                        arguments=json.dumps(block.input),
                    ),
                )
            )

    # Map Anthropic stop_reason to OpenAI finish_reason
    stop_reason_map = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    finish_reason = stop_reason_map.get(response.stop_reason, "stop")

    return (
        SimpleNamespace(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=None,
        ),
        finish_reason,
    )
