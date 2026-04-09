"""
Amazon Bedrock Converse API adapter for Hermes Agent.

Translates between Hermes' internal OpenAI-format messages and the
Bedrock Converse API, using boto3 with Bearer token authentication
(``AWS_BEARER_TOKEN_BEDROCK``).

Architecture:
  - ``build_bedrock_client()``      → boto3 bedrock-runtime client
  - ``build_converse_kwargs()``     → Converse API request dict
  - ``normalize_converse_response()``→ SimpleNamespace matching AIAgent expectations
  - ``convert_tools_to_converse()`` → tool schema translation
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def build_bedrock_client(region: str = "", base_url: str = ""):
    """Build a boto3 bedrock-runtime client.

    Authentication uses ``AWS_BEARER_TOKEN_BEDROCK`` (Bearer token) which
    boto3 picks up automatically.  Falls back to standard AWS credential
    chain (access key, instance role, etc.) if the env var is unset.

    Args:
        region:   AWS region (default from ``AWS_BEDROCK_REGION`` or ``us-east-1``).
        base_url: Full endpoint URL override (extracted from resolved base_url).
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for Amazon Bedrock support. "
            "Install it with: pip install boto3"
        )

    region = (
        region
        or os.getenv("AWS_BEDROCK_REGION", "").strip()
    )
    if not region:
        try:
            from hermes_cli.config import get_env_value
            region = (get_env_value("AWS_BEDROCK_REGION") or "").strip()
        except Exception:
            pass
    if not region:
        region = "us-east-1"

    kwargs: dict[str, Any] = {
        "service_name": "bedrock-runtime",
        "region_name": region,
    }

    # If user provided a full endpoint, extract region from it and use
    # the root domain as endpoint_url (boto3 expects the bare endpoint,
    # not /openai/v1).
    if base_url:
        # base_url is like https://bedrock-runtime.eu-central-1.amazonaws.com/openai/v1
        # boto3 needs     https://bedrock-runtime.eu-central-1.amazonaws.com
        import re
        match = re.search(r"(https://bedrock-runtime\.[^/]+)", base_url)
        if match:
            kwargs["endpoint_url"] = match.group(1)
            # Also extract region from URL
            region_match = re.search(r"bedrock-runtime\.([^.]+)\.", base_url)
            if region_match:
                kwargs["region_name"] = region_match.group(1)

    return boto3.client(**kwargs)


# ---------------------------------------------------------------------------
# Message conversion: OpenAI → Converse
# ---------------------------------------------------------------------------

# Converse-supported image formats
_MIME_TO_FORMAT = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


def _fetch_image_as_bytes(url: str, timeout: float = 10.0) -> Optional[Tuple[str, bytes]]:
    """Fetch an image URL and return ``(format, raw_bytes)``.

    Returns *None* on any failure (network error, unsupported format, too large).
    Bedrock Converse accepts jpeg, png, gif, webp up to ~20 MB.
    """
    import logging
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "").lower().split(";")[0].strip()
            data = resp.read(25 * 1024 * 1024)  # cap at 25 MB

            # Determine format from Content-Type first, then file extension
            fmt = _MIME_TO_FORMAT.get(content_type)
            if not fmt:
                # Try extension
                from urllib.parse import urlparse
                path = urlparse(url).path.lower()
                for ext, f in {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.items():
                    if path.endswith(f".{ext}"):
                        fmt = f
                        break
            if not fmt:
                logging.getLogger(__name__).debug(
                    "Unsupported image type %r from %s", content_type, url[:120],
                )
                return None
            return fmt, data
    except Exception as exc:
        logging.getLogger(__name__).debug("Failed to fetch image %s: %s", url[:120], exc)
        return None


def _convert_content_to_converse(content: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI message content to Converse content blocks."""
    if content is None:
        return []

    if isinstance(content, str):
        return [{"text": content}] if content else []

    if isinstance(content, list):
        blocks = []
        for part in content:
            if isinstance(part, str):
                blocks.append({"text": part})
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    text = part.get("text", "")
                    if text:
                        blocks.append({"text": text})
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                    if url.startswith("data:"):
                        # Base64 inline image: data:image/png;base64,<data>
                        import re
                        match = re.match(r"data:(image/\w+);base64,(.+)", url, re.DOTALL)
                        if match:
                            media_type = match.group(1)
                            data = match.group(2)
                            fmt = media_type.split("/")[1]
                            if fmt == "jpg":
                                fmt = "jpeg"
                            import base64
                            blocks.append({
                                "image": {
                                    "format": fmt,
                                    "source": {
                                        "bytes": base64.b64decode(data),
                                    },
                                },
                            })
                    elif url.startswith(("http://", "https://")):
                        # Converse doesn't accept image URLs — fetch and inline.
                        fetched = _fetch_image_as_bytes(url)
                        if fetched:
                            fmt, img_bytes = fetched
                            blocks.append({
                                "image": {
                                    "format": fmt,
                                    "source": {"bytes": img_bytes},
                                },
                            })
                        else:
                            logging.getLogger(__name__).warning(
                                "Bedrock Converse: could not fetch image from %s — skipping",
                                url[:120],
                            )
        return blocks

    return [{"text": str(content)}]


def _convert_tool_calls_to_converse(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool_calls to Converse toolUse blocks."""
    blocks = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}
        blocks.append({
            "toolUse": {
                "toolUseId": tc.get("id", ""),
                "name": name,
                "input": args,
            }
        })
    return blocks


def convert_messages_to_converse(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert OpenAI-format messages to Converse API format.

    Returns ``(system_blocks, converse_messages)`` where:
      - system_blocks: list of ``{"text": "..."}`` for the system prompt
      - converse_messages: list of ``{"role": ..., "content": [...]}``
    """
    system_blocks: List[Dict[str, Any]] = []
    converse_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            text = content if isinstance(content, str) else ""
            if isinstance(content, list):
                text = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            if text:
                system_blocks.append({"text": text})
            continue

        if role == "assistant":
            blocks = _convert_content_to_converse(content)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                blocks.extend(_convert_tool_calls_to_converse(tool_calls))
            if blocks:
                converse_messages.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            result_content = content if isinstance(content, str) else json.dumps(content)
            tool_result = {
                "toolResult": {
                    "toolUseId": tool_call_id,
                    "content": [{"text": result_content}],
                }
            }
            # Converse requires tool results in a "user" role message.
            # Merge consecutive tool results into one user message.
            if (
                converse_messages
                and converse_messages[-1]["role"] == "user"
                and any("toolResult" in b for b in converse_messages[-1]["content"])
            ):
                converse_messages[-1]["content"].append(tool_result)
            else:
                converse_messages.append({"role": "user", "content": [tool_result]})
            continue

        if role == "user":
            blocks = _convert_content_to_converse(content)
            if blocks:
                converse_messages.append({"role": "user", "content": blocks})
            continue

    # Converse requires alternating user/assistant roles.
    # Merge consecutive same-role messages.
    merged: List[Dict[str, Any]] = []
    for msg in converse_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)

    return system_blocks, merged


# ---------------------------------------------------------------------------
# Tool schema conversion: OpenAI → Converse
# ---------------------------------------------------------------------------

def convert_tools_to_converse(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert OpenAI tool definitions to Converse toolConfig format.

    Input:  [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
    Output: {"tools": [{"toolSpec": {"name": ..., "description": ..., "inputSchema": {"json": ...}}}]}
    """
    if not tools:
        return {}

    converse_tools = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {"type": "object", "properties": {}})

        converse_tools.append({
            "toolSpec": {
                "name": name,
                "description": description or name,
                "inputSchema": {
                    "json": parameters,
                },
            }
        })

    return {"tools": converse_tools}


# ---------------------------------------------------------------------------
# Build Converse kwargs
# ---------------------------------------------------------------------------

def build_converse_kwargs(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Build kwargs dict for ``bedrock_client.converse(**kwargs)``.

    Args:
        model:      Bedrock model ID (e.g. ``anthropic.claude-opus-4-6-v1``).
        messages:   OpenAI-format messages list.
        tools:      OpenAI-format tool definitions (or None).
        max_tokens: Output token limit.
    """
    system_blocks, converse_messages = convert_messages_to_converse(messages)

    kwargs: Dict[str, Any] = {
        "modelId": model,
        "messages": converse_messages,
    }

    if system_blocks:
        kwargs["system"] = system_blocks

    inference_config: Dict[str, Any] = {}
    if max_tokens:
        inference_config["maxTokens"] = max_tokens
    if inference_config:
        kwargs["inferenceConfig"] = inference_config

    if tools:
        kwargs["toolConfig"] = convert_tools_to_converse(tools)

    return kwargs


# ---------------------------------------------------------------------------
# Response normalization: Converse → OpenAI-like SimpleNamespace
# ---------------------------------------------------------------------------

def normalize_converse_response(
    response: Dict[str, Any],
) -> Tuple[SimpleNamespace, str]:
    """Normalize a Converse API response to the shape AIAgent expects.

    Returns ``(assistant_message, finish_reason)`` where assistant_message has:
      - ``.content``   — text string or None
      - ``.tool_calls`` — list of tool call SimpleNamespaces or None
      - ``.reasoning``  — None (Converse doesn't expose reasoning)
      - ``.reasoning_content`` — None
      - ``.reasoning_details`` — None
    """
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    text_parts: List[str] = []
    tool_calls: List[SimpleNamespace] = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                SimpleNamespace(
                    id=tu.get("toolUseId", ""),
                    type="function",
                    function=SimpleNamespace(
                        name=tu.get("name", ""),
                        arguments=json.dumps(tu.get("input", {})),
                    ),
                )
            )

    # Map Converse stopReason to OpenAI finish_reason
    stop_reason = response.get("stopReason", "end_turn")
    stop_map = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
    }
    finish_reason = stop_map.get(stop_reason, "stop")

    # Extract usage info
    usage = response.get("usage", {})
    usage_ns = SimpleNamespace(
        prompt_tokens=usage.get("inputTokens", 0),
        completion_tokens=usage.get("outputTokens", 0),
        total_tokens=usage.get("totalTokens", 0),
    )

    return (
        SimpleNamespace(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        ),
        finish_reason,
    )
