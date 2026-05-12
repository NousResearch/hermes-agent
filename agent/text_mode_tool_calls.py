"""
text_mode_tool_calls.py — Helper for synthesizing native-shaped tool_calls
from text-mode JSON content emitted by free-tier models that don't return
native OpenAI tool_calls structures (groq, cerebras, mistral, qwen).

Sprint D — root-cause fix for the agent rejecting groq's text-mode tool calls
and falling through to claude-sonnet (paid) in the F1 chain.

Public API:
    maybe_synthesize_tool_calls(message, model: str) -> bool
        Inspect message.content; if it contains a recognizable tool-call
        pattern AND the model is in the allowlist, attach synthesized
        SimpleNamespace tool_calls duck-typed to match the OpenAI
        ChatCompletionMessageToolCall structure. Returns True iff modified.

Detection patterns (in priority order, first match wins):
    1. <tool_call>{"name":...,"arguments":{...}}</tool_call>  (groq llama)
    2. ```json\n{"name":...,"arguments":...}\n```             (fenced)
    3. ```\n{"tool":"X","args":{...}}\n```                    (alt fenced)
    4. Bare top-level JSON if content IS that JSON (no prose)

Guards:
    - Allowlist by model prefix — only synthesize for known text-mode emitters
      (groq/, cerebras/, mistral/, qwen/, ollama/). Anthropic/OpenAI models
      that don't emit tool_calls did so intentionally — don't second-guess.
    - Idempotence — if message.tool_calls already has entries, return False.
    - False-positive guard — content with substantial natural-language prose
      around fenced blocks is treated cautiously; bare-JSON only fires when
      content is essentially the JSON alone.
"""
from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any, List, Optional

# Models known to emit text-mode JSON tool calls instead of native tool_calls.
# Add prefixes here as new free-tier providers surface the same pattern.
TEXT_MODE_MODEL_PREFIXES = (
    "groq/",
    "groq-",          # groq-fast, groq-llama aliases
    "cerebras/",
    "cerebras-",
    "mistral/",
    "mistral-small",
    "mistral-large",
    "mistral-medium",
    "qwen/",
    "qwen-",
    "qwen2-",
    "qwen3-",
    "ollama/",
    "meta-llama/",         # direct Meta-Llama route
    "openrouter/qwen",
    "openrouter/mistral",
    "openrouter/google",   # gemini-via-OR can also fall into this pattern
    "openrouter/meta-llama",   # OR Llama free tier — same family as groq Llama
    "openrouter/microsoft",    # Phi-4 via OR
    "openrouter/nousresearch", # Hermes-via-OR
)

# Hard cap on content length to prevent ReDoS / pathological regex on
# untrusted LLM responses. Any legitimate tool-call response is small.
MAX_CONTENT_LEN = 64 * 1024  # 64KB

# Compiled patterns
# 1. <tool_call>...</tool_call> wrapped tool calls (groq llama 3.3 style)
_TC_TAG_RE = re.compile(
    r"<tool_call\s*>\s*(\{.*?\})\s*</tool_call\s*>",
    re.DOTALL | re.IGNORECASE,
)

# 2. Markdown-fenced JSON: ```json {...} ```
_FENCED_JSON_RE = re.compile(
    r"```(?:json|JSON|tool_call)?\s*\n?(\{.*?\})\s*\n?```",
    re.DOTALL,
)

# 3. Bare top-level JSON (content is essentially the JSON alone — strict)
# Matches either an object {...} or a list [...] of tool calls.
_BARE_JSON_RE = re.compile(r"^\s*([\[\{].*[\]\}])\s*$", re.DOTALL)


def _model_is_text_mode(model: Optional[str]) -> bool:
    """Return True if the model is in our known-text-mode allowlist."""
    if not model or not isinstance(model, str):
        return False
    m = model.lower().strip()
    return any(m.startswith(p.lower()) for p in TEXT_MODE_MODEL_PREFIXES)


def _normalize_to_tool_call_shape(obj: Any) -> Optional[dict]:
    """Take any plausible tool-call dict shape and normalize to {name, arguments}.

    Accepted input shapes:
        {"name": "X", "arguments": {...}}   (OpenAI-canonical)
        {"name": "X", "arguments": "..."}   (OpenAI-canonical, args as JSON string)
        {"tool": "X", "args": {...}}        (Mistral/Qwen common)
        {"tool_name": "X", "tool_input": {...}}   (Anthropic-style if leaked)
        {"function": "X", "parameters": {...}}    (alt)
    Returns dict {"name": str, "arguments": str (json-encoded)} or None.
    """
    if not isinstance(obj, dict):
        return None

    name = (
        obj.get("name")
        or obj.get("tool")
        or obj.get("tool_name")
        or obj.get("function")
    )
    args = (
        obj.get("arguments")
        if "arguments" in obj
        else obj.get("args")
        if "args" in obj
        else obj.get("tool_input")
        if "tool_input" in obj
        else obj.get("parameters")
    )

    if not isinstance(name, str) or not name.strip():
        return None
    if args is None:
        args = {}

    # Normalize args to a JSON string (the OpenAI canonical shape).
    if isinstance(args, str):
        # If it parses, re-serialize; if not, keep as-is.
        try:
            parsed = json.loads(args)
            args_str = json.dumps(parsed, separators=(",", ":"))
        except Exception:
            args_str = args
    else:
        try:
            args_str = json.dumps(args, separators=(",", ":"))
        except Exception:
            return None  # unencodable — skip

    return {"name": name.strip(), "arguments": args_str}


def _extract_candidates(content: str) -> List[dict]:
    """Extract zero or more tool-call dicts from content. Returns normalized
    {name, arguments} dicts. Tries patterns in priority order and merges
    matches (a model can emit multiple tool calls in one response)."""
    candidates: List[dict] = []
    if not isinstance(content, str) or not content.strip():
        return candidates

    # 1. <tool_call>...</tool_call> wrapped
    for m in _TC_TAG_RE.finditer(content):
        raw = m.group(1)
        try:
            parsed = json.loads(raw)
        except Exception:
            # Allow trailing-comma / single-quote variants — strict=False is
            # not enough; try a light repair pass.
            try:
                parsed = json.loads(raw.replace("'", '"'))
            except Exception:
                continue
        norm = _normalize_to_tool_call_shape(parsed)
        if norm:
            candidates.append(norm)

    # 2. Markdown-fenced JSON (only run if no <tool_call> matches — avoid
    # double-counting the same call when models wrap fenced inside tags)
    if not candidates:
        for m in _FENCED_JSON_RE.finditer(content):
            raw = m.group(1)
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            # Heuristic: fenced JSON is a tool call only if it has a
            # tool-shaped key. Skip arbitrary structured output.
            if not isinstance(parsed, dict):
                continue
            if not any(k in parsed for k in ("name", "tool", "tool_name", "function")):
                continue
            norm = _normalize_to_tool_call_shape(parsed)
            if norm:
                candidates.append(norm)

    # 3. Bare top-level JSON — content IS the JSON, no prose
    if not candidates:
        m = _BARE_JSON_RE.match(content)
        if m:
            raw = m.group(1)
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                if any(k in parsed for k in ("name", "tool", "tool_name", "function")):
                    norm = _normalize_to_tool_call_shape(parsed)
                    if norm:
                        candidates.append(norm)
                # Also: top-level list of tool calls
            elif isinstance(parsed, list):
                for item in parsed:
                    norm = _normalize_to_tool_call_shape(item)
                    if norm:
                        candidates.append(norm)

    return candidates


def _make_tool_call_object(name: str, args_str: str, idx: int, content_hash: str) -> SimpleNamespace:
    """Build a SimpleNamespace duck-typed to match OpenAI's ChatCompletionMessageToolCall.
    The agent dispatcher reads .id, .call_id, .response_item_id, .type,
    .function.name, .function.arguments.
    """
    # Deterministic ID — derived from content+name+idx so retries hash the same
    call_id = "call_synth_" + hashlib.sha1(
        f"{content_hash}:{name}:{idx}".encode("utf-8")
    ).hexdigest()[:16]
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        response_item_id=None,  # only used by Codex Responses API
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=args_str,
        ),
    )


def maybe_synthesize_tool_calls(message: Any, model: Optional[str]) -> bool:
    """Inspect `message`; if appropriate, attach synthesized tool_calls.

    Args:
        message: a ChatCompletionMessage-shaped object (has .tool_calls,
                 .content; supports getattr/setattr or dict-like .get).
        model:   the LiteLLM model name that produced this response.

    Returns:
        True iff message.tool_calls was modified (synthesis happened).
        False otherwise (already had tool_calls, model not in allowlist,
        no candidates found in content).
    """
    # Guard 1: model allowlist
    if not _model_is_text_mode(model):
        return False

    # Guard 2: idempotence — don't double-fire if real tool_calls present
    existing = getattr(message, "tool_calls", None)
    if existing:
        return False

    # Read content (handles both attr-style and dict-style messages)
    if hasattr(message, "content"):
        content = getattr(message, "content", None)
    elif isinstance(message, dict):
        content = message.get("content")
    else:
        return False

    if not isinstance(content, str) or not content.strip():
        return False

    # ReDoS guard: skip pathologically large content. Tool-call JSON is small.
    if len(content) > MAX_CONTENT_LEN:
        return False

    # Extract candidates
    candidates = _extract_candidates(content)
    if not candidates:
        return False

    # Build SimpleNamespace tool_call objects
    content_hash = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:8]
    synthesized: List[SimpleNamespace] = []
    for idx, cand in enumerate(candidates):
        tc = _make_tool_call_object(
            name=cand["name"],
            args_str=cand["arguments"],
            idx=idx,
            content_hash=content_hash,
        )
        synthesized.append(tc)

    # Attach. If message is dict-like, also set the field.
    try:
        setattr(message, "tool_calls", synthesized)
    except Exception:
        if isinstance(message, dict):
            message["tool_calls"] = synthesized
        else:
            return False

    # Mark provenance — useful for observability and to avoid double-synthesis
    try:
        setattr(message, "_synthesized_tool_calls", True)
        setattr(message, "_synthesized_format_count", len(synthesized))
    except Exception:
        pass

    return True
