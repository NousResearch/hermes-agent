"""OpenAI-style -> Anthropic Messages API request conversion.

Everything here rewrites *request payloads*: model-id normalization, tool
schemas, and the message list (content blocks, thinking blocks and their
signatures, tool_use/tool_result pairing, cache_control placement, screenshot
eviction, blank-block scrubbing).

Split out of ``agent/anthropic_adapter.py`` so the adapter keeps client
construction and the API call itself, while the payload-shaping rules - by far
the largest and most fiddly part - have their own home. The endpoint-family
predicates a few of these rules branch on come from
``agent/anthropic_endpoints.py``, so this module never imports the adapter and
there is no import cycle.

``agent.anthropic_adapter`` re-exports every name below, so existing
``from agent.anthropic_adapter import convert_messages_to_anthropic`` imports
keep working.
"""

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.anthropic_endpoints import (
    _is_deepseek_anthropic_endpoint,
    _is_kimi_family_endpoint,
    _is_nous_portal_endpoint,
    _is_third_party_anthropic_endpoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message / tool / response format conversion
# ---------------------------------------------------------------------------


def _is_bedrock_model_id(model: str) -> bool:
    """Detect AWS Bedrock model IDs that use dots as namespace separators.

    Bedrock model IDs come in two forms:
    - Bare:    ``anthropic.claude-opus-4-7``
    - Regional (inference profiles): ``us.anthropic.claude-sonnet-4-5-v1:0``

    In both cases the dots separate namespace components, not version
    numbers, and must be preserved verbatim for the Bedrock API.
    """
    lower = model.lower()
    # Regional inference-profile prefixes
    if any(lower.startswith(p) for p in (
        "global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.",
        "ca.", "sa.", "me.", "af.",
    )):
        return True
    # Bare Bedrock model IDs: provider.model-family
    if lower.startswith("anthropic."):
        return True
    return False


def normalize_model_name(model: str, preserve_dots: bool = False) -> str:
    """Normalize a model name for the Anthropic API.

    - Strips 'anthropic/' prefix (OpenRouter format, case-insensitive)
    - Converts dots to hyphens in version numbers (OpenRouter uses dots,
      Anthropic uses hyphens: claude-opus-4.6 → claude-opus-4-6), unless
      preserve_dots is True (e.g. for Alibaba/DashScope: qwen3.5-plus).
    - Preserves Bedrock model IDs (``anthropic.claude-opus-4-7``) and
      regional inference profiles (``us.anthropic.claude-*``) whose dots
      are namespace separators, not version separators.
    """
    lower = model.lower()
    if lower.startswith("anthropic/"):
        model = model[len("anthropic/"):]
    if not preserve_dots:
        # Bedrock model IDs use dots as namespace separators
        # (e.g. "anthropic.claude-opus-4-7", "us.anthropic.claude-*").
        # These must not be converted to hyphens.  See issue #12295.
        if _is_bedrock_model_id(model):
            return model
        # Only convert dots to hyphens for Anthropic/Claude models.
        # Non-Anthropic models (gpt-5.4, gemini-2.5, etc.) use dots
        # as part of their canonical names.  See issue #17171.
        _lower = model.lower()
        if _lower.startswith("claude-") or _lower.startswith("anthropic/"):
            model = model.replace(".", "-")
    return model


def _sanitize_tool_id(tool_id: str) -> str:
    """Sanitize a tool call ID for the Anthropic API.

    Anthropic requires IDs matching [a-zA-Z0-9_-]. Replace invalid
    characters with underscores and ensure non-empty.
    """
    import re
    if not tool_id:
        return "tool_0"
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
    return sanitized or "tool_0"


def _normalize_tool_input_schema(schema: Any) -> Dict[str, Any]:
    """Normalize tool schemas before sending them to Anthropic.

    Anthropic's tool schema validator rejects nullable unions such as
    ``anyOf: [{"type": "string"}, {"type": "null"}]`` that Pydantic/MCP
    commonly emits for optional fields. Tool optionality is represented by
    the parent ``required`` array, so we delegate to the shared
    ``strip_nullable_unions`` helper to collapse nullable unions to the
    non-null branch while preserving metadata like description/default.

    ``keep_nullable_hint=False`` because the Anthropic validator does not
    recognize the OpenAPI-style ``nullable: true`` extension and strict
    schema-to-grammar converters may reject unknown keywords.

    Top-level ``oneOf``/``allOf``/``anyOf`` are also stripped here: the
    Anthropic API rejects union keywords at the schema root with a generic
    HTTP 400. Several upstream and plugin tools ship schemas with one of
    these keywords at the top level (commonly for Pydantic discriminated
    unions). If we land here with those keywords still present after
    nullable-union stripping, drop them and fall back to a plain object
    schema so the tool still validates at the Anthropic boundary.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    from tools.schema_sanitizer import strip_nullable_unions

    normalized = strip_nullable_unions(schema, keep_nullable_hint=False)
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}
    # Strip top-level union keywords that Anthropic's validator rejects.
    banned = {"oneOf", "allOf", "anyOf"}
    if banned & normalized.keys():
        normalized = {k: v for k, v in normalized.items() if k not in banned}
        if "type" not in normalized:
            normalized["type"] = "object"
    if normalized.get("type") == "object" and not isinstance(normalized.get("properties"), dict):
        normalized = {**normalized, "properties": {}}
    return normalized


def convert_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return []
    result = []
    seen_names: set = set()
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "")
        # Defensive dedup: Anthropic rejects requests with duplicate tool
        # names.  Upstream injection paths already dedup, but this guard
        # converts a hard API failure into a warning.  See: #18478
        if name and name in seen_names:
            logger.warning(
                "convert_tools_to_anthropic: duplicate tool name '%s' "
                "— dropping second occurrence",
                name,
            )
            continue
        if name:
            seen_names.add(name)
        anthropic_tool: Dict[str, Any] = {
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": _normalize_tool_input_schema(
                fn.get("parameters", {"type": "object", "properties": {}})
            ),
        }
        # Forward cache_control marker when present on the OpenAI-format
        # tool dict. Anthropic's tools array supports cache_control on the
        # last tool to cache the entire schema cross-session.
        cache_control = t.get("cache_control")
        if isinstance(cache_control, dict):
            anthropic_tool["cache_control"] = dict(cache_control)
        result.append(anthropic_tool)
    return result


def _image_source_from_openai_url(url: str) -> Dict[str, str]:
    """Convert an OpenAI-style image URL/data URL into Anthropic image source."""
    url = str(url or "").strip()
    if not url:
        return {"type": "url", "url": ""}

    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                media_type = mime_part
        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }

    return {"type": "url", "url": url}


def _convert_content_part_to_anthropic(part: Any) -> Optional[Dict[str, Any]]:
    """Convert a single OpenAI-style content part to Anthropic format."""
    if part is None:
        return None
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    ptype = part.get("type")

    if ptype == "input_text":
        block: Dict[str, Any] = {"type": "text", "text": part.get("text", "")}
    elif ptype == "text":
        # A stored Anthropic text block. Rebuild from whitelisted fields only —
        # SDK response text blocks carry output-only siblings (parsed_output,
        # citations=None) that the Messages INPUT schema rejects with HTTP 400
        # "Extra inputs are not permitted". Do NOT dict(part) it verbatim.
        block = {"type": "text", "text": part.get("text", "")}
        cits = part.get("citations")
        if isinstance(cits, list) and cits:
            block["citations"] = cits
    elif ptype in {"image_url", "input_image"}:
        image_value = part.get("image_url", {})
        url = image_value.get("url", "") if isinstance(image_value, dict) else str(image_value or "")
        block = {"type": "image", "source": _image_source_from_openai_url(url)}
    else:
        block = dict(part)

    if isinstance(part.get("cache_control"), dict) and "cache_control" not in block:
        block["cache_control"] = dict(part["cache_control"])
    return block


def _to_plain_data(value: Any, *, _depth: int = 0, _path: Optional[set] = None) -> Any:
    """Recursively convert SDK objects to plain Python data structures.

    Guards against circular references (``_path`` tracks ``id()`` of objects
    on the *current* recursion path) and runaway depth (capped at 20 levels).
    Uses path-based tracking so shared (but non-cyclic) objects referenced by
    multiple siblings are converted correctly rather than being stringified.
    """
    _MAX_DEPTH = 20
    if _depth > _MAX_DEPTH:
        return str(value)

    if _path is None:
        _path = set()

    obj_id = id(value)
    if obj_id in _path:
        return str(value)

    if hasattr(value, "model_dump"):
        _path.add(obj_id)
        try:
            # warnings=False: content blocks from the streaming accumulator
            # (ParsedTextBlock et al.) trip pydantic's serializer-mismatch
            # UserWarning against the generic Message union; the dump itself
            # is correct, and the warning leaks to the user's terminal.
            dumped = value.model_dump(warnings=False)
        except TypeError:
            # Duck-typed model_dump without pydantic's signature.
            dumped = value.model_dump()
        result = _to_plain_data(dumped, _depth=_depth + 1, _path=_path)
        _path.discard(obj_id)
        return result
    if isinstance(value, dict):
        _path.add(obj_id)
        result = {k: _to_plain_data(v, _depth=_depth + 1, _path=_path) for k, v in value.items()}
        _path.discard(obj_id)
        return result
    if isinstance(value, (list, tuple)):
        _path.add(obj_id)
        result = [_to_plain_data(v, _depth=_depth + 1, _path=_path) for v in value]
        _path.discard(obj_id)
        return result
    if hasattr(value, "__dict__"):
        _path.add(obj_id)
        result = {
            k: _to_plain_data(v, _depth=_depth + 1, _path=_path)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
        _path.discard(obj_id)
        return result
    return value


def _extract_preserved_thinking_blocks(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return Anthropic thinking blocks previously preserved on the message."""
    raw_details = message.get("reasoning_details")
    if not isinstance(raw_details, list):
        return []

    preserved: List[Dict[str, Any]] = []
    for detail in raw_details:
        if not isinstance(detail, dict):
            continue
        block_type = str(detail.get("type", "") or "").strip().lower()
        if block_type not in {"thinking", "redacted_thinking"}:
            continue
        preserved.append(copy.deepcopy(detail))
    return preserved


def _convert_content_to_anthropic(content: Any) -> Any:
    """Convert OpenAI-style multimodal content arrays to Anthropic blocks."""
    if not isinstance(content, list):
        return content

    converted = []
    for part in content:
        block = _convert_content_part_to_anthropic(part)
        if block is not None:
            converted.append(block)
    return converted


def _content_parts_to_anthropic_blocks(parts: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tool-message content parts → Anthropic tool_result inner blocks.

    Used for multimodal tool results (e.g. computer_use screenshots). Each
    part is normalized via `_convert_content_part_to_anthropic`, then
    filtered to the block types Anthropic tool_result accepts (text + image).
    """
    if not isinstance(parts, list):
        return []
    out: List[Dict[str, Any]] = []
    for part in parts:
        block = _convert_content_part_to_anthropic(part)
        if not block:
            continue
        btype = block.get("type")
        if btype == "text":
            text_val = block.get("text")
            if isinstance(text_val, str) and text_val:
                out.append({"type": "text", "text": text_val})
        elif btype == "image":
            src = block.get("source")
            if isinstance(src, dict) and src:
                out.append({"type": "image", "source": src})
    return out


_EMPTY_TEXT_PLACEHOLDER = "(empty)"


def _safe_text(text: Any) -> str:
    """Return ``text`` if it's non-whitespace, else a non-whitespace placeholder.

    The Anthropic Messages API rejects requests where a text content block is
    empty or whitespace-only (HTTP 400 "text content blocks must contain
    non-whitespace text"). When such a block gets stored in session history —
    e.g. produced by context compression — it is replayed verbatim on every
    subsequent turn, permanently wedging the session. Coercing to a
    non-whitespace placeholder is self-healing: the next API call recovers.

    Mirrors ``bedrock_adapter._safe_text`` (#9486); ref #69512.
    """
    if text is None:
        return _EMPTY_TEXT_PLACEHOLDER
    if not isinstance(text, str):
        text = str(text)
    return text if text.strip() else _EMPTY_TEXT_PLACEHOLDER


def _sanitize_replay_block(b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Strip output-only fields from a stored Anthropic content block so it is
    valid as REQUEST input on replay.

    The SDK response objects carry output-only attributes that the Messages
    *input* schema forbids ("Extra inputs are not permitted"): text blocks get
    ``parsed_output``/``citations`` (when null), tool_use blocks get ``caller``,
    etc. ``normalize_response`` captured blocks verbatim via ``_to_plain_data``,
    so these leak back as input on the next turn → HTTP 400.

    Whitelist per type (NOT a blacklist) so future SDK output-only fields can't
    reintroduce the bug. Returns a clean block, or None to drop it.
    """
    if not isinstance(b, dict):
        return None
    btype = b.get("type")
    if btype == "text":
        text_val = b.get("text", "")
        # Bedrock and strict Anthropic-compatible endpoints reject text
        # blocks where "text" is empty or whitespace-only (#69512). Drop the
        # blank block (the caller relocates any cache_control it carried and
        # falls back to a non-whitespace placeholder when nothing survives)
        # rather than coercing in place — a coerced "(empty)" block would be
        # model-visible noise next to surviving thinking/tool_use blocks.
        # Type-safe: captured blocks can carry text=None from an invalid
        # upstream payload, which a bare .strip() would crash on.
        if not isinstance(text_val, str) or not text_val.strip():
            return None
        out: Dict[str, Any] = {"type": "text", "text": text_val}
        # citations is input-valid ONLY when it's a non-empty list; the SDK
        # emits citations=None on responses, which the input schema rejects.
        cits = b.get("citations")
        if isinstance(cits, list) and cits:
            out["citations"] = cits
        if isinstance(b.get("cache_control"), dict):
            out["cache_control"] = b["cache_control"]
        return out
    if btype == "thinking":
        out = {"type": "thinking", "thinking": b.get("thinking", "")}
        if b.get("signature"):
            out["signature"] = b["signature"]
        return out
    if btype == "redacted_thinking":
        # Only valid with its data payload; drop if missing.
        return {"type": "redacted_thinking", "data": b["data"]} if b.get("data") else None
    if btype == "tool_use":
        out = {
            "type": "tool_use",
            "id": _sanitize_tool_id(b.get("id", "")),
            "name": b.get("name", ""),
            "input": b.get("input", {}),
        }
        if isinstance(b.get("cache_control"), dict):
            out["cache_control"] = b["cache_control"]
        return out
    if btype == "image":
        src = b.get("source")
        return {"type": "image", "source": src} if isinstance(src, dict) else None
    # Unknown/unsupported block type on the input path — drop rather than risk
    # another "Extra inputs are not permitted".
    return None


def _apply_assistant_cache_control_to_last_cacheable_block(
    blocks: List[Dict[str, Any]],
    cache_control: Any,
) -> None:
    if not isinstance(cache_control, dict):
        return
    for block in reversed(blocks):
        if isinstance(block, dict) and block.get("type") in {"text", "tool_use"}:
            block.setdefault("cache_control", dict(cache_control))
            break


def _convert_assistant_message(m: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an assistant message to Anthropic content blocks.

    Handles thinking blocks, regular content, tool calls, and
    reasoning_content injection for Kimi/DeepSeek endpoints.
    """
    content = m.get("content", "")
    # Anthropic interleaved-thinking fast path: when this turn carries a
    # verbatim, order-preserving block list (set by normalize_response only
    # for turns that interleave SIGNED thinking with tool_use), replay it.
    # Each block is run through _sanitize_replay_block to strip output-only
    # SDK fields (parsed_output, caller, citations=None, …) that the Messages
    # INPUT schema forbids — replaying them verbatim caused HTTP 400 "Extra
    # inputs are not permitted" (text.parsed_output). Block ORDER is preserved
    # (the reason this channel exists); only forbidden sibling fields are
    # dropped, leaving thinking signatures and tool_use id/name/input intact.
    ordered_blocks = m.get("anthropic_content_blocks")
    if isinstance(ordered_blocks, list) and ordered_blocks:
        # Re-source each tool_use input from the stored tool_calls map rather
        # than the captured block. The ordered-blocks list captures tool_use
        # input from the RAW API response (normalize_response), which is NOT
        # credential-redacted; tool_calls[].function.arguments IS redacted at
        # storage time (build_assistant_message, #19798). Replaying the raw
        # block input would resurrect a secret the model inlined into a tool
        # call (e.g. terminal(command="curl -H 'Authorization: Bearer sk-...'")
        # onto the wire, even though the same value is redacted everywhere else
        # in history. Keying by sanitized tool id preserves interleave order
        # (the reason this channel exists) while swapping in the redacted
        # input. Adapted from #36071 (replay-time tool-input re-sourcing).
        redacted_input_by_id: Dict[str, Any] = {}
        for tc in m.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, ValueError):
                parsed_args = {}
            redacted_input_by_id[_sanitize_tool_id(tc.get("id", ""))] = parsed_args
        replayed: List[Dict[str, Any]] = []
        _relocated_replay_cache_control = None
        _dropped_blank_text = False
        for b in ordered_blocks:
            clean = _sanitize_replay_block(b)
            if clean is None:
                if isinstance(b, dict) and b.get("type") == "text":
                    _dropped_blank_text = True
                if isinstance(b, dict) and isinstance(b.get("cache_control"), dict):
                    # A dropped blank text block can still carry the cache
                    # breakpoint marker -- relocate it rather than losing it.
                    _relocated_replay_cache_control = b["cache_control"]
                continue
            if clean.get("type") == "tool_use":
                # Override raw (un-redacted) input with the redacted copy when
                # we have one for this id; fall back to the sanitized block
                # input only if the tool_call is missing (shape mismatch).
                redacted = redacted_input_by_id.get(clean.get("id", ""))
                if redacted is not None:
                    clean["input"] = redacted
            replayed.append(clean)
        # When every text block was blank and nothing cacheable survived
        # (e.g. signed thinking + a blank text block, or a SOLE blank
        # cache-marked block), emit the non-whitespace placeholder so the
        # replayed message stays schema-valid (#69512) and a relocated cache
        # marker still has a carrier instead of being silently lost.
        _has_cacheable_replay = any(
            isinstance(b, dict) and b.get("type") in {"text", "tool_use"}
            for b in replayed
        )
        if not _has_cacheable_replay and (
            _dropped_blank_text or _relocated_replay_cache_control is not None
        ):
            replayed.append({"type": "text", "text": _EMPTY_TEXT_PLACEHOLDER})
        if replayed:
            if _relocated_replay_cache_control is not None:
                _apply_assistant_cache_control_to_last_cacheable_block(
                    replayed, _relocated_replay_cache_control
                )
            _apply_assistant_cache_control_to_last_cacheable_block(
                replayed, m.get("cache_control")
            )
            # apply_anthropic_cache_control marks an assistant turn with
            # non-empty text by writing cache_control INTO ``content`` (see
            # _apply_cache_marker's list branch), not at the top level. This
            # branch rebuilds the message from ordered_blocks and never reads
            # ``content``, so that marker would be dropped -- and because
            # _can_carry_marker already counted this message as a carrier, the
            # breakpoint is burned rather than relocated. #56195 covered the
            # complementary shape (blank content -> top-level marker); this is
            # the interleaved thinking + preamble-text + tool_use shape.
            _inline_cc = None
            _msg_content = m.get("content")
            if isinstance(_msg_content, list):
                for _blk in _msg_content:
                    if isinstance(_blk, dict) and isinstance(
                        _blk.get("cache_control"), dict
                    ):
                        _inline_cc = _blk["cache_control"]
                        break
            if _inline_cc is not None:
                _apply_assistant_cache_control_to_last_cacheable_block(
                    replayed, _inline_cc
                )
            return {"role": "assistant", "content": replayed}

    blocks = _extract_preserved_thinking_blocks(m)
    # Cache markers dropped along with a blank block are relocated onto the
    # last surviving cacheable block below (via
    # _apply_assistant_cache_control_to_last_cacheable_block), rather than
    # lost -- prompt_caching.py's _apply_cache_marker() sets cache_control
    # directly on content[-1] for list content, so if that last part happens
    # to be blank text, dropping it silently would lose the breakpoint.
    _relocated_cache_control = None
    if content:
        if isinstance(content, list):
            converted_content = _convert_content_to_anthropic(content)
            if isinstance(converted_content, list):
                # Bedrock and strict Anthropic-compatible endpoints reject
                # text blocks where "text" is empty or whitespace-only. The
                # ordered-replay path enforces the same invariant via
                # _sanitize_replay_block(). Type-safe against ANY invalid
                # "text" value from an upstream payload -- None, or a
                # truthy non-string like an int -- not just None: checking
                # isinstance() first (rather than `blk.get("text") or ""`)
                # means a non-string value is treated as blank/invalid
                # instead of reaching .strip() and raising AttributeError.
                for blk in converted_content:
                    _blk_text = blk.get("text") if isinstance(blk, dict) else None
                    if (
                        isinstance(blk, dict)
                        and blk.get("type") == "text"
                        and (not isinstance(_blk_text, str) or not _blk_text.strip())
                    ):
                        if isinstance(blk.get("cache_control"), dict):
                            _relocated_cache_control = blk["cache_control"]
                        continue
                    blocks.append(blk)
        else:
            # Scalar (non-list) content: a whitespace-only string is the
            # same invalid-payload case as an empty list block -- drop it
            # rather than emitting a blank text block.
            text_str = str(content)
            if text_str.strip():
                blocks.append({"type": "text", "text": text_str})
    for tc in m.get("tool_calls", []):
        if not tc or not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(args) if isinstance(args, str) else args
        except (json.JSONDecodeError, ValueError):
            parsed_args = {}
        blocks.append({
            "type": "tool_use",
            "id": _sanitize_tool_id(tc.get("id", "")),
            "name": fn.get("name", ""),
            "input": parsed_args,
        })
    # Kimi's /coding endpoint (Anthropic protocol) requires assistant
    # tool-call messages to carry reasoning_content when thinking is
    # enabled server-side.  Preserve it as a thinking block so Kimi
    # can validate the message history.  See hermes-agent#13848.
    #
    # Accept empty string "" — _copy_reasoning_content_for_api()
    # injects "" as a tier-3 fallback for Kimi tool-call messages
    # that had no reasoning.  Kimi requires the field to exist, even
    # if empty.
    #
    # Prepend (not append): Anthropic protocol requires thinking
    # blocks before text and tool_use blocks.
    #
    # Guard: only add when reasoning_details didn't already contribute
    # thinking blocks.  On native Anthropic, reasoning_details produces
    # signed thinking blocks — adding another unsigned one from
    # reasoning_content would create a duplicate (same text) that gets
    # downgraded to a spurious text block on the last assistant message.
    reasoning_content = m.get("reasoning_content")
    _already_has_thinking = any(
        isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
        for b in blocks
    )
    if isinstance(reasoning_content, str) and not _already_has_thinking:
        blocks.insert(0, {"type": "thinking", "thinking": reasoning_content})
    # Anthropic rejects empty assistant content. IMPORTANT: fall back only
    # to the placeholder, never to the raw `content` variable -- `content`
    # is the UNFILTERED original message content, and can itself be exactly
    # the blank/whitespace-only payload the filtering above just removed
    # (a sole blank text block, or scalar whitespace with no tool_calls).
    # `blocks or content` there would silently restore the invalid provider
    # payload this function exists to prevent (#69512).
    effective = blocks if blocks else [{"type": "text", "text": _EMPTY_TEXT_PLACEHOLDER}]
    # Applied here (after the empty-fallback resolution) rather than
    # earlier against `blocks` directly, so a cache_control relocated from
    # a dropped blank block that was the ONLY block still lands on the
    # (empty) placeholder instead of being silently lost when blocks was
    # empty at the point the marker would otherwise have been applied.
    if _relocated_cache_control is not None:
        _apply_assistant_cache_control_to_last_cacheable_block(
            effective, _relocated_cache_control
        )
    _apply_assistant_cache_control_to_last_cacheable_block(
        effective, m.get("cache_control")
    )
    return {"role": "assistant", "content": effective}


def _convert_tool_message_to_result(
    result: List[Dict[str, Any]], m: Dict[str, Any]
) -> None:
    """Convert a tool message to an Anthropic tool_result, merging consecutive
    results into one user message.

    Mutates ``result`` in place — either appends a new user message or extends
    the trailing user message's tool_result list.
    """
    content = m.get("content", "")
    multimodal_blocks: Optional[List[Dict[str, Any]]] = None
    if isinstance(content, dict) and content.get("_multimodal"):
        multimodal_blocks = _content_parts_to_anthropic_blocks(
            content.get("content") or []
        )
        # Fallback text if the conversion produced nothing usable.
        if not multimodal_blocks and content.get("text_summary"):
            multimodal_blocks = [
                {"type": "text", "text": str(content["text_summary"])}
            ]
    elif isinstance(content, list):
        converted = _content_parts_to_anthropic_blocks(content)
        if any(b.get("type") == "image" for b in converted):
            multimodal_blocks = converted
    # Back-compat: some callers stash blocks under a private key.
    if multimodal_blocks is None:
        stashed = m.get("_anthropic_content_blocks")
        if isinstance(stashed, list) and stashed:
            text_content = content if isinstance(content, str) and content.strip() else None
            multimodal_blocks = (
                [{"type": "text", "text": text_content}] + stashed
                if text_content else list(stashed)
            )

    if multimodal_blocks:
        result_content: Any = multimodal_blocks
    elif isinstance(content, str):
        result_content = content
    else:
        result_content = json.dumps(content) if content else "(no output)"
    if not result_content:
        result_content = "(no output)"
    tool_result = {
        "type": "tool_result",
        "tool_use_id": _sanitize_tool_id(m.get("tool_call_id", "")),
        "content": result_content,
    }
    if isinstance(m.get("cache_control"), dict):
        tool_result["cache_control"] = dict(m["cache_control"])
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


def _convert_user_message(content: Any) -> Dict[str, Any]:
    """Validate and convert a user message to anthropic format."""
    if isinstance(content, list):
        converted_blocks = _convert_content_to_anthropic(content)
        kept_blocks = _fix_blank_text_blocks_in_list(
            converted_blocks,
            placeholder_text="(empty message)",
            msg_index=-1,
            role="user",
            location="_convert_user_message",
        )
        return {"role": "user", "content": kept_blocks}
    else:
        if not content or (isinstance(content, str) and not content.strip()):
            content = "(empty message)"
        return {"role": "user", "content": content}


def _strip_orphaned_tool_blocks(result: List[Dict[str, Any]]) -> None:
    """Strip tool_use blocks with no matching tool_result, and vice versa.

    Context compression or session truncation can remove either side of a
    tool-call pair, or insert messages between a tool_use and its result.
    Anthropic requires each tool_use to have a matching tool_result in the
    IMMEDIATELY FOLLOWING user message — a global ID match is not enough.
    Mutates ``result`` in place.
    """
    # Pass 1: For each assistant message with tool_use blocks, check that
    # EACH tool_use ID has a matching tool_result in the immediately following
    # user message.  Strip tool_use blocks that lack an adjacent result —
    # Anthropic rejects non-adjacent pairs with HTTP 400 even when the IDs
    # match somewhere later in the conversation.
    for i, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        tool_use_ids_in_turn = {
            b.get("id")
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        if not tool_use_ids_in_turn:
            continue

        # Collect result IDs from the immediately following user message only.
        adjacent_result_ids: set = set()
        if i + 1 < len(result):
            nxt = result[i + 1]
            if nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
                for block in nxt["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        adjacent_result_ids.add(block.get("tool_use_id"))

        orphaned = tool_use_ids_in_turn - adjacent_result_ids
        if not orphaned:
            continue

        kept = [
            b
            for b in m["content"]
            if not (isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") in orphaned)
        ]
        # If stripping an orphaned tool_use mutated a turn that also carries a
        # signed thinking block, that block's Anthropic signature was computed
        # against the ORIGINAL (un-stripped) turn content and is now invalid.
        # Anthropic rejects the replayed turn with HTTP 400 "thinking blocks in
        # the latest assistant message cannot be modified".  Flag the turn so
        # _manage_thinking_signatures can demote the dead signature instead of
        # replaying it verbatim.  See hermes-agent: extended-thinking + parallel
        # tool batch interrupted mid-flight → non-retryable 400 crash-loop.
        if len(kept) != len(m["content"]) and any(
            isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
            for b in m["content"]
        ):
            m["_thinking_signature_invalidated"] = True
        m["content"] = kept if kept else [{"type": "text", "text": "(tool call removed)"}]

    # Pass 2: Rebuild the set of tool_use IDs that survived pass 1, then
    # strip tool_result blocks that no longer have any matching tool_use
    # anywhere in the conversation.
    surviving_tool_use_ids: set = set()
    for m in result:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    surviving_tool_use_ids.add(block.get("id"))

    for m in result:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            continue
        new_content = [
            b
            for b in m["content"]
            if not (isinstance(b, dict) and b.get("type") == "tool_result")
            or b.get("tool_use_id") in surviving_tool_use_ids
        ]
        if len(new_content) != len(m["content"]):
            m["content"] = new_content if new_content else [{"type": "text", "text": "(tool result removed)"}]


def _merge_consecutive_roles(result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive same-role messages to enforce Anthropic alternation.

    Returns a new list (caller must rebind ``result``).
    """
    fixed = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            if m["role"] == "user":
                prev_content = fixed[-1]["content"]
                curr_content = m["content"]
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    fixed[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    fixed[-1]["content"] = prev_content + curr_content
                else:
                    if isinstance(prev_content, str):
                        prev_content = [{"type": "text", "text": prev_content}]
                    if isinstance(curr_content, str):
                        curr_content = [{"type": "text", "text": curr_content}]
                    fixed[-1]["content"] = prev_content + curr_content
            else:
                # Consecutive assistant messages — merge text content.
                # Propagate the orphan-strip signature-invalidation flag onto the
                # surviving (prev) dict so _manage_thinking_signatures still sees it.
                if m.get("_thinking_signature_invalidated"):
                    fixed[-1]["_thinking_signature_invalidated"] = True
                # Drop thinking blocks from the *second* message: their
                # signature was computed against a different turn boundary
                # and becomes invalid once merged.
                if isinstance(m["content"], list):
                    m["content"] = [
                        b for b in m["content"]
                        if not (isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"})
                    ]
                prev_blocks = fixed[-1]["content"]
                curr_blocks = m["content"]
                if isinstance(prev_blocks, list) and isinstance(curr_blocks, list):
                    fixed[-1]["content"] = prev_blocks + curr_blocks
                elif isinstance(prev_blocks, str) and isinstance(curr_blocks, str):
                    fixed[-1]["content"] = prev_blocks + "\n" + curr_blocks
                else:
                    if isinstance(prev_blocks, str):
                        prev_blocks = [{"type": "text", "text": prev_blocks}]
                    if isinstance(curr_blocks, str):
                        curr_blocks = [{"type": "text", "text": curr_blocks}]
                    fixed[-1]["content"] = prev_blocks + curr_blocks
        else:
            fixed.append(m)
    return fixed


def _manage_thinking_signatures(
    result: List[Dict[str, Any]], base_url: str | None, model: str | None
) -> None:
    """Strip or preserve thinking blocks based on endpoint type.

    Anthropic signs thinking blocks against the full turn content.
    Any upstream mutation (context compression, session truncation, orphan
    stripping, message merging) invalidates the signature, causing HTTP 400
    "Invalid signature in thinking block".

    Signatures are Anthropic-proprietary.  Third-party endpoints (MiniMax,
    Azure AI Foundry, AWS Bedrock, self-hosted proxies) cannot validate them
    and will reject them outright.  Kimi's /coding and DeepSeek's /anthropic
    endpoints speak the Anthropic protocol upstream but require unsigned
    thinking blocks (synthesised from ``reasoning_content``) to round-trip on
    replayed assistant tool-call messages.  See hermes-agent#13848 (Kimi) and
    hermes-agent#16748 (DeepSeek).

    Nous Portal's ``/v1/messages`` route is the exception among third-party
    hosts: it proxies Claude to Anthropic/Vertex/Bedrock and validates the
    same signed thinking blocks.  Sticky ``session_id`` keeps a conversation
    on one upstream instance so those signatures stay warm — stripping them
    here would 400 the first tool-loop turn ("thinking must be passed back").
    Portal therefore takes the native Anthropic replay path below.

    Mutates ``result`` in place.
    """
    _THINKING_TYPES = frozenset(("thinking", "redacted_thinking"))
    # Portal speaks Anthropic's thinking contract end-to-end; do not treat it
    # as a signature-blind proxy even though the host is not anthropic.com.
    _is_third_party = (
        _is_third_party_anthropic_endpoint(base_url)
        and not _is_nous_portal_endpoint(base_url)
    )

    last_assistant_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    for idx, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue

        if _is_kimi_family_endpoint(base_url, model):
            # Kimi does not enforce thinking signatures — replay as-is
            # (shared cleanup below still strips cache markers + the internal flag).
            pass
        elif _is_deepseek_anthropic_endpoint(base_url):
            # DeepSeek: strip signed, preserve unsigned.
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if b.get("signature") or b.get("data"):
                    # Signed (or redacted-with-data) — upstream can't validate, strip.
                    continue
                new_content.append(b)
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]
        elif _is_third_party or idx != last_assistant_idx:
            # Third-party: strip ALL thinking blocks (signatures are proprietary).
            # Direct Anthropic: strip from non-latest assistant messages only.
            stripped = [
                b for b in m["content"]
                if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)
            ]
            m["content"] = stripped or [{"type": "text", "text": "(thinking elided)"}]
        else:
            # Latest assistant on direct Anthropic: keep signed, downgrade unsigned
            # to text so the reasoning isn't lost.
            #
            # Exception: if orphan-stripping (or another structural mutation) removed
            # a tool_use block from THIS turn, every thinking signature on it was
            # computed against the original turn content and is now dead.  Anthropic
            # rejects the turn either way — replaying the signed block 400s with
            # "thinking blocks in the latest assistant message cannot be modified",
            # and a bare signed block with no following tool_use is also invalid.
            # Demote ALL thinking blocks on this turn to text so the turn replays
            # cleanly and the model can re-plan from the surviving tool results.
            signature_dead = bool(m.get("_thinking_signature_invalidated"))
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if signature_dead:
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
                    continue
                if b.get("type") == "redacted_thinking":
                    # Redacted blocks use 'data' for the signature payload —
                    # drop the block when 'data' is missing (can't be validated).
                    if b.get("data"):
                        new_content.append(b)
                elif b.get("signature"):
                    new_content.append(b)
                else:
                    thinking_text = b.get("thinking", "")
                    if thinking_text:
                        new_content.append({"type": "text", "text": thinking_text})
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]

        # Strip cache_control from any remaining thinking/redacted_thinking
        # blocks — cache markers interfere with signature validation.
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") in _THINKING_TYPES:
                b.pop("cache_control", None)

        # Drop the internal bookkeeping flag — it must never reach the API payload.
        m.pop("_thinking_signature_invalidated", None)


def _evict_old_screenshots(result: List[Dict[str, Any]]) -> None:
    """Keep only the most recent ``_MAX_KEEP_IMAGES`` computer-use screenshots.

    Base64 images cost ~1,465 tokens each and accumulate across tool calls.
    Walk backward, keep the most recent N, replace older ones with a placeholder.

    Mutates ``result`` in place.
    """
    _MAX_KEEP_IMAGES = 3
    _image_count = 0
    for msg in reversed(result):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            inner = block.get("content")
            if not isinstance(inner, list):
                continue
            has_image = any(
                isinstance(b, dict) and b.get("type") == "image"
                for b in inner
            )
            if not has_image:
                continue
            _image_count += 1
            if _image_count > _MAX_KEEP_IMAGES:
                block["content"] = [
                    b if b.get("type") != "image"
                    else {"type": "text", "text": "[screenshot removed to save context]"}
                    for b in inner
                ]


def _ensure_leading_user_turn(result: List[Dict[str, Any]]) -> None:
    """Anthropic requires messages[0] to have role=user.

    After a second context compaction on the auto path the summary can be
    emitted as role=assistant with nothing in front of it (the system prompt
    lives outside messages[] or is extracted into the separate ``system``
    param), so messages[0] ends up assistant and the Messages API rejects
    the request with HTTP 400 — often masked by a misleading
    "tool_use ids were found without tool_result blocks" error (#52160).

    Mirror the Bedrock Converse adapter, which unconditionally prepends a
    minimal user turn when the first message is not user
    (convert_messages_to_converse).

    The inserted text block must be non-whitespace: Anthropic separately
    rejects any text content block whose text is empty or whitespace-only
    ("text content blocks must contain non-whitespace text"), so a single
    space here traded the "leading assistant turn" 400 for that one (#69512
    class). Uses the same placeholder as every other synthesized filler
    block in this module for consistency.
    """
    if result and result[0].get("role") != "user":
        result.insert(
            0, {"role": "user", "content": [{"type": "text", "text": _EMPTY_TEXT_PLACEHOLDER}]}
        )


def _fix_blank_text_blocks_in_list(
    blocks: List[Any],
    *,
    placeholder_text: str,
    msg_index: int,
    role: Any,
    location: str,
) -> List[Any]:
    """Drop blank/whitespace-only text blocks from ``blocks``, in place logic.

    Non-text blocks (tool_use, tool_result, image, document, thinking, …)
    and the relative order of everything else are left untouched. A
    cache_control marker riding on a dropped block is relocated onto the
    last surviving text/tool_use block so a breakpoint is never silently
    lost. If nothing survives, a single non-blank placeholder text block
    takes the dropped blocks' place (carrying the relocated cache_control,
    if any) so the message never has empty content.

    Returns a new list; does not mutate ``blocks``.
    """
    kept: List[Any] = []
    relocated_cache_control = None
    for block_index, blk in enumerate(blocks):
        if (
            isinstance(blk, dict)
            and blk.get("type") == "text"
            and not (isinstance(blk.get("text"), str) and blk["text"].strip())
        ):
            if isinstance(blk.get("cache_control"), dict):
                relocated_cache_control = blk["cache_control"]
            logger.warning(
                "Pre-call sanitizer: dropped blank text content block "
                "(message_index=%d role=%s location=%s block_index=%d "
                "block_type=text)",
                msg_index,
                role,
                location,
                block_index,
            )
            continue
        kept.append(blk)
    if not kept:
        placeholder: Dict[str, Any] = {"type": "text", "text": placeholder_text}
        if relocated_cache_control is not None:
            placeholder["cache_control"] = relocated_cache_control
        kept.append(placeholder)
    elif relocated_cache_control is not None:
        _apply_assistant_cache_control_to_last_cacheable_block(kept, relocated_cache_control)
    return kept


def _scrub_blank_text_blocks(result: List[Dict[str, Any]]) -> None:
    """Final provider-boundary guard against blank Anthropic text blocks.

    Anthropic rejects any text content block whose ``text`` is empty or
    whitespace-only with HTTP 400 ("text content blocks must contain
    non-whitespace text"). ``_convert_assistant_message``,
    ``_convert_user_message`` and ``_ensure_leading_user_turn`` already
    avoid emitting these for the paths that build them, but this pass runs
    last — after every other transform in ``convert_messages_to_anthropic``
    — so a blank block from any current or future producer (including one
    nested inside a ``tool_result``'s own content list) never reaches the
    wire. Diagnostics are structural only: message index, role, content
    location, block index/type. Never logs message text, tool arguments,
    tokens, or credentials. Mutates ``result`` in place.
    """
    for msg_index, msg in enumerate(result):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            continue
        placeholder_text = _EMPTY_TEXT_PLACEHOLDER if role == "assistant" else "(empty message)"
        new_content = _fix_blank_text_blocks_in_list(
            content,
            placeholder_text=placeholder_text,
            msg_index=msg_index,
            role=role,
            location="content",
        )
        for blk in new_content:
            if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                continue
            inner = blk.get("content")
            if isinstance(inner, list) and inner:
                blk["content"] = _fix_blank_text_blocks_in_list(
                    inner,
                    placeholder_text="(no output)",
                    msg_index=msg_index,
                    role=role,
                    location="tool_result",
                )
        msg["content"] = new_content


def convert_messages_to_anthropic(
    messages: List[Dict],
    base_url: str | None = None,
    model: str | None = None,
) -> Tuple[Optional[Any], List[Dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    System messages are extracted since Anthropic takes them as a separate param.
    system_prompt is a string or list of content blocks (when cache_control present).

    When *base_url* is provided and points to a third-party Anthropic-compatible
    endpoint, all thinking block signatures are stripped.  Signatures are
    Anthropic-proprietary — third-party endpoints cannot validate them and will
    reject them with HTTP 400 "Invalid signature in thinking block".

    When *model* is provided and matches the Kimi / Moonshot family (or
    *base_url* is a Kimi / Moonshot host), unsigned thinking blocks
    synthesised from ``reasoning_content`` are preserved on replayed
    assistant tool-call messages — Kimi requires the field to exist, even
    if empty.
    """
    system = None
    result: List[Dict[str, Any]] = []

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
                    # Copy blocks before coercing so the caller's message
                    # dicts are never mutated, then replace blank/whitespace
                    # text with the shared non-whitespace placeholder —
                    # Anthropic rejects a blank system text block with the
                    # same HTTP 400 as message blocks ("text content blocks
                    # must contain non-whitespace text"), and a blank block
                    # carrying a cache_control breakpoint cannot simply be
                    # dropped (#70909).
                    system = []
                    for p in content:
                        if not isinstance(p, dict):
                            continue
                        if (
                            p.get("type") == "text"
                            and isinstance(p.get("text"), str)
                            and not p["text"].strip()
                        ):
                            p = dict(p)
                            p["text"] = _EMPTY_TEXT_PLACEHOLDER
                        system.append(p)
                else:
                    system = "\n".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
            else:
                system = content
            continue

        if role == "assistant":
            result.append(_convert_assistant_message(m))
            continue

        if role == "tool":
            _convert_tool_message_to_result(result, m)
            continue

        # Regular user message
        result.append(_convert_user_message(content))

    _strip_orphaned_tool_blocks(result)
    result = _merge_consecutive_roles(result)
    _ensure_leading_user_turn(result)
    _manage_thinking_signatures(result, base_url, model)
    _evict_old_screenshots(result)
    _scrub_blank_text_blocks(result)

    return system, result

