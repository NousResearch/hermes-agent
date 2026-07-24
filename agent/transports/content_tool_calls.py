"""Detect tool calls models emit inside the response ``content`` field (as JSON,
XML, or special tokens) instead of the structured ``tool_calls`` field, and
promote them to executed calls. Strict fallback: only consulted when the
transport returned no structured tool_calls.

Leaf module: imports only stdlib + agent.transports.types (itself a leaf).
Do NOT import agent.codex_responses_adapter — it pulls ~92 modules via
prompt_builder and this module sits on cli.py's import path.

Ported from upstream Hermes PR #35129 to defend against M3 / MiniMax, Kimi K2,
Ollama qwen2.5-coder, GLM, and Gemma emitting tool calls in the content
channel rather than the structured tool_calls field. See
https://github.com/NousResearch/hermes-agent/pull/35129 for the full PR.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from agent.transports.types import ToolCall, build_tool_call

logger = logging.getLogger(__name__)


def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """``call_<sha256(name:args:index)[:12]>`` — stable id keeps the prompt cache
    warm. Mirrors agent/codex_responses_adapter.py (copied to keep this a leaf)."""
    digest = hashlib.sha256(
        f"{fn_name}:{arguments}:{index}".encode(errors="replace")
    ).hexdigest()
    return f"call_{digest[:12]}"


@dataclass(frozen=True)
class RawCall:
    name: str
    arguments: Any  # dict or JSON string
    span: str  # exact substring matched (removed to form residual)


@dataclass(frozen=True)
class ContentFormat:
    name: str
    find_calls: Callable[[str], list[RawCall]]


FORMATS: list[ContentFormat] = []


def _loads_lenient(raw: str) -> Any | None:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


# Line/content-start or tag-wrapped only (e.g. ``</think>\n<tool_call>`` or a
# ``<minimax:tool_call>`` wrapper), never mid-prose. Promotion EXECUTES the call,
# so narrated framing like "you'd emit <tool_call>{…}</tool_call>" must not fire.
# ``(?<=>)`` admits a preceding tag (reasoning-close / wrapper) without admitting
# prose, which always ends in a space or word char.
_TOOL_CALL_BLOCK_RE = re.compile(
    r"(?:(?<=^)|(?<=[\n\r])|(?<=>))[ \t]*"
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def find_tool_call_json(content: str) -> list[RawCall]:
    out: list[RawCall] = []
    for m in _TOOL_CALL_BLOCK_RE.finditer(content):
        obj = _loads_lenient(m.group(1))
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            out.append(
                RawCall(
                    name=obj["name"],
                    arguments=obj.get("arguments", {}),
                    span=m.group(0),
                )
            )
    return out


FORMATS.append(ContentFormat("tool_call_json", find_tool_call_json))


_MAX_BARE_JSON_ARGS = 16_000


def find_bare_json_object(content: str) -> list[RawCall]:
    s = content.strip()
    if not (s.startswith("{") and s.endswith("}")):  # whole-content-only
        return []
    obj = _loads_lenient(s)
    if not isinstance(obj, dict) or obj.keys() - {"name", "arguments"}:
        return []
    name, args = obj.get("name"), obj.get("arguments", {})
    if not name or not isinstance(name, str) or not isinstance(args, (dict, str)):
        return []
    serialized = json.dumps(args) if isinstance(args, dict) else args
    if len(serialized) > _MAX_BARE_JSON_ARGS:
        return []
    return [RawCall(name=name, arguments=args, span=content)]


FORMATS.append(ContentFormat("bare_json_object", find_bare_json_object))


_KIMI_SECTION_RE = re.compile(
    r"<\|tool_calls?_section_begin\|>(.*?)<\|tool_calls?_section_end\|>", re.DOTALL
)
_KIMI_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<id>.*?)\s*<\|tool_call_argument_begin\|>(?P<args>.*?)<\|tool_call_end\|>",
    re.DOTALL,
)


def _kimi_name(raw_id: str) -> str:
    name = raw_id.strip().removeprefix("functions.")
    return name.rsplit(":", 1)[0].strip()


def find_kimi_k2(content: str) -> list[RawCall]:
    if "<|tool_call" not in content:
        return []
    out: list[RawCall] = []
    for section in _KIMI_SECTION_RE.finditer(content):
        for m in _KIMI_CALL_RE.finditer(section.group(1)):
            obj = _loads_lenient(m.group("args").strip())
            name = _kimi_name(m.group("id"))
            # Per-call span (not the whole section): a section may hold several
            # parallel calls and they must not dedup against each other.
            if name and isinstance(obj, dict):
                out.append(RawCall(name=name, arguments=obj, span=m.group(0)))
    return out


FORMATS.append(ContentFormat("kimi_k2", find_kimi_k2))


# Same line-start / tag-wrapped gate as <tool_call>: the real capture is
# ``<minimax:tool_call>\n<invoke …>`` (newline- or wrapper-led), so mid-prose
# "call <invoke name=…>" stays inert.
_INVOKE_RE = re.compile(
    r'(?:(?<=^)|(?<=[\n\r])|(?<=>))[ \t]*'
    r'<invoke\b[^>]*\bname\s*=\s*"([^"]+)"[^>]*>(.*?)</invoke>',
    re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(
    r'<parameter\b[^>]*\bname\s*=\s*"([^"]+)"[^>]*>(.*?)</parameter>',
    re.DOTALL | re.IGNORECASE,
)


def find_minimax_invoke(content: str) -> list[RawCall]:
    if "<invoke" not in content.lower():
        return []
    out: list[RawCall] = []
    for m in _INVOKE_RE.finditer(content):
        name = m.group(1).strip()
        args = {pn.strip(): pv.strip() for pn, pv in _PARAM_RE.findall(m.group(2))}
        if name:
            out.append(RawCall(name=name, arguments=args, span=m.group(0)))
    return out


FORMATS.append(ContentFormat("minimax_invoke", find_minimax_invoke))


# Line-start + attribute gated so prose is not matched. Adapted from the
# strip_think_blocks gate (openclaw#67318) but STRICTER: that gate also allows
# sentence-ending punctuation (.!?:) as a boundary, which is safe for *display
# stripping* but not here — promotion EXECUTES the call, so "I'll call:
# <function name=...>" mid-sentence prose must not fire. Only line/content start.
_GEMMA_FUNC_RE = re.compile(
    r"(?:(?<=^)|(?<=[\n\r]))[ \t]*"
    r'<function\b[^>]*\bname\s*=\s*"(?P<name>[^"]+)"[^>]*>'
    r"(?P<body>(?:(?!</function>).)*)</function>",
    re.DOTALL | re.IGNORECASE,
)


def find_gemma_function(content: str) -> list[RawCall]:
    out: list[RawCall] = []
    for m in _GEMMA_FUNC_RE.finditer(content):
        obj = _loads_lenient(m.group("body").strip())
        if isinstance(obj, dict):
            out.append(
                RawCall(name=m.group("name").strip(), arguments=obj, span=m.group(0))
            )
    return out


FORMATS.append(ContentFormat("gemma_function", find_gemma_function))


# Pythonic / llama.cpp tool template: ``<function=NAME>{json args}</function>``
# (attribute-less ``=NAME``, distinct from Gemma's quoted ``<function name="NAME">``).
# Emitted by some Ollama-cloud proxies — issue #8965, whose "raw XML" is this form
# rather than <tool_call>. Same line/content-start gate as Gemma: promotion
# executes, so a narrated ``call <function=x>…`` mid-prose must not fire.
_PYTHONIC_FUNC_RE = re.compile(
    r"(?:(?<=^)|(?<=[\n\r]))[ \t]*"
    r"<function\s*=\s*(?P<name>[^\s>]+)\s*>"
    r"(?P<body>(?:(?!</function>).)*)</function>",
    re.DOTALL | re.IGNORECASE,
)


def find_pythonic_function(content: str) -> list[RawCall]:
    if "<function" not in content.lower():
        return []
    out: list[RawCall] = []
    for m in _PYTHONIC_FUNC_RE.finditer(content):
        obj = _loads_lenient(m.group("body").strip())
        if isinstance(obj, dict):
            out.append(
                RawCall(name=m.group("name").strip(), arguments=obj, span=m.group(0))
            )
    return out


FORMATS.append(ContentFormat("pythonic_function", find_pythonic_function))


_TRUTHY = {"1", "true", "yes", "on"}


def _env_on(name: str) -> bool:
    return os.getenv(name, "true").lower() in _TRUTHY


def _find_all(content: str) -> list[RawCall]:
    out: list[RawCall] = []
    for fmt in FORMATS:
        if fmt.name == "bare_json_object" and not _env_on(
            "HERMES_PROMOTE_BARE_JSON_TOOLCALL"
        ):
            continue
        try:
            out.extend(fmt.find_calls(content))
        except Exception:
            logger.warning(
                "content tool-call parser %s raised", fmt.name, exc_info=True
            )
    return out


def _dedupe_overlapping(raws: list[RawCall], content: str) -> list[RawCall]:
    """Drop any RawCall whose span overlaps an already-accepted span's byte range."""
    taken: list[tuple[int, int]] = []
    kept: list[RawCall] = []
    for rc in raws:
        # First-occurrence find: two byte-identical spans collapse to one. That
        # is the intent for cross-format overlap (bare-JSON vs <tool_call> on the
        # same bytes); a model emitting the same call verbatim twice is treated
        # as one. Distinct calls have distinct span text and survive.
        start = content.find(rc.span)
        if start < 0:
            continue
        end = start + len(rc.span)
        if any(start < t_end and t_start < end for t_start, t_end in taken):
            continue
        taken.append((start, end))
        kept.append(rc)
    return kept


def _residual(content: str, spans: list[str]) -> str:
    for span in spans:
        content = content.replace(span, "", 1)
    return content


def extract_content_tool_calls(
    content: str, valid_tool_names: set[str]
) -> tuple[list[ToolCall], str]:
    """Promote content-embedded tool calls whose name EXACTLY matches an active
    tool. Returns (promoted ToolCalls, residual content with consumed markup
    removed). Strict fallback: caller invokes only when structured tool_calls
    is empty. Promoted names are exact members, so the loop's fuzzy
    _repair_tool_call never touches them (native calls keep repair)."""
    if not content or not _env_on("HERMES_PROMOTE_TOOLCALLS"):
        return [], content
    raws = _dedupe_overlapping(_find_all(content), content)
    promoted: list[ToolCall] = []
    consumed: list[str] = []
    for i, rc in enumerate(raws):
        consumed.append(rc.span)  # remove markup whether or not it promotes
        # EXACT match only — deliberately NOT agent._repair_tool_call. Superseded
        # PR #26353 fuzzy-repaired promoted names (``search`` -> ``web_search``);
        # we drop that on purpose. Native tool_calls come from a structured field
        # so a near-miss is a safe typo to repair, but a name lifted from free-text
        # CONTENT is lower-trust — fuzzy-repairing it risks executing the WRONG
        # tool from prose. Fail closed; native calls keep repair at the loop seam.
        if rc.name not in valid_tool_names:
            continue
        args_str = (
            json.dumps(rc.arguments, ensure_ascii=False)
            if isinstance(rc.arguments, dict)
            else str(rc.arguments)
        )
        cid = _deterministic_call_id(rc.name, args_str, i)
        promoted.append(build_tool_call(id=cid, name=rc.name, arguments=args_str))
        logger.info("promoted content tool call: tool=%s id=%s", rc.name, cid)
    return promoted, _residual(content, consumed)
