"""Shared engine module for prompt-optimizer.

Contains all logic used by both the plugin gateway and CLI hooks:
- Data classes
- Template registry
- Pending preview store (thread-safe)
- Session model cache (thread-safe)
- SQLite metrics store
- Heuristic scoring
- Optimizer engine (LLM rewrite)
- Report formatting and rendering
"""

from __future__ import annotations

import dataclasses
import html
import json
import logging
import os
import re
import sqlite3
import threading
import textwrap
import time
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
PLUGIN_DIR = HERMES_HOME / "plugins" / "prompt-optimizer"
METRICS_DB = PLUGIN_DIR / "metrics.db"

OPTIMIZER_TIMEOUT_S = 30
OPTIMIZER_MODEL = "deepseek-v4-flash"

BYPASS_PREFIXES = ("/quick", "*simple", "#basic")

# Skill invocations are curated prompts injected by the skill loader, not
# user prose — rewriting them risks corrupting the skill contract and drags
# every skill invocation through an auxiliary LLM round-trip.
SKILL_INVOCATION_MARKER = '[IMPORTANT: The user has invoked the'


def is_skill_invocation(text: str) -> bool:
    """True when the prompt is a skill-loader invocation rather than user prose."""
    return text.lstrip().startswith(SKILL_INVOCATION_MARKER)


# Structured commands carry machine-readable payloads (delegation task lists,
# rubrics, fenced code) where a rewrite can silently corrupt the contract —
# same principle as skill invocations. Extend the verb list via the
# PROMPT_OPTIMIZER_BYPASS_VERBS env var (comma-separated, prepended).
STRUCTURED_COMMAND_VERBS = ("delegate_task", "delegate")


def _bypass_verbs() -> tuple:
    extra = os.getenv("PROMPT_OPTIMIZER_BYPASS_VERBS", "")
    verbs = tuple(v.strip().lower() for v in extra.split(",") if v.strip())
    return verbs + STRUCTURED_COMMAND_VERBS


def is_structured_command(text: str) -> bool:
    """True when the prompt is a structured command rather than user prose.

    Matches messages containing a fenced code block, or whose first word
    is a known orchestration verb (trailing ``:`` tolerated).
    """
    stripped = text.strip()
    if "```" in stripped:
        return True
    parts = stripped.split(None, 1)
    if not parts:
        return False
    first_word = parts[0].lower().rstrip(":")
    return first_word in _bypass_verbs()

_PENDING_TIMEOUT_S = 120

# Path to the model-profiles YAML; user-editable in place
_MODEL_PROFILES_PATH = PLUGIN_DIR / "model-profiles.yaml"

_LONDON = ZoneInfo("Europe/London") if ZoneInfo else None
_REPORT_DIR = PLUGIN_DIR / "reports"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RewriteRecord:
    original: str
    rewritten: str
    quality_before: float
    quality_after: float
    token_delta_pct: float
    model_profile: str
    approved: bool = True


@dataclass
class PromptTemplate:
    name: str
    description: str
    system_prompt: str


@dataclass
class PromptOptimizationPreview:
    session_key: str
    original: str
    rewritten: str
    quality_before: float
    quality_after: float
    token_delta_pct: float
    model_profile: str
    created_at: float = dataclasses.field(default_factory=time.time)
    template_name: Optional[str] = None


@dataclass
class PromptOptimizationDecision:
    session_key: str
    approved: bool
    applied_text: str
    record: RewriteRecord


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_TEMPLATES: List[PromptTemplate] = [
    PromptTemplate(
        "default",
        "Balanced clarity and token efficiency",
        "You rewrite user prompts for maximum clarity and token efficiency.\n\n"
        "RULES:\n"
        "1. Preserve ALL intent, constraints, and specific details.\n"
        "2. Remove filler, fluff, and redundant politeness.\n"
        "3. Be direct — start with the action verb.\n"
        "4. Assign quality scores based on the nuance and strength of each dimension.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
    PromptTemplate(
        "concise",
        "Maximum brevity without losing intent",
        "You aggressively compress user prompts to their shortest clear form.\n\n"
        "RULES:\n"
        "1. Strip every non-essential word.\n"
        "2. Preserve ONLY the core intent, constraints, and specific values.\n"
        "3. Be direct — start with the action verb.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
    PromptTemplate(
        "verbose",
        "Preserve full context while improving structure",
        "You restructure user prompts for maximum clarity while keeping EVERY detail.\n\n"
        "RULES:\n"
        "1. Reorder and rephrase for readability.\n"
        "2. Do NOT remove background, context, or examples.\n"
        "3. Be direct — start with the action verb.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
    PromptTemplate(
        "technical",
        "Optimize for code and technical tasks",
        "You rewrite user prompts for maximum clarity on software engineering tasks.\n\n"
        "RULES:\n"
        "1. Preserve ALL file paths, identifiers, stack traces, and constraints.\n"
        "2. Remove filler but keep technical precision.\n"
        "3. Be direct — start with the action verb.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
    PromptTemplate(
        "creative",
        "Optimize for creative writing and storytelling",
        "You rewrite user prompts for maximum clarity on creative tasks.\n\n"
        "RULES:\n"
        "1. Preserve tone, style cues, narrative structure, and character details.\n"
        "2. Remove filler and redundant phrasing.\n"
        "3. Be direct — start with the action verb.\n\n"
        "OUTPUT FORMAT:\n"
        "Return the optimised prompt first, then a line containing exactly ---SCORES---,\n"
        "then a JSON object with quality assessment per dimension (0-100 each):\n"
        '  - clarity: how unambiguously the intent and desired outcome are expressed\n'
        '  - specificity: how precise the constraints, inputs, and scope are\n'
        '  - terminology: how well domain-specific language is used (or avoided if generic)\n'
        '  - actionability: how clearly the reader knows what to do next\n'
        '  - structure: how well the prompt is organised (sections, lists, ordering)\n\n'
        "ORIGINAL PROMPT:\n{original}\n\n"
        "Rewritten prompt:",
    ),
]


def select_template(name: str = "") -> PromptTemplate:
    """Return a template by name, falling back to the default."""
    if not name:
        return _TEMPLATES[0]
    for t in _TEMPLATES:
        if t.name == name:
            return t
    return _TEMPLATES[0]


# ---------------------------------------------------------------------------
# Model profile loading + resolution (family + capability composition)
# ---------------------------------------------------------------------------
#
# Each model resolves along two axes:
#   1. family     — vendor / model family (claude, openai, deepseek, …)
#   2. capability — reasoning vs general
#
# The rewriter system prompt is composed by concatenating whichever axes
# resolved. If both come back None the base template runs untouched.
#
# The shape lives in ``model-profiles.yaml`` next to this file. If the
# YAML is missing or malformed the baked-in defaults below are used so
# the plugin still works without the data file. User edits to the YAML
# take precedence at load time.

_BAKED_PROFILES: Dict[str, Any] = {
    "families": {
        "claude": {
            "description": "Anthropic Claude — XML-friendly, long context",
            "prompt_tactics": [
                "Use XML tags (<thinking>, <answer>, <example>) to mark structure",
                "State constraints and boundaries explicitly",
                "Open with an imperative verb — no hedging or pleasantries",
                "Put examples in <example> blocks rather than inline prose",
            ],
            "token_efficiency_rules": [
                "Replace 'Could you please' with imperative verbs",
                "Remove filler adverbs (very, really, quite, just)",
                "Use bullet lists instead of prose enumerations",
            ],
        },
        "openai": {
            "description": "OpenAI GPT — tool use, function calling, generalist",
            "prompt_tactics": [
                "State the output format explicitly (JSON, markdown, plain)",
                "Put static instructions in the system role, dynamic input in user",
                "Use numbered steps for multi-part tasks",
                "Reference functions/tools by name when expecting calls",
            ],
            "token_efficiency_rules": [
                "Move repeated instructions to the system prompt",
                "Compress examples to minimal viable form",
                "Avoid restating role / persona mid-prompt",
            ],
        },
        "deepseek": {
            "description": "DeepSeek — coding-strong, cost-efficient",
            "prompt_tactics": [
                "Get to the task in the first sentence — no preamble",
                "Break complex requests into numbered sub-tasks",
                "Use markdown code blocks for any code-shaped input or output",
            ],
            "token_efficiency_rules": [
                "Use short variable names in code examples",
                "Strip redundant explanations after code blocks",
                "Skip the closing summary",
            ],
        },
        "google": {
            "description": "Google Gemini / Gemma — long context, multimodal",
            "prompt_tactics": [
                "Use clear section headers (## Context, ## Task, ## Output)",
                "Leverage long context — full documents beat excerpts",
                "State citation requirements upfront when grounding matters",
                "Prefer structured data (tables, JSON) over prose",
            ],
            "token_efficiency_rules": [
                "Group related instructions under one header",
                "Reference earlier sections by header name, don't repeat",
                "Use tables for parallel comparisons",
            ],
        },
        "nvidia": {
            "description": "Nvidia Nemotron — instruction-tuned, reasoning variants",
            "prompt_tactics": [
                "Open with an imperative verb stating the deliverable",
                "List constraints as a bulleted checklist",
                "Specify output structure (sections, headings) explicitly",
            ],
            "token_efficiency_rules": [
                "Drop politeness phrasing",
                "Use concise constraint language ('must', 'never', 'only')",
            ],
        },
        "kimi": {
            "description": "Moonshot Kimi — long context",
            "prompt_tactics": [
                "State the goal in one sentence at the top",
                "Provide full reference material rather than summaries",
                "Specify output length and format explicitly",
            ],
            "token_efficiency_rules": [
                "Skip self-introductions",
                "Use direct commands",
            ],
        },
        "qwen": {
            "description": "Alibaba Qwen — multilingual, instruction-following",
            "prompt_tactics": [
                "Lead with the action verb",
                "State output language explicitly if non-English",
                "Use markdown for structured output",
            ],
            "token_efficiency_rules": [
                "Strip filler and hedge words",
                "Use bullet points over prose",
            ],
        },
        "mistral": {
            "description": "Mistral / Magistral — fast, instruction-tuned",
            "prompt_tactics": [
                "Be concise — Mistral handles short prompts best",
                "State output format explicitly",
                "Use the system role for persistent context",
            ],
            "token_efficiency_rules": [
                "Drop pleasantries",
                "Move static context to system prompt",
            ],
        },
        "llama": {
            "description": "Meta Llama — open-weight generalist, instruction-tuned",
            "prompt_tactics": [
                "Use the system message for role + persistent constraints",
                "Provide examples for any structured-output task",
                "Number steps for multi-part workflows",
            ],
            "token_efficiency_rules": [
                "Remove conversational fillers",
                "Use direct commands; Llama responds well to imperatives",
            ],
        },
        "nousresearch": {
            "description": "NousResearch Hermes — agentic, tool-use focused",
            "prompt_tactics": [
                "State the task imperatively in the first line",
                "Define tool contracts explicitly when expecting calls",
                "Use JSON schemas for structured output",
                "Be explicit about reasoning vs action steps",
            ],
            "token_efficiency_rules": [
                "Skip pleasantries",
                "Use minimal prose around code/JSON blocks",
            ],
        },
        "xai": {
            "description": "xAI Grok — generalist, conversational, reasoning variants",
            "prompt_tactics": [
                "Be direct — Grok handles informal language well",
                "State the format requirement once, clearly",
                "Use markdown structure for technical content",
            ],
            "token_efficiency_rules": [
                "Drop unnecessary politeness",
                "Avoid restating context the model has just seen",
            ],
        },
        "amazon": {
            "description": "Amazon Nova — generalist, multimodal",
            "prompt_tactics": [
                "Structure prompts as: Context → Task → Output spec",
                "Be explicit about output format (JSON, markdown, plain text)",
                "Provide examples for any non-trivial schema",
            ],
            "token_efficiency_rules": [
                "Separate static instructions into system role",
                "Compress example data to minimal viable form",
            ],
        },
        "cohere": {
            "description": "Cohere Command — RAG-optimised, citation-aware",
            "prompt_tactics": [
                "Provide reference documents explicitly — Command excels at grounded answers",
                "Request citations when grounding matters",
                "Use clear section markers for documents vs instructions",
            ],
            "token_efficiency_rules": [
                "Reference documents by ID rather than re-quoting",
                "Strip explanatory padding around the core question",
            ],
        },
        "microsoft": {
            "description": "Microsoft Phi — small instruction-tuned models",
            "prompt_tactics": [
                "Keep prompts concise — Phi has limited context tolerance",
                "Use simple structure (one task per prompt)",
                "State output format explicitly",
            ],
            "token_efficiency_rules": [
                "Avoid long preambles",
                "One example is enough — don't pile on shots",
            ],
        },
        "perplexity": {
            "description": "Perplexity Sonar — search-augmented, citation-native",
            "prompt_tactics": [
                "Scope the query narrowly — Sonar searches what you ask for",
                "State freshness / recency requirements explicitly",
                "Ask for sources / citations when factuality matters",
            ],
            "token_efficiency_rules": [
                "Skip 'please search for' — Sonar searches by default",
                "Use specific terms over generic ones",
            ],
        },
        "zhipu": {
            "description": "Zhipu GLM / GLM-Z — bilingual, reasoning variants",
            "prompt_tactics": [
                "Be explicit about output language (English/Chinese)",
                "Use structured input (sections, bullets) for complex tasks",
                "State the deliverable in the opening sentence",
            ],
            "token_efficiency_rules": [
                "Strip hedge language",
                "Use direct constraint statements",
            ],
        },
        "liquid": {
            "description": "Liquid Foundation Models — efficient, edge-friendly",
            "prompt_tactics": [
                "Keep prompts focused — one task per call",
                "State output format explicitly",
                "Provide minimal but precise context",
            ],
            "token_efficiency_rules": [
                "Avoid padding and pleasantries",
                "Use lists over prose",
            ],
        },
        "minimax": {
            "description": "MiniMax M-series — long context, reasoning-capable",
            "prompt_tactics": [
                "State the deliverable in one sentence at the top",
                "Structure input with clear section headers",
                "Specify output length and format explicitly",
            ],
            "token_efficiency_rules": [
                "Move reusable context to a separate section",
                "Avoid restating constraints mid-prompt",
            ],
        },
        "ibm": {
            "description": "IBM Granite — enterprise instruction-tuned, code-aware",
            "prompt_tactics": [
                "Use structured prompts with clear sections",
                "State output format and length requirements upfront",
                "Provide schemas for any structured-output task",
            ],
            "token_efficiency_rules": [
                "Drop conversational filler",
                "Use precise technical terminology",
            ],
        },
        "inflection": {
            "description": "Inflection Pi — conversational, empathy-tuned",
            "prompt_tactics": [
                "Use a conversational tone — Pi responds best to natural dialogue",
                "State the desired outcome clearly",
            ],
            "token_efficiency_rules": [
                "Avoid robotic bullet-list-only prompts",
                "Keep instructions natural and contextual",
            ],
        },
        "xiaomi": {
            "description": "Xiaomi MiMo — reasoning-focused, multilingual",
            "prompt_tactics": [
                "State constraints upfront — MiMo reasons over the full context",
                "Use structured input for complex problems",
                "Specify reasoning depth requirements",
            ],
            "token_efficiency_rules": [
                "Avoid restating earlier context",
                "Use bullet structure for multi-part problems",
            ],
        },
    },
    "capabilities": {
        "reasoning": {
            "description": "Reasoning-native model (o-series, r-series, thinking, magistral, etc.)",
            "prompt_tactics": [
                "Front-load ALL constraints and examples — no incremental hints",
                "State the goal once, clearly, without back-references",
                "Skip 'think step by step' — these models reason by default",
                "Provide explicit verification criteria for correctness",
            ],
            "token_efficiency_rules": [
                "Avoid restating context the model has just seen",
                "Drop chain-of-thought scaffolding (the model adds its own)",
                "Keep prompts dense — these models tolerate complexity well",
            ],
        },
        "general": {
            "description": "Standard chat model",
            "prompt_tactics": [],
            "token_efficiency_rules": [],
        },
    },
    "family_aliases": {
        "claude":       ["claude-", "anthropic/"],
        "openai":       ["gpt-", "openai/", "o1", "o3", "o4"],
        "deepseek":     ["deepseek-", "deepseek/"],
        "google":       ["gemini-", "gemma-", "google/"],
        "nvidia":       ["nvidia/", "nemotron-"],
        "kimi":         ["kimi-", "moonshotai/"],
        "qwen":         ["qwen"],
        "mistral":      ["mistral-", "mistralai/", "magistral"],
        "llama":        ["llama-", "meta-llama/"],
        "nousresearch": ["nousresearch/", "hermes-4", "hermes-3"],
        "xai":          ["x-ai/", "grok-"],
        "amazon":       ["amazon/", "nova-"],
        "cohere":       ["cohere/", "command-"],
        "microsoft":    ["microsoft/", "phi-"],
        "perplexity":   ["perplexity/", "sonar-"],
        "zhipu":        ["z-ai/", "glm-"],
        "liquid":       ["liquid/", "lfm-"],
        "minimax":      ["minimax/", "minimaxai/"],
        "ibm":          ["ibm-granite/", "granite-"],
        "inflection":   ["inflection/"],
        "xiaomi":       ["xiaomi/", "mimo-"],
    },
    "reasoning_indicators": [
        "o1",
        "o3",
        "o4",
        "-r1",
        "deepseek-r",
        "thinking",
        "-think",
        "reasoning",
        "nemotron-3-super",
        "qwq",
        "magistral",
        "glm-z",
        "deepresearch",
    ],
}


_profiles_cache: Optional[Dict[str, Any]] = None


def _load_model_profiles() -> Dict[str, Any]:
    """Return the model-profile registry — cached.

    Reads ``model-profiles.yaml`` next to this module on first call.
    Missing file or malformed YAML falls through to ``_BAKED_PROFILES``
    so the plugin always has a working registry.
    """
    global _profiles_cache
    if _profiles_cache is not None:
        return _profiles_cache

    if not _MODEL_PROFILES_PATH.exists():
        _profiles_cache = _BAKED_PROFILES
        return _profiles_cache

    try:
        import yaml
        with open(_MODEL_PROFILES_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict) or "families" not in data:
            logger.warning(
                "prompt-optimizer: model-profiles.yaml missing 'families' "
                "block; using baked-in defaults"
            )
            _profiles_cache = _BAKED_PROFILES
            return _profiles_cache
        _profiles_cache = data
        return _profiles_cache
    except Exception as exc:
        logger.warning(
            "prompt-optimizer: failed to load model-profiles.yaml (%s); "
            "using baked-in defaults", exc,
        )
        _profiles_cache = _BAKED_PROFILES
        return _profiles_cache


def _reset_profiles_cache() -> None:
    """Test helper — drop the cached profiles so a reload happens."""
    global _profiles_cache
    _profiles_cache = None


def resolve_model_profile(model: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a model string to (family, capability).

    Family lookup uses substring matching against ``family_aliases`` from
    the profile registry (case-insensitive). Capability is "reasoning"
    if any ``reasoning_indicators`` substring is present, otherwise
    "general" when a family was detected, or None when family is also
    None.

    Both axes can be None independently:
      * `(claude, general)`  — Claude Sonnet, Opus, etc.
      * `(openai, reasoning)` — o3-mini, gpt-5.5 (thinking variants)
      * `(None, reasoning)`  — unknown vendor but obvious thinking model
      * `(None, None)`       — entirely unknown → base template
    """
    if not isinstance(model, str) or not model:
        return (None, None)
    needle = model.lower()
    profiles = _load_model_profiles()

    family: Optional[str] = None
    for fname, aliases in (profiles.get("family_aliases") or {}).items():
        if any(a.lower() in needle for a in (aliases or [])):
            family = fname
            break

    is_reasoning = any(
        ind.lower() in needle
        for ind in (profiles.get("reasoning_indicators") or [])
    )
    if is_reasoning:
        capability: Optional[str] = "reasoning"
    elif family is not None:
        capability = "general"
    else:
        capability = None

    return (family, capability)


def _render_profile_guidance(
    model: str,
    family: Optional[str],
    capability: Optional[str],
) -> str:
    """Build the MODEL-SPECIFIC GUIDANCE block for the rewriter prompt.

    Returns the empty string when neither axis resolved (the caller
    splices nothing and the base template is used as-is).
    """
    if family is None and capability is None:
        return ""

    profiles = _load_model_profiles()
    families = profiles.get("families") or {}
    capabilities = profiles.get("capabilities") or {}

    family_block = families.get(family) if family else None
    capability_block = capabilities.get(capability) if capability else None

    family_tactics = (family_block or {}).get("prompt_tactics") or []
    family_rules = (family_block or {}).get("token_efficiency_rules") or []
    cap_tactics = (capability_block or {}).get("prompt_tactics") or []
    cap_rules = (capability_block or {}).get("token_efficiency_rules") or []

    lines: List[str] = [
        "MODEL-SPECIFIC GUIDANCE (target: {})".format(model or "unknown"),
        "Family: {}  |  Capability: {}".format(
            family or "unknown", capability or "unknown",
        ),
    ]

    if family_tactics:
        lines.append("")
        lines.append("Family tactics ({}):".format(family))
        for t in family_tactics:
            lines.append("  - {}".format(t))

    if cap_tactics:
        lines.append("")
        lines.append("Capability tactics ({}):".format(capability))
        for t in cap_tactics:
            lines.append("  - {}".format(t))

    if family_rules or cap_rules:
        lines.append("")
        lines.append("Token efficiency:")
        for r in family_rules + cap_rules:
            lines.append("  - {}".format(r))

    lines.append("")
    return "\n".join(lines)


def _inject_guidance_into_template(
    template: PromptTemplate, guidance: str,
) -> PromptTemplate:
    """Return a new PromptTemplate with guidance spliced before OUTPUT FORMAT.

    The system prompt of each built-in template contains a
    ``"OUTPUT FORMAT:"`` marker. We insert the guidance block immediately
    before that line so the LLM sees RULES → MODEL GUIDANCE → OUTPUT FORMAT.
    If the marker is absent (e.g. a future custom template) we prepend
    the guidance at the start of the system prompt as a safe fallback.
    """
    if not guidance:
        return template

    marker = "OUTPUT FORMAT:"
    sp = template.system_prompt
    idx = sp.find(marker)
    if idx == -1:
        new_sp = guidance + "\n\n" + sp
    else:
        new_sp = sp[:idx] + guidance + "\n" + sp[idx:]

    return PromptTemplate(
        name=template.name,
        description=template.description,
        system_prompt=new_sp,
    )


# ---------------------------------------------------------------------------
# Pending preview store (thread-safe)
# ---------------------------------------------------------------------------

_pending_store: Dict[str, PromptOptimizationPreview] = {}
_pending_store_lock = threading.RLock()


def create_preview(session_key: str, preview: PromptOptimizationPreview) -> None:
    with _pending_store_lock:
        _pending_store[session_key] = preview


def resolve_preview(session_key: str) -> Optional[PromptOptimizationPreview]:
    with _pending_store_lock:
        return _pending_store.pop(session_key, None)


def get_pending_preview(session_key: str) -> Optional[PromptOptimizationPreview]:
    with _pending_store_lock:
        return _pending_store.get(session_key)


def pop_and_record(session_key: str) -> Optional[PromptOptimizationPreview]:
    """Pop preview for session_key without side effects."""
    with _pending_store_lock:
        return _pending_store.pop(session_key, None)


def expire_previews(timeout: float = _PENDING_TIMEOUT_S) -> List[str]:
    """Expire stale previews and return the removed session keys."""
    now = time.time()
    expired: List[str] = []
    with _pending_store_lock:
        for key, preview in list(_pending_store.items()):
            if now - preview.created_at > timeout:
                expired.append(key)
        for key in expired:
            del _pending_store[key]
    return expired


# ---------------------------------------------------------------------------
# Session model cache (thread-safe)
# ---------------------------------------------------------------------------

_session_model_cache: Dict[str, str] = {}
_session_model_lock = threading.RLock()


def cache_session_model(session_key: str, model: str) -> None:
    with _session_model_lock:
        _session_model_cache[session_key] = model


def get_cached_session_model(session_key: str) -> Optional[str]:
    with _session_model_lock:
        return _session_model_cache.get(session_key)


# ---------------------------------------------------------------------------
# SQLite metrics store
# ---------------------------------------------------------------------------

_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS rewrites (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    session_id  TEXT,
    platform    TEXT,
    original    TEXT NOT NULL,
    rewritten   TEXT NOT NULL,
    quality_before REAL,
    quality_after  REAL,
    token_delta_pct REAL,
    model_profile TEXT,
    model_used    TEXT,
    mode          TEXT,
    approved      INTEGER DEFAULT 1,
    bypassed      INTEGER DEFAULT 0,
    template_name TEXT,
    strategy      TEXT,
    heuristic_fallback INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_rewrites_ts ON rewrites(ts);
CREATE INDEX IF NOT EXISTS idx_rewrites_session ON rewrites(session_id);
CREATE INDEX IF NOT EXISTS idx_rewrites_model ON rewrites(model_used);
CREATE INDEX IF NOT EXISTS idx_rewrites_profile ON rewrites(model_profile);

CREATE TABLE IF NOT EXISTS daily_stats (
    day         TEXT PRIMARY KEY,
    rewrites    INTEGER DEFAULT 0,
    approved    INTEGER DEFAULT 0,
    bypassed    INTEGER DEFAULT 0,
    avg_quality_before REAL,
    avg_quality_after  REAL,
    avg_token_delta_pct REAL,
    top_profile TEXT
);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add new columns to existing databases."""
    cursor = conn.execute("PRAGMA table_info(rewrites)")
    existing = {row[1] for row in cursor.fetchall()}
    migrations = {
        "template_name": "ALTER TABLE rewrites ADD COLUMN template_name TEXT",
        "strategy": "ALTER TABLE rewrites ADD COLUMN strategy TEXT",
        "heuristic_fallback": "ALTER TABLE rewrites ADD COLUMN heuristic_fallback INTEGER DEFAULT 0",
    }
    for col, sql in migrations.items():
        if col not in existing:
            try:
                conn.execute(sql)
            except Exception:
                pass
    conn.commit()


def _ensure_db() -> sqlite3.Connection:
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(METRICS_DB), check_same_thread=False)
    conn.executescript(_METRICS_SCHEMA)
    _migrate_schema(conn)
    conn.commit()
    return conn


def _db() -> sqlite3.Connection:
    return _ensure_db()


def _prune_old(days: int = 90) -> None:
    cutoff = time.time() - days * 86400
    conn = _db()
    try:
        conn.execute("DELETE FROM rewrites WHERE ts < ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()


def _record(session_id, platform, original, rewritten, quality_before,
            quality_after, token_delta_pct, model_profile, model_used,
            mode, approved=True, bypassed=False, template_name=None,
            strategy=None, heuristic_fallback=False):
    conn = _db()
    try:
        conn.execute(
            """INSERT INTO rewrites
            (ts, session_id, platform, original, rewritten,
             quality_before, quality_after, token_delta_pct,
             model_profile, model_used, mode, approved, bypassed,
             template_name, strategy, heuristic_fallback)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), session_id, platform, original, rewritten,
             quality_before, quality_after, token_delta_pct,
             model_profile, model_used, mode,
             1 if approved else 0, 1 if bypassed else 0,
             template_name, strategy,
             1 if heuristic_fallback else 0))
        conn.commit()
    finally:
        conn.close()
    if hash(str(time.time())) % 100 == 0:
        _prune_old()


def _fetch_rewrites(limit: int = 20, since: float = 0.0, approved_only: bool = False) -> List[Dict[str, Any]]:
    conn = _db()
    conn.row_factory = sqlite3.Row
    try:
        where = ["ts >= ?"]
        params: List[Any] = [since]
        if approved_only:
            where.append("approved = 1")
        sql = (
            "SELECT id, ts, session_id, platform, original, rewritten, "
            "quality_before, quality_after, token_delta_pct, model_profile, model_used, mode, approved, bypassed, "
            "template_name, strategy, heuristic_fallback "
            "FROM rewrites WHERE " + " AND ".join(where) + " ORDER BY ts DESC LIMIT ?"
        )
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _summary_for_period(period: str) -> Dict[str, Any]:
    since = _period_start(period)
    conn = _db()
    try:
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN approved=1 THEN 1 ELSE 0 END), "
            "AVG(quality_before), AVG(quality_after), AVG(token_delta_pct) "
            "FROM rewrites WHERE ts >= ?",
            (since,),
        ).fetchone()
        count, approved, qb, qa, td = row or (0, 0, 0, 0, 0)
        return {
            "period": period,
            "count": int(count or 0),
            "approved": int(approved or 0),
            "quality_before": float(qb or 0),
            "quality_after": float(qa or 0),
            "quality_delta": float((qa or 0) - (qb or 0)),
            "token_delta_pct": float(td or 0),
        }
    finally:
        conn.close()


def _analytics_rows(kind: str) -> List[Tuple[str, int, float, float, float]]:
    kind = (kind or "all").lower()
    conn = _db()
    try:
        if kind in {"daily", "day", "today"}:
            expr = "date(ts, 'unixepoch', 'localtime')"
            since = _now_ts() - 14 * 86400
            limit = 14
        elif kind in {"monthly", "month"}:
            expr = "strftime('%Y-%m', ts, 'unixepoch', 'localtime')"
            since = _now_ts() - 366 * 86400
            limit = 12
        else:
            expr = "strftime('%Y-W%W', ts, 'unixepoch', 'localtime')"
            since = _now_ts() - 10 * 7 * 86400
            limit = 10
        rows = conn.execute(
            f"SELECT {expr} AS bucket, COUNT(*), AVG(quality_before), AVG(quality_after), AVG(token_delta_pct) "
            "FROM rewrites WHERE ts >= ? GROUP BY bucket ORDER BY bucket DESC LIMIT ?",
            (since, limit),
        ).fetchall()
        return [(str(r[0]), int(r[1] or 0), float(r[2] or 0), float(r[3] or 0), float(r[4] or 0)) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Scoring (heuristic fallback)
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> Optional[str]:
    """Detect if text is non-English. Returns ISO 639-1 code or None.

    Uses langid if installed (fast, 97 languages). Falls back to a
    Unicode-range heuristic for CJK, Arabic, Cyrillic, etc.

    False positives (flagging English as non-English) are safe — the
    rewriter just sees a harmless "preserve the language" instruction
    and still rewrites English correctly. False negatives miss real
    non-English prompts, which is the actual problem we're solving.
    """
    if not text or not text.strip():
        return None

    # 1. Unicode-range scan for scripts that are definitively not English
    #    (CJK, Arabic, Cyrillic, Hebrew, Hangul, Thai, Devanagari, etc.)
    non_latin = 0
    for c in text:
        cp = ord(c)
        if cp > 127:
            if (0x4E00 <= cp <= 0x9FFF or   # CJK Unified
                0x0600 <= cp <= 0x06FF or   # Arabic
                0x0400 <= cp <= 0x04FF or   # Cyrillic
                0x0590 <= cp <= 0x05FF or   # Hebrew
                0xAC00 <= cp <= 0xD7AF or   # Hangul
                0x0E00 <= cp <= 0x0E7F or   # Thai
                0x0900 <= cp <= 0x097F or   # Devanagari
                0x3040 <= cp <= 0x309F or   # Hiragana
                0x30A0 <= cp <= 0x30FF):    # Katakana
                non_latin += 1
    text_len = max(1, len(text))
    if non_latin / text_len > 0.1:
        try:
            import langid
            lang, _ = langid.classify(text)
            return lang if lang != "en" else "other"
        except ImportError:
            return "other"

    # 2. Latin-script non-English (French, Spanish, German, etc.) via langid
    try:
        import langid
        ranked = langid.rank(text)
        top_lang, top_conf = ranked[0]
        delta = top_conf - ranked[1][1]
        # Higher threshold for short text, lower for longer text
        min_delta = 2.0 if text_len < 30 else 2.5
        if top_lang != "en" and delta > min_delta:
            # Safety: find English in the ranking and check how far it is.
            # If English is a very close second (within 15 points), the
            # text is likely English. Otherwise we're confident it's non-English.
            en_conf = next((conf for lang, conf in ranked if lang == "en"), None)
            if en_conf is not None and (top_conf - en_conf) < 15.0:
                return None  # English is too close — probable false positive
            return top_lang
        if top_lang == "en" and delta > min_delta:
            return None  # confidently English
    except ImportError:
        pass

    return None


def _score_prompt_heuristic(text: str) -> Tuple[float, int]:
    """Return (quality_score_0_100, estimated_tokens) via heuristic."""
    if not text:
        return 0.0, 0
    est_tokens = max(1, len(text) // 4)
    score = 0.0

    # Language-aware scoring: skip English-only checks for non-English text
    is_ne = _detect_language(text) is not None

    # 1. Clarity — English action-verb check is meaningless for non-English
    if not is_ne:
        action_verbs = {"write", "build", "fix", "refactor", "add", "remove", "explain",
                        "compare", "review", "test", "debug", "create", "generate",
                        "list", "find", "search", "update", "delete"}
        first_ten = " ".join(text.split()[:10]).lower()
        if any(v in first_ten for v in action_verbs):
            score += 20.0
    # 2. Structure — markdown/bullets work in any language
    if any(m in text for m in ("```", "\n- ", "1. ", "##", "<", "| ")):
        score += 20.0
    # 3. Specificity — digits, extensions, @mentions are universal
    if re.search(r"\d+|\.[a-zA-Z]{2,4}|@[a-zA-Z]|\bfile\b|\bpath\b|\bfunction\b", text):
        score += 20.0
    # 4. Conciseness — sentence-length analysis is language-agnostic
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
    if sentences:
        avg_len = est_tokens / max(1, len(sentences))
        if 8 <= avg_len <= 15:
            score += 20.0
        elif avg_len < 25:
            score += 10.0
    # 5. Context — English context words are meaningless for non-English
    if is_ne:
        score += 15.0  # default credit — can't meaningfully assess in heuristic
    elif any(m in text.lower() for m in ("previous", "earlier", "as we", "continue")):
        score += 20.0
    return min(100.0, score), est_tokens


# ---------------------------------------------------------------------------
# Optimizer engine
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """You rewrite user prompts for maximum clarity and token efficiency.

RULES:
1. Preserve ALL intent, constraints, and specific details.
2. Remove filler, fluff, and redundant politeness.
3. Be direct — start with the action verb.
4. Assign quality scores based on the nuance and strength of each dimension.

OUTPUT FORMAT:
Return the optimised prompt first, then a line containing exactly ---SCORES---,
then a JSON object with quality assessment per dimension (0-100 each):
  - clarity: how unambiguously the intent and desired outcome are expressed
  - specificity: how precise the constraints, inputs, and scope are
  - terminology: how well domain-specific language is used (or avoided if generic)
  - actionability: how clearly the reader knows what to do next
  - structure: how well the prompt is organised (sections, lists, ordering)

ORIGINAL PROMPT:
{original}

Rewritten prompt:"""


def _parse_rewrite_response(raw: str, original: str) -> Optional[Tuple[str, Dict]]:
    """Parse LLM response into (rewritten_prompt, scores_dict) or None."""
    if not raw:
        return None
    if "---SCORES---" not in raw:
        return None
    parts = raw.split("---SCORES---", 1)
    rewritten = parts[0].strip()
    json_part = parts[1].strip() if len(parts) > 1 else ""

    json_part = re.sub(r"^```(?:json)?\s*", "", json_part).strip()
    json_part = re.sub(r"\s*```$", "", json_part).strip()

    if not rewritten or not json_part:
        return None
    if rewritten.lower() == original.lower():
        return None

    try:
        scores = json.loads(json_part)
    except (json.JSONDecodeError, ValueError):
        return None

    return rewritten, scores


def _extract_composite(scores: Dict) -> float:
    """Extract composite 0-100 score from LLM quality scores.

    Validates that all 5 dimensions are present and 0-100.
    Returns -1.0 sentinel on bad/partial data (triggers heuristic fallback).
    """
    required = {"clarity", "specificity", "terminology", "actionability", "structure"}
    if not required.issubset(scores.keys()):
        return -1.0
    for key in required:
        val = scores[key]
        if not isinstance(val, (int, float)) or val < 0 or val > 100:
            return -1.0
    return sum(scores[k] for k in required) / len(required)


def _try_rewrite_sync(text: str, pllm: Any,
                       template: Optional[PromptTemplate] = None) -> Optional[Tuple[str, float]]:
    """Call pllm.complete to rewrite the user prompt + get quality scores.

    Returns (rewritten_text, composite_score) on success, or None on failure.
    Composite score is 0-100 from LLM quality assessment; -1.0 sentinel when JSON parsed
    but scores were invalid (caller should use heuristic fallback).
    """
    if pllm is None:
        logger.debug("prompt-optimizer: pllm not set — bypassing")
        return None

    tpl = template or select_template()
    user_msg = tpl.system_prompt.format(original=text)

    try:
        # The timeout kwarg only bounds a single provider request — the
        # auxiliary client's fallback chain can stack several ~90s provider
        # timeouts on top of it, freezing the turn for minutes. Run the call
        # on a worker thread and enforce OPTIMIZER_TIMEOUT_S as a hard
        # wall-clock cap, failing open to "no rewrite". A hung provider call
        # may leave the worker thread lingering until its own timeout fires;
        # that is the trade-off for never blocking the user's turn.
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prompt-optimizer-rewrite")
        future = pool.submit(
            pllm.complete,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=512,
            timeout=OPTIMIZER_TIMEOUT_S,
        )
        # wait=False: never block on a hung provider thread; it dies with
        # its own timeout while the user's turn proceeds without a rewrite.
        pool.shutdown(wait=False)
        try:
            result = future.result(timeout=OPTIMIZER_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.info(
                "prompt-optimizer: rewrite abandoned after %ss wall-clock cap",
                OPTIMIZER_TIMEOUT_S,
            )
            return None
        raw = result.text.strip() if result and result.text else None
        if not raw:
            return None

        parsed = _parse_rewrite_response(raw, text)
        if parsed is None:
            return None

        rewritten, scores = parsed
        composite = _extract_composite(scores)
        return rewritten, composite

    except Exception as exc:
        logger.info("prompt-optimizer: LLM call failed — %s", exc)
        return None


def _run_optimizer(original, model, provider, pllm: Any,
                   template: Optional[PromptTemplate] = None) -> Optional[RewriteRecord]:
    """Run the optimizer: heuristic score before, LLM rewrite + score after.

    Composes a MODEL-SPECIFIC GUIDANCE block into the rewriter system
    prompt based on the target model's (family, capability) profile.
    Unknown models fall through to the base template.
    """
    quality_before, tokens_before = _score_prompt_heuristic(original)
    logger.info("prompt-optimizer: trying rewrite for: %r", original[:80])

    family, capability = resolve_model_profile(model)
    base_template = template or select_template()
    guidance = _render_profile_guidance(model, family, capability)
    effective_template = _inject_guidance_into_template(base_template, guidance)

    # Language preservation: detect if the prompt is non-English and
    # instruct the rewriter to preserve the original language.
    detected = _detect_language(original)
    if detected:
        label = detected if detected != "other" else "a non-English"
        lang_block = (
            "\nLANGUAGE PRESERVATION:\n"
            f"The user's prompt is written in {label} language. "
            "Rewrite it in the SAME language. Do NOT translate it.\n"
        )
        effective_template = _inject_guidance_into_template(effective_template, lang_block)

    result = _try_rewrite_sync(original, pllm, template=effective_template)
    if not result:
        logger.info("prompt-optimizer: no rewrite produced")
        return None

    rewritten_text, llm_composite = result

    if llm_composite >= 0:
        quality_after = llm_composite
    else:
        quality_after, _ = _score_prompt_heuristic(rewritten_text)

    tokens_after = max(1, len(rewritten_text) // 4)
    token_delta_pct = round(
        ((tokens_before - tokens_after) / max(1, tokens_before)) * 100, 1)

    # Composite profile key: "claude/general", "openai/reasoning", or
    # "unknown" when neither axis resolved. Lets analytics group rewrites
    # by family+capability while the raw model string lives in `model_used`.
    if family is None and capability is None:
        profile_key = "unknown"
    else:
        profile_key = "{}/{}".format(family or "unknown", capability or "unknown")

    return RewriteRecord(
        original=original, rewritten=rewritten_text,
        quality_before=quality_before, quality_after=quality_after,
        token_delta_pct=token_delta_pct,
        model_profile=profile_key,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_diff(record):
    """Compact diff for interactive approval.

    Shows original→rewritten quality scores, token delta, and both
    versions clipped for display.
    """
    if record.token_delta_pct > 0:
        token_msg = f"saved -{abs(record.token_delta_pct):.0f}%"
    elif record.token_delta_pct < 0:
        token_msg = f"used +{abs(record.token_delta_pct):.0f}%"
    else:
        token_msg = "no token change"

    orig = _clip(record.original, 200)
    rwt = _clip(record.rewritten, 200)
    q_diff = record.quality_after - record.quality_before

    return (
        f"Prompt Optimizer — Diff\n"
        f"Quality: {record.quality_before:.0f} \N{RIGHTWARDS ARROW} "
        f"{record.quality_after:.0f} ({q_diff:+.0f})  |  "
        f"Tokens: {token_msg}\n\n"
        f"[Original]\n{orig}\n\n"
        f"[Rewritten]\n{rwt}\n\n"
        f"Reply Y to use  |  N to keep original"
    )


def _coerce_text(value: Any) -> str:
    """Best-effort conversion for plugin inputs that sometimes arrive as tuples/lists."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_coerce_text(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "message", "value"):
            if key in value:
                return _coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# ---------------------------------------------------------------------------
# Report utilities
# ---------------------------------------------------------------------------

def _normalise_args(raw_args: Any) -> List[str]:
    text = _coerce_text(raw_args).strip()
    return text.split() if text else []


def _now_ts() -> float:
    return time.time()


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts or 0), tz=_LONDON)


def _fmt_ts(ts: float) -> str:
    return _dt(ts).strftime("%d/%m/%y %H:%M:%S")


def _fmt_day(ts: float) -> str:
    return _dt(ts).strftime("%d/%m/%y")


def _period_start(period: str) -> float:
    now = datetime.now(tz=_LONDON)
    period = (period or "week").lower()
    if period in {"today", "day", "daily"}:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period in {"month", "monthly"}:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period in {"all", "total"}:
        return 0.0
    else:
        start = now - timedelta(days=7)
    return start.timestamp()


def _limit_from_args(argv: Sequence[str], default: int = 5, maximum: int = 20) -> int:
    for idx, arg in enumerate(argv):
        if arg == "--limit" and idx + 1 < len(argv):
            try:
                return max(1, min(maximum, int(argv[idx + 1])))
            except ValueError:
                return default
        if arg.startswith("--limit="):
            try:
                return max(1, min(maximum, int(arg.split("=", 1)[1])))
            except ValueError:
                return default
    return default


def _delta_words(delta: float) -> str:
    if delta > 0:
        return f"saved {delta:.1f}%"
    if delta < 0:
        return f"expanded {abs(delta):.1f}%"
    return "flat"


def _quality_delta(row: Dict[str, Any]) -> float:
    return float(row.get("quality_after") or 0) - float(row.get("quality_before") or 0)


def _clip(text: Any, width: int = 110) -> str:
    value = " ".join(_coerce_text(text).split())
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)].rstrip() + "…"


def _wrap_block(text: Any, indent: str = "  ", width: int = 96) -> List[str]:
    value = _coerce_text(text).strip()
    if not value:
        return [indent + "(empty)"]
    lines: List[str] = []
    for paragraph in value.splitlines() or [value]:
        wrapped = textwrap.wrap(paragraph, width=width) or [""]
        lines.extend(indent + line for line in wrapped)
    return lines


def _bar(before: float, after: float, width: int = 16) -> str:
    before = max(0, min(100, float(before or 0)))
    after = max(0, min(100, float(after or 0)))
    filled = int(round(after / 100 * width))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _render_header(title: str, subtitle: str = "") -> List[str]:
    stamp = datetime.now(tz=_LONDON).strftime("%d/%m/%y %H:%M:%S")
    lines = [f"{title} · {stamp}"]
    if subtitle:
        lines.append(subtitle)
    lines.append("─" * max(64, min(100, len(lines[0]))))
    return lines


def _render_summary_cards() -> List[str]:
    today = _summary_for_period("daily")
    week = _summary_for_period("weekly")
    month = _summary_for_period("monthly")
    lines = ["Overview"]
    lines.append("  Period       Rewrites  Quality      Tokens")
    for label, data in (("Today", today), ("7 days", week), ("Month", month)):
        q = f"{data['quality_before']:.0f}→{data['quality_after']:.0f} ({data['quality_delta']:+.0f})"
        lines.append(f"  {label:<11} {data['count']:>7}  {q:<12} {_delta_words(data['token_delta_pct'])}")
    return lines


def _render_comparisons(rows: Sequence[Dict[str, Any]], *, title: str = "Before / after comparisons") -> List[str]:
    lines = [title]
    if not rows:
        return lines + ["  No rewrites recorded yet."]
    for idx, row in enumerate(rows, 1):
        q_before = float(row.get("quality_before") or 0)
        q_after = float(row.get("quality_after") or 0)
        lines.append("")
        lines.append(
            f"  #{idx} id={row.get('id')} · {_fmt_ts(row.get('ts') or 0)} · "
            f"{row.get('platform') or 'unknown'} · quality {q_before:.0f}→{q_after:.0f} "
            f"({_quality_delta(row):+.0f}) · tokens {_delta_words(float(row.get('token_delta_pct') or 0))}"
        )
        lines.append(f"  {_bar(q_before, q_after)}")
        lines.append("  Before")
        lines.extend(_wrap_block(row.get("original"), indent="    "))
        lines.append("  After")
        lines.extend(_wrap_block(row.get("rewritten"), indent="    "))
    return lines


def _render_suggestion_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["Prompt replacement suggestions"]
    if not rows:
        return lines + ["  No suggestions yet. Send a few prompts first."]
    for idx, row in enumerate(rows, 1):
        gain = _quality_delta(row)
        lines.append("")
        lines.append(f"  {idx}. Replace this pattern · quality {gain:+.0f} · tokens {_delta_words(float(row.get('token_delta_pct') or 0))}")
        lines.append(f"     Instead of: {_clip(row.get('original'), 100)}")
        lines.append(f"     Use:        {_clip(row.get('rewritten'), 100)}")
    return lines


def _render_insight_bullets(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["Optimization insights"]
    if not rows:
        return lines + ["  No data yet."]
    expanded = [r for r in rows if float(r.get("token_delta_pct") or 0) < 0]
    saved = [r for r in rows if float(r.get("token_delta_pct") or 0) > 0]
    avg_gain = sum(_quality_delta(r) for r in rows) / max(1, len(rows))
    avg_tokens = sum(float(r.get("token_delta_pct") or 0) for r in rows) / max(1, len(rows))
    lines.append(f"  • Quality is improving by {avg_gain:+.1f} points on average across recent prompts.")
    if avg_tokens < 0:
        lines.append("  • The optimiser is often expanding prompts. That can improve clarity, but it is not pure token saving.")
    else:
        lines.append("  • The optimiser is reducing token load while preserving stronger structure.")
    if expanded:
        lines.append(f"  • {len(expanded)}/{len(rows)} recent rewrites expanded the prompt. Watch for over-engineered short commands.")
    if saved:
        best = max(saved, key=lambda r: float(r.get("token_delta_pct") or 0))
        lines.append(f"  • Best token save: {_delta_words(float(best.get('token_delta_pct') or 0))} on id={best.get('id')}.")
    top = max(rows, key=_quality_delta)
    lines.append(f"  • Best quality gain: {_quality_delta(top):+.0f} on id={top.get('id')}.")
    lines.append("  • Use /quick, *simple, or #basic to bypass optimisation for tiny commands.")
    return lines


def _display_bucket(bucket: str, mode: str) -> str:
    try:
        if mode in {"daily", "day", "today"}:
            return datetime.strptime(bucket, "%Y-%m-%d").strftime("%d/%m/%y")
        if mode in {"monthly", "month"}:
            return datetime.strptime(bucket + "-01", "%Y-%m-%d").strftime("01/%m/%y")
        if "-W" in bucket:
            year, week = bucket.split("-W", 1)
            start = datetime.strptime(f"{year} {int(week)} 1", "%Y %W %w")
            return start.strftime("%d/%m/%y")
    except Exception:
        pass
    return bucket


def _render_analytics(kind: str = "all") -> str:
    argv_kind = (kind or "all").lower()
    modes = ["daily", "weekly", "monthly"] if argv_kind in {"", "all"} else [argv_kind]
    lines = _render_header("Prompt Analytics", "Daily, weekly, and monthly optimisation metrics")
    for mode in modes:
        rows = _analytics_rows(mode)
        lines.append("")
        lines.append(mode.capitalize())
        if not rows:
            lines.append("  No data.")
            continue
        lines.append("  Period       Rewrites  Quality      Tokens")
        for bucket, count, qb, qa, td in rows:
            label = _display_bucket(bucket, mode)
            lines.append(f"  {label:<12} {count:>7}  {qb:.0f}→{qa:.0f} ({qa-qb:+.0f})  {_delta_words(td)}")
    return "\n".join(lines)


def _build_full_report(limit: int = 5) -> str:
    recent = _fetch_rewrites(limit=max(limit, 10), since=0, approved_only=False)
    suggestions = sorted(recent, key=_quality_delta, reverse=True)[:limit]
    lines = _render_header(
        "Prompt Optimizer Insights",
        "Backend-wired prompt rewrites surfaced as a CLI/TUI report",
    )
    lines.extend(_render_summary_cards())
    lines.append("")
    lines.extend(_render_insight_bullets(recent[:10]))
    lines.append("")
    lines.extend(_render_suggestion_rows(suggestions))
    lines.append("")
    lines.extend(_render_comparisons(recent[:limit]))
    lines.append("")
    lines.append("Discoverability")
    lines.append("  /prompt-insights [--html]       Full report")
    lines.append("  /prompt-compare [--limit N]     Before/after comparisons")
    lines.append("  /prompt-suggestions [--limit N] Replacement suggestions")
    lines.append("  /prompt-analytics [all|daily|weekly|monthly]")
    return "\n".join(lines)


def _write_html_report(report_text: str) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=_LONDON).strftime("%Y-%m-%d-%H%M%S")
    path = _REPORT_DIR / f"report-{stamp}.html"
    body = html.escape(report_text)
    doc = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Prompt Optimizer Insights</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: #0f1115; color: #e8e6df; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 32px; }}
  pre {{ white-space: pre-wrap; line-height: 1.45; background: #171a21; border: 1px solid #2d3340; border-radius: 16px; padding: 24px; box-shadow: 0 18px 60px rgba(0,0,0,.35); }}
  .tag {{ color: #fbbf24; margin-bottom: 16px; }}
</style>
</head>
<body><main><div class=\"tag\">Prompt Optimizer Insights</div><pre>{body}</pre></main></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return str(path)
