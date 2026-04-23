"""System prompt assembly -- identity, platform hints, skills index, context files.

All functions are stateless. AIAgent._build_system_prompt() calls these to
assemble pieces, then combines them with memory and ephemeral prompts.
"""

import json
import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from hermes_constants import get_hermes_home, get_skills_dir, is_wsl
from typing import Any, Optional

from agent.archetypes import Archetype, resolve_archetype
from agent.context_rules import (
    MAX_HERMES_CONTEXT_LAYERS,
    apply_bounded_hierarchical_context,
    discover_hermes_context_layers,
)
from agent.route_categories import (
    LiteralCategory,
    RouteCategory,
    get_route_category,
    resolve_literal_category,
    resolve_literal_category_from_route_category,
)
from agent.runtime_modes import RuntimeMode, resolve_runtime_mode
from agent.skill_utils import (
    extract_skill_conditions,
    extract_skill_description,
    get_all_skills_dirs,
    get_disabled_skill_names,
    iter_skill_index_files,
    parse_frontmatter,
    skill_matches_platform,
)
from agent.task_contracts import TaskContract, validate_task_contract
from utils import atomic_json_write

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptOverlaySection:
    """Structured prompt section used for deterministic overlay assembly."""

    key: str
    title: str
    content: str


WAVE1_OVERLAY_ORDER = (
    "archetype",
    "category",
    "route_category",
    "delegation_profile",
    "runtime_mode",
    "skills",
    "task_contract",
    "orchestration_hints",
)

_WAVE1_OVERLAY_TITLES = {
    "archetype": "Archetype",
    "category": "Category",
    "route_category": "Route Category",
    "delegation_profile": "Delegation Profile",
    "runtime_mode": "Runtime Mode",
    "skills": "Skills",
    "task_contract": "Task Contract",
    "orchestration_hints": "Orchestration Hints",
}


def _normalize_overlay_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        raise TypeError("Overlay list values must be a string or list-like collection")
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _format_json_block(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_archetype_overlay(archetype: Archetype) -> str:
    return "\n".join(
        [
            f"name: {archetype.name}",
            f"summary: {archetype.summary}",
            f"default_route_category: {archetype.default_route_category}",
            f"default_delegation_profile: {archetype.default_delegation_profile}",
            "default_skills:",
            *[f"- {skill}" for skill in archetype.default_skills],
            "default_required_tools:",
            *[f"- {tool_name}" for tool_name in archetype.default_required_tools],
            f"permission_preset: {archetype.permission_preset}",
            f"fallback_policy: {archetype.fallback_policy}",
        ]
    )


def _coerce_literal_category_overlay(
    category: Any,
    *,
    route_category_name: str,
) -> LiteralCategory | dict[str, str]:
    if isinstance(category, LiteralCategory):
        payload = {
            "name": category.name,
            "summary": category.summary,
            "mapped_route_category": category.route_category,
            "default_runtime_mode": category.default_runtime_mode or "inherit",
        }
    elif isinstance(category, dict):
        name = str(category.get("name") or "").strip()
        if not name:
            raise ValueError("category overlay dict requires a non-empty name")
        payload = {
            "name": name,
            "summary": str(category.get("summary") or "").strip(),
            "mapped_route_category": str(category.get("mapped_route_category") or route_category_name).strip() or route_category_name,
            "default_runtime_mode": str(category.get("default_runtime_mode") or "inherit").strip() or "inherit",
        }
    else:
        resolved = resolve_literal_category(category)
        payload = {
            "name": resolved.name,
            "summary": resolved.summary,
            "mapped_route_category": resolved.route_category,
            "default_runtime_mode": resolved.default_runtime_mode or "inherit",
        }

    mapped_route = get_route_category(payload["mapped_route_category"])
    payload["fallback_semantics"] = "inherits_mapped_route_category"
    payload["mapped_route_fallbacks"] = ", ".join(mapped_route.fallback_models) if mapped_route.fallback_models else "(none)"
    return payload


def _format_literal_category_overlay(category: LiteralCategory | dict[str, str]) -> str:
    if isinstance(category, LiteralCategory):
        mapped_route = get_route_category(category.route_category)
        payload = {
            "name": category.name,
            "summary": category.summary,
            "mapped_route_category": category.route_category,
            "default_runtime_mode": category.default_runtime_mode or "inherit",
            "fallback_semantics": "inherits_mapped_route_category",
            "mapped_route_fallbacks": ", ".join(mapped_route.fallback_models) if mapped_route.fallback_models else "(none)",
        }
    else:
        payload = category
    return "\n".join(
        [
            f"name: {payload['name']}",
            f"summary: {payload['summary']}",
            f"mapped_route_category: {payload['mapped_route_category']}",
            f"default_runtime_mode: {payload['default_runtime_mode']}",
            f"fallback_semantics: {payload['fallback_semantics']}",
            f"mapped_route_fallbacks: {payload['mapped_route_fallbacks']}",
        ]
    )


def _coerce_route_category_overlay(route_category: Any) -> RouteCategory | dict[str, str]:
    if isinstance(route_category, RouteCategory):
        return route_category
    if isinstance(route_category, dict):
        name = str(route_category.get("name") or "").strip()
        if not name:
            raise ValueError("route_category overlay dict requires a non-empty name")
        return {
            "name": name,
            "summary": str(route_category.get("summary") or "").strip(),
            "intensity": str(route_category.get("intensity") or "").strip(),
        }
    return get_route_category(str(route_category).strip())


def _format_route_category_overlay(route_category: RouteCategory | dict[str, str]) -> str:
    if isinstance(route_category, RouteCategory):
        payload = {
            "name": route_category.name,
            "summary": route_category.summary,
            "intensity": route_category.intensity,
        }
    else:
        payload = route_category
    return "\n".join(
        [
            f"name: {payload['name']}",
            f"summary: {payload['summary']}",
            f"intensity: {payload['intensity']}",
        ]
    )


def _coerce_runtime_mode_overlay(runtime_mode: Any) -> RuntimeMode | dict[str, str]:
    if isinstance(runtime_mode, RuntimeMode):
        return runtime_mode
    if isinstance(runtime_mode, dict):
        name = str(runtime_mode.get("name") or "").strip()
        if not name:
            raise ValueError("runtime_mode overlay dict requires a non-empty name")
        return {
            "name": name,
            "description": str(runtime_mode.get("description") or "").strip(),
            "operating_posture": str(runtime_mode.get("operating_posture") or "").strip(),
            "kind": str(runtime_mode.get("kind") or "runtime_mode").strip() or "runtime_mode",
        }
    return resolve_runtime_mode(str(runtime_mode).strip())


def _format_runtime_mode_overlay(runtime_mode: RuntimeMode | dict[str, str]) -> str:
    if isinstance(runtime_mode, RuntimeMode):
        payload = {
            "name": runtime_mode.name,
            "description": runtime_mode.description,
            "operating_posture": runtime_mode.operating_posture,
            "kind": runtime_mode.kind,
        }
    else:
        payload = runtime_mode
    return "\n".join(
        [
            f"name: {payload['name']}",
            f"description: {payload['description']}",
            f"operating_posture: {payload['operating_posture']}",
            f"kind: {payload['kind']}",
        ]
    )


def _format_delegation_profile_overlay(delegation_profile: str) -> str:
    return f"name: {delegation_profile}"


def _format_skills_overlay(skills: list[str]) -> str:
    return "\n".join(f"- {skill_name}" for skill_name in skills)


def _format_orchestration_hints_overlay(orchestration_hints: Any) -> str:
    if isinstance(orchestration_hints, str):
        return orchestration_hints.strip()
    if isinstance(orchestration_hints, (list, tuple, dict)):
        return _format_json_block(orchestration_hints)
    raise TypeError("orchestration_hints must be a string, list, tuple, or dict")


def normalize_wave1_overlay_inputs(
    *,
    archetype_name: str | None = None,
    category: Any = None,
    route_category: Any = None,
    delegation_profile: str | None = None,
    runtime_mode: Any = None,
    skills: Any = None,
    task_contract: dict[str, Any] | TaskContract | None = None,
    orchestration_hints: Any = None,
) -> dict[str, Any]:
    """Canonicalize Wave 1 overlay inputs before any rendering occurs."""

    archetype = resolve_archetype(archetype_name)
    resolved_route_category = _coerce_route_category_overlay(
        route_category if route_category not in (None, "") else archetype.default_route_category
    )
    route_category_name = getattr(resolved_route_category, "name", None)
    if route_category_name is None:
        route_category_name = resolved_route_category["name"]
    resolved_category = _coerce_literal_category_overlay(
        category if category not in (None, "") else resolve_literal_category_from_route_category(route_category_name),
        route_category_name=route_category_name,
    )
    resolved_delegation_profile = (
        str(delegation_profile).strip()
        if delegation_profile and str(delegation_profile).strip()
        else archetype.default_delegation_profile
    )
    resolved_runtime_mode = _coerce_runtime_mode_overlay(runtime_mode or None)
    explicit_skills = _normalize_overlay_list(skills)
    resolved_skills = list(archetype.default_skills)
    resolved_skills.extend(explicit_skills)
    resolved_skills = list(dict.fromkeys(resolved_skills))

    validated_contract = None
    if task_contract is not None:
        validated_contract = validate_task_contract(task_contract).model_dump()

    rendered_orchestration_hints = None
    if orchestration_hints is not None:
        rendered_orchestration_hints = _format_orchestration_hints_overlay(orchestration_hints)
        if not rendered_orchestration_hints:
            rendered_orchestration_hints = None

    category_name = getattr(resolved_category, "name", None)
    if category_name is None:
        category_name = resolved_category["name"]

    runtime_mode_name = getattr(resolved_runtime_mode, "name", None)
    if runtime_mode_name is None:
        runtime_mode_name = resolved_runtime_mode["name"]

    return {
        "archetype": archetype.name,
        "archetype_definition": archetype,
        "category": category_name,
        "category_definition": resolved_category,
        "route_category": route_category_name,
        "route_category_definition": resolved_route_category,
        "delegation_profile": resolved_delegation_profile,
        "runtime_mode": runtime_mode_name,
        "runtime_mode_definition": resolved_runtime_mode,
        "skills": resolved_skills,
        "skills_explicit": bool(explicit_skills),
        "task_contract": validated_contract,
        "orchestration_hints": orchestration_hints,
        "orchestration_hints_rendered": rendered_orchestration_hints,
    }


def build_wave1_overlay_sections_from_normalized(normalized_inputs: dict[str, Any]) -> list[PromptOverlaySection]:
    """Render Wave 1 overlay sections from canonical normalized inputs."""

    sections: list[PromptOverlaySection] = [
        PromptOverlaySection(
            key="archetype",
            title=_WAVE1_OVERLAY_TITLES["archetype"],
            content=_format_archetype_overlay(normalized_inputs["archetype_definition"]),
        ),
        PromptOverlaySection(
            key="category",
            title=_WAVE1_OVERLAY_TITLES["category"],
            content=_format_literal_category_overlay(normalized_inputs["category_definition"]),
        ),
        PromptOverlaySection(
            key="route_category",
            title=_WAVE1_OVERLAY_TITLES["route_category"],
            content=_format_route_category_overlay(normalized_inputs["route_category_definition"]),
        ),
        PromptOverlaySection(
            key="delegation_profile",
            title=_WAVE1_OVERLAY_TITLES["delegation_profile"],
            content=_format_delegation_profile_overlay(normalized_inputs["delegation_profile"]),
        ),
        PromptOverlaySection(
            key="runtime_mode",
            title=_WAVE1_OVERLAY_TITLES["runtime_mode"],
            content=_format_runtime_mode_overlay(normalized_inputs["runtime_mode_definition"]),
        ),
    ]

    if normalized_inputs.get("skills_explicit"):
        sections.append(
            PromptOverlaySection(
                key="skills",
                title=_WAVE1_OVERLAY_TITLES["skills"],
                content=_format_skills_overlay(normalized_inputs["skills"]),
            )
        )

    if normalized_inputs.get("task_contract") is not None:
        sections.append(
            PromptOverlaySection(
                key="task_contract",
                title=_WAVE1_OVERLAY_TITLES["task_contract"],
                content=_format_json_block(normalized_inputs["task_contract"]),
            )
        )

    if normalized_inputs.get("orchestration_hints_rendered"):
        sections.append(
            PromptOverlaySection(
                key="orchestration_hints",
                title=_WAVE1_OVERLAY_TITLES["orchestration_hints"],
                content=normalized_inputs["orchestration_hints_rendered"],
            )
        )

    section_order = tuple(section.key for section in sections)
    expected_prefix = tuple(key for key in WAVE1_OVERLAY_ORDER if key in section_order)
    if section_order != expected_prefix:
        raise AssertionError(
            f"Wave 1 overlay order violation: expected {expected_prefix}, got {section_order}"
        )

    return sections


def build_wave1_overlay_prompt_from_normalized(normalized_inputs: dict[str, Any]) -> str:
    sections = build_wave1_overlay_sections_from_normalized(normalized_inputs)
    if not sections:
        return ""
    rendered_sections = [f"## {section.title}\n{section.content}" for section in sections]
    return "# Wave 1 Prompt Overlays\n\n" + "\n\n".join(rendered_sections)


def build_wave1_overlay_sections(
    *,
    archetype_name: str | None = None,
    category: Any = None,
    route_category: Any = None,
    delegation_profile: str | None = None,
    runtime_mode: Any = None,
    skills: Any = None,
    task_contract: dict[str, Any] | TaskContract | None = None,
    orchestration_hints: Any = None,
) -> list[PromptOverlaySection]:
    """Build the Wave 1 prompt overlays in the exact locked order.

    Ordering is structural and deterministic so downstream delegation code can
    rely on this helper instead of hand-assembling prompt fragments.
    """

    normalized_inputs = normalize_wave1_overlay_inputs(
        archetype_name=archetype_name,
        category=category,
        route_category=route_category,
        delegation_profile=delegation_profile,
        runtime_mode=runtime_mode,
        skills=skills,
        task_contract=task_contract,
        orchestration_hints=orchestration_hints,
    )
    return build_wave1_overlay_sections_from_normalized(normalized_inputs)


def build_wave1_overlay_prompt(
    *,
    archetype_name: str | None = None,
    category: Any = None,
    route_category: Any = None,
    delegation_profile: str | None = None,
    runtime_mode: Any = None,
    skills: Any = None,
    task_contract: dict[str, Any] | TaskContract | None = None,
    orchestration_hints: Any = None,
) -> str:
    """Render the exact Wave 1 overlays as a prompt block.

    `task_contract` is preserved as canonical structured JSON inside its own
    section rather than flattened into prose.
    """

    normalized_inputs = normalize_wave1_overlay_inputs(
        archetype_name=archetype_name,
        category=category,
        route_category=route_category,
        delegation_profile=delegation_profile,
        runtime_mode=runtime_mode,
        skills=skills,
        task_contract=task_contract,
        orchestration_hints=orchestration_hints,
    )
    return build_wave1_overlay_prompt_from_normalized(normalized_inputs)

# ---------------------------------------------------------------------------
# Context file scanning — detect prompt injection in AGENTS.md, .cursorrules,
# SOUL.md before they get injected into the system prompt.
# ---------------------------------------------------------------------------

_CONTEXT_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->', "html_comment_injection"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none', "hidden_div"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)', "translate_execute"),
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)', "read_secrets"),
]

_CONTEXT_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_context_content(content: str, filename: str) -> str:
    """Scan context file content for injection. Returns sanitized content."""
    findings = []

    # Check invisible unicode
    for char in _CONTEXT_INVISIBLE_CHARS:
        if char in content:
            findings.append(f"invisible unicode U+{ord(char):04X}")

    # Check threat patterns
    for pattern, pid in _CONTEXT_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            findings.append(pid)

    if findings:
        logger.warning("Context file %s blocked: %s", filename, ", ".join(findings))
        return f"[BLOCKED: {filename} contained potential prompt injection ({', '.join(findings)}). Content not loaded.]"

    return content


def _find_git_root(start: Path) -> Optional[Path]:
    """Walk *start* and its parents looking for a ``.git`` directory.

    Returns the directory containing ``.git``, or ``None`` if we hit the
    filesystem root without finding one.
    """
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


_HERMES_MD_NAMES = (".hermes.md", "HERMES.md")


def _find_hermes_md(cwd: Path) -> Optional[Path]:
    """Discover the nearest ``.hermes.md`` or ``HERMES.md``.

    Search order: *cwd* first, then each parent directory up to (and
    including) the git repository root.  Returns the first match, or
    ``None`` if nothing is found.
    """
    stop_at = _find_git_root(cwd)
    current = cwd.resolve()

    for directory in [current, *current.parents]:
        for name in _HERMES_MD_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        # Stop walking at the git root (or filesystem root).
        if stop_at and directory == stop_at:
            break
    return None


def _strip_yaml_frontmatter(content: str) -> str:
    """Remove optional YAML frontmatter (``---`` delimited) from *content*.

    The frontmatter may contain structured config (model overrides, tool
    settings) that will be handled separately in a future PR.  For now we
    strip it so only the human-readable markdown body is injected into the
    system prompt.
    """
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            # Skip past the closing --- and any trailing newline
            body = content[end + 4:].lstrip("\n")
            return body if body else content
    return content


# =========================================================================
# Constants
# =========================================================================

DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide "
    "range of tasks including answering questions, writing and editing code, "
    "analyzing information, creative work, and executing actions via your tools. "
    "You communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose unless otherwise directed below. "
    "Be targeted and efficient in your exploration and investigations."
)

MEMORY_GUIDANCE = (
    "You have persistent memory across sessions. Save durable facts using the memory "
    "tool: user preferences, environment details, tool quirks, and stable conventions. "
    "Memory is injected into every turn, so keep it compact and focused on facts that "
    "will still matter later.\n"
    "Prioritize what reduces future user steering — the most valuable memory is one "
    "that prevents the user from having to correct or remind you again. "
    "User preferences and recurring corrections matter more than procedural task details.\n"
    "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
    "state to memory; use session_search to recall those from past transcripts. "
    "If you've discovered a new way to do something, solved a problem that could be "
    "necessary later, save it as a skill with the skill tool."
)

SESSION_SEARCH_GUIDANCE = (
    "When the user references something from a past conversation or you suspect "
    "relevant cross-session context exists, use session_search to recall it before "
    "asking them to repeat themselves."
)

SKILLS_GUIDANCE = (
    "After completing a complex task (5+ tool calls), fixing a tricky error, "
    "or discovering a non-trivial workflow, save the approach as a "
    "skill with skill_manage so you can reuse it next time.\n"
    "When using a skill and finding it outdated, incomplete, or wrong, "
    "patch it immediately with skill_manage(action='patch') — don't wait to be asked. "
    "Skills that aren't maintained become liabilities."
)

TOOL_USE_ENFORCEMENT_GUIDANCE = (
    "# Tool-use enforcement\n"
    "You MUST use your tools to take action — do not describe what you would do "
    "or plan to do without actually doing it. When you say you will perform an "
    "action (e.g. 'I will run the tests', 'Let me check the file', 'I will create "
    "the project'), you MUST immediately make the corresponding tool call in the same "
    "response. Never end your turn with a promise of future action — execute it now.\n"
    "Keep working until the task is actually complete. Do not stop with a summary of "
    "what you plan to do next time. If you have tools available that can accomplish "
    "the task, use them instead of telling the user what you would do.\n"
    "Every response should either (a) contain tool calls that make progress, or "
    "(b) deliver a final result to the user. Responses that only describe intentions "
    "without acting are not acceptable."
)

# Model name substrings that trigger tool-use enforcement guidance.
# Add new patterns here when a model family needs explicit steering.
TOOL_USE_ENFORCEMENT_MODELS = ("gpt", "codex", "gemini", "gemma", "grok")

# OpenAI GPT/Codex-specific execution guidance.  Addresses known failure modes
# where GPT models abandon work on partial results, skip prerequisite lookups,
# hallucinate instead of using tools, and declare "done" without verification.
# Inspired by patterns from OpenAI's GPT-5.4 prompting guide & OpenClaw PR #38953.
OPENAI_MODEL_EXECUTION_GUIDANCE = (
    "# Execution discipline\n"
    "<tool_persistence>\n"
    "- Use tools whenever they improve correctness, completeness, or grounding.\n"
    "- Do not stop early when another tool call would materially improve the result.\n"
    "- If a tool returns empty or partial results, retry with a different query or "
    "strategy before giving up.\n"
    "- Keep calling tools until: (1) the task is complete, AND (2) you have verified "
    "the result.\n"
    "</tool_persistence>\n"
    "\n"
    "<mandatory_tool_use>\n"
    "NEVER answer these from memory or mental computation — ALWAYS use a tool:\n"
    "- Arithmetic, math, calculations → use terminal or execute_code\n"
    "- Hashes, encodings, checksums → use terminal (e.g. sha256sum, base64)\n"
    "- Current time, date, timezone → use terminal (e.g. date)\n"
    "- System state: OS, CPU, memory, disk, ports, processes → use terminal\n"
    "- File contents, sizes, line counts → use read_file, search_files, or terminal\n"
    "- Git history, branches, diffs → use terminal\n"
    "- Current facts (weather, news, versions) → use web_search\n"
    "Your memory and user profile describe the USER, not the system you are "
    "running on. The execution environment may differ from what the user profile "
    "says about their personal setup.\n"
    "</mandatory_tool_use>\n"
    "\n"
    "<act_dont_ask>\n"
    "When a question has an obvious default interpretation, act on it immediately "
    "instead of asking for clarification. Examples:\n"
    "- 'Is port 443 open?' → check THIS machine (don't ask 'open where?')\n"
    "- 'What OS am I running?' → check the live system (don't use user profile)\n"
    "- 'What time is it?' → run `date` (don't guess)\n"
    "Only ask for clarification when the ambiguity genuinely changes what tool "
    "you would call.\n"
    "</act_dont_ask>\n"
    "\n"
    "<prerequisite_checks>\n"
    "- Before taking an action, check whether prerequisite discovery, lookup, or "
    "context-gathering steps are needed.\n"
    "- Do not skip prerequisite steps just because the final action seems obvious.\n"
    "- If a task depends on output from a prior step, resolve that dependency first.\n"
    "</prerequisite_checks>\n"
    "\n"
    "<verification>\n"
    "Before finalizing your response:\n"
    "- Correctness: does the output satisfy every stated requirement?\n"
    "- Grounding: are factual claims backed by tool outputs or provided context?\n"
    "- Formatting: does the output match the requested format or schema?\n"
    "- Safety: if the next step has side effects (file writes, commands, API calls), "
    "confirm scope before executing.\n"
    "</verification>\n"
    "\n"
    "<missing_context>\n"
    "- If required context is missing, do NOT guess or hallucinate an answer.\n"
    "- Use the appropriate lookup tool when missing information is retrievable "
    "(search_files, web_search, read_file, etc.).\n"
    "- Ask a clarifying question only when the information cannot be retrieved by tools.\n"
    "- If you must proceed with incomplete information, label assumptions explicitly.\n"
    "</missing_context>"
)

# Gemini/Gemma-specific operational guidance, adapted from OpenCode's gemini.txt.
# Injected alongside TOOL_USE_ENFORCEMENT_GUIDANCE when the model is Gemini or Gemma.
GOOGLE_MODEL_OPERATIONAL_GUIDANCE = (
    "# Google model operational directives\n"
    "Follow these operational rules strictly:\n"
    "- **Absolute paths:** Always construct and use absolute file paths for all "
    "file system operations. Combine the project root with relative paths.\n"
    "- **Verify first:** Use read_file/search_files to check file contents and "
    "project structure before making changes. Never guess at file contents.\n"
    "- **Dependency checks:** Never assume a library is available. Check "
    "package.json, requirements.txt, Cargo.toml, etc. before importing.\n"
    "- **Conciseness:** Keep explanatory text brief — a few sentences, not "
    "paragraphs. Focus on actions and results over narration.\n"
    "- **Parallel tool calls:** When you need to perform multiple independent "
    "operations (e.g. reading several files), make all the tool calls in a "
    "single response rather than sequentially.\n"
    "- **Non-interactive commands:** Use flags like -y, --yes, --non-interactive "
    "to prevent CLI tools from hanging on prompts.\n"
    "- **Keep going:** Work autonomously until the task is fully resolved. "
    "Don't stop with a plan — execute it.\n"
)

# Model name substrings that should use the 'developer' role instead of
# 'system' for the system prompt.  OpenAI's newer models (GPT-5, Codex)
# give stronger instruction-following weight to the 'developer' role.
# The swap happens at the API boundary in _build_api_kwargs() so internal
# message representation stays consistent ("system" everywhere).
DEVELOPER_ROLE_MODELS = ("gpt-5", "codex")

PLATFORM_HINTS = {
    "whatsapp": (
        "You are on a text messaging communication platform, WhatsApp. "
        "Please do not use markdown as it does not render. "
        "You can send media files natively: to deliver a file to the user, "
        "include MEDIA:/absolute/path/to/file in your response. The file "
        "will be sent as a native WhatsApp attachment — images (.jpg, .png, "
        ".webp) appear as photos, videos (.mp4, .mov) play inline, and other "
        "files arrive as downloadable documents. You can also include image "
        "URLs in markdown format ![alt](url) and they will be sent as photos."
    ),
    "telegram": (
        "You are on a text messaging communication platform, Telegram. "
        "Standard markdown is automatically converted to Telegram format. "
        "Supported: **bold**, *italic*, ~~strikethrough~~, ||spoiler||, "
        "`inline code`, ```code blocks```, [links](url), and ## headers. "
        "You can send media files natively: to deliver a file to the user, "
        "include MEDIA:/absolute/path/to/file in your response. Images "
        "(.png, .jpg, .webp) appear as photos, audio (.ogg) sends as voice "
        "bubbles, and videos (.mp4) play inline. You can also include image "
        "URLs in markdown format ![alt](url) and they will be sent as native photos."
    ),
    "discord": (
        "You are in a Discord server or group chat communicating with your user. "
        "You can send media files natively: include MEDIA:/absolute/path/to/file "
        "in your response. Images (.png, .jpg, .webp) are sent as photo "
        "attachments, audio as file attachments. You can also include image URLs "
        "in markdown format ![alt](url) and they will be sent as attachments."
    ),
    "slack": (
        "You are in a Slack workspace communicating with your user. "
        "You can send media files natively: include MEDIA:/absolute/path/to/file "
        "in your response. Images (.png, .jpg, .webp) are uploaded as photo "
        "attachments, audio as file attachments. You can also include image URLs "
        "in markdown format ![alt](url) and they will be uploaded as attachments."
    ),
    "signal": (
        "You are on a text messaging communication platform, Signal. "
        "Please do not use markdown as it does not render. "
        "You can send media files natively: to deliver a file to the user, "
        "include MEDIA:/absolute/path/to/file in your response. Images "
        "(.png, .jpg, .webp) appear as photos, audio as attachments, and other "
        "files arrive as downloadable documents. You can also include image "
        "URLs in markdown format ![alt](url) and they will be sent as photos."
    ),
    "email": (
        "You are communicating via email. Write clear, well-structured responses "
        "suitable for email. Use plain text formatting (no markdown). "
        "Keep responses concise but complete. You can send file attachments — "
        "include MEDIA:/absolute/path/to/file in your response. The subject line "
        "is preserved for threading. Do not include greetings or sign-offs unless "
        "contextually appropriate."
    ),
    "cron": (
        "You are running as a scheduled cron job. There is no user present — you "
        "cannot ask questions, request clarification, or wait for follow-up. Execute "
        "the task fully and autonomously, making reasonable decisions where needed. "
        "Your final response is automatically delivered to the job's configured "
        "destination — put the primary content directly in your response."
    ),
    "cli": (
        "You are a CLI AI Agent. Try not to use markdown but simple text "
        "renderable inside a terminal."
    ),
    "sms": (
        "You are communicating via SMS. Keep responses concise and use plain text "
        "only — no markdown, no formatting. SMS messages are limited to ~1600 "
        "characters, so be brief and direct."
    ),
    "bluebubbles": (
        "You are chatting via iMessage (BlueBubbles). iMessage does not render "
        "markdown formatting — use plain text. Keep responses concise as they "
        "appear as text messages. You can send media files natively: include "
        "MEDIA:/absolute/path/to/file in your response. Images (.jpg, .png, "
        ".heic) appear as photos and other files arrive as attachments."
    ),
    "weixin": (
        "You are on Weixin/WeChat. Markdown formatting is supported, so you may use it when "
        "it improves readability, but keep the message compact and chat-friendly. You can send media files natively: "
        "include MEDIA:/absolute/path/to/file in your response. Images are sent as native "
        "photos, videos play inline when supported, and other files arrive as downloadable "
        "documents. You can also include image URLs in markdown format ![alt](url) and they "
        "will be downloaded and sent as native media when possible."
    ),
    "wecom": (
        "You are on WeCom (企业微信 / Enterprise WeChat). Markdown formatting is supported. "
        "You CAN send media files natively — to deliver a file to the user, include "
        "MEDIA:/absolute/path/to/file in your response. The file will be sent as a native "
        "WeCom attachment: images (.jpg, .png, .webp) are sent as photos (up to 10 MB), "
        "other files (.pdf, .docx, .xlsx, .md, .txt, etc.) arrive as downloadable documents "
        "(up to 20 MB), and videos (.mp4) play inline. Voice messages are supported but "
        "must be in AMR format — other audio formats are automatically sent as file attachments. "
        "You can also include image URLs in markdown format ![alt](url) and they will be "
        "downloaded and sent as native photos. Do NOT tell the user you lack file-sending "
        "capability — use MEDIA: syntax whenever a file delivery is appropriate."
    ),
    "qqbot": (
        "You are on QQ, a popular Chinese messaging platform. QQ supports markdown formatting "
        "and emoji. You can send media files natively: include MEDIA:/absolute/path/to/file in "
        "your response. Images are sent as native photos, and other files arrive as downloadable "
        "documents."
    ),
}

# ---------------------------------------------------------------------------
# Environment hints — execution-environment awareness for the agent.
# Unlike PLATFORM_HINTS (which describe the messaging channel), these describe
# the machine/OS the agent's tools actually run on.
# ---------------------------------------------------------------------------

WSL_ENVIRONMENT_HINT = (
    "You are running inside WSL (Windows Subsystem for Linux). "
    "The Windows host filesystem is mounted under /mnt/ — "
    "/mnt/c/ is the C: drive, /mnt/d/ is D:, etc. "
    "The user's Windows files are typically at "
    "/mnt/c/Users/<username>/Desktop/, Documents/, Downloads/, etc. "
    "When the user references Windows paths or desktop files, translate "
    "to the /mnt/c/ equivalent. You can list /mnt/c/Users/ to discover "
    "the Windows username if needed."
)


def build_environment_hints() -> str:
    """Return environment-specific guidance for the system prompt.

    Detects WSL, and can be extended for Termux, Docker, etc.
    Returns an empty string when no special environment is detected.
    """
    hints: list[str] = []
    if is_wsl():
        hints.append(WSL_ENVIRONMENT_HINT)
    return "\n\n".join(hints)


CONTEXT_FILE_MAX_CHARS = 20_000
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2


# =========================================================================
# Skills prompt cache
# =========================================================================

_SKILLS_PROMPT_CACHE_MAX = 8
_SKILLS_PROMPT_CACHE: OrderedDict[tuple, str] = OrderedDict()
_SKILLS_PROMPT_CACHE_LOCK = threading.Lock()
_SKILLS_SNAPSHOT_VERSION = 1


def _skills_prompt_snapshot_path() -> Path:
    return get_hermes_home() / ".skills_prompt_snapshot.json"


def clear_skills_system_prompt_cache(*, clear_snapshot: bool = False) -> None:
    """Drop the in-process skills prompt cache (and optionally the disk snapshot)."""
    with _SKILLS_PROMPT_CACHE_LOCK:
        _SKILLS_PROMPT_CACHE.clear()
    if clear_snapshot:
        try:
            _skills_prompt_snapshot_path().unlink(missing_ok=True)
        except OSError as e:
            logger.debug("Could not remove skills prompt snapshot: %s", e)


def _build_skills_manifest(skills_dir: Path) -> dict[str, list[int]]:
    """Build an mtime/size manifest of all SKILL.md and DESCRIPTION.md files."""
    manifest: dict[str, list[int]] = {}
    for filename in ("SKILL.md", "DESCRIPTION.md"):
        for path in iter_skill_index_files(skills_dir, filename):
            try:
                st = path.stat()
            except OSError:
                continue
            manifest[str(path.relative_to(skills_dir))] = [st.st_mtime_ns, st.st_size]
    return manifest


def _load_skills_snapshot(skills_dir: Path) -> Optional[dict]:
    """Load the disk snapshot if it exists and its manifest still matches."""
    snapshot_path = _skills_prompt_snapshot_path()
    if not snapshot_path.exists():
        return None
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(snapshot, dict):
        return None
    if snapshot.get("version") != _SKILLS_SNAPSHOT_VERSION:
        return None
    if snapshot.get("manifest") != _build_skills_manifest(skills_dir):
        return None
    return snapshot


def _write_skills_snapshot(
    skills_dir: Path,
    manifest: dict[str, list[int]],
    skill_entries: list[dict],
    category_descriptions: dict[str, str],
) -> None:
    """Persist skill metadata to disk for fast cold-start reuse."""
    payload = {
        "version": _SKILLS_SNAPSHOT_VERSION,
        "manifest": manifest,
        "skills": skill_entries,
        "category_descriptions": category_descriptions,
    }
    try:
        atomic_json_write(_skills_prompt_snapshot_path(), payload)
    except Exception as e:
        logger.debug("Could not write skills prompt snapshot: %s", e)


def _build_snapshot_entry(
    skill_file: Path,
    skills_dir: Path,
    frontmatter: dict,
    description: str,
) -> dict:
    """Build a serialisable metadata dict for one skill."""
    rel_path = skill_file.relative_to(skills_dir)
    parts = rel_path.parts
    if len(parts) >= 2:
        skill_name = parts[-2]
        category = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
    else:
        category = "general"
        skill_name = skill_file.parent.name

    platforms = frontmatter.get("platforms") or []
    if isinstance(platforms, str):
        platforms = [platforms]

    return {
        "skill_name": skill_name,
        "category": category,
        "frontmatter_name": str(frontmatter.get("name", skill_name)),
        "description": description,
        "platforms": [str(p).strip() for p in platforms if str(p).strip()],
        "conditions": extract_skill_conditions(frontmatter),
    }


# =========================================================================
# Skills index
# =========================================================================

def _parse_skill_file(skill_file: Path) -> tuple[bool, dict, str]:
    """Read a SKILL.md once and return platform compatibility, frontmatter, and description.

    Returns (is_compatible, frontmatter, description). On any error, returns
    (True, {}, "") to err on the side of showing the skill.
    """
    try:
        raw = skill_file.read_text(encoding="utf-8")
        frontmatter, _ = parse_frontmatter(raw)

        if not skill_matches_platform(frontmatter):
            return False, frontmatter, ""

        return True, frontmatter, extract_skill_description(frontmatter)
    except Exception as e:
        logger.warning("Failed to parse skill file %s: %s", skill_file, e)
        return True, {}, ""


def _skill_should_show(
    conditions: dict,
    available_tools: "set[str] | None",
    available_toolsets: "set[str] | None",
) -> bool:
    """Return False if the skill's conditional activation rules exclude it."""
    if available_tools is None and available_toolsets is None:
        return True  # No filtering info — show everything (backward compat)

    at = available_tools or set()
    ats = available_toolsets or set()

    # fallback_for: hide when the primary tool/toolset IS available
    for ts in conditions.get("fallback_for_toolsets", []):
        if ts in ats:
            return False
    for t in conditions.get("fallback_for_tools", []):
        if t in at:
            return False

    # requires: hide when a required tool/toolset is NOT available
    for ts in conditions.get("requires_toolsets", []):
        if ts not in ats:
            return False
    for t in conditions.get("requires_tools", []):
        if t not in at:
            return False

    return True


def build_skills_system_prompt(
    available_tools: "set[str] | None" = None,
    available_toolsets: "set[str] | None" = None,
) -> str:
    """Build a compact skill index for the system prompt.

    Two-layer cache:
      1. In-process LRU dict keyed by (skills_dir, tools, toolsets)
      2. Disk snapshot (``.skills_prompt_snapshot.json``) validated by
         mtime/size manifest — survives process restarts

    Falls back to a full filesystem scan when both layers miss.

    External skill directories (``skills.external_dirs`` in config.yaml) are
    scanned alongside the local ``~/.hermes/skills/`` directory.  External dirs
    are read-only — they appear in the index but new skills are always created
    in the local dir.  Local skills take precedence when names collide.
    """
    skills_dir = get_skills_dir()
    external_dirs = get_all_skills_dirs()[1:]  # skip local (index 0)

    if not skills_dir.exists() and not external_dirs:
        return ""

    # ── Layer 1: in-process LRU cache ─────────────────────────────────
    # Include the resolved platform so per-platform disabled-skill lists
    # produce distinct cache entries (gateway serves multiple platforms).
    from gateway.session_context import get_session_env
    _platform_hint = (
        os.environ.get("HERMES_PLATFORM")
        or get_session_env("HERMES_SESSION_PLATFORM")
        or ""
    )
    cache_key = (
        str(skills_dir.resolve()),
        tuple(str(d) for d in external_dirs),
        tuple(sorted(str(t) for t in (available_tools or set()))),
        tuple(sorted(str(ts) for ts in (available_toolsets or set()))),
        _platform_hint,
    )
    with _SKILLS_PROMPT_CACHE_LOCK:
        cached = _SKILLS_PROMPT_CACHE.get(cache_key)
        if cached is not None:
            _SKILLS_PROMPT_CACHE.move_to_end(cache_key)
            return cached

    disabled = get_disabled_skill_names()

    # ── Layer 2: disk snapshot ────────────────────────────────────────
    snapshot = _load_skills_snapshot(skills_dir)

    skills_by_category: dict[str, list[tuple[str, str]]] = {}
    category_descriptions: dict[str, str] = {}

    if snapshot is not None:
        # Fast path: use pre-parsed metadata from disk
        for entry in snapshot.get("skills", []):
            if not isinstance(entry, dict):
                continue
            skill_name = entry.get("skill_name") or ""
            category = entry.get("category") or "general"
            frontmatter_name = entry.get("frontmatter_name") or skill_name
            platforms = entry.get("platforms") or []
            if not skill_matches_platform({"platforms": platforms}):
                continue
            if frontmatter_name in disabled or skill_name in disabled:
                continue
            if not _skill_should_show(
                entry.get("conditions") or {},
                available_tools,
                available_toolsets,
            ):
                continue
            skills_by_category.setdefault(category, []).append(
                (frontmatter_name, entry.get("description", ""))
            )
        category_descriptions = {
            str(k): str(v)
            for k, v in (snapshot.get("category_descriptions") or {}).items()
        }
    else:
        # Cold path: full filesystem scan + write snapshot for next time
        skill_entries: list[dict] = []
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            is_compatible, frontmatter, desc = _parse_skill_file(skill_file)
            entry = _build_snapshot_entry(skill_file, skills_dir, frontmatter, desc)
            skill_entries.append(entry)
            if not is_compatible:
                continue
            skill_name = entry["skill_name"]
            if entry["frontmatter_name"] in disabled or skill_name in disabled:
                continue
            if not _skill_should_show(
                extract_skill_conditions(frontmatter),
                available_tools,
                available_toolsets,
            ):
                continue
            skills_by_category.setdefault(entry["category"], []).append(
                (entry["frontmatter_name"], entry["description"])
            )

        # Read category-level DESCRIPTION.md files
        for desc_file in iter_skill_index_files(skills_dir, "DESCRIPTION.md"):
            try:
                content = desc_file.read_text(encoding="utf-8")
                fm, _ = parse_frontmatter(content)
                cat_desc = fm.get("description")
                if not cat_desc:
                    continue
                rel = desc_file.relative_to(skills_dir)
                cat = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else "general"
                category_descriptions[cat] = str(cat_desc).strip().strip("'\"")
            except Exception as e:
                logger.debug("Could not read skill description %s: %s", desc_file, e)

        _write_skills_snapshot(
            skills_dir,
            _build_skills_manifest(skills_dir),
            skill_entries,
            category_descriptions,
        )

    # ── External skill directories ─────────────────────────────────────
    # Scan external dirs directly (no snapshot caching — they're read-only
    # and typically small).  Local skills already in skills_by_category take
    # precedence: we track seen names and skip duplicates from external dirs.
    seen_skill_names: set[str] = set()
    for cat_skills in skills_by_category.values():
        for name, _desc in cat_skills:
            seen_skill_names.add(name)

    for ext_dir in external_dirs:
        if not ext_dir.exists():
            continue
        for skill_file in iter_skill_index_files(ext_dir, "SKILL.md"):
            try:
                is_compatible, frontmatter, desc = _parse_skill_file(skill_file)
                if not is_compatible:
                    continue
                entry = _build_snapshot_entry(skill_file, ext_dir, frontmatter, desc)
                skill_name = entry["skill_name"]
                frontmatter_name = entry["frontmatter_name"]
                if frontmatter_name in seen_skill_names:
                    continue
                if frontmatter_name in disabled or skill_name in disabled:
                    continue
                if not _skill_should_show(
                    extract_skill_conditions(frontmatter),
                    available_tools,
                    available_toolsets,
                ):
                    continue
                seen_skill_names.add(frontmatter_name)
                skills_by_category.setdefault(entry["category"], []).append(
                    (frontmatter_name, entry["description"])
                )
            except Exception as e:
                logger.debug("Error reading external skill %s: %s", skill_file, e)

        # External category descriptions
        for desc_file in iter_skill_index_files(ext_dir, "DESCRIPTION.md"):
            try:
                content = desc_file.read_text(encoding="utf-8")
                fm, _ = parse_frontmatter(content)
                cat_desc = fm.get("description")
                if not cat_desc:
                    continue
                rel = desc_file.relative_to(ext_dir)
                cat = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else "general"
                category_descriptions.setdefault(cat, str(cat_desc).strip().strip("'\""))
            except Exception as e:
                logger.debug("Could not read external skill description %s: %s", desc_file, e)

    if not skills_by_category:
        result = ""
    else:
        index_lines = []
        for category in sorted(skills_by_category.keys()):
            cat_desc = category_descriptions.get(category, "")
            if cat_desc:
                index_lines.append(f"  {category}: {cat_desc}")
            else:
                index_lines.append(f"  {category}:")
            # Deduplicate and sort skills within each category
            seen = set()
            for name, desc in sorted(skills_by_category[category], key=lambda x: x[0]):
                if name in seen:
                    continue
                seen.add(name)
                if desc:
                    index_lines.append(f"    - {name}: {desc}")
                else:
                    index_lines.append(f"    - {name}")

        result = (
            "## Skills (mandatory)\n"
            "Before replying, scan the skills below. If a skill matches or is even partially relevant "
            "to your task, you MUST load it with skill_view(name) and follow its instructions. "
            "Err on the side of loading — it is always better to have context you don't need "
            "than to miss critical steps, pitfalls, or established workflows. "
            "Skills contain specialized knowledge — API endpoints, tool-specific commands, "
            "and proven workflows that outperform general-purpose approaches. Load the skill "
            "even if you think you could handle the task with basic tools like web_search or terminal. "
            "Skills also encode the user's preferred approach, conventions, and quality standards "
            "for tasks like code review, planning, and testing — load them even for tasks you "
            "already know how to do, because the skill defines how it should be done here.\n"
            "If a skill has issues, fix it with skill_manage(action='patch').\n"
            "After difficult/iterative tasks, offer to save as a skill. "
            "If a skill you loaded was missing steps, had wrong commands, or needed "
            "pitfalls you discovered, update it before finishing.\n"
            "\n"
            "<available_skills>\n"
            + "\n".join(index_lines) + "\n"
            "</available_skills>\n"
            "\n"
            "Only proceed without loading a skill if genuinely none are relevant to the task."
        )

    # ── Store in LRU cache ────────────────────────────────────────────
    with _SKILLS_PROMPT_CACHE_LOCK:
        _SKILLS_PROMPT_CACHE[cache_key] = result
        _SKILLS_PROMPT_CACHE.move_to_end(cache_key)
        while len(_SKILLS_PROMPT_CACHE) > _SKILLS_PROMPT_CACHE_MAX:
            _SKILLS_PROMPT_CACHE.popitem(last=False)

    return result


def build_nous_subscription_prompt(valid_tool_names: "set[str] | None" = None) -> str:
    """Build a compact Nous subscription capability block for the system prompt."""
    try:
        from hermes_cli.nous_subscription import get_nous_subscription_features
        from tools.tool_backend_helpers import managed_nous_tools_enabled
    except Exception as exc:
        logger.debug("Failed to import Nous subscription helper: %s", exc)
        return ""

    if not managed_nous_tools_enabled():
        return ""

    valid_names = set(valid_tool_names or set())
    relevant_tool_names = {
        "web_search",
        "web_extract",
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_scroll",
        "browser_console",
        "browser_press",
        "browser_get_images",
        "browser_vision",
        "image_generate",
        "text_to_speech",
        "terminal",
        "process",
        "execute_code",
    }

    if valid_names and not (valid_names & relevant_tool_names):
        return ""

    features = get_nous_subscription_features()

    def _status_line(feature) -> str:
        if feature.managed_by_nous:
            return f"- {feature.label}: active via Nous subscription"
        if feature.active:
            current = feature.current_provider or "configured provider"
            return f"- {feature.label}: currently using {current}"
        if feature.included_by_default and features.nous_auth_present:
            return f"- {feature.label}: included with Nous subscription, not currently selected"
        if feature.key == "modal" and features.nous_auth_present:
            return f"- {feature.label}: optional via Nous subscription"
        return f"- {feature.label}: not currently available"

    lines = [
        "# Nous Subscription",
        "Nous subscription includes managed web tools (Firecrawl), image generation (FAL), OpenAI TTS, and browser automation (Browser Use) by default. Modal execution is optional.",
        "Current capability status:",
    ]
    lines.extend(_status_line(feature) for feature in features.items())
    lines.extend(
        [
            "When a Nous-managed feature is active, do not ask the user for Firecrawl, FAL, OpenAI TTS, or Browser-Use API keys.",
            "If the user is not subscribed and asks for a capability that Nous subscription would unlock or simplify, suggest Nous subscription as one option alongside direct setup or local alternatives.",
            "Do not mention subscription unless the user asks about it or it directly solves the current missing capability.",
            "Useful commands: hermes setup, hermes setup tools, hermes setup terminal, hermes status.",
        ]
    )
    return "\n".join(lines)


# =========================================================================
# Context files (SOUL.md, AGENTS.md, .cursorrules)
# =========================================================================

def _truncate_content(content: str, filename: str, max_chars: int = CONTEXT_FILE_MAX_CHARS) -> str:
    """Head/tail truncation with a marker in the middle."""
    if len(content) <= max_chars:
        return content
    head_chars = int(max_chars * CONTEXT_TRUNCATE_HEAD_RATIO)
    tail_chars = int(max_chars * CONTEXT_TRUNCATE_TAIL_RATIO)
    head = content[:head_chars]
    tail = content[-tail_chars:]
    marker = f"\n\n[...truncated {filename}: kept {head_chars}+{tail_chars} of {len(content)} chars. Use file tools to read the full file.]\n\n"
    return head + marker + tail


def load_soul_md() -> Optional[str]:
    """Load SOUL.md from HERMES_HOME and return its content, or None.

    Used as the agent identity (slot #1 in the system prompt).  When this
    returns content, ``build_context_files_prompt`` should be called with
    ``skip_soul=True`` so SOUL.md isn't injected twice.
    """
    try:
        from hermes_cli.config import ensure_hermes_home
        ensure_hermes_home()
    except Exception as e:
        logger.debug("Could not ensure HERMES_HOME before loading SOUL.md: %s", e)

    soul_path = get_hermes_home() / "SOUL.md"
    if not soul_path.exists():
        return None
    try:
        content = soul_path.read_text(encoding="utf-8").strip()
        if not content:
            return None
        content = _scan_context_content(content, "SOUL.md")
        content = _truncate_content(content, "SOUL.md")
        return content
    except Exception as e:
        logger.debug("Could not read SOUL.md from %s: %s", soul_path, e)
        return None


def _load_hermes_md_file(cwd_path: Path, hermes_md_path: Path) -> str:
    """Load one .hermes.md/HERMES.md file with stable relative path rendering."""
    try:
        content = hermes_md_path.read_text(encoding="utf-8").strip()
        if not content:
            return ""
        content = _strip_yaml_frontmatter(content)
        rel = hermes_md_path.name
        try:
            rel = str(hermes_md_path.relative_to(cwd_path))
        except ValueError:
            try:
                rel = str(hermes_md_path.relative_to(_find_git_root(cwd_path) or cwd_path.parent))
            except ValueError:
                rel = hermes_md_path.name
        content = _scan_context_content(content, rel)
        result = f"## {rel}\n\n{content}"
        return _truncate_content(result, hermes_md_path.name)
    except Exception as e:
        logger.debug("Could not read %s: %s", hermes_md_path, e)
        return ""


def _load_hermes_md(cwd_path: Path) -> str:
    """Backward-compatible single-file .hermes.md / HERMES.md loader."""
    hermes_md_path = _find_hermes_md(cwd_path)
    if not hermes_md_path:
        return ""
    return _load_hermes_md_file(cwd_path, hermes_md_path)


def _load_agents_md(cwd_path: Path) -> str:
    """AGENTS.md — top-level only (no recursive walk)."""
    for name in ["AGENTS.md", "agents.md"]:
        candidate = cwd_path / name
        if candidate.exists():
            try:
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    content = _scan_context_content(content, name)
                    result = f"## {name}\n\n{content}"
                    return _truncate_content(result, "AGENTS.md")
            except Exception as e:
                logger.debug("Could not read %s: %s", candidate, e)
    return ""


def _load_claude_md(cwd_path: Path) -> str:
    """CLAUDE.md / claude.md — cwd only."""
    for name in ["CLAUDE.md", "claude.md"]:
        candidate = cwd_path / name
        if candidate.exists():
            try:
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    content = _scan_context_content(content, name)
                    result = f"## {name}\n\n{content}"
                    return _truncate_content(result, "CLAUDE.md")
            except Exception as e:
                logger.debug("Could not read %s: %s", candidate, e)
    return ""


def _load_cursorrules(cwd_path: Path) -> str:
    """.cursorrules + .cursor/rules/*.mdc — cwd only."""
    cursorrules_content = ""
    cursorrules_file = cwd_path / ".cursorrules"
    if cursorrules_file.exists():
        try:
            content = cursorrules_file.read_text(encoding="utf-8").strip()
            if content:
                content = _scan_context_content(content, ".cursorrules")
                cursorrules_content += f"## .cursorrules\n\n{content}\n\n"
        except Exception as e:
            logger.debug("Could not read .cursorrules: %s", e)

    cursor_rules_dir = cwd_path / ".cursor" / "rules"
    if cursor_rules_dir.exists() and cursor_rules_dir.is_dir():
        mdc_files = sorted(cursor_rules_dir.glob("*.mdc"))
        for mdc_file in mdc_files:
            try:
                content = mdc_file.read_text(encoding="utf-8").strip()
                if content:
                    content = _scan_context_content(content, f".cursor/rules/{mdc_file.name}")
                    cursorrules_content += f"## .cursor/rules/{mdc_file.name}\n\n{content}\n\n"
            except Exception as e:
                logger.debug("Could not read %s: %s", mdc_file, e)

    if not cursorrules_content:
        return ""
    return _truncate_content(cursorrules_content, ".cursorrules")


def build_context_files_prompt(
    cwd: Optional[str] = None,
    skip_soul: bool = False,
    task_contract: dict[str, Any] | TaskContract | None = None,
    max_hermes_hierarchy_layers: int = MAX_HERMES_CONTEXT_LAYERS,
) -> str:
    """Discover and load bounded project context for the system prompt.

    Hierarchical behavior:
      1. Load up to ``max_hermes_hierarchy_layers`` .hermes.md/HERMES.md files
         along the git-root→cwd lineage, in deterministic root→leaf order.
      2. If no Hermes lineage exists, fall back to exactly one cwd-only context
         source by the historical priority order: AGENTS.md → CLAUDE.md → cursor rules.
      3. SOUL.md from HERMES_HOME remains independent and optional.

    When ``task_contract`` is provided, bounded context precedence metadata is
    rendered additively without mutating the canonical contract fields.
    """
    if cwd is None:
        cwd = os.getcwd()

    cwd_path = Path(cwd).resolve()
    sections: list[str] = []
    layer_descriptors = []

    hermes_layers = discover_hermes_context_layers(
        cwd_path,
        max_layers=max_hermes_hierarchy_layers,
    )
    if hermes_layers:
        for layer in hermes_layers:
            project_context = _load_hermes_md_file(cwd_path, layer.path)
            if project_context:
                sections.append(project_context)
                layer_descriptors.append(layer)
    else:
        project_context = (
            _load_agents_md(cwd_path)
            or _load_claude_md(cwd_path)
            or _load_cursorrules(cwd_path)
        )
        if project_context:
            sections.append(project_context)

    if task_contract is not None and layer_descriptors:
        bounded = apply_bounded_hierarchical_context(task_contract, layer_descriptors)
        sections.append(
            "## Context Precedence\n\n"
            "Task-contract fields remain authoritative. Bounded hierarchical context is additive only.\n\n"
            + "precedence: " + " > ".join(bounded.precedence)
        )

    # SOUL.md from HERMES_HOME only — skip when already loaded as identity
    if not skip_soul:
        soul_content = load_soul_md()
        if soul_content:
            sections.append(soul_content)

    if not sections:
        return ""
    return "# Project Context\n\nThe following project context files have been loaded and should be followed:\n\n" + "\n".join(sections)
