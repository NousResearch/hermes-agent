"""Core CodeAct recipe injection.

Recipes are small, high-level Python functions injected into the persistent
CodeAct namespace. They orchestrate Hermes tools but return structured data so
the assistant can synthesize the final answer outside ``run_code``.
"""

from __future__ import annotations

import textwrap


def get_recipe_names(enabled_tool_names: set[str] | None) -> list[str]:
    tools = enabled_tool_names or set()
    names: list[str] = []
    if "research_gather" in tools:
        names.append("research_web")
    return names


def build_recipe_catalogue(enabled_tool_names: set[str] | None) -> str:
    lines = []
    if "research_web" in get_recipe_names(enabled_tool_names):
        lines.append(
            "  research_web(question, topic_type='auto', freshness='auto', "
            "depth='thorough', max_sources=8) — gather citation-ready web evidence"
        )
    return "\n".join(lines)


def build_recipe_help_registry(
    enabled_tool_names: set[str] | None,
) -> dict[str, tuple[str, str]]:
    registry: dict[str, tuple[str, str]] = {}
    if "research_web" in get_recipe_names(enabled_tool_names):
        compact = (
            "  research_web(question, topic_type='auto', freshness='auto', "
            "depth='thorough', max_sources=8) — gather citation-ready web evidence"
        )
        full = (
            "Core CodeAct recipe for source-grounded web research.\n\n"
            "Runs Hermes' local-first research stack: topic classification, "
            "typed query fan-out, local research memory lookup, web search, "
            "page extraction, optional browser fallback, and gap/adversarial "
            "checks. Returns a structured evidence bundle; do not use it to "
            "print the final answer. After calling it, write the final answer "
            "as normal assistant text with source references from the bundle. "
            "Use research_help() if you need the lower-level research_search "
            "tool workflow."
        )
        registry["research_web"] = (compact, full)
    return registry


def build_recipe_source(enabled_tool_names: set[str] | None) -> str:
    if "research_web" not in get_recipe_names(enabled_tool_names):
        return ""

    return textwrap.dedent(
        """\
        def research_web(question, topic_type='auto', freshness='auto',
                         depth='thorough', max_sources=8):
            \"\"\"Gather source-grounded web evidence as a structured dict.

            Use this before manually sequencing web_search/web_extract for
            current facts, source-grounded answers, comparisons, or research.
            Return value is evidence for the assistant to synthesize; it is not
            a final prose answer.
            \"\"\"
            import json as _json
            raw = _call_tool('research_gather', {
                'question': question,
                'topic_type': topic_type,
                'freshness': freshness,
                'depth': depth,
                'max_pages': max_sources,
            })
            try:
                parsed = _json.loads(raw)
            except Exception as _exc:
                return {
                    'success': False,
                    'error': f'research_gather returned malformed JSON: {_exc}',
                    'raw': raw,
                }
            if not isinstance(parsed, dict):
                return {
                    'success': False,
                    'error': 'research_gather returned non-dict JSON',
                    'raw': parsed,
                }
            return parsed
        """
    )
