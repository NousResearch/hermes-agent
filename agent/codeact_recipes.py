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
        names.append("medical_pharma_research")
    return names


def build_recipe_catalogue(enabled_tool_names: set[str] | None) -> str:
    lines = []
    if "research_web" in get_recipe_names(enabled_tool_names):
        lines.append(
            "  research_web(question, topic_type='auto', freshness='auto', "
            "depth='thorough', max_sources=8) — FIRST CALL for search/research/report/latest/current tasks; returns citation-ready evidence"
        )
    if "medical_pharma_research" in get_recipe_names(enabled_tool_names):
        lines.append(
            "  medical_pharma_research(question, freshness='latest', depth='thorough', "
            "max_sources=10) — FDA/EMA/NMPA + company IR + ClinicalTrials.gov + PubMed source recipe"
        )
    return "\n".join(lines)


def build_recipe_help_registry(
    enabled_tool_names: set[str] | None,
) -> dict[str, tuple[str, str]]:
    registry: dict[str, tuple[str, str]] = {}
    if "research_web" in get_recipe_names(enabled_tool_names):
        compact = (
            "  research_web(question, topic_type='auto', freshness='auto', "
            "depth='thorough', max_sources=8) — FIRST CALL for search/research/report/latest/current tasks"
        )
        full = (
            "Core CodeAct recipe for source-grounded web research.\n\n"
            "Use this first for prompts that ask to search, research, produce "
            "a report, find latest/current information, or answer as of a date. "
            "Do not start with raw web_search/browser/curl/Wikipedia unless this "
            "recipe fails or you only need a one-off URL discovery.\n\n"
            "Runs Hermes' local-first research stack: topic classification, "
            "typed query fan-out, local research memory lookup, web search, "
            "page extraction, optional browser fallback, and gap/adversarial "
            "checks. Returns a structured evidence bundle; do not use it to "
            "print the final answer. After calling it, write the final answer "
            "as normal assistant text with source references from source_table "
            "or citation_metadata in the bundle. Final research reports must "
            "include citations or a source table. "
            "If a source hits rate limits, JS challenges, bot protection, or "
            "Cloudflare/Wikipedia blocks, use browser/Camofox/Scrapling "
            "fallbacks when available and record the limitation. "
            "Use research_help() if you need the lower-level research_search "
            "tool workflow."
        )
        registry["research_web"] = (compact, full)
    if "medical_pharma_research" in get_recipe_names(enabled_tool_names):
        compact = (
            "  medical_pharma_research(question, freshness='latest', "
            "depth='thorough', max_sources=10) — medical/pharma source-quality recipe"
        )
        full = (
            "Use for pharma, GLP-1/GIP, clinical-trial, drug-development, "
            "approval, or pipeline reports. It routes research_gather through "
            "topic_type='medical_pharma', prioritizing FDA/EMA/NMPA, company "
            "investor relations and press releases, ClinicalTrials.gov, PubMed "
            "and major journals, then reputable news. Wikipedia is orientation "
            "only and should not support final status claims."
        )
        registry["medical_pharma_research"] = (compact, full)
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
            search, research, report, latest/current, source-grounded answers,
            comparisons, or as-of-date tasks. Return value is evidence for the
            assistant to synthesize; it is not a final prose answer. Final
            reports must cite source_table/citation_metadata from the result.
            \"\"\"
            import json as _json
            try:
                from agent.research_search.intent import classify_research_intent as _classify_research_intent

                _intent = _classify_research_intent(question)
                _intent_topic = _intent.get('topic_type')
                _intent_freshness = _intent.get('freshness')
                if topic_type == 'auto' and _intent_topic and _intent_topic != 'general':
                    topic_type = _intent_topic
                if freshness == 'auto' and _intent_freshness in ('latest', 'recent'):
                    freshness = _intent_freshness
            except Exception:
                pass
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

        def medical_pharma_research(question, freshness='latest',
                                    depth='thorough', max_sources=10):
            \"\"\"Gather medical/pharma evidence with a regulator-first source profile.

            Prioritizes FDA/EMA/NMPA, company IR/press releases,
            ClinicalTrials.gov, PubMed/major journals, then reputable news.
            Wikipedia is orientation only. Final reports must cite the returned
            source_table/citation_metadata.
            \"\"\"
            return research_web(
                question,
                topic_type='medical_pharma',
                freshness=freshness,
                depth=depth,
                max_sources=max_sources,
            )
        """
    )
